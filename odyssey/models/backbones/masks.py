"""Attention masks and position ids shared by every attention-based backbone.

Three backbones need exactly the same mask arithmetic -- the native
transformer control, the BigBird-shaped restriction of it, and the
``transformers`` adapter, which hands the result straight to a Hugging Face
model as a 4D mask. Keeping it here rather than in any one of them means the
guarantee is written and tested once.

The guarantee, in one sentence: **a position may attend only to positions
belonging to the same packed patient, and, in the causal regime, only to
positions at or before it.** Everything else in this module is that sentence
plus the bookkeeping to make it true under packing.

Two independent axes combine here:

*Context regime* (:mod:`odyssey.models.backbones`) decides which same-segment
pairs are allowed at all -- ``causal`` (lower triangle), ``encoder`` (all of
them), ``prefix`` (lower triangle, plus both directions inside a sampled
prefix). *Sparsity* then removes pairs a dense stack would have kept, which is
how a BigBird-shaped receptive field is expressed without a sparse kernel.

Why a mask and not an argument to the attention kernel: the same boolean array
is valid for ``torch.nn.functional.scaled_dot_product_attention`` and for any
``transformers`` model, because HF returns an already-4D mask untouched
(``transformers.masking_utils._preprocess_mask_arguments``). One
implementation therefore covers our own blocks and every wrapped HF stack, and
one test proves both.
"""

from typing import NamedTuple

import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch


def segment_ids(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return a per-position segment id: increments at every reset.

    Only equality between two positions' ids is meaningful (whether they
    belong to the same packed patient); the absolute values carry no
    other information.
    """
    return torch.cumsum(reset_mask.long(), dim=1)


def position_ids(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return each position's index since its most recent reset (for RoPE).

    Position 0 of every segment gets id 0, whether or not that segment is
    the first in the row -- a packed patient's rotary angles are identical
    to processing that same patient alone.
    """
    seq_len = reset_mask.shape[1]
    idx = (
        torch.arange(seq_len, device=reset_mask.device)
        .unsqueeze(0)
        .expand_as(reset_mask)
    )
    segment_start = torch.where(reset_mask, idx, torch.zeros_like(idx))
    segment_start = torch.cummax(segment_start, dim=1).values
    return idx - segment_start


def segment_lengths(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return, per position, the total length of the segment it belongs to."""
    ids = segment_ids(reset_mask)
    counts = torch.zeros_like(ids).scatter_add_(1, ids, torch.ones_like(ids))
    return counts.gather(1, ids)


def rebase_time_stamps(
    time_stamps: torch.Tensor, reset_mask: torch.Tensor
) -> torch.Tensor:
    """Force every segment boundary's time delta to exactly 0, elsewhere unchanged.

    :class:`~odyssey.models.embeddings.TimeEmbeddingLayer` computes
    time-since-previous-event as a delta over the *whole row's* raw
    timestamps, uniformly -- it has no notion of a packed segment
    boundary. Left alone, a packed segment's own first position would get
    whatever ``time_stamps[boundary] - time_stamps[boundary - 1]`` happens
    to be: a value that depends on the *previous* segment's absolute
    timestamps, exactly the cross-patient leakage this must not have.
    Zeroing the delta at every reset (matching the "fresh sequence start"
    convention :class:`TimeEmbeddingLayer` already uses at row position 0
    when no ``prev_value`` is given) and reconstructing the series by
    cumulative sum makes every non-boundary delta come out identical to the
    original (nothing but the boundary deltas changes), so this is a
    correction, not an approximation. Doing this from ``reset_mask`` alone,
    rather than trusting a caller to have pre-shifted timestamps, means the
    no-leakage guarantee holds for *any* valid ``reset_mask``, not only ones
    a particular sampler happens to construct carefully.
    """
    deltas = time_stamps[:, 1:] - time_stamps[:, :-1]
    deltas = deltas.masked_fill(reset_mask[:, 1:], 0.0)
    first = time_stamps[:, :1]
    return torch.cat([first, first + torch.cumsum(deltas, dim=1)], dim=1)


def sample_prefix_mask(
    reset_mask: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``(batch, seq)``: True where a position is inside its prefix.

    One fraction is drawn per *segment* from ``U[0, 2 * fraction)`` (so the
    expected prefix is ``fraction`` of the segment, and the model sees many
    boundaries rather than memorizing one), and the segment's first
    ``round(draw * length)`` positions form its prefix. Sampling per segment,
    not per row, keeps a packed patient's treatment independent of whatever
    it was packed beside.
    """
    within = position_ids(reset_mask)
    lengths = segment_lengths(reset_mask)
    draw = torch.rand(
        reset_mask.shape, device=reset_mask.device, generator=generator
    ) * (2.0 * fraction)
    # One draw per segment: each position takes the draw made at its own
    # segment's start. Offsetting by 1 before the running max keeps a genuine
    # draw of 0.0 distinguishable from "no segment start seen yet".
    seg_start_draw = (
        torch.cummax(
            torch.where(within == 0, draw + 1.0, torch.zeros_like(draw)), dim=1
        ).values
        - 1.0
    )
    cut = torch.round(seg_start_draw.clamp(0.0, 1.0) * lengths.to(draw.dtype))
    return within < cut.long()


def bigbird_block_mask(
    reset_mask: torch.Tensor,
    *,
    block_size: int,
    num_global_blocks: int,
    num_random_blocks: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``(batch, seq, seq)``: True where BigBird's pattern allows a pair.

    Blocks are numbered *within each packed segment*, so a patient's own
    receptive field is the same whether or not it shares a row: its first
    blocks are its own global blocks, its window is its own neighbourhood.
    Numbering blocks by absolute row position instead would give a packed
    neighbour's opening tokens the global role for everyone, which is not
    BigBird and not reproducible from a patient alone.

    Reproduces the receptive field, not the sparse kernel: the result is a
    dense boolean mask, so compute stays quadratic. The speed claim is not
    what this arm is for -- the Mamba arms already answer that -- and a real
    block-sparse kernel is weeks of work for a question nobody is asking.

    ``generator`` fixes the random blocks. Callers must pass a seeded one at
    evaluation: a paired comparison across backbones cannot attribute noise
    that changes between two scoring passes of the same model.
    """
    batch, seq_len = reset_mask.shape
    if seq_len == 0:
        return reset_mask.new_ones((batch, 0, 0), dtype=torch.bool)
    within = position_ids(reset_mask)
    block = within // block_size
    num_blocks = int(block.max().item()) + 1

    q_blk = block.unsqueeze(2)
    k_blk = block.unsqueeze(1)

    window = (q_blk - k_blk).abs() <= 1
    is_global = block < num_global_blocks
    glob = is_global.unsqueeze(2) | is_global.unsqueeze(1)

    allowed = window | glob
    if num_random_blocks > 0 and num_blocks > 1:
        scores = torch.rand(
            (batch, num_blocks, num_blocks),
            device=reset_mask.device,
            generator=generator,
        )
        keep = min(num_random_blocks, num_blocks)
        threshold = scores.topk(keep, dim=-1).values[..., -1:]
        random_adj = scores >= threshold  # (batch, q_block, k_block)
        # out[b, i, j] = random_adj[b, block[b, i], block[b, j]], in two
        # gathers: pick each query's row of the block adjacency, then each
        # key's column of that row.
        rows = random_adj.gather(
            1, block.unsqueeze(-1).expand(batch, seq_len, num_blocks)
        )
        allowed = allowed | rows.gather(
            2, block.unsqueeze(1).expand(batch, seq_len, seq_len)
        )
    return allowed


def build_attn_mask(
    reset_mask: torch.Tensor,
    *,
    mode: str = "causal",
    prefix_mask: torch.Tensor | None = None,
    sparsity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(batch, 1, seq, seq)`` bool: True where attention is allowed.

    Always intersected with same-segment, so no regime and no sparsity
    pattern can ever let one packed patient see another. Within a segment:

    - ``causal``: position ``i`` may attend to ``j <= i``.
    - ``encoder``: every pair, both directions. Only legal when the loss no
      longer supervises anything a position can see -- see
      :mod:`odyssey.models.backbones`.
    - ``prefix``: causal, plus both directions among positions flagged in
      ``prefix_mask``. The training loop must drop those positions from every
      forecasting loss; they have read their own targets.
    """
    batch, seq_len = reset_mask.shape
    segment = segment_ids(reset_mask)
    same_segment = segment.unsqueeze(2) == segment.unsqueeze(1)

    if mode == "encoder":
        allowed = same_segment
    else:
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=reset_mask.device)
        ).unsqueeze(0)
        if mode == "prefix":
            if prefix_mask is None:
                raise ValueError("mode='prefix' needs a prefix_mask")
            both_in_prefix = prefix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)
            causal = causal | both_in_prefix
        elif mode != "causal":
            raise ValueError(f"unknown attention mode {mode!r}")
        allowed = same_segment & causal

    if sparsity_mask is not None:
        allowed = allowed & sparsity_mask
    return allowed.unsqueeze(1).expand(batch, 1, seq_len, seq_len)


def resolve_reset_mask(
    reset_mask: torch.Tensor | None, concept_ids: torch.Tensor
) -> torch.Tensor:
    """Return a reset mask that always opens a segment at row position 0.

    ``None`` (or an all-``False`` mask) means "the whole row is one segment",
    the ordinary one-patient-per-row case every existing test batch uses.
    """
    batch_size, seq_len = concept_ids.shape
    resolved = (
        concept_ids.new_zeros(batch_size, seq_len, dtype=torch.bool)
        if reset_mask is None
        else reset_mask
    )
    if seq_len > 0 and not bool(resolved[:, 0].all()):
        resolved = resolved.clone()
        resolved[:, 0] = True
    return resolved


class AttentionInputs(NamedTuple):
    """Everything an attention stack needs derived from one batch + reset mask."""

    reset_mask: torch.Tensor
    """``(batch, seq)`` with a guaranteed segment opening at position 0."""
    position_ids: torch.Tensor
    """``(batch, seq)`` index within the segment, for RoPE or HF position ids."""
    attn_mask: torch.Tensor
    """``(batch, 1, seq, seq)`` bool, True where attention is allowed."""
    prefix_mask: torch.Tensor | None
    """``(batch, seq)`` True inside a bidirectional prefix, or None outside
    ``mode="prefix"``. The training loop needs this to drop those positions
    from every forecasting loss."""
    aux: AuxiliaryInputs
    """The batch's auxiliary inputs with segment-boundary time deltas zeroed."""


class MaskedAttentionMixin:
    """Shared mask construction for every attention-based backbone.

    Holds the context regime and the optional sparsity pattern, and turns a
    batch plus a ``reset_mask`` into an :class:`AttentionInputs`. Both the
    native transformer stack and the ``transformers`` adapter mix this in, so
    ``causal`` / ``prefix`` / ``encoder`` and the BigBird receptive field mean
    exactly the same thing in each, and the leakage tests that prove it are
    written once.
    """

    #: Deterministic seed for the BigBird random blocks outside training. A
    #: paired comparison across backbones cannot attribute noise that changes
    #: between two scoring passes of the same model, so eval must not resample.
    EVAL_SPARSITY_SEED = 20260904

    def _init_masking(
        self,
        *,
        mode: str = "causal",
        prefix_fraction: float = 0.5,
        sparsity: str | None = None,
        block_size: int = 64,
        num_global_blocks: int = 1,
        num_random_blocks: int = 3,
    ) -> None:
        """Record the regime and sparsity pattern; call from ``__init__``."""
        if mode not in ("causal", "prefix", "encoder"):
            raise ValueError(f"unknown attention mode {mode!r}")
        if sparsity not in (None, "bigbird"):
            raise ValueError(f"unknown sparsity {sparsity!r}; use None or 'bigbird'")
        self.mode = mode
        self.prefix_fraction = float(prefix_fraction)
        self.sparsity = sparsity
        self.block_size = int(block_size)
        self.num_global_blocks = int(num_global_blocks)
        self.num_random_blocks = int(num_random_blocks)

    def attention_inputs(
        self, batch: ClinicalSequenceBatch, reset_mask: torch.Tensor | None
    ) -> AttentionInputs:
        """Derive positions, mask and corrected timestamps for one batch."""
        resolved = resolve_reset_mask(reset_mask, batch.concept_ids)
        seq_len = resolved.shape[1]

        prefix_mask = None
        if self.mode == "prefix":
            prefix_mask = sample_prefix_mask(
                resolved,
                self.prefix_fraction,
                generator=None
                if getattr(self, "training", True)
                else torch.Generator(device=resolved.device).manual_seed(
                    self.EVAL_SPARSITY_SEED
                ),
            )

        sparsity_mask = None
        if self.sparsity == "bigbird":
            generator = (
                None
                if getattr(self, "training", True)
                else torch.Generator(device=resolved.device).manual_seed(
                    self.EVAL_SPARSITY_SEED
                )
            )
            sparsity_mask = bigbird_block_mask(
                resolved,
                block_size=self.block_size,
                num_global_blocks=self.num_global_blocks,
                num_random_blocks=self.num_random_blocks,
                generator=generator,
            )

        # Recorded so the loss can find it. Under mode="prefix" the positions
        # inside the prefix attended bidirectionally and have therefore already
        # read their own forecasting targets, so they must be dropped from
        # every next-event loss -- and the cut is drawn here, per forward, so
        # this is the only place that knows where it fell. Read immediately
        # after the forward by ``_SequenceModelBase._drop_prefix_positions``,
        # in the same synchronous call; it is a hand-off, not carried state.
        self.last_prefix_mask = prefix_mask

        aux = batch.aux
        if seq_len > 0:
            aux = aux._replace(
                time_stamps=rebase_time_stamps(aux.time_stamps, resolved)
            )
        return AttentionInputs(
            reset_mask=resolved,
            position_ids=position_ids(resolved),
            attn_mask=build_attn_mask(
                resolved,
                mode=self.mode,
                prefix_mask=prefix_mask,
                sparsity_mask=sparsity_mask,
            ),
            prefix_mask=prefix_mask,
            aux=aux,
        )
