"""Any ``transformers`` architecture as a backbone, with our mask, not theirs.

One adapter instead of one module per architecture. ``AutoConfig.for_model``
builds the config from a ``model_type`` string, ``AutoModel.from_config``
builds a randomly-initialized stack from it, and this class supplies the
clinical embeddings and, critically, the attention mask.

**We own the mask; Hugging Face owns the blocks.** ``transformers`` returns an
already-4D attention mask untouched
(``masking_utils._preprocess_mask_arguments``: *"If the mask is already 4D,
simply return as-is"*), so the block-diagonal, same-segment mask built in
:mod:`odyssey.models.backbones.masks` is what the wrapped stack actually
attends under. That is what makes ``causal`` / ``prefix`` / ``encoder`` mean
the same thing here as in the native transformer arm, rather than depending on
each family's own ``is_decoder`` handling.

Random init, deliberately: these are *architecture* arms, trained on this
project's data behind this project's heads, at a matched budget against the
hybrid. Loading pretrained natural-language weights would confound the
comparison with a different pretraining corpus, and no public checkpoint is
pretrained on our token vocabulary anyway.

Two behaviours to know about, neither of them a bug:

1. **HF re-embeds on top of ``inputs_embeds``.** ``BertEmbeddings.forward``
   adds its own learned absolute position embeddings and token-type
   embeddings, then applies LayerNorm and dropout, to whatever it is handed.
   So a wrapped BERT carries a second positional signal and a second
   LayerNorm after :class:`~odyssey.models.embeddings.ClinicalEventEmbeddings`
   has already normalized. Harmless, and measurable: ``use_hf_embeddings=False``
   (the default) still goes through their embedding layer -- there is no
   supported way to bypass it generically -- but explicit segment-reset
   ``position_ids`` are passed so a packed patient gets the same positions it
   would get alone.
2. **``max_position_embeddings`` must cover ``max_context``**, or a long row
   indexes past the position table. Set from the run's ``max_context``.

Families are refused rather than wrapped when their attention is not a plain
masked softmax -- see :data:`REFUSED`. A silently-ignored mask is the failure
mode that matters here: the model trains, converges, and has seen the future,
and nothing downstream would notice. That is also why every family, refused or
not, is checked at construction by :meth:`HFBackbone.assert_no_leakage`
rather than trusted to a whitelist that rots on the next ``transformers``
release.
"""

from typing import Any, cast

import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.backbones.masks import MaskedAttentionMixin
from odyssey.models.embeddings import CachedEHREmbeddings


REFUSED: dict[str, str] = {
    "longformer": (
        "Longformer builds its own sliding-window + global attention mask "
        "internally and does not honour a custom 4D mask, so packed rows "
        "would leak across patients undetected. Use backbone='bigbird' for a "
        "sparse receptive field under our own mask."
    ),
    "modernbert": (
        "ModernBERT alternates local and global attention with its own mask "
        "construction, which a custom 4D mask does not override."
    ),
    "deberta": "DeBERTa's disentangled attention does not take a 4D mask.",
    "deberta_v2": "DeBERTa's disentangled attention does not take a 4D mask.",
    "mamba": (
        "Mamba has no attention mask at all -- its receptive field is a "
        "property of the scan. Use backbone='ehr_mamba' (Mamba-1) or "
        "'pure_mamba2', which handle packing by giving each patient its own "
        "row instead of by masking."
    ),
    "mamba2": (
        "Mamba-2 has no attention mask at all. Use backbone='pure_mamba2', "
        "which carries recurrent state across chunks properly."
    ),
    "big_bird": (
        "HF BigBird refuses block-sparse attention as a decoder "
        "(modeling_big_bird.py: 'BigBird cannot be used as a decoder when "
        "config.attention_type != original_full') and silently downgrades to "
        "dense otherwise. Use backbone='bigbird', which implements the "
        "receptive field as a mask over our own attention."
    ),
}

#: Config keys every wrapped family understands, mapped by
#: ``PretrainedConfig.attribute_map`` where a family spells them differently
#: (GPT-2's ``n_embd`` / ``n_layer`` / ``n_head`` / ``n_positions``).
_CANONICAL_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "max_position_embeddings",
)


class HFBackbone(MaskedAttentionMixin, SequenceBackbone):
    """A ``transformers`` stack behind this project's embeddings and masks."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 256,
        padding_idx: int = 0,
        *,
        model_type: str = "bert",
        mode: str = "causal",
        max_context: int = 4096,
        prefix_fraction: float = 0.5,
        verify_no_leakage: bool = True,
        hf_overrides: dict[str, Any] | None = None,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        ffn_mult: int = 4,
        **kwargs: Any,
    ) -> None:
        """Build ``model_type``'s stack at this width, depth and context.

        Extra keyword arguments that are not embedding options are passed to
        the HF config, so an arm can set ``intermediate_size``,
        ``hidden_dropout_prob`` and so on through
        ``TrainingConfig.backbone_kwargs`` without this signature growing.
        """
        try:
            from transformers import AutoConfig, AutoModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "backbone requires transformers, an optional extra: "
                "`uv sync --extra text`."
            ) from exc

        if model_type in REFUSED:
            raise ValueError(
                f"model_type={model_type!r} is not supported: {REFUSED[model_type]}"
            )

        super().__init__()
        self._init_masking(mode=mode, prefix_fraction=prefix_fraction)
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_context = max_context
        """The widest row this backbone can take. Unlike the native
        transformer arm, which places positions with per-segment RoPE and has
        no upper bound, most wrapped families index a *learned* position table
        sized here -- so a wider row is an out-of-range index deep inside
        ``transformers``, not a graceful extrapolation. Callers that probe a
        model at a fixed width (:func:`odyssey.utils.env_fingerprint.numeric_canary`)
        read this to stay inside it."""

        embedding_keys = {
            "type_vocab_size",
            "max_num_visits",
            "time_embeddings_size",
            "visit_order_size",
            "layer_norm_eps",
            "hidden_dropout_prob",
            "use_values",
            "use_value_fourier",
        }
        embedding_kwargs = {k: v for k, v in kwargs.items() if k in embedding_keys}
        config_kwargs = {k: v for k, v in kwargs.items() if k not in embedding_keys}
        config_kwargs.update(hf_overrides or {})

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )

        config = AutoConfig.for_model(
            model_type,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            # Every row this backbone ever sees is at most max_context long,
            # and a learned position table shorter than that indexes out of
            # bounds on the first long batch rather than at construction.
            max_position_embeddings=max_context,
            pad_token_id=padding_idx,
            # Our mask decides the regime; is_decoder would only pick which
            # mask HF builds when we do not hand it one.
            is_decoder=False,
            **config_kwargs,
        )
        self._check_config_applied(config, hidden_size, num_hidden_layers, max_context)
        _scale_feedforward(config, hidden_size, ffn_mult, config_kwargs)
        _clamp_special_tokens(config, vocab_size, padding_idx)
        # AutoModel is typed only when the optional `transformers` extra is
        # installed, so a direct call is a no-untyped-call error in one
        # environment and clean in the other -- and a type: ignore for it is
        # flagged unused in the second. An explicit Any alias is right in both.
        auto_model: Any = AutoModel
        self.model = auto_model.from_config(config)

        if verify_no_leakage and mode == "causal":
            self.assert_no_leakage()

    @staticmethod
    def _check_config_applied(
        config: Any, hidden_size: int, num_hidden_layers: int, max_context: int
    ) -> None:
        """Fail loudly if a family ignored the sizes we asked for.

        ``PretrainedConfig.__setattr__`` routes canonical names through each
        family's ``attribute_map`` (GPT-2 stores ``hidden_size`` as
        ``n_embd``), but a family with no mapping and no such field would
        silently keep its own default -- producing an arm that is not the size
        the comparison assumes it is.
        """
        expected = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "max_position_embeddings": max_context,
        }
        for key, want in expected.items():
            got = getattr(config, key, None)
            if got != want:
                raise ValueError(
                    f"{type(config).__name__} ignored {key}={want} (it reports "
                    f"{got!r}); this architecture cannot be sized through the "
                    "generic adapter"
                )

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        Stateless: ``state`` is accepted for interface conformance and always
        ignored, and the returned state carries only the last timestamp.
        """
        seq_len = batch.concept_ids.shape[1]
        if seq_len > self.max_context:
            raise ValueError(
                f"{self.model_type!r} was built for rows of at most "
                f"{self.max_context} tokens (its learned position table is "
                f"that size) but got {seq_len}. Raise the run's max_context, "
                f"or hand this backbone narrower rows -- the alternative is "
                f"an out-of-range index inside transformers with no useful "
                f"message."
            )
        prepared = self.attention_inputs(batch, reset_mask)
        self.embeddings.set_aux_inputs(prepared.aux, prev_time_stamps=None)
        inputs_embeds = self.embeddings(batch.concept_ids)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attn_mask,
            position_ids=prepared.position_ids,
        )
        hidden_states = cast(torch.Tensor, out.last_hidden_state)
        new_state = TimeAwareState(
            recurrent=None, prev_time_stamps=batch.aux.time_stamps[:, -1]
        )
        return hidden_states, new_state

    @torch.no_grad()
    def assert_no_leakage(self, seq_len: int = 12) -> None:
        """Raise unless changing a late token leaves earlier states untouched.

        The only check that actually proves the wrapped family honoured our
        mask. A whitelist encodes today's ``transformers`` behaviour and rots
        on the next release; this does not. Cheap enough to run at
        construction: one tiny batch, two forwards, no gradients.
        """
        was_training = self.training
        self.eval()
        try:
            batch = _probe_batch(self.embeddings, seq_len)
            baseline, _ = self.forward(batch)
            perturbed_ids = batch.concept_ids.clone()
            cut = seq_len // 2
            perturbed_ids[:, cut:] = (perturbed_ids[:, cut:] + 1) % max(
                2, int(self.embeddings.embeddings.word_embeddings.num_embeddings)
            )
            changed, _ = self.forward(batch._replace(concept_ids=perturbed_ids))
            if not torch.equal(baseline[:, :cut], changed[:, :cut]):
                worst = (baseline[:, :cut] - changed[:, :cut]).abs().max().item()
                raise ValueError(
                    f"model_type={self.model_type!r} leaks future context: "
                    f"changing tokens at position >= {cut} moved earlier "
                    f"hidden states by up to {worst:.3e}. Its attention does "
                    "not honour the 4D mask this adapter supplies, so it "
                    "cannot be used as a causal backbone. Add it to "
                    "odyssey.models.backbones.hf.REFUSED."
                )
        finally:
            self.train(was_training)


def _probe_batch(
    embeddings: CachedEHREmbeddings, seq_len: int
) -> ClinicalSequenceBatch:
    """Return a small synthetic batch for :meth:`HFBackbone.assert_no_leakage`."""
    vocab_size = int(embeddings.embeddings.word_embeddings.num_embeddings)
    type_vocab = int(embeddings.embeddings.token_type_embeddings.num_embeddings)
    generator = torch.Generator().manual_seed(0)
    shape = (1, seq_len)
    concept_ids = torch.randint(1, max(2, vocab_size), shape, generator=generator)
    return ClinicalSequenceBatch(
        concept_ids=concept_ids,
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, max(1, type_vocab), shape, generator=generator),
            time_stamps=torch.arange(seq_len, dtype=torch.float32)
            .unsqueeze(0)
            .expand(shape)
            .contiguous(),
            ages=torch.full(shape, 50.0),
            visit_orders=torch.zeros(shape, dtype=torch.long),
            visit_segments=torch.ones(shape, dtype=torch.long),
        ),
    )


def _scale_feedforward(
    config: Any, hidden_size: int, ffn_mult: int, explicit: dict[str, Any]
) -> None:
    """Scale the MLP width with ``hidden_size`` unless the caller set it.

    Every family ships an ``intermediate_size`` default tuned for its own
    published width (BERT 3072, Llama 11008) and does **not** derive it from
    whatever ``hidden_size`` we ask for. Left alone, a 256-wide "small" arm
    would carry an 11008-wide MLP and dwarf the hybrid it is supposed to be
    matched against -- a parameter-budget error that no test would catch and
    that would be read as an architecture result. GPT-2 already derives its
    own (``n_inner=None`` means ``4 * n_embd``), so it is left alone.
    """
    if "intermediate_size" in explicit:
        return
    if getattr(config, "intermediate_size", None) is not None:
        config.intermediate_size = ffn_mult * hidden_size


def _clamp_special_tokens(config: Any, vocab_size: int, padding_idx: int) -> None:
    """Point out-of-range bos/eos ids at padding, silencing a spurious warning.

    Families carry their pretrained tokenizer's special-token ids (GPT-2's
    50256), which are outside our clinical vocabulary. Nothing in this project
    generates text, so the ids are unused -- but left out of range they emit a
    config warning on every construction that looks like a real problem.
    """
    for name in ("bos_token_id", "eos_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value >= vocab_size:
            setattr(config, name, padding_idx)
