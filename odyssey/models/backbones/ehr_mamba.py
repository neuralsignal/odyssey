"""The EHRMamba paper's backbone: a Mamba-1 stack, ported onto this interface.

This is the architecture from *EHRMamba: Towards Generalizable and Scalable
Foundation Models for Electronic Health Records* (arXiv:2405.14567) --
the original arm, kept so a run can be compared against the thing it
descends from. It is a port, not a reproduction: same block structure and
same Mamba-1 mixer, but reading this project's
:class:`~odyssey.models.embeddings.CachedEHREmbeddings` and sitting under
this project's heads, so a comparison against the hybrid isolates the
sequence mixer rather than the whole training setup.

**Stateless, and that is a property of Mamba-1, not a shortcut.** Mamba-1's
``selective_scan_fn`` takes no initial-state argument at all -- the scan
always begins from zero -- so unlike
:class:`~odyssey.models.backbones.hybrid.EHRHybridBackbone` (whose Mamba-2
kernel accepts ``initial_states``, wired up in ``_make_mamba2_with_state_cls``)
there is nowhere to put a carried state. Truncated BPTT across chunks would
therefore restart the recurrence at every chunk boundary while the training
loop believed it was continuing, which is why this arm is registered
``stateless=True, one_patient_per_row=True``: the sampler hands it whole
records and never asks it to continue one. :meth:`EHRMambaBackbone.forward`
raises rather than accept a carried recurrent state, so a misconfiguration
fails loudly instead of training a subtly wrong model.

Blocks are :class:`~odyssey.models.backbones.hybrid.HybridBlock` with the
attention branch left out, so the prenorm/residual structure is literally the
same code the hybrid arm uses -- only the mixer differs.
"""

from functools import partial

import torch

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import (
    SequenceBackbone,
    TimeAwareState,
    resolve_prev_time_stamps,
)
from odyssey.models.backbones.hybrid import HybridBlock
from odyssey.models.embeddings import CachedEHREmbeddings


class EHRMambaBackbone(SequenceBackbone):
    """A stack of Mamba-1 blocks, one patient per row."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 768,
        padding_idx: int = 0,
        num_hidden_layers: int = 8,
        mamba_state_size: int = 16,
        mamba_conv_size: int = 4,
        mamba_expand: int = 2,
        norm_epsilon: float = 1e-5,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the Mamba-1 backbone.

        The mixer defaults (``d_state=16``, ``d_conv=4``, ``expand=2``) are
        Mamba-1's own, which the paper did not change.
        """
        try:
            # Deferred: mamba-ssm needs CUDA. See hybrid.py's module docstring.
            from mamba_ssm.modules.mamba_simple import Mamba  # noqa: PLC0415
            from mamba_ssm.ops.triton.layer_norm import RMSNorm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "EHRMambaBackbone requires mamba-ssm, which needs a CUDA "
                "build: `uv sync --extra cuda --no-build-isolation`. Use "
                "backbone='lstm' or backbone='transformer' for CPU "
                "development instead."
            ) from exc

        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )

        def _make_block(layer_idx: int) -> HybridBlock:
            mamba_cls = partial(
                Mamba,
                layer_idx=layer_idx,
                d_state=mamba_state_size,
                d_conv=mamba_conv_size,
                expand=mamba_expand,
            )
            return HybridBlock(
                hidden_size,
                mamba_cls,
                None,  # no attention branch: this is the pure Mamba-1 arm
                partial(RMSNorm, eps=norm_epsilon),
            )

        self.layers = torch.nn.ModuleList(
            [_make_block(i) for i in range(num_hidden_layers)]
        )
        self.norm_f = RMSNorm(hidden_size, eps=norm_epsilon)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        ``state`` may carry a previous chunk's last timestamps (which the
        embeddings do use, to get the first time-delta right), but its
        ``recurrent`` field must be ``None``: Mamba-1 cannot resume a scan, so
        a non-``None`` recurrent state means the caller believes this arm is
        continuing a sequence that it is in fact restarting. Likewise a
        ``reset_mask`` with any reset after position 0 means the row packs more
        than one patient, which this arm has no mask to separate.
        """
        if state is not None and state.recurrent is not None:
            raise NotImplementedError(
                "EHRMambaBackbone cannot resume a scan: Mamba-1's "
                "selective_scan_fn has no initial-state argument, so a "
                "carried recurrent state would be silently dropped. Train it "
                "with PackedContextSampler (stateless=True in the registry), "
                "or use backbone='hybrid'/'pure_mamba2' for truncated BPTT."
            )
        if (
            reset_mask is not None
            and reset_mask.shape[1] > 1
            and reset_mask[:, 1:].any()
        ):
            raise NotImplementedError(
                "EHRMambaBackbone does not support packed multi-patient rows: "
                "a scan has no attention mask to block carry-over between "
                "patients. It is registered one_patient_per_row=True for "
                "exactly this reason."
            )

        prev_time_stamps = resolve_prev_time_stamps(state, batch, reset_mask)
        self.embeddings.set_aux_inputs(batch.aux, prev_time_stamps=prev_time_stamps)
        hidden_states = self.embeddings(batch.concept_ids)

        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        residual = hidden_states + residual if residual is not None else hidden_states
        result: torch.Tensor = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        new_state = TimeAwareState(
            recurrent=None, prev_time_stamps=batch.aux.time_stamps[:, -1]
        )
        return result, new_state
