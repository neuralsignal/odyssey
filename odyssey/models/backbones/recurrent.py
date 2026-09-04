"""LSTM / Bi-LSTM arm -- the pre-transformer baseline, on this project's inputs.

The point of this arm is a control: if a Mamba or transformer arm beats a
plain recurrent net by little, the sequence model is not where the signal is
coming from. It is a port of the architecture, not of the paper's training
setup, so it eats the same
:class:`~odyssey.models.embeddings.CachedEHREmbeddings` and produces the same
hidden states as every other backbone.

``mode`` picks the context regime and nothing else has to change:

``causal``
    A unidirectional ``nn.LSTM``. Position *t* has seen tokens ``<= t``, so
    every head -- forecasting, time-to-event, value -- is valid.
``encoder``
    A bidirectional ``nn.LSTM``, each direction ``hidden_size // 2`` wide so
    the concatenation is exactly ``hidden_size`` and the parameter budget
    still matches the causal arm. Every position has seen the whole record,
    so only the final-position supervision the encoder regime sets up is
    meaningful; the training loop is what enforces that.

This backbone is registered ``one_patient_per_row=True``. A recurrent scan has
no attention mask to block cross-patient carry-over, so packing two patients
into one row would leak the first into the second through the hidden state
with nothing to stop it. One patient per row makes the question moot and lets
the fused cuDNN ``nn.LSTM`` path run, instead of the per-token ``LSTMCell``
loop that mid-sequence resets would otherwise force (see
:class:`~odyssey.models.backbones.tiny_gru.TinyGRUBackbone` for what that
costs).
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import (
    SequenceBackbone,
    TimeAwareState,
    resolve_prev_time_stamps,
)
from odyssey.models.embeddings import CachedEHREmbeddings


class RecurrentBackbone(SequenceBackbone):
    """An ``nn.LSTM`` behind this project's clinical embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        padding_idx: int = 0,
        *,
        num_layers: int = 2,
        mode: str = "causal",
        dropout: float = 0.0,
        **embedding_kwargs: object,
    ) -> None:
        """Build the stack; ``mode`` chooses unidirectional or bidirectional."""
        if mode not in ("causal", "encoder"):
            raise ValueError(
                f"RecurrentBackbone supports mode='causal' or 'encoder', "
                f"got {mode!r}. The prefix regime needs a per-position choice "
                f"of receptive field, which a single LSTM pass cannot express."
            )
        bidirectional = mode == "encoder"
        if bidirectional and hidden_size % 2:
            raise ValueError(
                f"mode='encoder' splits hidden_size across two directions, so "
                f"it must be even; got hidden_size={hidden_size}."
            )

        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        self.padding_idx = padding_idx
        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        ``state`` and ``reset_mask`` are accepted for interface compatibility
        and ignored: this arm is registered stateless and one-patient-per-row,
        so the sampler hands it whole records with nothing to carry in and no
        mid-row boundary to reset at.
        """
        prev_time_stamps = resolve_prev_time_stamps(state, batch, reset_mask)
        self.embeddings.set_aux_inputs(batch.aux, prev_time_stamps=prev_time_stamps)
        embeds = self.embeddings(batch.concept_ids)

        # Trailing padding has to be excluded, not just ignored downstream: in
        # the bidirectional case the backward pass starts at the last column,
        # so padding would be the first thing every real position sees.
        lengths = (batch.concept_ids != self.padding_idx).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        hidden_states, _ = pad_packed_sequence(
            output, batch_first=True, total_length=embeds.shape[1]
        )

        new_state = TimeAwareState(
            recurrent=None, prev_time_stamps=batch.aux.time_stamps[:, -1]
        )
        return hidden_states, new_state
