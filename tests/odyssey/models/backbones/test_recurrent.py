"""Tests for RecurrentBackbone: the LSTM / Bi-LSTM control arm.

The two properties worth testing are the two this arm can silently get
wrong: that ``mode='causal'`` really is causal (an ``nn.LSTM`` built with
``bidirectional=True`` by accident produces the right shapes and trains
fine), and that ``mode='encoder'`` does not let trailing padding into the
backward pass (which it would, unpacked, because the backward direction
starts at the last column).
"""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.backbones.recurrent import RecurrentBackbone


VOCAB_SIZE = 40
HIDDEN_SIZE = 16
PADDING_IDX = 0


def _make_batch(
    batch: int, seq_len: int, *, seed: int = 0, lengths: list[int] | None = None
) -> ClinicalSequenceBatch:
    """A batch of random records, optionally right-padded to ``lengths``."""
    gen = torch.Generator().manual_seed(seed)
    concept_ids = torch.randint(1, VOCAB_SIZE, (batch, seq_len), generator=gen)
    if lengths is not None:
        for row, length in enumerate(lengths):
            concept_ids[row, length:] = PADDING_IDX
    return ClinicalSequenceBatch(
        concept_ids=concept_ids,
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len), generator=gen),
            time_stamps=torch.cumsum(torch.rand(batch, seq_len, generator=gen), dim=1),
            ages=torch.rand(batch, seq_len, generator=gen) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len), generator=gen),
            visit_segments=torch.randint(0, 3, (batch, seq_len), generator=gen),
        ),
    )


def _make_backbone(mode: str = "causal") -> RecurrentBackbone:
    """Build in eval mode so the embeddings' dropout doesn't add per-call noise."""
    return RecurrentBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        num_layers=2,
        mode=mode,
    ).eval()


@pytest.mark.parametrize("mode", ["causal", "encoder"])
def test_conforms_to_interface(mode: str) -> None:
    """Both modes are SequenceBackbones returning (batch, seq, hidden_size)."""
    backbone = _make_backbone(mode)
    assert isinstance(backbone, SequenceBackbone)

    hidden_states, state = backbone(_make_batch(3, 12))

    assert hidden_states.shape == (3, 12, HIDDEN_SIZE)
    assert isinstance(state, TimeAwareState)
    assert torch.equal(state.prev_time_stamps, _make_batch(3, 12).aux.time_stamps[:, -1])


def test_causal_mode_does_not_see_the_future() -> None:
    """Changing a token at position >= cut leaves every earlier position alone.

    This is the property that makes the forecasting, time-to-event and value
    heads valid on this arm. A ``bidirectional=True`` slip would break it
    while leaving shapes and training loss looking entirely healthy.
    """
    backbone = _make_backbone("causal")
    cut = 6
    original = _make_batch(2, 12, seed=1)
    perturbed = _make_batch(2, 12, seed=1)
    perturbed.concept_ids[:, cut:] = (perturbed.concept_ids[:, cut:] % 7) + 20

    with torch.no_grad():
        before, _ = backbone(original)
        after, _ = backbone(perturbed)

    assert torch.equal(before[:, :cut], after[:, :cut])
    assert not torch.allclose(before[:, cut:], after[:, cut:])


def test_encoder_mode_sees_the_future() -> None:
    """The bidirectional arm is genuinely bidirectional -- the point of it."""
    backbone = _make_backbone("encoder")
    original = _make_batch(2, 12, seed=1)
    perturbed = _make_batch(2, 12, seed=1)
    perturbed.concept_ids[:, 6:] = (perturbed.concept_ids[:, 6:] % 7) + 20

    with torch.no_grad():
        before, _ = backbone(original)
        after, _ = backbone(perturbed)

    assert not torch.allclose(before[:, 0], after[:, 0])


def test_encoder_mode_ignores_trailing_padding() -> None:
    """A short record's hidden states don't depend on how far the row is padded.

    Unpacked, the backward direction would start on padding and carry it into
    every real position, making a record's representation a function of its
    neighbours' lengths in the same batch.
    """
    backbone = _make_backbone("encoder")
    length = 7
    padded = _make_batch(1, 20, seed=2, lengths=[length])
    short = ClinicalSequenceBatch(
        concept_ids=padded.concept_ids[:, :length],
        aux=AuxiliaryInputs(
            type_ids=padded.aux.type_ids[:, :length],
            time_stamps=padded.aux.time_stamps[:, :length],
            ages=padded.aux.ages[:, :length],
            visit_orders=padded.aux.visit_orders[:, :length],
            visit_segments=padded.aux.visit_segments[:, :length],
        ),
    )

    with torch.no_grad():
        from_padded, _ = backbone(padded)
        from_short, _ = backbone(short)

    torch.testing.assert_close(from_padded[:, :length], from_short)


def test_bidirectional_matches_causal_parameter_budget() -> None:
    """Each direction is half-width, so the two modes are budget-comparable."""
    causal = sum(p.numel() for p in _make_backbone("causal").rnn.parameters())
    encoder = sum(p.numel() for p in _make_backbone("encoder").rnn.parameters())

    assert encoder < causal  # two half-width directions are cheaper than one full


def test_prefix_mode_is_refused() -> None:
    """A single LSTM pass can't express a per-position receptive field."""
    with pytest.raises(ValueError, match="prefix regime"):
        _make_backbone("prefix")


def test_odd_hidden_size_is_refused_in_encoder_mode() -> None:
    """Splitting an odd width across two directions would silently truncate."""
    with pytest.raises(ValueError, match="must be even"):
        RecurrentBackbone(vocab_size=VOCAB_SIZE, hidden_size=15, mode="encoder")
