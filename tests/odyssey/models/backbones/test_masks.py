"""Tests for the shared attention-mask arithmetic.

Every masked backbone -- the native transformer, BigBird, and every wrapped
``transformers`` family -- gets its receptive field from these functions, so a
bug here is a bug in all of them at once, and the symptom is a better loss
rather than a crash.
"""

import pytest
import torch

from odyssey.models.backbones.masks import (
    bigbird_block_mask,
    build_attn_mask,
    position_ids,
    rebase_time_stamps,
    sample_prefix_mask,
    segment_ids,
    segment_lengths,
)


def _resets(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def test_segment_ids_group_positions_by_packed_patient() -> None:
    """Only equality between ids is meaningful -- so that is what is tested.

    The absolute values are a cumulative sum and carry no other meaning; a
    test asserting them would break on an implementation change that is not
    a behaviour change.
    """
    ids = segment_ids(_resets([[True, False, False, True, False]]))[0]

    assert (ids[:3] == ids[0]).all()
    assert (ids[3:] == ids[3]).all()
    assert ids[0] != ids[3]


def test_position_ids_restart_per_segment() -> None:
    """RoPE must see a packed patient's own positions, not the row's."""
    resets = _resets([[True, False, False, True, False]])
    assert position_ids(resets).tolist() == [[0, 1, 2, 0, 1]]


def test_segment_lengths_counts_each_segment() -> None:
    """Used to size per-segment work; must not count across a boundary."""
    resets = _resets([[True, False, False, True, False]])
    assert segment_lengths(resets).tolist() == [[3, 3, 3, 2, 2]]


def test_rebase_time_stamps_zeroes_the_delta_at_each_segment_start() -> None:
    """The boundary delta goes to 0; every other delta is left exactly alone.

    It is the *deltas* the time embedding reads, so that is what has to be
    corrected -- the absolute base stays the row's own, which nothing sees.
    """
    resets = _resets([[True, False, True, False]])
    times = torch.tensor([[10.0, 12.0, 100.0, 105.0]])

    rebased = rebase_time_stamps(times, resets)[0]
    deltas = (rebased[1:] - rebased[:-1]).tolist()

    assert deltas == [2.0, 0.0, 5.0]


def test_causal_mask_blocks_the_future_and_other_patients() -> None:
    """The two guarantees that make packing safe, in one mask."""
    resets = _resets([[True, False, True, False]])

    mask = build_attn_mask(resets, mode="causal")[0, 0]

    assert not mask[0, 1]  # no attending forward
    assert mask[1, 0]  # attending back within the segment
    assert not mask[2, 1]  # no attending into the previous patient
    assert mask[3, 2]  # attending back within the second patient


def test_encoder_mask_is_bidirectional_within_a_segment_only() -> None:
    """Bidirectional does not mean cross-patient."""
    resets = _resets([[True, False, True, False]])

    mask = build_attn_mask(resets, mode="encoder")[0, 0]

    assert mask[0, 1]  # forward within the segment: allowed here
    assert not mask[1, 2]  # still never across the segment boundary
    assert not mask[2, 0]


def test_prefix_mask_lies_between_causal_and_encoder() -> None:
    """PrefixLM is bidirectional inside the prefix and causal after it."""
    resets = _resets([[True] + [False] * 7])
    prefix = sample_prefix_mask(resets, fraction=0.5, generator=torch.Generator())

    causal = build_attn_mask(resets, mode="causal")
    encoder = build_attn_mask(resets, mode="encoder")
    prefix_mask = build_attn_mask(resets, mode="prefix", prefix_mask=prefix)

    assert (prefix_mask >= causal).all()  # at least as permissive as causal
    assert (prefix_mask <= encoder).all()  # never more than fully bidirectional


def test_bigbird_receptive_field_is_a_subset_of_causal() -> None:
    """Sparse attention may only remove edges, never add a look-ahead."""
    resets = _resets([[True] + [False] * 31])
    sparsity = bigbird_block_mask(
        resets,
        block_size=4,
        num_global_blocks=1,
        num_random_blocks=1,
        generator=torch.Generator().manual_seed(0),
    )

    causal = build_attn_mask(resets, mode="causal")
    sparse = build_attn_mask(resets, mode="causal", sparsity_mask=sparsity)

    assert (sparse <= causal).all()
    assert sparse.sum() < causal.sum()  # it actually removed something


def test_every_position_can_attend_to_itself() -> None:
    """A row with no visible position at all produces NaN after softmax."""
    resets = _resets([[True, False, True, False, False]])

    for mode in ("causal", "encoder"):
        mask = build_attn_mask(resets, mode=mode)[0, 0]
        assert mask.diagonal().all(), mode


def test_unknown_mode_is_refused() -> None:
    """A typo in a config field must not silently pick a default regime."""
    with pytest.raises(ValueError, match="mode"):
        build_attn_mask(_resets([[True, False]]), mode="bidirectional")
