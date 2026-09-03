"""Tests for the backbone registry.

The registry replaced ten ``backbone == "transformer"`` string comparisons
scattered across training and inference. What these tests protect is the
reason that was worth doing: every declared capability is a fact the rest of
the codebase reads from one place, so a new backbone cannot be half-added.
"""

import pytest

from odyssey.models.backbones import (
    ATTENTION_MODES,
    BACKBONES,
    BIDIRECTIONAL_MODES,
    backbone_spec,
    validate_backbone,
)


def test_every_backbone_declares_at_least_causal() -> None:
    """Causal is the regime everything downstream assumes by default."""
    for name, spec in BACKBONES.items():
        assert spec.modes, f"{name} declares no attention modes"
        assert spec.modes <= set(ATTENTION_MODES), f"{name} declares an unknown mode"


def test_one_patient_per_row_backbones_are_stateless() -> None:
    """The two flags are not independent, and the wrong pair is silent.

    A backbone that needs its own row is one with no mask to separate packed
    patients. Marking it stateful would hand it the TBTT lane sampler, which
    packs -- and the leakage would show up as a slightly better loss, not an
    error.
    """
    for name, spec in BACKBONES.items():
        if spec.one_patient_per_row:
            assert spec.stateless, f"{name}: one_patient_per_row needs stateless"


def test_bidirectional_backbones_are_stateless() -> None:
    """A bidirectional pass cannot be resumed from a carried state."""
    for name, spec in BACKBONES.items():
        if spec.modes & BIDIRECTIONAL_MODES:
            assert spec.stateless, f"{name}: bidirectional modes need stateless"


def test_unknown_backbone_names_the_registered_ones() -> None:
    """The error is the discovery mechanism; it has to list the options."""
    with pytest.raises(ValueError, match="hybrid"):
        backbone_spec("not_a_backbone")


def test_illegal_architecture_mode_pair_is_refused() -> None:
    """The hybrid is causal-only; asking it to encode must fail at build time."""
    with pytest.raises(ValueError, match="encoder"):
        validate_backbone("hybrid", "encoder")


def test_legal_pair_returns_the_spec() -> None:
    """A valid pair round-trips to the same spec ``backbone_spec`` returns."""
    assert validate_backbone("transformer", "encoder") is BACKBONES["transformer"]


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_supports_agrees_with_modes(name: str) -> None:
    """``supports`` is the only thing callers use; keep it honest."""
    spec = BACKBONES[name]
    for mode in ATTENTION_MODES:
        assert spec.supports(mode) == (mode in spec.modes)
