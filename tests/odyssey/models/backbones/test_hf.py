"""Tests for the ``transformers`` adapter.

The whole adapter rests on one claim: ``transformers`` returns an already-4D
attention mask untouched, so our block-diagonal, same-segment mask is what the
wrapped stack actually attends under. If that stops being true for a family,
the model still trains -- it just gets to see the future. So the tests here
are mostly about the probe that catches exactly that, and about the probe
itself being able to fail.
"""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch


transformers = pytest.importorskip("transformers")

from odyssey.models.backbones.hf import REFUSED, HFBackbone  # noqa: E402


VOCAB_SIZE = 50
HIDDEN_SIZE = 32
FAMILIES = ["bert", "roberta", "gpt2", "llama"]


def _make_backbone(model_type: str = "bert", mode: str = "causal") -> HFBackbone:
    return HFBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        model_type=model_type,
        mode=mode,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_context=64,
    ).eval()


def _make_batch(batch: int, seq_len: int, *, seed: int = 0) -> ClinicalSequenceBatch:
    gen = torch.Generator().manual_seed(seed)
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len), generator=gen),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len), generator=gen),
            time_stamps=torch.cumsum(torch.rand(batch, seq_len, generator=gen), dim=1),
            ages=torch.rand(batch, seq_len, generator=gen) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len), generator=gen),
            visit_segments=torch.randint(0, 3, (batch, seq_len), generator=gen),
        ),
    )


@pytest.mark.parametrize("model_type", FAMILIES)
def test_family_builds_and_produces_our_hidden_size(model_type: str) -> None:
    """Construction runs the leakage probe, so this also asserts causality."""
    backbone = _make_backbone(model_type)

    hidden_states, _ = backbone(_make_batch(2, 16))

    assert hidden_states.shape == (2, 16, HIDDEN_SIZE)


@pytest.mark.parametrize("model_type", FAMILIES)
def test_feedforward_width_follows_hidden_size(model_type: str) -> None:
    """Otherwise a 'small' arm silently carries Llama's 11008-wide MLP.

    Nothing would fail; the arm would just be many times the budget it was
    supposed to be matched at, and the result would read as an architecture
    difference. Comparing families against each other is the check: at the
    same width and depth they should be within a small factor.
    """
    params = sum(p.numel() for p in _make_backbone(model_type).model.parameters())

    assert params < 100_000, f"{model_type} did not scale its MLP to hidden_size"


def test_leakage_probe_catches_a_bidirectional_mask() -> None:
    """The safety net has to be able to fail, or it is not a safety net.

    Forcing a causal-built model to hand its stack an encoder mask is exactly
    what a family silently ignoring our 4D mask would look like from outside.
    """
    backbone = _make_backbone("bert")
    backbone.mode = "encoder"

    with pytest.raises(ValueError, match="leaks future context"):
        backbone.assert_no_leakage()


def test_causal_mode_does_not_see_the_future() -> None:
    """The same property the probe asserts, checked through a real forward."""
    backbone = _make_backbone("bert")
    cut = 8
    original = _make_batch(2, 16, seed=3)
    perturbed = _make_batch(2, 16, seed=3)
    perturbed.concept_ids[:, cut:] = (perturbed.concept_ids[:, cut:] % 11) + 20

    with torch.no_grad():
        before, _ = backbone(original)
        after, _ = backbone(perturbed)

    torch.testing.assert_close(before[:, :cut], after[:, :cut])


@pytest.mark.parametrize("model_type", sorted(REFUSED))
def test_refused_families_explain_themselves(model_type: str) -> None:
    """Each refusal names the alternative; a bare 'unsupported' is useless."""
    with pytest.raises(ValueError, match="not supported"):
        _make_backbone(model_type)

    assert "backbone=" in REFUSED[model_type] or "attention" in REFUSED[model_type]
