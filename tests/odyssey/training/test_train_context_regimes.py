"""End-to-end tests for the bidirectional context regimes and the arms using them.

The regime's whole content is a restriction: a bidirectional pass has already
read the tokens that the forecasting, time-to-event and value heads are
supposed to predict, so those three are switched off and supervision moves to
the landmark. Nothing about that fails loudly if it is wrong -- the loss goes
down either way -- so it is tested directly rather than through a converging
run.

The shard fixture is imported from the transformer integration test rather
than copied a third time; it is a plain function with no GPU dependency.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.sequence_model import BaselineSequenceModel, ForecastObjective
from odyssey.training.event_targets import EventHazardTargets
from odyssey.training.train import TrainingConfig, restrict_to_landmark
from tests.odyssey.training.test_train_transformer import _write_shards


def _chunk(lanes: int = 2, seq_len: int = 6) -> StreamingChunk:
    """Build a chunk with one patient per row, ending at the last position."""
    ones = torch.ones(lanes, seq_len)
    patient_end = torch.zeros(lanes, seq_len, dtype=torch.bool)
    patient_end[:, -1] = True
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.arange(1, lanes * seq_len + 1).reshape(lanes, seq_len),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(lanes, seq_len, dtype=torch.long),
                time_stamps=ones.cumsum(dim=1).double(),
                ages=ones * 40.0,
                visit_orders=torch.zeros(lanes, seq_len, dtype=torch.long),
                visit_segments=torch.zeros(lanes, seq_len, dtype=torch.long),
            ),
        ),
        targets=torch.arange(2, lanes * seq_len + 2).reshape(lanes, seq_len),
        reset_mask=patient_end.flip(1),
        real_mask=~patient_end,
        subject_ids=torch.zeros(lanes, seq_len, dtype=torch.long),
        patient_end=patient_end,
        visit_ids=torch.zeros(lanes, seq_len, dtype=torch.long),
        visit_end=patient_end,
    )


def _event_targets(lanes: int = 2, seq_len: int = 6) -> EventHazardTargets:
    shape = (lanes, seq_len, 3)
    return EventHazardTargets(
        gap_hours=torch.ones(shape),
        observed=torch.ones(shape, dtype=torch.bool),
        at_risk=torch.ones(shape, dtype=torch.bool),
    )


def test_restrict_to_landmark_empties_the_leaky_supervision() -> None:
    """Forecast, time and value all read ``real_mask`` or ``targets``.

    Both have to be emptied: the plain cross-entropy path scores every
    non-padding target and never consults ``real_mask``.
    """
    scored, _ = restrict_to_landmark(_chunk(), None)

    assert not scored.real_mask.any()
    assert (scored.targets == 0).all()


def test_restrict_to_landmark_keeps_event_hazards_at_the_landmark() -> None:
    """The hazards survive, because their targets are not in the sequence.

    They come from the onset/censoring tables, so the landmark position's
    target depends on what happens *after* the record -- which a
    bidirectional pass has not seen.
    """
    chunk = _chunk()
    _, targets = restrict_to_landmark(chunk, _event_targets())

    assert targets is not None
    assert torch.equal(targets.at_risk.any(dim=-1), chunk.patient_end)


def test_restrict_to_landmark_leaves_the_inputs_alone() -> None:
    """Supervision is restricted; the model still reads the whole record."""
    chunk = _chunk()
    scored, _ = restrict_to_landmark(chunk, None)

    assert torch.equal(scored.batch.concept_ids, chunk.batch.concept_ids)
    assert torch.equal(scored.patient_end, chunk.patient_end)


def test_objective_carries_the_regime_flag(tmp_path: Path) -> None:
    """The flag is built once, from the config, for both loops to read."""
    assert not ForecastObjective().encoder_regime


@pytest.mark.parametrize("backbone", ["bert", "lstm"])
def test_encoder_arm_trains_end_to_end(tmp_path: Path, backbone: str) -> None:
    """The real training script, in the real regime, on real-shaped shards.

    Covers the pieces that only meet each other in ``train()``: the random
    landmark truncation, ``pack=False``, the registry's mode validation, the
    restriction above, and a checkpoint that still writes.
    """
    pytest.importorskip("transformers")
    from odyssey.training.train import train  # noqa: PLC0415

    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    overrides: dict[str, Any] = {
        "train_shard_dir": str(train_dir),
        "tuning_shard_dir": str(tuning_dir),
        "output_dir": str(output_dir),
        "backbone": backbone,
        "attention_mode": "encoder",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "attn_num_heads": 4,
        "embedding_dim": 8,
        "vocab_min_count": 1,
        "quantile_min_count": 1,
        "num_lanes": 2,
        "max_context": 32,
        "num_epochs": 1,
        "log_every": 2,
        "eval_every": 4,
        "eval_max_chunks": 2,
        "checkpoint_every": 4,
    }

    assert train(TrainingConfig(**overrides)) == output_dir
    assert (output_dir / "checkpoint_final.pt").exists()


def test_baseline_model_is_refused_in_a_bidirectional_regime() -> None:
    """Nothing would be left to supervise, and the run would not error.

    The baseline has no concept bottleneck and no event heads, so the
    restriction above would leave it training against a constant zero.
    """
    from odyssey.training.train import build_model  # noqa: PLC0415

    config = TrainingConfig(
        train_shard_dir="",
        tuning_shard_dir="",
        output_dir="",
        model_kind="baseline",
        backbone="bert",
        attention_mode="encoder",
    )

    with pytest.raises(ValueError, match="baseline"):
        build_model(config, vocab_size=32, num_concepts=4)


# ---------------------------------------------------------------------------
# The prefix regime
# ---------------------------------------------------------------------------


def test_prefix_positions_are_dropped_from_supervision() -> None:
    """The cut is drawn inside the forward, so the loss reads it from there.

    Positions before the cut attended both ways and have already read their own
    next token; positions after it are causal and stay supervised -- that split
    is what separates ``prefix`` from ``encoder``.
    """
    backbone = TransformerBackbone(
        vocab_size=64, hidden_size=16, num_hidden_layers=1, num_heads=2, mode="prefix"
    )
    model = BaselineSequenceModel(backbone, vocab_size=64)
    chunk = _chunk()

    # Stand in for what a forward would have recorded: the first half is prefix.
    prefix = torch.zeros_like(chunk.real_mask)
    prefix[:, :3] = True
    backbone.last_prefix_mask = prefix

    scored, targets = model._drop_prefix_positions(chunk, _event_targets())

    assert not scored.real_mask[:, :3].any()  # prefix half: nothing supervised
    assert scored.real_mask[:, 3:].any()  # causal half: still supervised
    assert targets is not None
    assert not targets.at_risk[:, :3].any()


def test_no_prefix_leaves_the_chunk_untouched() -> None:
    """Every non-prefix backbone leaves ``last_prefix_mask`` at None."""
    model = BaselineSequenceModel(
        TransformerBackbone(
            vocab_size=64, hidden_size=16, num_hidden_layers=1, num_heads=2
        ),
        vocab_size=64,
    )
    chunk = _chunk()

    scored, targets = model._drop_prefix_positions(chunk, None)

    assert scored is chunk
    assert targets is None


def test_prefix_arm_trains_end_to_end(tmp_path: Path) -> None:
    """The regime through the real script, including the mask/loss hand-off."""
    from odyssey.training.train import train  # noqa: PLC0415

    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    config = TrainingConfig(
        train_shard_dir=str(train_dir),
        tuning_shard_dir=str(tuning_dir),
        output_dir=str(output_dir),
        backbone="transformer",
        attention_mode="prefix",
        hidden_size=32,
        num_hidden_layers=2,
        attn_num_heads=4,
        embedding_dim=8,
        vocab_min_count=1,
        quantile_min_count=1,
        num_lanes=2,
        max_context=32,
        num_epochs=1,
        log_every=2,
        eval_every=4,
        eval_max_chunks=2,
        checkpoint_every=4,
    )

    assert train(config) == output_dir
    assert (output_dir / "checkpoint_final.pt").exists()
