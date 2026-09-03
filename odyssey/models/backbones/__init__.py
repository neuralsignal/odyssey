"""Sequence backbones the concept bottleneck can sit on top of, and the registry.

An arm of this project is a **pair**, not a name: an *architecture* (what the
blocks are -- Mamba-2, attention, an LSTM, any ``transformers`` model type) and
a *context regime* (what each position is allowed to see). Those two axes are
independent, and conflating them is what makes "add BERT" sound harder than it
is: BERT is not a bidirectional model, it is a transformer stack that is
usually *run* bidirectionally.

:data:`BACKBONES` is the one place that mapping lives. Adding an architecture
is one entry here plus one module; nothing downstream needs a new ``if``.

Context regimes (:data:`ATTENTION_MODES`), in order of what they cost:

``causal``
    Position *t* sees positions <= *t*. Every head and loss this project has is
    valid, and evaluation is a single streaming pass with
    :func:`odyssey.inference.alerts._landmark_mask` picking scoring positions.
    The default, and the only regime that existed before the registry.

``prefix``
    Bidirectional among a sampled prefix of each row, causal after it, with the
    loss zeroed on prefix positions (they saw their own targets). Standard
    PrefixLM/UL2 training. Buys bidirectional *encoding of history* while
    keeping every existing loss and the single-pass evaluation.

``encoder``
    Fully bidirectional over a record truncated at a landmark, supervised from
    the final position only. What CEHR-BERT and the paper's Bi-LSTM actually
    are. Forecast/time/value heads are disabled -- definitionally, this regime
    cannot do them -- and evaluation costs one forward *per landmark* instead of
    one pass per patient. See ``docs/multi_backbone_research.md`` section 8.

Not every pair is legal, which is why :class:`BackboneSpec` carries ``modes``.
An SSM's receptive field is a property of its scan, not of a mask, so Mamba
arms are causal-only: a bidirectional SSM means a second reversed scan, which
is a different architecture rather than a different mode.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from odyssey.models.backbones.base import SequenceBackbone


CAUSAL = "causal"
PREFIX = "prefix"
ENCODER = "encoder"

ATTENTION_MODES: tuple[str, ...] = (CAUSAL, PREFIX, ENCODER)
"""Every context regime the training loop knows how to drive."""

BIDIRECTIONAL_MODES: frozenset[str] = frozenset({PREFIX, ENCODER})
"""Regimes in which some position sees a position after it. These are exactly
the regimes where the forecasting, time-to-next-event and value heads would be
supervised on tokens they can already see, so the training loop must mask (or
disable) those losses -- see
:func:`odyssey.training.train.supervised_position_mask`."""


@dataclass(frozen=True)
class BackboneSpec:
    """How to build one architecture, and what the rest of the code must know.

    ``build`` takes the run's :class:`~odyssey.training.train.TrainingConfig`
    (typed ``Any`` here only to avoid a circular import; every builder reads it
    with ``getattr`` the way ``build_model`` always has) plus ``vocab_size``,
    and defers any heavy import -- ``mamba-ssm`` needs a CUDA build and
    ``transformers`` is an optional extra, so neither may be imported at module
    scope.
    """

    build: Callable[..., "SequenceBackbone"]
    stateless: bool
    """No cross-call recurrent state. Drives two decisions that used to be
    spelled ``backbone == "transformer"``: the training loop uses
    :class:`~odyssey.data.packed_context.PackedContextSampler` over whole (or
    head-truncated) patients rather than
    :class:`~odyssey.data.streaming.PackedLaneSampler`'s TBTT chunking, and the
    inference paths that cannot handle a re-encoded context refuse instead of
    silently producing wrong numbers."""

    modes: frozenset[str] = frozenset({CAUSAL})
    """Context regimes this architecture can express."""

    one_patient_per_row: bool = False
    """This backbone cannot honour ``reset_mask`` in the middle of a row, so
    packing several patients into one row would leak across them. True for
    every recurrent stateless arm (an SSM or RNN scan runs straight through a
    segment boundary and there is no mask to stop it) and for bidirectional
    arms under ``encoder`` (the supervised position must be the row's last
    real one). Forces ``pack=False`` on
    :class:`~odyssey.data.packed_context.PackedContextSampler`."""

    def supports(self, mode: str) -> bool:
        """Return whether this architecture can be run in ``mode``."""
        return mode in self.modes


def _extra_kwargs(config: Any) -> dict[str, Any]:
    """Return this run's free-form per-architecture overrides.

    ``TrainingConfig.backbone_kwargs`` exists so an arm can carry its own
    hyperparameters (an HF ``model_type``'s ``intermediate_size``, an LSTM's
    ``num_layers``) without a ``TrainingConfig`` field per architecture. It
    round-trips through ``asdict`` into ``config.json`` like everything else.
    """
    return dict(getattr(config, "backbone_kwargs", None) or {})


def _common(config: Any, *, vocab_size: int) -> dict[str, Any]:
    """Return the constructor arguments every backbone in this package takes."""
    from odyssey.data.vocabulary import PAD_ID  # noqa: PLC0415

    return {
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "padding_idx": PAD_ID,
        "use_values": bool(getattr(config, "value_embeddings", False)),
        "use_value_fourier": bool(getattr(config, "value_fourier", False)),
    }


def _build_hybrid(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the Mamba-2 + attention hybrid (needs a CUDA ``mamba-ssm``)."""
    from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: PLC0415

    return EHRHybridBackbone(
        **_common(config, vocab_size=vocab_size),
        num_hidden_layers=config.num_hidden_layers,
        mamba_state_size=config.mamba_state_size,
        mamba_headdim=config.mamba_headdim,
        mamba_chunk_size=config.mamba_chunk_size,
        attn_num_heads=config.attn_num_heads,
        **_extra_kwargs(config),
    )


def _build_pure_mamba2(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the hybrid with its attention branch removed."""
    from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: PLC0415

    return EHRHybridBackbone(
        **_common(config, vocab_size=vocab_size),
        num_hidden_layers=config.num_hidden_layers,
        mamba_state_size=config.mamba_state_size,
        mamba_headdim=config.mamba_headdim,
        mamba_chunk_size=config.mamba_chunk_size,
        attn_num_heads=config.attn_num_heads,
        use_attention=False,
        **_extra_kwargs(config),
    )


def _build_ehr_mamba(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the paper's Mamba-1 stack (stateless; needs CUDA ``mamba-ssm``)."""
    from odyssey.models.backbones.ehr_mamba import EHRMambaBackbone  # noqa: PLC0415

    return EHRMambaBackbone(
        **_common(config, vocab_size=vocab_size),
        num_hidden_layers=config.num_hidden_layers,
        **_extra_kwargs(config),
    )


def _build_transformer(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the modern-vanilla decoder-only transformer control."""
    from odyssey.models.backbones.transformer import (  # noqa: PLC0415
        TransformerBackbone,
    )

    return TransformerBackbone(
        **_common(config, vocab_size=vocab_size),
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.attn_num_heads,
        mode=getattr(config, "attention_mode", CAUSAL),
        **_extra_kwargs(config),
    )


def _build_bigbird(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the transformer control restricted to BigBird's receptive field."""
    from odyssey.models.backbones.transformer import (  # noqa: PLC0415
        TransformerBackbone,
    )

    kwargs = _extra_kwargs(config)
    kwargs.setdefault("sparsity", "bigbird")
    return TransformerBackbone(
        **_common(config, vocab_size=vocab_size),
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.attn_num_heads,
        mode=getattr(config, "attention_mode", CAUSAL),
        **kwargs,
    )


def _build_tiny_gru(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the CPU/CI stand-in (see :mod:`odyssey.models.backbones.tiny_gru`)."""
    from odyssey.models.backbones.tiny_gru import TinyGRUBackbone  # noqa: PLC0415

    return TinyGRUBackbone(
        **_common(config, vocab_size=vocab_size),
        num_layers=config.num_hidden_layers,
        **_extra_kwargs(config),
    )


def _build_recurrent(config: Any, *, vocab_size: int) -> "SequenceBackbone":
    """Build the LSTM/GRU arm (``bidirectional`` via ``attention_mode``)."""
    from odyssey.models.backbones.recurrent import (  # noqa: PLC0415
        RecurrentBackbone,
    )

    return RecurrentBackbone(
        **_common(config, vocab_size=vocab_size),
        num_layers=config.num_hidden_layers,
        mode=getattr(config, "attention_mode", CAUSAL),
        **_extra_kwargs(config),
    )


def _hf_builder(
    model_type: str, **defaults: Any
) -> Callable[..., "SequenceBackbone"]:
    """Return a builder for one ``transformers`` architecture.

    ``defaults`` are this arm's own hyperparameters (e.g. CEHR-BERT's published
    depth and width); a run's ``backbone_kwargs`` override them.
    """

    def build(config: Any, *, vocab_size: int) -> "SequenceBackbone":
        from odyssey.models.backbones.hf import HFBackbone  # noqa: PLC0415

        kwargs = {**defaults, **_extra_kwargs(config)}
        kwargs.setdefault("num_hidden_layers", config.num_hidden_layers)
        kwargs.setdefault("num_attention_heads", config.attn_num_heads)
        return HFBackbone(
            **_common(config, vocab_size=vocab_size),
            model_type=model_type,
            mode=getattr(config, "attention_mode", CAUSAL),
            max_context=int(getattr(config, "max_context", 4096)),
            **kwargs,
        )

    return build


_ALL_MODES = frozenset(ATTENTION_MODES)


BACKBONES: dict[str, BackboneSpec] = {
    # --- native ------------------------------------------------------------
    "hybrid": BackboneSpec(_build_hybrid, stateless=False),
    "pure_mamba2": BackboneSpec(_build_pure_mamba2, stateless=False),
    "ehr_mamba": BackboneSpec(
        _build_ehr_mamba, stateless=True, one_patient_per_row=True
    ),
    "transformer": BackboneSpec(_build_transformer, stateless=True, modes=_ALL_MODES),
    "bigbird": BackboneSpec(_build_bigbird, stateless=True, modes=_ALL_MODES),
    "lstm": BackboneSpec(
        _build_recurrent,
        stateless=True,
        modes=frozenset({CAUSAL, ENCODER}),
        one_patient_per_row=True,
    ),
    "tiny_gru": BackboneSpec(_build_tiny_gru, stateless=False),
    # --- transformers ------------------------------------------------------
    # Only families verified to route a custom 4D mask through untouched; see
    # odyssey.models.backbones.hf for the ones that are refused and why.
    "bert": BackboneSpec(_hf_builder("bert"), stateless=True, modes=_ALL_MODES),
    "cehr_bert": BackboneSpec(
        # The published CEHR-BERT configuration (odyssey/models/configs/
        # cehr_bert.yaml on release_article): 768 wide, 5 layers, 8 heads.
        # hidden_size still comes from the run config so a matched-budget
        # comparison against the hybrid stays possible.
        _hf_builder("bert", num_hidden_layers=5, num_attention_heads=8),
        stateless=True,
        modes=_ALL_MODES,
    ),
    "roberta": BackboneSpec(_hf_builder("roberta"), stateless=True, modes=_ALL_MODES),
    "gpt2": BackboneSpec(_hf_builder("gpt2"), stateless=True, modes=_ALL_MODES),
    "llama": BackboneSpec(_hf_builder("llama"), stateless=True, modes=_ALL_MODES),
}


def backbone_spec(name: str) -> BackboneSpec:
    """Return the spec for ``name``, or raise naming every registered arm."""
    try:
        return BACKBONES[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone {name!r}; registered: "
            f"{', '.join(sorted(BACKBONES))}"
        ) from None


def validate_backbone(name: str, mode: str) -> BackboneSpec:
    """Return the spec for ``name`` after checking it can run in ``mode``.

    Called at config validation rather than at the first training step: an
    illegal pair (a bidirectional mode on a Mamba arm, say) is a typo that must
    fail before a run allocates a GPU, not after it has logged a loss.
    """
    if mode not in ATTENTION_MODES:
        raise ValueError(
            f"unknown attention_mode {mode!r}; known: {', '.join(ATTENTION_MODES)}"
        )
    spec = backbone_spec(name)
    if not spec.supports(mode):
        raise ValueError(
            f"backbone {name!r} cannot run in attention_mode {mode!r}; it "
            f"supports: {', '.join(sorted(spec.modes))}"
        )
    return spec
