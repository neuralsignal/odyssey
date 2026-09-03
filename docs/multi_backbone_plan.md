# Plan: make the backbone a knob, add causal and bidirectional arms, load `transformers` models dynamically

Companion to [`docs/multi_backbone_research.md`](multi_backbone_research.md), which has
the evidence for every claim below. Branch: `multi-backbone` on `neuralsignal/odyssey`
(fork fast-forwarded to `upstream/main` @ `ca0589c`).

**Scope, as decided:** architecture port, not paper reproduction. The goal is a menu of
architectures behind main's existing tokenization, heads and evaluation. Bidirectional
arms are wanted. `transformers` models should be usable without a file each.

**The organizing idea** (research doc §7): an arm is a **pair** — an *architecture*
(what the blocks are) and a *context regime* (what each position may see: `causal`,
`prefix`, `encoder`). "BERT" is not a bidirectional model; it is a transformer stack
usually run bidirectionally. Keeping those two axes separate is what makes this a
registry and a mask function rather than a combinatorial pile of model files.

**Out of scope:** XGBoost (main already runs a tuned GBM plus EBM/TabICL/SurvivalPFN/
MEDS-Tab), MLM pretraining, MPF. Reasons in research doc §6.

---

## Phase 0 — the registry

**Why first:** `backbone` is a bare string compared in 10 places (research doc §5), and
three of them (`case_study.py:319`, `concept_attribution.py:414`,
`interventions.py:521`) *refuse or warn* for stateless backbones. Every later phase
pays that tax again otherwise, and a new arm must inherit those refusals explicitly
rather than silently emit wrong attributions.

In `odyssey/models/backbones/__init__.py` (currently a one-line docstring — no new file):

```python
@dataclass(frozen=True)
class BackboneSpec:
    build: Callable[..., SequenceBackbone]   # deferred import inside; mamba-ssm needs CUDA
    stateless: bool
    """No cross-call recurrent state: driven by PackedContextSampler over
    whole/truncated patients, not PackedLaneSampler's TBTT chunking."""
    modes: frozenset[str] = frozenset({"causal"})
    """Context regimes this architecture can express. Mamba has no attention
    mask to invert, so it is causal-only; see the matrix in research doc §7."""
    one_patient_per_row: bool = False
    """Cannot honour reset_mask mid-row (recurrent scan, or an HF model whose
    packing behaviour is unverified). Forces pack=False."""

BACKBONES: dict[str, BackboneSpec] = {...}
```

Two new `TrainingConfig` fields, both round-tripping through `asdict` → `config.json`
and both safe for old runs (`run_inference.load_run` already drops unknown fields and
defaults missing ones, `run_inference.py:339-350`):

- `attention_mode: str = "causal"` — validated against `BACKBONES[backbone].modes`.
- `backbone_kwargs: dict[str, Any] = field(default_factory=dict)` — free-form per-arm
  overrides, so an HF arm can carry `num_hidden_layers`, `intermediate_size`, etc.
  without a `TrainingConfig` field per architecture.

Mechanical replacements:

| Site | Now | After |
|---|---|---|
| `train.py:629,642` | `if backbone_kind == "hybrid" / elif "transformer"` | `BACKBONES[kind].build(config, vocab_size=...)` |
| `train.py:1551,1569` | `if config.backbone == "transformer"` | `if BACKBONES[config.backbone].stateless` |
| `alerts.py:449,667` | `packed = backbone == "transformer"` | `packed = BACKBONES[backbone].stateless` |
| `run_inference.py:862` | same shape | same |
| `case_study.py:319`, `concept_attribution.py:414`, `interventions.py:521` | `== "transformer"` guard | `BACKBONES[...].stateless` guard, message names the actual backbone |

No behavior change; `backbone: str = "hybrid"` stays the default.

**Check:** `tests/odyssey/models/backbones/test_registry.py` — every registered name
builds on CPU or raises a clean `ImportError` naming its extra; `"hybrid"` is stateful
and `"transformer"` stateless; the inference guards accept exactly `BACKBONES.keys()`;
an illegal `(backbone, attention_mode)` pair raises at config validation, not at step 1
of training. Existing suites green and unchanged is the real regression test.

*Effort: small. 6 files, ~+80/−30 lines.*

---

## Phase 1 — the `transformers` adapter

**Highest value per line in this plan.** One file turns `bert`, `cehr_bert`, `roberta`,
`gpt2`, `llama` and friends into arms, in both regimes where the family supports it,
with no per-model module. `transformers>=5.16.1` is already an optional extra
(`pyproject.toml`, `text`), and `odyssey/text/embed_notes.py:52` already has the
deferred-import idiom to copy.

`odyssey/models/backbones/hf.py`:

```python
class HFBackbone(SequenceBackbone):
    def __init__(self, model_type, vocab_size, hidden_size, *, mode="causal",
                 hf_overrides=None, use_hf_embeddings=False, **embedding_kwargs): ...
```

- `AutoConfig.for_model(model_type, hidden_size=..., max_position_embeddings=max_context, **hf_overrides)`,
  then `AutoModel.from_config(cfg)` — random init; we are porting architectures, not
  loading clinical weights.
- Embeddings from `CachedEHREmbeddings`, handed over as `inputs_embeds`.
- **We build the mask, HF applies it.** A 4D mask passes through HF untouched
  (`masking_utils.py:811`), so `causal` / `prefix` / `encoder` is our mask function,
  not an HF config flag. Reuse `transformer.py`'s `_build_attn_mask`, `_segment_ids`,
  `_position_ids`, `_rebase_time_stamps` verbatim.
- Pass explicit segment-reset `position_ids`, or packed rows get row-absolute ones.
- Return `out.last_hidden_state`, stateless `TimeAwareState(recurrent=None, ...)`.

Registry entries built on it:

| Name | `model_type` | Notes |
|---|---|---|
| `bert` | `bert` | plain HF BERT stack |
| `cehr_bert` | `bert` | paper config: `hidden_size=768`, depth 5, 8 heads, `intermediate_size=3072`, 512 ctx |
| `roberta`, `gpt2`, `llama` | as named | available for free once the adapter exists |

**Two gotchas to document in the module docstring, not leave to be discovered**
(research doc §9): HF re-embeds on top of `inputs_embeds` (a second learned positional
signal and a second LayerNorm after `ClinicalEventEmbeddings`' own), and
`max_position_embeddings` must be ≥ `max_context` or long rows crash. Offer
`use_hf_embeddings=True` as the ablation that prices the CEHR embedding fusion itself.

**Refuse, loudly, rather than wrap:** `longformer`, `modernbert` (local/sliding
attention builds its own mask), `deberta_v2` (disentangled attention, neither mask
helper), `mamba` (no attention mask at all — use the native arms). A silently-ignored
mask produces a model that trains fine, converges fine, and has seen the future.

**Check — this is the important one.** Registration of *any* HF arm requires passing
`test_transformer.py::test_causal_position_invariant_to_future_token_changes`: change a
token at *t+k*, assert the hidden state at *t* is bit-identical. It is the only check
that proves the mask was honoured, it already exists, and it is cheap. Run it as a
build-time assertion for any `model_type` the adapter has not seen before — a static
whitelist encodes today's `transformers` behaviour and rots on the next release. Plus
the packing/leakage tests from the same file.

*Effort: medium. One file, ~200 lines, most of it the mask plumbing and the refusal list.*

---

## Phase 2 — LSTM / Bi-LSTM

`odyssey/models/backbones/recurrent.py`: `CachedEHREmbeddings` → `nn.LSTM` → hidden
states. `bidirectional` is one constructor flag; `modes={"causal", "encoder"}`.

Note what *not* to copy: `TinyGRUBackbone` steps an `nn.GRUCell` per token in Python
because it must honour `reset_mask` mid-row. This arm sets `one_patient_per_row=True`,
so there are no mid-sequence resets and the fused cuDNN `nn.LSTM` path works directly.
~40 lines, not 106.

Causal (unidirectional) works the moment this lands. Bidirectional needs Phase 3.

**Check:** the same leakage tests, plus one asserting the bidirectional variant is
*rejected* under `attention_mode="causal"` — a wrong-way-round config here is
undetectable downstream.

*Effort: small.*

---

## Phase 3 — the `encoder` context regime

What makes CEHR-BERT and Bi-LSTM actually themselves (research doc §8). Cut the record
at a landmark, encode it bidirectionally, supervise from the final position.

Cheaper than it sounds, because three of the four pieces already exist:

1. **Sampler: a generator, not surgery.** `PackedContextSampler` takes an *iterator* of
   `PatientSequence`; right-truncation at a random cut is a wrapping generator plus a
   `_truncate_tail` mirror of the existing `_truncate_head`. ~10 lines.
2. **Targets: unchanged.** `EventHazardTargets` is already per-position from the
   `(subject, visit)` onset/censoring tables. At the truncation position they are
   already correct. The change is which positions are supervised, not new targets.
3. **Losses: a subset.** Event hazards and concept heads only; forecast, time and value
   heads are disabled — definitionally, this regime cannot do them.
4. **`pack=False`** on the sampler (~5 lines), needed by every `one_patient_per_row` arm.

**The cost is evaluation, and it is real.** `alerts.py` scores every landmark in one
streaming pass via `_landmark_mask`. A bidirectional model cannot: each landmark needs
its own forward over its own truncated prefix. Landmarks are every 4h per visit, so a
ten-day stay is ~60 forwards over growing prefixes instead of one pass.

**Decide before training an encoder arm, not after:** subsample landmarks for these
arms — and run *every* arm on the same subsampled set, or `long_history_compare.py`'s
paired comparison is no longer paired — or budget for the full bill. This is the one
place where "just add an architecture" turns into a protocol change.

*Effort: medium for training, and an open question for eval.*

---

## Phase 4 — native Mamba arms

Independent of Phases 1–3; both causal-only (an SSM's receptive field is a property of
the scan, not a mask).

**`pure_mamba2`** — the hybrid minus its attention branch. `main`'s central
architectural claim is that parallel Mamba-2 + attention beats either alone; the
transformer arm prices one half and nothing prices the other. Cheapest way in: give
`HybridBlock` an optional `attn_mixer_cls=None` and `EHRHybridBackbone` a
`use_attention: bool = True`, so this is a registry entry rather than a new file —
provided the diff to `hybrid.py` stays under ~20 lines. State carrying, reset zeroing
and the aliasing-clone fix all come for free. GPU test mirroring
`test_mamba2_patch_gpu.py`.

**`ehr_mamba`** — the paper's Mamba-1. `mamba_ssm.modules.mamba_simple.Mamba` is
already importable from the pinned `mamba-ssm==2.3.0`. Paper config: 32 layers, 768
wide, `state_size=16`, `expand=2`, `conv_kernel=4`, 2048 ctx.

**Stateless, deliberately.** Mamba-1's kernel cannot accept an initial SSM state
(`selective_scan_fn` has no such parameter; the `inference_params` path is single-token
decode only — research doc §4.3). It also has no `cu_seqlens` varlen path, so a
recurrent scan cannot be cut at a packed-row boundary: this arm needs
`one_patient_per_row=True`, or patient B's state is contaminated by patient A's. Say
both facts in the module docstring next to the kernel line numbers, or someone will
"fix" the statelessness later. The paper used fixed 2048-token windows carrying
nothing, so this is the faithful configuration, not a compromise.

*Effort: small for `pure_mamba2`, medium for `ehr_mamba`.*

---

## Phase 5 — BigBird, causal, sparsity as a mask

The one genuinely distinct attention *shape* missing. HF cannot supply it causally
(`modeling_big_bird.py:1136` raises, `:1501` silently downgrades to dense — research
doc §4.2), so the adapter is no help and the pattern gets implemented directly.

Best as a `sparsity=` option on `transformer.py`'s `CausalSelfAttention`: build
BigBird's receptive field — global blocks, sliding window, random blocks — as a boolean
mask and `&` it with the existing causal ∧ same-segment mask. Everything else in
`TransformerBackbone` is reused unchanged.

Docstring must say plainly: this reproduces BigBird's receptive field, **not** its
sparse-kernel speed — compute stays O(n²). That is the right trade, because the speed
question is already answered by the Mamba arms and the open question is whether the
restricted receptive field costs forecast quality. A real sparse kernel is weeks of
work for a question nobody is asking.

Random blocks: resampled per forward in training, **fixed by seed at eval**, or the
paired subject-clustered comparison gets noise it cannot attribute.

**Check:** the mask is the whole model — assert it is a strict subset of the dense
causal ∧ same-segment mask, contains the global and window blocks it claims, and is
identical across two seeded eval forwards.

*Effort: medium.*

---

## Phase 6 (optional) — the `prefix` regime

Bidirectional among a sampled prefix, causal after, loss zeroed on prefix positions.
A mask change plus `real_mask &= ~in_prefix`; no new sampler, no new eval, no head
changes. Buys bidirectional *encoding of history* while keeping every existing loss and
the single-pass evaluation.

Value is genuinely uncertain — which is an argument for measuring it, not for building
it speculatively. Do it only if a `causal` vs `encoder` result makes the question
interesting.

---

## Running the comparison

Unchanged. Every arm trains through `train.py` with `backbone=<name>`,
`attention_mode=<regime>`, and is compared with the machinery that landed in `ca0589c`:

```
scripts/long_history_compare.py <dump_a> <dump_b> --label-a hybrid --label-b cehr_bert
scripts/make_backbone_table.py  <banked json> --output paper/.../backbone_<name>.tex
```

Matched parameter and compute budget per roadmap Track A item 5: subset scale first,
full scale only if the subset result is interesting either way. For stateless arms
report the `whole` and `truncated` strata separately — `truncated_subject_ids` exists
for exactly this, and pooling makes a context-limited arm look worse than it is.

---

## Sequencing

| Phase | Delivers | Effort | Depends on |
|---|---|---|---|
| 0 | registry, `attention_mode`, `backbone_kwargs` | small | — |
| 1 | HF adapter → `bert`, `cehr_bert`, `roberta`, `gpt2`, `llama` (causal) | medium | 0 |
| 2 | `lstm` (causal) | small | 0 |
| 3 | `encoder` regime → bidirectional BERT and Bi-LSTM | medium + eval decision | 0, 1, 2 |
| 4 | `pure_mamba2`, `ehr_mamba` | small / medium | 0 |
| 5 | `bigbird` causal | medium | 0 |
| 6 | `prefix` regime | small | 0, and a reason |

Phases 1, 2, 4 and 5 are independent once 0 lands. Phase 3 is the only one that touches
the evaluation protocol.

After Phase 2 the menu is already: `hybrid`, `transformer`, `bert`, `cehr_bert`,
`roberta`, `gpt2`, `llama`, `lstm`, `tiny_gru` — all causal, all comparable, all
through one config field.

---

## Open decisions

1. **Landmark budget for `encoder` arms** (Phase 3). Subsample landmarks — and then
   re-score every arm on the same subsample to keep the comparison paired — or pay for
   ~60× forwards per patient? This is a protocol change, so it wants an answer before
   the first encoder arm trains, not after.
2. **Which HF families to expose beyond the verified set.** `bert` and `roberta` are
   verified for both regimes. `llama`/`gpt2` are verified causal, mask-forced
   bidirectional and unverified there. The leakage probe decides per family at
   registration; the question is whether unverified families are exposed at all or
   gated behind an explicit opt-in.
3. **Does `prefix` (Phase 6) earn a slot**, or is `causal` vs `encoder` the whole
   question worth measuring?
