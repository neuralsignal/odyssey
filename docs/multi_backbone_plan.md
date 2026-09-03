# Plan: make the backbone a knob, and put the EHRMamba-paper architectures back behind it

Companion to [`docs/multi_backbone_research.md`](multi_backbone_research.md), which
has the evidence for every claim below. Branch: `multi-backbone` on
`neuralsignal/odyssey` (fork already fast-forwarded to `upstream/main` @ `ca0589c`).

**Shape of the work:** `main` already has the abstraction (`SequenceBackbone`) and the
comparison harness (`long_history_compare.py` → `make_backbone_table.py`). What it does
not have is (a) a way to name a backbone without editing 10 `if backbone == "..."`
sites, and (b) any arm between "our hybrid" and "a vanilla transformer". So: one small
plumbing phase, then one file per architecture.

**Not in scope** (§6 of the research doc): XGBoost, Bi-LSTM, MLM pretraining, MPF.
Each is either superseded by something stronger already in `main`, structurally
incompatible with the causal objective, or answering a question `main` does not have.

---

## Phase 0 — backbone registry

**Why first:** every later phase otherwise pays the 10-site edit tax again, and three
of those sites (`case_study.py:319`, `concept_attribution.py:414`,
`interventions.py:521`) currently *refuse or warn* for stateless backbones. A new arm
must inherit that refusal explicitly, not silently produce wrong attributions.

Add to `odyssey/models/backbones/__init__.py` (currently a one-line docstring — no new
file):

```python
@dataclass(frozen=True)
class BackboneSpec:
    build: Callable[..., SequenceBackbone]   # deferred import inside; mamba-ssm needs CUDA
    stateless: bool
    """No cross-call recurrent state: driven by PackedContextSampler over
    whole/truncated patients, not PackedLaneSampler's TBTT chunking."""

BACKBONES: dict[str, BackboneSpec] = {...}
```

Then, mechanically:

| Site | Now | After |
|---|---|---|
| `train.py:629,642` | `if backbone_kind == "hybrid" / elif "transformer"` | `BACKBONES[kind].build(config, vocab_size=...)` |
| `train.py:1551,1569` | `if config.backbone == "transformer"` | `if BACKBONES[config.backbone].stateless` |
| `alerts.py:449,667` | `packed = backbone == "transformer"` | `packed = BACKBONES[backbone].stateless` |
| `run_inference.py:862` | same shape | same |
| `case_study.py:319`, `concept_attribution.py:414`, `interventions.py:521` | `== "transformer"` guard | `BACKBONES[...].stateless` guard, message names the actual backbone |

No behavior change. `backbone: str = "hybrid"` stays the default, so every saved
`config.json` keeps loading.

**Check:** one `tests/odyssey/models/backbones/test_registry.py` asserting (1) every
registered name builds on CPU or raises a clean `ImportError` naming `mamba-ssm`,
(2) `"hybrid"` is not stateless and `"transformer"` is, (3) the set of names the
inference guards accept equals `BACKBONES.keys()`. Existing suites must stay green
unchanged — that is the real regression test for a no-op refactor.

*Effort: small. Touches 6 files, adds ~50 lines, deletes ~30.*

---

## Phase 1 — `pure_mamba2`: the hybrid minus its attention branch

**The most valuable arm and the cheapest.** `main`'s central architectural claim is
that parallel Mamba-2 + attention with merge-attention beats either alone
(`hybrid.py` module docstring, research journal entry 03). The transformer arm prices
one half of that. Nothing prices the other half.

`odyssey/models/backbones/pure_mamba.py`: the same block stack as `HybridBlock`, with
`self.attn` and `self.merge` dropped. Reuses `_make_mamba2_with_state_cls` and
`HybridState` verbatim, so cross-chunk state carrying, the reset-zeroing loop, and the
aliasing-clone fix all come for free. Stateful → `PackedLaneSampler`, no sampler work.

Cleanest way to avoid a forked copy of the block loop: give `HybridBlock` an optional
`attn_mixer_cls=None`, and have `EHRHybridBackbone` take `use_attention: bool = True`.
Then `pure_mamba2` is a registry entry with `use_attention=False`, not a new file at all.
Prefer that if the diff to `hybrid.py` stays under ~20 lines; fall back to a separate
file if the branching makes `HybridBlock.forward` hard to read.

**Check:** CPU-unavailable (needs `mamba-ssm`), so a GPU test mirroring
`tests/odyssey/models/backbones/test_mamba2_patch_gpu.py`: state carrying across two
chunks equals one contiguous forward, and `reset_mask[:,0]` zeroes the right rows.

*Effort: small. Highest research value per line in this plan.*

---

## Phase 2 — `ehr_mamba`: the paper's Mamba-1, stateless over packed context

The paper's headline architecture. `mamba_ssm.modules.mamba_simple.Mamba` is already
importable from the pinned `mamba-ssm==2.3.0` — no new dependency.

`odyssey/models/backbones/ehr_mamba.py`: a plain pre-norm stack of Mamba-1 mixers on
`CachedEHREmbeddings`, `RMSNorm` final. Paper config for reference: 32 layers,
`hidden_size=768`, `state_size=16`, `expand=2`, `conv_kernel=4`, 2048 context.

**Stateless, deliberately.** Mamba-1's kernel physically cannot accept an initial SSM
state (`selective_scan_fn` has no such parameter; the `inference_params` path is
single-token decode only — research doc §4.3). So this arm returns
`TimeAwareState(recurrent=None, ...)`, registers `stateless=True`, and is driven by
`PackedContextSampler` exactly like the transformer arm. This is what the paper did
too (fixed 2048-token windows, nothing carried), so the arm is *more* faithful this
way, not less. The module docstring must say this plainly, next to the kernel line
numbers, or someone will "fix" it later.

**The one real complication: packing leaks into a recurrent scan.**
`PackedContextSampler` greedily packs *several* whole patients into each
`max_context` row (`packed_context.py:262-270`) and relies on the backbone honouring
`reset_mask` as a segment boundary. `TransformerBackbone` honours it with a mask.
Mamba scans recurrently along the sequence axis, so patient B's hidden state is
contaminated by patient A's and there is no mask to apply. Mamba-1's kernel offers no
`cu_seqlens` varlen path to cut the scan mid-row either, so the state cannot be zeroed
at the boundary.

Fix: **one patient per row for this arm.** `PackedContextSampler` has no such option
today — add `pack: bool = True`, and in the row loop
(`packed_context.py:270`) break after the first patient when it is `False`. ~5 lines,
no effect on the transformer arm. Cost is wasted padding tokens on short patients;
say so in the docstring. Do not silently accept the leak — it would look like a
mediocre-but-plausible result rather than a bug.

**Check:** copy `test_transformer.py`'s leakage tests
(`test_no_cross_patient_leakage_*`, `test_packed_patient_matches_processing_alone`).
They are the acceptance bar precisely because they would catch the leak above.

*Effort: medium — the backbone itself is small; the `pack=False` option and its tests are the rest.*

---

## Phase 3 — `bigbird`: causal block-sparse attention, as a mask

The one genuinely distinct attention *shape* missing from `main`. HF cannot supply it
causally (`modeling_big_bird.py:1136` raises; `:1501` silently downgrades to dense —
research doc §4.2), so implement the pattern directly.

`odyssey/models/backbones/bigbird.py`, or better, a `sparsity=` option on
`transformer.py`'s `CausalSelfAttention`: build BigBird's receptive field — global
blocks, sliding window, random blocks — as a boolean mask, `&` it with the existing
`_build_attn_mask` output (causal ∧ same-segment), pass it to
`scaled_dot_product_attention`. Everything else in `TransformerBackbone` is reused
unchanged: RoPE per-segment position ids, timestamp rebasing, packing safety.

**Stated honestly in the docstring:** this reproduces BigBird's receptive field, not
its sparse-kernel speed — compute stays O(n²). That is the right trade here: the
speed question is already answered by the Mamba arms, and the open question is whether
restricting attention to global+window+random costs forecast quality. A real sparse
kernel is weeks of work for a question nobody is asking.

Random blocks must be resampled per forward in training and **fixed by seed at eval**,
or the paired subject-clustered comparison in `long_history_compare.py` gets noise it
cannot attribute.

**Check:** the mask is the whole model — one test asserting the built mask is a strict
subset of the dense causal ∧ same-segment mask, contains the global and window blocks
it claims to, and is identical across two eval-mode forwards with the same seed. Plus
the copied leakage tests from Phase 2.

*Effort: medium.*

---

## Phase 4 (optional) — `bert`-flavoured transformer

Only if someone explicitly wants the CEHR-BERT arm. Once made causal it differs from
`main`'s existing transformer arm by post-norm vs pre-norm, LayerNorm vs RMSNorm,
GELU-MLP vs SwiGLU, learned absolute positions vs RoPE — four flags on
`TransformerBackbone`, not a new file. Cheap if wanted, and worth nothing if not asked
for: it prices normalization and activation choices, not an architecture.

---

## Running the comparison

Nothing new needed. Each arm trains through the same `train.py` with
`backbone="<name>"`, and is compared with the machinery that landed in `ca0589c`:

```
scripts/long_history_compare.py <dump_a> <dump_b> --label-a hybrid --label-b pure_mamba2
scripts/make_backbone_table.py  <banked json> --output paper/.../backbone_<name>.tex
```

Matched parameter and compute budget per the roadmap Track A item 5 protocol: subset
scale first, full scale only if the subset result is interesting in either direction.
For stateless arms, report the `whole` and `truncated` strata separately —
`PackedContextSampler.truncated_subject_ids` exists for exactly this, and pooling them
makes a context-limited arm look worse than it is.

---

## Sequencing and effort

| Phase | Arm | Effort | Value |
|---|---|---|---|
| 0 | registry | small | unblocks everything; removes a 10-site tax |
| 1 | `pure_mamba2` | small | prices `main`'s own central architecture claim |
| 2 | `ehr_mamba` | medium | the paper's headline model, faithfully |
| 3 | `bigbird` | medium | the one missing attention shape |
| 4 | `bert` flags | small | only on request |

Phases 1–3 are independent once 0 lands.

---

## Open question, needs an answer before Phase 2

**Architecture port, or paper reproduction?** This plan assumes the first: bring the
architectures across and run them behind `main`'s causal forecasting objective, same
tokenization, same heads, same eval — so the arms are comparable to the hybrid and to
each other.

Reproducing the *paper* instead (MLM pretraining for BERT/BigBird, per-task finetuning,
MPF) means a second training pipeline and a second eval harness living alongside the
current one. It answers "what did the 2024 paper get?" rather than "which architecture
is best for this project's task". If that reproduction is what is wanted, say so —
it changes the plan substantially and Phase 3 in particular.
