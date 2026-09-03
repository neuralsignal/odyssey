# Multi-backbone research: what the EHRMamba paper had, what `main` has, what is actually missing

Written 2026-09-03 against `upstream/main` @ `ca0589c`. Companion to
[`docs/multi_backbone_plan.md`](multi_backbone_plan.md), which turns this into work.

Goal of the investigation: the paper *EHRMamba: Towards Generalizable and Scalable
Foundation Models for Electronic Health Records* (arXiv:2405.14567) shipped with a
multi-architecture Odyssey. Today's `main` trains exactly one architecture. Find out
what was lost, what silently survived, and what it would actually cost to get a
"pick your architecture" knob back.

## 1. Three code lineages, not two

The user asked to reconcile `main` with `release_article`. There is a third, better
source that the request did not name.

| Ref | Date | What it is |
|---|---|---|
| `upstream/release_article` @ `2e992ac` | 2024-05-24 | The paper snapshot. 1 commit past the fork point. |
| `78878c8` (parent of `68b87c0`) | 2026-04-04 | **Last commit on `main` that still had every model.** Two years of refactoring past `release_article`, and it has *more* models than the paper snapshot. |
| `upstream/main` @ `ca0589c` | 2026-09-04 | Today. One backbone family, plus a control. 912 commits past the old fork state. |

`68b87c0` *"Replace all models with EHR-Mamba3; remove dead code"* is the deletion
commit: −8648 lines, +1015. It removed `cehr_bert`, `cehr_big_bird`, `ehr_mamba`,
`ehr_mamba2`, the `baseline/` notebooks, `odyssey/interp/`, and the Lightning
`pretrain.py`/`finetune.py` entry points in one go.

**Port from `78878c8`, not from `release_article`.** `release_article`'s models are
pre-refactor (`c869316` "Refactored the codebase to remove redundant code" landed
after the fork), and `release_article` never had `ehr_mamba2` or `cehr_multibird`
at all.

The user's fork `neuralsignal/odyssey` was a clean ancestor of `upstream/main`
(0 divergent commits, 912 behind) and has been fast-forwarded to `ca0589c`.

## 2. Model inventory

| Model | `release_article` | `78878c8` | `main` today | Notes |
|---|---|---|---|---|
| CEHR-BERT | ✅ `cehr_bert/model.py` | ✅ | ❌ | HF `BertForMaskedLM`, bidirectional MLM, 512 ctx, depth 5 |
| CEHR-BigBird | ✅ `cehr_big_bird/model.py` | ✅ | ❌ | HF `BigBirdForMaskedLM`, `block_sparse`, 2048 ctx, depth 6 |
| CEHR-MultiBird | ❌ | ✅ (config only survives) | ❌ | BigBird variant; only `configs/cehr_multibird.yaml` remained by `78878c8` |
| EHRMamba (Mamba-1) | ✅ `cehr_mamba/model.py` | ✅ `ehr_mamba/` | ❌ | HF `MambaForCausalLM`, 2048 ctx, 32 layers, `state_size=16` |
| EHR-Mamba2 | ❌ | ✅ `ehr_mamba2/model.py` | ❌ | HF `Mamba2ForCausalLM`. **No CEHR embeddings** — plain token ids |
| EHR-Mamba3 | ❌ | ❌ | ❌ | Added by `68b87c0`, later replaced by the hybrid |
| Hybrid Mamba-2 + attention | ❌ | ❌ | ✅ `backbones/hybrid.py:454` | The current default. Own implementation, `mamba-ssm` kernels |
| Modern-vanilla transformer | ❌ | ❌ | ✅ `backbones/transformer.py:293` | RoPE / pre-norm RMSNorm / SwiGLU. The existing architecture control |
| Tiny GRU | ❌ | ❌ | ✅ `backbones/tiny_gru.py:29` | CPU/CI stand-in, not a research arm |
| XGBoost | ✅ notebook | ✅ notebook | ➖ superseded | `main` has a *tuned* GBM at `odyssey/inference/baseline_features.py`, plus EBM, TabICL, SurvivalPFN, MEDS-Tab |
| Bi-LSTM | ✅ `baseline/Bi-LSTM.py` | ✅ | ❌ | Bidirectional; see §4.1 |

## 3. What already survived — the port is smaller than it looks

The paper's *inputs* were never lost. They were refactored into `main` under new
names, and every existing backbone consumes them:

| `release_article` | `main` today |
|---|---|
| `BERTEmbeddingsForCEHR` / `BigBirdEmbeddingsForCEHR` / `MambaEmbeddingsForCEHR` (3 near-duplicate classes) | one `ClinicalEventEmbeddings` (`odyssey/models/embeddings.py:123`) + `CachedEHREmbeddings` bridge (`:242`) |
| concept + time + age + visit-order + visit-segment + token-type fusion | identical fusion, same `tanh(scale_back_concat_layer(...))` shape, plus an opt-in numeric-value channel |
| `ConceptTokenizer` | `odyssey/data/vocabulary.py` (+ ICD-3 backoff, quantile/clinical value bins) |
| `PretrainDataset` / `PretrainDatasetDecoder` | `odyssey/data/streaming.py`, `odyssey/data/packed_context.py` |
| per-model Lightning `training_step` + HF LM head | `odyssey/training/train.py` + `odyssey/models/sequence_model.py` heads |

And `main` already has the abstraction the request is asking for:

- `SequenceBackbone` ABC (`odyssey/models/backbones/base.py:56`) — `forward(batch, state, reset_mask) -> (hidden_states, state)`, `hidden_size`, `embeddings`.
- Three concrete implementations proving it holds for a stateful CUDA hybrid, a stateless attention stack, and a toy RNN.
- `scripts/long_history_compare.py` + `scripts/make_backbone_table.py` — a paired, subject-clustered A/B harness that already exists to compare two backbones and emit the paper table.

**So this is not "reintroduce a model abstraction". It is "add arms to one that already
exists, and stop hard-coding the arm names".**

## 4. The four real incompatibilities

### 4.1 Objective: MLM/bidirectional vs. autoregressive

The paper's CEHR-BERT and CEHR-BigBird are **masked-LM bidirectional encoders**
(`mask_prob: 0.15` in both configs), finetuned per task with a classification head.
`main` is a **causal next-bundle forecaster** — a marked temporal point process, with
every head (forecast, time-to-event, hazards, concept bottleneck) reading a hidden
state that must not have seen the future.

Bi-LSTM has the same problem, structurally.

There is no honest way to drop a bidirectional encoder into `main`'s loss. Two
options, and only one of them is cheap:

- **(a) Port the architecture, causally.** Run BERT/BigBird as decoders behind
  `main`'s existing objective. Answers "does this attention *shape* work here?" —
  which is exactly the question `main`'s existing transformer arm was built to ask
  (README roadmap Track A item 5). Cheap.
- **(b) Port the paper's protocol.** Add an MLM pretraining path and a per-task
  finetune path alongside the forecasting one. That is a second training pipeline,
  a second dataset path, a second eval harness, and it reproduces 2024 numbers
  rather than answering a 2026 question. Expensive, and it fights the codebase.

(a) unless someone explicitly wants a paper reproduction. This is the single decision
that most changes the size of the work.

### 4.2 BigBird block-sparse cannot be a decoder in HF

Verified in `huggingface/transformers` `modeling_big_bird.py`:

```
1136:  raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
1501:  "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting ..."
```

So `BigBirdForCausalLM` silently downgrades to dense attention. A causal BigBird arm
means writing the sparsity pattern ourselves. Two ways:

- **Sparsity as a mask** over `scaled_dot_product_attention`: global + sliding-window
  + random blocks, intersected with `main`'s existing block-diagonal causal mask
  (`_build_attn_mask`, `transformer.py:179`). ~40 lines on top of the existing
  `CausalSelfAttention`. Keeps O(n²) *compute* but reproduces BigBird's *receptive
  field* exactly. Answers the quality question, not the speed question.
- **A real sparse kernel**: reproduces the speed claim, weeks of work, and the speed
  question is already answered by the Mamba arms.

Mask it. Say so in the docstring.

### 4.3 State carrying: Mamba-1 physically cannot do it with the fast kernel

`main`'s hybrid backbone carries recurrent state across TBTT chunks via a patched
`Mamba2` subclass (`hybrid.py`, `_make_mamba2_with_state_cls`) that seeds
`initial_states` into `mamba_chunk_scan_combined`.

**Mamba-1 has no equivalent.** In `mamba-ssm==2.3.0` (the version already pinned in
the `cuda` extra):

- `mamba_ssm/ops/selective_scan_interface.py:106` —
  `selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)`.
  It can *return* a last state. It has **no parameter to accept one.**
- `mamba_ssm/modules/mamba_simple.py:129` — with `inference_params` set and
  `seqlen_offset > 0`, `Mamba.forward` routes to `step()`, the single-token decode
  path. There is no multi-token-chunk-with-seeded-state path at all.

The workaround is not a patch, it is a different training regime: run Mamba-1
**stateless over packed context windows**, the way `TransformerBackbone` already runs
(`PackedContextSampler`, `train.py:1551`). This is also *faithful to the paper*, which
used a fixed `max_seq_length: 2048` and carried nothing across windows.

`Mamba` (Mamba-1) is exported from `mamba_ssm/__init__.py` in 2.3.0, so this needs no
new dependency.

Mamba-2 as a *pure* SSM arm (hybrid minus the attention branch) keeps full state
carrying for free — the patched class already exists.

### 4.4 Packed multi-patient rows

`PackedContextSampler` packs several whole patients into one row and relies on the
backbone honouring `reset_mask` as a segment boundary. Consequences for new arms:

- Stateless attention arms (BERT-ish, BigBird) must intersect their attention mask
  with the same-segment mask and reset RoPE/position ids per segment — `transformer.py`
  already has `_segment_ids`, `_position_ids`, `_rebase_time_stamps`, `_build_attn_mask`
  to reuse verbatim.
- `EHRHybridBackbone.forward` raises `NotImplementedError` on any reset past position 0,
  so stateful arms stay on `PackedLaneSampler`. A pure-Mamba arm inherits that limit.
- A *stateless recurrent* arm (Mamba-1, §4.3) is the awkward case: it needs
  `PackedContextSampler`, but a recurrent scan cannot honour `reset_mask` the way a
  mask can, and Mamba-1 has no `cu_seqlens` varlen path. It needs one patient per row,
  which the sampler cannot do today (it greedily packs several, `packed_context.py:262`).

The leakage tests already written for the transformer arm
(`tests/odyssey/models/backbones/test_transformer.py`, e.g.
`test_no_cross_patient_leakage_*`, `test_packed_patient_matches_processing_alone`)
are the acceptance bar for any new stateless arm — copy them, do not re-derive them.

## 5. `backbone` is a string compared in 10 places

Adding a third value today means editing all of these:

| File | Lines |
|---|---|
| `odyssey/training/train.py` | 629, 642 (construction), 1551, 1569 (sampler choice) |
| `odyssey/inference/alerts.py` | 449, 667 (`packed = backbone == "transformer"`) |
| `odyssey/inference/run_inference.py` | 862 |
| `odyssey/inference/case_study.py` | 319 (raises "not wired for transformer") |
| `odyssey/inference/concept_attribution.py` | 414 (same) |
| `odyssey/inference/interventions.py` | 521 (warns) |

Every one of those is really asking one of two questions: *is this backbone
stateless?* or *does it need packed context rows?* — which are the same question
today. That is the registry's job. Note the three `inference/` sites that refuse or
warn for `transformer`: a new stateless arm inherits those gaps, and pretending
otherwise would silently produce wrong case studies and attributions.

## 6. Per-model verdict

| Arm | Verdict | Why |
|---|---|---|
| **Pure Mamba-2** (hybrid minus attention) | **Do it first** | ~30 lines. Reuses the patched stateful `Mamba2`. Isolates whether the attention branch earns its keep — a question the project has not measured and `main`'s own architecture rests on |
| **EHRMamba / Mamba-1** | **Do it** | The paper's headline model. Stateless over packed context (§4.3), which is what the paper did anyway. No new dependency |
| **CEHR-BigBird (causal, masked sparsity)** | **Do it, third** | The one genuinely distinct attention *shape* missing. ~40 lines over the existing attention block. Honest caveat: reproduces the receptive field, not the sparse-kernel speed |
| **CEHR-BERT (causal)** | **Skip unless asked** | Once made causal, it is `main`'s transformer arm with post-norm + LayerNorm + GELU + learned absolute positions instead of pre-norm + RMSNorm + SwiGLU + RoPE. That is a hyperparameter sweep, not an architecture. If someone wants it, it is a flag on `TransformerBackbone`, not a new file |
| **Bi-LSTM** | **Skip** | Bidirectional; incompatible with the objective (§4.1). Causal-and-scaled, it is `TinyGRUBackbone` with a bigger `hidden_size` |
| **XGBoost** | **Skip** | Superseded. `main` already runs a *tuned* GBM as the strongest reference bar, plus EBM, TabICL, SurvivalPFN and MEDS-Tab. Adding a 2024 untuned XGBoost notebook makes the comparison worse |
| **MLM pretraining + per-task finetuning path** | **Ask first** | See §4.1(b). Real work, different question |
| **Multitask Prompted Finetuning (MPF)** | **Skip** | The paper's MPF exists to avoid one finetune per task. `main` already trains every task head jointly in one run — MPF's problem does not exist here |

## 7. Repo context worth knowing before opening a PR upstream

`HANDOFF.md` and `docs/experiment_plan.md` show `main` is mid-crunch on an ML4H 2026
submission (deadline Sept 10 AoE). The backbone-comparison machinery
(`long_history_compare.py`, `make_backbone_table.py`) landed in the most recent commit,
`ca0589c`. A backbone-registry PR is aligned with roadmap Track A item 5, but the
timing is theirs to pick — this work lands on the fork first regardless.
