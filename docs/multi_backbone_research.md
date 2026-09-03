# Multi-backbone research: what the EHRMamba paper had, what `main` has, what is actually missing

Written 2026-09-03 against `upstream/main` @ `ca0589c`; §§6-9 added after the scope
decision. Companion to [`docs/multi_backbone_plan.md`](multi_backbone_plan.md), which
turns this into work.

**Scope, decided:** architecture port, not paper reproduction. Causal *and*
bidirectional arms wanted. `transformers` models loadable without a module each.

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

### 4.1 Objective: bidirectional context leaks every head main has

The paper's CEHR-BERT and CEHR-BigBird are **masked-LM bidirectional encoders**
(`mask_prob: 0.15` in both configs), finetuned per task with a classification head.
The Bi-LSTM was never pretrained at all — `release_article`'s `baseline/Bi-LSTM.py:95`
is CEHR embeddings into `nn.LSTM(bidirectional=True)`, pooled, one linear layer,
`BCEWithLogitsLoss`. A landmark-anchored discriminative classifier, nothing more.

`main` is a **causal next-bundle forecaster**. Every head reads the hidden state at
position *t* and supervises something strictly after *t*:

| Head | Supervises | Leaks under bidirectional context? |
|---|---|---|
| forecast / LM (`_streaming_task_loss`) | the target's bundle members at positions ≥ *t* | **Yes, directly** — those positions are the targets |
| time-to-next-event (`_streaming_time_loss`) | gap to the next event | Yes |
| per-event hazards (`_streaming_event_loss`) | onset within a horizon | Yes — future events are in the context |
| value quantiles (`_streaming_value_loss`) | the next event's magnitude | Yes |
| concept readout | the patient's state *at* *t* | Deployment-invalid: a real-time readout cannot see later labs |

**A rejected shortcut, worth recording so nobody re-proposes it.** "Bidirectional
within a same-timestamp bundle, causal across bundles" looks principled — main's own
framing is that within-bundle order is meaningless and bundles are set-scored. It does
not work here. `_bundle_log_likelihood` (`sequence_model.py:142`) credits position *i*
with the probability of every *not-yet-emitted* member of its bundle, i.e. positions
*j ≥ i* of that same bundle. Those are exactly the positions bidirectional
within-bundle attention would expose. And the variant that *is* safe —
bidirectional over strictly earlier bundles — is a subset of the causal mask, because
bundles are time-ordered and contiguous. There is no free bidirectionality to take.

So bidirectional arms need a **different supervision regime**, not a different
pretraining objective. That distinction matters: it is not MLM, not paper
reproduction, and it reuses main's existing heads and eval. See §8.

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

Updated after the scope decision: architecture port (not paper reproduction), causal
*and* bidirectional arms wanted, HF models loadable dynamically.

| Arm | Verdict | Why |
|---|---|---|
| **Pure Mamba-2** (hybrid minus attention) | **Do it first** | ~30 lines. Reuses the patched stateful `Mamba2`. Isolates whether the attention branch earns its keep — a question the project has not measured and `main`'s own architecture rests on |
| **EHRMamba / Mamba-1** | **Do it** | The paper's headline model. Stateless over packed context (§4.3), which is what the paper did anyway. No new dependency. Causal only — no attention mask exists to invert |
| **HF adapter** (`AutoModel` behind `SequenceBackbone`) | **Do it** | One file buys `bert`, `roberta`, `gpt2`, `llama`, … as arms, causal or bidirectional, without a file each. `transformers` is *already* an optional extra (`text`). See §9 for what it can and cannot safely wrap |
| **CEHR-BERT** | **Do it, via the adapter** | `model_type="bert"` + `ClinicalEventEmbeddings` + paper hyperparameters (768 wide, depth 5, 512 ctx) is a registry entry, not a new module |
| **Plain BERT** | **Do it, via the adapter** | Same adapter, HF's own embeddings unmodified — the ablation that prices the CEHR embedding fusion itself |
| **Bi-LSTM** | **Do it** | ~40 lines: `nn.LSTM(bidirectional=...)` on `CachedEHREmbeddings`. With one patient per row there are no mid-sequence resets, so the fused cuDNN path works and the `TinyGRUBackbone` per-token Python loop is unnecessary. Both directions from one flag |
| **CEHR-BigBird (causal, masked sparsity)** | **Do it, later** | The one genuinely distinct attention *shape* missing. ~40 lines over the existing attention block. Honest caveat: reproduces the receptive field, not the sparse-kernel speed. The HF adapter cannot supply it (§4.2) |
| **XGBoost** | **Skip** | Superseded. `main` already runs a *tuned* GBM as the strongest reference bar, plus EBM, TabICL, SurvivalPFN and MEDS-Tab. A 2024 untuned XGBoost notebook makes the comparison worse, not broader |
| **MLM pretraining path** | **Skip** | Explicitly out of scope: architecture port, not paper reproduction. `encoder` mode (§8) gives bidirectional arms without it |
| **Multitask Prompted Finetuning (MPF)** | **Skip** | MPF exists to avoid one finetune per task. `main` already trains every task head jointly in one run — the problem does not exist here |

## 7. Two things that are orthogonal: architecture and context regime

The cleanest way to hold all of this is that an arm is a **pair**, not a name:

- **architecture** — what the blocks are (Mamba-1, Mamba-2, hybrid, transformer, LSTM, any HF `model_type`)
- **context regime** — what each position is allowed to see (`causal`, `prefix`, `encoder`; §8)

Most of the confusion in the original request ("both causal and bidirectional") comes
from those two being conflated. BERT is not a bidirectional model; BERT is a
transformer stack that is *usually run* bidirectionally. Separating them means one
registry axis for blocks and one for masks, and every legal combination works without
a special case.

Not every pair is legal, and the registry has to know which:

| Architecture | `causal` | `prefix` | `encoder` |
|---|---|---|---|
| hybrid, transformer, bigbird | ✅ | ✅ | ✅ |
| HF `bert` / `roberta` | ✅ | ✅ | ✅ |
| HF decoder-only (`llama`, `gpt2`) | ✅ | ⚠️ mask-forced, unverified | ⚠️ same |
| Mamba-1 / Mamba-2 / pure SSM | ✅ | ❌ | ❌ |
| LSTM / GRU | ✅ | ❌ | ✅ (as bidirectional) |

Mamba has no attention mask to invert; an SSM's receptive field is a property of the
scan, not of a mask. A bidirectional SSM would mean running a second reversed scan
and concatenating — a different architecture (BiMamba), not a mode. Out of scope
unless asked.

## 8. How a bidirectional arm can be trained here at all

Three context regimes. Only the first exists today.

### `causal` — what main does now

Position *t* sees ≤ *t*. Every head valid, every loss valid, evaluation is one
streaming pass with the landmark mask picking scoring positions
(`alerts.py:_landmark_mask`). Nothing changes.

### `encoder` — fully bidirectional over a truncated history

The regime CEHR-BERT and the Bi-LSTM actually are. Cut the record at a landmark time,
encode what remains bidirectionally, supervise from the final position only.

What it needs, and what it does not:

- **Data: nothing new in the sampler.** `PackedContextSampler` already takes an
  *iterator* of `PatientSequence`. Right-truncation at a random cut is a generator
  wrapping that iterator plus a `_truncate_tail` mirror of the existing
  `_truncate_head` (`packed_context.py`). ~10 lines, no sampler surgery.
- **Targets: nothing new.** `EventHazardTargets` (`training/event_targets.py`) is
  already computed per position from `(subject, visit)` onset/censoring tables. At the
  truncation position they are already the right targets. The change is a mask —
  supervise the last real position, not all of them — not new target machinery.
- **Losses: a subset of the existing ones.** Event hazards and concept heads only.
  Forecast, time-to-next-event and value heads are disabled (they are, definitionally,
  what this regime cannot do).
- **One patient per row.** Required anyway for any arm that cannot honour `reset_mask`
  (§4.4).

**The real cost is evaluation, and it is not small.** `alerts.py` scores every
landmark in a *single* streaming pass, reading hazard outputs at masked positions. A
bidirectional model cannot do that: each landmark needs its own forward over its own
truncated prefix. Landmarks are every 4 hours per visit, so a ten-day stay is ~60
forwards over growing prefixes instead of one pass. Either subsample landmarks for
these arms (and compare *all* arms on the same subsampled set, or the comparison is
not paired), or accept a large eval bill. This should be decided before any encoder
arm is trained, not after.

### `prefix` — bidirectional over a sampled prefix, causal after

The cheap middle. Sample a prefix length per row; attention is bidirectional among
prefix positions and causal everywhere else; the loss is zeroed on prefix positions
(they saw their own targets). Standard PrefixLM/UL2 training.

- Cost: a mask change plus `real_mask &= ~in_prefix`. No new sampler, no new eval, no
  head changes — a prefix-trained model is still scored causally at landmarks.
- Value: genuinely uncertain. It buys bidirectional *encoding of history* while
  keeping every existing loss and the single-pass eval. Whether that helps here is
  exactly the kind of thing this project measures rather than assumes.
- Not a substitute for `encoder`: it does not make CEHR-BERT "the paper's CEHR-BERT".

## 9. Wrapping arbitrary `transformers` models: what actually works

`transformers>=5.16.1` is **already an optional extra** (`pyproject.toml`, `text`), and
`odyssey/text/embed_notes.py:52` already has the deferred-import pattern. So an HF
adapter costs no new dependency policy.

The shape: `AutoConfig.for_model(model_type, **overrides)` → `AutoModel.from_config(cfg)`
(random init — we are porting architectures, not clinical weights), embeddings from
`CachedEHREmbeddings`, hand HF `inputs_embeds`, read `last_hidden_state`.

### The design rule that makes this safe: we own the mask, HF owns the blocks

Verified in `transformers` `masking_utils.py:811`:

```
# If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
    return True, attention_mask, None, None, None, None, None
```

So the block-diagonal causal ∧ same-segment mask `transformer.py:_build_attn_mask`
already builds can be handed straight to an HF model, and `causal` vs `prefix` vs
`encoder` becomes *our* mask function rather than an HF config flag. `BertModel`
routes through `create_causal_mask` when `config.is_decoder` and
`create_bidirectional_mask` otherwise (`modeling_bert.py`, `_create_attention_masks`),
and both early-exit on a 4D mask.

Bonus: HF also infers packing from `position_ids` that do not increment by 1
(`masking_utils.py:729`, `find_packed_sequence_indices`) — and
`transformer.py:_position_ids` already resets to 0 per segment.

### Which families the adapter can honestly accept

Surveyed by grepping each `modeling_*.py` on `transformers` main:

| Family | `inputs_embeds` | causal path | bidirectional path | Verdict |
|---|---|---|---|---|
| `bert`, `roberta` | ✅ | `create_causal_mask` + `is_decoder` | `create_bidirectional_mask` | **Supported.** Both regimes from config |
| `gpt2` | ✅ | ✅ | present in module | **Supported**, bidirectional via forced 4D mask |
| `llama` (and decoder-only kin) | ✅ | ✅ | ✗ | Causal supported; bidirectional is mask-forced and **unverified** |
| `big_bird` | ✅ | only with `original_full` (§4.2) | ✅ | Dense-attention BigBird only — not the sparse arm |
| `longformer`, `modernbert` | ✅ | ✗ | ✅ | **Refuse.** Local/sliding attention builds its own mask; a custom 4D mask may be silently ignored |
| `deberta_v2` | ✅ | ✗ | ✗ | **Refuse.** Disentangled attention, neither mask helper |
| `mamba` | ✅ | n/a | n/a | **Refuse.** No attention mask at all; use the native Mamba arms |

### Two gotchas that must be documented, not discovered

1. **HF re-embeds on top of `inputs_embeds`.** `BertEmbeddings.forward` adds its own
   learned absolute `position_embeddings` and `token_type_embeddings`, then LayerNorm
   and dropout, on top of whatever `inputs_embeds` it is handed. So a wrapped BERT gets
   a second LayerNorm after `ClinicalEventEmbeddings`' own, and a second positional
   signal. Neither is fatal; both must be stated, and `max_position_embeddings` must be
   set ≥ our `max_context` or long rows crash. Pass explicit segment-reset
   `position_ids` (`BertModel.forward` accepts them) or packed rows get row-absolute
   positions.
2. **A silently-ignored mask is the failure mode that matters.** If a family builds its
   own mask, the adapter produces a model that trains fine, converges fine, and has
   seen the future. Nothing downstream would notice.

### Therefore: registration requires a leakage probe, not a whitelist

Every HF arm must pass, at registration time, the test
`test_transformer.py::test_causal_position_invariant_to_future_token_changes` —
change a token at position *t+k*, assert the hidden state at *t* is bit-identical.
It is the only check that actually proves the mask was honoured, it already exists,
and it is cheap. A whitelist encodes today's `transformers` behaviour and rots on the
next release; the probe does not. Run it as a build-time assertion for any arm the
adapter has not seen before.

## 10. Repo context worth knowing before opening a PR upstream

`HANDOFF.md` and `docs/experiment_plan.md` show `main` is mid-crunch on an ML4H 2026
submission (deadline Sept 10 AoE). The backbone-comparison machinery
(`long_history_compare.py`, `make_backbone_table.py`) landed in the most recent commit,
`ca0589c`. A backbone-registry PR is aligned with roadmap Track A item 5, but the
timing is theirs to pick — this work lands on the fork first regardless.
