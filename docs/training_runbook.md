# Training runbook: MIMIC-IV to a trained backbone

Everything from a bare GPU host to a finished run, for any of the 12 arms in
`odyssey/models/backbones/__init__.py`. Written for the `multi-backbone`
branch.

The short version: get credentialed, download MIMIC-IV, extract to MEDS,
point `--train-shard-dir` / `--tuning-shard-dir` at the result, and pick the
arm with two fields in `--config-json`. Steps 1-4 are one-time per machine;
step 5 is once per arm.

---

## 1. PhysioNet credentials

MIMIC-IV is credentialed access, and this is the long pole -- approval takes
days to weeks, so start it before anything else.

1. Register at <https://physionet.org/register/>.
2. Complete CITI "Data or Specimens Only Research" training and upload the
   completion report to your PhysioNet profile.
3. Sign the data use agreement on the
   [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) page.

The public **demo** (<https://physionet.org/content/mimic-iv-demo/2.2/>, 100
subjects) needs none of this and is the right way to prove the pipeline works
while the real application is pending. Everything below works on it unchanged.

## 2. Download

```bash
# Full MIMIC-IV 3.1 (credentialed). Large; use a disk with room to spare,
# and expect this to take a while.
wget -r -N -c -np --user <physionet-user> --ask-password \
  https://physionet.org/files/mimiciv/3.1/
```

`meds-extract-run` can also download it for you (omit `do_download=false` in
step 4), which is simpler but gives you no resumable local copy.

`hosp` and `icu` modules only. MIMIC-IV-ED is a separate dataset under its own
DUA and is not wired in.

## 3. Install

Python >= 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone git@github.com:neuralsignal/odyssey.git
cd odyssey
git checkout multi-backbone
uv sync --dev
```

Then add the extras for the arms you actually plan to train:

```bash
# transformers arms: bert, cehr_bert, roberta, gpt2, llama
uv sync --extra text
```

```bash
# Mamba arms: hybrid, pure_mamba2, ehr_mamba. CUDA/nvcc required.
# TWO STEPS. Do not collapse them -- both orderings of the one-step form
# are confirmed to fail, for two different reasons (see the long comment
# on the `cuda` extra in pyproject.toml).
uv sync --extra cuda

PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=12 \
MAMBA_FORCE_BUILD=TRUE \
uv pip install --no-build-isolation --no-binary mamba-ssm \
  --no-deps --no-cache --reinstall 'mamba-ssm==2.3.0'
```

> The README's one-line `uv sync --extra cuda --no-build-isolation` predates
> that finding and does not work. Use the two-step above.

Verify before trusting a green `uv sync` -- it has exited 0 on a broken
install more than once:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "import mamba_ssm"          # only if you installed the cuda extra
uv run python -c "import transformers"       # only if you installed the text extra
uv run pytest -m "not integration_test" tests/ -q
```

**The transformer, BigBird, LSTM and every `transformers` arm need no CUDA
build at all** -- they are plain PyTorch. If you are not training a Mamba arm,
skip the `cuda` extra entirely and skip the build pain with it. A GPU still
helps; `mamba-ssm` is what needs `nvcc`, not the GPU.

## 4. Extract to MEDS

```bash
# Demo, no credentials, ~minutes. Do this first even if you have the real data.
uv run meds-extract-run spec=MIMIC-IV output_dir=~/data/mimic_demo dataset_key=demo

# Full MIMIC-IV 3.1 from the local copy downloaded in step 2.
uv run meds-extract-run spec=MIMIC-IV output_dir=~/data/mimiciv_3.1_v1 \
    do_download=false input_dir=~/physionet.org/files/mimiciv/3.1
```

`do_download=false` skips **all** downloads, including ten small
concept-mapping CSVs the pipeline pulls from `MIT-LCP/mimic-code` on GitHub
(not from PhysioNet -- they are not part of the MIMIC-IV release). Fetch them
into the input directory's root first, or `extract_code_metadata` fails:

```bash
BASE="https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map"
for f in meas_chartevents_main.csv inputevents_to_rxnorm.csv lab_itemid_to_loinc.csv \
         meas_chartevents_value.csv numerics-summary.csv outputevents_to_loinc.csv \
         d_labitems_to_loinc.csv proc_datetimeevents.csv waveforms-summary.csv proc_itemid.csv; do
  curl -sSL -o "$HOME/physionet.org/files/mimiciv/3.1/$f" "$BASE/$f"
done
```

Check the layout before going further. Everything downstream assumes it:

```bash
ls ~/data/mimiciv_3.1_v1/data/     # -> train/  tuning/  held_out/
ls ~/data/mimiciv_3.1_v1/data/train/*.parquet | head
```

That `data/` directory is `DATA_ROOT` everywhere below.

## 5. Train

Four CLI flags, and every other `TrainingConfig` field through
`--config-json` (inline JSON or a path to a JSON file).

```bash
DATA=~/data/mimiciv_3.1_v1/data

uv run python -m odyssey.training.train \
    --train-shard-dir  $DATA/train \
    --tuning-shard-dir $DATA/tuning \
    --output-dir runs/bert_causal \
    --config-json '{"backbone": "bert", "max_context": 2048, "num_lanes": 8}'
```

Start with `{"max_train_shards": 2, "max_tuning_shards": 1, "num_epochs": 1}`
on the demo extraction to prove the arm runs end to end before committing a
GPU to it.

### Picking the arm

An arm is a pair: `backbone` x `attention_mode`.

| `backbone` | `attention_mode` | Needs | Sampler field |
|---|---|---|---|
| `hybrid` (default) | `causal` | cuda extra | `chunk_size` |
| `pure_mamba2` | `causal` | cuda extra | `chunk_size` |
| `ehr_mamba` | `causal` | cuda extra | `max_context` |
| `transformer` | `causal` / `prefix` / `encoder` | — | `max_context` |
| `bigbird` | `causal` / `prefix` / `encoder` | — | `max_context` |
| `bert`, `cehr_bert`, `roberta`, `gpt2`, `llama` | `causal` / `prefix` / `encoder` | text extra | `max_context` |
| `lstm` | `causal` / `encoder` | — | `max_context` |
| `tiny_gru` | `causal` | — | `chunk_size` |

Stateful arms (`hybrid`, `pure_mamba2`, `tiny_gru`) stream with truncated BPTT
and read `chunk_size`; every other arm gets whole-patient context windows and
reads `max_context`. The wrong one is silently ignored, not an error, so set
the one its row names.

An illegal pair fails at build time with the reason -- `validate_backbone`
raises before anything trains.

### Examples

```jsonc
// The existing default, unchanged.
{"backbone": "hybrid", "chunk_size": 512, "num_lanes": 64}

// Causal BERT at a matched budget.
{"backbone": "bert", "max_context": 2048, "num_lanes": 8}

// CEHR-BERT as published (768 wide, 5 layers, 8 heads), bidirectional.
{"backbone": "cehr_bert", "attention_mode": "encoder", "max_context": 2048}

// Bi-LSTM.
{"backbone": "lstm", "attention_mode": "encoder", "max_context": 2048}

// The paper's Mamba-1 stack.
{"backbone": "ehr_mamba", "max_context": 2048}

// Sparse attention; block geometry via backbone_kwargs.
{"backbone": "bigbird", "max_context": 4096,
 "backbone_kwargs": {"block_size": 64, "num_random_blocks": 3}}
```

`backbone_kwargs` is a free-form dict passed to the backbone constructor. For
`transformers` arms, anything that is not an embedding option goes to the HF
config, so `{"hidden_dropout_prob": 0.2}` or `{"intermediate_size": 1024}`
work without a new config field.

### Four things to know before a real run

**Keep the budget matched.** `hidden_size`, `num_hidden_layers` and
`attn_num_heads` come from the run config for every arm, so hold them fixed
across arms or the comparison measures parameter count. The MLP width is
handled: HF families ignore `hidden_size` when sizing their feedforward
(BERT defaults to 3072, Llama to 11008, regardless), so `_scale_feedforward`
overrides it to `4 * hidden_size` unless you set `intermediate_size` yourself.

**`max_context` is a hard limit on HF arms**, not a soft one. It sizes the
learned position table, and a wider row raises a `ValueError` naming the
limit. Attention is dense over that window, so start at 1024-2048 and raise it
only if memory allows.

**Bidirectional arms need the concept bottleneck.** Under `encoder` or
`prefix` the forecast, time and value heads are switched off (they would be
reading their own targets out of the input), leaving the concept and event
hazard heads to carry the loss. `model_kind: "baseline"` has neither, so that
combination is refused at build time rather than training against a constant
zero.

**Encoder arms change the evaluation protocol.** A bidirectional model cannot
score every landmark in one streaming pass -- each landmark needs its own
forward over its own truncated prefix. `alerts.py` has not been changed for
this. Decide the landmark budget (subsample, and re-score every arm on the
same subsample to keep the comparison paired -- or pay for roughly 60 forwards
per patient) before training an encoder arm at scale, not after.

### Resuming

```bash
--config-json '{"resume_from": "runs/bert_causal/checkpoint_3.pt", ...}'
```

Fast-forwards that epoch's sampler to the checkpoint's position, which is only
correct if `num_lanes` / `chunk_size` / `reset_prob` / `seed` are unchanged.
They are saved alongside the checkpoint and checked; if they differ, the epoch
restarts from its beginning with a warning.

## 6. Evaluate

```bash
scripts/eval_run.sh runs/bert_causal ~/data/mimiciv_3.1_v1/data
```

Stages: `eval`, `interventions`, `attribution`, `alerts`, `cases`, `report`.
A failed stage does not stop the later ones -- read the `=== STAGE ... EXIT`
lines.

For any **stateless** arm (everything but `hybrid`, `pure_mamba2`,
`tiny_gru`), three stages refuse by design and exit non-zero:
`interventions`, `attribution` and `cases` drive the TBTT lane sampler, which
is not the context those arms trained on. `eval`, `alerts` and `report` run
normally, and `alerts` is the one the backbone comparison actually needs.
The refusal messages say which arm and why.

`eval_run.sh` reads `max_context` from the run's own saved `config.json`, so
there is nothing to keep in sync by hand.

## 7. Compare arms

`scripts/long_history_compare.py` produces the paired, subject-clustered
comparison between two runs; `scripts/make_backbone_table.py` renders it.
Report the `whole` and `truncated` strata separately for stateless arms --
`truncated_subject_ids` exists for exactly this, and pooling makes a
context-limited arm look worse than it is.

## Known gaps

- `ehr_mamba` and `pure_mamba2` are ported but have never been executed;
  `mamba-ssm` would not build on the development host. Run them on the demo
  extraction first.
- Landmark evaluation for `encoder` arms still costs one forward per landmark.
- `case_study`, `concept_attribution` and `interventions` refuse stateless
  backbones rather than supporting them.
