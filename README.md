# transformers-paper

A from-scratch PyTorch implementation of the encoder-decoder Transformer from *Attention Is All You Need*
(Vaswani et al., 2017), trained for English → French translation on the Tatoeba sentence pairs. Training
runs on Modal GPUs with PyTorch DistributedDataParallel and bf16 mixed precision; every experiment is
tracked in MLflow with checkpoints, metrics, plots and hardware metadata.

## Project layout

```
src/transformer_nmt/
  config.py          Pydantic settings (all knobs are environment variables, see below)
  data/              download, normalisation, word tokenizer, vocabulary, dataset, DDP-aware loaders
  model/             masks, multi-head attention (SDPA), encoder, decoder, Seq2SeqTransformer
  training/          trainer (AMP + DDP), Noam/cosine schedules, metrics (loss, ppl, accuracy, BLEU),
                     checkpoints, MLflow tracker, plots
  inference/         Translator: greedy and beam-search decoding from a checkpoint
  utils/             seeding, logging, hardware introspection
scripts/
  train.py           training entry point (single process or under torchrun)
  translate.py       CLI inference with the best checkpoint found under artifacts/
  modal_train.py     Modal layer: builds the image, runs torchrun on N GPUs, syncs results back
  modal_serve.py     Modal web API serving the best checkpoint (FastAPI endpoint)
tests/               pytest suite (data, model, schedules, metrics, checkpoints, tracking, end to end)
artifacts/           data cache, checkpoints, plots, MLflow store (git-ignored)
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it), a Modal account with a
payment method.

```bash
uv sync                      # creates .venv and installs the locked dependencies
cp .env.example .env         # then fill in your Modal tokens
uv run pytest                # 60 unit and integration tests, CPU only, ~1 min
```

`.env` must contain `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `MODAL_PROFILE`
(from https://modal.com/settings/tokens). Every other variable is optional.

## Configuration

All hyper-parameters live in `src/transformer_nmt/config.py` as Pydantic settings sections. Any field can
be overridden with `PREFIX_FIELD` in `.env` or the shell; the local configuration is exported unchanged to
the Modal container.

| Prefix | Section | Examples |
|---|---|---|
| `DATA_` | corpus and preprocessing | `DATA_MAX_PAIRS=150000`, `DATA_MAX_SEQ_LEN=40`, `DATA_MIN_TOKEN_FREQ=2` |
| `MODEL_` | architecture | `MODEL_D_MODEL=256`, `MODEL_NUM_HEADS=8`, `MODEL_NUM_ENCODER_LAYERS=3`, `MODEL_NORM_FIRST=true` |
| `TRAINING_` | optimisation | `TRAINING_EPOCHS=25`, `TRAINING_BATCH_SIZE=128`, `TRAINING_PRECISION=auto`, `TRAINING_WARMUP_STEPS=2000` |
| `TRACKING_` | artifacts and MLflow | `TRACKING_ARTIFACTS_DIR=artifacts`, `TRACKING_EXPERIMENT_NAME=transformer-nmt-eng-fra` |
| `INFERENCE_` | decoding | `INFERENCE_BEAM_SIZE=4`, `INFERENCE_MAX_DECODE_LEN=40` |
| `MODAL_` | compute | `MODAL_GPU=A100-40GB`, `MODAL_GPU_COUNT=2`, `MODAL_INFERENCE_GPU=T4` |

Defaults follow the paper where it makes sense for a 135k-pair corpus: d_model 256, 8 heads, 3+3 layers,
FFN 512, dropout 0.1, sinusoidal positions, tied target embeddings, label smoothing 0.1, AdamW
(β=(0.9, 0.98), ε=1e-9) with the Noam warm-up/inverse-sqrt schedule, gradient clipping at 1.0.
Pre-LayerNorm blocks are the default (`MODEL_NORM_FIRST=false` gives the paper's Post-LN).

## Training on Modal

```bash
# 1. smoke test on the configured GPUs: 3k pairs, 2 epochs, exercises DDP + bf16 + MLflow + sync
uv run --env-file .env modal run scripts/modal_train.py --smoke

# 2. full run, detached so it survives a closed laptop
uv run --env-file .env modal run --detach scripts/modal_train.py --epochs 25

# follow logs of a detached run
uv run --env-file .env modal app logs -f transformer-nmt
```

`scripts/modal_train.py` builds the image from `pyproject.toml`/`uv.lock` (`Image.uv_sync`), mounts the
package, launches `torchrun --standalone --nproc_per_node=$MODAL_GPU_COUNT scripts/train.py` and persists
the SQLite MLflow store, run artifacts, checkpoints and plots to the `transformer-nmt-artifacts` Volume.
When the run finishes the local entrypoint downloads everything to `artifacts/modal/` and rewrites the
MLflow artifact URIs so the UI works locally:

```bash
uv run mlflow ui --backend-store-uri sqlite:///artifacts/modal/mlflow.db --port 5001
```

Hosting the training function as a persistent Modal app is one command: `uv run --env-file .env modal deploy
scripts/modal_train.py` (deployed app `transformer-nmt`, function `train_on_gpu`).

### Choosing a GPU

Modal on-demand prices on 2026-10-30 (https://modal.com/pricing); `MODAL_GPU` accepts any of these names.

| GPU | VRAM | bf16 tensor cores | $/GPU-hour | Fit for this project |
|---|---|---|---|---|
| T4 | 16 GB | no (fp16 only) | 0.59 | inference API, fp16 training |
| L4 | 24 GB | yes | 0.80 | cheap training, slower memory |
| A10 | 24 GB | yes | 1.10 | cheapest comfortable training GPU |
| L40S | 48 GB | yes | 1.95 | fast single-GPU training |
| **A100-40GB** | 40 GB | yes | 2.10 | **default: 2× with DDP** |
| A100-80GB | 80 GB | yes | 2.50 | same speed, more memory |
| H100 | 80 GB | yes (+fp8) | 3.95 | overkill for a 10M-parameter model |

This model has ~10M parameters and sentences of ≤40 tokens, so a training step is bound by kernel-launch
latency rather than FLOPs: an H100 is not meaningfully faster than an A100 here, and two A100-40GB with
DDP double the throughput for about the price of one H100. Set `MODAL_GPU=H100 MODAL_GPU_COUNT=1` if you
prefer a single faster card; Modal may upgrade `A100` and `H100` requests to larger cards at no extra cost.

### Precision

`TRAINING_PRECISION=auto` resolves to the best option for the device:

| Mode | What runs | When |
|---|---|---|
| `bf16` | `torch.autocast(dtype=bfloat16)`, no loss scaling | Ampere/Ada/Hopper GPUs (A10, L4, L40S, A100, H100) |
| `fp16` | autocast float16 + `GradScaler` | older GPUs such as T4/V100 |
| `fp32` | full precision, TF32 matmuls enabled on CUDA | CPU, Apple MPS, or when requested |

bf16 keeps fp32's exponent range so it needs no gradient scaler and is the recommended choice; fp16 is
only a fallback. `torch.set_float32_matmul_precision("high")` lets the remaining fp32 matmuls use TF32.

### Distributed training

The trainer uses `DistributedDataParallel` with one process per GPU, `DistributedSampler` sharding,
`all_reduce` of loss/token counts, rank-0-only checkpointing and tracking, and a broadcast early-stopping
decision. The same code runs single-process locally; `WORLD_SIZE` from `torchrun` switches the mode.
The effective batch is `TRAINING_BATCH_SIZE × MODAL_GPU_COUNT`.

## Experiment tracking

Each run creates an MLflow run named like `20261030-1950-a100-sxm4-40gb-bf16-bs256-full` in experiment
`transformer-nmt-eng-fra` (`-smoke` suffix for smoke tests) and logs:

- **params**: every settings field, dataset statistics (pairs, vocab sizes, lengths), model parameter count,
  hardware (GPU name, VRAM, CUDA/cuDNN/torch versions, host), world size, steps per epoch;
- **tags**: project, task, language pair, run type, launcher, device, precision, git commit;
- **metrics per step**: `train/step_loss`, `train/learning_rate`, `train/grad_norm`, `train/tokens_per_second`;
- **metrics per epoch**: `train/loss` (label-smoothed), `val/loss` (NLL), `val/perplexity`,
  `val/token_accuracy`, `val/bleu` (sacreBLEU, greedy decoding of 500 validation sentences), epoch time,
  `system/peak_memory_gb`, plus MLflow system metrics (CPU, RAM, GPU utilisation and memory);
- **artifacts**: `best.pt`, `settings.json`, both vocabularies, `history.json/csv`, step records, sample
  translations, `training_curves.png` and `step_dynamics.png`.

Checkpoints live in `artifacts/checkpoints/<run>/` (`best.pt`, `last.pt`, `epoch-XXX.pt` every 5 epochs) with
a `best.json` sidecar that the inference tools use to pick the best run.

## Inference

```bash
# best checkpoint under artifacts/ (smoke runs are ignored when a full run exists)
uv run python scripts/translate.py "I love this movie." "Where is the train station?"
uv run python scripts/translate.py --beam-size 1 --checkpoint artifacts/modal/checkpoints/<run>/best.pt
uv run python scripts/translate.py            # interactive mode
```

Hosted API on Modal (loads the best checkpoint from the Volume on a `MODAL_INFERENCE_GPU`):

```bash
uv run --env-file .env modal deploy scripts/modal_serve.py
curl -X POST "$TRANSLATE_URL" -H "Content-Type: application/json" \
     -d '{"text": ["I love this movie.", "Where is the station?"], "beam_size": 4}'
```

`modal deploy` prints the endpoint URLs (`.../translate` and `.../health`; `/translate/docs` serves the
OpenAPI UI). `uv run --env-file .env modal run scripts/modal_serve.py --text "Good morning."` spins up an
ephemeral copy and sends one request.

## Results

Final run `20261030-1947-a100-sxm4-40gb-bf16-bs256-full`: 2× NVIDIA A100-SXM4-40GB on Modal, DDP with
2 ranks, bf16 autocast, global batch 256, 25 epochs (11,950 optimizer steps, 478 per epoch), 9.95M
parameters, 122,251 training pairs / 13,584 validation pairs, vocabularies 8,735 (EN) and 14,681 (FR).

| Metric (validation, 13.6k pairs) | Value |
|---|---|
| Best NLL loss (epoch 22, `best.pt`) | 0.974 |
| Perplexity | 2.6 |
| Token accuracy | 0.80 |
| sacreBLEU, greedy, 500-sentence subset (best epoch 14 / final) | 45.2 / 44.5 |
| Training throughput | ~79k target tokens/s |
| Wall time for 25 epochs (excluding validation) | 5.6 min |
| Peak GPU memory per rank | 1.5 GB |

Validation loss plateaus after epoch 17 while training loss keeps falling, so the default of saving
`best.pt` on validation loss (and the optional `TRAINING_EARLY_STOPPING_PATIENCE`) matters more than
extra epochs. Sample outputs from the best checkpoint (beam size 4):

| English | Model output |
|---|---|
| I love this movie. | j'adore ce film. |
| Where is the train station? | où se trouve la gare? |
| She promised to call me tomorrow. | elle a promis de m'appeler demain. |
| Could you help me, please? | pourriez vous m'aider s'il vous plait? |

The deployed Modal API lives at
`https://amos-luna-16--transformer-nmt-api-translationservice-translate.us-east.modal.run`
(`.../health` for the probe); the training function is deployed as app `transformer-nmt`.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.
(2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems 30.
https://arxiv.org/abs/1706.03762

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```
