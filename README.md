# transformers-paper

A from-scratch PyTorch implementation of the encoder-decoder Transformer from *Attention Is All You Need*
(Vaswani et al., 2017), trained on Tatoeba sentence pairs for English → French, English → Spanish and
Spanish → English translation (one model per direction). Training
runs on Modal GPUs with PyTorch DistributedDataParallel and bf16 mixed precision; every experiment is
tracked in MLflow with checkpoints, metrics, plots and hardware metadata.

A plain-English walkthrough of how every file, Modal, MLflow and the API connect (with sequence
diagrams) lives in [docs/workflow.md](docs/workflow.md); [docs/logic.md](docs/logic.md) follows the code
step by step, from the command you type to the tensors in a training step and the inference path.

## Project layout

```
src/transformer_nmt/
  config.py          Pydantic settings (all knobs are environment variables, see below)
  data/              download, normalisation, word tokenizer, vocabulary, dataset, DDP-aware loaders
  model/             attention, layers (positional encoding, feed-forward), EncoderBlock/DecoderBlock,
                     Encoder, Decoder, EncoderDecoder
  training/          trainer (AMP + DDP), Noam/cosine schedules, metrics (loss, ppl, accuracy, BLEU),
                     checkpoints, MLflow tracker, plots
  inference/         Translator: greedy, beam-search and token-streaming decoding from a checkpoint
  api/               FastAPI inference service: health, languages, model selection, translate, stream
  utils/             seeding, logging, hardware introspection
scripts/
  train.py           training entry point, executed inside the Modal container by torchrun
  modal_train.py     the command you run: rents the GPUs, runs train.py there, syncs results back
  modal_serve.py     hosts the same FastAPI app on a Modal GPU, reading checkpoints from the Volume
tests/               pytest suite (data, model, schedules, metrics, checkpoints, tracking, end to end)
artifacts/           data cache, checkpoints, plots, MLflow store (git-ignored)
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it), a Modal account with a
payment method.

```bash
uv sync                      # creates .venv and installs the locked dependencies
cp .env.example .env         # then fill in your Modal tokens
uv run pytest                # 83 unit and integration tests, CPU only, ~2 min
```

`.env` must contain `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `MODAL_PROFILE`
(from https://modal.com/settings/tokens). Every other variable is optional.

## Configuration

All hyper-parameters live in `src/transformer_nmt/config.py` as Pydantic settings sections. Any field can
be overridden with `PREFIX_FIELD` in `.env` or the shell; the local configuration is exported unchanged to
the Modal container.

| Prefix | Section | Examples |
|---|---|---|
| `DATA_` | corpus and preprocessing | `DATA_SOURCE_URL`, `DATA_ARCHIVE_MEMBER`, `DATA_SOURCE_LANGUAGE`, `DATA_TARGET_LANGUAGE`, `DATA_SWAP_PAIR_ORDER`, `DATA_MAX_SEQ_LEN=40` |
| `MODEL_` | architecture | `MODEL_D_MODEL=256`, `MODEL_NUM_HEADS=8`, `MODEL_NUM_ENCODER_LAYERS=3`, `MODEL_NORM_FIRST=true` |
| `TRAINING_` | optimisation | `TRAINING_EPOCHS=50`, `TRAINING_BATCH_SIZE=128`, `TRAINING_PRECISION=auto`, `TRAINING_WARMUP_STEPS=2000` |
| `TRACKING_` | artifacts and MLflow | `TRACKING_ARTIFACTS_DIR=artifacts`, `TRACKING_EXPERIMENT_NAME=transformer-nmt-eng-fra` |
| `INFERENCE_` | decoding and API | `INFERENCE_BEAM_SIZE=4`, `INFERENCE_MAX_DECODE_LEN=40`, `INFERENCE_MODELS_ROOT=artifacts`, `INFERENCE_AUTOLOAD_BEST_MODEL=true` |
| `MODAL_` | compute | `MODAL_GPU=A100-40GB`, `MODAL_GPU_COUNT=2`, `MODAL_INFERENCE_GPU=T4` |

Defaults follow the paper where it makes sense for a 135k-pair corpus: d_model 256, 8 heads, 3+3 blocks,
FFN 512, dropout 0.1, sinusoidal positions, tied target embeddings, label smoothing 0.1, AdamW
(β=(0.9, 0.98), ε=1e-9) with the Noam warm-up/inverse-sqrt schedule, gradient clipping at 1.0.

The model follows the block structure of the reference course material: `Encoder` and `Decoder`
hold an `nn.ModuleList` of `EncoderBlock`s / `DecoderBlock`s (one block per layer, each with
pre-LayerNorm, multi-head attention, a feed-forward network and residual dropout), built on
`MultiHeadAttention`, `PositionalEncoding` and `FeedForward` from `model/layers.py`; `EncoderDecoder`
ties them together. Additions over the reference are marked with short comments in the code (fused
attention kernel, weight tying, final LayerNorm).

Data: the Tatoeba file is normalised (lower-case, accents and `œ` kept, punctuation split), exact
duplicates are dropped, and the 90/10 split is **grouped by English sentence** so none of the many
sources with several French translations leaks from training into validation.

## Training on Modal

Training never runs on your laptop. The command below is executed locally, but everything it does
happens on Modal:

1. `modal run` imports `scripts/modal_train.py`, authenticates with the tokens from `.env`, builds (or
   reuses) the container image and sends the code to Modal.
2. Modal starts a container with the requested GPUs and runs `train_on_gpu()`, which launches
   `torchrun scripts/train.py` with one process per GPU.
3. The container's output (dataset statistics, one line per epoch, sample translations) is streamed
   back and printed in your terminal in real time while the command is attached.
4. When training ends, results are persisted on the Modal Volume and downloaded to `artifacts/modal/`.

With `--detach` the job keeps running if you close the terminal; `modal app logs -f transformer-nmt`
re-attaches to the live logs, and `--no-train` downloads the finished results later.

```bash
# 1. smoke test on the configured GPUs: 3k pairs, 2 epochs, exercises DDP + bf16 + MLflow + sync
uv run --env-file .env modal run scripts/modal_train.py --smoke

# 2. full run, detached so it survives a closed laptop
uv run --env-file .env modal run --detach scripts/modal_train.py --epochs 25

# follow logs of a detached run, then fetch its results without training again
uv run --env-file .env modal app logs -f transformer-nmt
uv run --env-file .env modal run scripts/modal_train.py --no-train
```

### Training other language pairs

The corpus, the direction and the experiment name are settings, so another Tatoeba pair is a matter of
environment variables. The archives below share the format of the French one (`English<TAB>Other`):

```bash
# English -> Spanish (TensorFlow mirror of the Tatoeba/ManyThings pairs, 118,964 sentences)
DATA_SOURCE_URL=https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip \
DATA_ARCHIVE_MEMBER=spa-eng/spa.txt DATA_TARGET_LANGUAGE=Spanish \
TRACKING_EXPERIMENT_NAME=transformer-nmt-eng-spa \
uv run --env-file .env modal run --detach scripts/modal_train.py --epochs 50

# Spanish -> English: same file, columns swapped
DATA_SOURCE_URL=https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip \
DATA_ARCHIVE_MEMBER=spa-eng/spa.txt DATA_SWAP_PAIR_ORDER=true \
DATA_SOURCE_LANGUAGE=Spanish DATA_TARGET_LANGUAGE=English \
TRACKING_EXPERIMENT_NAME=transformer-nmt-spa-eng \
uv run --env-file .env modal run --detach scripts/modal_train.py --epochs 50
```

Run names carry the direction (`...-eng-spa-...`), each run's `settings.json` records the languages, and
the inference API groups models by direction automatically.

`scripts/modal_train.py` builds the image from `pyproject.toml`/`uv.lock` (`Image.uv_sync`), mounts the
package, launches `torchrun --standalone --nproc_per_node=$MODAL_GPU_COUNT scripts/train.py` and persists
the SQLite MLflow store, run artifacts, checkpoints and plots to the `transformer-nmt-artifacts` Volume
(also on failure, under an account-wide lock so concurrent runs cannot clobber the store).
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
- **metrics per optimizer step** (`step/*`): loss, learning rate, gradient norm, tokens/s;
- **metrics per epoch** (`train/*`, `val/*`): `train/loss` (label-smoothed), `train/grad_norm`,
  `train/skipped_steps` (fp16 overflow steps), `val/loss` (NLL), `val/perplexity`,
  `val/token_accuracy`, `val/bleu` (sacreBLEU, greedy decoding of 500 validation sentences), epoch time,
  `system/peak_memory_gb`, plus MLflow system metrics (CPU, RAM, GPU utilisation and memory);
- **artifacts**: `best.pt`, `settings.json`, both vocabularies, `history.json/csv`, step records, sample
  translations, `training_curves.png` and `step_dynamics.png`.

Checkpoints live in `artifacts/checkpoints/<run>/` (`best.pt`, `last.pt`, `epoch-XXX.pt` every 5 epochs) with
a `best.json` sidecar that the inference tools use to pick the best run.

## Inference

### FastAPI service (recommended)

The API lives in `src/transformer_nmt/api/` and is registered as the FastAPI entrypoint in
`pyproject.toml`, so the [FastAPI CLI](https://fastapi.tiangolo.com/fastapi-cli/) needs no arguments:

```bash
uv run fastapi dev            # http://127.0.0.1:8000, auto-reload, docs at /docs
uv run fastapi run --port 8000   # production mode, binds 0.0.0.0
```

On startup it scans `INFERENCE_MODELS_ROOT` (default `artifacts/`, so both local and Modal-synced runs
are found), reads each run's language direction from its `settings.json`, ranks the runs of every
direction by their recorded validation metric and loads the best non-smoke one per direction.
Languages are two-letter codes (`en`, `es`, `fr`).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | device, GPU name, number of runs, the active model of each direction |
| GET | `/languages` | | available directions with model counts and the active run |
| GET | `/models` | | every run: `run_name`, `source_language`, `target_language`, metric, value, epoch, `run_type`, `active` |
| POST | `/models/select` | `{"run_name"}` | loads that run and makes it active for its direction |
| POST | `/translate` | `{"text", "source_language", "target_language", "beam_size"?}` | `{"translation", "model"}` |
| POST | `/translate/stream` | `{"text", "source_language", "target_language"}` | server-sent events, one `data: {"token": ...}` per word, then `event: done` |

```bash
curl -s localhost:8000/health
curl -s localhost:8000/languages
curl -s -X POST localhost:8000/models/select -H 'Content-Type: application/json' \
     -d '{"run_name": "20261030-2020-a100-sxm4-40gb-bf16-bs256-full"}'
curl -s -X POST localhost:8000/translate -H 'Content-Type: application/json' \
     -d '{"text": "I would like a cup of coffee, please.", "source_language": "en", "target_language": "es"}'
curl -N -X POST localhost:8000/translate/stream -H 'Content-Type: application/json' \
     -d '{"text": "She promised to call me tomorrow.", "source_language": "es", "target_language": "en"}'
```

A direction with no trained model answers `404`; with `INFERENCE_AUTOLOAD_BEST_MODEL=false` a direction
whose model was not selected yet answers `409` until you call `/models/select`.

### Hosted on Modal

`scripts/modal_serve.py` serves the very same FastAPI app from a `MODAL_INFERENCE_GPU` container that
scans the checkpoints persisted on the Volume:

```bash
uv run --env-file .env modal deploy scripts/modal_serve.py   # prints the public URL
curl -s "$URL/health"; curl -s "$URL/languages"
curl -s -X POST "$URL/translate" -H "Content-Type: application/json" \
     -d '{"text": "Good morning.", "source_language": "en", "target_language": "fr"}'
```

## Results

Three models, one per direction, all trained on Modal with DDP (2 ranks), bf16 autocast and a global
batch of 256. Validation sets are source-grouped (no English sentence shared with training) and exact
duplicates are removed before splitting. BLEU is sacreBLEU with greedy decoding on a 500-sentence
validation subset.

| Direction | Run | GPUs | Epochs | Pairs (train / val) | Vocab (src / tgt) | Best val loss (epoch) | Perplexity | Token acc. | BLEU at best / peak | Train time |
|---|---|---|---|---|---|---|---|---|---|---|
| English → French | `20261030-2020-a100-sxm4-40gb-bf16-bs256-full` | 2× A100-40GB | 25 | 122,048 / 13,561 | 8,720 / 14,700 | 1.112 (13) | 3.0 | 0.77 | 42.3 / 44.5 | 7.3 min |
| English → Spanish | `20261030-2141-eng-spa-h100-80gb-hbm3-bf16-bs256-full` | 2× H100 80GB | 50 | 107,040 / 11,893 | 8,527 / 14,305 | 1.140 (12) | 3.1 | 0.77 | 41.8 / 46.6 | 10.2 min |
| Spanish → English | `20261030-2156-spa-eng-h100-80gb-hbm3-bf16-bs256-full` | 2× H100 80GB | 50 | 107,040 / 11,893 | 14,333 / 8,515 | 0.965 (19) | 2.6 | 0.82 | 58.4 / 60.0 | 9.7 min |

In every run the validation loss bottoms out between epochs 12 and 20 while BLEU keeps creeping up a
little as the learning rate decays; `best.pt` (selected on validation loss) is what the API serves, and
the extra epochs cost about 10 GPU-minutes per run. The two Spanish runs were started on A100s but no
A100 pair was available at the time, so they ran on H100s (`MODAL_GPU=H100`); the model is
latency-bound, so the H100s were only marginally faster (about 11.5 s per epoch versus 13.5 s).

Sample outputs from the deployed API (beam size 4):

| Direction | Input | Output |
|---|---|---|
| en → fr | I would like a cup of coffee, please. | j'aimerais une tasse de café s'il vous plaît. |
| en → es | I would like a cup of coffee, please. | quisiera una taza de café por favor. |
| en → es | She promised to call me tomorrow. | ella prometió llamarme mañana. |
| es → en | Ella prometió llamarme mañana. | she promised to call me up tomorrow. |
| es → en | No tenemos suficiente tiempo. | we don't have enough time. |

An earlier English → French run with a plain random split reached a lower validation loss (0.97) only
because thousands of English sentences with several French translations appeared on both sides; its
checkpoints were removed so that `GET /models` only ranks runs trained on comparable data.

Deployed Modal apps: `transformer-nmt` (function `train_on_gpu`) and `transformer-nmt-api`
(FastAPI service at `https://amos-luna-16--transformer-nmt-api-translationservice-web.us-east.modal.run`,
serving all three directions).

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
