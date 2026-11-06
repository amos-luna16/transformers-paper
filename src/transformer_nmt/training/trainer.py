from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from transformer_nmt.config import Settings
from transformer_nmt.data.dataset import PadCollator, build_dataloader
from transformer_nmt.data.pipeline import DataBundle, prepare_data
from transformer_nmt.model.encoder_decoder import EncoderDecoder
from transformer_nmt.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    BEST_SUMMARY_NAME,
    LAST_CHECKPOINT_NAME,
    SETTINGS_NAME,
    SOURCE_VOCAB_NAME,
    TARGET_VOCAB_NAME,
    BestCheckpointSummary,
    save_checkpoint,
    save_run_metadata,
    unwrap_model,
)
from transformer_nmt.training.distributed import (
    DistributedContext,
    all_reduce_sum,
    barrier,
    initialize_distributed,
    shutdown_distributed,
)
from transformer_nmt.training.metrics import TokenLossAccumulator, TrainingHistory, corpus_bleu
from transformer_nmt.training.plotting import plot_step_dynamics, plot_training_curves
from transformer_nmt.training.precision import (
    PrecisionPolicy,
    autocast_context,
    create_grad_scaler,
    enable_tensor_core_math,
    resolve_precision,
)
from transformer_nmt.training.scheduler import create_lr_scheduler
from transformer_nmt.training.tracking import ExperimentTracker
from transformer_nmt.utils.hardware import collect_hardware_info, git_commit_hash, peak_memory_gb
from transformer_nmt.utils.logging import configure_logging
from transformer_nmt.utils.seed import seed_everything

logger = logging.getLogger(__name__)

HISTORY_JSON_NAME = "history.json"
HISTORY_CSV_NAME = "history.csv"
STEP_RECORDS_NAME = "step_records.json"
TRAINING_CURVES_NAME = "training_curves.png"
STEP_DYNAMICS_NAME = "step_dynamics.png"
SAMPLE_TRANSLATIONS_NAME = "sample_translations.json"
NUM_SAMPLE_TRANSLATIONS = 12


@dataclass(frozen=True)
class TrainingSummary:
    """Outcome of a training run, returned to callers and serialised by the Modal layer."""

    run_name: str
    mlflow_run_id: str | None
    epochs_completed: int
    best_metric: str
    best_metric_value: float | None
    best_epoch: int | None
    best_checkpoint_path: str | None
    total_train_time_seconds: float
    final_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


def language_pair_slug(
    source_language: str,
    target_language: str,
) -> str:
    """Short corpus-style tag for a language pair, e.g. ``eng-spa`` for English to Spanish."""
    return f"{source_language[:3].lower()}-{target_language[:3].lower()}"


def generate_run_name(
    device_name: str,
    precision: str,
    global_batch_size: int,
    *,
    run_type: str,
    language_pair: str = "eng-fra",
) -> str:
    """Build a descriptive run name such as ``20261030-1432-eng-spa-a100-bf16-bs256-full``."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    gpu_slug = device_name.lower().replace("nvidia", "").strip().replace(" ", "-").replace("_", "-")
    gpu_slug = "-".join(part for part in gpu_slug.split("-") if part)[:24] or "cpu"
    return f"{timestamp}-{language_pair}-{gpu_slug}-{precision}-bs{global_batch_size}-{run_type}"


class Trainer:
    """Mixed-precision, optionally distributed trainer for :class:`EncoderDecoder`."""

    def __init__(
        self,
        settings: Settings,
        data: DataBundle,
        context: DistributedContext,
        tracker: ExperimentTracker,
        run_name: str,
        run_type: str = "full",
    ) -> None:
        """Build model, optimiser, schedule, loaders and output locations.

        Args:
            settings: Global settings.
            data: Prepared datasets and vocabularies.
            context: Distributed coordinates of this process.
            tracker: Experiment tracker (no-op on non-main ranks).
            run_name: Name of the run, used for the checkpoint directory.
            run_type: ``"full"`` or ``"smoke"``; recorded in the best-checkpoint sidecar.
        """
        self.settings = settings
        self.data = data
        self.context = context
        self.tracker = tracker
        self.run_name = run_name
        self.run_type = run_type
        self.device = context.device
        self.pad_index = data.target_vocab.pad_index
        self.run_dir = settings.tracking.checkpoints_dir / run_name
        self.plots_dir = settings.tracking.plots_dir / run_name

        enable_tensor_core_math()
        self.precision: PrecisionPolicy = resolve_precision(
            settings.training.precision, self.device
        )
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scaler = create_grad_scaler(self.precision, self.device)
        self.train_loader, self.train_sampler = self._build_loader(train=True)
        self.validation_loader, _ = self._build_loader(train=False)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * settings.training.epochs
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            name=settings.training.scheduler,
            d_model=settings.model.d_model,
            warmup_steps=settings.training.warmup_steps,
            total_steps=self.total_steps,
            noam_scale=settings.training.noam_lr_scale,
        )
        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_index, 
            label_smoothing=settings.training.label_smoothing
        )
        self.eval_criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_index, 
            reduction="sum"
        )

        self.history = TrainingHistory()
        self.step_records: list[dict[str, float]] = []
        self.global_step = 0
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.show_progress_bar = context.is_main_process and sys.stdout.isatty()
        # MPS has no float64; keep reductions in the widest dtype the device supports.
        self.accumulator_dtype = torch.float32 if self.device.type == "mps" else torch.float64

    # ------------------------------------------------------------------ construction helpers
    def _build_model(self) -> nn.Module:
        """Instantiate the Transformer and wrap it for compilation/DDP as configured."""
        model: nn.Module = EncoderDecoder(
            src_vocab_size=len(self.data.source_vocab),
            tgt_vocab_size=len(self.data.target_vocab),
            config=self.settings.model,
            pad_idx=self.pad_index,
        ).to(self.device)
        if self.settings.training.compile_model:
            model = torch.compile(model, dynamic=True)
        if self.context.is_distributed:
            device_ids = [self.context.local_rank] if self.device.type == "cuda" else None
            model = DistributedDataParallel(
                model, 
                device_ids=device_ids
            )
        return model

    def _build_optimizer(self) -> torch.optim.AdamW:
        """AdamW with the paper's betas/epsilon; no decay on biases, norms and embeddings."""
        training = self.settings.training
        decay, no_decay = [], []
        for name, parameter in unwrap_model(self.model).named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.dim() < 2 or "token_emb" in name else decay).append(parameter)
        base_lr = 1.0 if training.scheduler == "noam" else training.peak_learning_rate
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_epsilon,
        )

    def _build_loader(
        self,
        *,
        train: bool,
    ) -> tuple[Any, Any]:
        """Create the train or validation DataLoader for this rank."""
        collator = PadCollator(self.pad_index, self.settings.data.pad_to_multiple_of)
        return build_dataloader(
            self.data.train_dataset if train else self.data.validation_dataset,
            batch_size=self.settings.training.batch_size,
            shuffle=train,
            collator=collator,
            num_workers=self.settings.data.num_workers,
            distributed=self.context.is_distributed,
            seed=self.settings.training.seed,
            pin_memory=self.device.type == "cuda",
        )

    @property
    def global_batch_size(self) -> int:
        """Sentences processed per optimisation step across all ranks."""
        return self.settings.training.batch_size * self.context.world_size

    # ------------------------------------------------------------------ one epoch
    def _split_target(
        self,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return decoder input (``<sos>`` + tokens) and expected output (tokens + ``<eos>``)."""
        return target[:, :-1], target[:, 1:]

    def _batch_statistics(
        self,
        logits: Tensor,
        target_output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(non_pad_tokens, correct_tokens)`` as device tensors (no host sync)."""
        mask = target_output != self.pad_index
        correct = ((logits.argmax(dim=-1) == target_output) & mask).sum()
        return mask.sum(), correct

    def _train_epoch(
        self,
        epoch: int,
    ) -> dict[str, float]:
        """Run one optimisation pass over the training shard of this rank."""
        training = self.settings.training
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        sums = torch.zeros(
            3, dtype=self.accumulator_dtype, device=self.device
        )  # loss, tokens, correct
        grad_norm_sum = torch.zeros((), dtype=self.accumulator_dtype, device=self.device)
        finite_grad_steps = torch.zeros((), dtype=self.accumulator_dtype, device=self.device)
        window_tokens = torch.zeros((), dtype=self.accumulator_dtype, device=self.device)
        window_start = time.perf_counter()
        epoch_start = window_start
        max_norm = training.grad_clip_norm if training.grad_clip_norm > 0 else float("inf")

        progress = tqdm(
            self.train_loader,
            desc=f"epoch {epoch}/{training.epochs}",
            disable=not self.show_progress_bar,
            leave=False,
        )
        for step_in_epoch, (source, target) in enumerate(progress, start=1):
            source = source.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            target_input, target_output = self._split_target(target)

            with autocast_context(self.precision, self.device):
                logits = self.model(source, target_input)
                loss = self.train_criterion(
                    logits.reshape(-1, logits.size(-1)), target_output.reshape(-1)
                )

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            with torch.no_grad():
                tokens, correct = self._batch_statistics(logits.detach(), target_output)
                sums += torch.stack([loss.detach() * tokens, tokens, correct]).to(
                    self.accumulator_dtype
                )
                # fp16 overflow steps (skipped by GradScaler) report a non-finite norm.
                finite = torch.isfinite(grad_norm)
                grad_norm_sum += torch.where(finite, grad_norm, torch.zeros_like(grad_norm)).to(
                    self.accumulator_dtype
                )
                finite_grad_steps += finite.to(self.accumulator_dtype)
                window_tokens += tokens.to(self.accumulator_dtype)

            if self.global_step % training.log_every_n_steps == 0:
                step_loss = float(loss.item())  # the only host sync of the window
                step_grad_norm = float(grad_norm.item())
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                record = {
                    "step": float(self.global_step),
                    "step/loss": step_loss,
                    "step/learning_rate": float(self.scheduler.get_last_lr()[0]),
                    "step/grad_norm": step_grad_norm if math.isfinite(step_grad_norm) else None,
                    "step/tokens_per_second": float(window_tokens.item())
                    * self.context.world_size
                    / elapsed,
                }
                self.step_records.append(record)
                self.tracker.log_metrics(
                    {k: v for k, v in record.items() if k != "step"}, step=self.global_step
                )
                if self.show_progress_bar:
                    progress.set_postfix(
                        loss=f"{record['step/loss']:.3f}", lr=f"{record['step/learning_rate']:.2e}"
                    )
                else:
                    logger.info(
                        "epoch %d step %d/%d | loss %.4f | lr %.2e | grad %s | %.0f tok/s",
                        epoch,
                        step_in_epoch,
                        self.steps_per_epoch,
                        record["step/loss"],
                        record["step/learning_rate"],
                        f"{step_grad_norm:.2f}" if math.isfinite(step_grad_norm) else "inf",
                        record["step/tokens_per_second"],
                    )
                window_tokens.zero_()
                window_start = time.perf_counter()

        epoch_seconds = time.perf_counter() - epoch_start
        reduced = TokenLossAccumulator.from_tensor(all_reduce_sum(sums, self.context))
        finite_steps = float(finite_grad_steps.item())
        return {
            "train/loss": reduced.mean_loss,
            "train/perplexity": reduced.perplexity,
            "train/token_accuracy": reduced.accuracy,
            "train/grad_norm": float(grad_norm_sum.item()) / max(finite_steps, 1.0),
            "train/skipped_steps": float(self.steps_per_epoch) - finite_steps,
            "train/learning_rate": float(self.scheduler.get_last_lr()[0]),
            "train/tokens_per_second": reduced.token_count / max(epoch_seconds, 1e-9),
            "train/epoch_seconds": epoch_seconds,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Compute NLL, perplexity and token accuracy over the validation set (all ranks)."""
        self.model.eval()
        sums = torch.zeros(3, dtype=self.accumulator_dtype, device=self.device)
        for source, target in self.validation_loader:
            source = source.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            target_input, target_output = self._split_target(target)
            with autocast_context(self.precision, self.device):
                logits = self.model(source, target_input)
            loss_sum = self.eval_criterion(
                logits.float().reshape(-1, logits.size(-1)), target_output.reshape(-1)
            )
            tokens, correct = self._batch_statistics(logits, target_output)
            sums += torch.stack([loss_sum, tokens, correct]).to(self.accumulator_dtype)
        reduced = TokenLossAccumulator.from_tensor(all_reduce_sum(sums, self.context))
        return {
            "val/loss": reduced.mean_loss,
            "val/perplexity": reduced.perplexity,
            "val/token_accuracy": reduced.accuracy,
        }

    @torch.no_grad()
    def _decode_validation_subset(
        self,
        num_samples: int,
        batch_size: int = 128,
    ) -> tuple[list[str], list[str], list[str]]:
        """Greedy-decode the first ``num_samples`` validation sentences (main process only)."""
        model = unwrap_model(self.model)
        collator = PadCollator(self.pad_index, self.settings.data.pad_to_multiple_of)
        sources, references, hypotheses = [], [], []
        count = min(num_samples, len(self.data.validation_dataset))
        for start in range(0, count, batch_size):
            indices = range(start, min(start + batch_size, count))
            source, _ = collator([self.data.validation_dataset[i] for i in indices])
            with autocast_context(self.precision, self.device):
                generated = model.greedy_decode(
                    source.to(self.device),
                    sos_idx=self.data.target_vocab.sos_index,
                    eos_idx=self.data.target_vocab.eos_index,
                    max_len=self.settings.data.max_seq_len,
                )
            for i, row in zip(indices, generated.tolist(), strict=True):
                src_text, ref_text = self.data.validation_pairs[i]
                sources.append(src_text)
                references.append(ref_text)
                hypotheses.append(" ".join(self.data.target_vocab.decode(row)))
        return sources, references, hypotheses

    def _evaluate_bleu(self) -> float | None:
        """Corpus BLEU on a fixed validation subset, computed by the main process."""
        if self.settings.training.bleu_eval_samples == 0:
            return None
        bleu: float | None = None
        if self.context.is_main_process:
            _, references, hypotheses = self._decode_validation_subset(
                self.settings.training.bleu_eval_samples
            )
            bleu = corpus_bleu(hypotheses, references)
        barrier(self.context)
        return bleu

    # ------------------------------------------------------------------ checkpoints & bookkeeping
    def _is_improvement(
        self,
        value: float,
    ) -> bool:
        """Whether ``value`` beats the best value recorded so far for the selection metric."""
        if self.best_value is None:
            return True
        minimize = self.settings.training.best_metric.endswith("loss")
        return value < self.best_value if minimize else value > self.best_value

    def _save_checkpoints(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Write last/best/periodic checkpoints (main process only)."""
        if not self.context.is_main_process:
            return
        training = self.settings.training
        common = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric_value": self.best_value,
            "extra": {"run_name": self.run_name, "metrics": metrics},
        }
        save_checkpoint(self.run_dir / LAST_CHECKPOINT_NAME, **common)
        if training.checkpoint_every_n_epochs and epoch % training.checkpoint_every_n_epochs == 0:
            save_checkpoint(self.run_dir / f"epoch-{epoch:03d}.pt", **common)
        selection_value = metrics.get(training.best_metric)
        if selection_value is None:
            raise RuntimeError(
                f"Selection metric {training.best_metric!r} missing from epoch metrics "
                f"{sorted(metrics)}; cannot select the best checkpoint"
            )
        if self._is_improvement(selection_value):
            self.best_value, self.best_epoch, self.epochs_without_improvement = (
                selection_value,
                epoch,
                0,
            )
            common["best_metric_value"] = self.best_value
            best_path = save_checkpoint(self.run_dir / BEST_CHECKPOINT_NAME, **common)
            BestCheckpointSummary(
                run_name=self.run_name,
                metric=training.best_metric,
                value=selection_value,
                epoch=epoch,
                checkpoint_file=best_path.name,
                run_type=self.run_type,
            ).save(self.run_dir / BEST_SUMMARY_NAME)
            logger.info(
                "New best %s=%.4f at epoch %d -> %s",
                training.best_metric,
                selection_value,
                epoch,
                best_path,
            )
        else:
            self.epochs_without_improvement += 1

    def _should_stop_early(self) -> bool:
        """Early-stopping decision taken on rank 0 and shared with every rank."""
        patience = self.settings.training.early_stopping_patience
        stop = bool(
            patience
            and self.context.is_main_process
            and self.epochs_without_improvement >= patience
        )
        flag = torch.tensor(
            [1.0 if stop else 0.0], dtype=self.accumulator_dtype, device=self.device
        )
        return bool(all_reduce_sum(flag, self.context).item() > 0)

    def _log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Emit the one-line epoch summary."""
        bleu = metrics.get("val/bleu")
        logger.info(
            "epoch %d/%d | train loss %.4f ppl %.1f acc %.3f | val loss %.4f ppl %.1f acc %.3f%s"
            " | lr %.2e | %.1fs | %.0f tok/s%s",
            epoch,
            self.settings.training.epochs,
            metrics["train/loss"],
            metrics["train/perplexity"],
            metrics["train/token_accuracy"],
            metrics["val/loss"],
            metrics["val/perplexity"],
            metrics["val/token_accuracy"],
            f" bleu {bleu:.2f}" if bleu is not None else "",
            metrics["train/learning_rate"],
            metrics["train/epoch_seconds"],
            metrics["train/tokens_per_second"],
            f" | peak mem {metrics['system/peak_memory_gb']:.2f} GB"
            if "system/peak_memory_gb" in metrics
            else "",
        )

    def _finalize(
        self,
        total_seconds: float,
    ) -> TrainingSummary:
        """Persist history, plots, sample translations and artifacts (main process only)."""
        final_metrics = dict(self.history.rows[-1]) if self.history.rows else {}
        final_metrics.pop("epoch", None)
        summary = TrainingSummary(
            run_name=self.run_name,
            mlflow_run_id=self.tracker.run_id,
            epochs_completed=len(self.history.rows),
            best_metric=self.settings.training.best_metric,
            best_metric_value=self.best_value,
            best_epoch=self.best_epoch,
            best_checkpoint_path=str((self.run_dir / BEST_CHECKPOINT_NAME).resolve())
            if self.best_epoch
            else None,
            total_train_time_seconds=total_seconds,
            final_metrics={
                k: float(v) for k, v in final_metrics.items() if isinstance(v, int | float)
            },
        )
        if not self.context.is_main_process:
            return summary

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history.save_json(self.run_dir / HISTORY_JSON_NAME)
        self.history.save_csv(self.run_dir / HISTORY_CSV_NAME)
        curves = plot_training_curves(
            self.history, self.plots_dir / TRAINING_CURVES_NAME, title=self.run_name
        )
        self.tracker.log_figure(curves, f"plots/{TRAINING_CURVES_NAME}")
        if self.step_records:
            dynamics = plot_step_dynamics(
                self.step_records, self.plots_dir / STEP_DYNAMICS_NAME, title=self.run_name
            )
            self.tracker.log_figure(dynamics, f"plots/{STEP_DYNAMICS_NAME}")
            self.tracker.log_dict(self.step_records, STEP_RECORDS_NAME)

        sources, references, hypotheses = self._decode_validation_subset(NUM_SAMPLE_TRANSLATIONS)
        samples = [
            {"source": s, "reference": r, "hypothesis": h}
            for s, r, h in zip(sources, references, hypotheses, strict=True)
        ]
        self.tracker.log_dict(samples, SAMPLE_TRANSLATIONS_NAME)
        source_name = self.settings.data.source_language
        target_name = self.settings.data.target_language
        for sample in samples[:5]:
            logger.info(
                "%s: %s | %s ref: %s | %s hyp: %s",
                source_name,
                sample["source"],
                target_name,
                sample["reference"],
                target_name,
                sample["hypothesis"],
            )

        for name in (
            HISTORY_JSON_NAME,
            HISTORY_CSV_NAME,
            SETTINGS_NAME,
            SOURCE_VOCAB_NAME,
            TARGET_VOCAB_NAME,
            BEST_SUMMARY_NAME,
            BEST_CHECKPOINT_NAME,
        ):
            path = self.run_dir / name
            if path.exists():
                in_checkpoints = name in (BEST_CHECKPOINT_NAME, BEST_SUMMARY_NAME)
                self.tracker.log_artifact(
                    path, artifact_path="checkpoints" if in_checkpoints else "run"
                )
        self.tracker.log_metrics(
            {
                "summary/best_metric_value": self.best_value,
                "summary/best_epoch": self.best_epoch,
                "summary/epochs_completed": summary.epochs_completed,
                "summary/total_train_time_seconds": total_seconds,
            },
            step=summary.epochs_completed,
        )
        self.tracker.set_tags(
            {"best_checkpoint": summary.best_checkpoint_path or "", "status_detail": "completed"}
        )
        return summary

    # ------------------------------------------------------------------ public API
    def fit(self) -> TrainingSummary:
        """Train for the configured epochs, validating, checkpointing and tracking each one."""
        if self.context.is_main_process:
            save_run_metadata(
                self.run_dir, self.settings, self.data.source_vocab, self.data.target_vocab
            )
        logger.info(
            "Training %s parameters on %s | precision %s | world size %d | global batch %d"
            " | %d steps/epoch",
            f"{unwrap_model(self.model).count_parameters():,}",
            self.device,
            self.precision.name,
            self.context.world_size,
            self.global_batch_size,
            self.steps_per_epoch,
        )
        start = time.perf_counter()
        for epoch in range(1, self.settings.training.epochs + 1):
            metrics = self._train_epoch(epoch)
            metrics.update(self._validate())
            bleu = self._evaluate_bleu()
            if bleu is not None:
                metrics["val/bleu"] = bleu
            peak = peak_memory_gb(self.device)
            if peak is not None:
                metrics["system/peak_memory_gb"] = peak
            self.history.append({"epoch": epoch, **metrics})
            self.tracker.log_metrics(metrics, step=epoch)
            self._save_checkpoints(epoch, metrics)
            if self.context.is_main_process:
                self._log_epoch(epoch, metrics)
            if self._should_stop_early():
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    self.epochs_without_improvement,
                )
                break
        return self._finalize(time.perf_counter() - start)


def run_training(
    settings: Settings,
    *,
    run_type: str = "full",
    launcher: str = "local",
    device: str | None = None,
) -> TrainingSummary:
    """End-to-end entry point: distributed setup, data, tracking, training and teardown.

    Args:
        settings: Fully resolved settings for the run.
        run_type: ``"full"`` or ``"smoke"``; recorded as an MLflow tag and in the run name.
        launcher: ``"local"`` or ``"modal"``; recorded as an MLflow tag.
        device: Explicit device for single-process runs.

    Returns:
        The :class:`TrainingSummary` of the run.
    """
    context = initialize_distributed(device)
    configure_logging(context.rank)
    seed_everything(settings.training.seed)
    try:
        if context.is_main_process:
            data = prepare_data(settings)
        barrier(context)
        if not context.is_main_process:
            data = prepare_data(settings)

        hardware = collect_hardware_info(context.device)
        precision = resolve_precision(settings.training.precision, context.device)
        global_batch = settings.training.batch_size * context.world_size
        run_name = settings.tracking.run_name or generate_run_name(
            hardware.device_name,
            precision.name,
            global_batch,
            run_type=run_type,
            language_pair=language_pair_slug(
                settings.data.source_language, settings.data.target_language
            ),
        )
        tracker = ExperimentTracker(settings.tracking, enabled=context.is_main_process)
        tags = {
            "project": settings.tracking.project_name,
            "task": "machine-translation",
            "language_pair": f"{settings.data.source_language}->{settings.data.target_language}",
            "model_family": "transformer-encoder-decoder",
            "run_type": run_type,
            "launcher": launcher,
            "device": hardware.device_name,
            "precision": precision.name,
            "world_size": context.world_size,
            "git_commit": git_commit_hash() or "unknown",
        }
        with tracker.run(run_name, tags=tags):
            trainer = Trainer(settings, data, context, tracker, run_name, run_type=run_type)
            tracker.log_params(
                {
                    **settings.to_flat_dict(),
                    **{f"dataset.{k}": v for k, v in data.statistics.items()},
                    **{f"hardware.{k}": v for k, v in hardware.to_dict().items()},
                    "model.parameters": unwrap_model(trainer.model).count_parameters(),
                    "training.global_batch_size": trainer.global_batch_size,
                    "training.steps_per_epoch": trainer.steps_per_epoch,
                    "training.total_steps": trainer.total_steps,
                    "training.resolved_precision": precision.name,
                    "training.world_size": context.world_size,
                }
            )
            summary = trainer.fit()
        logger.info("Training finished: %s", summary.to_dict())
        return summary
    finally:
        shutdown_distributed(context)
