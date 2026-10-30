from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN)
PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3

PrecisionMode = Literal["auto", "bf16", "fp16", "fp32"]
SchedulerName = Literal["noam", "cosine"]
BestMetricName = Literal["val/loss", "val/bleu"]

ENV_FILE = ".env"


def _section_config(prefix: str) -> SettingsConfigDict:
    """Build the shared pydantic-settings configuration for one settings section.

    Args:
        prefix: Environment-variable prefix for the section (for example ``TRAINING_``).

    Returns:
        A ``SettingsConfigDict`` that reads ``.env`` from the working directory, ignores
        unrelated variables and makes the section immutable after construction.
    """
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


class DataSettings(BaseSettings):
    """Dataset location and preprocessing hyper-parameters (``DATA_*`` variables)."""

    model_config = _section_config("DATA_")

    source_url: str = Field(
        default="https://download.pytorch.org/tutorial/data.zip",
        description="Archive with the Tatoeba English-French sentence pairs.",
    )
    archive_member: str = Field(
        default="data/eng-fra.txt",
        description="Path of the tab-separated pairs file inside the archive.",
    )
    local_file: Path | None = Field(
        default=None,
        description="Optional pre-downloaded TSV file (English<TAB>French per line).",
    )
    source_language: str = "English"
    target_language: str = "French"
    max_pairs: int = Field(default=150_000, ge=1, description="Pairs read from the file.")
    max_seq_len: int = Field(
        default=40, ge=4, description="Maximum tokens per sentence, including <sos>/<eos>."
    )
    min_token_freq: int = Field(default=2, ge=1, description="Minimum count to enter the vocab.")
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    split_seed: int = 42
    num_workers: int = Field(default=0, ge=0, description="DataLoader worker processes.")
    pad_to_multiple_of: int = Field(
        default=8, ge=1, description="Pad batch lengths to a multiple of this (tensor cores)."
    )


class ModelSettings(BaseSettings):
    """Transformer architecture hyper-parameters (``MODEL_*`` variables)."""

    model_config = _section_config("MODEL_")

    d_model: int = Field(default=256, ge=8)
    num_heads: int = Field(default=8, ge=1)
    num_encoder_layers: int = Field(default=3, ge=1)
    num_decoder_layers: int = Field(default=3, ge=1)
    dim_feedforward: int = Field(default=512, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_position_embeddings: int = Field(
        default=256, ge=8, description="Length of the sinusoidal positional-encoding table."
    )
    norm_first: bool = Field(
        default=True,
        description="Pre-LayerNorm residual blocks (True) or the paper's Post-LayerNorm (False).",
    )
    tie_target_embeddings: bool = Field(
        default=True,
        description="Share the decoder embedding matrix with the output projection (paper §3.4).",
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> ModelSettings:
        """Ensure ``d_model`` is even and divisible by the number of attention heads."""
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encodings")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self


class TrainingSettings(BaseSettings):
    """Optimisation loop hyper-parameters (``TRAINING_*`` variables)."""

    model_config = _section_config("TRAINING_")

    epochs: int = Field(default=25, ge=1)
    batch_size: int = Field(default=128, ge=1, description="Per-device batch size.")
    precision: PrecisionMode = Field(
        default="auto",
        description=(
            "auto picks bf16 on Ampere+/Hopper GPUs, fp16+GradScaler on older GPUs, fp32 otherwise."
        ),
    )
    scheduler: SchedulerName = "noam"
    peak_learning_rate: float = Field(
        default=1e-3, gt=0.0, description="Peak LR for the cosine schedule."
    )
    noam_lr_scale: float = Field(
        default=1.0, gt=0.0, description="Multiplier on the paper's d_model^-0.5 Noam schedule."
    )
    warmup_steps: int = Field(
        default=2000,
        ge=1,
        description="Linear warm-up steps; the paper used 4000 for ~100k steps, we train ~12k.",
    )
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.98, ge=0.0, lt=1.0)
    adam_epsilon: float = Field(default=1e-9, gt=0.0)
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    grad_clip_norm: float = Field(default=1.0, ge=0.0, description="0 disables clipping.")
    seed: int = 42
    log_every_n_steps: int = Field(default=50, ge=1)
    bleu_eval_samples: int = Field(
        default=500, ge=0, description="Validation sentences decoded for BLEU each epoch."
    )
    checkpoint_every_n_epochs: int = Field(default=5, ge=0, description="0 keeps only last/best.")
    early_stopping_patience: int = Field(default=0, ge=0, description="0 disables early stopping.")
    best_metric: BestMetricName = "val/loss"
    compile_model: bool = Field(default=False, description="Wrap the model with torch.compile.")

    @model_validator(mode="after")
    def _validate_selection_metric(self) -> TrainingSettings:
        """BLEU can only drive checkpoint selection when it is computed every epoch."""
        if self.best_metric == "val/bleu" and self.bleu_eval_samples == 0:
            raise ValueError("best_metric='val/bleu' requires bleu_eval_samples > 0")
        return self


class TrackingSettings(BaseSettings):
    """Artifact locations and MLflow experiment tracking (``TRACKING_*`` variables)."""

    model_config = _section_config("TRACKING_")

    artifacts_dir: Path = Field(default=Path("artifacts"))
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="Defaults to a SQLite store at ``artifacts_dir/mlflow.db`` (MLflow 3 default).",
    )
    project_name: str = "attention-is-all-you-need"
    experiment_name: str = "transformer-nmt-eng-fra"
    run_name: str | None = Field(default=None, description="Auto-generated when omitted.")
    log_system_metrics: bool = True
    system_metrics_interval_seconds: int = Field(default=10, ge=1)

    @property
    def data_dir(self) -> Path:
        """Directory holding the downloaded and extracted dataset."""
        return self.artifacts_dir / "data"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory holding one sub-directory of checkpoints per run."""
        return self.artifacts_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        """Directory holding rendered training-curve figures."""
        return self.artifacts_dir / "plots"

    @property
    def mlflow_db_path(self) -> Path:
        """SQLite database file of the default MLflow tracking store."""
        return self.artifacts_dir / "mlflow.db"

    @property
    def mlflow_artifacts_dir(self) -> Path:
        """Root directory where MLflow stores run artifacts (one folder per experiment)."""
        return self.artifacts_dir / "mlartifacts"

    @property
    def resolved_tracking_uri(self) -> str:
        """MLflow tracking URI, defaulting to the local SQLite database."""
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{self.mlflow_db_path.resolve()}"


class InferenceSettings(BaseSettings):
    """Decoding defaults for translation (``INFERENCE_*`` variables)."""

    model_config = _section_config("INFERENCE_")

    checkpoint_path: Path | None = Field(
        default=None, description="Explicit checkpoint; defaults to the best run on disk."
    )
    beam_size: int = Field(default=4, ge=1, description="1 selects greedy decoding.")
    max_decode_len: int = Field(default=40, ge=2)
    length_penalty: float = Field(default=0.6, ge=0.0, description="Beam-search length penalty.")
    models_root: Path | None = Field(
        default=None,
        description="Directory scanned for run checkpoints; defaults to the artifacts directory.",
    )
    autoload_best_model: bool = Field(
        default=True, description="Load the best available run when the API starts."
    )


class ModalSettings(BaseSettings):
    """Modal credentials and compute shape (``MODAL_*`` variables)."""

    model_config = _section_config("MODAL_")

    token_id: SecretStr | None = None
    token_secret: SecretStr | None = None
    profile: str | None = None
    app_name: str = "transformer-nmt"
    gpu: str = Field(default="A100-40GB", description="Modal GPU type, e.g. L40S, A100-40GB, H100.")
    gpu_count: int = Field(default=2, ge=1, le=8, description="GPUs per container (DDP ranks).")
    inference_gpu: str = Field(
        default="T4",
        description="GPU for the hosted translation API (a 10M-param model needs little).",
    )
    inference_scaledown_seconds: int = Field(default=120, ge=2, le=1200)
    volume_name: str = "transformer-nmt-artifacts"
    volume_mount_path: Path = Path("/vol")
    python_version: str = "3.12"
    cpu_cores: float = Field(default=6.0, gt=0.0)
    memory_mb: int = Field(default=16_384, ge=1024)
    timeout_seconds: int = Field(default=4 * 3600, ge=60, le=24 * 3600)

    @field_validator("gpu")
    @classmethod
    def _normalise_gpu(cls, value: str) -> str:
        """Strip whitespace and reject an embedded ``:count`` suffix."""
        value = value.strip()
        if ":" in value:
            raise ValueError("Use MODAL_GPU_COUNT for the GPU count instead of 'TYPE:N'")
        return value

    @property
    def gpu_spec(self) -> str:
        """GPU string accepted by ``modal.App.function(gpu=...)``, e.g. ``L40S:2``."""
        return f"{self.gpu}:{self.gpu_count}" if self.gpu_count > 1 else self.gpu

    def credential_environment(self) -> dict[str, str]:
        """Return the Modal token pair that the client reads from the process env at call time.

        ``MODAL_PROFILE`` is deliberately excluded: the CLI resolves the profile when ``modal`` is
        imported, before any script runs, so it must come from the shell (``uv run --env-file
        .env modal ...``). An empty dict means the client falls back to ``~/.modal.toml``.
        """
        pairs = (
            ("MODAL_TOKEN_ID", self.token_id.get_secret_value() if self.token_id else None),
            (
                "MODAL_TOKEN_SECRET",
                self.token_secret.get_secret_value() if self.token_secret else None,
            ),
        )
        return {key: value for key, value in pairs if value}


class Settings(BaseModel):
    """Aggregate of every configuration section used across the pipeline."""

    model_config = {"frozen": True}

    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    modal: ModalSettings = Field(default_factory=ModalSettings)

    @model_validator(mode="after")
    def _validate_sequence_budget(self) -> Settings:
        """Reject configurations whose padded batches would overflow the positional table."""
        multiple = self.data.pad_to_multiple_of
        padded = -(-self.data.max_seq_len // multiple) * multiple
        if self.model.max_position_embeddings < padded:
            raise ValueError(
                f"model.max_position_embeddings={self.model.max_position_embeddings} is shorter "
                f"than the padded sequence length {padded} (data.max_seq_len rounded up to a "
                f"multiple of data.pad_to_multiple_of={multiple})"
            )
        return self

    def with_overrides(self, **sections: dict[str, Any]) -> Settings:
        """Return a copy with field overrides applied per section.

        Args:
            **sections: Mapping from section name (``training``, ``data``...) to a dict of
                field overrides, for example ``training={"epochs": 2}``.

        Returns:
            A new, validated ``Settings`` instance.
        """
        updated: dict[str, Any] = {}
        for section_name, overrides in sections.items():
            section = getattr(self, section_name)
            unknown = set(overrides) - set(type(section).model_fields)
            if unknown:
                raise ValueError(f"Unknown {section_name} override(s): {sorted(unknown)}")
            updated[section_name] = type(section).model_validate(
                {**section.model_dump(), **overrides}
            )
        return type(self).model_validate({**dict(self), **updated})

    def to_flat_dict(self, *, include_secrets: bool = False) -> dict[str, Any]:
        """Flatten every section to ``section.field -> value`` for parameter logging.

        Args:
            include_secrets: Whether ``SecretStr`` values are revealed. They are dropped
                by default so that credentials never reach experiment trackers.

        Returns:
            A flat dictionary with JSON-compatible values.
        """
        flat: dict[str, Any] = {}
        for section_name, section in self.model_dump(mode="json").items():
            for field_name, value in section.items():
                raw_value = getattr(getattr(self, section_name), field_name)
                if isinstance(raw_value, SecretStr):
                    if not include_secrets:
                        continue
                    value = raw_value.get_secret_value()
                flat[f"{section_name}.{field_name}"] = value
        return flat

    def to_environment(self, *sections: str) -> dict[str, str]:
        """Serialise sections as ``PREFIX_FIELD=value`` pairs understood by the same settings.

        This is how the local configuration travels to a remote container: every value is
        rendered the way pydantic-settings parses it back, ``None`` fields are skipped and
        secrets are never included.

        Args:
            *sections: Section names to export (defaults to data, model, training, tracking).

        Returns:
            Environment-variable mapping.
        """
        chosen = sections or ("data", "model", "training", "tracking")
        environment: dict[str, str] = {}
        for section_name in chosen:
            section = getattr(self, section_name)
            prefix = section.model_config.get("env_prefix", "")
            for field_name, value in section.model_dump(mode="json").items():
                if value is None or isinstance(getattr(section, field_name), SecretStr):
                    continue
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                environment[f"{prefix}{field_name.upper()}"] = rendered
        return environment

    def smoke_test_variant(self) -> Settings:
        """Return settings tuned for a fast end-to-end validation run.

        The variant keeps the real dataset but limits the number of pairs, epochs and BLEU
        samples so the full pipeline executes in a couple of minutes on a laptop.
        """
        return self.with_overrides(
            data={"max_pairs": 3_000},
            training={
                "epochs": 2,
                "batch_size": 32,
                "warmup_steps": 20,
                "log_every_n_steps": 10,
                "bleu_eval_samples": 50,
                "checkpoint_every_n_epochs": 0,
            },
            tracking={"experiment_name": f"{self.tracking.experiment_name}-smoke"},
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the process-wide settings from defaults, ``.env`` and the environment."""
    return Settings()
