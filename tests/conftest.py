from __future__ import annotations

import pytest

from transformer_nmt.config import (
    DataSettings,
    InferenceSettings,
    ModalSettings,
    ModelSettings,
    Settings,
    TrackingSettings,
    TrainingSettings,
)
from transformer_nmt.data.dataset import TranslationDataset
from transformer_nmt.data.pipeline import DataBundle
from transformer_nmt.data.preprocessing import WordTokenizer
from transformer_nmt.data.vocabulary import Vocabulary

TINY_PAIRS: list[tuple[str, str]] = [
    ("i am happy .", "je suis heureux ."),
    ("you are tired .", "tu es fatigué ."),
    ("he is late .", "il est en retard ."),
    ("we are friends .", "nous sommes amis ."),
    ("they are here .", "ils sont ici ."),
    ("i love cats .", "j'aime les chats ."),
    ("she likes dogs .", "elle aime les chiens ."),
    ("it is cold .", "il fait froid ."),
    ("i am hungry .", "j'ai faim ."),
    ("you are right .", "tu as raison ."),
    ("he is my brother .", "il est mon frère ."),
    ("we are ready .", "nous sommes prêts ."),
    ("they are students .", "ils sont étudiants ."),
    ("i am a teacher .", "je suis professeur ."),
    ("she is a doctor .", "elle est médecin ."),
    ("it is raining .", "il pleut ."),
    ("where are you ?", "où es tu ?"),
    ("what is this ?", "qu'est ce que c'est ?"),
    ("come here !", "viens ici !"),
    ("thank you very much .", "merci beaucoup ."),
]


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Run every test from a temporary directory with no settings-related env vars."""
    import os

    for key in list(os.environ):
        if key.split("_")[0] in {"DATA", "MODEL", "TRAINING", "TRACKING", "INFERENCE", "MODAL"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def make_tiny_settings(artifacts_dir) -> Settings:
    """Small, fast settings for CPU tests writing under ``artifacts_dir``."""
    return Settings(
        data=DataSettings(
            _env_file=None,
            max_pairs=50,
            max_seq_len=12,
            min_token_freq=1,
            validation_fraction=0.2,
            pad_to_multiple_of=4,
        ),
        model=ModelSettings(
            _env_file=None,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_position_embeddings=32,
        ),
        training=TrainingSettings(
            _env_file=None,
            epochs=1,
            batch_size=4,
            precision="fp32",
            warmup_steps=2,
            log_every_n_steps=1,
            bleu_eval_samples=4,
            checkpoint_every_n_epochs=0,
            seed=0,
        ),
        tracking=TrackingSettings(
            _env_file=None,
            artifacts_dir=artifacts_dir,
            experiment_name="unit-tests",
            log_system_metrics=False,
        ),
        inference=InferenceSettings(_env_file=None),
        modal=ModalSettings(_env_file=None),
    )


def make_tiny_bundle(settings: Settings) -> DataBundle:
    """A DataBundle built from the in-memory tiny corpus."""
    tokenizer = WordTokenizer()
    train_pairs, validation_pairs = TINY_PAIRS[:16], TINY_PAIRS[16:]
    source_vocab = Vocabulary.build((s for s, _ in TINY_PAIRS), tokenizer, 1)
    target_vocab = Vocabulary.build((t for _, t in TINY_PAIRS), tokenizer, 1)
    max_len = settings.data.max_seq_len
    return DataBundle(
        tokenizer=tokenizer,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        train_dataset=TranslationDataset(
            train_pairs, tokenizer, source_vocab, target_vocab, max_len
        ),
        validation_dataset=TranslationDataset(
            validation_pairs, tokenizer, source_vocab, target_vocab, max_len
        ),
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        statistics={"train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs)},
    )


@pytest.fixture
def tiny_settings(tmp_path) -> Settings:
    """Small, fast settings for CPU tests."""
    return make_tiny_settings(tmp_path / "artifacts")


@pytest.fixture
def tokenizer() -> WordTokenizer:
    """Shared tokenizer."""
    return WordTokenizer()


@pytest.fixture
def tiny_bundle(tiny_settings: Settings) -> DataBundle:
    """A DataBundle built from the in-memory tiny corpus."""
    return make_tiny_bundle(tiny_settings)
