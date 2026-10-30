import math

import torch

from transformer_nmt.training.metrics import (
    TokenLossAccumulator,
    TrainingHistory,
    corpus_bleu,
    count_correct_tokens,
    perplexity_from_loss,
)


def test_count_correct_tokens_ignores_padding():
    logits = torch.tensor([[[0.1, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
    targets = torch.tensor([[1, 0, 1]])  # middle position is padding
    assert count_correct_tokens(logits, targets, pad_index=0) == (1, 2)


def test_perplexity_clamps():
    assert math.isclose(perplexity_from_loss(math.log(10)), 10.0)
    assert perplexity_from_loss(1000.0) == math.exp(20.0)


def test_corpus_bleu_bounds():
    assert math.isclose(corpus_bleu(["je suis là ."], ["je suis là ."]), 100.0)
    assert corpus_bleu([], []) == 0.0
    assert corpus_bleu(["rien du tout"], ["je suis là ."]) < 5.0


def test_accumulator_tensor_roundtrip():
    accumulator = TokenLossAccumulator()
    accumulator.update(loss_sum=6.0, token_count=3, correct_tokens=2)
    restored = TokenLossAccumulator.from_tensor(accumulator.as_tensor(torch.device("cpu")))
    assert restored.mean_loss == 2.0 and math.isclose(restored.accuracy, 2 / 3)


def test_history_series_best_and_export(tmp_path):
    history = TrainingHistory()
    history.append({"epoch": 1, "val/loss": 2.0, "val/bleu": 5.0})
    history.append({"epoch": 2, "val/loss": 1.5})
    assert history.series("val/loss") == [2.0, 1.5]
    assert history.best("val/loss", minimize=True) == (2, 1.5)
    assert history.best("val/bleu", minimize=False) == (1, 5.0)
    history.save_json(tmp_path / "h.json")
    history.save_csv(tmp_path / "h.csv")
    assert (tmp_path / "h.csv").read_text().splitlines()[0] == "epoch,val/loss,val/bleu"
