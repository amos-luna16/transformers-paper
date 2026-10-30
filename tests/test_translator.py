import torch
from torch import nn

from transformer_nmt.config import SPECIAL_TOKENS, ModelSettings
from transformer_nmt.data.preprocessing import WordTokenizer
from transformer_nmt.data.vocabulary import Vocabulary
from transformer_nmt.inference.translator import Translator

TOKEN_A, TOKEN_B = 4, 5


class _ScriptedDecoder(nn.Module):
    """Stub model whose next-token log-probabilities follow a fixed script per step."""

    def __init__(self, vocab_size: int, script) -> None:
        super().__init__()
        self.config = ModelSettings(
            _env_file=None, d_model=8, num_heads=2, max_position_embeddings=16
        )
        self.pad_index = 0
        self.vocab_size = vocab_size
        self.script = script

    def encode(self, source):
        return torch.zeros(1, source.size(1), 8), torch.ones(
            1, 1, 1, source.size(1), dtype=torch.bool
        )

    def decode(self, target, memory, memory_mask):
        step = target.size(1) - 1
        logits = torch.full((target.size(0), target.size(1), self.vocab_size), -30.0)
        for token, log_prob in self.script(step).items():
            logits[:, -1, token] = log_prob
        return logits

    def greedy_decode(self, source, *, sos_index, eos_index, max_len):
        tokens = [
            max(self.script(step).items(), key=lambda kv: kv[1])[0] for step in range(max_len)
        ]
        return torch.tensor([tokens])


def _translator(script, *, max_decode_len):
    source_vocab = Vocabulary([*SPECIAL_TOKENS, "x"])
    target_vocab = Vocabulary([*SPECIAL_TOKENS, "a", "b"])
    model = _ScriptedDecoder(len(target_vocab), script)
    return Translator(
        model,
        source_vocab,
        target_vocab,
        WordTokenizer(),
        torch.device("cpu"),
        max_decode_len=max_decode_len,
    )


def test_beam_search_lets_alive_beams_compete_when_budget_is_exhausted():
    eos = 3
    script = lambda step: {TOKEN_A: -0.1, TOKEN_B: -0.2, eos: -8.0}  # noqa: E731
    translator = _translator(script, max_decode_len=3)
    assert (
        translator.beam_search(torch.tensor([[4]]), beam_size=2, length_penalty=0.6)
        == [TOKEN_A] * 3
    )
    assert translator.translate("x", beam_size=2) == "a a a"


def test_beam_search_returns_immediately_finished_hypothesis():
    eos = 3
    script = lambda step: {eos: -0.01, TOKEN_A: -5.0}  # noqa: E731
    translator = _translator(script, max_decode_len=5)
    assert translator.beam_search(torch.tensor([[4]]), beam_size=3, length_penalty=0.6) == [eos]
    assert translator.translate("x", beam_size=3) == ""


def test_beam_search_prefers_better_scored_longer_hypothesis():
    eos = 3
    script = lambda step: {TOKEN_A: -0.1, eos: -3.0} if step < 2 else {eos: -0.05, TOKEN_A: -4.0}  # noqa: E731
    translator = _translator(script, max_decode_len=6)
    assert translator.beam_search(torch.tensor([[4]]), beam_size=2, length_penalty=0.6) == [
        TOKEN_A,
        TOKEN_A,
        eos,
    ]


def test_untranslatable_sentences_yield_empty_strings_without_failing_the_batch():
    script = lambda step: {TOKEN_A: -0.1, 3: -1.0}  # noqa: E731
    translator = _translator(script, max_decode_len=2)
    assert translator.translate_batch(["123", "x"]) == ["", "a a"]
    assert translator.translate("456", beam_size=2) == ""


def test_source_is_truncated_to_the_training_limit():
    script = lambda step: {3: -0.1}  # noqa: E731
    translator = _translator(script, max_decode_len=4)
    translator.max_source_len = 2
    encoded = translator._encode_source("x x x x x")
    assert encoded is not None and encoded.shape == (1, 2)
