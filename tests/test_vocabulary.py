import pytest

from transformer_nmt.config import EOS_INDEX, PAD_INDEX, SOS_INDEX, SPECIAL_TOKENS, UNK_INDEX
from transformer_nmt.data.vocabulary import Vocabulary


def test_build_orders_special_tokens_first_and_applies_min_freq(
    tokenizer,
):
    vocab = Vocabulary.build(["a a b .", "a c ."], tokenizer, min_freq=2)
    assert vocab.tokens[:4] == list(SPECIAL_TOKENS)
    assert "a" in vocab and "." in vocab
    assert "b" not in vocab and "c" not in vocab


def test_build_is_deterministic_by_frequency_then_alphabet(
    tokenizer,
):
    vocab = Vocabulary.build(["z y x", "z y", "z"], tokenizer)
    assert vocab.tokens[4:] == ["z", "y", "x"]


def test_encode_decode_roundtrip_with_specials(
    tokenizer,
):
    vocab = Vocabulary.build(["je suis là ."], tokenizer)
    ids = vocab.encode(["je", "suis", "là", "."], add_special_tokens=True)
    assert ids[0] == SOS_INDEX and ids[-1] == EOS_INDEX
    assert vocab.decode(ids) == ["je", "suis", "là", "."]
    assert vocab.decode([*ids, PAD_INDEX, PAD_INDEX]) == ["je", "suis", "là", "."]


def test_unknown_tokens_map_to_unk(
    tokenizer,
):
    vocab = Vocabulary.build(["bonjour"], tokenizer)
    assert vocab.token_to_index("inconnu") == UNK_INDEX


def test_save_and_load(
    tmp_path,
    tokenizer,
):
    vocab = Vocabulary.build(["le chat dort ."], tokenizer)
    vocab.save(tmp_path / "v.json")
    restored = Vocabulary.load(tmp_path / "v.json")
    assert restored.tokens == vocab.tokens


def test_rejects_invalid_token_lists():
    with pytest.raises(ValueError):
        Vocabulary(["a", "b"])
    with pytest.raises(ValueError):
        Vocabulary([*SPECIAL_TOKENS, "x", "x"])
