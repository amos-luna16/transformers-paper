import torch

from transformer_nmt.config import EOS_INDEX, PAD_INDEX, SOS_INDEX
from transformer_nmt.data.dataset import (
    PadCollator,
    TranslationDataset,
    build_dataloader,
    deduplicate_pairs,
    load_sentence_pairs,
    normalize_and_filter_pairs,
    split_pairs,
)
from transformer_nmt.data.vocabulary import Vocabulary


def test_load_sentence_pairs_reads_first_two_columns(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("Go.\tVa !\tCC-BY\nHi.\tSalut !\nbroken line\n", encoding="utf-8")
    assert load_sentence_pairs(path, max_pairs=10) == [("Go.", "Va !"), ("Hi.", "Salut !")]
    assert len(load_sentence_pairs(path, max_pairs=1)) == 1


def test_normalize_and_filter_pairs_respects_length_budget(tokenizer):
    pairs = [("Go.", "Va !"), ("one two three four five", "un deux trois"), ("", "vide")]
    kept = normalize_and_filter_pairs(pairs, tokenizer, max_seq_len=4)
    assert kept == [("go .", "va !")]


def test_normalize_and_filter_pairs_drops_zero_token_sides(tokenizer):
    assert normalize_and_filter_pairs([("'", "oui ."), ("5 × 3", "quinze .")], tokenizer, 40) == []


def test_deduplicate_pairs_keeps_first_occurrence():
    assert deduplicate_pairs([("a", "b"), ("c", "d"), ("a", "b")]) == [("a", "b"), ("c", "d")]


def test_split_pairs_keeps_all_translations_of_a_source_on_one_side():
    pairs = [(f"src{i}", f"tgt{i}-{j}") for i in range(50) for j in range(3)]
    train, validation = split_pairs(pairs, 0.2, seed=3)
    assert len(train) + len(validation) == len(pairs) and len(validation) == 30
    assert not {src for src, _ in train} & {src for src, _ in validation}


def test_split_pairs_is_deterministic_and_disjoint():
    pairs = [(str(i), str(i)) for i in range(100)]
    train_a, val_a = split_pairs(pairs, 0.2, seed=1)
    train_b, val_b = split_pairs(pairs, 0.2, seed=1)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 20 and len(train_a) == 80
    assert not set(train_a) & set(val_a)


def test_dataset_wraps_target_with_specials_and_truncates(tokenizer):
    pairs = [("hello there", "bonjour à tous mes amis")]
    src_vocab = Vocabulary.build([p[0] for p in pairs], tokenizer)
    tgt_vocab = Vocabulary.build([p[1] for p in pairs], tokenizer)
    dataset = TranslationDataset(pairs, tokenizer, src_vocab, tgt_vocab, max_seq_len=4)
    source, target = dataset[0]
    assert source.tolist() == src_vocab.encode(["hello", "there"])
    assert target[0].item() == SOS_INDEX and target[-1].item() == EOS_INDEX and target.numel() == 4


def test_collator_pads_to_multiple(tokenizer):
    collator = PadCollator(PAD_INDEX, pad_to_multiple_of=8)
    source, target = collator(
        [(torch.tensor([5, 6]), torch.tensor([2, 7, 3])), (torch.tensor([9]), torch.tensor([2, 3]))]
    )
    assert source.shape == (2, 8) and target.shape == (2, 8)
    assert source[1].tolist() == [9] + [PAD_INDEX] * 7


def test_build_dataloader_yields_padded_batches(tiny_bundle):
    loader, sampler = build_dataloader(
        tiny_bundle.train_dataset,
        batch_size=4,
        shuffle=True,
        collator=PadCollator(PAD_INDEX, 4),
        num_workers=0,
        distributed=False,
        seed=0,
        pin_memory=False,
    )
    assert sampler is None
    source, target = next(iter(loader))
    assert source.shape[0] == 4 and target.shape[1] % 4 == 0
