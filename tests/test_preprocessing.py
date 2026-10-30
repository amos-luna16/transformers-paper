from transformer_nmt.data.preprocessing import WordTokenizer, normalize_text


def test_normalize_keeps_accents_and_apostrophes():
    assert normalize_text("C'est très  Facile!") == "c'est très facile !"


def test_normalize_separates_punctuation_and_drops_symbols():
    assert normalize_text("Hello, world... 123?") == "hello world . . . ?"


def test_normalize_handles_non_breaking_space():
    assert normalize_text("Salut !") == "salut !"


def test_tokenizer_keeps_contractions():
    assert WordTokenizer()("I can't believe it's working!") == [
        "i",
        "can't",
        "believe",
        "it's",
        "working",
        "!",
    ]


def test_detokenize_reattaches_punctuation():
    assert WordTokenizer().detokenize(["je", "suis", "là", "."]) == "je suis là."
