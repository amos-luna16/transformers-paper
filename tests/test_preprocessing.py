from transformer_nmt.data.preprocessing import WordTokenizer, normalize_text


def test_normalize_keeps_accents_and_apostrophes():
    assert normalize_text("C'est très  Facile!") == "c'est très facile !"


def test_normalize_separates_punctuation_and_drops_symbols():
    assert normalize_text("Hello, world... 123?") == "hello world . . . ?"


def test_normalize_handles_non_breaking_space():
    assert normalize_text("Salut\u202f!") == "salut !"


def test_normalize_keeps_oe_ligature():
    assert normalize_text("Ma sœur a un cœur d'or.") == "ma sœur a un cœur d'or ."
    assert WordTokenizer()("ma sœur a un cœur d'or .") == [
        "ma",
        "sœur",
        "a",
        "un",
        "cœur",
        "d'or",
        ".",
    ]


def test_normalize_maps_typographic_apostrophes():
    assert normalize_text("J\u2019ai faim.") == normalize_text("J'ai faim.") == "j'ai faim ."
    assert WordTokenizer()(normalize_text("I don\u2019t know")) == ["i", "don't", "know"]


def test_normalize_drops_multiplication_and_division_signs():
    assert normalize_text("3 × 4 ÷ 2") == ""


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
