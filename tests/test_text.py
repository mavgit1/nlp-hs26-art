from art_nlp.text import PAD, SEP, UNK, Vocab, tokenize


def test_tokenize_splits_punctuation_and_contractions():
    assert tokenize("He didn't go, she's late.") == ["He", "did", "n't", "go", ",", "she", "'s", "late", "."]


def test_tokenize_keeps_hyphens_and_curly_apostrophes():
    assert tokenize("A well-known man’s dog") == ["A", "well-known", "man", "'s", "dog"]


def test_tokenize_lower_and_no_split():
    assert tokenize("Don't Stop", lower=True, split_contractions=False) == ["don't", "stop"]
    assert tokenize("a,b (c)", split_contractions=False) == ["a", ",", "b", "(", "c", ")"]


def test_vocab_specials_first_and_unk():
    v = Vocab(["the", "cat", "the"])
    assert v.itos[:3] == [PAD, UNK, SEP] and len(v) == 5
    assert v.encode(["the", "dog"]) == [3, v.stoi[UNK]]


def test_vocab_build_min_freq():
    v = Vocab.build([["a", "b", "a"], ["a", "c"]], min_freq=2)
    assert "a" in v and "b" not in v
