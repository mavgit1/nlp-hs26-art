from art_nlp.batching import collate_bag, collate_seq, encode, sep_id
from art_nlp.data import Example
from art_nlp.text import Vocab

EXAMPLES = [Example("I ran.", "I won.", ("I trained hard.", "I slept."), 0),
            Example("She cooked.", "It was great.", ("It burned.", "She used a recipe."), 1)]
VOCAB = Vocab(w for e in EXAMPLES for t in (e.obs1, e.obs2, *e.hyps) for w in t.replace(".", " .").split())
BATCH = encode(EXAMPLES, VOCAB)


def test_collate_bag_shapes_and_padding():
    b = collate_bag(BATCH)
    assert b["hyps"].shape == (2, 2, 5) and b["obs1"].shape == (2, 3)
    assert b["hyps"][0, 1].tolist() == VOCAB.encode(["I", "slept", "."]) + [0, 0]
    assert b["label"].tolist() == [0, 1]


def test_collate_seq_score_builds_story_per_hypothesis():
    b = collate_seq(BATCH, "score", sep_id(VOCAB))
    assert b["tokens"].shape[:2] == (2, 2)
    assert b["lengths"][0].tolist() == [3 + 1 + 4 + 1 + 3, 3 + 1 + 3 + 1 + 3]
    first = b["tokens"][0, 0, : b["lengths"][0, 0]].tolist()
    assert first == VOCAB.encode("I ran . <sep> I trained hard . <sep> I won .".split())


def test_collate_seq_concat_has_both_hypotheses():
    b = collate_seq(BATCH, "concat", sep_id(VOCAB))
    assert b["tokens"].shape[1] == 1 and b["lengths"][0, 0] == 3 + 4 + 3 + 3 + 3
