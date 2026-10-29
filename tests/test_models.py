import pytest
import torch

from art_nlp.models import BagModel, RNNModel

torch.manual_seed(0)
MATRIX = torch.randn(20, 8)
MATRIX[:2] = 0


def bag_batch():
    ids = lambda *s: torch.randint(3, 20, s)
    return {"obs1": ids(4, 5), "obs2": ids(4, 6), "hyps": ids(4, 2, 7)}


def seq_batch(k):
    return {"tokens": torch.randint(3, 20, (4, k, 9)), "lengths": torch.randint(3, 10, (4, k))}


def swap(batch, key):
    return {**batch, key: batch[key].flip(1)}


@pytest.mark.parametrize("fmt,interactions", [("score", False), ("score", True), ("concat", False)])
def test_bag_shapes_and_frozen(fmt, interactions):
    m = BagModel(MATRIX, fmt, hidden=16, dropout=0.0, interactions=interactions)
    assert m(**bag_batch()).shape == (4, 2)
    assert not m.emb.weight.requires_grad


def test_bag_hyp_only_ignores_observations():
    m, b = BagModel(MATRIX, "score", 16, 0.0, hyp_only=True).eval(), bag_batch()
    assert torch.allclose(m(**b), m(**{**b, "obs1": b["obs1"].flip(0), "obs2": b["obs2"].flip(0)}))


def test_bag_score_swapping_hypotheses_swaps_scores():
    m, b = BagModel(MATRIX, "score", 16, 0.0, interactions=True).eval(), bag_batch()
    assert torch.allclose(m(**b).flip(1), m(**swap(b, "hyps")))


def test_bag_mean_ignores_padding():
    m, b = BagModel(MATRIX, "score", 16, 0.0).eval(), bag_batch()
    padded = {**b, "obs1": torch.cat([b["obs1"], torch.zeros(4, 3, dtype=torch.long)], 1)}
    assert torch.allclose(m(**b), m(**padded))


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("fmt,k", [("score", 2), ("concat", 1)])
@pytest.mark.parametrize("bi", [False, True])
def test_rnn_shapes(cell, fmt, k, bi):
    m = RNNModel(MATRIX, fmt, cell, hidden=6, clf_hidden=5, dropout=0.0, bidirectional=bi)
    assert m(**seq_batch(k)).shape == (4, 2)


def test_rnn_ignores_padding_and_is_swap_equivariant():
    m, b = RNNModel(MATRIX, "score", "lstm", 6, 5, 0.0).eval(), seq_batch(2)
    padded = {**b, "tokens": torch.cat([b["tokens"], torch.zeros(4, 2, 4, dtype=torch.long)], 2)}
    assert torch.allclose(m(**b), m(**padded), atol=1e-6)
    assert torch.allclose(m(**b).flip(1), m(**swap(swap(b, "tokens"), "lengths")), atol=1e-6)
