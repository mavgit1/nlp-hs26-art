import numpy as np

from art_nlp.embeddings import DIM, build_matrix
from art_nlp.text import PAD, SEP, UNK


def test_build_matrix_rows_and_specials():
    vectors = {"cat": np.ones(DIM, np.float32), "dog": np.full(DIM, 2, np.float32)}
    vocab, m = build_matrix(vectors)
    assert m.shape == (5, DIM)
    assert m[vocab.stoi["dog"]].eq(2).all()
    assert m[vocab.stoi[PAD]].eq(0).all() and m[vocab.stoi[UNK]].eq(0).all()
    assert m[vocab.stoi[SEP]].norm() > 0
