import gzip
import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .text import PAD, SEP, UNK, Vocab

CACHE = Path(os.environ.get("ART_EMBEDDINGS", Path.home() / ".cache/nlp-hs26-art/embeddings"))
DIM = 300

SOURCES = {
    "fasttext": ("crawl-300d-2M.vec.zip", "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"),
    "glove": ("glove.840B.300d.zip", "https://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip"),
    "word2vec": ("word2vec-google-news-300.gz",
                 "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
                 "word2vec-google-news-300/word2vec-google-news-300.gz"),
}


def _text_lines(path: Path) -> Iterator[str]:
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        yield from io.TextIOWrapper(f, encoding="utf-8", errors="replace")


def _iter_text(path: Path, wanted: set[str]) -> Iterator[tuple[str, np.ndarray]]:
    for line in _text_lines(path):
        parts = line.rstrip().split(" ")
        if len(parts) <= DIM:  # fastText header line
            continue
        word = " ".join(parts[:-DIM])  # GloVe has a few tokens with spaces
        if word in wanted:
            yield word, np.asarray(parts[-DIM:], dtype=np.float32)


def _iter_word2vec(path: Path, wanted: set[str]) -> Iterator[tuple[str, np.ndarray]]:
    with gzip.open(path, "rb") as f:
        count, dim = map(int, f.readline().split())
        for _ in range(count):
            word = bytearray()
            while (c := f.read(1)) != b" ":
                if c != b"\n":
                    word += c
            vec = f.read(4 * dim)
            w = word.decode("utf-8", errors="replace")
            if w in wanted:
                yield w, np.frombuffer(vec, dtype=np.float32).copy()


def load_vectors(name: str, wanted: set[str]) -> dict[str, np.ndarray]:
    """Stream a pretrained file, keeping only the wanted words."""
    path = CACHE / SOURCES[name][0]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download from {SOURCES[name][1]}")
    reader = _iter_word2vec if name == "word2vec" else _iter_text
    vectors = {}
    for w, v in reader(path, wanted):
        vectors.setdefault(w, v)  # first entry wins, as in the original files
    return vectors


def build_matrix(vectors: dict[str, np.ndarray], seed: int = 0) -> tuple[Vocab, torch.Tensor]:
    """Vocab over the given vectors; PAD and UNK are zero, SEP a fixed random vector."""
    vocab = Vocab(vectors)
    matrix = torch.zeros(len(vocab), DIM)
    for i, w in enumerate(vocab.itos):
        if w in vectors:
            matrix[i] = torch.from_numpy(vectors[w])
    norm = matrix[3:].norm(dim=1).mean()
    sep = torch.randn(DIM, generator=torch.Generator().manual_seed(seed))
    matrix[vocab.stoi[SEP]] = sep / sep.norm() * norm
    return vocab, matrix
