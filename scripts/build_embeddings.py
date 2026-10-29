"""Cache the frozen embedding matrix for every token in the dataset texts.

Embeddings are frozen, so restricting the 2M+ pretrained rows to the dataset's
tokens is equivalent to using the full table. Only text is read, no labels.
"""
import argparse
from pathlib import Path

import torch

from art_nlp.data import SPLITS, load_split
from art_nlp.embeddings import SOURCES, build_matrix, load_vectors
from art_nlp.text import tokenize

p = argparse.ArgumentParser()
p.add_argument("--embedding", choices=SOURCES, required=True)
p.add_argument("--lower", action="store_true")
p.add_argument("--no-split", dest="split", action="store_false", help="keep contractions whole")
args = p.parse_args()

tokens = {t for s in SPLITS for e in load_split(s) for text in (e.obs1, e.obs2, *e.hyps)
          for t in tokenize(text, args.lower, args.split)}
found = load_vectors(args.embedding, tokens | {t.lower() for t in tokens})
vectors = {t: found.get(t, found.get(t.lower())) for t in sorted(tokens)}
vocab, matrix = build_matrix({t: v for t, v in vectors.items() if v is not None})

name = f"{args.embedding}-{'lower' if args.lower else 'cased'}-{'split' if args.split else 'nosplit'}"
out = Path("artifacts/embeddings") / f"{name}.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"itos": vocab.itos, "matrix": matrix, "lower": args.lower, "split": args.split}, out)
print(f"{out}: {len(vocab)} rows, {len(tokens) - len(vocab) + 3} dataset tokens map to <unk>")
