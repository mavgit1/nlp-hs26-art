"""Coverage of the train text by each pretrained embedding and tokenizer variant."""
from collections import Counter
from itertools import product
from pathlib import Path

from art_nlp.data import load_split
from art_nlp.embeddings import SOURCES, load_vectors
from art_nlp.text import tokenize

VARIANTS = {f"{'lower' if lo else 'cased'}/{'split' if sp else 'nosplit'}": (lo, sp)
            for lo, sp in product((False, True), (True, False))}

texts = [t for e in load_split("train") for t in (e.obs1, e.obs2, *e.hyps)]
counts = {name: Counter(t for s in texts for t in tokenize(s, *opts)) for name, opts in VARIANTS.items()}
wanted = set().union(*counts.values())
wanted |= {w.lower() for w in wanted}

rows, oov = [], {}
for emb in SOURCES:
    vectors = load_vectors(emb, wanted)
    for name, c in counts.items():
        hit = {w for w in c if w in vectors}
        fallback = {w for w in c if w not in vectors and w.lower() in vectors}
        n = sum(c.values())
        rows.append((emb, name, len(hit) / len(c), sum(c[w] for w in hit) / n,
                     sum(c[w] for w in hit | fallback) / n))
        oov[emb, name] = [(w, k) for w, k in c.most_common() if w not in vectors][:15]

lines = ["# Embedding coverage on train", "",
         "| embedding | tokenizer | type coverage | token coverage | token coverage + lowercase fallback |",
         "|---|---|---|---|---|"]
lines += [f"| {e} | {v} | {a:.1%} | {b:.1%} | {c:.1%} |" for e, v, a, b, c in rows]
lines += ["", "## Most frequent missing tokens", ""]
lines += [f"- **{e} {v}**: " + ", ".join(f"`{w}` ({k})" for w, k in ws) for (e, v), ws in oov.items()]
out = Path("results/embedding_coverage.md")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
