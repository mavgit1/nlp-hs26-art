"""Evaluate checkpoints on a split and save per-example predictions.

Run on test only once, for the final models.
"""
import argparse
import csv
from pathlib import Path

from art_nlp.training import device, evaluate, load_checkpoint, loader

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+", type=Path)
p.add_argument("--split", default="validation", choices=["validation", "test"])
args = p.parse_args()

out_dir = Path("results/predictions")
out_dir.mkdir(parents=True, exist_ok=True)
for path in args.checkpoints:
    model, cfg, vocab, meta = load_checkpoint(path)
    data = loader(cfg, vocab, meta, args.split, shuffle=False)
    loss, acc, probs = evaluate(model, data, device())
    out = out_dir / f"{path.stem}-{args.split}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "prob_h2", "pred"])
        for i, (ex, pr) in enumerate(zip(data.dataset, probs.tolist())):
            w.writerow([i, ex.label, f"{pr:.4f}", int(pr > 0.5)])
    print(f"{path.stem}\t{args.split}\tloss {loss:.4f}\tacc {acc:.4f}\t-> {out}")
