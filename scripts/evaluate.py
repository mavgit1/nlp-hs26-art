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
p.add_argument("--log-wandb", action="store_true", help="write metrics into the matching final/baseline run")
args = p.parse_args()
wandb_runs = {}
if args.log_wandb:
    import os
    import wandb
    wb_path = f"{os.environ.get('WANDB_ENTITY', 'marvin-huber-hochschule-luzern')}/" \
           f"{os.environ.get('WANDB_PROJECT', 'nlp-hs26-art-p1')}"
    wandb_runs = {r.name: r for r in wandb.Api().runs(wb_path) if {"final", "baseline"} & set(r.tags)}

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
    if path.stem in wandb_runs:
        run = wandb_runs[path.stem]
        run.summary[f"{args.split}_eval/loss"], run.summary[f"{args.split}_eval/acc"] = loss, acc
        run.summary.update()
    print(f"{path.stem}\t{args.split}\tloss {loss:.4f}\tacc {acc:.4f}\t-> {out}")
