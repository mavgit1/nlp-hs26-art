"""Export all finished W&B runs (config + best validation metrics) to results/runs.csv."""
import os

import pandas as pd
import wandb

path = f"{os.environ.get('WANDB_ENTITY', 'marvin-huber-hochschule-luzern')}/" \
       f"{os.environ.get('WANDB_PROJECT', 'nlp-hs26-art-p1')}"
rows = []
for r in wandb.Api().runs(path):
    if r.state != "finished":
        continue
    cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
    rows.append({"name": r.name, "id": r.id, "sweep": r.sweep.name if r.sweep else "", "tags": " ".join(r.tags),
                 **cfg, **{k: r.summary.get(k) for k in ("best/val/loss", "best/val/acc", "best/epoch",
                                                          "trainable_params", "test/acc")}})
df = pd.DataFrame(rows).sort_values("best/val/loss")
os.makedirs("results", exist_ok=True)
df.to_csv("results/runs.csv", index=False)
print(df[["name", "sweep", "best/val/loss", "best/val/acc", "best/epoch"]].to_string(index=False))
