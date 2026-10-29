"""Train one model; every Config field is a --flag (also used by W&B sweeps)."""
import argparse
import os
from dataclasses import asdict, fields

import wandb
from dotenv import load_dotenv

from art_nlp.training import Config, train

load_dotenv()
p = argparse.ArgumentParser()
for f in fields(Config):
    kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
    p.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
p.add_argument("--no-wandb", action="store_true")
p.add_argument("--tags", nargs="*", default=[])
args = vars(p.parse_args())
no_wandb, tags = args.pop("no_wandb"), args.pop("tags")
cfg = Config(**args)

if no_wandb:
    print(train(cfg))
else:
    with wandb.init(entity=os.environ.get("WANDB_ENTITY", "marvin-huber-hochschule-luzern"),
                    project=os.environ.get("WANDB_PROJECT", "nlp-hs26-art-p1"),
                    name=cfg.run_name(), group=f"{cfg.model}-{cfg.format}", tags=tags, config=asdict(cfg)) as run:
        # sweeps may override config values
        cfg = Config(**{**asdict(cfg), **dict(run.config)})
        run.name = cfg.run_name()
        print(train(cfg, run))
