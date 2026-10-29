"""Which original allenai/art split each course split comes from (exact pair match)."""
import csv
from pathlib import Path

from datasets import load_dataset

from art_nlp.data import DATASET, SPLITS

key = lambda d: set(zip(d["observation_1"], d["observation_2"], d["hypothesis_1"], d["hypothesis_2"]))
orig = {s: key(load_dataset("allenai/art", split=s)) for s in ("train", "validation")}

out = Path("results/split_origin.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["course_split", "n", "in_original_train", "in_original_validation"])
    for s in SPLITS:
        pairs = key(load_dataset(DATASET, split=s))
        w.writerow([s, len(pairs), *(f"{len(pairs & orig[o]) / len(pairs):.4f}" for o in orig)])
print(out.read_text())
