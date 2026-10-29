from dataclasses import dataclass

from datasets import load_dataset

DATASET = "amarfurt/art-nlp-hs26"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Example:
    obs1: str
    obs2: str
    hyps: tuple[str, str]
    label: int  # index of the plausible hypothesis: 0 or 1


def load_split(split: str) -> list[Example]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [
        Example(r["observation_1"], r["observation_2"],
                (r["hypothesis_1"], r["hypothesis_2"]), int(r["label"]) - 1)
        for r in load_dataset(DATASET, split=split)
    ]
