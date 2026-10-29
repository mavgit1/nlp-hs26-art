from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import Example
from .text import SEP, Vocab, tokenize

FORMATS = ("score", "concat")


@dataclass(frozen=True)
class Encoded:
    obs1: list[int]
    obs2: list[int]
    hyps: tuple[list[int], list[int]]
    label: int


def encode(examples: list[Example], vocab: Vocab, lower: bool = False, split: bool = True) -> list[Encoded]:
    enc = lambda t: vocab.encode(tokenize(t, lower, split))
    return [Encoded(enc(e.obs1), enc(e.obs2), (enc(e.hyps[0]), enc(e.hyps[1])), e.label) for e in examples]


def _pad(seqs: list[list[int]]) -> torch.Tensor:
    return pad_sequence([torch.tensor(s) for s in seqs], batch_first=True)


def collate_bag(batch: list[Encoded]) -> dict[str, torch.Tensor]:
    """Separate padded segments: obs1, obs2 [B, L]; hyps [B, 2, L]."""
    hyps = _pad([h for b in batch for h in b.hyps])
    return {"obs1": _pad([b.obs1 for b in batch]), "obs2": _pad([b.obs2 for b in batch]),
            "hyps": hyps.view(len(batch), 2, -1), "label": torch.tensor([b.label for b in batch])}


def story(b: Encoded, fmt: str, sep: int) -> list[list[int]]:
    """Token sequences in narrative order: one per hypothesis, or one with both."""
    if fmt == "score":
        return [b.obs1 + [sep] + h + [sep] + b.obs2 for h in b.hyps]
    return [b.obs1 + [sep] + b.hyps[0] + [sep] + b.hyps[1] + [sep] + b.obs2]


def collate_seq(batch: list[Encoded], fmt: str, sep: int) -> dict[str, torch.Tensor]:
    """tokens [B, K, L] and lengths [B, K]; K = 2 for score, 1 for concat."""
    seqs = [s for b in batch for s in story(b, fmt, sep)]
    k = len(seqs) // len(batch)
    return {"tokens": _pad(seqs).view(len(batch), k, -1),
            "lengths": torch.tensor([len(s) for s in seqs]).view(len(batch), k),
            "label": torch.tensor([b.label for b in batch])}


def sep_id(vocab: Vocab) -> int:
    return vocab.stoi[SEP]
