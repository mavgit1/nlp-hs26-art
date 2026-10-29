import re
from collections import Counter
from collections.abc import Iterable

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
SPECIALS = (PAD, UNK, SEP)

_CONTRACTION = re.compile(r"(?i)(\w)(n't|'s|'re|'ve|'ll|'m|'d)\b")
_TOKEN = re.compile(r"n't|'\w+|\w+(?:[-.]\w+)*|[^\w\s]", re.IGNORECASE)


def tokenize(text: str, lower: bool = False, split_contractions: bool = True) -> list[str]:
    text = text.replace("’", "'")
    if split_contractions:
        text = _CONTRACTION.sub(r"\1 \2", text)
        tokens = _TOKEN.findall(text)
    else:
        tokens = re.findall(r"\w+(?:['.\-]\w+)*|[^\w\s]", text)
    return [t.lower() for t in tokens] if lower else tokens


class Vocab:
    def __init__(self, words: Iterable[str]):
        self.itos = list(SPECIALS) + [w for w in dict.fromkeys(words) if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    @classmethod
    def build(cls, corpus: Iterable[list[str]], min_freq: int = 1) -> "Vocab":
        counts = Counter(t for tokens in corpus for t in tokens)
        return cls(w for w, c in counts.most_common() if c >= min_freq)
