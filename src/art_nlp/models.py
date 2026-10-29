import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

FIRST_WORD_ID = 3  # ids 0-2 are <pad>, <unk>, <sep>


class Classifier(nn.Sequential):
    """2-layer classifier with ReLU; no dropout after the output."""

    def __init__(self, d_in: int, hidden: int, d_out: int, dropout: float):
        super().__init__(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


def frozen_embedding(matrix: torch.Tensor) -> nn.Embedding:
    return nn.Embedding.from_pretrained(matrix, freeze=True, padding_idx=0)


class BagModel(nn.Module):
    """Model 1: averaged frozen word embeddings per text part + classifier.

    score: one score per hypothesis from [o1, o2, h] (+ h*o1, h*o2), softmax over both.
    concat: [o1, o2, h1, h2] -> 2 logits.
    """

    def __init__(self, matrix: torch.Tensor, fmt: str, hidden: int, dropout: float, interactions: bool = False):
        super().__init__()
        self.fmt, self.interactions = fmt, interactions
        self.emb = frozen_embedding(matrix)
        d = matrix.shape[1]
        d_in = (5 if interactions else 3) * d if fmt == "score" else 4 * d
        self.clf = Classifier(d_in, hidden, 1 if fmt == "score" else 2, dropout)

    def mean(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids >= FIRST_WORD_ID).unsqueeze(-1).float()  # skip <pad>, <unk>
        return (self.emb(ids) * mask).sum(-2) / mask.sum(-2).clamp(min=1)

    def forward(self, obs1, obs2, hyps, **_) -> torch.Tensor:
        o1, o2, h = self.mean(obs1), self.mean(obs2), self.mean(hyps)
        if self.fmt == "concat":
            return self.clf(torch.cat([o1, o2, h[:, 0], h[:, 1]], -1))
        o1, o2 = o1.unsqueeze(1).expand_as(h), o2.unsqueeze(1).expand_as(h)
        parts = [o1, o2, h] + ([h * o1, h * o2] if self.interactions else [])
        return self.clf(torch.cat(parts, -1)).squeeze(-1)


class RNNModel(nn.Module):
    """Model 2: frozen embeddings -> 2-layer LSTM/GRU over the story -> classifier on final hidden state."""

    def __init__(self, matrix: torch.Tensor, fmt: str, cell: str, hidden: int, clf_hidden: int,
                 dropout: float, bidirectional: bool = False, layers: int = 2):
        super().__init__()
        self.fmt, self.layers, self.dirs = fmt, layers, 2 if bidirectional else 1
        self.emb = frozen_embedding(matrix)
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(matrix.shape[1], hidden, num_layers=layers, batch_first=True,
                       dropout=dropout, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.clf = Classifier(hidden * self.dirs, clf_hidden, 1 if fmt == "score" else 2, dropout)

    def forward(self, tokens, lengths, **_) -> torch.Tensor:
        b, k, l = tokens.shape
        packed = pack_padded_sequence(self.emb(tokens.view(b * k, l)), lengths.view(-1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h  # LSTM returns (h_n, c_n)
        last = h.view(self.layers, self.dirs, b * k, -1)[-1]  # top layer, all directions
        logits = self.clf(self.drop(last.transpose(0, 1).reshape(b * k, -1)))
        return logits.view(b, k) if self.fmt == "score" else logits.view(b, 2)
