import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .batching import collate_bag, collate_seq, encode, sep_id
from .data import load_split
from .models import BagModel, RNNModel
from .text import Vocab

CHECKPOINTS = Path("artifacts/checkpoints")


@dataclass
class Config:
    model: str = "bag"  # bag | rnn
    format: str = "score"  # score | concat
    interactions: bool = False  # bag only
    hyp_only: bool = False  # bag only: ignore observations
    cell: str = "lstm"  # rnn only
    bidirectional: bool = False  # rnn only
    hidden: int = 256  # bag: classifier hidden; rnn: RNN hidden
    clf_hidden: int = 256  # rnn only
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 30
    patience: int = 3
    clip: float = 1.0
    seed: int = 0
    embeddings: str = "artifacts/embeddings/glove-cased-split.pt"
    log_every: int = 50
    train_limit: int = 0  # >0: use only the first N train examples (sanity checks)

    def run_name(self) -> str:
        if self.model == "bag":
            arch = f"bag-{'hyponly' if self.hyp_only else self.format}{'-ix' if self.interactions else ''}-h{self.hidden}"
        else:
            arch = f"rnn-{self.format}-{self.cell}{'-bi' if self.bidirectional else ''}-h{self.hidden}"
        extra = "" if self.batch_size == 64 else f"-b{self.batch_size}"
        if Path(self.embeddings).stem != "glove-cased-split":
            extra += f"-{Path(self.embeddings).stem}"
        return f"{arch}-lr{self.lr:g}-d{self.dropout:g}{extra}-s{self.seed}"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_embeddings(path: str) -> tuple[Vocab, torch.Tensor, dict]:
    ckpt = torch.load(path)
    return Vocab(ckpt["itos"][3:]), ckpt["matrix"], ckpt


def build_model(cfg: Config, matrix: torch.Tensor) -> torch.nn.Module:
    if cfg.model == "bag":
        return BagModel(matrix, cfg.format, cfg.hidden, cfg.dropout, cfg.interactions, cfg.hyp_only)
    return RNNModel(matrix, cfg.format, cfg.cell, cfg.hidden, cfg.clf_hidden, cfg.dropout, cfg.bidirectional)


def loader(cfg: Config, vocab: Vocab, emb_meta: dict, split: str, shuffle: bool, limit: int = 0) -> DataLoader:
    data = encode(load_split(split), vocab, emb_meta["lower"], emb_meta["split"])
    data = data[:limit] if limit else data
    collate = collate_bag if cfg.model == "bag" else partial(collate_seq, fmt=cfg.format, sep=sep_id(vocab))
    gen = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(data, cfg.batch_size, shuffle=shuffle, collate_fn=collate, generator=gen)


def to(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev) if k != "lengths" else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, data: DataLoader, dev) -> tuple[float, float, torch.Tensor]:
    """Mean loss, accuracy and predicted probabilities of hypothesis 2."""
    model.eval()
    loss, correct, probs = 0.0, 0, []
    for batch in data:
        batch = to(batch, dev)
        logits = model(**batch)
        loss += F.cross_entropy(logits, batch["label"], reduction="sum").item()
        correct += (logits.argmax(-1) == batch["label"]).sum().item()
        probs.append(logits.softmax(-1)[:, 1].cpu())
    n = len(data.dataset)
    return loss / n, correct / n, torch.cat(probs)


def trainable_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith("emb.")}


def load_checkpoint(path: Path, dev=None) -> tuple[torch.nn.Module, Config, Vocab, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    vocab, matrix, meta = load_embeddings(cfg.embeddings)
    model = build_model(cfg, matrix)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    assert all(k.startswith("emb.") for k in missing) and not unexpected
    return model.to(dev or device()), cfg, vocab, meta


def train(cfg: Config, run=None) -> dict:
    """Train with early stopping on validation loss; keeps the best checkpoint."""
    seed_everything(cfg.seed)
    dev = device()
    vocab, matrix, meta = load_embeddings(cfg.embeddings)
    train_dl, val_dl = loader(cfg, vocab, meta, "train", True, cfg.train_limit), loader(cfg, vocab, meta, "validation", False)
    model = build_model(cfg, matrix).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ckpt_path = CHECKPOINTS / f"{run.name if run else cfg.run_name()}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log = run.log if run else (lambda d: None)
    if run:
        run.summary["trainable_params"] = sum(p.numel() for p in params)

    best = {"val/loss": float("inf")}
    bad_epochs, step = 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        win_loss, win_correct, win_n = 0.0, 0, 0
        for i, batch in enumerate(train_dl):
            batch = to(batch, dev)
            logits = model(**batch)
            loss = F.cross_entropy(logits, batch["label"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            step += 1
            n = len(batch["label"])
            win_loss += loss.item() * n
            win_correct += (logits.argmax(-1) == batch["label"]).sum().item()
            win_n += n
            if step % cfg.log_every == 0 or i == len(train_dl) - 1:
                log({"train/loss": win_loss / win_n, "train/acc": win_correct / win_n,
                     "epoch": epoch + (i + 1) / len(train_dl)})
                win_loss, win_correct, win_n = 0.0, 0, 0
        val_loss, val_acc, _ = evaluate(model, val_dl, dev)
        log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch + 1})
        if val_loss < best["val/loss"]:
            best = {"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch + 1}
            torch.save({"config": asdict(cfg), "state": trainable_state(model), **best}, ckpt_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    if run:
        for k, v in best.items():
            run.summary[f"best/{k}"] = v
        run.summary["checkpoint"] = str(ckpt_path)
    return {**best, "checkpoint": ckpt_path}
