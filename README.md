# NLP HS26 Project 1: ART with word embeddings and RNNs

Abductive commonsense reasoning (ART): given two observations O1 and O2, pick the
more plausible of two hypotheses. Two models, both on frozen pretrained word embeddings:

1. averaged word embeddings + 2-layer ReLU classifier
2. 2-layer LSTM/GRU + 2-layer ReLU classifier

My expected test accuracies are in [EXPECTATIONS.md](EXPECTATIONS.md), written before training.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

## Reproduce

1. Explore the data: `notebooks/01_exploration.ipynb`
2. Download the pretrained embeddings into `~/.cache/nlp-hs26-art/embeddings`
   (or set `ART_EMBEDDINGS`); URLs are in `src/art_nlp/embeddings.py`.
3. Compare embedding coverage: `uv run python scripts/embedding_coverage.py`
   (writes `results/embedding_coverage.md`)
4. Cache the chosen embedding matrix, e.g.
   `uv run python scripts/build_embeddings.py --embedding fasttext --no-split`

## Sources

- Dataset: [amarfurt/art-nlp-hs26](https://huggingface.co/datasets/amarfurt/art-nlp-hs26),
  a deduplicated fork of [allenai/art](https://huggingface.co/datasets/allenai/art).
  Bhagavatula et al. (2020), *Abductive Commonsense Reasoning*, ICLR.
- fastText: Mikolov et al. (2018), *Advances in Pre-Training Distributed Word Representations*.
- GloVe: Pennington et al. (2014), *GloVe: Global Vectors for Word Representation*.
- word2vec: Mikolov et al. (2013), *Distributed Representations of Words and Phrases*.

## Tools

Code written with the help of Claude Code (Anthropic). I reviewed and understand all code.
