# NLP HS26 Project 1: ART with word embeddings and RNNs

Abductive commonsense reasoning (ART): given two observations O1 and O2, pick the
more plausible of two hypotheses. Two models, both on frozen GloVe 840B embeddings:

1. averaged word embeddings + 2-layer ReLU classifier
2. 2-layer bidirectional GRU + 2-layer ReLU classifier

My expected test accuracies are in [EXPECTATIONS.md](EXPECTATIONS.md), written before training.

## Results

Mean ± std over 3 seeds. Test was evaluated once, after all choices were made on validation.

| Model | Validation acc | Test acc | Expected |
|---|---|---|---|
| 1: averaged embeddings, score per hypothesis, interaction features | 73.6 ± 0.7 | **53.5 ± 0.4** | 53 |
| 2: 2-layer bi-GRU over O1 `<sep>` H `<sep>` O2 | 74.0 ± 0.8 | **54.5 ± 0.6** | 57 |
| baseline: averaged embeddings of the hypothesis only | 69.3 ± 0.2 | 52.2 ± 0.4 | – |

- Validation pairs are minimal edits (median 4 differing words), test pairs are not
  (median 13); validation accuracy does not transfer to test.
- The hypothesis-only baseline reaches 69% on validation: most of the signal is
  in hypothesis artifacts, not in reasoning about O1 and O2.
- Model 2 beats model 1 on negation and outcome edits, but not on entity swaps
  (`notebooks/02_results.ipynb`).

Experiment tracking (W&B): saved views *Project 1: final models* and *Project 1: sweeps*
in `marvin-huber-hochschule-luzern/nlp-hs26-art-p1`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

W&B logging needs `wandb login`. The optional edit-type labelling needs
`ANTHROPIC_API_KEY` in `.env`.

## Reproduce

1. Explore the data: `notebooks/01_exploration.ipynb`
2. Download the pretrained embeddings into `~/.cache/nlp-hs26-art/embeddings`
   (or set `ART_EMBEDDINGS`); URLs are in `src/art_nlp/embeddings.py`.
3. Compare embedding coverage: `uv run python scripts/embedding_coverage.py`
   (writes `results/embedding_coverage.csv`)
4. Cache the chosen embedding matrix (GloVe 840B, cased, contractions split:
   best token coverage, keeps `n't` and punctuation):
   `uv run python scripts/build_embeddings.py --embedding glove`
5. Sweeps: `uv run wandb sweep sweeps/<name>.yaml`, then `uv run wandb agent <id>`
   (`bag_score`, `bag_concat`, `rnn_base`, `rnn_tune`)
6. Final runs, 3 seeds each: `scripts/final_runs.sh`
7. Evaluate (test only once):
   `uv run python scripts/evaluate.py artifacts/checkpoints/<run>.pt --split test`
8. Export runs and build the W&B views: `scripts/collect_runs.py`, `scripts/workspace_views.py`
9. Label validation edit types (Claude API): `scripts/label_edit_types.py`
10. Results and error analysis: `notebooks/02_results.ipynb`

Training: AdamW (weight decay 0.01), cross-entropy over the two hypothesis scores,
gradient clipping at 1.0, early stopping on validation loss (patience 3), best
checkpoint kept (trainable weights only).

## Code

- `src/art_nlp/data.py`: dataset loading (labels as 0/1)
- `src/art_nlp/text.py`: tokenizer, vocabulary
- `src/art_nlp/embeddings.py`: pretrained vector loading, frozen embedding matrix
- `src/art_nlp/batching.py`: padding and story sequences
- `src/art_nlp/models.py`: both models
- `src/art_nlp/training.py`: config, training loop, evaluation, checkpoints

## Sources

- Dataset: [amarfurt/art-nlp-hs26](https://huggingface.co/datasets/amarfurt/art-nlp-hs26),
  a deduplicated fork of [allenai/art](https://huggingface.co/datasets/allenai/art).
  Bhagavatula et al. (2020), *Abductive Commonsense Reasoning*, ICLR.
- GloVe: Pennington et al. (2014), *GloVe: Global Vectors for Word Representation*
  (used); fastText (Mikolov et al., 2018) and word2vec (Mikolov et al., 2013) were
  only compared for coverage.

## Tools

Code written with the help of Claude Code (Anthropic). Validation edit types were
labelled with the Claude API (Claude Opus 5.5), for error analysis only.
