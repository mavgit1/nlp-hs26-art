# Expectations

Written on 29.10.2026, before any model was trained.
Metric: accuracy on the test split (1532 examples, about 51% label 1).

| Model | Expected best test accuracy | Plausible range |
|---|---|---|
| 1: frozen word embeddings + 2-layer ReLU classifier | **53%** | 50–56% |
| 2: frozen word embeddings + 2-layer LSTM/GRU + 2-layer ReLU classifier | **57%** | 52–62% |

## Reasoning

- In train and validation, h1 and h2 differ in only about 4 words (median).
  Averaged embeddings of the two hypotheses are therefore almost identical,
  so model 1 sees little signal. Close to chance, with a small gain from lexical cues.
- The pairs were filtered so that BERT could not solve them, and the task needs
  commonsense about what happens between O1 and O2. Neither model has that knowledge.
- Model 2 sees word order and can relate the hypothesis to the observations
  (e.g. negation, who does what). The ART paper reports ESIM+ELMo at around 58%,
  a stronger RNN baseline than mine, so I expect to stay at or below that.
- Test hypothesis pairs differ more (about 13 words) than in validation, so
  validation accuracy may over- or underestimate test accuracy.
