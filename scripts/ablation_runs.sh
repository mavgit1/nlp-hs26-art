#!/usr/bin/env bash
# One-factor ablations around the final configs (seed 0 unless noted).
set -euo pipefail
cd "$(dirname "$0")/.."
BAG=(--model bag --format score --interactions true --hidden 512 --lr 0.001)
RNN=(--model rnn --format score --cell gru --bidirectional true --dropout 0.5 --lr 0.001)
FT=artifacts/embeddings/fasttext-cased-nosplit.pt
run() { uv run python scripts/train.py --tags sweep ablation "$@"; }

for d in 0.0 0.5; do run "${BAG[@]}" --dropout $d; done
for h in 128 512; do run "${RNN[@]}" --hidden $h; done
for b in 32 128; do run "${BAG[@]}" --batch-size $b; run "${RNN[@]}" --batch-size $b; done
for s in 0 1 2; do run "${BAG[@]}" --embeddings $FT --seed $s; run "${RNN[@]}" --embeddings $FT --seed $s; done
# runner-up RNN config, three seeds, to check the seed-0 selection
for s in 0 1 2; do run --model rnn --format score --cell gru --bidirectional true --dropout 0.2 --lr 0.0003 --seed $s; done
