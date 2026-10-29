#!/usr/bin/env bash
# Final runs: best validation config per model, three seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."
which=${1:-all}
for s in 0 1 2; do
  if [[ $which == all || $which == bag ]]; then
    uv run python scripts/train.py --tags final --model bag --format score --interactions true --hidden 512 --lr 0.001 --seed $s
    uv run python scripts/train.py --tags baseline --model bag --format score --hyp-only true --hidden 512 --lr 0.001 --seed $s
  fi
done
