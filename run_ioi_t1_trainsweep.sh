#!/bin/bash
# Train-size sweep for IOI-t1, run on login node (test set is only 50 examples).
# Runs train sizes 20/50/100/200 sequentially, then prints a side-by-side comparison.

set -e

for n in 20 50 100 200; do
  echo "=== Train examples: $n ==="
  python baselines_examples/eap_ioi_paper.py \
    --dataset-path data/data/datasets/ioi-t1/ \
    --train-examples $n \
    --eval-examples 50 \
    --eval-split test \
    --sparsities 93,94,95,96,97,98,99 \
    --out-dir results/ioi-t1-eap-test-train$n
done

echo ""
echo "=== Comparison ==="
python baselines_examples/compare_train_sweep.py \
  results/ioi-t1-eap-test-train20 \
  results/ioi-t1-eap-test-train50 \
  results/ioi-t1-eap-test-train100 \
  results/ioi-t1-eap-test-train200
