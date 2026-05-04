#!/bin/bash
# Train-size sweep for GP, run on login node (test set is 378 examples).
# Runs train sizes 20/50/100/200 sequentially, then prints a side-by-side comparison.

set -e

for n in 20 50 100 200; do
  echo "=== Train examples: $n ==="
  python baselines_examples/eap_gp_paper.py \
    --dataset-path data/data/datasets/gp/ \
    --train-examples $n \
    --eval-examples 378 \
    --eval-split test \
    --sparsities 93,94,95,96,97,98,99 \
    --out-dir results/gp-eap-test-train$n
done

echo ""
echo "=== Comparison ==="
python baselines_examples/compare_train_sweep.py \
  results/gp-eap-test-train20 \
  results/gp-eap-test-train50 \
  results/gp-eap-test-train100 \
  results/gp-eap-test-train200
