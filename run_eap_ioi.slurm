#!/bin/bash
#SBATCH --job-name=eap_ioi
#SBATCH --output=logs/eap_ioi_%j.out
#SBATCH --error=logs/eap_ioi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd ~/Edge-Pruning

mkdir -p logs results/ioi-eap-val results/ioi-eap-test

source ~/.bashrc
conda activate /scratch/network/$USER/edgepruning_env_py311

export PYTHONPATH=$PYTHONPATH:/home/$USER/edge-attribution-patching
export HF_HOME=/scratch/network/$USER/hf-cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/hf-cache
export TMPDIR=/scratch/network/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

# 1) Validation run: 200 examples (fast, sanity check + matches paper's val split)
python baselines_examples/eap_ioi_paper.py \
  --dataset-path data/data/datasets/ioi/ \
  --train-examples 200 \
  --eval-examples 200 \
  --eval-split validation \
  --sparsities "93,94,95,96,97,98,99" \
  --out-dir results/ioi-eap-val

# 2) Test run: 36,084 examples (paper-grade numbers)
python baselines_examples/eap_ioi_paper.py \
  --dataset-path data/data/datasets/ioi/ \
  --train-examples 200 \
  --eval-examples 36084 \
  --eval-split test \
  --sparsities "93,94,95,96,97,98,99" \
  --out-dir results/ioi-eap-test
