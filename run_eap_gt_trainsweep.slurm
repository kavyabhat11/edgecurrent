#!/bin/bash
#SBATCH --job-name=eap_gt_sweep
#SBATCH --output=logs/eap_gt_sweep_%j.out
#SBATCH --error=logs/eap_gt_sweep_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd ~/Edge-Pruning

mkdir -p logs results

source ~/.bashrc
conda activate /scratch/network/$USER/edgepruning_env_py311

export PYTHONPATH=$PYTHONPATH:/home/$USER/edge-attribution-patching
export HF_HOME=/scratch/network/$USER/hf-cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/scratch/network/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

for n in 20 50 100; do
  python baselines_examples/eap_gt_paper.py \
    --dataset-path data/data/datasets/gt/ \
    --train-examples $n \
    --eval-examples 12240 \
    --eval-split test \
    --sparsities 93,94,95,96,97,98,99 \
    --out-dir results/gt-eap-test-train$n
done
