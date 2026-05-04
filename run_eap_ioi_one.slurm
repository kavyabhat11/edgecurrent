#!/bin/bash
#SBATCH --job-name=eap_ioi
#SBATCH --output=logs/eap_ioi_train%x_%j.out
#SBATCH --error=logs/eap_ioi_train%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Submit with: sbatch --export=N=20 --job-name=ioi_t20 run_eap_ioi_one.slurm
# (and similarly for N=50, 100, 200 in parallel)

set -e
: "${N:?Set N via --export=N=20 (or 50/100/200)}"

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

python baselines_examples/eap_ioi_paper.py \
  --dataset-path data/data/datasets/ioi/ \
  --train-examples $N \
  --eval-examples 36084 \
  --eval-split test \
  --sparsities 93,94,95,96,97,98,99 \
  --out-dir results/ioi-eap-test-train$N
