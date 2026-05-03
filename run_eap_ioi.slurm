#!/bin/bash
#SBATCH --job-name=eap_ioi
#SBATCH --output=logs/eap_ioi_%j.out
#SBATCH --error=logs/eap_ioi_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd ~/Edge-Pruning

mkdir -p logs results/ioi-eap

source ~/.bashrc
conda activate /scratch/network/$USER/edgepruning_env_py311

export PYTHONPATH=$PYTHONPATH:/home/$USER/edge-attribution-patching
export HF_HOME=/scratch/network/$USER/hf-cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/hf-cache
export TMPDIR=/scratch/network/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

python baselines_examples/run_eap_ioi.py \
  --dataset-path data/datasets/merged/ioi/ \
  --max-examples 200 \
  --batch-size 20 \
  --out-json-path results/ioi-eap/eap_edges.json
