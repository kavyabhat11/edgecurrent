#!/bin/bash
#SBATCH --job-name=ep_blimp
#SBATCH --output=logs/ep_blimp_%A_%a.out
#SBATCH --error=logs/ep_blimp_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5

# 6 sparsity levels (one per array task) for Edge Pruning on BLiMP s-v agreement.

set -e

cd ~/Edge-Pruning
mkdir -p logs data/runs

source ~/.bashrc
conda activate /scratch/network/$USER/edgepruning_env_py311

export PYTHONPATH=$PYTHONPATH:$(pwd)/src/modeling
export HF_HOME=/scratch/network/$USER/hf-cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/network/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

SPARSITIES=(0.94 0.95 0.96 0.97 0.98 0.99)
EDGE_SPARSITY=${SPARSITIES[$SLURM_ARRAY_TASK_ID]}

echo "Array idx: $SLURM_ARRAY_TASK_ID  ->  edge_sparsity=$EDGE_SPARSITY"

WANDB_MODE=disabled python src/prune/fpt2_blimp.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/blimp_regular_plural_subject_verb_agreement_1/ \
    --train_split train \
    --initialize_from gpt2 \
    --max_seq_length 32 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate 0.8 \
    --layer_learning_rate 0.8 \
    --reg_edge_learning_rate 0.8 \
    --reg_layer_learning_rate 0.8 \
    --max_steps 3000 \
    --warmup_steps 200 \
    --eval_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity 0.72 \
    --num_sparsity_warmup_steps 2500 \
    --max_train_samples 200 \
    --max_eval_samples 200 \
    --output_dir ./data/runs/blimp-es${EDGE_SPARSITY}-ns0.72-t3000/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --disable_node_loss
