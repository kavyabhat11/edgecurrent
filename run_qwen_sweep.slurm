#!/bin/bash
#SBATCH --job-name=qwen_ep
#SBATCH --output=logs/qwen_ep_%A_%a.out
#SBATCH --error=logs/qwen_ep_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20

# Each array task = one (task, sparsity) pair.
# Submit with: sbatch run_qwen_sweep.slurm
#
# Tasks: ioi, ioi-t1, gp, gt
# Sparsities: 0.94 0.95 0.96 0.97 0.98 0.99 0.995  (7 each)
# 4 tasks * 7 sparsities = 28 jobs (array indexes 0-27); set --array accordingly if expanding.
# To start, we run only IOI + GT first (14 jobs, indexes 0-13).

set -e

cd ~/Edge-Pruning
mkdir -p logs data/runs

source ~/.bashrc
conda activate /scratch/network/$USER/edgepruning_env_py311

export PYTHONPATH=$PYTHONPATH:/home/$USER/edge-attribution-patching:$(pwd)/src/modeling
export HF_HOME=/scratch/network/$USER/hf-cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/network/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

SPARSITIES=(0.94 0.95 0.96 0.97 0.98 0.99 0.995)
TASKS=(ioi ioi-t1 gp gt)
DATASETS=(ioi ioi-t1 gp gt)
SCRIPTS=(fqwen_ioi.py fqwen_ioi.py fqwen_gp.py fqwen_gt.py)

NUM_SP=${#SPARSITIES[@]}                                    # 7
TASK_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SP ))                # 0..3
SP_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SP ))                  # 0..6

TASK=${TASKS[$TASK_IDX]}
DATASET=${DATASETS[$TASK_IDX]}
SCRIPT=${SCRIPTS[$TASK_IDX]}
EDGE_SPARSITY=${SPARSITIES[$SP_IDX]}

echo "Array idx: $SLURM_ARRAY_TASK_ID  ->  task=$TASK  sparsity=$EDGE_SPARSITY  script=$SCRIPT"

NODE_SPARSITY=0.72
TOTAL=3000
WARMUP=2500

WANDB_MODE=disabled python src/prune/$SCRIPT \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/$DATASET/ \
    --train_split train \
    --initialize_from Qwen/Qwen2-0.5B \
    --max_seq_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate 0.8 \
    --layer_learning_rate 0.8 \
    --reg_edge_learning_rate 0.8 \
    --reg_layer_learning_rate 0.8 \
    --max_steps $TOTAL \
    --warmup_steps 200 \
    --eval_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples 1000000 \
    --max_eval_samples 200 \
    --output_dir ./data/runs/qwen-${TASK}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --disable_node_loss
