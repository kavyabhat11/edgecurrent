#!/bin/bash
# Edge Pruning sweep on GPT-2 small for BLiMP regular_plural_subject_verb_agreement_1.
# Adapted from run_scripts/ioi_sweep.sh.

EDGE_SPARSITIES=(0.94 0.95 0.96 0.97 0.98 0.99 0.995)

for i in "${!EDGE_SPARSITIES[@]}"; do

EDGE_SPARSITY=${EDGE_SPARSITIES[i]}
NODE_SPARSITY=0.72
ELR=0.8
LLR=0.8
RELR=0.8
RLLR=0.8
TOTAL=3000
WARMUP=2500

EXTRA="--disable_node_loss"
TAG="wo_node_loss"

train_split="train"
N_TRAIN=200
N_VAL=200

WANDB_MODE=disabled python src/prune/fpt2_blimp.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/blimp_regular_plural_subject_verb_agreement_1/ \
    --train_split $train_split \
    --initialize_from gpt2 \
    --max_seq_length 32 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate $ELR \
    --layer_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_layer_learning_rate $RLLR \
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
    --max_train_samples $N_TRAIN \
    --max_eval_samples $N_VAL \
    --output_dir ./data/runs/blimp-${TAG}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    $EXTRA

done
