#!/bin/bash
# Launch SSA-LLaMA training on a single node with 8 GPUs.
# Usage: bash launch_single_node.sh
set -e

cd "$(dirname "$0")"

torchrun \
    --nproc_per_node=8 \
    train_ssa.py \
    --model-id meta-llama/Llama-3.1-8B \
    --data-dir data \
    --out-dir out_ssa \
    --batch-size 2 \
    --grad-accum 8 \
    --block-size 2048 \
    --lr 2e-5 \
    --max-iters 10000 \
    --eval-interval 500 \
    --grad-checkpoint
