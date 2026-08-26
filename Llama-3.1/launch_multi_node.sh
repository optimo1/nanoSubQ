#!/bin/bash
# Launch SSA-LLaMA training across 8 DGX H100 nodes (64 GPUs).
#
# Run this on EACH node with a different NODE_RANK:
#   Node 0: bash launch_multi_node.sh 0
#   Node 1: bash launch_multi_node.sh 1
#   ...
#   Node 7: bash launch_multi_node.sh 7
#
# Or use a cluster scheduler (SLURM, Kubernetes) to set NODE_RANK automatically.
set -e

NODE_RANK=${1:?Usage: bash launch_multi_node.sh <node_rank>}
NUM_NODES=${2:-8}
MASTER_ADDR=${MASTER_ADDR:-"10.0.0.1"}   # IP of node 0 (set by your cluster)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=8

cd "$(dirname "$0")"

echo "=== Node $NODE_RANK / $NUM_NODES | Master: $MASTER_ADDR:$MASTER_PORT ==="

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
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
