#!/bin/bash
#SBATCH --job-name=ssa-llama
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/ssa_%j.out
#SBATCH --error=logs/ssa_%j.err

# SLURM: launch one torchrun per node
# Requires: --nodes=8 --gpus-per-node=8
#
# Submit:  sbatch launch_slurm.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/ssa_<jobid>.out

mkdir -p logs

# SLURM sets SLURM_NODELIST, SLURM_PROCID, etc.
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=29500
NUM_NODES=$SLURM_NNODES

# Each task (node) runs this script
srun --nodes=$NUM_NODES --ntasks-per-node=1 --gpus-per-node=8 \
    bash -c "
        NODE_RANK=\$SLURM_PROCID
        echo \"Node \$NODE_RANK starting — master=$MASTER_ADDR:$MASTER_PORT\"
        torchrun \
            --nproc_per_node=8 \
            --nnodes=$NUM_NODES \
            --node_rank=\$NODE_RANK \
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
    "
