#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 23:30:00
#SBATCH -o logs/%x-%j.log
# LoRA training of MiniMax-H3 Ref2VA: one sample per GPU, the frozen model FSDP-sharded over the GPUs of each node.
# Resumes from the newest checkpoint in OUT, so a job that ran out of time can be resubmitted unchanged.
# The paper's SimPWM-H3 run: 32 GPUs (8 nodes x 4 A100 80 GB, or 4 nodes x 8), ARGS as below. Submit from the
# repository root.
# usage: sbatch -N 8 --gres=gpu:4 scripts/h3/train.sbatch DS_DIR CACHE OUT \
#            --ids resources/droid/splits/train_h3.txt --steps 500 --stop-at 250 --lr 1.4e-4 --warmup 25 \
#            --save-every 125 --act-offload off
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
source "$REPO/scripts/h3/env.sh"
DS=$1 CACHE=$2 OUT=$3
shift 3
G=$(nvidia-smi -L | wc -l)
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
cd "$REPO"
srun --kill-on-bad-exit=1 bash -c "torchrun --nnodes=$SLURM_NNODES --nproc-per-node=$G --node-rank=\$SLURM_NODEID \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29544 --rdzv-id=$SLURM_JOB_ID \
  -m simpwm.h3.train --data $DS --cache $CACHE --out $OUT $*"
