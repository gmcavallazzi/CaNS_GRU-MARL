#!/bin/bash
#SBATCH -J eval_gru
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=E-%x.%j.err
#SBATCH --gres=gpu:1

# Evaluate a GRU-MARL checkpoint on one full episode.
#
# Usage:
#   sbatch launch_eval.sh                                          # defaults: best_checkpoint
#   sbatch launch_eval.sh checkpoints_gru/checkpoint_ep_400.pth    # specific checkpoint
#   sbatch launch_eval.sh checkpoints_gru/checkpoint_ep_400.pth 1800   # custom episode length

CHECKPOINT="${1:-model/checkpoint_ep_108.pth}"
EP_LENGTH="${2:-}"

echo "=========================================="
echo "CaNS DRL: Evaluate checkpoint"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

module load nvhpc/26.1 miniforge3/26.1.0
conda activate cans_drl

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/../dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=1
export OMPI_MCA_rmaps_base_oversubscribe=1

nvidia-smi

# Build output dir name from checkpoint
CKPT_NAME=$(basename "$CHECKPOINT" .pth)
OUT_DIR="eval_${CKPT_NAME}"

EXTRA_ARGS=""
if [ -n "$EP_LENGTH" ]; then
    EXTRA_ARGS="--episode-length $EP_LENGTH"
fi

mpirun --bind-to none --mca coll ^hcoll \
  -n 1 python eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --config config_gru_refined.yaml \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS : \
  -n 1 ./cans

echo "=========================================="
echo "Eval completed at: $(date)"
echo "Plots in: $OUT_DIR/"
echo "=========================================="
