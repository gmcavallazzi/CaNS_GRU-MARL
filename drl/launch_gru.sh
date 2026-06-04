#!/bin/bash
#SBATCH -J drl_gru
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=E-%x.%j.err
#SBATCH --gres=gpu:1

echo "=========================================="
echo "CaNS DRL: GRU-based MARL channel flow"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=========================================="

module load nvhpc/26.1 miniforge3/26.1.0
conda activate cans_drl

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/../dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=1
export OMPI_MCA_rmaps_base_oversubscribe=1

echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

mpirun --bind-to none --mca coll ^hcoll \
  -n 1 python stwStart_gru.py --config config_gru.yaml : \
  -n 1 ./cans

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
