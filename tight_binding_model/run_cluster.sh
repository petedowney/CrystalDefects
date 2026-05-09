#!/bin/bash

#SBATCH -J compute-tbm-values
#SBATCH --account= put your account here
#SBATCH --partition=normal_q
#SBATCH --nodes=2
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
#SBATCH --time=0-15:00:00

# module load SciPy-bundle/2024.05-gfbf-2024a
# module load mpi4py/4.0.1-gompi-2024a
# source ~/mpi-env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
srun --mpi=pmix -n 100 python train_mpi.py --epochs 150 --sample-limit -1 --batch-size 60