#!/bin/sh
#SBATCH -N 16
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -C knl

module load taskfarmer
cd $SCRATCH/taskfarmer
export THREADS=24
export HDF5_USE_FILE_LOCKING=FALSE

runcommands.sh tasks.txt
