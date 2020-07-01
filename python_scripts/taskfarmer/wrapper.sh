#!/usr/bin/env bash

cd $SCRATCH/taskfarmer
module load python
source activate spe3sm_env
export HDF5_USE_FILE_LOCKING=FALSE
python vinterp_with_args.py --date $1 --model $2
