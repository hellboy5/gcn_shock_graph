#!/bin/bash
#SBATCH -J COMPUTE_SHOCK_CONVERT
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00


cd $PWD

./driver_shock_convert.sh
