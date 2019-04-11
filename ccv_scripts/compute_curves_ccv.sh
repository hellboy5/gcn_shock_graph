#!/bin/bash
#SBATCH -J COMPUTE_CURVES_STL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00


cd $PWD

./compute_curves_stl.sh
