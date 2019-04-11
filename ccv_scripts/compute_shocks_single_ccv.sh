#!/bin/bash
#SBATCH -J COMPUTE_SHOCKS_X7
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G

cd $PWD

# This script runs 16 independent MATLAB tasks across 2 oscar nodes by creating
# a "job array" of 16 separate 1-core jobs.
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"


# MATLAB function, you can use lines 1 through 16 of asay
# 'inputs.txt', that is in the same directory as where you script.
FILE=$(awk "NR==$SLURM_ARRAY_TASK_ID" /users/mnarayan/scratch/missing_files.txt)
echo "Starting job $FILE on $HOSTNAME"


matlab -r "addpath /users/mnarayan/scratch/Topological_Contour_Graph/util/io; cem2cemv('$FILE'); exit"

/users/mnarayan/scratch/compute_shocks.sh $FILE
esf_file=`echo $FILE | sed s/\.png/_to_msel\.esf/g`
echo "Esf file: $esf_file"
    
python /users/mnarayan/scratch/gcn_shock_graph/shock_graph.py $esf_file $FILE

