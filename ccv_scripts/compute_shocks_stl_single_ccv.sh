#!/bin/bash
#SBATCH -J COMPUTE_SHOCKS_TINY_IN
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=2G

cd $PWD



# This script runs 16 independent MATLAB tasks across 2 oscar nodes by creating
# a "job array" of 16 separate 1-core jobs.
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

FILE=$(awk "NR==$SLURM_ARRAY_TASK_ID" /users/mnarayan/scratch/scripts/stl_lo.txt)

echo "Working on $FILE line number $SLURM_ARRAY_TASK_ID"
matlab -r "addpath /users/mnarayan/scratch/Topological_Contour_Graph/util/io; cem2cemv('$FILE'); exit"

/users/mnarayan/scratch/scripts/compute_shocks.sh $FILE
esf_file=`echo $FILE | sed s/\.png/_se_tcg\.esf/g`
echo "Esf file: $esf_file"


python /users/mnarayan/scratch/gcn_shock_graph/shock_graph.py $esf_file $FILE

cemv_file=`echo $FILE | sed s/\.png/_se_tcg\.cemv/g`
h5_file=`echo $FILE | sed s/\.png/_se_tcg\*h5/g`

val1=`ls $esf_file`
val2=`ls $cemv_file`
val3=`ls $h5_file`


if [ ! -z "$val1" ] && [ ! -z "$val2" ] && [ ! -z "$val3" ]
then
    echo "Found all three files"
    zip_file=`echo $FILE | sed s/\.png/_se_tcg\.zip/g`
    zip -j $zip_file $val1 $val2 $val3
    rm -f $val1
    rm -f $val2
    rm -f $val3
fi





