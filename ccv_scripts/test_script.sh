#!/bin/bash


SLURM_ARRAY_TASK_ID=1
cd $PWD

# This script runs 16 independent MATLAB tasks across 2 oscar nodes by creating
# a "job array" of 16 separate 1-core jobs.
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

offset=$(( $SLURM_ARRAY_TASK_ID-1 ))
start=$(( $offset*2 ))
let start++
stop=$(( $SLURM_ARRAY_TASK_ID*2 ))

echo "Working on files between $start and $stop"
for i in $(seq $start $stop)
do
    FILE=$(awk "NR==$i" /users/mnarayan/scratch/cifar_100.txt)
    echo "Working on $FILE line number $i"
    matlab -r "addpath /users/mnarayan/scratch/Topological_Contour_Graph/util/io; cem2cemv('$FILE'); exit"

    dir=`dirname $FILE`
    base=`basename $FILE | cut -d"." -f1`
    final_name="$dir/resize_$base.png"
    orig_cem_name=$dir"/"$base"_to_msel_x7.cemv"
    final_cem_name=$dir"/resize_"$base"_to_msel_x7.cemv"

    mv $orig_cem_name $final_cem_name
    convert $FILE -resize 224x224 $final_name

    echo "Resize file: $final_name"
    
    /users/mnarayan/scratch/compute_shocks.sh $final_name
    esf_file=`echo $final_name | sed s/\.png/_to_msel_x7\.esf/g`
    echo "Esf file: $esf_file"
    python /users/mnarayan/scratch/gcn_shock_graph/shock_graph.py $esf_file $final_name

    rm -f $final_name
done

