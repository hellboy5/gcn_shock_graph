#!/bin/bash

files=`find /users/mnarayan/scratch/cifar_100 -type f -name "*_to_msel.esf"`

for f in $files
do
    png_file=`echo $f | sed s/_to_msel\.esf/\.png/g`

    echo "Working on $f with $png_file"
   
    python /users/mnarayan/scratch/gcn_shock_graph/shock_graph.py $f $png_file
    
done
