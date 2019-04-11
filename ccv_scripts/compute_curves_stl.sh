#!/bin/bash

files=`cat stl_files.txt`


for f in $files
do
    echo "Working on "$f
    base=`basename $f | cut -d"." -f1`
    dir=`dirname $f`
    out_name=$dir"/"$base"_to_msel.cem" 
    echo "Writing to "$out_name
    
    /users/mnarayan/scratch/Topological_Contour_Graph/bin/MSEL_img2CFs $f $out_name 200 1.5 1





done

