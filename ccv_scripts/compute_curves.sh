#!/bin/bash

files=`cat cifar_100.txt`


for f in $files
do
    base=`basename $f | cut -d"." -f1`
    dir=`dirname $f`
    final_name="$dir/resize_$base.png"
    out_name=$dir"/"$base"_to_msel_x7.cem" 

    val=`ls $out_name`

    echo $final_name
    echo $out_name

    if [ ! -z $val ]
    then
        echo "Found : "$out_name" exiting"
        continue
    fi

    echo "Working on $f"

    convert $f -resize 224x224 $final_name

    out_name=$dir"/"$base"_to_msel_x7.cem" 
    /users/mnarayan/scratch/Topological_Contour_Graph/bin/MSEL_img2CFs $final_name $out_name 200 0.5 1

    rm -f final_name


done

