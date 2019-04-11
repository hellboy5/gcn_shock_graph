#!/bin/bash

input_object=`basename $1 | cut -d"." -f1`
input_folder=`dirname $1`

out_name=`echo $1 | sed s/\.JPEG/_se_tcg.esf/g`
val=`ls $out_name`

if [ ! -z $val ]
then
    echo "Found : "$out_name" exiting"
    exit
fi

input_filename_shocks=$input_object"_shocks.xml"

temp1=$input_object"_one_var.txt"
temp2=$input_object"_temp.xml"
temp3=$input_object"_temp3.xml"
temp4=$input_object"_temp4.xml"

echo "Writing results to "$output_folder
echo " "
echo "Working on "$input_object

echo $input_folder | sed 's/\//\\\//g' > $temp1
var=`cat $temp1`


sed 's/input_object_dir=\"\"/input_object_dir="'$var$'"/' /users/mnarayan/scratch/scripts/base/input_defaults_shocks.xml > $temp3
sed 's/input_object_name=\"\"/input_object_name="'$input_object'"/' $temp3 > $input_filename_shocks

/users/mnarayan/scratch/scripts/base/dbsk2d_ishock_esf_computation -x $input_filename_shocks

rm -f $temp1
rm -f $temp2
rm -f $temp3
rm -f $temp4
rm -f $input_filename_shocks

