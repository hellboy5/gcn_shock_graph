#!/bin/bash

files=`cat stl_files.txt`

for f in $files
do
    image=`basename $f | cut -d"." -f1`
    dir=`dirname $f`
    step=$image"_to_msel.zip"
    foo=`ls $dir/$step`

    if [ -z "$foo" ]
    then
	echo $f
    fi

done
