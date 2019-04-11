#!/bin/bash

files=`cat missing_files.txt`

for f in $files
do
    image=`basename $f | cut -d"." -f1`
    dir=`dirname $f`
    
    find $dir -type f -name "resize_$image*" -delete

done

