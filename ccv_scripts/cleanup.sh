#!/bin/bash

# find cifar_100/train -type f -name "*_to_msel.esf" -exec zip -j train_esf_files.zip {} \;
# find cifar_100/train -type f -name "*_to_msel-n[0-9]*.h5" -exec zip -j train_sg_files.zip {} \;
# find cifar_100/train -type f -name "*_to_msel.cemv" -exec zip -j train_cemv_files.zip {} \;

# find cifar_100/test -type f -name "*_to_msel.esf" -exec zip -j test_esf_files.zip {} \;
# find cifar_100/test -type f -name "*_to_msel-n[0-9]*.h5" -exec zip -j test_sg_files.zip {} \;
# find cifar_100/test -type f -name "*_to_msel.cemv" -exec zip -j test_cemv_files.zip {} \;

# ./find_missing.sh > missing_files.txt

find ../cifar_100/test -type f -name "*_to_msel-n[0-9]*.h5" -delete
find ../cifar_100/test -type f -name "*_to_msel.esf" -delete
find ../cifar_100/test -type f -name "*_to_msel.cem" -delete
find ../cifar_100/test -type f -name "*_to_msel.cemv" -delete

