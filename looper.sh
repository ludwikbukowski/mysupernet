#!/bin/bash

DIR=single_mgr
RES_DIR=single_mgr_turbo

for filename in ${DIR}/*.h5; do
python3 super_training.py --model=${filename} --output=tmp.h5 --notwo --noone --epochs1=4 --epochs2=4 --epochs3=4  --batch-size1=1024 --batch-size2=1024 --batch-size3=1024
python3 super_training.py --model=tmp.h5 --output=tmp.h5 --notwo --nolast --epochs1=4 --epochs2=4 --epochs3=4  --batch-size1=1024 --batch-size2=1024 --batch-size3=1024
N=`basename ${filename}`
python3 super_training.py --model=tmp.h5 --output=$RES_DIR/${N} --nolast --noone --epochs1=4 --epochs2=4 --epochs3=4  --batch-size1=1024 --batch-size2=1024 --batch-size3=1024
done

