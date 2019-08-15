#!/bin/bash

BATCH_SIZE1=1024
EPOCHS=3

python3 super_chosen.py --model=adams/adam5.h5 --epochs=3 --batch_size=${BATCH_SIZE} --output=tmp.h5 --index=1
python3 super_chosen.py --model=tmp.h5 --epochs=3 --batch_size=${BATCH_SIZE} --output=tmp.h5 --index=2
python3 super_chosen.py --model=tmp.h5 --epochs=5 --batch_size=${BATCH_SIZE} --output=tmp.h5 --index=3
python3 super_chosen.py --model=tmp.h5 --epochs=7 --batch_size=${BATCH_SIZE} --output=tmp.h5 --index=4

