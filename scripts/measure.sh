#!/bin/bash
source out_files/exp_num
OUT_DIR="out_files/exp_$EXP_NUM"
mkdir -p $OUT_DIR
pinpoint -c -r 1 -i 250 -e CPU,GPU -- python exp_runner.py k_point uniform bpp --n_gens 20 2> $OUT_DIR/raw.txt
echo EXP_NUM=$((EXP_NUM + 1)) > out_files/exp_num