#!/bin/bash
export PYTHONPATH="/home/debian/.local/lib/python3.9/site-packages:$PYTHONPATH"
source out_files/exp_num
python --version
OUT_DIR="out_files/exp_$EXP_NUM"
mkdir -p $OUT_DIR
sudo chmod 777 $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r 3 -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python exp_runner.py k_point uniform bpp -n 10 -stats -o$OUT_DIR
chmod 0777 $OUT_FILE
echo EXP_NUM=$((EXP_NUM + 1)) > out_files/exp_num

