#!/bin/bash
export PYTHONPATH="/home/debian/.local/lib/python3.9/site-packages:$PYTHONPATH"
source out_files/exp_num
python --version
OUT_DIR="out_files/exp_$EXP_NUM"
echo EXP_NUM=$((EXP_NUM + 1)) > out_files/exp_num
mkdir -p $OUT_DIR
sudo chmod a+w, a+r $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r 5 -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python nothing.py -o$OUT_DIR -t12
chmod a+w, a+r $OUT_FILE
