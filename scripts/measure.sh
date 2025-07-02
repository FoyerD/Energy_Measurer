#!/bin/bash
export PATH="/home/debian/anaconda3/envs/EM/bin/:/home/debian/repos/pinpoint/build:$PATH"
source out_files/exp_num
python --version
OUT_DIR="out_files/exp_$EXP_NUM"
echo EXP_NUM=$((EXP_NUM + 1)) > out_files/exp_num
mkdir -p $OUT_DIR
sudo chmod a+w,a+r $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r 5 -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python exp_runner.py dnc uniform bpp -n 6000 -stats -o$OUT_DIR
chmod a+w,a+r $OUT_FILE

