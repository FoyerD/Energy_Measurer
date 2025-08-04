#!/bin/bash
export PATH="/home/debian/anaconda3/envs/EM/bin/:/home/debian/repos/pinpoint/build:$PATH"  
OUT_DIR="out_files/exp_baseline"
mkdir -p $OUT_DIR
sudo chmod a+w,a+r $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r 5 -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python nothing.py -o$OUT_DIR -t$1
chmod a+w,a+r $OUT_FILE
