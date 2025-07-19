#!/bin/bash

export PATH="/home/debian/anaconda3/envs/EM/bin/:/home/debian/repos/pinpoint/build:$PATH"
source out_files/exp_num

OUT_DIR=""


while getopts "o:" opt; do
  case "$opt" in
    o)
      OUT_DIR="out_files/exp_$OPTARG"
      ;;
  esac
done

if [ -z "$OUT_DIR" ]; then
	OUT_DIR="out_files/exp_$EXP_NUM"
	echo EXP_NUM=$((EXP_NUM + 1)) > out_files/exp_num
fi

mkdir -p $OUT_DIR
sudo chmod a+w,a+r $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r 5 -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python exp_runner.py dnc uniform bpp -n 6000 -stats -o$OUT_DIR

chmod a+w,a+r $OUT_FILE

