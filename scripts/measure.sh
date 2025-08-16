#!/bin/sh

export PATH="/home/foyer/.conda/envs/energy_measure/bin/:/home/debian/anaconda3/envs/EM/bin/:/home/debian/repos/pinpoint/build:$PATH"
#source out_files/exp_num

OUT_DIR=""
NUM_EXPS=1
SETUP_FILE="setups/setup.toml"


while getopts "o:n:s:" opt; do
  case "$opt" in
    o)
        OUT_DIR="out_files/exp_$OPTARG"
        ;;
    
    n)
        NUM_EXPS=$OPTARG
        ;;
    
    s)
        SETUP_FILE=$OPTARG
        ;;
  esac
done

if [ -z "$OUT_DIR" ]; then
        echo "Output directory not specified. Use -o <output_directory>."
        exit 1
fi
if [ -z "$SETUP_FILE" ]; then
        echo "Setup file not specified. Use -s <setup_file>."
        exit 1
fi



mkdir -p $OUT_DIR
sudo chmod a+w,a+r $OUT_DIR
OUT_FILE=$OUT_DIR/raw.txt

pinpoint -c --timestamp -r $NUM_EXPS -i 250 -e rapl:pkg,GPU -o $OUT_FILE -- python exp_runner.py --setup_file $SETUP_FILE -o $OUT_DIR
chmod a+w,a+r $OUT_FILE

