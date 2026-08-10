#!/bin/sh

export PATH="/home/debian/repos/pinpoint/build:$PATH"

OUT_DIR=""
NUM_EXPS=1
SETUP_FILE=""
SETUP_COMMAND=cp
PYTHON_COMMAND=""


while getopts "o:n:s:rp:" opt; do
  case "$opt" in
	o)
		OUT_DIR="$OPTARG"
		;;
    
	n)
		NUM_EXPS=$OPTARG
		;;
    
	s)
		SETUP_FILE="$OPTARG"
		;;
    p)
        PYTHON_COMMAND="$OPTARG"
        ;;

	r)
		SETUP_COMMAND=mv
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
if [ ! -f "$SETUP_FILE" ]; then
		echo "Setup file $SETUP_FILE does not exist."
		exit 1
fi
if [ -z "$PYTHON_COMMAND" ]; then
        echo "Please provide full path to a python binary using -p."
        exit 1
fi
if [ ! -f "$PYTHON_COMMAND" ]; then
		echo "Python binary $PYTHON_COMMAND does not exist."
		exit 1
fi

EXP_DIR=$($PYTHON_COMMAND exp_namer.py $SETUP_FILE)
status=$?
if [ $status -ne 0 ]; then
    echo "Naming of exp dir failed with exit code $status"
	exit 1
fi

OUT_DIR="$OUT_DIR/$EXP_DIR"
OUT_FILE=$OUT_DIR/raw.txt

mkdir -p $OUT_DIR
chmod a+w,a+r $OUT_DIR

which python
pinpoint -c --timestamp -r $NUM_EXPS -e rapl:pkg,GPU -o $OUT_FILE -- $PYTHON_COMMAND exp_runner.py --setup_file $SETUP_FILE -o $OUT_DIR

$SETUP_COMMAND $SETUP_FILE $OUT_DIR

chmod a+r $OUT_FILE
