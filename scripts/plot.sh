#!/bin/sh

min_gen=0
max_gen=6000

if [ $2 ]; then
	min_gen=$2
fi
if [ $3 ]; then
   	max_gen=$3
fi

python plot.py ./imp_outs/exp_$1/mean_measures.csv ./imp_outs/exp_$1/mean_statistics.csv ./imp_outs/exp_$1/imgs --min_gen $min_gen --max_gen $max_gen $4
