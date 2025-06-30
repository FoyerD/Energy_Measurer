#!/bin/sh

min_gen=0
max_gen=6000

if [ $3 ]; then
	min_gen=$3
fi
if [ $4 ]; then
   	max_gen=$4
fi

python plot.py ./imp_outs/exp_$1/mean_measures.csv ./imp_outs/exp_$1/mean_statistics.csv ./imp_outs/exp_$1/imgs $2 --min_gen $min_gen --max_gen $max_gen
