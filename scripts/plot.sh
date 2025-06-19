#!/bin/bash

# Plotting the data
python plot.py ./imp_outs/exp_$1/mean_measures.csv ./imp_outs/exp_$1/mean_statistics.csv ./imp_outs/exp_$1 $2
