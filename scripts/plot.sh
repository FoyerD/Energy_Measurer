#!/bin/bash

# Plotting the data
python plot.py ./out_files/exp_$1/mean_measures.csv ./out_files/exp_$1/mean_statistics.csv ./out_files/exp_$1 $2
