#!/bin/bash

# Parsing the results into individual files
python parse_pinpoint.py ./imp_outs/exp_$1/raw.txt ./imp_outs/exp_$1/parsed_measures
python parse_statistics.py ./imp_outs/exp_$1/statistics.csv ./imp_outs/exp_$1/parsed_stats

base_pkg=0
base_gpu=0
if [ -n "$2" ]; then
    read base_pkg base_gpu < <(python parse_nothing.py ./imp_outs/exp_$2/raw.txt) 
fi

# Prepering the data for plotting
python merge_files.py ./imp_outs/exp_$1/parsed_measures ./imp_outs/exp_$1/parsed_stats ./imp_outs/exp_$1 --sdatetime --base_pkg $base_pkg --base_gpu $base_gpu
