#!/bin/bash

# Parsing the results into individual files
python parse_pinpoint.py ./out_files/exp_$1/raw.txt ./out_files/exp_$1/parsed_measures
python parse_statistics.py ./out_files/exp_$1/statistics.csv ./out_files/exp_$1/parsed_stats

base_pkg=0
base_gpu=0
if [ -n "$2" ]; then
    read base_pkg base_gpu < <(python parse_nothing.py ./out_files/exp_$2/raw.txt) 
fi

# Prepering the data for plotting
python merge_files.py ./out_files/exp_$1/parsed_measures ./out_files/exp_$1/parsed_stats ./out_files/exp_$1 --sdatetime --base_pkg $base_pkg --base_gpu $base_gpu
