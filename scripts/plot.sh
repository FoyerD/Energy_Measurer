#!/bin/bash
# Parsing the results into individual files
python parse_pinpoint.py ./out_files/exp_$1/raw.txt ./out_files/exp_$1/parsed_measures
python parse_statistics.py ./out_files/exp_$1/statistics.csv ./out_files/exp_$1/parsed_stats

# Prepering the data for plotting
python merge_files.py ./out_files/exp_$1/parsed_measures ./out_files/exp_$1/parsed_stats ./out_files/exp_$1

# Plotting the data
python plot.py ./out_files/exp_$1/mean_measure.csv ./out_files/exp_$1/mean_statistics.csv ./out_files/exp_$1
