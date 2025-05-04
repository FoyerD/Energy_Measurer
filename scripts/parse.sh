#!/bin/bash

python parse_pinpoint.py ./out_files/exp_$1/raw.txt ./out_files/exp_$1

python parse_statistics.py ./out_files/exp_$1/statistics.csv ./out_files/exp_$1