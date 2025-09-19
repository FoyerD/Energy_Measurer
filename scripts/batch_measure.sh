#!/bin/sh

for setup_file in setups/batch_setups/*.toml; do
	#scripts/measure.sh -n 20 -s "$setup_file" -o out_files
	echo $setup_file
done

