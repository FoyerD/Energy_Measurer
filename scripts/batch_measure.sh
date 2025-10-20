#!/bin/sh

for setup_file in setups/batch_setups/*.toml; do
	echo "---------$setup_file----------"
	scripts/measure.sh -n 5 -s "$setup_file" -o out_files -r
done

