#!/bin/sh

if [ $1 ]; then
	scp -r -i ~/.ssh/id_amit -P 5050 debian@100.95.44.40:~/repos/Energy_Measurer/out_files/exp_$1 ~/repos/Energy_Measurer/out_files/exp_$1
fi
else
	echo "Please provide the experiment name as an argument."
fi
