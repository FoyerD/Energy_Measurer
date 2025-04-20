#!/bin/bash

# Run the command with superuser privileges
su -c 'pinpoint -c -r 2 -i 250 -e CPU,GPU -- python exp_runner.py k_point uniform bpp --n_gens 20'