# Papers-GreenDNC

This repository contains the code and experimental setup for the GreenDNC paper. It includes configuration files, measurement scripts to run experiments, and parsing utilities to transform raw system data into the final result tables presented in the paper.

## Repository Contents

### 1. Configuration (`setups/`)

All experiment parameters are defined using `.toml` configuration files located in the `setups/` directory.
* The repository includes **4 base configurations** representing the core combinations of our experiments (2 operators × 2 domains).
* These `.toml` files are designed to be easily modified if you wish to tweak parameters or test new configurations.

### 2. Measurement Scripts

Measurement scripts are located in the `scripts/` directory and generally require `sudo` privileges to accurately measure system-level energy and performance metrics.

* **`scripts/measure.sh`**
    The core script for running a single experimental setup.
    * `-o`: Output directory (a new subdirectory will be created here).
    * `-n`: Number of experiments to run.
    * `-s`: The setup `.toml` file to use.
    * `-p`: Path to the Python binary to use.
    * `-r`: Setup command (binary). Defaults to copying the config to the output. If set, it moves it to config.
    * *Example Usage:*
      ```bash
      sudo scripts/measure.sh -n 2 -s setups/dnc_gc.toml -o out_files -p /home/foyer/.conda/envs/energy_measure/bin/python
      ```

* **`scripts/batch_measure.sh`**
    Automates running multiple configurations. It runs the experiment 5 times for each configuration found in the `setups/batch_setups` directory.
    * *Example Usage:*
      ```bash
      sudo scripts/batch_measure.sh /home/foyer/.conda/envs/energy_measure/bin/python
      ```

* **`scripts/measure_nothing.sh`**
    Runs a dummy experiment to establish a system energy/performance baseline.
    * `-o`: Output directory (creates an `exp_baseline` folder inside the specified path).
    * `-n`: Number of experiments to run.
    * `-p`: Path to the Python binary.
    * *Example Usage:*
      ```bash
      sudo scripts/measure_nothing.sh -n 5 -o out_files -p /home/foyer/.conda/envs/energy_measure/bin/python
      ```

### 3. Data Parsing & Results

Once raw data is generated, use the Python scripts in the root directory to process the metrics.

* **`parse.py`**
    Takes a raw experiment directory and parses it to generate readable results, subtracting the system baseline.
    * *Example Usage:*
      ```bash
      python parse.py out_files/exp_dir --baseline_dir out_files/exp_baseline
      ```

* **`results.py`**
    Takes a directory containing multiple experiment subdirectories and aggregates them into a formatted table (as seen in the paper). Use the `pivot` argument to format the output table correctly.
    * *Example Usage:*
      ```bash
      python results.py imp_outs pivot
      ```

### 4. Utilities

* **`merge_exps.py`**
    Merges two separate raw experiment directories into a single directory (e.g., combining an older run of 5 experiments with a newer run of 5 experiments to create a 10-experiment dataset).
    * *Note:* You must run `clear_exps.sh` and then re-parse the data after merging.
    * *Example Usage:*
      ```bash
      python merge_exps.py old_exps new_exps
      ```

* **`scripts/clear_exps.sh`**
    Receives a list of experiment directories and cleans them, leaving *only* the raw unparsed data. Essential for resetting state before re-parsing or after merging.
    * *Example Usage:*
      ```bash
      scripts/clear_exps.sh out_files/*
      ```

### 5. Advanced / Internal Tools

* **`config_maker.py`**
    An internal utility that reads the 4 main config files from `setups/` and populates the `setups/batch_setups/` directory with individual configurations for each experiment run in the paper. 
    * *Example Usage:*
      ```bash
      python config_maker.py
      ```

## Recommended Workflow

1. **Configure:** Modify the `.toml` files in `setups/` as needed, or use `config_maker.py` to generate the batch setups.
2. **Establish Baseline:** Run `scripts/measure_nothing.sh` to get the system energy baseline.
3. **Run Experiments:** Use `scripts/batch_measure.sh` to collect raw data across your configurations.
4. **Merge (Optional):** If combining multiple runs, use `merge_exps.py` to consolidate directories, followed by `scripts/clear_exps.sh`.
5. **Parse Data:** Use `parse.py` on your outputs alongside your generated baseline.
6. **Generate Tables:** Run `results.py pivot` to compile the parsed data into the final tables for the paper.
