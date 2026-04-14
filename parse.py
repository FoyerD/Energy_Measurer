import argparse
import csv
import os
from statistics import mean
import pandas as pd
from functools import reduce
from pathlib import Path
        
_gc_metadata_cache = None
def transform_gc_fitness(instance, raw_value, metadata_path="datasets_dnc/graph_coloring/graph_baselines.csv", penalty=1000):
    if (raw_value < 1):
        return raw_value

    global _gc_metadata_cache
    
    if _gc_metadata_cache is None:
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata CSV missing: {metadata_path}")
        df = pd.read_csv(metadata_path)
        _gc_metadata_cache = df.set_index('instance_name').to_dict('index')
    if instance not in _gc_metadata_cache:
        raise ValueError(f"Instance '{instance}' not found in metadata.")
    if raw_value >= penalty:
        return -float('inf')

    k = raw_value
    n_nodes = _gc_metadata_cache[instance]['nodes']

    if n_nodes > 1:
        maximizing_fitness = (n_nodes - k) / (n_nodes - 1)
    else:
        maximizing_fitness = 1.0

    return maximizing_fitness

def transform_measures(df: pd.DataFrame, base_pkg:float=0.0, base_gpu:float=0.0):
    df['PKG'] = (df['PKG'].astype(float) - base_pkg) / 1000
    df['GPU'] = (df['GPU'].astype(float) - base_gpu) / 1000
    df['PKG'] = df['PKG'].cumsum()
    df['GPU'] = df['GPU'].cumsum()
    df['TOTAL'] = df['GPU'] + df['PKG']
    return df

def transform_statistics(df: pd.DataFrame, domain:str, instance:str):
    if domain == "graph_coloring":
        df['best_of_gen'] = df['best_of_gen'].apply(lambda x: transform_gc_fitness(instance, x))
    return df

def preprocess_df(df):
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].iloc[0]
    return df

def add_gen_to_df(measures_df, gen_df):
    measures_df['type'] = 'MEASURE'
    gen_df['type'] = 'GEN'
    merged_df = pd.concat([measures_df, gen_df]).sort_values(by='time')
    merged_df['gen'] = merged_df['gen'].ffill().bfill() #filling empty gen entries of GPU
    # Split the merged_db into two DataFrames based on 'type'
    measure_df_split = merged_df[merged_df['type'] == 'MEASURE'].drop(columns=['type']).reset_index(drop=True)
    
    gen_df_split = merged_df[merged_df['type'] == 'GEN'].drop(columns=['type']).reset_index(drop=True)
    gened_measures_df_sorted = measure_df_split.sort_values('time', ascending=False)
    gened_measures_df = gened_measures_df_sorted.drop_duplicates(subset='gen')
    measure_df_split_single_gen = gened_measures_df.sort_values('gen').reset_index(drop=True)

    return measure_df_split_single_gen, gen_df_split



def parse_pinpoint(pinpoint_file:str, output_dir:str):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(pinpoint_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')

    for i, measure in enumerate(measures[1:]):
        lines = measure.strip().splitlines()
        header = ['time', 'PKG', 'GPU']
        data_lines = lines[1:] if lines[0].startswith("Run") else lines

        # Prepare data rows
        data_rows = [line.strip().split(',') for line in data_lines if line.strip()]

        # Write to CSV
        output_path = os.path.join(output_dir, f'pinpoint_{i}.csv')
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_rows)


def parse_statistics(statistics_file:str, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    lines = None
    with open(statistics_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')
    for i, measure in enumerate(measures):
        with open(os.path.join(output_dir, f'statistics_{i}.csv'), 'a') as f:
            f.write(measure.strip())
        

def group_df(df: pd.DataFrame, key_col: str):
    # Separate columns by type
    
    bool_cols = df.select_dtypes(bool).columns.tolist()
    other_cols = [c for c in df.columns if c not in bool_cols + [key_col]]

    # Build aggregation dictionary
    agg_dict = {col: list for col in bool_cols}
    agg_dict.update({col: 'mean' for col in other_cols})
    
    # Aggregate
    merged_measures_df = df.groupby('gen', as_index=False).agg(agg_dict)
    return merged_measures_df

def merge_files(measures_dir, statistics_dir, out_dir, domain, instance, base_pkg:float=0.0, base_gpu:float=0.0):
    measures = []
    statistics = []
    gened_measures = []
    gened_statistics = []
    
    # collecting measures dataframes
    for root, dirs, files in os.walk(measures_dir):
        for file in files:
            curr_df = pd.read_csv(os.path.join(root, file))
            curr_df = transform_measures(curr_df, base_pkg=base_pkg, base_gpu=base_gpu)
            measures.append(preprocess_df(curr_df))

    # collecting statistics dataframes
    for root, dirs, files in os.walk(statistics_dir):
        for file in files:
            curr_df = pd.read_csv(os.path.join(root, file))
            curr_df = transform_statistics(curr_df, domain=domain, instance=instance)
            statistics.append(preprocess_df(curr_df))
    
    if len(measures) != len(statistics):
        raise RuntimeError(f"The number of measures and statistics files must be the same\nstat:{len(statistics)}, mes: {len(measures)}")
    
    
    # adding gen column to each measures df based on corresponding statistics df
    for measure_df, statistics_df in zip(measures, statistics):
        gened_measures_df, gened_statistics_df = add_gen_to_df(measure_df, statistics_df)

        gen_mapping = gened_measures_df.set_index('gen')['TOTAL']
        gened_statistics_df['best_of_gen/TOTAL'] = gened_statistics_df['best_of_gen'] / gen_mapping.reindex(gened_statistics_df['gen']).values
        gened_statistics_df['best_of_gen/TOTAL'] = gened_statistics_df['best_of_gen/TOTAL'].ffill().bfill()
        # if (len(gened_statistics_df) != len(gened_measures_df)):
        #     print(f"stat num rows: {len(gened_statistics_df)}, mes num rows: {len(gened_measures_df)}")
        #     print(f"{gened_statistics_df['best_of_gen'].iloc[-1]}/{gened_measures_df['TOTAL'].iloc[-1]}={gened_statistics_df['best_of_gen/TOTAL'].iloc[-1]}")

        gened_measures.append(gened_measures_df)
        gened_statistics.append(gened_statistics_df)
        
    # concating
    all_measures_df = pd.concat(gened_measures).reset_index(drop=True)
    all_statistics_df = pd.concat(gened_statistics).reset_index(drop=True)
    all_statistics_df['best_of_gen'] = all_statistics_df['best_of_gen'].astype(float)

    
    # getting the std of columns
    measures_value_stds = {'PKG': 0, 'GPU': 0, 'MEMORY': 0, 'TOTAL': 0}
    statistics_value_stds = {'best_of_gen': 0, 'time': 0, 'best_of_gen/TOTAL': 0}

    grouped_mesures = all_measures_df.groupby('gen')
    grouped_statistics = all_statistics_df.groupby('gen')

    for i, col in enumerate(measures_value_stds):
        measures_value_stds[col] = grouped_mesures[col].std().reset_index().fillna(0)
        measures_value_stds[col].columns = ['gen', f'{col}_std']

    for i, col in enumerate(statistics_value_stds):
        statistics_value_stds[col] = grouped_statistics[col].std().reset_index().fillna(0)
        statistics_value_stds[col].columns = ['gen', f'{col}_std']

    # grouping by gen 
    all_statistics_df['TRAINED'] = all_statistics_df['TRAINED'].astype(bool)
    merged_measures_df = group_df(all_measures_df, 'gen').reset_index()
    merged_statistics_df = group_df(all_statistics_df, 'gen').reset_index()

    # adding stds
    final_measures_df = reduce(
            lambda curr_df, col: pd.merge(curr_df, measures_value_stds[col], on='gen', how='left'),                                 
            measures_value_stds.keys(),
            merged_measures_df)

    final_statistics_df = reduce(
            lambda curr_df, col: pd.merge(curr_df, statistics_value_stds[col], on='gen', how='left'),
            statistics_value_stds.keys(),
            merged_statistics_df)

    final_measures_df.to_csv(os.path.join(out_dir, 'mean_measures.csv'), index=False)
    final_statistics_df.to_csv(os.path.join(out_dir, 'mean_statistics.csv'), index=False)
    
    return final_measures_df, final_statistics_df

def get_baseline_stats(baseline_file: str):
    dfs = []
    mean_pkgs = []
    mean_gpus = []
    with open(baseline_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')
    header = ['time', 'PKG', 'GPU']

    for i, measure in enumerate(measures[1:]):
        lines = measure.strip().splitlines()
        data_lines = lines[1:] if lines[0].startswith("Run") else lines

        # Prepare data rows
        data_rows = [line.strip().split(',') for line in data_lines if line.strip()]

        # Turn to Pandas DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        dfs.append(df)
        mean_pkgs.append(df['PKG'].astype(float).mean())
        mean_gpus.append(df['GPU'].astype(float).mean())

    return dfs, mean(mean_pkgs), mean(mean_gpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help='The program must recive dir containing measures files')
    parser.add_argument('--baseline_dir', type=str,
                        help='The program could recive the baseline directory to be used')
    args = parser.parse_args()
    

    baseline_dir = args.baseline_dir
    base_gpu = 0
    base_pkg = 0
    statistics_dir = os.path.join(args.exp_dir, 'parsed_statistics')
    measures_dir = os.path.join(args.exp_dir, 'parsed_measures')
    os.makedirs(statistics_dir, exist_ok=True)
    os.makedirs(measures_dir, exist_ok=True)

    print(f"------ Parsing {args.exp_dir} ------")

    parse_statistics(os.path.join(args.exp_dir, 'statistics.csv'), statistics_dir)
    parse_pinpoint(os.path.join(args.exp_dir, 'raw.txt'), measures_dir)
    if baseline_dir:
        _, base_pkg, base_gpu = get_baseline_stats(os.path.join(baseline_dir, 'raw.txt'))

    domain = args.exp_dir.split('/')[-1].split('__')[0]
    instance = args.exp_dir.split('/')[-1].split('__')[1]
    merge_files(measures_dir, statistics_dir, args.exp_dir, domain, instance, base_pkg, base_gpu)
