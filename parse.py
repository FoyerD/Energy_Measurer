
import argparse
import csv
import os
from statistics import mean
import pandas as pd

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
    return measure_df_split, gen_df_split



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

        print(f"Wrote {output_path}")

def parse_statistics(statistics_file:str, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    lines = None
    with open(statistics_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')
    for i, measure in enumerate(measures):
        with open(os.path.join(output_dir, f'statistics_{i}.csv'), 'a') as f:
            f.write(measure)
        
        print(f"Wrote {os.path.join(output_dir, f'statistics_{i}.csv')}")


def merge_files(measures_dir, statistics_dir, out_dir, base_pkg:float=0.0, base_gpu:float=0.0):
    measures = []
    statistics = []
    
    gened_measures = []
    gened_statistics = []
    
    for root, dirs, files in os.walk(measures_dir):
        for file in files:
            curr_df = pd.read_csv(os.path.join(root, file))
            curr_df['PKG'] = ((curr_df['PKG'] - base_pkg) / 1000) * 0.25
            curr_df['GPU'] = ((curr_df['GPU'] - base_gpu) / 1000) * 0.25
            curr_df['PKG'] = curr_df['PKG'].cumsum()
            curr_df['GPU'] = curr_df['GPU'].cumsum()
            measures.append(preprocess_df(curr_df))
    for root, dirs, files in os.walk(statistics_dir):
        for file in files:
            curr_df = pd.read_csv(os.path.join(root, file))
            statistics.append(preprocess_df(curr_df))
    
    assert len(measures) == len(statistics), "The number of measures and statistics files must be the same"
    
    
    
    for measure_df, statistics_df in zip(measures, statistics):
        gened_measures_df, gened_statistics_df = add_gen_to_df(measure_df, statistics_df)
        gened_measures.append(gened_measures_df)
        gened_statistics.append(gened_statistics_df)
        
    all_measures_df = pd.concat(gened_measures).reset_index(drop=True)
    all_statistics_df = pd.concat(gened_statistics).reset_index(drop=True)
    print(all_statistics_df.head()['best_of_gen'])
    all_statistics_df['best_of_gen'] = all_statistics_df['best_of_gen'].astype(float)

    measures_value_stds = {'PKG': 0, 'GPU': 0, 'MEMORY': 0}
    statistics_value_stds = {'best_of_gen': 0}

    for i, col in enumerate(measures_value_stds):
        measures_value_stds[col] = all_measures_df.groupby('gen')[col].std().reset_index().fillna(0)
        measures_value_stds[col].columns = ['gen', f'{col}_std']

    for i, col in enumerate(statistics_value_stds):
        statistics_value_stds[col] = all_statistics_df.groupby('gen')[col].std().reset_index().fillna(0)
        statistics_value_stds[col].columns = ['gen', f'{col}_std']

    merged_measures_df = all_measures_df.groupby('gen').mean().reset_index()
    merged_statistics_df = all_statistics_df.groupby('gen').mean().reset_index()

    final_measures_df = pd.merge(merged_measures_df, measures_value_stds['PKG'], on='gen', how='left').fillna(0)
    final_measures_df = pd.merge(final_measures_df, measures_value_stds['GPU'], on='gen', how='left').fillna(0)
    final_measures_df = pd.merge(final_measures_df, measures_value_stds['MEMORY'], on='gen', how='left').fillna(0)
    final_statistics_df = pd.merge(merged_statistics_df, statistics_value_stds['best_of_gen'], on='gen', how='left').fillna(0)


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

    print(f'mean_pkgs: {mean(mean_pkgs)}')
    print(f'mean_gpus: {mean(mean_gpus)}')
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

    parse_statistics(os.path.join(args.exp_dir, 'statistics.csv'),
                     statistics_dir)
    parse_pinpoint(os.path.join(args.exp_dir, 'raw.txt'),
                   measures_dir)
    if baseline_dir:
        _, base_pkg, base_gpu = get_baseline_stats(os.path.join(baseline_dir, 'raw.txt'))

    merge_files(measures_dir,
                statistics_dir,
                args.exp_dir, base_pkg, base_gpu)