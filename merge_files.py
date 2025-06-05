
import argparse
import os
import pandas as pd
import Utilities.DfHelper as dfh

def preprocess_df(df, datetime:bool=False):
    new_df = df.sort_values('time')
    #TODO! Remove after meeting
    if not datetime:
        new_df['time'] = pd.to_datetime(new_df['time'], unit='s') + pd.Timedelta(hours=3)
    else:
        new_df['time'] = pd.to_datetime(new_df['time'])
    new_df = dfh.add_seconds_passed(new_df, col='time', new_col='seconds_passed')
    return new_df

def add_gen_to_df(measures_df, gen_df):
    measures_df['type'] = 'MEASURE'
    gen_df['type'] = 'GEN'
    merged_df = pd.concat([measures_df, gen_df]).sort_values(by='time')
    merged_df = merged_df.ffill().bfill() #filling empty gen entries of GPU
    # Split the merged_db into two DataFrames based on 'type'
    measure_df_split = merged_df[merged_df['type'] == 'MEASURE'].drop(columns=['type']).reset_index(drop=True)
    gen_df_split = merged_df[merged_df['type'] == 'GEN'].drop(columns=['type']).reset_index(drop=True)
    return measure_df_split, gen_df_split


def main(measures_dir, statistics_dir, out_dir, mdatetime:bool=False, sdatetime:bool=False, base_pkg:float=0.0, base_gpu:float=0.0):
    measures = []
    statistics = []
    
    gened_measures = []
    gened_statistics = []
    
    for root, dirs, files in os.walk(measures_dir):
        for file in files:
            if file.endswith('.csv'):
                curr_df = pd.read_csv(os.path.join(root, file))
                curr_df['PKG'] = (curr_df['PKG'] - base_pkg) / 1000 * 0.25
                curr_df['GPU'] = (curr_df['GPU'] - base_gpu) / 1000 * 0.25
                curr_df['PKG'] = curr_df['PKG'].cumsum()
                curr_df['GPU'] = curr_df['GPU'].cumsum()
                measures.append(preprocess_df(curr_df, mdatetime))
    for root, dirs, files in os.walk(statistics_dir):
        for file in files:
            if file.endswith('.csv'):
                curr_df = pd.read_csv(os.path.join(root, file))
                statistics.append(preprocess_df(curr_df, sdatetime))
    
    assert len(measures) == len(statistics), "The number of measures and statistics files must be the same"
    
    
    
    for measure_df, statistics_df in zip(measures, statistics):
        gened_measures_df, gened_statistics_df = add_gen_to_df(measure_df, statistics_df)
        gened_measures.append(gened_measures_df)
        gened_statistics.append(gened_statistics_df)
        
    all_measures_df = pd.concat(gened_measures).reset_index(drop=True)
    all_statistics_df = pd.concat(gened_statistics).reset_index(drop=True)
        
    pkg_std = dfh.std_by_group(all_measures_df, group_col='gen', value_col='PKG')
    gpu_std = dfh.std_by_group(all_measures_df, group_col='gen', value_col='GPU')
    bog_std = dfh.std_by_group(all_statistics_df, group_col='gen', value_col='best_of_gen')
    
    merged_measures_df = dfh.mean_by_group(all_measures_df, group_col='gen')
    merged_statistics_df = dfh.mean_by_group(all_statistics_df, group_col='gen')
    
    final_measures_df = pd.merge(merged_measures_df, pkg_std, on='gen', how='left').fillna(0)
    final_measures_df = pd.merge(final_measures_df, gpu_std, on='gen', how='left').fillna(0)
    final_statistics_df = pd.merge(merged_statistics_df, bog_std, on='gen', how='left').fillna(0)
    
    final_measures_df.to_csv(os.path.join(out_dir, 'mean_measures.csv'), index=False)
    final_statistics_df.to_csv(os.path.join(out_dir, 'mean_statistics.csv'), index=False)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('measures_dir', type=str,
                        help='The program must recive dir containing measures files')
    parser.add_argument('statistics_dir', type=str,
                        help='The program must recive dir containing statistics files')
    parser.add_argument('out_dir', type=str,
                        help='The program must recive the output directory to be used')
    parser.add_argument('--mdatetime', action='store_true',
                        help='Indcate if using datetime or timestamp')
    parser.add_argument('--sdatetime', action='store_true',
                        help='Indcate if using datetime or timestamp')
    parser.add_argument('--base_pkg', type=float, default=0.0,
                        help='Base PKG power in Watts')
    parser.add_argument('--base_gpu', type=float, default=0.0,
                        help='Base GPU power in Watts')
    args = parser.parse_args()
    measures_dir = args.measures_dir
    statistics_dir = args.statistics_dir
    output_dir = args.out_dir
    base_pkg = args.base_pkg
    base_gpu = args.base_gpu
    os.makedirs(output_dir, exist_ok=True)
    main(measures_dir, statistics_dir, output_dir, args.mdatetime, args.sdatetime, base_pkg, base_gpu)