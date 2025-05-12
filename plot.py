import argparse
from os import walk
import os
from sys import argv

from matplotlib import pyplot as plt
import pandas as pd
from Utilities.Plotter import Plotter
import Utilities.DfHelper as dfh

def unzip(tuples):
    # Using zip() with * to unzip the list
    a, b = zip(*tuples)

    # Convert the results to lists
    a = list(a)
    b = list(b)
    
    return a, b

def subtract_per_diff(df, avg, col, time_col='seconds_passed'):
    df[col] -= avg * df[time_col].diff().fillna(0)
    return df 
    

def plot_dual_graph(cpu_dfs, gpu_dfs, statistics_dfs, output_dir, markers:list=None, cpu_avg:float=0, gpu_avg:float=0):
    assert(len(cpu_dfs) == len(gpu_dfs) == len(statistics_dfs))
    if markers is None:
        markers = []
    new_cpu_dfs = [subtract_per_diff(dfh.get_diff_col(preprocess_df(df).sort_values(by='gen'), 'measure', 'measure'), avg=cpu_avg, col='measure', time_col='seconds_passed') for df in cpu_dfs]
    new_gpu_dfs = [dfh.add_cumsum(dfh.subtract_amount(preprocess_df(df), gpu_avg, 'measure').sort_values(by='gen'), col='measure', new_col='measure') for df in gpu_dfs]
    new_statistics_dfs = [preprocess_df(df).sort_values(by='gen') for df in statistics_dfs]
    
    new_cpu_dfs, new_gpu_dfs = unzip([add_gen_to_df(dfs[0], dfs[1]) for dfs in zip(new_cpu_dfs, new_gpu_dfs)])
    new_statistics_dfs = [df[df['best_of_gen'] >= 0] for df in new_statistics_dfs]
    
    new_cpu_dfs = [dfh.max_by_group(df, 'gen', 'time') for df in new_cpu_dfs]
    new_gpu_dfs = [dfh.max_by_group(df, 'gen', 'time') for df in new_gpu_dfs]
    new_statistics_dfs = [dfh.max_by_group(df, 'gen', 'time') for df in new_statistics_dfs]
    
    concat_cpu_df = pd.concat(new_cpu_dfs, ignore_index=True)
    concat_gpu_df = pd.concat(new_gpu_dfs, ignore_index=True)
    concat_statistics_df = pd.concat(new_statistics_dfs, ignore_index=True)
    
    cpu_std = dfh.calculate_grouped_std(concat_cpu_df, value_column='measure', group_column='gen')
    gpu_std = dfh.calculate_grouped_std(concat_gpu_df, value_column='measure', group_column='gen')
    statistics_std = dfh.calculate_grouped_std(concat_statistics_df, value_column='best_of_gen', group_column='gen')
    
    final_cpu_df = dfh.mean_by_group(concat_cpu_df.drop(columns='type'), group_col='gen', col='measure')
    final_gpu_df = dfh.mean_by_group(concat_gpu_df.drop(columns='type'), group_col='gen', col='measure')
    final_statistics_df = dfh.mean_by_group(concat_statistics_df, group_col='gen', col='best_of_gen')
    
    final_cpu_df = pd.merge(final_cpu_df, cpu_std, on='gen', how='left').fillna(0)
    final_gpu_df = pd.merge(final_gpu_df, gpu_std, on='gen', how='left').fillna(0)
    final_statistics_df = pd.merge(final_statistics_df, statistics_std, on='gen', how='left').fillna(0)
    final_dfs = {
        "CPU": final_cpu_df,
        "GPU": final_gpu_df,
        "statistics": final_statistics_df
    }
    
    plotter = Plotter(x_col='gen', dbs=final_dfs)
    
    plotter.add_plot(col='measure', db_name='CPU', axes_n=0, label='CPU joules', color='red')
    plotter.add_plot(col='measure', db_name='GPU', axes_n=0, label='GPU joules', color='blue')
    plotter.add_plot(col='best_of_gen', db_name='statistics', axes_n=1, label='Best of Gen Fitness', color='green')
    
    if(len(cpu_dfs) > 1):
        plotter.fill_between(col='measure', db_name='CPU', axes_n=0, color='red', dev=final_cpu_df['measure_std'])
        plotter.fill_between(col='measure', db_name='GPU', axes_n=0, color='blue', dev=final_gpu_df['measure_std'])
        plotter.fill_between(col='best_of_gen', db_name='statistics', axes_n=1, color='green', dev=final_statistics_df['best_of_gen_std'])
    
    for marker in markers:
        plotter.add_marker(time=marker['time'], time_col='seconds_passed', col=marker['col'], axes_n=1, db_name='statistics')
    
    plotter.save_fig(
        path=f'{output_dir}/dual_plot.png',
        title='Measure/Statistics vs Time',
        x_labels=['Generation', 'Generation'],
        y_labels=['Joules', 'Fitness']
    )

def read_dfs(output_dir):
    gpu_dfs = []
    cpu_dfs = []
    statistics_dfs = []
    
    for root, dirs, files in walk(output_dir):
        for name in dirs:
            curr_run_dir = f'{root}/{name}'
            cpu_df = pd.read_csv(f'{curr_run_dir}/cpu_measures.csv')
            cpu_dfs.append(cpu_df)
            statistics_df = pd.read_csv(f'{curr_run_dir}/statistics.csv')
            statistics_dfs.append(statistics_df)
            if os.path.exists(f'{curr_run_dir}/gpu_measures.csv'):
                gpu_df = pd.read_csv(f'{curr_run_dir}/gpu_measures.csv')
                gpu_dfs.append(gpu_df)
    
    return cpu_dfs, gpu_dfs, statistics_dfs


def main(measures_file:str, statistics_file:str, output_dir:str):
    measures_df = preprocess_df(pd.read_csv(measures_file))
    statistics_df = preprocess_df(pd.read_csv(statistics_file))
    os.mkdir(output_dir, exist_ok=True)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('measures_file', type=str,
                    help='The program must recive the measures file to be parsed')
    parser.add_argument('statistics_file', type=str,
                    help='The program must recive the statistics file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    args = parser.parse_args()

    main(measures_file=args.measures_file,
        statistics_file=args.statistics_file,
         output_dir=args.out_dir)