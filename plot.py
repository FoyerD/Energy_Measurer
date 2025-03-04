from os import walk
from sys import argv

import pandas as pd
from Measurer.Plotter import Plotter
import Measurer.DfHelper as dfh

def unzip(tuples):
    # Using zip() with * to unzip the list
    a, b = zip(*tuples)

    # Convert the results to lists
    a = list(a)
    b = list(b)
    
    return a, b

def preprocess_df(df):
    new_df = df.sort_values('time')
    new_df = dfh.convert_to_datetime(new_df, 'time')
    new_df = dfh.add_seconds_passed(new_df, col='time')
    return new_df

def add_gen_to_gpu_df(gpu_df, cpu_df):
    merged_df = pd.concat([gpu_df, cpu_df]).sort_values(by='seconds_passed')
    merged_df = merged_df.ffill().bfill() #filling empty gen entries of GPU
    # Split the merged_db into two DataFrames based on 'type'
    gpu_df_split = merged_df[merged_df['type'] == 'GPU'].reset_index(drop=True)
    cpu_df_split = merged_df[merged_df['type'] == 'CPU'].reset_index(drop=True)
    
    return cpu_df_split, gpu_df_split

def plot_dual_graph(cpu_dfs, gpu_dfs, statistics_dfs, output_dir, take_above:int=0, markers:list=None):
    if markers is None:
        markers = []

    new_cpu_dfs = [preprocess_df(df).sort_values(by='gen') for df in cpu_dfs]
    new_gpu_dfs = [dfh.add_cumsum(preprocess_df(df),col='measure', new_col='measure') for df in gpu_dfs]
    new_statistics_dfs = [preprocess_df(df).sort_values(by='gen') for df in statistics_dfs]
    
    new_cpu_dfs, new_gpu_dfs = unzip([add_gen_to_gpu_df(dfs[0], dfs[1]) for dfs in zip(new_cpu_dfs, new_gpu_dfs)])
    #new_cpu_dfs = [dfh.get_diff_col(df, 'measure', 'measure') for df in new_cpu_dfs]
    
    new_cpu_dfs = [dfh.max_by_group(df, 'gen', 'seconds_passed') for df in new_cpu_dfs]
    new_gpu_dfs = [dfh.max_by_group(df, 'gen', 'seconds_passed') for df in new_gpu_dfs]
    new_statistics_dfs = [dfh.max_by_group(df, 'gen', 'seconds_passed') for df in new_statistics_dfs]
    
    concat_cpu_df = pd.concat(new_cpu_dfs, ignore_index=True)
    concat_gpu_df = pd.concat(new_gpu_dfs, ignore_index=True)
    concat_statistics_df = pd.concat(new_statistics_dfs, ignore_index=True)
    
    final_cpu_df = dfh.mean_by_group(concat_cpu_df, 'gen', 'measure')
    final_gpu_df = dfh.mean_by_group(concat_gpu_df, 'gen', 'measure')
    final_statistics_df = dfh.mean_by_group(concat_statistics_df, 'gen', 'best_of_gen')
    
    plotter = Plotter(x_col='gen', dbs=[final_cpu_df, final_gpu_df, final_statistics_df])
    
    plotter.add_plot(col='measure', db_n=0, axes_n=0, label='cpu joules', color='red')
    plotter.add_plot(col='measure', db_n=1, axes_n=0, label='gpu joules', color='blue')
    plotter.take_above(col='best_of_gen', value=take_above, db_n=2)        
    plotter.add_plot(col='best_of_gen', db_n=2, axes_n=1, label='best of gen fitness', color='green')
    
    #plotter.add_std_dev(col='measure', db_n=0, axes_n=0, color='red')
    #plotter.add_std_dev(col='measure', db_n=1, axes_n=0, color='blue')
    #plotter.add_std_dev(col='best_of_gen', db_n=2, axes_n=1, color='green')
    
    for marker in markers:
        plotter.add_marker(time=marker['time'], col=marker['col'], axes_n=1, db_n=2, marker=marker['marker'])
    
    plotter.save_fig(path=f'{output_dir}/dual_plot.png', title='Measure/Statistics vs time', x_labels=['generation', 'generation'], y_labels=['joules', 'fitness'])

def read_dfs(output_dir):
    gpu_dfs = []
    cpu_dfs = []
    statistics_dfs = []
    
    for root, dirs, files in walk(output_dir):
        for name in dirs:
            curr_run_dir = f'{root}/{name}'
            cpu_df = pd.read_csv(f'{curr_run_dir}/cpu_measures.csv')
            gpu_df = pd.read_csv(f'{curr_run_dir}/gpu_measures.csv')
            statistics_df = pd.read_csv(f'{curr_run_dir}/statistics.csv')
            cpu_dfs.append(cpu_df)
            gpu_dfs.append(gpu_df)
            statistics_dfs.append(statistics_df)
    
    return cpu_dfs, gpu_dfs, statistics_dfs


def main(output_dir):
    cpu_dfs, gpu_dfs, statistics_dfs = read_dfs(output_dir)
    plot_dual_graph(cpu_dfs, gpu_dfs, statistics_dfs, output_dir)
    print(f'Finished plotting {job_id}, found at {output_dir}')



if __name__ == "__main__":
    job_id = str(argv[1])
    main(f'./code_files/energy_measurer/out_files/{job_id}')