import os
import sys

import pandas as pd
from Measurer.Measurer import Measurer
from Measurer.Plotter import Plotter
import Measurer.DfHelper as dfh


def run_n_measures(n:int, operator:str, num_gens:int=100):
    measurers = []
    parent_output_dir = f"./code_files/energy_measurer/out_files/{job_id}"
    gpu_dfs = []
    cpu_dfs = []
    statistics_dfs = []
    
    for i in range(n):
        output_dir = f"{parent_output_dir}/{i}"
        os.makedirs(output_dir, exist_ok=True)
        measurer = Measurer(job_id=job_id, output_dir=output_dir)
        measurers.append(measurer)
        measurer.setup_dnc(max_generation=num_gens, embedding_dim=64, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
        measurer.start_measure(prober_path="./code_files/energy_measurer/prob_nvsmi.py")
        measurer.save_measures()
        measurer.get_dual_graph(take_above=0, markers=[{'time':5*60, 'marker':'o', 'col':'best_of_gen'}])
        
        gpu_df = measurer.get_gpu_df()
        cpu_df = measurer.get_cpu_df()
        gpu_df = dfh.add_cumsum(gpu_df, 'measure', 'measure')
        merged_df = pd.concat([gpu_df, cpu_df]).sort_values(by='time')
        # Replace GPU measure by the cumulative sum of GPU measures
        merged_df.loc[merged_df['type'] == 'GPU', 'measure'] = merged_df.loc[
            merged_df['type'] == 'GPU', 'measure'
        ].cumsum() 
        merged_df = merged_df.bfill().ffill() #filling empty gen entries of GPU
        # Split the merged_db into two DataFrames based on 'type'
        gpu_df_split = merged_df[merged_df['type'] == 'GPU'].reset_index(drop=True)
        cpu_df_split = merged_df[merged_df['type'] == 'CPU'].reset_index(drop=True)
        
        gpu_df_max_time = gpu_df_split.groupby('gen')['time'].max().reset_index()
        cpu_df_max_time = cpu_df_split.groupby('gen')['time'].max().reset_index()
        
        gpu_df_split = pd.merge(gpu_df_split, gpu_df_max_time[['gen']], on='gen')
        cpu_df_split = pd.merge(cpu_df_split, cpu_df_max_time[['gen']], on='gen') 
        gpu_df_split.columns = gpu_df_split.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
        cpu_df_split.columns = cpu_df_split.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
 
        
        gpu_dfs.append(gpu_df_split)
        cpu_dfs.append(cpu_df_split)
        statistics_dfs.append(measurer.get_statistics_df()) 
    
    concat_cpu_df = pd.concat(cpu_dfs, ignore_index=True)
    concat_gpu_df = pd.concat(gpu_dfs, ignore_index=True)
    concat_statistics_df = pd.concat(statistics_dfs, ignore_index=True)
    
    cpu_df_mean_measure = concat_cpu_df.groupby('gen')['measure'].mean().reset_index()
    gpu_df_mean_measure = concat_gpu_df.groupby('gen')['measure'].mean().reset_index()
    statistics_df_mean_bog = concat_statistics_df.groupby('gen')['best_of_gen'].mean().reset_index()
    
    cpu_df_final = pd.merge(concat_cpu_df.drop(columns=['measure']), cpu_df_mean_measure[['gen', 'measure']], on='gen')
    gpu_df_final = pd.merge(concat_gpu_df.drop(columns=['measure']), gpu_df_mean_measure[['gen', 'measure']], on='gen')
    statistics_df_final = pd.merge(concat_statistics_df.drop(columns=['best_of_gen']), statistics_df_mean_bog[['gen', 'best_of_gen']], on='gen')

    cpu_df_final.columns = cpu_df_final.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
    gpu_df_final.columns = gpu_df_final.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
    statistics_df_final.columns = statistics_df_final.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)

    cpu_df_mean_measure['time'] = concat_cpu_df.groupby('gen')['time'].max().reset_index()['time']
    gpu_df_mean_measure['time'] = concat_gpu_df.groupby('gen')['time'].max().reset_index()['time']
    statistics_df_mean_bog['time'] = concat_statistics_df.groupby('gen')['time'].max().reset_index()['time']

    plotter = Plotter(x_col='gen', dbs=[cpu_df_mean_measure, gpu_df_mean_measure, statistics_df_mean_bog])
    
    #plotter.take_above(col='average', value=take_above, db_n=2)
    #plotter.add_plot(col='average', db_n=2, axes_n=1, label='average fitness')
    
    plotter.take_above(col='best_of_gen', db_n=2, value=0)
    
    plotter.add_plot(col='measure', db_n=0, axes_n=0, label='cpu joules', color='red')
    plotter.add_groupby_max_plot(col='measure', db_n=1, axes_n=0, label='gpu joules', color='blue')
    plotter.add_plot(col='best_of_gen', db_n=2, axes_n=1, label='best of gen fitness')
    
    
    plotter.save_fig(path=f'{parent_output_dir}/dual_plot.png', title='Measure/Statistics vs time', x_labels=['generation', 'generation'], y_labels=['joules', 'fitness'])

    
def main():
    run_n_measures(n=1, operator="dnc", num_gens=100)

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()