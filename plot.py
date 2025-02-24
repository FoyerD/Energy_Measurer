import sys

import pandas as pd
import Plotter

def main():
    csv_dir = './code_files/energy_measurer/out_files'
    take_above = 0
    markers = []
    cpu_db = pd.read_csv(csv_dir + f'/measures/cpu_{job_id}.csv')
    gpu_db = pd.read_csv(csv_dir + f'/measures/gpu_{job_id}.csv')
    statistics_db = pd.read_csv(csv_dir + f'/statistics/statistics_{job_id}.csv')
    
    merged_db = pd.concat([gpu_db, cpu_db]).sort_values(by='time')
    # Replace GPU measure by the cumulative sum of GPU measures
    merged_db.loc[merged_db['type'] == 'GPU', 'measure'] = merged_db.loc[
        merged_db['type'] == 'GPU', 'measure'
    ].cumsum() 
    merged_db = merged_db.bfill().ffill() #filling empty gen entries of GPU
    # Split the merged_db into two DataFrames based on 'type'
    gpu_db_split = merged_db[merged_db['type'] == 'GPU'].reset_index(drop=True)
    cpu_db_split = merged_db[merged_db['type'] == 'CPU'].reset_index(drop=True)

    plotter = Plotter(x_col='gen', dbs=[cpu_db_split, gpu_db_split, statistics_db])
    
    #plotter.take_above(col='average', value=take_above, db_n=2)
    #plotter.add_plot(col='average', db_n=2, axes_n=1, label='average fitness')
    
    plotter.take_above(col='best_of_gen', value=take_above, db_n=2)        
    plotter.add_plot(col='best_of_gen', db_n=2, axes_n=1, label='best of gen fitness')
    
    plotter.add_plot(col='measure', db_n=0, axes_n=0, label='cpu joules', color='red')
    plotter.add_groupby_max_plot(col='measure', db_n=1, axes_n=0, label='gpu joules', color='blue')
    
    for marker in markers:
        plotter.add_marker(time=marker['time'], col=marker['col'], axes_n=1, db_n=2, marker=marker['marker'])
    
    plotter.save_fig(path=f'{csv_dir}/plots/dual_{job_id}.png', title='Measure/Statistics vs time', x_labels=['generation', 'generation'], y_labels=['joules', 'fitness'])




if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()