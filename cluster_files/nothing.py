from time import sleep
import datetime
import pandas as pd
import sys
from Utilities.Logger import Logger
from cluster_files.Measurer import Plotter
import Utilities.DfHelper as dfh

def get_nothing_avg(job_id, output_dir, total_time=0):
        cpu_file = f'{output_dir}/cpu_nothing.csv'
        gpu_file = f'{output_dir}/gpu_nothing.csv'
        sleep_time = 1
        print('Total time:', total_time)
        if total_time > 0:
            cpu_logger = Logger()
            cpu_logger.add_time_col()
            cpu_logger.add_cpu_measure_col(job_id)

            gpu_logger = Logger()
            gpu_logger.add_time_col()
            gpu_logger.add_gpu_measure_col()
            start_time = datetime.datetime.now()
            while (datetime.datetime.now() - start_time).seconds < total_time:
                cpu_logger.log()
                gpu_logger.log()
                sleep(sleep_time)

            cpu_df = cpu_logger.get_df()
            gpu_df = gpu_logger.get_df()
            cpu_logger.to_csv(cpu_file)
            gpu_logger.to_csv(gpu_file)
        else:
            cpu_df = pd.read_csv(cpu_file)
            gpu_df = pd.read_csv(gpu_file)
            gpu_df['time'] = pd.to_datetime(gpu_df['time'])  # Ensure it's datetime
            total_time = (gpu_df['time'].max() - gpu_df['time'].min()).total_seconds()
                
        gpu_df['measure'] = pd.to_numeric(gpu_df['measure'], errors='coerce')
        cpu_df['measure'] = pd.to_numeric(cpu_df['measure'], errors='coerce')

        gpu_avg = gpu_df['measure'].mean()
        cpu_avg = cpu_df.iloc[-1]['measure'] / total_time
        
        return cpu_avg, gpu_avg

def plot_nothing(job_id, output_dir):
    cpu_file = f'{output_dir}/cpu_nothing.csv'
    gpu_file = f'{output_dir}/gpu_nothing.csv'
    
    cpu_df = pd.read_csv(cpu_file)
    gpu_df = pd.read_csv(gpu_file)
    
    plotter = Plotter(dbs={'CPU': cpu_df, 'GPU': gpu_df}, x_col='time')
    gpu_df = dfh.convert_to_datetime(gpu_df, 'time')
    cpu_df = dfh.convert_to_datetime(cpu_df, 'time')
    gpu_df = dfh.add_cumsum(gpu_df, 'measure', 'measure')
    cpu_df = dfh.add_seconds_passed(cpu_df, 'time', 'time')
    gpu_df = dfh.add_seconds_passed(gpu_df, 'time', 'time')
    plotter.add_plot(col='measure', db_name='CPU', color='red', axes_n=0, label='CPU Energy Consumption')
    plotter.add_plot(col='measure', db_name='GPU', color='blue', axes_n=1, label='GPU Energy Consumption')
    plotter.save_fig(path=f'{output_dir}/nothing_plot.png', title='CPU/GPU Jouls vs Time', x_labels=['Time', 'Time'], y_labels=['Joules', 'Joules'])
    

def main(job_id, output_dir, total_time):
    cpu_avg, gpu_avg = get_nothing_avg(job_id, output_dir, total_time)
    print(f'CPU: {cpu_avg}')
    print(f'GPU: {gpu_avg}')
    plot_nothing(job_id, output_dir)
    return cpu_avg, gpu_avg
    


if __name__ == "__main__":
    job_id = sys.argv[1]
    output_dir = sys.argv[2]
    total_time = int(sys.argv[3])
    print('Total time:', total_time)
    main(job_id=job_id, output_dir=output_dir, total_time=total_time)