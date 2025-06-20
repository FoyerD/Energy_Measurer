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
    

def plot_dual_graph(measures_df, statistics_df, output_dir:str, markers:list):
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    dbs = {"MEASURES": measures_df, "STATISTICS": statistics_df}
    plotter = Plotter(x_col='gen', dbs=dbs)
    
    plotter.add_plot(col='PKG', db_name='MEASURES', axes_n=0, label='PKG Jouls', color='red')
    plotter.add_plot(col='GPU', db_name='MEASURES', axes_n=0, label='GPU Jouls', color='blue')
    plotter.add_plot(col='TOTAL', db_name='MEASURES', axes_n=0, label='Total Jouls', color='purple')
    plotter.add_plot(col='best_of_gen', db_name='STATISTICS', axes_n=1, label='Best of Gen Fitness', color='green')
    
    
    plotter.fill_between(col='PKG', db_name='MEASURES', axes_n=0, color='red', dev=measures_df['PKG_std'])
    plotter.fill_between(col='GPU', db_name='MEASURES', axes_n=0, color='blue', dev=measures_df['GPU_std'])
    plotter.fill_between(col='best_of_gen', db_name='STATISTICS', axes_n=1, color='green', dev=statistics_df['best_of_gen_std'])
    
    for marker in markers:
        plotter.add_marker(time=marker['time'], time_col='seconds_passed', col=marker['col'], axes_n=1, db_name='statistics')
    
    plotter.save_fig(
        path=f'{output_dir}/dual_plot.png',
        title='MEASURES/Statistics vs Time',
        x_labels=['Generation', 'Generation'],
        y_labels=['Jouls', 'Fitness']
    )

def plot_statistics_over_total(measures_df, statistics_df, output_dir: str, markers:list):
    # Ensure TOTAL is available
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    
    # Merge dataframes on 'gen' to align TOTAL and best_of_gen
    merged_df = pd.merge(statistics_df, measures_df[['gen', 'TOTAL']], on='gen', how='inner')
    
    # Sort by TOTAL (optional, for cleaner plots)
    merged_df = merged_df.sort_values(by='TOTAL')

    # Plot best_of_gen vs TOTAL
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged_df['TOTAL'],
        merged_df['best_of_gen'],
        label='Best of Gen Fitness vs TOTAL Energy',
        color='green'
    )
    
    # Optional: add confidence band using std
    if 'best_of_gen_std' in statistics_df.columns:
        std_map = statistics_df.set_index('gen')['best_of_gen_std']
        std_vals = merged_df['gen'].map(std_map).fillna(0)
        plt.fill_between(
            merged_df['TOTAL'],
            merged_df['best_of_gen'] - std_vals,
            merged_df['best_of_gen'] + std_vals,
            color='green',
            alpha=0.2,
            label='Std Dev'
        )
    
    plt.xlabel('TOTAL Energy (Joules)')
    plt.ylabel('Best of Gen Fitness')
    plt.title('Best Fitness vs Total Energy Consumed')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{output_dir}/statistics_over_total.png')
    plt.close()


def main(measures_file:str, statistics_file:str, output_dir:str, over_energy:bool=False):
    measures_df = pd.read_csv(measures_file)
    statistics_df = pd.read_csv(statistics_file)
    statistics_df = statistics_df[statistics_df['best_of_gen'] > 0]
    #TODO! REMOVE AFTER MEETING
    statistics_df = statistics_df[statistics_df['gen'] <= 2000]
    measures_df = measures_df[measures_df['gen'] <= 2000]
    markers = [
            # {'time': 0, 'col': 'best_of_gen'},
            # {'time': 60*5, 'col': 'best_of_gen'},
            # {'time': 60*10, 'col': 'best_of_gen'},
            # {'time': 60*15, 'col': 'best_of_gen'},
            # {'time': 60*20, 'col': 'best_of_gen'},
        ]
    if over_energy:
        plot_statistics_over_total(measures_df, statistics_df, output_dir, markers=markers)
    else:
        plot_dual_graph(measures_df, statistics_df, output_dir, markers=markers)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('measures_file', type=str,
                    help='The program must recive the measures file to be parsed')
    parser.add_argument('statistics_file', type=str,
                    help='The program must recive the statistics file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    parser.add_argument('--over_energy', action='store_true',
                    help='Indicate if plotting statistics over total energy consumed')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(measures_file=args.measures_file,
        statistics_file=args.statistics_file,
         output_dir=args.out_dir,
         over_energy=args.over_energy)