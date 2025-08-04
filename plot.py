import argparse
import os

from matplotlib import pyplot as plt
import pandas as pd
from math import inf

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
    

def plot_dual_graph(measures_df, statistics_df, output_dir:str, markers:list, name:str='dual_plot'):
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    axes = [ax1, ax1.twinx()]    
    
    axes[0].set_title('Energy Consumption')
    axes[0].set_ylabel('Jouls')
    axes[1].set_ylabel('Fitness')
    
    axes[0].plot(measures_df['gen'], measures_df['PKG'], color='red', label='PKG Jouls')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['PKG'] - measures_df['PKG_std'],
        measures_df['PKG'] + measures_df['PKG_std'],
        color='red', alpha=0.2
    )

    axes[0].plot(measures_df['gen'], measures_df['GPU'], color='blue', label='GPU Jouls')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['GPU'] - measures_df['GPU_std'],
        measures_df['GPU'] + measures_df['GPU_std'],
        color='blue', alpha=0.2
    )
    
    
    axes[0].plot(measures_df['gen'], measures_df['TOTAL'], color='purple', label='GPU Jouls')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['GPU'] - measures_df['GPU_std'],
        measures_df['GPU'] + measures_df['GPU_std'],
        color='purple', alpha=0.2
    )
    
    
    axes[1].plot(statistics_df['gen'], statistics_df['best_of_gen'], color='green', label='Best of Gen Fitness')
    axes[1].fill_between(
        statistics_df['gen'],
        statistics_df['best_of_gen'] - statistics_df['best_of_gen_std'],
        statistics_df['best_of_gen'] + statistics_df['best_of_gen_std'],
        color='green', alpha=0.2
    )

    points = statistics_df.loc[statistics_df['TRAINED'] == True, 'gen']
    for total_val in points:
        axes[0].axvline(x=total_val, color='blue', linestyle='--', alpha=0.5)


    # for marker in markers:
    #     plotter.add_marker(time=marker['time'], time_col='seconds_passed', col=marker['col'], axes_n=1, db_name='statistics')
    
    fig.legend(loc='upper left')
    fig.savefig(f'{output_dir}/svgs/{name}.svg')
    fig.savefig(f'{output_dir}/pngs/{name}.png')


def plot_memory_over_gen(measures_df, statistics_df, output_dir: str, name:str='memory_over_gen'):
    # Ensure TOTAL is available
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    
    # Merge dataframes on 'gen' to align TOTAL and best_of_gen
    merged_df = pd.merge(statistics_df, measures_df[['gen', 'TOTAL']], on='gen', how='inner')
    merged_df = merged_df.sort_values(by='gen')

    # Plot memory vs gen
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged_df['gen'],
        merged_df['MEMORY'],
        label='Memory Usage (KB)',
        color='orange'
    )
    
    # Optional: add confidence band using std
    if 'MEMORY_std' in statistics_df.columns:
        std_map = statistics_df.set_index('gen')['MEMORY_std']
        std_vals = merged_df['gen'].map(std_map).fillna(0)
        plt.fill_between(
            merged_df['gen'],
            merged_df['MEMORY'] - std_vals,
            merged_df['MEMORY'] + std_vals,
            color='orange',
            alpha=0.2,
            label='Std Dev'
        )
    
    trained_points = merged_df.loc[merged_df['TRAINED'] == 1.0, 'gen']
    for gen_val in trained_points:
        plt.axvline(x=gen_val, color='blue', linestyle='--', alpha=0.5)
    
    plt.xlabel('Generation')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage Over Generations')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{output_dir}/svgs/{name}.svg')
    plt.savefig(f'{output_dir}/pngs/{name}.png')
    plt.close()

def plot_statistics_over_total(measures_df, statistics_df, output_dir: str, markers, name:str='statistics_over_jouls'):
    # Ensure TOTAL is available
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    
    # Merge dataframes on 'gen' to align TOTAL and best_of_gen
    merged_df = pd.merge(statistics_df, measures_df[['gen', 'TOTAL']], on='gen', how='inner')
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
    
    trained_points = merged_df.loc[merged_df['TRAINED'] == 1.0, 'TOTAL']
    for total_val in trained_points:
        plt.axvline(x=total_val, color='blue', linestyle='--', alpha=0.5)
    
    
    plt.xlabel('TOTAL Energy (Joules)')
    plt.ylabel('Best of Gen Fitness')
    plt.title('Best Fitness vs Total Energy Consumed')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{output_dir}/svgs/{name}.svg')
    plt.savefig(f'{output_dir}/pngs/{name}.png')
    plt.close()


def plot_memory_over_jouls(measures_df, statistics_df, output_dir: str, name:str='memory_over_jouls'):
    # Ensure TOTAL is available
    measures_df['TOTAL'] = measures_df['PKG'] + measures_df['GPU']
    
    # Merge dataframes on 'gen' to align TOTAL and MEMORY
    merged_df = pd.merge(statistics_df, measures_df[['gen', 'TOTAL']], on='gen', how='inner')
    merged_df = merged_df.sort_values(by='TOTAL')

    # Plot memory vs TOTAL
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged_df['TOTAL'],
        merged_df['MEMORY'],
        label='Memory Usage (KB) vs TOTAL Energy',
        color='orange'
    )
    
    # Optional: add confidence band using std
    if 'MEMORY_std' in statistics_df.columns:
        std_map = statistics_df.set_index('gen')['MEMORY_std']
        std_vals = merged_df['gen'].map(std_map).fillna(0)
        plt.fill_between(
            merged_df['TOTAL'],
            merged_df['MEMORY'] - std_vals,
            merged_df['MEMORY'] + std_vals,
            color='orange',
            alpha=0.2,
            label='Std Dev'
        )
    
    trained_points = merged_df.loc[merged_df['TRAINED'] == True, 'TOTAL']
    for total_val in trained_points:
        plt.axvline(x=total_val, color='blue', linestyle='--', alpha=0.5)
    
    plt.xlabel('TOTAL Energy (Joules)')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs Total Energy Consumed')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{output_dir}/svgs/{name}.svg')
    plt.savefig(f'{output_dir}/pngs/{name}.png')
    plt.close()

def main(measures_file:str, statistics_file:str, output_dir:str, over_energy:bool=False, min_gen:int=-inf, max_gen:int=inf):
    measures_df = pd.read_csv(measures_file)
    statistics_df = pd.read_csv(statistics_file)
    statistics_df = statistics_df[statistics_df['best_of_gen'] > 0]

    measures_df = measures_df[measures_df['gen'] <= max_gen][measures_df['gen'] > min_gen]
    statistics_df = statistics_df[statistics_df['gen'] <= max_gen][statistics_df['gen'] > min_gen]
    markers = [
            # {'time': 0, 'col': 'best_of_gen'},
            # {'time': 60*5, 'col': 'best_of_gen'},
            # {'time': 60*10, 'col': 'best_of_gen'},
            # {'time': 60*15, 'col': 'best_of_gen'},
            # {'time': 60*20, 'col': 'best_of_gen'},
        ]
    plot_statistics_over_total(measures_df, statistics_df, output_dir, markers=markers, name=f'statistics_over_jouls_{min_gen}_to_{max_gen}')
    plot_memory_over_jouls(measures_df, statistics_df, output_dir, name=f'memory_over_jouls_{min_gen}_to_{max_gen}')
    plot_memory_over_gen(measures_df, statistics_df, output_dir, name=f'memory_over_gen_{min_gen}_to_{max_gen}')
    plot_dual_graph(measures_df, statistics_df, output_dir, markers=markers, name=f'dual_over_gen_{min_gen}_to_{max_gen}')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('measures_file', type=str,
                    help='The program must recive the measures file to be parsed')
    parser.add_argument('statistics_file', type=str,
                    help='The program must recive the statistics file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    parser.add_argument('--min_gen', type=int, default=0,
                    help='Minimum generation to consider in the plots')
    parser.add_argument('--max_gen', type=int, default=6000,
                    help='Maximum generation to consider in the plots')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path_svgs = os.path.join(args.out_dir, 'svgs')
    path_pngs = os.path.join(args.out_dir, 'pngs')
    os.makedirs(path_svgs, exist_ok=True)
    os.makedirs(path_pngs, exist_ok=True)
    main(measures_file=args.measures_file,
        statistics_file=args.statistics_file,
         output_dir=args.out_dir,
         min_gen=args.min_gen,
         max_gen=args.max_gen)