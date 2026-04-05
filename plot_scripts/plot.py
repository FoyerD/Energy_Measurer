import argparse
import os

from matplotlib import pyplot as plt
import pandas as pd
from math import inf
import ast

known_formats = ['pdf', 'png', 'svg']

def unzip(tuples):
    a, b = zip(*tuples)

    # Convert the results to lists
    a = list(a)
    b = list(b)
    
    return a, b

def subtract_per_diff(df, avg, col, time_col='time'):
    df[col] -= avg * df[time_col].diff().fillna(0)
    return df 
    
# def add_trained_markers(df: pd.DataFrame, x_col: str, ax):
#     trained_points = df.loc[df['TRAINED'] > 0, [x_col, 'TRAINED']]
#     for _, row in trained_points.iterrows():
#         ax.axvline(x=row[x_col], color='blue', linestyle='--', alpha=row['TRAINED'])

def add_trained_markers(df: pd.DataFrame, x_col: str, ax, num_exps_to_plot: int = inf):
    # Assume each df['TRAINED'] is a list of same length
    n_experiments = len(df.iloc[0]['TRAINED'])
    cmap = plt.get_cmap('tab10', n_experiments)  # or use another colormap
    
    for i in range(min(n_experiments, num_exps_to_plot)):
        color = cmap(i)
        # get x values where this experiment was trained (True)
        trained_xs = df.loc[df['TRAINED'].apply(lambda lst: lst[i]), x_col]
        for x in trained_xs:
            ax.axvline(x=x, color=color, linestyle='--', alpha=0.5)

def plot_dual_graph(measures_df, statistics_df, output_dir:str, markers:list, name:str='dual_plot', num_exps_to_plot: int = inf, formats:list=['png']):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    axes = [ax1, ax1.twinx()]    
    
    # axes[0].set_title('Energy Consumption & Best Fitness')
    axes[0].set_xlabel('Generation', fontsize=20)
    axes[0].set_ylabel('Megajoules', fontsize=20)
    axes[1].set_ylabel('Fitness', fontsize=20)

    axes[0].tick_params(axis='both', labelsize=18)
    axes[1].tick_params(axis='both', labelsize=18)

    axes[0].plot(measures_df['gen'], measures_df['PKG'], color='red', label='PKG MJ')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['PKG'] - measures_df['PKG_std'],
        measures_df['PKG'] + measures_df['PKG_std'],
        color='red', alpha=0.2
    )

    axes[0].plot(measures_df['gen'], measures_df['GPU'], color='blue', label='GPU MJ')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['GPU'] - measures_df['GPU_std'],
        measures_df['GPU'] + measures_df['GPU_std'],
        color='blue', alpha=0.2
    )
    
    
    axes[0].plot(measures_df['gen'], measures_df['TOTAL'], color='purple', label='Total MJ')
    axes[0].fill_between(
        measures_df['gen'],
        measures_df['TOTAL'] - measures_df['TOTAL_std'],
        measures_df['TOTAL'] + measures_df['TOTAL_std'],
        color='purple', alpha=0.2
    )
    
    
    axes[1].plot(statistics_df['gen'], statistics_df['best_of_gen'], color='green', label='Best of Gen Fitness')
    axes[1].fill_between(
        statistics_df['gen'],
        statistics_df['best_of_gen'] - statistics_df['best_of_gen_std'],
        statistics_df['best_of_gen'] + statistics_df['best_of_gen_std'],
        color='green', alpha=0.2
    )

    add_trained_markers(statistics_df, 'gen', axes[0], num_exps_to_plot)


    # for marker in markers:
    #     plotter.add_marker(time=marker['time'], time_col='time', col=marker['col'], axes_n=1, db_name='statistics')
    
    # fig.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    for format in formats:
        fig.savefig(f'{output_dir}/{format}s/{name}.{format}')

def plot_statistics_over_time(measures_df, statistics_df, output_dir:str, markers:list, name:str='dual_plot', num_exps_to_plot: int = inf, formats:list=['png']):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    axes = [ax1, ax1.twinx()]    
    
    # axes[0].set_title('Energy Consumption & Best Fitness')
    axes[0].set_xlabel('Seconds', fontsize=20)
    axes[0].set_ylabel('Megajoules', fontsize=20)
    axes[1].set_ylabel('Fitness', fontsize=20)

    axes[0].tick_params(axis='both', labelsize=12)
    axes[1].tick_params(axis='both', labelsize=12)

    axes[0].plot(measures_df['time'], measures_df['PKG'], color='red', label='PKG MJ')
    axes[0].fill_between(
        measures_df['time'],
        measures_df['PKG'] - measures_df['PKG_std'],
        measures_df['PKG'] + measures_df['PKG_std'],
        color='red', alpha=0.2
    )

    axes[0].plot(measures_df['time'], measures_df['GPU'], color='blue', label='GPU MJ')
    axes[0].fill_between(
        measures_df['time'],
        measures_df['GPU'] - measures_df['GPU_std'],
        measures_df['GPU'] + measures_df['GPU_std'],
        color='blue', alpha=0.2
    )
    
    
    axes[0].plot(measures_df['time'], measures_df['TOTAL'], color='purple', label='Total MJ')
    axes[0].fill_between(
        measures_df['time'],
        measures_df['TOTAL'] - measures_df['TOTAL_std'],
        measures_df['TOTAL'] + measures_df['TOTAL_std'],
        color='purple', alpha=0.2
    )
    
    
    axes[1].plot(statistics_df['time'], statistics_df['best_of_gen'], color='green', label='Best of Gen Fitness')
    axes[1].fill_between(
        statistics_df['time'],
        statistics_df['best_of_gen'] - statistics_df['best_of_gen_std'],
        statistics_df['best_of_gen'] + statistics_df['best_of_gen_std'],
        color='green', alpha=0.2
    )

    add_trained_markers(statistics_df, 'time', axes[0], num_exps_to_plot)


    # for marker in markers:
    #     plotter.add_marker(time=marker['time'], time_col='time', col=marker['col'], axes_n=1, db_name='statistics')
    
    # fig.legend(loc='upper left')
    for format in formats:
        fig.savefig(f'{output_dir}/{format}s/{name}.{format}')


def plot_memory_over_gen(measures_df, statistics_df, output_dir: str, name:str='memory_over_gen', num_exps_to_plot: int = inf, formats:list=['png']):
    
    # Merge dataframes on 'gen' to align TOTAL and best_of_gen
    merged_df = pd.merge(statistics_df.drop(columns=['TOTAL']), measures_df[['gen', 'TOTAL']], on='gen', how='inner')
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
    
    add_trained_markers(merged_df, 'gen', plt, num_exps_to_plot)
    
    plt.xlabel('Generation')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage Over Generations')
    # plt.legend(loc='upper left')
    plt.grid(True)
    
    for format in formats:
        plt.savefig(f'{output_dir}/{format}s/{name}.{format}')
    plt.close()

def plot_statistics_over_total(measures_df, statistics_df, output_dir: str, markers, name:str='statistics_over_joules', num_exps_to_plot: int = inf, formats:list=['png']):
    # Merge dataframes on 'gen' to align TOTAL and best_of_gen
    merged_df = pd.merge(statistics_df.drop(columns=['TOTAL']), measures_df[['gen', 'TOTAL']], on='gen', how='inner')
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
    
    add_trained_markers(merged_df, 'TOTAL', plt, num_exps_to_plot) 
    
    plt.xlabel('TOTAL Energy (Joules)')
    plt.ylabel('Best of Gen Fitness')
    plt.title('Best Fitness vs Total Energy Consumed')
    # plt.legend()
    plt.grid(True)
    
    for format in formats:
        plt.savefig(f'{output_dir}/{format}s/{name}.{format}')
    plt.close()


def plot_memory_over_joules(measures_df, statistics_df, output_dir: str, name:str='memory_over_joules', num_exps_to_plot: int = inf, formats:list=['png']):
    
    # Merge dataframes on 'gen' to align TOTAL and MEMORY
    merged_df = pd.merge(statistics_df.drop(columns=['TOTAL']), measures_df[['gen', 'TOTAL']], on='gen', how='inner')
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
    
    add_trained_markers(merged_df, 'TOTAL', plt, num_exps_to_plot)

    plt.xlabel('TOTAL Energy (Joules)')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs Total Energy Consumed')
    # plt.legend()
    plt.grid(True)
    
    for format in formats:
        plt.savefig(f'{output_dir}/{format}s/{name}.{format}')
    plt.close()

    

def main(measures_file:str, statistics_file:str, output_dir:str, num_exps_to_plot: int, min_gen:int=-inf, max_gen:int=inf, formats:list=['png']):
    measures_df = pd.read_csv(measures_file)
    statistics_df = pd.read_csv(statistics_file)
    statistics_df = statistics_df[statistics_df['best_of_gen'] > 0][statistics_df['best_of_gen'] <= 100]
    statistics_df['TRAINED'] = statistics_df['TRAINED'].apply(ast.literal_eval)

    measures_df['PKG'] = measures_df['PKG'] / 1e6
    measures_df['PKG_std'] = measures_df['PKG_std'] / 1e6

    measures_df['GPU'] = measures_df['GPU'] / 1e6
    measures_df['GPU_std'] = measures_df['GPU_std'] / 1e6

    measures_df['TOTAL'] = measures_df['TOTAL'] / 1e6
    measures_df['TOTAL_std'] = measures_df['TOTAL_std'] / 1e6


    measures_df = measures_df[measures_df['gen'] <= max_gen][measures_df['gen'] > min_gen]
    statistics_df = statistics_df[statistics_df['gen'] <= max_gen][statistics_df['gen'] > min_gen]
    markers = [
            # {'time': 0, 'col': 'best_of_gen'},
            # {'time': 60*5, 'col': 'best_of_gen'},
            # {'time': 60*10, 'col': 'best_of_gen'},
            # {'time': 60*15, 'col': 'best_of_gen'},
            # {'time': 60*20, 'col': 'best_of_gen'},
        ]
    plot_statistics_over_total(measures_df, statistics_df, output_dir, markers=markers, name=f'statistics_over_joules_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)
    # plot_memory_over_joules(measures_df, statistics_df, output_dir, name=f'memory_over_joules_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)
    # plot_memory_over_gen(measures_df, statistics_df, output_dir, name=f'memory_over_gen_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)
    plot_dual_graph(measures_df, statistics_df, output_dir, markers=markers, name=f'dual_over_gen_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)
    # plot_statistics_over_time(measures_df, statistics_df, output_dir, markers=markers, name=f'statistics_over_time_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)
    # plot_ratio_over_gen(measures_df, statistics_df, output_dir, name=f'ratio_over_gen_{min_gen}_to_{max_gen}', num_exps_to_plot=num_exps_to_plot, formats=args.formats)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str,
                    help='The program must recive the measures file to be parsed')
    parser.add_argument('--num_exps_to_plot', type=int, default=inf,
                    help='The number of experiments to draw trained lines for')
    parser.add_argument('--min_gen', type=int, default=0,
                    help='Minimum generation to consider in the plots')
    parser.add_argument('--max_gen', type=int, default=6000,
                    help='Maximum generation to consider in the plots')
    parser.add_argument(
            '--formats', 
            nargs='+',
            type=str,
            help='List of formats (e.g., --formats pdf png jpeg)',
            default=['json'],
        )

    args = parser.parse_args()

    images_dir = os.path.join(args.experiment_dir, 'imgs')
    for format in args.formats:
        if format not in known_formats:
            raise ValueError(f"Unknown format '{format}'. Supported formats are: {', '.join(known_formats)}")
        else:
            path_format = os.path.join(images_dir, f'{format}s')
            os.makedirs(path_format, exist_ok=True)
    
    main(measures_file=os.path.join(args.experiment_dir, 'mean_measures.csv'),
         statistics_file=os.path.join(args.experiment_dir, 'mean_statistics.csv'),
         output_dir=images_dir,
         num_exps_to_plot=args.num_exps_to_plot,
         min_gen=args.min_gen,
         max_gen=args.max_gen,
         formats=args.formats)

