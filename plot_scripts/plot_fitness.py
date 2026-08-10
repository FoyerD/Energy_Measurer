import argparse
import os
import pandas as pd
from matplotlib import pyplot as plt
from math import inf

def plot_combined_fitness(experiment_dirs, labels, output_dir, min_gen=-inf, max_gen=inf, name='combined_fitness'):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # If labels aren't provided or don't match the number of directories, use folder names
    if not labels or len(labels) != len(experiment_dirs):
        labels = [os.path.basename(os.path.normpath(d)) for d in experiment_dirs]
        
    for d, label in zip(experiment_dirs, labels):
        stat_file = os.path.join(d, 'mean_statistics.csv')
        
        if not os.path.exists(stat_file):
            print(f"Warning: {stat_file} not found in {d}. Skipping.")
            continue
            
        df = pd.read_csv(stat_file)
        
        # Apply the same filtering rules as your original script
        df = df[(df['best_of_gen'] > 0) & (df['best_of_gen'] <= 100)]
        df = df[(df['gen'] >= min_gen) & (df['gen'] <= max_gen)]
        
        # Plot the main fitness line and grab its color
        label = '__'.join(label.split('__')[:2])  # Use only the first two parts of the label
        line = ax.plot(df['gen'], df['best_of_gen'], label=label, linewidth=2)[0]
        color = line.get_color()
        
        # Plot standard deviation band if the column exists
        if 'best_of_gen_std' in df.columns:
            ax.fill_between(
                df['gen'],
                df['best_of_gen'] - df['best_of_gen_std'],
                df['best_of_gen'] + df['best_of_gen_std'],
                color=color, 
                alpha=0.2
            )
            
    # Styling
    ax.set_xlabel('Generation', fontsize=18)
    ax.set_ylabel('Fitness (Best of Gen)', fontsize=18)
    ax.set_title('Fitness Comparison Across Experiments', fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, 'svgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pngs'), exist_ok=True)
    
    # Save the plot
    fig.savefig(os.path.join(output_dir, f'svgs/{name}.svg'))
    fig.savefig(os.path.join(output_dir, f'pngs/{name}.png'))
    print(f"Plot saved successfully to {output_dir}/pngs/{name}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot combined fitness over generations for multiple experiments.')
    
    # Accept one or more directories
    parser.add_argument('experiment_dirs', type=str, nargs='+',
                        help='Paths to the experiment directories (separated by spaces)')
    
    # Optional arguments
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for the legend (must match the number of directories provided)')
    parser.add_argument('--output_dir', type=str, default='imp_outs/combined_plots',
                        help='Directory to save the generated plots (default: imp_outs/combined_plots)')
    parser.add_argument('--min_gen', type=int, default=0,
                        help='Minimum generation to consider in the plots')
    parser.add_argument('--max_gen', type=int, default=6000,
                        help='Maximum generation to consider in the plots')
    
    args = parser.parse_args()
    
    plot_combined_fitness(
        experiment_dirs=args.experiment_dirs,
        labels=args.labels,
        output_dir=args.output_dir,
        min_gen=args.min_gen,
        max_gen=args.max_gen,
        name=f'combined_fitness_{args.min_gen}_to_{args.max_gen}'
    )