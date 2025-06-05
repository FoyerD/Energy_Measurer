import argparse
import csv
import os
from statistics import mean
import pandas as pd


def main(nothing_file: str):
    dfs = []
    mean_pkgs = []
    mean_gpus = []
    with open(nothing_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')
    header = ['time', 'PKG', 'GPU']
    
    for i, measure in enumerate(measures[1:]):
        lines = measure.strip().splitlines()
        data_lines = lines[1:] if lines[0].startswith("Run") else lines

        # Prepare data rows
        data_rows = [line.strip().split(',') for line in data_lines if line.strip()]

        # Turn to Pandas DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        dfs.append(df)
        mean_pkgs.append(df['PKG'].astype(float).mean())
        mean_gpus.append(df['GPU'].astype(float).mean())
        
    print(f'{mean(mean_pkgs):.2f} {mean(mean_gpus):.2f}')

        
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nothing_file', type=str,
                    help='The program must recive the output file to be parsed')
    args = parser.parse_args()

    main(nothing_file=args.nothing_file)