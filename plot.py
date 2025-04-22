
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(pinpoint_file, output_dir):
    """
    Main function to parse the pinpoint file and generate plots.
    """

    # Read the pinpoint file
    df = pd.read_csv(pinpoint_file)
    print(df)

    # Create a directory for the plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a plot for each column in the dataframe
    for column in df.columns:
        plt.figure()
        df[column].plot()
        plt.title(column)
        plt.savefig(os.path.join(output_dir, f'{column}.png'))
        plt.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pinpoint_file', type=str,
                    help='The program must recive the output file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    args = parser.parse_args()

    main(pinpoint_file=args.pinpoint_file,
         output_dir=args.out_dir)