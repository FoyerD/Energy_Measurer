import argparse
import os

def main(statistics_file:str, output_dir:str):
    lines = None
    with open(statistics_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')
    for i, measure in enumerate(measures):
        with open(os.path.join(output_dir, f'statistics_{i}.csv'), 'a') as f:
            f.write(measure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('statistics_file', type=str,
                    help='The program must recive the output file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    args = parser.parse_args()

    main(statistics_file=args.statistics_file,
         output_dir=args.out_dir)