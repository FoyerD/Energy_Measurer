import argparse
import os
import csv

def main(pinpoint_file:str, output_dir:str):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(pinpoint_file, 'r') as f:
        lines = f.readlines()

    measures = ''.join(lines).split('###')

    for i, measure in enumerate(measures[1:]):
        lines = measure.strip().splitlines()
        header = ['time', 'PKG', 'GPU']
        data_lines = lines[1:] if lines[0].startswith("Run") else lines

        # Prepare data rows
        data_rows = [line.strip().split(',') for line in data_lines if line.strip()]

        # Write to CSV
        output_path = os.path.join(output_dir, f'pinpoint_{i}.csv')
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_rows)

        print(f"Wrote {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pinpoint_file', type=str,
                    help='The program must recive the output file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    args = parser.parse_args()

    main(pinpoint_file=args.pinpoint_file,
         output_dir=args.out_dir)