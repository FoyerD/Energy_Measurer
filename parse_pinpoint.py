import argparse
import os

def main(pinpoint_file:str, output_dir:str):
    lines = None
    with open(pinpoint_file, 'r') as f:
        lines = f.readlines()
    
    new_measure_indices = []
    new_measure_indices = [i for i, line in enumerate(lines) if line[0] == '#']
    if(len(new_measure_indices) == 0):
        with open(os.path.join(output_dir, 'measure_0.csv'), 'a') as f:
            f.write('CPU,GPU\n')
            f.write(''.join(lines))
    else:
        new_measure_indices.append(len(lines))
        measures = []
        for i, index in enumerate(new_measure_indices[:-1]):
            measures.append(lines[index+1:new_measure_indices[i+1]])
        for i, measure in enumerate(measures):
            with open(os.path.join(output_dir, f'measure_{i}.csv'), 'a') as f:
                f.write('CPU,GPU\n')
                f.write(''.join(measure))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pinpoint_file', type=str,
                    help='The program must recive the output file to be parsed')
    parser.add_argument('out_dir', type=str,
                    help='The program must recive the output directory to be used')
    args = parser.parse_args()

    main(pinpoint_file=args.pinpoint_file,
         output_dir=args.out_dir)