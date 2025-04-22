import argparse

def main(pinpoint_file:str, output_dir:str):
    with open(pinpoint_file, 'r') as f:
        lines = f.readlines()
    
    new_measure_indices = [i for i, line in enumerate(lines) if line[0] == '#']
    if(len(new_measure_indices) == 0):
        with open(f'{output_dir}/measure_0.csv', 'a') as f:
            f.write('CPU,GPU\n')
            f.write(''.join(lines))
    else:
        measures = []
        for i in range(len(new_measure_indices)-1):
            measures.append(lines[new_measure_indices[i]+1:new_measure_indices[i+1]])
        measures.append(lines[new_measure_indices[-1]+1:])
        for i, measure in enumerate(measures):
            with open(f'{output_dir}/measure_{i}.csv', 'a') as f:
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