import os
import pandas as pd
import sys

def main(experimets_path: str):
    totals = []
    bests = []
    names = []
    for root, dirs, files in os.walk(experimets_path):
        if('imgs' not in dirs): continue

        names.append(root.split('/')[-1])
        if('mean_measures.csv' in files):
            df = pd.read_csv(os.path.join(root, 'mean_measures.csv'))
            pkg, gpu = df.loc[df.index[-1], ['PKG', 'GPU']]
            totals.append(pkg + gpu)
        else:
            totals.append(-1)

        if('mean_statistics.csv' in files):
            df = pd.read_csv(os.path.join(root, 'mean_statistics.csv'))
            best = df.loc[df.index[-1], ['best_of_gen']]
            bests.append(best)
        else:
            bests.append(-1)

        zipped = [item for item in zip(names, bests, totals)]
        zipped = [str((item[0], float(item[1].values[0]), float(item[2]))) for item in zipped]
        print(len(zipped))
        return '\n'.join(zipped)






if __name__ == "__main__":
    assert len(sys.argv) == 2
    print(main(sys.argv[1]))
