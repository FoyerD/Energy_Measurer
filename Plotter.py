import pandas as pd
import matplotlib.pyplot as plt
import sys

class Plotter:
    def __init__(self, x_col, db1, db2=None):
        self._x_col = x_col
        self._db1 = db1.sort_values(by='time')
        self._db2 = db2.sort_values(by='time') if db2 is not None else None
        self._fig, self._ax_left = plt.subplots()
        self._ax_right = self._ax_left.twinx() if db2 is not None else None
        
        self._axes = [self._ax_left, self._ax_right]
        self._dbs = [self._db1, self._db2]
        
        for db in self._dbs:
            db['time'] = pd.to_datetime(db['time'])
            db['time_diff'] = (db['time'] - db['time'].min()).dt.total_seconds()
        
    def add_marker(self, time, col, db_n=0,marker='o'):
        timedif_col = f'time_diff_{time/60}'
        self._dbs[db_n][timedif_col] = (self._dbs[db_n]['time'] - time).abs()
        marker_idx = self._dbs[db_n][timedif_col].idxmin()
        marker_row = self._dbs[db_n].loc[marker_idx]
        self._axs[db_n].scatter(marker_row[self._x_col], marker_row[col], color='black', zorder=5, label=f'{time/60} min', marker=marker)
    
    def take_above(self, col, value, db_n=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] >= value]
    
    def take_below(self, col, value, db_n=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] <= value]
    
    def add_plot(self, col, color, db_n=0, label=None):
        self._axs[db_n].plot(self._dbs[db_n][self._x_col], self._dbs[db_n][col], label=label if label is not None else col, color=color)
    
    def save_fig(self, path, title, x_labels, y_labels):
        for i, labels in enumerate(zip(x_labels, y_labels)):
            self.axs[i].set_xlabel(labels[0])
            self.axs[i].set_ylabel(labels[1])
        self._fig.suptitle(title)
        plt.xticks(self._db[self._x_col])
        self._fig.legend()
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=300)  # Adjust dpi for resolution