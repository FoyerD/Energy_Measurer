import pandas as pd
import matplotlib.pyplot as plt
import sys

class Plotter:
    def __init__(self, db, x_col, ):
        self._db = db.sort_values(by='time')
        self._x_col = x_col
        self._fig, self._ax = plt.subplots()
        
        self._db['time'] = pd.to_datetime(self._db['time'])
        self._db['time_diff'] = (self._db['time_diff'] - self._db['time_diff'].min()).dt.total_seconds()
    
    def add_marker(self, time, col, marker='o'):
        timedif_col = f'time_diff_{time/60}'
        self._db[timedif_col] = (self._db['time'] - time).abs()
        marker_idx = self._db[timedif_col].idxmin()
        marker_row = self.statistics_db.loc[marker_idx]
        self._ax.scatter(marker_row[self._x_col], marker_row[col], color='black', zorder=5, label='20 min', marker=marker)
    
    def take_above(self, col, value):
        self._db = self._db[self._db[col] >= value]
    
    def take_below(self, col, value):
        self._db = self._db[self._db[col] <= value]
    
    def add_plot(self, col, color, label=None):
        self._ax.plot(self._db[self._x_col], self._db[col], label=label if label is not None else col, color=color)
    
    def save_fig(self, path, title, x_label, y_label):
        self._ax.set_ylabel(x_label)
        self._ax.set_xlabel(y_label)
        self._fig.suptitle(title)
        plt.xticks(self._db[self._x_col])
        self._fig.legend()
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=300)  # Adjust dpi for resolution