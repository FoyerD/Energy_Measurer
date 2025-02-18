import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, x_col, dbs: list):
        self._x_col = x_col
        self._fig, self._ax_left = plt.subplots()
        
        self._axes = [self._ax_left, self._ax_right]
        self._dbs = []
        
        for db in dbs:
            new_db = db.sort_values(by='time')
            new_db['time'] = pd.to_datetime(new_db['time'])
            new_db['time_diff'] = (new_db['time'] - new_db['time'].min()).dt.total_seconds()
            self._dbs.append(new_db)
        
    def add_marker(self, time:float, col, db_n: int=0, axes_n:int=0, marker='o'):
        timedif_col = f'time_diff_{time/60}'
        self._dbs[db_n][timedif_col] = (self._dbs[db_n]['time'] - time).abs()
        marker_idx = self._dbs[db_n][timedif_col].idxmin()
        marker_row = self._dbs[db_n].loc[marker_idx]
        self._axs[axes_n].scatter(marker_row[self._x_col], marker_row[col], color='black', zorder=5, label=f'{time/60} min', marker=marker)
    
    def take_above(self, col, value, db_n:int=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] >= value]
    
    def take_below(self, col, value, db_n:int=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] <= value]
    
    def add_plot(self, col, color=None, db_n:int=0, axes_n:int=0, label:str=None):
        self._axs[axes_n].plot(self._dbs[db_n][self._x_col], self._dbs[db_n][col], label=label if label is not None else col, color=color)
    
    def add_groupby_max_plot(self, col, db_n:int=0, axes_n:int=0, label:str=None):
        grouped = self._dbs[db_n].groupby(self._x_col)[col].max()
        grouped.plot(kind='line', ax=self._axes[axes_n], label=label if label is not None else col)
    
    def save_fig(self, path:str, title:str, x_labels:str, y_labels:str):
        for i, labels in enumerate(zip(x_labels, y_labels)):
            self.axs[i].set_xlabel(labels[0])
            self.axs[i].set_ylabel(labels[1])
        self._fig.suptitle(title)
        plt.xticks(self._db[self._x_col])
        self._fig.legend()
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=300)  # Adjust dpi for resolution