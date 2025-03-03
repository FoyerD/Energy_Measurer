import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Measurer.DfHelper as dfh 

class Plotter:
    def __init__(self, x_col, dbs: list):
        self._x_col = x_col
        self._fig, self._ax_left = plt.subplots()
        self._ax_right = self._ax_left.twinx()
        self._axes = [self._ax_left, self._ax_right]
        self._dbs = dbs
        
    def add_marker(self, time:float, col, db_n: int=0, axes_n:int=0, marker='o'):
        self._dbs[db_n], timedif_col = dfh.add_time_diff(self._dbs[db_n], time, col)
        marker_idx = self._dbs[db_n][timedif_col].idxmin()
        marker_row = self._dbs[db_n].loc[marker_idx]
        self._axes[axes_n].scatter(marker_row[self._x_col], marker_row[col], color='black', zorder=5, label=f'{time/60} min', marker=marker)
    
    def take_above(self, col, value, db_n:int=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] >= value]
    
    def take_below(self, col, value, db_n:int=0):
        self._dbs[db_n] = self._dbs[db_n][self._dbs[db_n][col] <= value]
    
    def add_plot(self, col, color=None, db_n:int=0, axes_n:int=0, label:str=None):
        self._axes[axes_n].plot(self._dbs[db_n][self._x_col], self._dbs[db_n][col], label=label if label is not None else col, color=color)
    
    def add_groupby_max_plot(self, col, db_n:int=0, axes_n:int=0, label:str=None, color=None):
        grouped = self._dbs[db_n].groupby(self._x_col)[col].max()
        grouped.plot(kind='line', ax=self._axes[axes_n], label=label if label is not None else col, color=color)
    
    def add_std_dev(self, col:str, db_n:int=0, axes_n:int=0, color=None):
        dev = dfh.calculate_grouped_std(self._dbs[db_n], col, self._x_col)
        x = self._dbs[db_n][self._x_col]
        y = self._dbs[db_n][col]
        self._axes[axes_n].fill_between(x, y - dev, y + dev, color=color, alpha=0.2)
    
    def save_fig(self, path:str, title:str, x_labels:str, y_labels:str):
        for i, labels in enumerate(zip(x_labels, y_labels)):
            self._axes[i].set_xlabel(labels[0])
            self._axes[i].set_ylabel(labels[1])
        self._fig.suptitle(title)
        
        
        # Determine ticks
        num_ticks = 5
        # Get unique ticks
        ticks = np.linspace(self._dbs[0]['gen'].min(), self._dbs[0]['gen'].max(), num=num_ticks).tolist()
        # Convert ticks to integers
        ticks = list(map(int, ticks))
        # Remove duplicates and sort the ticks
        ticks = sorted(set(ticks))
        
        plt.xticks(ticks)
        self._fig.legend()
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=300)  # Adjust dpi for resolution
        