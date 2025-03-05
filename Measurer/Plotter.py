import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Measurer.DfHelper as dfh 

class Plotter:
    def __init__(self, x_col, dbs: dict):
        self._x_col = x_col
        self._fig, self._ax_left = plt.subplots()
        self._ax_right = self._ax_left.twinx()
        self._axes = [self._ax_left, self._ax_right]
        self._dbs_dict = dbs  # Dictionary mapping names to DataFrames
        
    def add_marker(self, time: float, col, db_name: str, axes_n: int = 0, marker='o'):
        self._dbs_dict[db_name], timedif_col = dfh.add_time_diff(self._dbs_dict[db_name], time, col)
        marker_idx = self._dbs_dict[db_name][timedif_col].idxmin()
        marker_row = self._dbs_dict[db_name].loc[marker_idx]
        self._axes[axes_n].scatter(marker_row[self._x_col], marker_row[col], color='black', zorder=5, label=f'{time/60} min', marker=marker)
    
    def take_above(self, col, value, db_name: str):
        self._dbs_dict[db_name] = self._dbs_dict[db_name][self._dbs_dict[db_name][col] >= value]
    
    def take_below(self, col, value, db_name: str):
        self._dbs_dict[db_name] = self._dbs_dict[db_name][self._dbs_dict[db_name][col] <= value]
    
    def add_plot(self, col, db_name: str, axes_n: int = 0, color=None, label: str = None):
        self._axes[axes_n].plot(self._dbs_dict[db_name][self._x_col], self._dbs_dict[db_name][col], label=label if label else col, color=color)
    
    def add_groupby_max_plot(self, col, db_name: str, axes_n: int = 0, label: str = None, color=None):
        grouped = self._dbs_dict[db_name].groupby(self._x_col)[col].max()
        grouped.plot(kind='line', ax=self._axes[axes_n], label=label if label else col, color=color)
    
    def fill_between(self, col: str, db_name: str, dev:pd.Series, axes_n: int = 0, color=None):
        x = self._dbs_dict[db_name][self._x_col]
        y = self._dbs_dict[db_name][col]
        self._axes[axes_n].fill_between(x, y - dev, y + dev, color=color, alpha=0.5)
    
    def save_fig(self, path: str, title: str, x_labels: str, y_labels: str):
        for i, labels in enumerate(zip(x_labels, y_labels)):
            self._axes[i].set_xlabel(labels[0])
            self._axes[i].set_ylabel(labels[1])
        self._fig.suptitle(title)
        
        # Determine ticks
        num_ticks = 5
        ticks = np.linspace(self._dbs_dict[list(self._dbs_dict.keys())[0]]['gen'].min(),
                            self._dbs_dict[list(self._dbs_dict.keys())[0]]['gen'].max(),
                            num=num_ticks).tolist()
        ticks = list(map(int, sorted(set(ticks))))  # Convert to integers and remove duplicates
        
        plt.xticks(ticks)
        self._fig.legend()
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=300)
