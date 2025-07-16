import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import Utilities.DfHelper as dfh 
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class PlotUtils:
    def add_marker(ax:Axes ,df:DataFrame, value: float, col_to_diff:str, x_col:str, y_col:str, color=None, marker='o', label:str='marker'):
        df, dif_col = dfh.add_diff_col(df, value, col_to_diff)
        marker_idx = df[dif_col].idxmin()
        marker_row = df.loc[marker_idx]
        ax.scatter(marker_row[x_col], marker_row[y_col], color=color, zorder=5, label=dif_col, marker=marker)
    
    def add_vertical_line_cont(ax:plt.Axes, df:DataFrame, value: float, col_name:str, x_col:str, color=None, linestyle='--', label:str='marker'):
        df, timedif_col = dfh.add_diff_col(df, value, col_name)
        marker_idx = df[timedif_col].idxmin()
        marker_row = df.loc[marker_idx]
        ax.axvline(x=marker_row[x_col], color=color, linestyle=linestyle, label=timedif_col)
    
    
    def fill_between(ax:plt.Axes, df:DataFrame, x_col:str, y_col: str, dev:pd.Series, color=None):
        x = df[x_col]
        y = df[y_col]
        
        y_low = y - dev
        y_high = y + dev
        
        ax.fill_between(x, y_low, y_high, color=color, alpha=0.2)
    
    def save_fig(fig:Figure, axes:list[Axes], path: str, title: str, x_labels: str, y_labels: str, ticks=list[int]):
        for i, labels in enumerate(zip(x_labels, y_labels)):
            axes[i].set_xlabel(labels[0])
            axes[i].set_ylabel(labels[1])
        fig.suptitle(title)
    
        axes[0].set_xticks(ticks)
        fig.legend()
        fig.tight_layout()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(path, dpi=300)
    
    
    def add_vertical_line_cont(ax:plt.Axes, df:DataFrame, value, col_name:str, x_col:str, color='blue', linestyle='--', label:str='marker'):
        points = df.loc[df[col_name] == value, x_col]
        for total_val in points:
            ax.axvline(x=total_val, color=color, linestyle=linestyle, alpha=0.5)
