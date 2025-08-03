import pandas as pd
from pandas.core.frame import DataFrame
from numpy import linspace


def convert_to_datetime(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col])
    return df

def add_seconds_passed(df: pd.DataFrame, col: str='time', new_col:str='seconds_passed'):
    df[new_col] = (df[col] - df[col].min()).dt.total_seconds()
    df[new_col] = pd.to_numeric(df[new_col], errors='coerce')
    return df

def get_diff_col(df: pd.DataFrame, col: str, new_col:str):
    df[new_col] = (df[col] - df[col].min())
    return df
 
def subtract_amount(df: pd.DataFrame, amount: float, col: str):
    df[col] = df[col] - amount
    return df

def add_diff_col(df: pd.DataFrame, point: float, col_name: str='seconds_passed'):
    diff_col = f'{col_name}_diff_{point}'
    df[diff_col] = (df[col_name] - point).abs()
    return df, diff_col

def take_above(df: pd.DataFrame, col_name:str, value:float):
    df[col_name] = df[df[col_name] >= value]

def take_below(df: pd.DataFrame, col_name:str, value:float):
    df[col_name] = df[df[col_name] <= value]

def add_cumsum(df: pd.DataFrame, col: str, new_col: str):
    df[new_col] = df[col].cumsum()
    return df

def max_by_group(df: pd.DataFrame, group_col: str, col: str):
    return df.loc[df.groupby(group_col)[col].idxmax()]

def mean_by_group(df: pd.DataFrame, group_col: str):
    mean_df = df.groupby(group_col).mean().reset_index()
    return mean_df

    
def std_by_group(df, value_col, group_col):
    """
    Calculate the standard deviation of a value column grouped by another column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        value_column (str): Name of the column for which standard deviation is calculated
        group_column (str): Name of the column to group by
    
    Returns:
        pd.Series: Standard deviation of value_column for each group
    """
    stds = df.groupby(group_col)[value_col].std().reset_index().fillna(0)
    stds.columns = [group_col, f'{value_col}_std']
    return stds

def get_ticks(df:DataFrame, x_col:str, num_ticks:int=5):
    """
    Generate evenly spaced ticks for the x-axis based on the minimum and maximum values of a specified column.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The column name to base the ticks on.
        num_ticks (int): The number of ticks to generate.
    
    Returns:
        list: A list of tick values.
    """
    min_val = df[x_col].min()
    max_val = df[x_col].max()
    ticks = linspace(min_val, max_val, num=num_ticks).tolist()
    return list(map(int, sorted(set(ticks))))  # Convert to integers and remove duplicates
