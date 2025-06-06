import pandas as pd

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

def add_time_diff(df: pd.DataFrame, time: float, col: str='seconds_passed'):
    timedif_col = f'time_diff_{time/60}'
    df[timedif_col] = (df[col] - time).abs()
    return df, timedif_col

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