import pandas as pd

def convert_to_datetime(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col])
    return df

def add_seconds_passed(df: pd.DataFrame, col: str='time'):
    df['seconds_passed'] = (df[col] - df[col].min()).dt.total_seconds()
    return df

def add_time_diff(df: pd.DataFrame, time: float, col: str='seconds_passed'):
    timedif_col = f'time_diff_{time/60}'
    df[timedif_col] = (df[col] - time).abs()
    return df, timedif_col

def add_cumsum(df: pd.DataFrame, col: str, new_col: str):
    df[new_col] = df[col].cumsum()
    return df

def max_by_group(df: pd.DataFrame, group_col: str, col: str):
    max_df = df.groupby(group_col)[col].max().reset_index()
    
    final_df = pd.merge(df.drop(columns=[col]), max_df[[group_col, col]], on=group_col, how='right')
    final_df.columns = final_df.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
    return final_df

def mean_by_group(df: pd.DataFrame, group_col: str, col:str):
    mean_df = df.groupby(group_col)[col].mean().reset_index()
    
    final_df = pd.merge(df.drop(columns=[col]), mean_df[[group_col, col]], on=group_col, how='right')
    final_df.columns = final_df.columns.str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
    return final_df

    
def calculate_grouped_std(df, value_column, group_column):
    """
    Calculate the standard deviation of a value column grouped by another column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        value_column (str): Name of the column for which standard deviation is calculated
        group_column (str): Name of the column to group by
    
    Returns:
        pd.Series: Standard deviation of value_column for each group
    """
    return df.groupby(group_column)[value_column].std()