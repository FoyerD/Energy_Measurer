import pandas as pd


def convert_to_datetime(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col])
    return df

def add_seconds_passed(df: pd.DataFrame, col: str='time'):
    df['time_diff'] = df[col] - df[col].min()
    return df

def add_time_diff(df: pd.DataFrame, time: float, col: str='seconds_passed'):
    timedif_col = f'time_diff_{time/60}'
    df[timedif_col] = (df[col] - time).abs()
    return df, timedif_col

def add_cumsum(df: pd.DataFrame, col: str, new_col: str):
    df[new_col] = df[col].cumsum()
    return df