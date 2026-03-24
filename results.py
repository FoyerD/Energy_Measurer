
import pandas as pd
import sys
import os
import tomllib
import numpy as np
from sympy import latex
import parse




def extract_toml_fields(toml_path: str):
    """Extract relevant fields from the .toml file, handling different domains and crossovers."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    # --- Domain handling ---
    domain = data.get("domain", {})
    domain_args = domain.get("args", {})
    domain_name = domain.get("name", None)

    dataset_name = None
    if domain_name == "bpp":
        dataset_name = domain_args.get("dataset_name", None)
    elif domain_name == "graph_coloring":
        graph_path = domain_args.get("graph_path", "")
        dataset_name = os.path.splitext(os.path.basename(graph_path))[0] if graph_path else None

    # --- Crossover handling ---
    crossover = data.get("crossover", {})
    crossover_name = crossover.get("name", None)
    crossover_args = crossover.get("args", {})

    batch_size = np.nan
    training_scheduling = np.nan

    if crossover_name == "dnc":
        dnc_conf = crossover_args.get("dnc_config", {})
        batch_size = dnc_conf.get("batch_size", np.nan)
        training_scheduling = dnc_conf.get("fitness_epsilon", np.nan)

    return {
        "domain_name": domain_name,
        "instance": dataset_name.replace("_", "\_"),
        "crossover_name": crossover_name,
        "bs": batch_size,
        "ts": training_scheduling,
    }


def extract_csv_values(experiment_dir: str):
    """Extract total (PKG+GPU), GPU, PKG, best_of_gen and time from experiment CSVs."""
    mes_results = {}
    mes_stds = {}
    stat_results = {}
    stat_stds = {}

    mes_values = ['TOTAL', 'GPU', 'PKG']
    stat_values = ['best_of_gen', 'time', 'best_of_gen/TOTAL']

    mes_transform = {'TOTAL': lambda x: x / 10**6, 'GPU': lambda x: x / 10**6, 'PKG': lambda x: x / 10**6}
    stat_transform = {'time': lambda x: x / 60**2, 'best_of_gen/TOTAL': lambda x: x * 10**6}

    measures_path = os.path.join(experiment_dir, "mean_measures.csv")
    stats_path = os.path.join(experiment_dir, "mean_statistics.csv")

    if os.path.exists(measures_path):
        df = pd.read_csv(measures_path)
        for value in mes_values:
            std_col = f'{value}_std'

            if value in mes_transform.keys():
                df[value] = mes_transform[value](df[value])
                df[std_col] = mes_transform[value](df[std_col])

            mes_results[value] = float(df.loc[df.index[-1], value])
            mes_stds[value] = df.loc[df.index[-1], std_col]

            

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        for value in stat_values:
            std_col = f'{value}_std'

            if value in stat_transform.keys():
                df[value] = stat_transform[value](df[value])
                df[std_col] = stat_transform[value](df[std_col])

            stat_results[value] = float(df.loc[df.index[-1], value])
            stat_stds[value] = df.loc[df.index[-1], std_col]

    return mes_results, mes_stds, stat_results, stat_stds


def make_row(toml_info: dict, mes_results: dict, mes_stds: dict, stat_results: dict, stat_stds: dict):
    row = {
        **toml_info,
        "MJ": mes_results.get("TOTAL", np.nan),
        "MJ-std": mes_stds.get("TOTAL", np.nan),
        "Fitness": stat_results.get("best_of_gen", np.nan),
        "Fitness-std": stat_stds.get("best_of_gen", np.nan),
        "Hours": stat_results.get("time", np.nan),
        "Hours-std": stat_stds.get("time", np.nan),
        "MJ/Fitness": stat_results.get("best_of_gen/TOTAL", np.nan),
        "MJ/Fitness-std": stat_stds.get("best_of_gen/TOTAL", np.nan),
        "GPU": mes_results.get("GPU", np.nan),
        "GPU-std": mes_stds.get("GPU", np.nan),
        "PKG": mes_results.get("PKG", np.nan),
        "PKG-std": mes_stds.get("PKG", np.nan),
    }
    return row

def get_data(experiments_path: str, domains: list[str]) -> dict[str, pd.DataFrame]: 
    """
Returns a dict with dfs, each row corresponding to an experiment, and columns for:
- domain_name
- instance
- crossover_name
- bs (if applicable, else NaN)
- ts (if applicable, else NaN)
- MJ
- MJ-std
- Fitness
- Fitness-std
- Hours
- Hours-std
- MJ/Fitness
- MJ/Fitness-std
- GPU
- GPU-std
- PKG
- PKG-std
    """
    df_rows = {domain: [] for domain in domains}
    for root, dirs, files in os.walk(experiments_path):
        toml_files = [f for f in files if f.endswith(".toml")]
        if len(toml_files) != 1:
            continue  # only consider directories with exactly one .toml file

        toml_path = os.path.join(root, toml_files[0])
        toml_info = extract_toml_fields(toml_path)
        domain_name = toml_info['domain_name']
        if domain_name not in domains:
            continue  # skip if domain is not in the specified list
        mes_results, mes_stds, stat_results, stat_stds = extract_csv_values(root)
        row = make_row(toml_info, mes_results, mes_stds, stat_results, stat_stds)
        df_rows[domain_name].append(row)


    dfs = {domain_name: pd.DataFrame(df_rows[domain_name]) for domain_name in df_rows if len(df_rows[domain_name]) > 0}
    return dfs

    
def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a df of multiple experiments of the same domain (not instances), pivot it so each row is for an instance, and the columns are:
    1. Instance name.
    2. The Fitness for each configuration "{crossover_name}_{bs}_{ts}".
    3. The Fitness_std for each configuration "{crossover_name}_{bs}_{ts}".
    4. The MJ for each configuration "{crossover_name}_{bs}_{ts}".
    5. The MJ_std for each configuration "{crossover_name}_{bs}_{ts}".
    6. The Hours for each configuration "{crossover_name}_{bs}_{ts}".
    7. The Hours_std for each configuration "{crossover_name}_{bs}_{ts}".
    8. The MJ/Fitness for each configuration "{crossover_name}_{bs}_{ts}".
    9. The MJ/Fitness_std for each configuration "{crossover_name}_{bs}_{ts}".
    """

    df = df.copy().drop(columns=['domain_name'])  # drop domain_name as it's the same for all rows
    df['bs'] = pd.to_numeric(df['bs'], errors='coerce').fillna(0).astype(int)  # convert bs to int, treating NaN as 0
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce').fillna(0)  # convert ts to int, treating NaN as 0
    df['config'] = df.apply(lambda row: f"{row['crossover_name']}_{row['bs']}_{row['ts']}", axis=1)

    pivoted = df.pivot(index='instance', columns='config', values=['Fitness', 'Fitness-std', 'MJ', 'MJ-std', 'Hours', 'Hours-std', 'MJ/Fitness', 'MJ/Fitness-std', 'GPU', 'GPU-std', 'PKG', 'PKG-std'])
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted.reset_index(inplace=True)
    return pivoted


# functions for displaying results in paper
def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_order = {
        "Fitness": 0,
        "MJ": 1,
        "MJ/Fitness": 2,
        "Hours": 3,
        "Fitness-std": 4,
        "MJ-std": 5,
        "MJ/Fitness-std": 6,
        "Hours-std": 7,
        "GPU": 8,
        "GPU-std": 9,
        "PKG": 10,
        "PKG-std": 11
    }

    algo_order = {
        "kpoint": 0,
        "dnc": 1
    }

    def col_key(col):
        if col == "instance":
            return (-1, 0, 0, 0)  # force first

        metric, algo, batch, ts = col.split("_")

        bs = int(batch)

        ts = float(ts)
        ts = -100 if ts == 0 else -ts

        return (
            algo_order.get(algo, 99),
            bs,
            ts,
            metric_order.get(metric, 99)
        )

    ordered_cols = sorted(df.columns, key=col_key)
    df = df[ordered_cols]
    return df


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if any(metric in col for metric in ["Fitness", "MJ", "MJ/Fitness"]) and "std" not in col]
    return df[["instance"] + cols]

def std_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if any(metric in col for metric in ["Fitness", "MJ", "MJ/Fitness"]) and "std" in col]
    return df[["instance"] + cols]

def time_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if "Hours" in col]
    cols = [col for col in cols if "kpoint" in col or ("dnc_2048_0.0" in col and '1' not in col) or "dnc_2048_0.1" in col]
    return df[["instance"] + cols]
    



def get_latex_format(df):
    df = df.copy()
    df = df.round(2)
    bold_coords = set()

    mj_cols = [c for c in df.columns if c.startswith("MJ_") and "kpoint" not in c]
    time_cols = [c for c in df.columns if c.startswith("Hours_") and "kpoint" not in c]
    fit_cols = [c for c in df.columns if c.startswith("Fitness_")]
    ratio_cols = [c for c in df.columns if c.startswith("MJ/Fitness_") and "kpoint" not in c]

    for idx, row in df.iterrows():
        def mark_extremes(cols, find_max=True):
            vals = pd.to_numeric(row[cols], errors='coerce')
            vals_for_calc = vals.replace({np.nan: -np.inf if find_max else np.inf})
            target = vals_for_calc.max() if find_max else vals_for_calc.min()            
            matches = vals[vals.round(2) == round(target, 4)].index
            
            for col in matches:
                if pd.notna(row[col]):
                    bold_coords.add((idx, col))
                    df.at[idx, col] = f"\\textbf{{{row[col]}}}"

        mark_extremes(mj_cols, find_max=False)
        mark_extremes(time_cols, find_max=False)
        mark_extremes(fit_cols, find_max=True)
        mark_extremes(ratio_cols, find_max=True)

    for col in df.columns:
        if col != "instance":
            df[col] = df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x
            )

    return df.to_latex(index=False, escape=False), bold_coords

def get_latex_format_std(df_std, bold_coords):
    df_std = df_std.copy()

    for idx, row in df_std.iterrows():
        for col in df_std.columns:
            if col == "instance":
                continue
            
            val = row[col]
            formatted_val = f"{val:.2f}" if pd.notna(val) else "nan"
            
            if (idx, col.replace("-std", "")) in bold_coords:
                df_std.at[idx, col] = f"\\textbf{{{formatted_val}}}"
            else:
                df_std.at[idx, col] = formatted_val

    return df_std.to_latex(index=False, escape=False)
    

if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: python collect_experiments.py <experiments_dir>"
    
    exps_path = sys.argv[1]
    domains = ['bpp', 'graph_coloring'] # 'graph_coloring'
    if len(sys.argv) == 3:
        parsed_dfs = {domain: pd.read_csv(f"{exps_path}/{domain}_results.csv") for domain in domains}
    else:
        dfs = get_data(exps_path, domains)
        parsed_dfs = {domain_name: sort_columns(pivot_df(df)) for domain_name, df in dfs.items()}
        for domain_name, df in parsed_dfs.items():
            df.to_csv(f"{exps_path}/{domain_name}_results.csv", index=False)

    for domain_name, df in parsed_dfs.items():
        main_df = main_table(df)
        std_df = std_table(df)
        time_df = time_table(df)

        latex_main, bold_map = get_latex_format(main_df)
        latex_std = get_latex_format_std(std_df, bold_map)
        latex_time, _ = get_latex_format(time_df)

        print(f"-----{domain_name}-----")
        print(latex_main)
        print("----")
        print(latex_std)
        print("----")
        print(latex_time)
