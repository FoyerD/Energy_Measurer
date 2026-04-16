import pandas as pd
import sys
import os
import tomllib
import numpy as np
from sympy import latex
import re
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations


unwanted_instances = ["BPP\\_195", "BPP\\_359", "BPP\\_360"]


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
        "st": training_scheduling,
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
- st (if applicable, else NaN)
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
    2. The Fitness for each configuration "{crossover_name}_{bs}_{st}".
    3. The Fitness_std for each configuration "{crossover_name}_{bs}_{st}".
    4. The MJ for each configuration "{crossover_name}_{bs}_{st}".
    5. The MJ_std for each configuration "{crossover_name}_{bs}_{st}".
    6. The Hours for each configuration "{crossover_name}_{bs}_{st}".
    7. The Hours_std for each configuration "{crossover_name}_{bs}_{st}".
    8. The MJ/Fitness for each configuration "{crossover_name}_{bs}_{st}".
    9. The MJ/Fitness_std for each configuration "{crossover_name}_{bs}_{st}".
    """

    df = df.copy().drop(columns=['domain_name'])  # drop domain_name as it's the same for all rows
    df['bs'] = pd.to_numeric(df['bs'], errors='coerce').fillna(0).astype(int)  # convert bs to int, treating NaN as 0
    df['st'] = pd.to_numeric(df['st'], errors='coerce').fillna(0)  # convert st to int, treating NaN as 0
    df['config'] = df.apply(lambda row: f"{row['crossover_name']}_{row['bs']}_{row['st']}", axis=1)

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

        metric, algo, batch, st = col.split("_")

        bs = int(batch)

        st = float(st)
        st = -100 if st == 0 else -st

        return (
            algo_order.get(algo, 99),
            bs,
            st,
            metric_order.get(metric, 99)
        )

    ordered_cols = sorted(df.columns, key=col_key)
    df = df[ordered_cols]
    return df


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    # cols = [col for col in df.columns if any(metric in col for metric in ["Fitness", "MJ", "MJ/Fitness"])]
    # df = df[["instance"] + cols]
    df = df[~df['instance'].isin(unwanted_instances)]
    return df

def std_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if any(metric in col for metric in ["Fitness", "MJ", "MJ/Fitness"]) and "std" in col]
    return df[["instance"] + cols]

def time_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if "Hours" in col]
    cols = [col for col in cols if "kpoint" in col or ("dnc_2048_0.0" in col and '1' not in col) or "dnc_2048_0.1" in col]
    return df[["instance"] + cols]
    




def get_latex_format(df, pivot=False):
    df = df.copy()
    df = df.round(2)
    bold_coords = set()

    # Define groups
    def filter_mean_col(columns, prefix):
        return [c for c in columns if c.startswith(prefix) and "std" not in c and "kpoint" not in c]
    def filter_std_col(columns, prefix):
        return [c for c in columns if c.startswith(prefix) and "std" in c and "kpoint" not in c]
    
    mj_cols = filter_mean_col(df.columns, "MJ_")
    fit_cols = filter_mean_col(df.columns, "Fitness")
    ratio_cols = filter_mean_col(df.columns, "MJ/Fitness")
    hours_cols = filter_mean_col(df.columns, "Hours")

    # --- Bolding Logic (Only applied to mean columns) ---
    for idx, row in df.iterrows():
        def mark_extremes(cols, find_max=True):
            if not cols: return
            vals = pd.to_numeric(row[cols], errors='coerce')
            if vals.isna().all(): return
            target = vals.max() if find_max else vals.min()            
            matches = vals[np.isclose(vals.astype(float), float(target), atol=1e-4)].index
            for col in matches:
                bold_coords.add((idx, col))
                df.at[idx, col] = f"\\textbf{{{row[col]:.2f}}}"

        mark_extremes(mj_cols, find_max=False)
        mark_extremes(fit_cols, find_max=True)
        mark_extremes(ratio_cols, find_max=True)
        mark_extremes(hours_cols, find_max=False)

    # Format numbers to strings to prepare for combination
    for col in df.columns:
        if col != "instance":
            df[col] = df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x
            )
        
    all_std_cols = df.columns[df.columns.str.contains("std")].tolist()
    if pivot:
        # --- COMBINE MEAN AND STD ---
        for std_col in all_std_cols:
            # Map "Metric-std_XYZ" to "Metric_XYZ"
            mean_col = std_col.replace("-std", "")
            
            if mean_col in df.columns:
                # Merge into: 0.26 ($\pm$ 0.00) or \textbf{0.26} ($\pm$ 0.00)
                df[mean_col] = df[mean_col].astype(str) + " ($\\pm$ " + df[std_col].astype(str) + ")"
        
        # Drop the std columns so they do not pivot into new rows
        df = df.drop(columns=all_std_cols)

        # --- PIVOT ---
        df_p = df.set_index("instance").T
        
        new_index = []
        metric_map = {"Fitness": "$f$", "MJ": "MJ", "MJ/Fitness": "$f/MJ$", "Hours": "t"}

        for full_name in df_p.index:
            # Safely unpack assuming naming like: MJ_dnc_512_0.0
            parts = full_name.split("_")
            raw_metric = parts[0]
            
            if "kpoint" in full_name:
                method_name = "One\\newline Point"
            elif "dnc" in full_name and len(parts) >= 4:
                op = parts[1]
                bs = parts[2]
                st = parts[3]
                if st == "0.0":
                    method_name = f"DNC\\newline bs={bs}"
                else:
                    method_name = f"DNC\\newline st={st}"        
            else:
                method_name = full_name
            
            metric_tex = metric_map.get(raw_metric, raw_metric)
            new_index.append((method_name, metric_tex))
        
        df_p.index = pd.MultiIndex.from_tuples(new_index)
        
        # Sort rows so they always appear as f, MJ, f/MJ
        df_p = df_p.reindex(["$f$", "MJ", "$f/MJ$", "t"], level=1)

        latex_str = df_p.to_latex(
            index=True,
            multirow=True,
            escape=False,
            column_format="ll" + "c" * len(df_p.columns)
        )
        
        # Cleanups
        latex_str = latex_str.replace("[t]", "")
        # Dynamically replace any \cline{} with \midrule, regardless of column count
        latex_str = re.sub(r"\\cline\{.*?\}", r"\\midrule", latex_str)
        latex_str = re.sub(r"\{\*\}", r"{1.5cm}", latex_str)

    else:
        latex_str = df.to_latex(index=False, escape=False)

    return latex_str, bold_coords

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


def wilcoxon_test(df):
    """
    Performs pairwise Wilcoxon signed-rank tests for all configurations 
    within each metric found in the DataFrame.
    """
    # 1. Identify mean columns (exclude 'instance' and '-std' columns)
    mean_cols = [c for c in df.columns if "-std" not in c and c != "instance"]
    
    # 2. Group columns by metric (the part before the first underscore)
    metric_groups = {}
    for col in mean_cols:
        metric_name = col.split('_')[0]
        if metric_name not in metric_groups:
            metric_groups[metric_name] = []
        metric_groups[metric_name].append(col)
    
    results = []

    # 3. Iterate through each metric and its associated configurations
    for metric, cols in metric_groups.items():
        # Generate all unique pairs of configurations for this metric
        for col_a, col_b in combinations(cols, 2):
            # Extract configuration details for cleaner reporting
            # (Removes the metric prefix from the comparison name)
            config_a = "_".join(col_a.split('_')[1:])
            config_b = "_".join(col_b.split('_')[1:])
            
            try:
                # Perform the paired test
                # zero_method='pratt' is used to handle identical values in small samples
                stat, p_val = wilcoxon(df[col_a], df[col_b], zero_method='pratt')
            except ValueError:
                # This occurs if all differences between pairs are zero
                stat, p_val = None, 1.0

            results.append({
                "Metric": metric,
                "Config_A": config_a,
                "Config_B": config_b,
                "Statistic": stat,
                "p-value": p_val,
                "Significant": p_val < 0.05 if p_val is not None else False
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: python collect_experiments.py <experiments_dir>"
    
    exps_path = sys.argv[1]
    pivot = "pivot" in sys.argv
    use_existing = "existing" in sys.argv
    domains = ['bpp', 'graph_coloring']

    if use_existing:
        parsed_dfs = {domain_name: pd.read_csv(f"{exps_path}/{domain_name}_results.csv") for domain_name in domains}
    else:
        dfs = get_data(exps_path, domains)
        parsed_dfs = {domain_name: sort_columns(pivot_df(df)) for domain_name, df in dfs.items()}
        for domain_name, df in parsed_dfs.items():
            df.to_csv(f"{exps_path}/{domain_name}_results.csv", index=False)

    for domain_name, df in parsed_dfs.items():
        print(f"-------------{domain_name}-------------")
        main_df = main_table(df)
        wilc_main = wilcoxon_test(main_df)
        # wilc_main.to_csv(f"main_wilcoxon_{domain_name}.csv")
        std_df = std_table(df)
        time_df = time_table(df)

        latex_main, bold_map = get_latex_format(main_df, pivot=pivot)
        if (not pivot):
            latex_std = get_latex_format_std(std_df, bold_map)
        latex_time, _ = get_latex_format(time_df)

        print("----main table   ---")
        print(latex_main)
        print("----wilcoxon test---")
        print(wilc_main.to_string())
        # print("----")
        # if (not pivot):
        #     print(latex_std)
        #     print("----")
        # print(latex_time)
