import os
import sys
import tomllib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
        "instance": dataset_name.replace('BPP_', ''),
        "crossover_name": crossover_name,
        "bs": batch_size,
        "ts": training_scheduling,
    }


def extract_csv_values(experiment_dir: str):
    """Extract total (PKG+GPU), best_of_gen and time from experiment CSVs."""
    mes_results = {}
    mes_stds = {}
    stat_results = {}
    stat_stds = {}
    mes_values = ['TOTAL']
    stat_values = ['best_of_gen', 'time', 'best_of_gen/TOTAL']
    mes_transform = {'TOTAL': lambda x: x / 10**6}
    stat_transform = {'time': lambda x: x / 60**2}

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


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame so that for each 'instance',
    there is a separate column for each combination of 'bs' or 'ts'
    when crossover_name == 'dnc'.
    Produces columns like Fitness_bs512, MJ_bs512, Fitness_ts0.001, MJ_ts0.001, etc.
    """
    # Keep only DNC rows
    dnc_df = df.copy()

    # Create a 'setting' column describing each config
    def make_label(r):
        label_parts = []
        if pd.notna(r['bs']):
            label_parts.append(f"bs{int(r['bs'])}")
        if pd.notna(r['ts']):
            label_parts.append(f"st{r['ts']:.6g}")
        return "_".join(label_parts) if label_parts else "unknown"

    dnc_df['setting'] = dnc_df.apply(make_label, axis=1)

    # Pivot: create separate Fitness and MJ columns per setting
    flat = dnc_df.pivot_table(
        index='instance',
        columns='setting',
        values=['Fitness', 'MJ', 'Hours', 'Fit/MJ', 'Fitness_std', 'MJ_std', 'Hours_std', 'Fit/MJ_std'],
        aggfunc='first'
    )
    

    # Flatten multiindex columns
    flat.columns = [f"{metric}_{setting}" for metric, setting in flat.columns]
    flat = flat.reset_index()


    order = [
        "instance",

        # (optional) One Point crossover columns
        "Fitness_kpoint", "MJ_kpoint", "Hours_kpoint", "Fit/MJ_kpoint",

        # DNC bs variations
        "Fitness_unknown",   "MJ_unknown", "Hours_unknown", "Fit/MJ_unknown",
        "Fitness_bs512_st0", "MJ_bs512_st0", "Hours_bs512_st0", "Fit/MJ_bs512_st0",
        "Fitness_bs1024_st0", "MJ_bs1024_st0", "Hours_bs1024_st0", "Fit/MJ_bs1024_st0",
        "Fitness_bs2048_st0", "MJ_bs2048_st0", "Hours_bs2048_st0", "Fit/MJ_bs2048_st0",

        # DNC stability (st) variations
        "Fitness_bs2048_st0.1", "MJ_bs2048_st0.1", "Hours_bs2048_st0.1", "Fit/MJ_bs2048_st0.1",
        "Fitness_bs2048_st0.01", "MJ_bs2048_st0.01", "Hours_bs2048_st0.01", "Fit/MJ_bs2048_st0.01",
        "Fitness_bs2048_st0.001", "MJ_bs2048_st0.001", "Hours_bs2048_st0.001", "Fit/MJ_bs2048_st0.001",
    ] 

    # Keep only existing columns from `order`
    existing_cols = [c for c in order if c in flat.columns]
    
    return flat[existing_cols]

def format_df_stds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame so that for each 'instance',
    there is a separate column for each combination of 'bs' or 'ts'
    when crossover_name == 'dnc'.
    Produces columns like Fitness_bs512, MJ_bs512, Fitness_ts0.001, MJ_ts0.001, etc.
    """
    # Keep only DNC rows
    dnc_df = df.copy()

    # Create a 'setting' column describing each config
    def make_label(r):
        label_parts = []
        if pd.notna(r['bs']):
            label_parts.append(f"bs{int(r['bs'])}")
        if pd.notna(r['ts']):
            label_parts.append(f"st{r['ts']:.6g}")
        return "_".join(label_parts) if label_parts else "unknown"

    dnc_df['setting'] = dnc_df.apply(make_label, axis=1)

    # Pivot: create separate Fitness and MJ columns per setting
    flat = dnc_df.pivot_table(
        index='instance',
        columns='setting',
        values=['Fitness', 'MJ', 'Hours', 'Fitness_std', 'MJ_std', 'Hours_std', 'Fit/MJ', 'Fit/MJ_std'],
        aggfunc='first'
    )
    

    # Flatten multiindex columns
    flat.columns = [f"{metric}_{setting}" for metric, setting in flat.columns]
    flat = flat.reset_index()


    order = [
        "instance",

        # (optional) One Point crossover columns
        "Fitness_kpoint", "MJ_kpoint", "Hours_kpoint", "Fitness/MJ_kpoint",

        # DNC bs variations
        "Fitness_unknown",   "MJ_unknown", "Hours_unknown", "Fit/MJ_unknown",
        "Fitness_bs512_st0", "MJ_bs512_st0", "Hours_bs512_st0", "Fit/MJ_bs512_st0",
        "Fitness_bs1024_st0", "MJ_bs1024_st0", "Hours_bs1024_st0", "Fit/MJ_bs1024_st0",
        "Fitness_bs2048_st0", "MJ_bs2048_st0", "Hours_bs2048_st0", "Fit/MJ_bs2048_st0",

        # DNC stability (st) variations
        "Fitness_bs2048_st0.1", "MJ_bs2048_st0.1", "Hours_bs2048_st0.1", "Fit/MJ_bs2048_st0.1",
        "Fitness_bs2048_st0.01", "MJ_bs2048_st0.01", "Hours_bs2048_st0.01", "Fit/MJ_bs2048_st0.01",
        "Fitness_bs2048_st0.001", "MJ_bs2048_st0.001", "Hours_bs2048_st0.001", "Fit/MJ_bs2048_st0.001",
    ] 
    expanded_order = []
    for col in order:
        expanded_order.append(col)
        if col != "instance":  # skip instance
            expanded_order.append(col.replace("Fitness", "Fitness_std") if "Fitness" in col else
                                  col.replace("MJ", "MJ_std") if "MJ" in col else
                                  col.replace("Hours", "Hours_std"))

    # Keep only existing columns from `order`
    existing_cols = [c for c in expanded_order if c in flat.columns]
    # Add any leftover columns (to avoid losing data)
    remaining_cols = [c for c in flat.columns if c not in existing_cols]
    
    return flat[existing_cols]   


def parse_df(df: pd.DataFrame) -> str:
    df = df.round(2)
    mj_cols = [col for col in df.columns if 'MJ' in col and 'unknown' not in col and 'std' not in col]
    fit_cols = [col for col in df.columns if 'Fitness' in col and 'std' not in col]
    time_cols = [col for col in df.columns if 'Hours' in col and 'unknown' not in col and 'std' not in col]
    std_cols = [col for col in df.columns if 'std' in col]
    ratio_cols = [col for col in df.columns if 'Fit/MJ' in col and 'std' not in col]
    std_ratio_cols = [col for col in df.columns if 'Fit/MJ_std' in col]

    for idx, row in df.iterrows():
        # --- MIN, MJ ---
        mj_vals = row[mj_cols].replace({np.nan: np.inf})
        min_val = mj_vals.min()
        min_cols = mj_vals[mj_vals == min_val].index
        for col in min_cols:
            val = row[col]
            if pd.notna(val):
                df.at[idx, col] = f"\\textbf{{{val:.2f}}}"
    
        # --- MIN, Hours ---
        time_vals = row[time_cols].replace({np.nan: np.inf})
        min_val = time_vals.min()
        min_cols = time_vals[time_vals == min_val].index
        for col in min_cols:
            val = row[col]
            if pd.notna(val):
                df.at[idx, col] = f"\\textbf{{{val:.2f}}}"
    
        # --- MAX, Fitness ---
        fit_vals = row[fit_cols].replace({np.nan: -np.inf})  # fix: -inf for max
        max_val = fit_vals.max()
        max_cols = fit_vals[fit_vals == max_val].index
        for col in max_cols:
            val = row[col]
            if pd.notna(val):
                df.at[idx, col] = f"\\textbf{{{val:.2f}}}"

        # --- MAX, Fit/MJ ---
        ratio_vals = row[ratio_cols].replace({np.nan: -np.inf})
        max_val = ratio_vals.max()
        max_cols = ratio_vals[ratio_vals == max_val].index
        for col in max_cols:
            val = row[col]
            if pd.notna(val):
                df.at[idx, col] = f"\\textbf{{{val:.2f}}}"
   

    # --- Combine std values into base columns ---
    for std_col in std_cols:
        base_col = std_col.replace("_std", "")
        if base_col in df.columns:
            for idx, row in df.iterrows():
                val = row[base_col]
                std_val = row[std_col]
                if pd.notna(val) and pd.notna(std_val):
                    df.at[idx, base_col] = f"{val} ({std_val})"
                elif pd.notna(val):
                    df.at[idx, base_col] = f"{val}"
            # Optionally drop std column

    # print LaTeX table lines
    df = df.drop(columns=std_cols)
    print(df.columns)
    csv_str = ''
    for _, row in df.iterrows():
        vals = [f'{v:.2f}' if pd.notna(v) and (type(v) == type(0.0) or type(v) == type(0)) else v for v in row]
        vals = [str(v) if pd.notna(v) else "0" for v in vals]
        csv_str += " & ".join(vals) + " \\\\\n"
    return csv_str


def main(experiments_path: str) -> dict[str, pd.DataFrame]: 
    df_rows = {'bpp': [], 'graph_coloring': []}

    for root, dirs, files in os.walk(experiments_path):
        toml_files = [f for f in files if f.endswith(".toml")]
        if len(toml_files) != 1:
            continue  # only consider directories with exactly one .toml file

        toml_path = os.path.join(root, toml_files[0])
        toml_info = extract_toml_fields(toml_path)
        domain_name = toml_info['domain_name']
        #toml_info.pop('domain_name')
        mes_results, mes_stds, stat_results, stat_stds = extract_csv_values(root)
        row = {
            **toml_info,
            "MJ": mes_results['TOTAL'],
            "MJ_std": mes_stds['TOTAL'],
            "Fitness": stat_results['best_of_gen'],
            "Fitness_std": stat_stds['best_of_gen'],
            "Hours": stat_results['time'],
            "Hours_std": stat_stds['time'],
            "Fit/MJ": stat_results['best_of_gen/TOTAL'] if 'best_of_gen/TOTAL' in stat_results else np.nan,
            "Fit/MJ_std": stat_stds['best_of_gen/TOTAL'] if 'best_of_gen/TOTAL' in stat_stds else np.nan,
        }
        df_rows[domain_name].append(row)

    dfs = {domain_name: pd.DataFrame(df_rows[domain_name]) for domain_name in df_rows if len(df_rows[domain_name]) > 0}
    formated_dfs = {domain_name: format_df(dfs[domain_name]) for domain_name in dfs}
    formated_dfs_stds = {f'{domain_name}_std': format_df_stds(dfs[domain_name]) for domain_name in dfs}
    formated_dfs.update(formated_dfs_stds)
    return formated_dfs


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python collect_experiments.py <experiments_dir>"
    exps_path = sys.argv[1]
    dfs = main(exps_path)
    for domain_name, df in dfs.items():
        print(f"-----{domain_name}-----")
        print(parse_df(df))
        print()

