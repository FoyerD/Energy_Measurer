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
        "instance": dataset_name.replace('_', r'\_'),
        "crossover_name": crossover_name,
        "bs": batch_size,
        "ts": training_scheduling,
    }


def extract_csv_values(experiment_dir: str):
    """Extract total (PKG+GPU), best_of_gen and time from experiment CSVs."""
    total = np.nan
    best_of_gen = np.nan
    time = np.nan

    measures_path = os.path.join(experiment_dir, "mean_measures.csv")
    stats_path = os.path.join(experiment_dir, "mean_statistics.csv")

    if os.path.exists(measures_path):
        try:
            df = pd.read_csv(measures_path)
            pkg, gpu = df.loc[df.index[-1], ["PKG", "GPU"]]
            time = df.loc[df.index[-1], "time"]
            total = float(pkg + gpu)
        except Exception:
            pass

    if os.path.exists(stats_path):
        try:
            df = pd.read_csv(stats_path)
            best_of_gen = float(df.loc[df.index[-1], "best_of_gen"])
        except Exception:
            pass

    return total / 10**6, best_of_gen, time / 60**2


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
        values=['Fitness', 'MJ', 'Hours'],
        aggfunc='first'  # or 'mean', depending on how you want to combine duplicates
    )
    

    # Flatten multiindex columns
    flat.columns = [f"{metric}_{setting}" for metric, setting in flat.columns]
    flat = flat.reset_index()


    order = [
        "instance",

        # (optional) One Point crossover columns
        "Fitness_kpoint", "MJ_kpoint", "Hours_kpoint",

        # DNC bs variations
        "Fitness_unknown",   "MJ_unknown", "Hours_unknown",
        "Fitness_bs512_st0", "MJ_bs512_st0", "Hours_bs512_st0",
        "Fitness_bs1024_st0", "MJ_bs1024_st0", "Hours_bs1024_st0",
        "Fitness_bs2048_st0", "MJ_bs2048_st0", "Hours_bs2048_st0",

        # DNC stability (st) variations
        "Fitness_bs2048_st0.1", "MJ_bs2048_st0.1", "Hours_bs2048_st0.1",
        "Fitness_bs2048_st0.01", "MJ_bs2048_st0.01", "Hours_bs2048_st0.01",
        "Fitness_bs2048_st0.001", "MJ_bs2048_st0.001", "Hours_bs2048_st0.001",
    ] 

    # Keep only existing columns from `order`
    existing_cols = [c for c in order if c in flat.columns]
    # Add any leftover columns (to avoid losing data)
    remaining_cols = [c for c in flat.columns if c not in existing_cols]
    
    return flat[existing_cols + remaining_cols]

    


def parse_df(df: pd.DataFrame) -> str:
    df = df.round(3)
    mj_cols = [col for col in df.columns if 'MJ' in col and 'unknown' not in col]
    fit_cols = [col for col in df.columns if 'Fitness' in col]
    time_cols = [col for col in df.columns if 'Hours' in col and 'unknown' not in col]
    for idx, row in df.iterrows():
        # MIN, MJ
        mj_vals = row[mj_cols].replace({np.nan: np.inf})
        mj_col = mj_vals.idxmin()

        val = row[mj_col]
        if pd.notna(val):
            df.at[idx, mj_col] = f"\\textbf{{{val}}}"
        
        # MIN, Hours
        time_vals = row[time_cols].replace({np.nan: np.inf})
        time_col = time_vals.idxmin()

        val = row[time_col]
        if pd.notna(val):
            df.at[idx, time_col] = f"\\textbf{{{val}}}"

        # MAX, Fitness
        fit_vals = row[fit_cols].replace({np.nan: np.inf})
        fit_col = fit_vals.idxmax()

        val = row[fit_col]
        if pd.notna(val):
            df.at[idx, fit_col] = f"\\textbf{{{val}}}"


    # print LaTeX table lines
    print(df.columns)
    csv_str = ''
    for _, row in df.iterrows():
        vals = [str(v) if pd.notna(v) else "0" for v in row]
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
        total, best_of_gen, time = extract_csv_values(root)
        row = {
            **toml_info,
            "MJ": total,
            "Fitness": best_of_gen,
            "Hours": time
        }
        df_rows[domain_name].append(row)

    dfs = {domain_name: pd.DataFrame(df_rows[domain_name]) for domain_name in df_rows if len(df_rows[domain_name]) > 0}
    formated_dfs = {domain_name: format_df(dfs[domain_name]) for domain_name in dfs}
    return formated_dfs


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python collect_experiments.py <experiments_dir>"
    exps_path = sys.argv[1]
    dfs = main(exps_path)
    for domain_name, df in dfs.items():
        print(f"-----{domain_name}-----")
        print(parse_df(df))
        print()

