
import pandas as pd
import sys
import os
import tomllib
import numpy as np

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
        "MJ_std": mes_stds.get("TOTAL", np.nan),
        "Fitness": stat_results.get("best_of_gen", np.nan),
        "Fitness_std": stat_stds.get("best_of_gen", np.nan),
        "Hours": stat_results.get("time", np.nan),
        "Hours_std": stat_stds.get("time", np.nan),
        "MJ/Fitness": stat_results.get("best_of_gen/TOTAL", np.nan),
        "MJ/Fitness_std": stat_stds.get("best_of_gen/TOTAL", np.nan),
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
- MJ_std
- Fitness
- Fitness_std
- Hours
- Hours_std
- MJ/Fitness
- MJ/Fitness_std
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
    df['config'] = df.apply(lambda row: f"{row['crossover_name']}_{row['bs']}_{row['ts']}", axis=1)

    pivoted = df.pivot(index='instance', columns='config', values=['Fitness', 'Fitness_std', 'MJ', 'MJ_std', 'Hours', 'Hours_std', 'MJ/Fitness', 'MJ/Fitness_std'])
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted.reset_index(inplace=True)
    return pivoted



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python collect_experiments.py <experiments_dir>"
    exps_path = sys.argv[1]

    domains = ['bpp', 'graph_coloring']
    dfs = get_data(exps_path, domains)

    parsed_dfs = {domain_name: pivot_df(df) for domain_name, df in dfs.items()}
    for domain_name, df in parsed_dfs.items():
        df.to_csv(f"{exps_path}/{domain_name}_results.csv", index=False)
        print(f"-----{domain_name}-----")
        print(df)
        print()