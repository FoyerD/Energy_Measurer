import os
import sys
import tomllib
import pandas as pd
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
    fitness_epsilon = np.nan
    k = np.nan
    arity = np.nan

    if crossover_name == "dnc":
        dnc_conf = crossover_args.get("dnc_config", {})
        batch_size = dnc_conf.get("batch_size", np.nan)
        fitness_epsilon = dnc_conf.get("fitness_epsilon", np.nan)
    elif crossover_name == "kpoint":
        k = crossover_args.get("k", np.nan)
        arity = crossover_args.get("arity", np.nan)

    return {
        "domain_name": domain_name,
        "dataset_name": dataset_name,
        "crossover_name": crossover_name,
        "batch_size": batch_size,
        "fitness_epsilon": fitness_epsilon,
        "k": k,
        "arity": arity,
    }


def extract_csv_values(experiment_dir: str):
    """Extract total (PKG+GPU) and best_of_gen from experiment CSVs."""
    total = np.nan
    best_of_gen = np.nan

    measures_path = os.path.join(experiment_dir, "mean_measures.csv")
    stats_path = os.path.join(experiment_dir, "mean_statistics.csv")

    if os.path.exists(measures_path):
        try:
            df = pd.read_csv(measures_path)
            pkg, gpu = df.loc[df.index[-1], ["PKG", "GPU"]]
            total = float(pkg + gpu)
        except Exception:
            pass

    if os.path.exists(stats_path):
        try:
            df = pd.read_csv(stats_path)
            best_of_gen = float(df.loc[df.index[-1], "best_of_gen"])
        except Exception:
            pass

    return total, best_of_gen


def group_by_dataset(df):
    """
    Split into two DataFrames: BPP and Graph Coloring.
    Group by dataset_name and take the last value for Fitness and MJ if multiple runs exist.
    """
    # Fill missing dataset_name with experiment name
    df["dataset_name"] = df["dataset_name"].fillna(df["experiment"])

    # Select relevant columns
    simple_df = df[["domain_name", "dataset_name", "best_of_gen", "total"]].copy()
    simple_df.rename(columns={"best_of_gen": "Fitness", "total": "MJ"}, inplace=True)

    # Group by domain_name and dataset_name
    grouped = simple_df.groupby(["domain_name", "dataset_name"], as_index=False).last()

    # Split into two separate DataFrames
    bpp_df = grouped[grouped["domain_name"] == "bpp"].copy()
    gc_df = grouped[grouped["domain_name"] == "graph_coloring"].copy()


    bpp_df.drop("domain_name", axis=1, inplace=True) 
    gc_df.drop("domain_name", axis=1, inplace=True) 

    return bpp_df, gc_df

def export_latex(df, filename):
    df.to_csv(filename, sep="&", index=False, header=False, lineterminator=" \\\\\n")
    print(f"Table written to {filename}")

def main(experiments_path: str):
    rows = []

    for root, dirs, files in os.walk(experiments_path):
        toml_files = [f for f in files if f.endswith(".toml")]
        if len(toml_files) != 1:
            continue  # only consider directories with exactly one .toml file

        toml_path = os.path.join(root, toml_files[0])
        toml_info = extract_toml_fields(toml_path)
        total, best_of_gen = extract_csv_values(root)

        row = {
            "experiment": os.path.basename(root),
            **toml_info,
            "total": total,
            "best_of_gen": best_of_gen,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    return df


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python collect_experiments.py <experiments_dir>"
    df = main(sys.argv[1])
    bpp_df, gc_df = group_by_dataset(df)

    export_latex(bpp_df, "bpp_table.txt")
    export_latex(gc_df, "graph_coloring_table.txt")

