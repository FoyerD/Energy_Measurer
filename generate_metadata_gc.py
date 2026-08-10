import pandas as pd
from pathlib import Path
import networkx as nx
from ECkityFactory import get_gc_graph

def calculate_chromatic_number(G):
    """
    Attempts to find the chromatic number. 
    Note: For very large graphs, this may be computationally expensive.
    """
    # DSATUR is a high-quality heuristic that often finds the optimal 
    # for many benchmark instances.
    coloring = nx.coloring.greedy_color(G, strategy="DSATUR")
    return max(coloring.values()) + 1

def update_master_baselines(directory_path, master_csv="graph_baselines.csv", penalty=1000):
    base_path = Path(directory_path)
    master_path = base_path / master_csv
    
    if master_path.exists():
        master_df = pd.read_csv(master_path)
        # Ensure the column exists if the CSV was created by an older version
        if 'optimal_coloring' not in master_df.columns:
            master_df['optimal_coloring'] = None
    else:
        master_df = pd.DataFrame(columns=[
            'instance_name', 'nodes', 'edges', 'optimal_coloring', 'penalty_used', 'max_raw_possible'
        ])

    graph_files = [f for f in base_path.iterdir() if f.is_file() and f.suffix != '.csv' and not f.name.startswith('.')]
    
    new_rows = []

    for graph_file in graph_files:
        if graph_file.name in master_df['instance_name'].values:
            print(f"Skipping: {graph_file.name} (Already in master CSV)")
            continue

        try:
            print(f"Processing: {graph_file.name}...")
            
            # Load the graph
            G = get_gc_graph(str(graph_file))
            
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            # Calculate Optimal Coloring (Chromatic Number)
            print(f"Calculating optimal coloring for {graph_file.name}...")
            optimal_val = calculate_chromatic_number(G)

            # Your existing logic for max_raw
            max_raw = (n_edges * penalty) + n_nodes

            new_rows.append({
                'instance_name': graph_file.name,
                'nodes': n_nodes,
                'edges': n_edges,
                'optimal_coloring': optimal_val,
                'penalty_used': penalty,
                'max_raw_possible': max_raw
            })
            print(f"Successfully processed: {graph_file.name} (Chi: {optimal_val})")

        except Exception as e:
            print(f"Error processing {graph_file.name}: {e}")

    if new_rows:
        updated_df = pd.concat([master_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df.to_csv(master_path, index=False)
        print(f"\nDone! Master CSV updated at: {master_path}")
    else:
        print("\nNo new graphs found. Master CSV is up to date.")

if __name__ == "__main__":
    target_dir = "datasets_dnc/graph_coloring"
    update_master_baselines(target_dir)