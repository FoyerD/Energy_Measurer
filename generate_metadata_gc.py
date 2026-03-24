import pandas as pd
import os
from pathlib import Path
from ECkityFactory import get_gc_graph

def update_master_baselines(directory_path, master_csv="graph_baselines.csv", penalty=1000):
    base_path = Path(directory_path)
    master_path = base_path / master_csv
    
    
    if master_path.exists():
        master_df = pd.read_csv(master_path)
    else:
        master_df = pd.DataFrame(columns=['instance_name', 'nodes', 'edges', 'penalty_used', 'max_raw_possible'])

    
    graph_files = [f for f in base_path.iterdir() if f.is_file() and f.suffix != '.csv']
    
    new_rows = []

    for graph_file in graph_files:
        
        if graph_file.name in master_df['instance_name'].values:
            print(f"Skipping: {graph_file.name} (Already in master CSV)")
            continue

        try:
            print(f"Processing: {graph_file.name}...")
            
            
            G = get_gc_graph(str(graph_file))
            
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            
            max_raw = (n_edges * penalty) + n_nodes

            
            new_rows.append({
                'instance_name': graph_file.name,
                'nodes': n_nodes,
                'edges': n_edges,
                'penalty_used': penalty,
                'max_raw_possible': max_raw
            })
            print(f"Successfully calculated: {graph_file.name}")

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
