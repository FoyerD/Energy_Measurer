import os
import sys

import pandas as pd
from Measurer.Measurer import Measurer
from Measurer.Plotter import Plotter
import Measurer.DfHelper as dfh
from plot import main as plot_dual_graph

def run_n_measures(n:int, operator:str, num_gens:int=100):
    measurers = []
    parent_output_dir = f"./code_files/energy_measurer/out_files/{job_id}"
    
    for i in range(n):
        output_dir = f"{parent_output_dir}/{i}"
        os.makedirs(output_dir, exist_ok=True)
        measurer = Measurer(job_id=job_id, output_dir=output_dir)
        measurers.append(measurer)
        if(operator == 'dnc'):
            measurer.setup_dnc(max_generation=num_gens, embedding_dim=64, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
        elif(operator == 'k_point'):
            measurer.setup_k_point_crossover(max_generation=num_gens, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
        else:
            raise ValueError(f'Operator {operator} not recognized')
        measurer.start_measure(prober_path="./code_files/energy_measurer/prob_nvsmi.py", write_each=5)
        measurer.save_measures()
        measurer.get_dual_graph(take_above=0, markers=[])#[{'time':5*60, 'marker':'o', 'col':'best_of_gen'}]
    
    plot_dual_graph(parent_output_dir, job_id)

    
def main():
    run_n_measures(n=5, operator=operator, num_gens=6000)

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    operator = str(sys.argv[2])
    main()