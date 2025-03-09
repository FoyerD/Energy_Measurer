import os
import sys

import argparse
from Measurer.Measurer import Measurer
from plot import main as plot_dual_graph

def run_n_measures(job_id:str, cross_op:str, mutation_op:str, domain:str, n_runs:int=1, n_gens:int=100):
    measurers = []
    parent_output_dir = f"./code_files/energy_measurer/out_files/{job_id}"
    
    for i in range(n_runs):
        output_dir = f"{parent_output_dir}/{i}"
        os.makedirs(output_dir, exist_ok=True)
        measurer = Measurer(job_id=job_id, output_dir=output_dir)
        measurers.append(measurer)
        if(cross_op == 'dnc'):
            measurer.setup_dnc(max_generation=n_gens, embedding_dim=64, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
        elif(cross_op == 'k_point'):
            measurer.setup_k_point_crossover(max_generation=n_gens, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
        else:
            raise ValueError(f'Operator {cross_op} not recognized')
        measurer.start_measure(prober_path="./code_files/energy_measurer/prob_nvsmi.py", write_each=5)
        measurer.save_measures()
        measurer.get_dual_graph(take_above=0, markers=[])#[{'time':5*60, 'marker':'o', 'col':'best_of_gen'}]
    
    plot_dual_graph(parent_output_dir, job_id)

    
def main(job_id:str, cross_op:str, mutation_op:str, domain:str, n_runs:int=1, n_gens:int=100):
    run_n_measures(n_runs=n_runs, cross_op=cross_op, mutation_op=mutation_op, domain=domain, n_gens=n_gens, job_id=job_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=int,
                    help='The program must recive the ID of current job')
    parser.add_argument('crossover_op', type=str,
                    help='The program must recive the crossover operator to be used')
    parser.add_argument('mutation_op', type=str,
                    help='The program must recive the mutation operator to be used')
    parser.add_argument('domain', type=str,
                    help='The program must recive the domain of the problem')
    parser.add_argument('--n_runs', type=int, default=1,
                    help='The program may recive the number of measures to be taken')
    parser.add_argument('--n_gens', type=int, default=100,
                    help='The program may recive the number of generations to be taken')
    
    
    args = parser.parse_args()
    main(job_id=str(args.job_id), 
         cross_op=args.crossover_op,
         mutation_op=args.mutation_op,
         domain=args.domain, 
         n_runs=args.n_runs, 
         n_gens=args.n_gens)
    