import datetime
from logging import Logger
import os
import sys

import argparse
from time import sleep
from Measurer.Measurer import Measurer
from plot import main as plot_dual_graph


def get_evaluator(measurer:Measurer, domain:str):
    if(domain == 'bpp'):
        return measurer.setup_bpp_evaluator(db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json', dataset_name='BPP_14')
    else:
        raise ValueError(f'Domain {domain} not recognized')

def get_crossover_op(measurer:Measurer, cross_op:str):
    if(cross_op == 'dnc'):
        return measurer.setup_dnc(embedding_dim=64)
    elif(cross_op == 'k_point'):
        return measurer.setup_k_point_crossover()
    else:
        raise ValueError(f'Operator {cross_op} not recognized')

def get_mutation_op(measurer:Measurer, mutation_op:str):
    if(mutation_op == 'uniform'):
        return measurer.setup_uniform_mutation()
    else:
        raise ValueError(f'Operator {mutation_op} not recognized')

def get_nothing_avg(job_id, output_dir, total_time):
        cpu_file = f'{output_dir}/cpu_nothing.csv'
        gpu_file = f'{output_dir}/gpu_nothing.csv'
        sleep_time = 1

        cpu_logger = Logger()
        cpu_logger.add_time_col()
        cpu_logger.add_cpu_measure_col(job_id)

        gpu_logger = Logger()
        gpu_logger.add_time_col()
        gpu_logger.add_gpu_measure_col()
        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).seconds < total_time:
            cpu_logger.log()
            gpu_logger.log()
            sleep(sleep_time)

        cpu_df = cpu_logger.get_df()
        gpu_df = gpu_logger.get_df()
        cpu_logger.to_csv(cpu_file)
        gpu_logger.to_csv(gpu_file)
        
        gpu_avg = gpu_df['measure'].mean()
        cpu_avg = cpu_df.iloc[-1]['measure'] / total_time
        return cpu_avg, gpu_avg

def run_n_measures(job_id:str, cross_op:str, mutation_op:str, domain:str, n_runs:int=1, n_gens:int=100, sleep_time:int=0):
    measurers = []
    parent_output_dir = f"./code_files/energy_measurer/out_files/{job_id}"
    cpu_avg, gpu_avg = get_nothing_avg(job_id, parent_output_dir, total_time=60 * 20)
    for i in range(n_runs):
        output_dir = f"{parent_output_dir}/{i}"
        os.makedirs(output_dir, exist_ok=True)
        measurer = Measurer(job_id=job_id, output_dir=output_dir)
        measurers.append(measurer)
        
        # evaluator
        get_evaluator(measurer, domain)
        
        # crossover operator
        get_crossover_op(measurer, cross_op)
        
        # mutation operator
        get_mutation_op(measurer, mutation_op)
        
        measurer.create_simple_evo(population_size=100, max_generation=n_gens)

        nothing(sleep_time)
        measurer.start_measure(prober_path="./code_files/energy_measurer/prob_nvsmi.py", write_each=5)
        measurer.save_measures()
        markers = [{'time':3*60, 'marker':'o', 'col':'best_of_gen'},
                   {'time':20*60, 'marker':'*', 'col':'best_of_gen'}]
        measurer.get_dual_graph(markers=markers, cpu_avg=cpu_avg, gpu_avg=gpu_avg)
    
    plot_dual_graph(parent_output_dir, job_id)

    
def main(job_id:str, cross_op:str, mutation_op:str, domain:str, n_runs:int=1, n_gens:int=100, sleep_time:int=0):
    run_n_measures(n_runs=n_runs, cross_op=cross_op, mutation_op=mutation_op, domain=domain, n_gens=n_gens, job_id=job_id, sleep_time=sleep_time)

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
    parser.add_argument('--sleep_time', type=int, default=0,
                        help='The program may recive the sleep time before measures')
    
    
    args = parser.parse_args()
    main(job_id=str(args.job_id), 
         cross_op=args.crossover_op,
         mutation_op=args.mutation_op,
         domain=args.domain, 
         n_runs=args.n_runs, 
         n_gens=args.n_gens,
         sleep_time=args.sleep_time)
    