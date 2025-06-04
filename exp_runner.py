import datetime

import os
import argparse
from EckityExtended.EckityWrapper import EckityWrapper





def get_evaluator(wrapper:EckityWrapper, domain:str):
    if(domain == 'bpp'):
        return wrapper.setup_bpp_evaluator(db_path='./datasets_dnc/hard_parsed.json', dataset_name='BPP_14')
    elif(domain == 'frozen_lake'):
        return wrapper.setup_frozen_lake_evaluator()
    else:
        raise ValueError(f'Domain {domain} not recognized')

def get_crossover_op(wrapper:EckityWrapper, cross_op:str, logging:bool=False):
    if(cross_op == 'dnc'):
        return wrapper.setup_dnc(embedding_dim=64, logging=logging)
    elif(cross_op == 'k_point'):
        return wrapper.setup_k_point_crossover()
    else:
        raise ValueError(f'Operator {cross_op} not recognized')

def get_mutation_op(wrapper:EckityWrapper, mutation_op:str, logging:bool=False):
    if(mutation_op == 'uniform'):
        return wrapper.setup_uniform_mutation()
    else:
        raise ValueError(f'Operator {mutation_op} not recognized')





def main(cross_op:str, mutation_op:str, domain:str, output_dir:str, n_gens:int=100, sleep_time:int=0, log_cpu:bool=False, log_gpu:bool=False, logging:bool=False, log_statistics:bool=False):
    wrappers = []
    output_dir = output_dir#os.path.join(os.getcwd(), "out_files", "exp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    #os.makedirs(output_dir, exist_ok=True)
    wrapper = EckityWrapper(output_dir=output_dir)
    wrappers.append(wrapper)
    
    # evaluator
    get_evaluator(wrapper, domain)
    
    # crossover operator
    get_crossover_op(wrapper, cross_op, logging=logging)
    
    # mutation operator
    get_mutation_op(wrapper, mutation_op, logging=logging)
    
    wrapper.create_simple_evo(population_size=100, max_generation=n_gens, log_cpu=log_cpu, log_statistics=log_statistics)

    prober_path = None#os.path.join(os.getcwd(), "code_files", "energy_wrapper", "prob_nvsmi.py")
    wrapper.start_measure(prober_path=prober_path, write_each=5)
    wrapper.save_measures()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('crossover_op', type=str,
                    help='The program must recive the crossover operator to be used')
    parser.add_argument('mutation_op', type=str,
                    help='The program must recive the mutation operator to be used')
    parser.add_argument('domain', type=str,
                    help='The program must recive the domain of the problem')
    parser.add_argument('-n', '--n_gens', type=int, default=100,
                    help='The program may recive the number of generations to be taken')
    parser.add_argument('-l', '--logging', action='store_true',
                    help='Enable logging during the execution')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                    help='The program may recive the output directory to save the results')
    parser.add_argument('-cpu', '--log_cpu', action='store_true',
                    help='Enable CPU logging')
    parser.add_argument('-stats', '--log_statistics', action='store_true',
                    help='Enable statistics logging')
    
    
    args = parser.parse_args()
    
    
    main(cross_op=args.crossover_op,
         mutation_op=args.mutation_op,
         domain=args.domain,  
         n_gens=args.n_gens,
         logging=args.logging,
         output_dir=args.output_dir,
         log_cpu=args.log_cpu,
         log_statistics=args.log_statistics)
