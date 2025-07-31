import argparse
import os
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators import Creator
from eckity.genetic_operators import GeneticOperator
from eckity.evaluators import IndividualEvaluator
import EckityExtended.ECkityFactory as EckityFactory
from DNC_mid_train.DNC_eckity_wrapper import GAIntegerStringVectorCreator
from Utilities.Logger import Logger
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.algorithms.simple_evolution import AFTER_GENERATION_EVENT_NAME




def main(crossover_op_name:str, mutation_op:str, domain:str, output_dir:str, n_gen:int=100, sleep_time:int=0, log_cpu:bool=False, log_gpu:bool=False, logging:bool=False, log_statistics:bool=False, **kwargs):

    evo_algo: SimpleEvolution = None
    output_dir: str = None
    crossover_op: GeneticOperator = None
    mutation_op: GeneticOperator = None
    evaluator: IndividualEvaluator = None
    creator: Creator = None
    individual_length: int = None
    higher_is_better: bool = True
    
    # evaluator
    if(domain == 'bpp'):
        assert 'db_path' in kwargs, "db_path must be provided for BPP domain"
        assert 'dataset_name' in kwargs, "dataset_name must be provided for BPP domain"

        evaluator, individual_length, min_bound, max_bound = EckityFactory.make_bppp_evaluator(db_path=kwargs['db_path'], dataset_name=kwargs['dataset_name'])
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(min_bound, max_bound))
        higher_is_better = True
    
    elif(domain == 'frozen_lake'):
        evaluator, individual_length = EckityFactory.make_frozen_lake_evaluator(map=map, slippery=kwargs.get('slippery', False), num_games=kwargs.get('num_games', 5))
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(0, 3))
        higher_is_better = True

    else:
        raise ValueError(f'Domain {domain} not recognized')
    
    # crossover operator
    if(crossover_op_name == 'dnc'):
        crossover_op = EckityFactory.create_dnc_op(individual_creator=creator,
                                                     evaluator=evaluator,
                                                     individual_length=individual_length,
                                                     population_size=100,
                                                     embedding_dim=64,
                                                     loggers=None,
                                                     log_events=None,
                                                     batch_size=1024)
    elif(crossover_op_name == 'k_point'):
        crossover_op = EckityFactory.create_k_point_crossover(probability=kwargs.get('probability', 0.5),
                                                              arity=kwargs.get('c_arity', 2),
                                                              k=kwargs.get('k', 1))
    else:
        raise ValueError(f'Operator {crossover_op_name} not recognized')
    
    # mutation operator
    if(mutation_op == 'uniform'):
        mutation_op = EckityFactory.create_uniform_mutation(probability=kwargs.get('probability', 0.1),
                                                     arity=kwargs.get('m_arity', 1))
    else:
        raise ValueError(f'Operator {mutation_op} not recognized')
    
    
    # Logger setup
    statistics_logger = Logger()
    statistics_logger.add_time_col()
    statistics_logger.add_memory_col(units='KB')
    
    # Selection operator
    selection = TournamentSelection(tournament_size=5, higher_is_better=higher_is_better)
    evo_algo = EckityFactory.create_simple_evo(population_size=kwargs.get('population_size', 100),
                                                           max_generation=n_gen,
                                                           individual_creator=creator,
                                                           evaluator=evaluator,
                                                           selection_methods=[selection],
                                                           higher_is_better=higher_is_better,
                                                           operators_sequence=[crossover_op, mutation_op],
                                                           loggers=[statistics_logger],
                                                           log_events=[AFTER_GENERATION_EVENT_NAME])
    
    statistics_logger.add_best_of_gen_col(evo_algo)
    statistics_logger.add_average_col(evo_algo)
    statistics_logger.add_gen_col(evo_algo)
    
    if(crossover_op_name == 'dnc'):
        statistics_logger.update_column("TRAINED", crossover_op.dnc_wrapper.is_trained)
    else:
        statistics_logger.update_column("TRAINED", False)

    # Start the experiment
    evo_algo.evolve()
    evo_algo.execute()
    

    if(statistics_logger.num_logs() == 0):
        return

    if(os.path.exists(output_dir + f'/statistics.csv')):
        with open(output_dir + f'/statistics.csv', 'a') as f:
            f.write('###\n')
        statistics_logger.to_csv(output_dir + f'/statistics.csv', append=True, header=True)
    else:
        statistics_logger.to_csv(output_dir + f'/statistics.csv', append=False, header=True)
    statistics_logger.empty_logs()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('crossover_op', type=str,
                    help='The program must recive the crossover operator to be used')
    parser.add_argument('mutation_op', type=str,
                    help='The program must recive the mutation operator to be used')
    parser.add_argument('domain', type=str,
                    help='The program must recive the domain of the problem')
    parser.add_argument('-n', '--n_gen', type=int, default=100,
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
    
    
    main(crossover_op_name=args.crossover_op,
         mutation_op=args.mutation_op,
         domain=args.domain,  
         n_gen=args.n_gen,
         logging=args.logging,
         output_dir=args.output_dir,
         log_cpu=args.log_cpu,
         log_statistics=args.log_statistics)
