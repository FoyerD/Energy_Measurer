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
import tomllib

def main(crossover_op_name:str, mutation_op_name:str, domain:str, output_dir:str, setup_file:str=None):

    with open(setup_file, 'rb') as f:
        config = tomllib.load(f)
        
    evo_algo: SimpleEvolution = None
    crossover_op: GeneticOperator = None
    mutation_op: GeneticOperator = None
    evaluator: IndividualEvaluator = None
    creator: Creator = None
    individual_length: int = None
    higher_is_better: bool = True
    
    evolution_args = config['evolution']
    
    # evaluator
    domain_args = config['domain'][domain]
    if(domain == 'bpp'):
        evaluator, individual_length, min_bound, max_bound = EckityFactory.make_bpp_evaluator(db_path=domain_args['db_path'], dataset_name=domain_args['dataset_name'])
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(min_bound, max_bound))
        higher_is_better = True
    
    elif(domain == 'frozen_lake'):
        evaluator, individual_length = EckityFactory.make_frozen_lake_evaluator(slippery=domain_args.get('slippery', False), num_games=domain_args.get('num_games', 5))
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(0, 3))
        higher_is_better = True

    else:
        raise ValueError(f'Domain {domain} not recognized')
    
    # crossover operator
    crossover_args = config['crossover'][crossover_op_name]
    if(crossover_op_name == 'dnc'):
        crossover_op = EckityFactory.create_dnc_op(individual_creator=creator,
                                                     evaluator=evaluator,
                                                     individual_length=individual_length,
                                                     population_size=evolution_args['population_size'],
                                                     embedding_dim=crossover_args['embedding_dim'],
                                                     loggers=None,
                                                     log_events=None,
                                                     batch_size=crossover_args['batch_size'])
    elif(crossover_op_name == 'k_point'):
        crossover_op = EckityFactory.create_k_point_crossover(probability=crossover_args['probability'],
                                                              arity=crossover_args['arity'],
                                                              k=crossover_args['k'])
    else:
        raise ValueError(f'Operator {crossover_op_name} not recognized')
    
    # mutation operator
    mutation_args = config['mutation'][mutation_op_name]
    if(mutation_op_name == 'uniform'):
        mutation_op = EckityFactory.create_uniform_mutation(probability=mutation_args['probability'],
                                                     arity=mutation_args['arity'])
    else:
        raise ValueError(f'Operator {mutation_op_name} not recognized')

    
    # Logger setup
    statistics_logger = Logger(output_path=os.path.join(output_dir, 'statistics.csv'))
    statistics_logger.add_time_col()
    statistics_logger.add_memory_col(units='KB')
    
    # Selection operator
    selection = TournamentSelection(tournament_size=5, higher_is_better=higher_is_better)
    evo_algo = EckityFactory.create_simple_evo(population_size=evolution_args['population_size'],
                                                           max_generation=evolution_args['max_generation'],
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
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                    help='The program may recive the output directory to save the results')
    parser.add_argument('--setup_file', type=str, default='setup.toml',
                        help='Path to the setup file containing additional parameters')
    
    
    args = parser.parse_args()

    main(crossover_op_name=args.crossover_op,
         mutation_op_name=args.mutation_op,
         domain=args.domain,  
         output_dir=args.output_dir,
         setup_file=args.setup_file)
