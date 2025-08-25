import argparse
import os
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators import Creator
from eckity.genetic_operators import GeneticOperator
from eckity.evaluators import IndividualEvaluator
import EckityExtended.ECkityFactory as EckityFactory
from DNC_mid_train.DNC_eckity_wrapper import GAIntegerStringVectorCreator
from Logger import Logger
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.algorithms.simple_evolution import AFTER_GENERATION_EVENT_NAME
import tomllib
from torch.cuda import is_available as is_cuda_aviable
from DNC_mid_train.DNC_eckity_wrapper import DeepNeuralCrossoverConfig

def main(output_dir:str, setup_file:str=None):

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
    domain_name = config['domain']['name']
    if(domain_name == 'bpp'):
        evaluator, individual_length, min_bound, max_bound = EckityFactory.make_bpp_evaluator(**config['domain']['args'])
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(min_bound, max_bound))
        higher_is_better = True

    elif(domain_name == 'frozen_lake'):
        evaluator, individual_length = EckityFactory.make_frozen_lake_evaluator(**config['domain']['args'])
        creator = GAIntegerStringVectorCreator(length=individual_length, bounds=(0, 3))
        higher_is_better = True

    else:
        raise ValueError(f'Domain {domain_name} not recognized')



    # crossover operator
    crossover_name = config['crossover']['name']
    if(crossover_name == 'dnc'):
        config['crossover']['args']['population_size'] = evolution_args['population_size']
        config['crossover']['args']['dnc_config']['use_device'] = 'cuda' if is_cuda_aviable() else 'cpu'
        config['crossover']['args']['dnc_config']['sequence_length'] = individual_length
        config['crossover']['args']['dnc_config']['num_embeddings'] = individual_length + 1

        dnc_config = DeepNeuralCrossoverConfig(**config['crossover']['args']['dnc_config'])
        config['crossover']['args']['dnc_config'] = dnc_config

        crossover_op = EckityFactory.create_dnc_op(individual_creator=creator,
                                                   evaluator=evaluator,
                                                   **config['crossover']['args'])
        
    elif(crossover_name == 'kpoint'):
        crossover_op = EckityFactory.create_kpoint_crossover(**config['crossover']['args'])
    else:
        raise ValueError(f'Operator {crossover_name} not recognized')


    
    # mutation operator
    mutation_name = config['mutation']['name']
    if(mutation_name == 'uniform'):
        mutation_op = EckityFactory.create_uniform_mutation(**config['mutation']['args'])
    else:
        raise ValueError(f'Operator {mutation_name} not recognized')


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

    if(crossover_name == 'dnc'):
        statistics_logger.update_column("TRAINED", lambda: crossover_op.dnc_wrapper.trained)
        crossover_op.dnc_wrapper.set_best_of_gen_callback(lambda: evo_algo.best_of_gen.fitness.get_pure_fitness() if evo_algo.best_of_gen.fitness is not None else 0)
    else:
        statistics_logger.update_column("TRAINED", lambda: False)

    # Start the experiment
    evo_algo.evolve()
    evo_algo.execute()
    

    if(statistics_logger.num_logs() == 0):
        return

    if(os.path.exists(output_dir + f'/statistics.csv')):
        header = statistics_logger._first
        if statistics_logger._first:
            with open(output_dir + f'/statistics.csv', "a") as file:
                file.write('###\n')
        statistics_logger.to_csv(output_dir + f'/statistics.csv', append=True, header=header)
    else:
        statistics_logger.to_csv(output_dir + f'/statistics.csv', append=False, header=True)
    statistics_logger.empty_logs()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                    help='The program may recive the output directory to save the results')
    parser.add_argument('--setup_file', type=str, default='setup.toml',
                        help='Path to the setup file containing additional parameters')
    
    
    args = parser.parse_args()

    main(output_dir=args.output_dir, setup_file=args.setup_file)
