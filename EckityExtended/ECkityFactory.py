from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import DeepNeuralCrossover
from DNC_mid_train import dnc_runner_eckity
from eckity.creators import Creator
from eckity.evaluators import IndividualEvaluator
from eckity.breeders import Breeder, SimpleBreeder
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.statistics import Statistics, BestAverageWorstStatistics
from Utilities.Logger import Logger

def get_statistics_logger():
    logger_statistics = Logger()
    logger_statistics.add_time_col()
    logger_statistics.add_memory_col(units='KB')
    return logger_statistics

def get_bpp_info(db_path:str, dataset_name:str):
    datasets_json = json.load(open(db_path, 'r'))
    dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
    dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    dataset_n_items = len(dataset_item_weights)
    return dataset_item_weights, dataset_bin_capacity, dataset_n_items


def make_bpp_evaluator(db_path:str, dataset_name:str):
    fitness_dict = {}
    dataset_item_weights, dataset_bin_capacity, dataset_n_items = get_bpp_info(db_path, dataset_name)
    ind_length = dataset_n_items
    min_bound, max_bound = 0, dataset_n_items - 1

    bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict)
    
    return bpp_eval, ind_length, min_bound, max_bound

def make_frozen_lake_evaluator(map = None, **kwargs):
    fl_eval = dnc_runner_eckity.FrozenLakeEvaluator(map=map, **kwargs)
    ind_length = fl_eval.get_individual_length()
    return fl_eval, ind_length
    

def create_dnc_op(individual_creator:Creator,
                  evaluator: IndividualEvaluator,
                  loggers: list = None,
                  log_events:list = None,
                  **kwargs):

    dnc_op = DeepNeuralCrossover(vector_creator=individual_creator,
                                 individual_evaluator=evaluator,
                                 **kwargs)

    if(loggers is not None and log_events is not None
        and len(loggers) == len(log_events)):
        for logger, log_event in zip(loggers, log_events):
            dnc_op.dnc_wrapper.register(log_event, logger.log)
    
    return dnc_op

def create_k_point_crossover(loggers: list = None,
                             log_events:list = None,
                             **kwargs):
    
    op = VectorKPointsCrossover(**kwargs)
    
    if(loggers is not None and log_events is not None
        and len(loggers) == len(log_events)):
        for logger, log_event in zip(loggers, log_events):
            op.register(log_event, logger.log)
    return op

def create_uniform_mutation(loggers: list = None,
                            log_events:list = None,
                            **kwargs):
    op = dnc_runner_eckity.IntVectorUniformMutation(**kwargs)

    if(loggers is not None and log_events is not None
        and len(loggers) == len(log_events)):
        for logger, log_event in zip(loggers, log_events):
            op.register(log_event, logger.log)

    return op 

def create_simple_evo(
                        individual_creator:Creator,
                        evaluator: IndividualEvaluator,
                        operators_sequence:list,
                        selection_methods:list,
                        statistics:Statistics=None,
                        population_size:int=100,
                        breeder:Breeder=None,
                        higher_is_better:bool=True,
                        max_workers:int=1,
                        max_generation:int=1000,
                        loggers:list=None,
                        log_events:list=None):
    algo = SimpleEvolution(
        Subpopulation(creators=individual_creator,
                        population_size=population_size,
                        evaluator=evaluator,
                        higher_is_better=higher_is_better,
                        operators_sequence=operators_sequence,
                        selection_methods=list(map(lambda x: (x, 1/len(selection_methods)),selection_methods))),
        breeder=breeder if breeder is not None else SimpleBreeder(),
        max_workers=max_workers,
        max_generation=max_generation,
        statistics=statistics if statistics is not None else BestAverageWorstStatistics(should_print=False), random_seed=4242,should_print=False
    )
    if(loggers is not None and log_events is not None
        and len(loggers) == len(log_events)):
        for logger, log_event in zip(loggers, log_events):
            algo.register(log_event, logger.log)
    return algo
    
    

