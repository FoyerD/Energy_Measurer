from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import DeepNeuralCrossoverConfig, GAIntegerStringVectorCreator, DeepNeuralCrossover
from DNC_mid_train import dnc_runner_eckity
from eckity.creators import Creator
from eckity.evaluators import IndividualEvaluator
from eckity.breeders import Breeder, SimpleBreeder
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.statistics import Statistics, BestAverageWorstStatistics
from torch.cuda import is_available


class ECkittyFactory:
    def __init__(self):
        pass
    
    def get_bpp_info(self, db_path:str, dataset_name:str):
        datasets_json = json.load(open(db_path, 'r'))
        dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
        dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
        dataset_n_items = len(dataset_item_weights)
        return dataset_item_weights, dataset_bin_capacity, dataset_n_items
    
    
    def make_bpp_evaluator(self, db_path:str, dataset_name:str):
        fitness_dict = {}
        dataset_item_weights, dataset_bin_capacity, dataset_n_items = self.get_bpp_info(db_path, dataset_name)
        ind_length = dataset_n_items
        min_bound, max_bound = 0, dataset_n_items - 1

        bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                    bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict)
        
        return bpp_eval, ind_length, min_bound, max_bound
    
    def create_dnc_op(self,
                      individual_creator:Creator,
                      evaluator: IndividualEvaluator,
                      individual_length:int,
                      embedding_dim: int = 64,
                      running_mean_decay: float = 0.95,
                      batch_size: int = 1024,
                      learning_rate: float = 1e-4,
                      n_parents: int = 2,
                      epsilon_greedy: float = 0.3,
                      population_size:int = 100,
                      events = None,
                      loggers: list = None,
                      log_events:list = None):

        dnc_config = DeepNeuralCrossoverConfig(
            embedding_dim=embedding_dim,
            sequence_length=individual_length,
            num_embeddings=individual_length + 1,
            running_mean_decay=running_mean_decay,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_device='cuda' if is_available() else 'cpu',
            n_parents=n_parents,
            epsilon_greedy=epsilon_greedy
        )

        dnc_op = DeepNeuralCrossover(probability=0.8, population_size=population_size, dnc_config=dnc_config,
                                    individual_evaluator=evaluator, vector_creator=individual_creator, events=events)
        
        if(loggers is not None and log_events is not None
           and len(loggers) == len(log_events)):
            for logger, log_event in zip(loggers, log_events):
                dnc_op.dnc_wrapper.register(log_event, logger.log)
        
        return dnc_op
    
    def create_k_point_crossover(self,
                                probability:int=1, 
                                arity:int=2, 
                                k:int=1, 
                                events=None, 
                                loggers: list = None,
                                log_events:list = None):
        
        op = VectorKPointsCrossover(probability=probability, k=k, arity=arity, events=events)
        
        if(loggers is not None and log_events is not None
           and len(loggers) == len(log_events)):
            for logger, log_event in zip(loggers, log_events):
                op.register(log_event, logger.log)
        return op
    
    def create_uniform_mutation(self, probability:float=0.5, arity:int=1, probability_for_each:float=0.1, events=None, 
                                loggers: list = None,
                                log_events:list = None):
        op = dnc_runner_eckity.IntVectorUniformMutation(probability=probability, probability_for_each=probability_for_each, arity=arity, events=events)

        if(loggers is not None and log_events is not None
           and len(loggers) == len(log_events)):
            for logger, log_event in zip(loggers, log_events):
                op.register(log_event, logger.log)

        return op 
    
    def create_simple_evo(self,
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
        
        

    