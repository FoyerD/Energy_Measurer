from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import DeepNeuralCrossoverConfig, GAIntegerStringVectorCreator, DeepNeuralCrossover
from DNC_mid_train import dnc_runner_eckity
from eckity.creators import Creator
from eckity.evaluators import IndividualEvaluator
from eckity.breeders import Breeder
from eckity.statistics import Statistics
from DNC_mid_train.DNC_eckity_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME


class ECkittyFactory:
    def __init__(self, job_id: int):
        self._job_id = str(job_id)
    
    def create_dnc_op(self,
                      embedding_dim: int = 64,
                      running_mean_decay: float = 0.95,
                      batch_size: int = 512,
                      learning_rate: float = 1e-4,
                      n_parents: int = 2,
                      epsilon_greedy: float = 0.2,
                      population_size:int = 100,
                      events = None,
                      loggers: list = None,
                      log_events:list = None):
        fitness_dict = {}
        datasets_json = json.load(open('./datasets/hard_parsed.json', 'r'))
        dataset_name = 'BPP_14'
        dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
        dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
        dataset_n_items = len(dataset_item_weights)

        ind_length = dataset_n_items
        min_bound, max_bound = 0, dataset_n_items - 1

        individual_creator = GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound))
        bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                    bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict)

        dnc_config = DeepNeuralCrossoverConfig(
            embedding_dim=embedding_dim,
            sequence_length=ind_length,
            num_embeddings=dataset_n_items + 1,
            running_mean_decay=running_mean_decay,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_device='cuda',
            n_parents=n_parents,
            epsilon_greedy=epsilon_greedy
        )

        dnc_op = DeepNeuralCrossover(probability=0.8, population_size=population_size, dnc_config=dnc_config,
                                    individual_evaluator=bpp_eval, vector_creator=individual_creator, events=events)
        
        if(loggers is not None and log_events is not None
           and len(loggers) == len(log_events)):
            for logger, log_event in zip(loggers, log_events):
                dnc_op.dnc_wrapper.register(log_event, logger.log)
                
    def create_simple_evo(self,
                          individual_creator:Creator,
                          evaluator: IndividualEvaluator,
                          operators_sequence:list,
                          selection_methods:list,
                          statistics:Statistics,
                          population_size:int=100,
                          breeder:Breeder=None,
                          max_workers:int=1,
                          max_generation:int=1000,
                          loggers:list=None,
                          log_events:list=None):
        algo = SimpleEvolution(
            Subpopulation(creators=individual_creator,
                          population_size=population_size,
                          evaluator=evaluator,
                          higher_is_better=True,
                          operators_sequence=operators_sequence,
                          selection_methods=selection_methods),
            breeder=breeder,
            max_workers=max_workers,
            max_generation=max_generation,
            statistics=statistics, random_seed=4242
        )
        
        if(loggers is not None and log_events is not None
           and len(loggers) == len(log_events)):
            for logger, log_event in zip(loggers, log_events):
                algo.register(log_event, logger.log)
        
        return algo
        
        

    