import os
import subprocess
import pandas as pd
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME 
from Measurer.ECkittyFactory import ECkittyFactory
from Measurer.Plotter import Plotter
from Measurer.Logger import Logger
from Measurer.Logger import Logger
from eckity.algorithms.simple_evolution import AFTER_GENERATION_EVENT_NAME
import subprocess
import pandas as pd
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import GAIntegerStringVectorCreator
from DNC_mid_train import dnc_runner_eckity
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME 
from DNC_mid_train.dnc_runner_eckity import IntVectorUniformMutation
from plot import plot_dual_graph

class Measurer:
    def __init__(self, job_id:int, output_dir:str):
        self._job_id = job_id
        self._eckitty_factory = ECkittyFactory(job_id)
        self._cpu_loggers = []
        self._statistics_loggers = []
        self._evo_algo = None
        self._output_dir = output_dir
        
        
        
    def setup_dnc(self, db_path:str, max_generation:int=100, embedding_dim:int=64, population_size:int=100):
        logger_before_train = Logger()
        logger_before_train.add_time_col()
        logger_before_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_before_train)
        
        logger_after_train = Logger()
        logger_after_train.add_time_col()
        logger_after_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_train)
        
        dnc_op, dataset = self._eckitty_factory.create_dnc_op(population_size=population_size, embedding_dim=embedding_dim, loggers=[logger_before_train, logger_after_train], log_events=[BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME], db_path=db_path)
        dataset_item_weights = np.array(dataset['items'])
        dataset_bin_capacity = dataset['max_bin_weight']
        dataset_n_items = len(dataset_item_weights)
        ind_length = dataset_n_items
        min_bound, max_bound = 0, dataset_n_items - 1
        
        
        logger_after_generation = Logger()
        logger_after_generation.add_time_col()
        logger_after_generation.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_generation)

        logger_statistics = Logger()
        logger_statistics.add_time_col()
        self._statistics_loggers.append(logger_statistics)
        
        higher_is_better = True
        individual_creator = GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound))
        bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                   bin_capacity=dataset_bin_capacity, fitness_dict={})
        selection = TournamentSelection(tournament_size=5, higher_is_better=higher_is_better)
        mutation = IntVectorUniformMutation(probability=0.5, probability_for_each=0.1)
        
        self._evo_algo = self._eckitty_factory.create_simple_evo(population_size=population_size,
                                                           max_generation=max_generation,
                                                           individual_creator=individual_creator,
                                                           evaluator=bpp_eval,
                                                           selection_methods=[selection],
                                                           higher_is_better=higher_is_better,
                                                           operators_sequence=[dnc_op, mutation],
                                                           loggers=[logger_after_generation, logger_statistics],
                                                           log_events=[AFTER_GENERATION_EVENT_NAME, AFTER_GENERATION_EVENT_NAME])
        logger_statistics.add_best_of_gen_col(self._evo_algo)
        logger_statistics.add_average_col(self._evo_algo)
        
        for logger in self._cpu_loggers + self._statistics_loggers:
            logger.add_gen_col(self._evo_algo)
    
    
    def start_measure(self, prober_path:str):
        gpu_prober = self._start_prober(path=prober_path)
        self._evo_algo.evolve()
        self._evo_algo.execute()
        gpu_prober.kill()
        
    def _start_prober(self, path:str):
        return subprocess.Popen(["python", path, self._job_id, self._output_dir])
     
    def save_measures(self):
        first = True
        for logger in self._cpu_loggers:
            if(logger.num_logs() == 0):
                continue
            logger.to_csv(self._output_dir + f'/cpu_measures.csv', append=not first)
            logger.empty_logs()
            first = False
        
        first = True
        for logger in self._statistics_loggers:
            if(logger.num_logs() == 0):
                continue
            logger.to_csv(self._output_dir + f'/statistics.csv', append=not first)
            logger.empty_logs()
            first = False
            
    def get_dual_graph(self, take_above:int=0, markers:list=None):
        plot_dual_graph([self.get_cpu_df()],
                            [self.get_gpu_df()],
                            [self.get_statistics_df()],
                            self._output_dir, take_above=take_above, markers=markers)
    
    def get_cpu_df(self):
        return pd.read_csv(f'{self._output_dir}/cpu_measures.csv')  
    
    def get_gpu_df(self):
        return pd.read_csv(f'{self._output_dir}/gpu_measures.csv')
    
    def get_statistics_df(self):
        return pd.read_csv(f'{self._output_dir}/statistics.csv')

    