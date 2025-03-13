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
from eckity.creators.creator import Creator
from eckity.evaluators.individual_evaluator import IndividualEvaluator
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
        self._crossover_op = None
        self._mutation_op = None
        self._evaliator = None
        self._creator = None
        self._individual_length = None
        self._min_bound = None
        self._max_bound = None
        
        
    def setup_dnc(self, embedding_dim:int=64, population_size:int=100):
        assert self._evaluator is not None, 'Evaluator must be set'
        assert self._creator is not None, 'Creator must be set'
        assert self._individual_length is not None, 'Individual length must be set'
        
        
        logger_before_train = Logger()
        logger_before_train.add_time_col()
        logger_before_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_before_train)
        
        logger_after_train = Logger()
        logger_after_train.add_time_col()
        logger_after_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_train)
        
        self._crossover_op = self._eckitty_factory.create_dnc_op(individual_creator=self._creator, evaluator=self._evaluator, individual_length=self._individual_length, population_size=population_size, embedding_dim=embedding_dim, loggers=[logger_before_train, logger_after_train], log_events=[BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME])        
        return self._crossover_op
    
    def setup_bpp_evaluator(self, db_path:str, dataset_name:str):
        self._evaluator, self._individual_length, min_bound, max_bound = self._eckitty_factory.make_bpp_evaluator(db_path=db_path, dataset_name=dataset_name)
        self._creator = GAIntegerStringVectorCreator(length=self._individual_length, bounds=(min_bound, max_bound))
        return self._evaluator, self._individual_length, min_bound, max_bound
    
    def setup_k_point_crossover(self, probability:float=0.5, arity:int=2, k:int=1):
        self._crossover_op = self._eckitty_factory.create_k_point_crossover(probability=probability, arity=arity, k=k)
        return self._crossover_op
        
    def setup_uniform_mutation(self, probability:float=0.5, arity:int=1, probability_for_each:float=0.1):
        self._mutation_op = self._eckitty_factory.create_uniform_mutation(probability=probability, arity=arity, probability_for_each=probability_for_each)
        return self._mutation_op
    
    def create_simple_evo(self, population_size:int, max_generation:int, higher_is_better:bool=True):
        assert self._crossover_op is not None, 'Crossover operator must be set'
        assert self._mutation_op is not None, 'Mutation operator must be set'
        
        logger_after_generation = Logger()
        logger_after_generation.add_time_col()
        logger_after_generation.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_generation)

        logger_statistics = Logger()
        logger_statistics.add_time_col()
        self._statistics_loggers.append(logger_statistics)
        
        
        selection = TournamentSelection(tournament_size=5, higher_is_better=higher_is_better)
        
        self._evo_algo = self._eckitty_factory.create_simple_evo(population_size=population_size,
                                                           max_generation=max_generation,
                                                           individual_creator=self._creator,
                                                           evaluator=self._evaluator,
                                                           selection_methods=[selection],
                                                           higher_is_better=higher_is_better,
                                                           operators_sequence=[self._crossover_op, self._mutation_op],
                                                           loggers=[logger_after_generation, logger_statistics],
                                                           log_events=[AFTER_GENERATION_EVENT_NAME, AFTER_GENERATION_EVENT_NAME])
        logger_statistics.add_best_of_gen_col(self._evo_algo)
        logger_statistics.add_average_col(self._evo_algo)
        
        for logger in self._cpu_loggers + self._statistics_loggers:
            logger.add_gen_col(self._evo_algo)
    
    def start_measure(self, prober_path:str, write_each:int=1):
        gpu_prober = self._start_prober(path=prober_path, write_each=write_each)
        self._cpu_loggers[0].log()
        self._evo_algo.evolve()
        self._evo_algo.execute()
        gpu_prober.kill()
        
    def _start_prober(self, path:str, write_each:int=1):
        return subprocess.Popen(["python", path, self._job_id, self._output_dir, str(write_each)])
     
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
            
    def get_dual_graph(self, markers:list=None):
        plot_dual_graph([self.get_cpu_df()],
                            [self.get_gpu_df()],
                            [self.get_statistics_df()],
                            self._output_dir, markers=markers)
    
    def get_cpu_df(self):
        return pd.read_csv(f'{self._output_dir}/cpu_measures.csv')  
    
    def get_gpu_df(self):
        return pd.read_csv(f'{self._output_dir}/gpu_measures.csv')
    
    def get_statistics_df(self):
        return pd.read_csv(f'{self._output_dir}/statistics.csv')

