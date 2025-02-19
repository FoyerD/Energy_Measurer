import subprocess
import pandas as pd
from ECkittyFactory import ECkittyFactory
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME 
from Logger import Logger
from eckity.algorithms.simple_evolution import AFTER_GENERATION_EVENT_NAME
from Plotter import Plotter
import subprocess
import pandas as pd
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import GAIntegerStringVectorCreator
from DNC_mid_train import dnc_runner_eckity
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME 
from Logger import Logger

class Measurer:
    def __init__(self, job_id:int, csv_dir:str):
        self._job_id = job_id
        self._eckitty_factory = ECkittyFactory(job_id)
        self._csv_dir = csv_dir
        self._cpu_loggers = []
        self._statistics_loggers = []
        self._evo_algo = None
        
        
    def setup_dnc(self, max_generation:int=100, embedding_dim:int=64, population_size:int=100):
        logger_before_train = Logger()
        logger_before_train.add_time_col()
        logger_before_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_before_train)
        
        logger_after_train = Logger()
        logger_after_train.add_time_col()
        logger_after_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_train)
        
        dnc_op, dataset = self._eckitty_factory.create_dnc_op(population_size=population_size, embedding_dim=embedding_dim, loggers=[logger_before_train, logger_after_train], log_events=[BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME])
        dataset_item_weights = np.array(dataset['items'])
        dataset_bin_capacity = dataset['max_bin_weight']
        dataset_n_items = len(dataset_item_weights)
        ind_length = dataset_n_items
        min_bound, max_bound = 0, dataset_n_items - 1
        
        
        logger_after_generation = Logger()
        logger_after_generation.add_time_col()
        logger_after_train.add_cpu_measure_col(self._job_id)
        self._cpu_loggers.append(logger_after_generation)

        logger_statistics = Logger()
        logger_statistics.add_time_col()
        self._statistics_loggers.append(logger_statistics)
        
        higher_is_better = True
        individual_creator = GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound))
        bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                   bin_capacity=dataset_bin_capacity, fitness_dict={})
        selection = TournamentSelection(tournament_size=5, higher_is_better=higher_is_better)
        
        self._evo_algo = self._eckitty_factory.create_simple_evo(population_size=population_size,
                                                           max_generation=max_generation,
                                                           individual_creator=individual_creator,
                                                           evaluator=bpp_eval,
                                                           selection_methods=[(selection, 1)],
                                                           higher_is_better=higher_is_better,
                                                           operators_sequence=[dnc_op],
                                                           loggers=[logger_after_generation, logger_statistics],
                                                           log_events=[AFTER_GENERATION_EVENT_NAME, AFTER_GENERATION_EVENT_NAME])
        logger_statistics.add_best_of_gen_col(self._evo_algo)
        logger_statistics.add_average_col(self._evo_algo)
        
        for logger in self._cpu_loggers + self._statistics_loggers:
            logger.add_gen_col(self._evo_algo)
    
    
    def start_measure(self):
        gpu_prober = self._start_prober()
        self._evo_algo.evolve()
        self._evo_algo.execute()
        gpu_prober.kill()
        
    def _start_prober(self):
        return subprocess.Popen(["python", "./code_files/energy_measurer/prob_nvsmi.py", self._job_id])
     
    def save_measures(self):
        first = True
        for logger in self._cpu_loggers:
            logger.to_csv(self._csv_dir + f'/measures/cpu_{self._job_id}.csv', append=not first)
            first = False
        
        first = True
        for logger in self._statistics_loggers:
            logger.to_csv(self._csv_dir + f'/statistics/statistics_{self._job_id}.csv', append=not first)
            first = False
            
    def get_dual_graph(self):
        cpu_db = pd.read_csv(self._csv_dir + f'/cpu_{self._job_id}.csv')
        gpu_db = pd.read_csv(self._csv_dir + f'/gpu_{self._job_id}.csv')
        statistics_db = pd.read_csv(self._csv_dir + f'/statistics_{self._job_id}.csv')
        
        merged_db = pd.concat([gpu_db, cpu_db]).sort_values(by='time')
        # Replace GPU measure by the cumulative sum of GPU measures
        merged_db.loc[merged_db['type'] == 'GPU', 'measure'] = merged_db.loc[
            merged_db['type'] == 'GPU', 'measure'
        ].cumsum() 
        merged_db = merged_db.bfill().ffill() #filling empty gen entries of GPU
        # Split the merged_db into two DataFrames based on 'type'
        gpu_db_split = merged_db[merged_db['type'] == 'GPU'].reset_index(drop=True)
        cpu_db_split = merged_db[merged_db['type'] == 'CPU'].reset_index(drop=True)

        
        
        
        plotter = Plotter(x_col='gen', dbs=[cpu_db_split, gpu_db_split, statistics_db])
        plotter.add_groupby_max_plot(col='measure', db_n=1, axes_n=0, label='gpu joules')
        plotter.add_plot(col='measure', db_n=0, axes_n=0, label='cpu joules')
        plotter.add_plot(col='average', db_n=2, axes_n=1, label='average fitness')
        plotter.add_plot(col='best_of_gen', db_n=2, axes_n=1, label='best of gen fitness')
        
        plotter.add_marker(time = 5*60, col='best_of_gen', db_n=1, axes_n=1)
        plotter.take_above(col='average', value=0, db_n=1)
        plotter.take_above(col='best_of_gen', value=0, db_n=1)
        
        plotter.save_fig(path=f'./out_files/plots/dual_{self._job_id}.png', title='Measure/Statistics vs time', x_labels=['generation', 'generation'], y_labels=['joules', 'fitness'])
        

    