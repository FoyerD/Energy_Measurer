import subprocess

import pandas as pd
from code_files.energy_mesurer.ECkittyFactory import ECkittyFactory
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME
from code_files.energy_mesurer.Logger import Logger
from eckity.algorithms.simple_evolution import SimpleEvolution, AFTER_GENERATION_EVENT_NAME
from code_files.energy_mesurer.Plotter import Plotter

class Mesurer:
    def __init__(self, job_id:int, csv_dir:str):
        self._job_id = job_id
        self._eckitty_factory = ECkittyFactory(job_id)
        self._csv_dir = csv_dir
        self._cpu_loggers = []
        self._statistics_loggers = []
        
        
    def setup_dnc(self, max_generation:int=100, embedding_dim:int=64):
        logger_before_train = Logger(job_id=self._job_id)
        logger_before_train.add_time_col()
        logger_before_train.add_cpu_mesure_col(self._job_id)
        self._cpu_loggers.append(logger_before_train)
        
        logger_after_train = Logger(job_id=self._job_id)
        logger_after_train.add_time_col()
        logger_after_train.add_cpu_mesure_col(self._job_id)
        self._cpu_loggers.append(logger_after_train)
        
        dnc_op = self._eckitty_factory.create_dnc_op(embedding_dim=embedding_dim, loggers=[logger_before_train, logger_after_train], log_events=[BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME])
        
        
        
        logger_after_generation = Logger(job_id=self._job_id)
        logger_after_generation.add_time_col()
        logger_after_train.add_cpu_mesure_col(self._job_id)
        self._cpu_loggers.append(logger_after_generation)

        logger_statistics = Logger(job_id=self._job_id)
        logger_statistics.add_time_col()
        self._statistics_loggers.append(logger_statistics)
        
        evo_algo = self._ec_kitty_factory.create_simple_evo(max_generation=max_generation, embedding_dim=embedding_dim, operators_sequence=[dnc_op], loggers=[logger_after_generation, logger_statistics], log_events=[AFTER_GENERATION_EVENT_NAME, AFTER_GENERATION_EVENT_NAME])
        logger_statistics.add_best_of_gen_col(evo_algo)
        logger_statistics.add_average_col(evo_algo)
        
        for logger in self._cpu_loggers + self._statistics_loggers:
            logger.add_gen_col(evo_algo)
    
    
    def start_mesure(self):
        gpu_prober = self.start_prober()
        self._evo_algo.evolve()
        self._evo_algo.execute()
        gpu_prober.kill()
        
    def _start_prober(self):
        return subprocess.Popen(["python", "./code_files/energy_mesurer/prob_nvsmi.py", self._job_id])
     
    def save_mesures(self):
        first = True
        for logger in self._cpu_loggers:
            logger.to_csv(self._csv_dir + f'/cpu_{self._job_id}.csv', append=not first)
            first = False
        
        first = True
        for logger in self._statistics_loggers:
            logger.to_csv(self._csv_dir + f'/statistics_{self._job_id}.csv', append=not first)
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
        
        plotter = Plotter(x_col='gen', db1=merged_db, db2=statistics_db, job_id=self._job_id)
        plotter.add_groupby_max_plot(col='mesure', db_n=0, label='cpu joules', )
        plotter.add_plot(col='average', db_n=1)
        plotter.add_plot(col='best_of_gen', db_n=1)
        
        plotter.add_marker(time = 5*60, col='best_of_gen', db_n=1)
        plotter.take_above(col='average', value=0, db_n=1)
        plotter.take_above(col='best_of_gen', value=0, db_n=1)
        
        plotter.save_fig(path=f'./out_files/plots/dual_{self._job_id}.png', title='Measure/Statistics vs time', x_labels=['generation', 'generation'], y_labels=['joules', 'fitness'])
        

    