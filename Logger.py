import subprocess
from datetime import datetime
import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution

class Logger():
    def __init__(self, columns: dict = None):
        self._columns = columns if columns is not None else {}
        self._log_data = []  # List to hold log entries

    def update_column(self, name: str, lamd):
        '''
        :param name: name of column in logger
        :param lamd: lambda, will be probed with each log
        '''
        self._columns[name] = lamd

    def log(self):
        log_entry = {key: str(self._columns[key]()) for key in self._columns}
        self._log_data.append(log_entry)

    def add_gen_col(self, algo: SimpleEvolution = None):
        self.update_column("gen", (lambda: algo.event_name_to_data('')['generation_num']) if algo is not None else (lambda: None))

    def add_time_col(self):
        self.update_column("time", lambda: datetime.now())
    
    def add_str_col(self, name: str, value: str):
        self.update_column(name, lambda: value)
        
    def add_cpu_mesure_col(self, job_id: str):
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.add_str_col("type", "CPU")
    
    def add_gpu_mesure_col(self):
        self.update_column("measure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])    
        self.add_str_col("type", "GPU")
        
    def add_best_of_gen_col(self, algo: SimpleEvolution):
        self.update_column("best_of_gen", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_gen) if algo.best_of_gen is not None else 0)
    
    def add_average_col(self, algo: SimpleEvolution):
        self.update_column("average", lambda: float(algo.get_average_fitness()[0]))
        
    def to_csv(self, path: str, append: bool = False):
        df = pd.DataFrame(self._log_data)
        if append:
            df.to_csv(path, index=False, mode='a', header=False)
        else:
            df.to_csv(path, index=False)



