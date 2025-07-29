import subprocess
import time
import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution
import os
class Logger():
    def __init__(self, columns: dict = None, output_file: str = None):
        self._columns = columns if columns is not None else {}
        self._log_data = []  # List to hold log entries
        self._dump_each = 10
        self._first_dump = True
        self.output_file = output_file

    def update_column(self, name: str, lamd: callable):
        '''
        :param name: name of column in logger
        :param lamd: lambda, will be probed with each log
        '''
        self._columns[name] = lamd

    def log(self, *args, **kwargs):
        log_entry = {key: str(self._columns[key]()) for key in self._columns}
        self._log_data.append(log_entry)
        if len(self._log_data) >= self._dump_each:
            if self._first_dump:
                if(os.path.exists(self.output_file)):
                    with open(self.output_file, 'a') as f:
                        f.write('###\n')
                self.log_headers(self.output_file)
                self._first_dump = False
            self.dump_rows(self.output_file)
            self.empty_logs()

    def add_gen_col(self, algo: SimpleEvolution = None):
        self.update_column("gen", (lambda: algo.event_name_to_data('')['generation_num']) if algo is not None else (lambda: None))

    def add_time_col(self):
        self.update_column("time", lambda: time.time())
    
    def add_str_col(self, name: str, value: str):
        self.update_column(name, lambda: value)
        
    def add_cpu_measure_col(self):
        self.update_column("measure", lambda: 0)
        self.update_column("type", lambda: "CPU")

    def add_slurm_cpu_measure_col(self, job_id: str):
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.add_str_col("type", "CPU")
    
    def add_gpu_measure_col(self):
        self.update_column("measure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])    
        self.add_str_col("type", "GPU")
        
    def add_best_of_gen_col(self, algo: SimpleEvolution):
        self.update_column("best_of_gen", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_gen) if algo.best_of_gen is not None else 0)
    
    def add_average_col(self, algo: SimpleEvolution):
        self.update_column("average", lambda: float(algo.get_average_fitness()[0]))
        
    def to_csv(self, path: str, append: bool = False, header: bool = False):
        df = pd.DataFrame(self._log_data)
        if append:
            df.to_csv(path, index=False, mode='a', header=header)
        else:
            df.to_csv(path, index=False, header=header)
    
    def empty_logs(self):
        self._log_data = []
    
    def log_headers(self, path: str):
        headers = list(self._columns.keys())
        with open(path, "a") as file_obj:
            file_obj.write(','.join(headers) + '\n')

    def num_logs(self):
        return len(self._log_data)
    
    def get_df(self):
        return pd.DataFrame(self._log_data)
    
    def dump_rows(self, path: str):
        df = self.get_df()
        if not df.empty:
            df.to_csv(path, index=False, mode='a', header=False)
        else:
            print("No logs to dump.")