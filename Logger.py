import subprocess
from datetime import datetime
import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution

class Logger():
    def __init__(self, job_id:str, columns: dict = None):
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

    def setup_GPU(self):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: None)
        self.update_column("type", lambda: "GPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])
        self.update_column("train_status", lambda: "Middle")

    def setup_CPU_before_train(self, algo: SimpleEvolution, job_id: str):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("type", lambda: "CPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.update_column("train_status", lambda: "Start")

    def setup_CPU_after_train(self, algo: SimpleEvolution, job_id: str):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("type", lambda: "CPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.update_column("train_status", lambda: "Finish")

    def setup_evolution_statistics(self, algo: SimpleEvolution):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("best_of_gen", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_gen) if algo.best_of_gen is not None else 0)
        self.update_column("average", lambda: float(algo.get_average_fitness()[0]))

    def to_csv(self, path: str):
        df = pd.DataFrame(self._log_data)
        df.to_csv(path, index=False)


