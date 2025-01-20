import subprocess
from datetime import datetime

class Logger():
    def __init__(self, columns: dict = None, output_file = None, job_id : str = "1"):
        self._columns = columns if columns != None else {}
        self._output_file = output_file
        self._job_id = job_id

    def update_column(self, name: str, lamd):
        '''

        :param name: name of column in logger
        :param lamd: lambda, will be probed with each log
        '''
        self._columns[name] = lamd

    def set_file(self,file=None):
        self._output_file = file

    def log_headers(self):
        if(self._output_file == None):
            print(",".join(list(self._columns.keys())))
        else:
            with open(self._output_file, "a") as file_obj:
                print(",".join(list(self._columns.keys())), file=file_obj)

    def set_job_id(self,job_id):
        self._job_id= job_id
        print(self._job_id)

    def log(self, aux1=None, aux2=None):
        if(self._output_file == None):
            print(",".join([str(self._columns[key]()) for key in self._columns]))
        else:
            with open(self._output_file, "a") as file_obj:
                print(",".join([str(self._columns[key]()) for key in self._columns]), file=file_obj)

    def raw_log(self,line):
        if(self._output_file == None):
            print(line)
        else:
            with open(self._output_file, "a") as file_obj:
                print(line, file=file_obj)

    def setup_GPU(self):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: None)
        self.update_column("type", lambda: "GPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])
        self.update_column("train_status", lambda: "Middle")

    def setup_CPU_before_train(self, algo):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("type", lambda: "CPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + self._job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.update_column("train_status", lambda: "Start")
    
    def setup_CPU_after_train(self, algo):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("type", lambda: "CPU")
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + self._job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.update_column("train_status", lambda: "Finish")
        
    def setup_evolution_statistics(self, algo):
        self.update_column("time", lambda: datetime.now())
        self.update_column("gen", lambda: algo.event_name_to_data('')['generation_num'])
        self.update_column("best_of_gen", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_gen) if algo.best_of_gen != None  else 0)
        self.update_column("best_of_run", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_run_) if algo.best_of_gen != None  else 0)
        self.update_column("average", lambda: float(algo.get_average_fitness()[0]))
        #self.update_column("median", lambda: algo)
