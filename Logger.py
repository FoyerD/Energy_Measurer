import subprocess
from datetime import datetime

class Logger():
    def __init__(self, columns: dict = {}, output_file = None, job_id : str = "1"):
        self._columns = columns
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
        if(self._output_file == None): print(",".join(list(self._columns.keys())))
        else:
            with open(self._output_file, "a") as output_file:
                print(",".join(list(self._columns.keys())), file=output_file)

    def set_job_id(self,job_id):
        self._job_id= job_id
        print(self._job_id)

    def log(self, aux1=None, aux2=None):
        if(self._output_file == None): print(",".join([str(self._columns[key]()) for key in self._columns]))
        else:
            with open(self._output_file, "a") as output_file:
                print(",".join([str(self._columns[key]()) for key in self._columns]), file=output_file)

    def raw_log(self,line):
        if(self._output_file == None): print(line)
        else:
            with open(self._output_file, "a") as output_file:
                print(line, file=output_file)

    def setup_GPU(self):
        self.update_column("time", lambda: datetime.now())
        self.update_column("computation", lambda: "GPU")
        self.update_column("mesure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])
        self.update_column("start_gen", lambda: False)

    def setup_CPU(self):
        self.update_column("time", lambda: datetime.now())
        self.update_column("computation", lambda: "CPU")
        self.update_column("CPU", lambda: str(subprocess.check_output(["sstat", "-j" + self._job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.update_column("start_gen", lambda: True)
        
    def setup_evolution_statistics(self, algo):
        self.update_column("best", lambda: algo)
        self.update_column("avg", lambda: algo)
        self.update_column("median", lambda: algo)
