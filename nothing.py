from Measurer.Logger import Logger
import sys
from time import sleep
import datetime

def main():
    cpu_file = f'{sys.argv[2]}/cpu_measures.csv'
    gpu_file = f'{sys.argv[2]}/gpu_measures.csv'
    sleep_time = 1
    iters_untill_dump = int(sys.argv[3])
    total_time = int(sys.argv[4])
    iters = iters_untill_dump
    append = False

    cpu_logger = Logger()
    cpu_logger.add_time_col()
    cpu_logger.add_cpu_measure_col(sys.argv[1])

    gpu_logger = Logger()
    gpu_logger.add_time_col()
    gpu_logger.add_gpu_measure_col()
    print('iters_untill_dump:' + str(iters_untill_dump))
    start_time = datetime.datetime.now()
    while (datetime.datetime.now() - start_time).seconds < total_time:
        cpu_logger.log()
        gpu_logger.log()
        if(iters <= 0):
            cpu_logger.to_csv(cpu_file, append)
            gpu_logger.to_csv(gpu_file, append)

            cpu_logger.empty_logs()
            gpu_logger.empty_logs()
            append = True
            iters = iters_untill_dump
        else:
            iters -= 1
        sleep(sleep_time)

if __name__ == "__main__":
    main()