from Measurer.Logger import Logger
import sys
from datetime import datetime
from time import sleep

def main():
    ouput_file = f'{sys.argv[2]}/gpu_measures.csv'
    sleep_time = 1
    iters_untill_dump = 5
    iters = iters_untill_dump
    append = False
    logger = Logger()
    logger.add_time_col()
    logger.add_gpu_measure_col()
    logger.add_gen_col()
    
    while True:
        logger.log()
        if(iters == 0):
            logger.to_csv(ouput_file, append)
            logger.empty_logs()
            append = True
            iters = iters_untill_dump
        else:
            iters -= 1
        print(iters)
        sleep(sleep_time)

if __name__ == "__main__":
    main()