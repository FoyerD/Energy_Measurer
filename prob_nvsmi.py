from Logger import Logger
import sys
from datetime import datetime
from time import sleep

def main():
    ouput_file = "./out_files/mesures/gpu_" + str(sys.argv[1]) + ".csv"
    sleep_time = 1
    iters_untill_dump = 5
    iters = iters_untill_dump
    append = False
    logger = Logger()
    logger.add_time_col()
    logger.add_gen_col()
    logger.add_gpu_mesure_col()
    logger.log_headers(path=ouput_file)
    
    while True:
        logger.log()
        if(iters == 0):
            logger.to_csv(ouput_file, append)
            append = True
            iters = iters_untill_dump
        else:
            iters -= 1
        sleep(sleep_time)

if __name__ == "__main__":
    main()