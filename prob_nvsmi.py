from Logger import Logger
import sys
from datetime import datetime
from time import sleep

def main():
    ouput_file = "./out_files/mesures/gpu_" + str(sys.argv[1]) + ".csv"
    sleep_time = 1
    logger = Logger(output_file=ouput_file, job_id=str(sys.argv[1]))
    logger.setup_GPU()
    logger.log_headers()
    while True:
        logger.log()
        sleep(sleep_time)

if __name__ == "__main__":
    main()