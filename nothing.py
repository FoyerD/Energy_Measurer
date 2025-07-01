from Utilities.Logger import Logger
import argparse
import time
import psutil

def main(out_dir:str, measure_time:int):
    logger = Logger()
    logger.add_time_col()
    logger.add_str_col("COL", "void")
    logger.update_column("MEMORY", psutil.virtual_memory().total / (1024))  # Total memory in KB
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        logger.log(elapsed_time, "void")
        if elapsed_time > measure_time:
            break
        time.sleep(1)
    logger.to_csv(f"{out_dir}/nothing.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('-t', '--time', type=int, default=0, help='Time to measure in seconds')
    
    args = parser.parse_args()
    main(out_dir=args.out_dir, measure_time=args.time)