import sys
from Measurer import Measurer


def main():
    measurer = Measurer(job_id=job_id, csv_dir="./out_files/")
    measurer.setup_dnc(max_generation=100, embedding_dim=64)
    measurer.start_measure()
    measurer.save_measures()
    measurer.get_dual_graph()

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()