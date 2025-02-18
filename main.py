import sys
from code_files.energy_mesurer import Mesurer


def main():
    mesurer = Mesurer(opertor_name="dnc", job_id=job_id, csv_dir="./out_files/mesures")
    mesurer.setup_dnc(max_generation=100, embedding_dim=64)
    mesurer.start_mesure()
    mesurer.save_mesures()
    mesurer.get_dual_graph()

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()