import sys
from Measurer.Measurer import Measurer

def main():
    measurer = Measurer(job_id=job_id, output_dir="./code_files/energy_measurer/out_files/{job_id}")
    measurer.setup_dnc(max_generation=100, embedding_dim=64, db_path='./code_files/energy_measurer/datasets_dnc/hard_parsed.json')
    measurer.start_measure(prober_path="./code_files/energy_measurer/prob_nvsmi.py")
    measurer.save_measures()
    measurer.get_dual_graph(take_above=0, markers=[{'time':5*60, 'marker':'o', 'col':'best_of_gen'}])

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()