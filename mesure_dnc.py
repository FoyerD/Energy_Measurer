import subprocess
import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from DNC_mid_train.DNC_eckity_wrapper import DeepNeuralCrossoverConfig, GAIntegerStringVectorCreator, DeepNeuralCrossover
from DNC_mid_train import dnc_runner_eckity
from DNC_mid_train.multiparent_wrapper import BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME 
from Logger import Logger
import sys
from time import sleep
import matplotlib.pyplot as plt


def setup_dnc_algo():
    fitness_dict = {}
    datasets_json = json.load(open('./datasets/hard_parsed.json', 'r'))
    print("read db")
    dataset_name = 'BPP_14'
    dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
    dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    dataset_n_items = len(dataset_item_weights)

    ind_length = dataset_n_items
    min_bound, max_bound = 0, dataset_n_items - 1
    population_size = 1000

    individual_creator = GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound))
    bpp_eval = dnc_runner_eckity.BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                   bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict)

    dnc_config = DeepNeuralCrossoverConfig(
        embedding_dim=64,
        sequence_length=ind_length,
        num_embeddings=dataset_n_items + 1,
        running_mean_decay=0.95,
        batch_size=1024,
        learning_rate=1e-4,
        use_device='cuda',
        n_parents=2,
        epsilon_greedy=0.3
    )

    dnc_op = DeepNeuralCrossover(probability=0.8, population_size=population_size, dnc_config=dnc_config,
                                 individual_evaluator=bpp_eval, vector_creator=individual_creator)
    
    logger_start = Logger(job_id=job_id, output_file="./out_files/mesures/cpu_" + job_id + ".csv")
    logger_start.setup_CPU_before_train()
    logger_start.log_headers()
    dnc_op.dnc_wrapper.register(BEFORE_TRAIN_EVENT_NAME, logger_start.log)
    
    logger_fin = Logger(job_id=job_id, output_file="./out_files/mesures/cpu_" + job_id + ".csv")
    logger_fin.setup_CPU_after_train()
    dnc_op.dnc_wrapper.register(AFTER_TRAIN_EVENT_NAME, logger_fin.log)


    # Initialize the evolutionary algorithm
    return SimpleEvolution(
        Subpopulation(creators=individual_creator,
                      population_size=population_size,
                      # user-defined fitness evaluation method
                      evaluator=bpp_eval,
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      # elitism_rate=1 / 300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          dnc_op,
                          dnc_runner_eckity.IntVectorUniformMutation(probability=0.5, probability_for_each=0.1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=5, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=50000,
        # termination_checker=ThresholdFromTargetTerminationChecker(optimal=100, threshold=0.0),
        statistics=BestAverageWorstStatistics(), random_seed=4242
    )


def merge_csv():
    # Load the CSV files
    cpu_data = pd.read_csv('./out_files/mesures/cpu_' + job_id + '.csv')  
    gpu_data = pd.read_csv('./out_files/mesures/gpu_' + job_id + '.csv')  
    # Merge the two DataFrames
    merged_data = pd.concat([cpu_data, gpu_data])

    # Optionally, sort the merged data by time
    merged_data['time'] = pd.to_datetime(merged_data['time'])  # Ensure 'time' column is datetime
    merged_data = merged_data.sort_values(by='time')

    # Save the merged data to a new CSV file
    merged_data.to_csv('./out_files/mesures/merged_' + job_id + '.csv', index=False)
    return merged_data
    
def plot_graphs(merged_data, statistics):
    # Replace GPU measure by the cumulative sum of GPU measures
    merged_data.loc[merged_data['type'] == 'GPU', 'measure'] = merged_data.loc[
        merged_data['type'] == 'GPU', 'measure'
    ].cumsum()   

    fig, ax_left = plt.subplots()
    # Plot the data
    plt.figure(figsize=(12, 6))
    for data_type, group in merged_data.groupby('type'):
        ax_left.plot(group['time'], group['measure'], label=data_type)
    for train_status, group in merged_data.groupby('train_status'):
        if(train_status == 'Start'):
            [ax_left.axvline(x=time, ymax=0.1, color='green') for time in group['time']]
        elif(train_status == 'Finish'):
            [ax_left.axvline(x=time, ymax=0.1, color='red') for time in group['time']]
    ax_left.set_ylabel('mesures')
    ax_left.set_xlabel('time')

    ax_right = ax_left.twinx()
    ax_right.plot(statistics['time'], statistics['best_of_gen'], label='best_of_gen', color='silver')
    ax_right.plot(statistics['time'], statistics['best_of_run'], label='best_of_run', color='gold')
    ax_right.plot(statistics['time'], statistics['average'], label='average', color='chocolate')
    ax_right.set_ylabel('statistics')

    fig.suptitle('Measure/Statistics vs Time')
    fig.legend()
    fig.savefig('out_files/plots/plot_' + job_id + '.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution


def main():
    assert (len(sys.argv) >= 1)
    sleep_time = 60 * 4
    prober = subprocess.Popen(["python", "./code_files/energy_mesurer/prob_nvsmi.py", job_id])
    sleep(sleep_time)

    algo = setup_dnc_algo()
    statistics_logger = Logger(output_file="./out_files/statistics/statistics_" + job_id + ".csv" , job_id=job_id)
    statistics_logger.setup_evolution_statistics(algo)
    statistics_logger.log_headers()
    algo.register("after_generation", statistics_logger.log)
    # evolve the generated initial population
    algo.evolve()
    # Execute (show) the best solution
    algo.execute()
    prober.kill()
    merged_data = merge_csv()
    statistics = pd.read_csv('./out_files/statistics/statistics_' + job_id + '.csv')
    statistics['time'] = pd.to_datetime(statistics['time'])  # Ensure 'time' column is datetime
    statistics = statistics.sort_values(by='time')
    plot_graphs(merged_data, statistics)

if __name__ == "__main__":
    job_id = str(sys.argv[1])
    main()