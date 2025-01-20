import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_graphs(gpu_db, cpu_db, statistics_db, job_id):
    merged_db = pd.concat([gpu_db, cpu_db]).sort_values(by='time')
    # Replace GPU measure by the cumulative sum of GPU measures
    merged_db.loc[merged_db['type'] == 'GPU', 'measure'] = merged_db.loc[
        merged_db['type'] == 'GPU', 'measure'
    ].cumsum() 
    print(merged_db)  
    merged_db = merged_db.ffill()
    fig, ax_left = plt.subplots()
    # Plot the data
    plt.figure(figsize=(18, 9))
    for data_type, group in merged_db.groupby('type'):
        if(data_type == 'GPU'):
            gen_grouped = group.groupby('gen').max()
            print(gen_grouped)
            ax_left.plot(gen_grouped['gen'], gen_grouped['measure'], label=data_type)
        else:
            ax_left.plot(group['gen'], group['measure'], label=data_type)
    for train_status, group in merged_db.groupby('train_status'):
        if(train_status == 'Start'):
            [ax_left.axvline(x=time, ymax=0.1, color='green') for time in group['time']]
        elif(train_status == 'Finish'):
            [ax_left.axvline(x=time, ymax=0.1, color='red') for time in group['time']]
    ax_left.set_ylabel('joules')
    ax_left.set_xlabel('generation')

    ax_right = ax_left.twinx()
    ax_right.plot(statistics_db['gen'], statistics_db['best_of_gen'], label='best_of_gen', color='silver')
    ax_right.plot(statistics_db['gen'], statistics_db['best_of_run'], label='best_of_run', color='gold')
    #x_right.plot(statistics_db['gen'], statistics_db['average'], label='average', color='chocolate')
    ax_right.set_ylabel('fitness')

    fig.suptitle('Measure/Statistics vs time')
    plt.xticks(merged_db['gen'])
    fig.legend()
    fig.tight_layout()
    fig.savefig('out_files/plots/dual_' + job_id + '.png', dpi=300)  # Adjust dpi for resolution

def plot_mesures_graph(merged_db, job_id):
    # Replace GPU measure by the cumulative sum of GPU measures
    merged_db.loc[merged_db['type'] == 'GPU', 'measure'] = merged_db.loc[
        merged_db['type'] == 'GPU', 'measure'
    ].cumsum()

    plt.figure(figsize=(18, 9))
    for data_type, group in merged_db.groupby('type'):
        plt.plot(group['gen'], group['measure'], label=data_type)
    for train_status, group in merged_db.groupby('train_status'):
        if(train_status == 'Start'):
            [plt.axvline(x=time, ymax=0.1, color='green') for time in group['time']]
        elif(train_status == 'Finish'):
            [plt.axvline(x=time, ymax=0.1, color='red') for time in group['time']]
    plt.ylabel('joules')
    plt.xlabel('time')

    plt.suptitle('Measure vs time')
    #plt.xticks(statistics_db['time'][0::3])
    plt.legend()
    plt.savefig('out_files/plots/mesures_' + job_id + '.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution

def plot_statistics_graph(statistics_db, job_id):
    plt.plot(statistics_db['gen'], statistics_db['best_of_gen'], label='best_of_gen', color='silver')
    plt.plot(statistics_db['gen'], statistics_db['best_of_run'], label='best_of_run', color='gold')
    #ax_right.plot(statistics_db['gen'], statistics_db['average'], label='average', color='chocolate')
    plt.ylabel('fitness')

    plt.suptitle('Statistics vs gen')
    #plt.xticks(statistics_db['time'][0::3])
    plt.legend()
    plt.savefig('out_files/plots/statistics_' + job_id + '.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution


def get_dual_graph(job_id):
    # Load the CSV files
    cpu_db = pd.read_csv('./out_files/mesures/cpu_' + job_id + '.csv')
    gpu_db = pd.read_csv('./out_files/mesures/gpu_' + job_id + '.csv')
    statistics_db = pd.read_csv('./out_files/statistics/statistics_' + job_id + '.csv')
    statistics_db = statistics_db.sort_values(by='time')

    plot_graphs(gpu_db, cpu_db, statistics_db, job_id)

'''def get_mesure_graph(job_id):
    # Load the CSV files
    cpu_db = pd.read_csv('./out_files/mesures/cpu_' + job_id + '.csv')
    gpu_db = pd.read_csv('./out_files/mesures/gpu_' + job_id + '.csv')

    merged_db = merge_and_sort(cpu_db, gpu_db, 'time')
    plot_mesures_graph(merged_db, job_id)

def get_statistics_graph(job_id):
    statistics_db = pd.read_csv('./out_files/statistics/statistics_' + job_id + '.csv')
    statistics_db = statistics_db.sort_values(by='time')
    plot_statistics_graph(statistics_db, job_id)'''


plt_clear = plt.clf
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    get_dual_graph(sys.argv[1])
    plt.clf()
    #get_mesure_graph(sys.args[1])
    plt.clf()
    #get_statistics_graph(sys.args[1])
