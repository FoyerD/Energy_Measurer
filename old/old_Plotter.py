import pandas as pd
import matplotlib.pyplot as plt
import sys



def plot_graphs(gpu_db, cpu_db, statistics_db, job_id):
    fig, ax_left = plt.subplots()
    
    #-----statistics------
    # Add markers at timestamps
    statistics_db['time'] = pd.to_datetime(statistics_db['time'])
    statistics_db['time'] = (statistics_db['time'] - statistics_db['time'].min()).dt.total_seconds()
    minutes_marker_5 = 60*5
    minutes_marker_10 = 60*10
    minutes_marker_20 = 60*20
    minutes_marker_3 = 60*3
    statistics_db['time_diff_10'] = (statistics_db['time'] - minutes_marker_10).abs()
    statistics_db['time_diff_5'] = (statistics_db['time'] - minutes_marker_5).abs()
    statistics_db['time_diff_20'] = (statistics_db['time'] - minutes_marker_20).abs()
    statistics_db['time_diff_3'] = (statistics_db['time'] - minutes_marker_3).abs()
    
    statistics_db = statistics_db[statistics_db['best_of_run'] >= 0]
    statistics_db = statistics_db[statistics_db['best_of_gen'] >= 0]
    idx_at_20_minutes = statistics_db['time_diff_20'].idxmin()
    idx_at_3_minutes = statistics_db['time_diff_3'].idxmin()
    idx_at_10_minutes = statistics_db['time_diff_10'].idxmin()
    idx_at_5_minutes = statistics_db['time_diff_5'].idxmin()
    row_at_10_minutes = statistics_db.loc[idx_at_10_minutes]
    row_at_5_minutes = statistics_db.loc[idx_at_5_minutes]
    row_at_20_minutes = statistics_db.loc[idx_at_20_minutes]
    row_at_3_minutes = statistics_db.loc[idx_at_3_minutes]
    

    
    #statistics_db = statistics_db[(statistics_db['gen'] > 100) & (statistics_db['gen'] < 200)]
    ax_right = ax_left.twinx()
    ax_right.plot(statistics_db['gen'], statistics_db['best_of_gen'], label='best_of_gen', color='silver')
    ax_right.plot(statistics_db['gen'], statistics_db['best_of_run'], label='best_of_run', color='gold')
    
    
    #ax_right.scatter(row_at_5_minutes['gen'], row_at_5_minutes['best_of_gen'], color='black', zorder=5, label='5 min', marker='o')
    #ax_right.scatter(row_at_10_minutes['gen'], row_at_10_minutes['best_of_gen'], color='black', zorder=5, label='10 min', marker='*')
    ax_right.scatter(row_at_3_minutes['gen'], row_at_3_minutes['best_of_gen'], color='black', zorder=5, label='3 min', marker='o')
    ax_right.scatter(row_at_20_minutes['gen'], row_at_20_minutes['best_of_gen'], color='black', zorder=5, label='20 min', marker='*')
    #x_right.plot(statistics_db['gen'], statistics_db['average'], label='average', color='chocolate')
    ax_right.set_ylabel('fitness')
    
    #
    
    #-----mesurments------
    merged_db = pd.concat([gpu_db, cpu_db]).sort_values(by='time')
    # Replace GPU measure by the cumulative sum of GPU measures
    merged_db.loc[merged_db['type'] == 'GPU', 'measure'] = merged_db.loc[
        merged_db['type'] == 'GPU', 'measure'
    ].cumsum() 
    merged_db = merged_db.bfill().ffill() #filling empty gen entries of GPU
    

    #merged_db = merged_db[(merged_db['gen'] > 100) & (merged_db['gen'] < 200)]
    
    plt.figure(figsize=(18, 9))
    for data_type, group in merged_db.groupby('type'):
        if(data_type == 'GPU'):
            grouped_gens = group.groupby('gen')['measure'].max()
            grouped_gens.plot(kind='line', ax=ax_left, label=data_type)
        else:
            ax_left.plot(group['gen'], group['measure'], label=data_type)
    '''for train_status, group in merged_db.groupby('train_status'):
        if(train_status == 'Start'):
            [ax_left.axvline(x=gen, ymax=0.1, color='green') for gen in group['gen']]
        elif(train_status == 'Finish'):
            [ax_left.axvline(x=gen, ymax=0.1, color='red') for gen in group['gen']]'''
    ax_left.set_ylabel('joules')
    ax_left.set_xlabel('generation')


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
    plt.xlabel('gen')

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
