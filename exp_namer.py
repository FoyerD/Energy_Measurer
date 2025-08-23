from tomllib import load as tload
from sys import argv
from time import time



def get_name(config):
    exp_dir_name = ''

    domain_name = config['domain']['name']
    if (domain_name == 'bpp'):
        exp_dir_name += f"{config['domain']['args']['dataset_name']}__".lower()
    else:
        exp_dir_name += f"{domain_name}__"

    crossover_name = config['crossover']['name']
    if(crossover_name == 'dnc'):
        exp_dir_name += f"{crossover_name}_bz{config['crossover']['args']['dnc_config']['batch_size']}_fe{config['crossover']['args']['dnc_config']['fitness_epsilon']}__"
    else:
        exp_dir_name += f"{crossover_name}__"

    exp_dir_name += f"{int(time())}"
    return exp_dir_name




if __name__ == "__main__":
    setup_file = argv[1]
    with open(setup_file, 'rb') as f:
        config = tload(f)

    print(get_name(config))
