import toml
import os

base_configs_paths = ["setups/dnc_bpp.toml", "setups/kpoint_bpp.toml", "setups/dnc_gc.toml", "setups/kpoint_gc.toml"]
output_dir = "setups/batch_setups"

batch_sizes = [512, 1024, 2048]
sts = [0, 0.1, 0.01, 0.001]
domain_to_dataset_names = {'gc': ['games120.col','myciel7.col','miles1500.col','mulsol.i.2.col','queen8_12.col', 'zeroin.i.2.col'],
                'bpp': ['BPP_14', 'BPP_181', 'BPP_40', 'BPP_47', 'BPP_60', 'BPP_645', 'BPP_785', 'BPP_832']}

os.makedirs(output_dir, exist_ok=True)


def change_instance(cfg, domain, dataset_name):
    if (domain == "gc"):
        cfg["domain"]["args"]["graph_path"] = f"datasets_dnc/graph_coloring/{dataset_name}"
    elif (domain == "bpp"):
        cfg["domain"]["args"]["dataset_name"] = dataset_name

for base_config_path in base_configs_paths:
    with open(base_config_path, "r") as f:
        base_cfg = toml.load(f)
    
    domain = base_config_path.split('_')[1].replace(".toml", '')
    dataset_names = domain_to_dataset_names[domain]

    for dataset_name in dataset_names:
        for st in sts:
            for batch_size in batch_sizes:
                if (st != 0 and batch_size != 2048): continue
                cfg = base_cfg.copy()

                change_instance(cfg, domain, dataset_name)
                if("dnc" in base_config_path):
                    cfg["crossover"]["args"]["dnc_config"]["batch_size"] = batch_size
                    cfg["crossover"]["args"]["dnc_config"]["fitness_epsilon"] = st 
                    filename = f"config_{dataset_name}_bs{batch_size}_st{st}.toml"
                else:
                    filename = f"config_{dataset_name}_kpoint.toml"

                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w") as f:
                    toml.dump(cfg, f)

                print(f"✅ Created {filepath}")
