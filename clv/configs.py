import yaml
from os.path import join, abspath


def read_config(directory):
    with open(join(directory, "docs", "configs.yaml")) as file:
        config = yaml.full_load(file)
    return config


def read_params(directory):
    with open(join(directory, "docs", "test_parameters.yaml")) as file:
        config = yaml.full_load(file)
    return config


def conf(var):
    config = read_config(abspath(__file__).split("configs.py")[0])
    return {
             'config': {c: config[c] for c in config
                        if c not in
                        ['data_main_path', 'model_main_path', 'log_main_path', 'docs_main_path', 'folder_name']}
    }[var]


def hyper_conf(var):
    config = read_params(abspath(__file__).split("configs.py")[0])
    return {'purchase_amount_hyper': config['models']['purchase_amount']['hyper_params'],
            'next_purchase_hyper': config['models']['next_purchase']['hyper_params'],
            'newcomers_hyper': config['models']['newcomers']['hyper_params'],
            'purchase_amount': config['models']['purchase_amount']['params'],
            'next_purchase': config['models']['next_purchase']['params'],
            'newcomers': config['models']['newcomers']['params']
            }[var]


iteration = 30
boostrap_ratio = 0.5
weekdays = ['Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays']
web_port_default = 7002
accepted_ratio_of_actual_order = 0.75
accept_threshold_for_loss_diff = 0.05
parameter_tuning_trials = 10