import datetime
import yaml
from os.path import join, abspath
from dateutil.parser import parse


init_directory = abspath(__file__).split("configs.py")[0]  # "/".join(abspath(__file__).split("/")[:-1]) + "/"
init_directory2 = abspath("")


def get_directory(path):
    directory, web, web_host = abspath(__file__).split("configs.py")[0], 7000, '127.0.0.1'
    with open(join(path, "instance.yaml")) as file:
        instances = yaml.full_load(file)
    active_ins = []
    for ins in instances['instances']:
        if ins['active'] is True and init_directory2 == ins['absolute_path']:
            active_ins.append(ins)
    if len(active_ins) == 0:
        directory = init_directory
        web = 7070
    if len(active_ins) == 1:
        active = active_ins[0]
        directory = active['directory']
        web = active['web']
        web_host = active['web_host']
    if len(active_ins) > 1:
        now = datetime.datetime.now()
        actives = list(map(lambda x: ((now - x['start_date']).total_seconds(), x['directory'], x['web'], x['web_host']), active_ins))
        directory = sorted(actives)[0][1]
        web = sorted(actives)[0][2]
        web_host = sorted(actives)[0][3]
    return directory, web, web_host


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
            'purchase_amount': config['models']['purchase_amount']['params'],
            'next_purchase': config['models']['next_purchase']['params']
            }[var]


alpha = 0.01
iteration = 30
boostrap_ratio = 0.5
time_dimensions = ['year', 'quarter', 'month', 'week', 'week_part', 'week_day', 'day_part', 'hour', 'min', 'second']
weekdays = ['Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays']
folder_name = 'anomaly_detection_framework'
web_port_default = 7002
day_of_year = 366
time_indicator_accept_threshold = {
    'threshold': 0.9, 'accept_ratio_value': 0.5
}
s_size_ratio = 0.6
normal_dist_confidence_intervals = [.9, .99, .01]
descriptive_columns = ['mean1', 'mean2', 'size1', 'size2', 'var1', 'var2']