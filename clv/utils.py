import yaml
import json
import datetime
from os.path import join, abspath
from multiprocessing import cpu_count
import threading


def read_yaml(directory, filename):
    with open(join(directory, "", filename)) as file:
        docs = yaml.full_load(file)
    return docs


def write_yaml(directory, filename, data, ignoring_aliases=False):
    if ignoring_aliases:
        yaml.Dumper.ignore_aliases = lambda *args : True

    with open(join(directory, "", filename), 'w') as file:
        if ignoring_aliases:
            yaml.dump(data, file, default_flow_style=False)
        else:
            yaml.dump(data, file)


def read_write_to_json(directory, filename, data, is_writing):
    if is_writing:
        with open(join(directory, "", filename), 'w') as file:
            json.dump(data, file)
    else:
        with open(join(directory, "", filename), "r") as file:
            data = json.loads(file.read())
        return data


def get_day_part(hour):
    if 0 <= hour < 7:
        return 'night'
    if 7 <= hour < 12:
        return 'morning'
    if 12 <= hour < 17:
        return 'afternoon'
    if 17 <= hour < 21:
        return 'evening1'
    if 21 <= hour < 24:
        return 'evening2'


def model_path(comb, group, model):
    return "_".join(["_".join([str(i[0]) + "*" + str(i[1]) for i in zip(group, comb)]), model]) + ".json"


def convert_date(date):
    if date not in ['', None]:
        if len(date) == 0:
            format_str = '%Y-%m-%d'
        if len(date) == 16:
            format_str = '%Y-%m-%d %H:%M'
        if len(date) > 16:
            format_str = '%Y-%m-%d %H:%M:%S.%f'
        if not str:
            date = datetime.datetime.strptime(date, format_str)
    else:
        date = datetime.datetime.now() + datetime.timedelta(minutes=2)
    return date


def get_current_day(replace=True):
    return str(datetime.datetime.now())[0:10].replace("-", "") if replace else str(datetime.datetime.now())[0:10]


def get_result_data_path(directory, time_period, max_date):
    return join(directory, "results_" + time_period + "_" +
                str(max_date)[0:10].replace("-", "") + "_" + get_current_day() + ".csv")


def convert_str_to_day(x):
    try:
        return datetime.datetime.strptime(str(x)[0:10], "%Y-%m-%d")
    except Exception as e:
        return None


def convert_feature(value):
    try:
        if value == value:
            return value
        else:
            return None
    except Exception as e:
        print(e)
        return None


def get_folder_path():
    return abspath(__file__).split("utils.py")[0]


def check_for_existing_parameters(directory, model):
    try:
        params = read_yaml(directory, "test_parameters.yaml")[model]
    except Exception as e:
        params = None
    return params


def get_iter_sample(s_values, i, iters, cpus):
    if i != iters - 1:
        return s_values[(i * cpus): ((i+1) * cpus)]
    else:
        return s_values[(i * cpus):]


def execute_parallel_run(values, executor, parallel=2, arguments=None):
    global process
    cpus = cpu_count()
    if parallel > cpus * 4:
        parallel = cpus * 4
    iters = int(len(values) / parallel) + 1
    print("number of iterations :", iters)
    for i in range(iters):
        print("iteration :", i)
        _sample_values = get_iter_sample(values, i, iters, parallel)
        all_processes = []
        for v in _sample_values:
            if arguments:
                process = threading.Thread(target=executor, args=(v, arguments, ))
            else:
                process = threading.Thread(target=executor, args=(v,))
            process.deamon = True
            process.start()
        process.join()
    return "done !!!"





