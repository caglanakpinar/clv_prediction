import datetime
import sys
import importlib
import os
import yaml
import json
import pickle
import datetime
import subprocess
import os
from os.path import join
import signal
import math
import random
import socket
import errno
import urllib
import time
from os import listdir
from os.path import dirname, join, abspath
import pandas as pd
from multiprocessing import cpu_count
import threading

try:
    from .configs import weekdays, conf, boostrap_ratio, web_port_default
except Exception as e:
    from configs import weekdays, conf, boostrap_ratio, web_port_default


def callfunc(my_file):
    pathname, filename = os.path.split(my_file)
    sys.path.append(os.path.abspath(pathname))
    modname = os.path.splitext(filename)[0]
    my_mod = importlib.import_module(modname)
    return my_mod


def get_running_pids(process_name, argument=None):
    pids = []
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if process_name in line.decode('utf-8'):
            pid = int(line.decode('utf-8').split(None, 1)[0])
            if argument:
                if len(line.decode('utf-8').split(process_name)) != 0:
                    args = line.decode('utf-8').split(process_name)[1].split(None, 1)
                    if str(argument) in args:
                        print("running job :", line.decode('utf-8'))
                        pids.append(pid)
            else:
                print("running job :", line.decode('utf-8'))
                pids.append(pid)
        else:
            print("no initialized ", argument, " is detected")
    return pids


def kill_process_with_name(process_name, argument=None):
    pids = get_running_pids(process_name, argument=argument)
    if len(pids) != 0:
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
    else:
        print("no running jobs")


def is_port_in_use(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    use = False
    try:
        s.bind(("0.0.0.0", port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            use = True
    return use


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


def model_from_to_pkl(directory=None, path=None, model=None, is_writing=False):
    if is_writing:
        with open(join(directory, "", path), "wb") as f:
            pickle.dump(model, f)
    else:
        with open(join(directory, "", path), 'rb') as f:
            model = pickle.load(f)
        return model


def get_col(c1, c2):
    if c2 == 1:
        return c1
    else:
        return c1 + '_' + str(c2)


def split_groups(groups):
    s_groups = []
    if groups not in [None, 'None']:
        if "+" not in groups:
            s_groups = [groups]
        else:
            s_groups = groups.split("+")
    return s_groups


def split_test_groups(group, data):
    return sorted(list(data[group].unique()))[-1]


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


def date_part(date, part):
    if part == 'year':
        return date.year
    if part == 'quarter':
        return get_quarter(date)
    if part == 'month':
        return date.month
    if part == 'week':
        return date.isocalendar()[1]
    if part == 'week_part':
        return 1 if date.isoweekday() in [6, 7] else 0
    if part == 'week_day':
        return date.isoweekday()
    if part == 'day_part':
        return get_day_part(date.hour)
    if part == 'hour':
        return date.hour
    if part == 'min':
        return date.min
    if part == 'second':
        return date.second
    if part == 'day':
        return str(date)[0:10]


def get_quarter(d):
    return "Q%d_%d" % (math.ceil(d.month/3), d.year)


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


def get_result_data_path(directory, time_period):
    return join(directory, "results_" + time_period + "_" + get_current_day())


def convert_date_v2(x):
    return datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")


def convert_str_to_day(x):
    try:
        return datetime.datetime.strptime(str(x)[0:10], "%Y-%m-%d")
    except Exception as e:
        return None


def get_ratio_of_date(max_date, min_date, ratio):
    range_dates = (max_date - min_date).total_seconds()
    return min_date + datetime.timedelta(seconds=int(range_dates * ratio))


def url_string(value, res=False):
    if value is not None:
        if res:
            return value.replace("\r", " ").replace("\n", " ").replace(" ", "+")
        else:
            return value.replace("+", " ")
    else:
        return None


def request_url(url, params=None):
    if params is not None:
        url += '?'
        for p in params:
            url += p + '=' + url_string(str(params[p]), res=True) + '&'
    response = 404
    while response != 200:
        try:
            res = urllib.request.urlopen(url)
            response = res.code
        except Exception as e:
            print(e)
        time.sleep(2)


def convert_feature(value):
    try:
        if value == value:
            return value
        else:
            return None
    except Exception as e:
        print(e)
        return None


def get_sample_size(size):
    if size > 30:
        if size >= 2000:
            return 1000
        else:
            return int(size * boostrap_ratio)
    else:
        return 5


def get_residuals(residuals, ratio):
    if len(residuals) > 10000:
        return random.sample(residuals, 10000)
    else:
        return random.sample(residuals, len(residuals) * ratio)


convert_date_to_day = lambda x:  datetime.datetime.strptime(str(x)[0:10], "%Y-%m-%d %H:%M:%S")


def show_chart(df, x, y1, y2, is_bar_chart):
    """
    plots line chart with to lines
    params: df; x, y1, y2 fields included, x: x axis of data set, y1, y2 and y3; y axis of data sets
    params: is_bar_chart if True shows Bar chart, is_sorted: if True sorts y1, y2 and y3
    return: return multi dimensional line chart or bar chart on plotly
    """
    import plotly.graph_objs as go
    import plotly.offline as offline
    offline.init_notebook_mode()

    chart = go.Bar if is_bar_chart else go.Scatter
    marker = {'size': 15, 'opacity': 0.5, 'line': {'width': 0.5, 'color': 'white'}}
    x = df[x]
    _y1, _y2 = df[y1], df[y2]
    trace = []
    names = [y1, y2]
    counter = 0
    for _y in [_y1, _y2]:
        if not is_bar_chart:
            trace.append(chart(x=x, y=_y, mode='lines+markers', name=names[counter], marker=marker))
        else:
            trace.append(chart(x=x, y=_y, name=names[counter]))
        counter += 1
    offline.iplot(trace)


def get_results(date_col):
    results = []
    for f in listdir(dirname(join(conf('data_main_path'), ""))):
        f_splits = f.split(conf('result_file'))
        if f_splits[0] == "":
            results += pd.read_csv(join(conf('data_main_path'), "", f)).to_dict('results')
    results = pd.DataFrame(results)
    if len(results) >= 1000:
        results = results.sort_values(by=date_col, ascending=True)[-1000:]
    return results


def find_web_port():
    web_port = web_port_default
    while is_port_in_use(web_port):
        web_port += 1
    return web_port


def convert_dt_str(date, replace=True):
    if replace:
        return str(date)[0:10].replace("-", "")
    else:
        return str(date)[0:10]


def check_result_data_exits(path):
    file = []
    for f in listdir(dirname(join(path, ""))):
        if len(f.split("_")) != 1:
            if f.split("_")[-1] == "results.csv":
                file += [f]
    return file


def get_result_file_name(path, date, time_period=None):
    if time_period is None:
        if date is not None:
            return join(path, convert_dt_str(date, replace=True) + "_results.csv")
        else:
            return join(path,  "results.csv")
    else:
        if date is not None:
            return join(path, date[0:19].replace("-", "").replace(" ", "") + "_results.csv")
        else:
            return join(path,  time_period + "_results.csv")


def get_folder_path():
    return abspath(__file__).split("utils.py")[0]


def check_for_existing_parameters(directory, model):
    try:
        params = read_yaml(directory, "test_parameters")[model]['params']
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
    print(cpu_count(), parallel)
    if parallel > cpus:
        parallel = cpus
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





