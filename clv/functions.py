import numpy as np
import pandas as pd
from math import sqrt
import random
import glob
from os import listdir
from os.path import dirname
from dateutil.parser import parse
from statsmodels.tsa.arima.model import ARIMA

try:
    from data_access import GetData
    from utils import *
    from configs import accepted_ratio_of_actual_order
except Exception as e:
    from .data_access import GetData
    from .utils import *
    from .configs import accepted_ratio_of_actual_order


def data_manipulation_nc(date,
                         order_count,
                         amount_indicator,
                         time_indicator,
                         data_source,
                         data_query_path,
                         customer_indicator,
                         directory):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=amount_indicator,
                           date=date)
    data_process.data_execute()
    print("data size :", len(data_process.data))
    data = data_process.data
    data[time_indicator] = data[time_indicator].apply(lambda x: convert_str_to_day(x))
    data = data.sort_values(by=[customer_indicator, time_indicator], ascending=True)

    # list of newcomer users
    newcomers = find_newcomers_with_order_count(data, directory, order_count, customer_indicator, time_indicator)
    # this value will be assigned when prediction process is initialized
    average_amount = np.mean(data[data[amount_indicator] == data[amount_indicator]][amount_indicator])
    data, min_max = order_count_normalization(data, time_indicator)
    return data, "order_count", average_amount, min_max


def data_manipulation_np(date,
                         time_indicator,
                         data_source,
                         data_query_path,
                         feature,
                         customer_indicator,
                         params,
                         time_period,
                         directory):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=feature, date=date)
    data_process.data_execute()
    print("data size :", len(data_process.data))
    data = data_process.data
    data[time_indicator] = data[time_indicator].apply(lambda x: convert_str_to_day(x))
    data['last_days'] = data.sort_values(by=[customer_indicator, time_indicator],
                                         ascending=True).groupby(customer_indicator)[time_indicator].shift(1)
    data = data.query("last_days == last_days")
    data = pd.merge(data, data.rename(columns={"last_days": "last_days_2"}).groupby(
                                            customer_indicator)['last_days_2'].max(),
                    on=customer_indicator, how='left')
    data['time_diff'] = data.apply(lambda row: calculate_time_diff(row['last_days'],
                                                                   row[time_indicator],
                                                                   time_period), axis=1)
    opt_lag = OptimumLagDecision(data, customer_indicator, time_indicator, params, directory)
    opt_lag.find_optimum_lag()
    params['lahead'], params['lag'] = opt_lag.best_lag, opt_lag.best_lag
    params['batch_size'] = max((params['tsteps'] - 1), (params['lahead'] - 1)) + (params['lahead'] * 2)

    data, customer_min_max = get_customer_min_max_data(data, 'time_diff', customer_indicator)
    return data, 'time_diff_norm', customer_min_max, params


def data_manipulation(date,
                      time_indicator,
                      order_count,
                      data_source,
                      data_query_path,
                      amount_indicator,
                      customer_indicator,
                      directory):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=amount_indicator,
                           date=date)
    data_process.data_execute()
    print("data size :", len(data_process.data))
    data = data_process.data
    data[time_indicator] = data[time_indicator].apply(lambda x: convert_str_to_day(x))
    max_date = max(data[time_indicator])
    data = data.sort_values(by=[customer_indicator, time_indicator], ascending=True)
    data['order_seq_num'] = data.sort_values(by=[customer_indicator,
                                                 time_indicator]).groupby([customer_indicator]).cumcount() + 1
    data = data.sort_values(by=[customer_indicator, time_indicator])
    data = pd.merge(data,
                    data.groupby(customer_indicator)['order_seq_num'].max().reset_index().rename(
                        columns={"order_seq_num": "max_order"}),
                    on=customer_indicator, how='left')
    order_count = order_count_decision(data, order_count, customer_indicator, directory)
    data['prev_orders'] = data['max_order'] - order_count
    data = data.query("order_seq_num > prev_orders")
    data['order_seq_num'] = data.sort_values(by=[customer_indicator, time_indicator]).groupby(
        [customer_indicator]).cumcount() + 1
    data['order_seq_num'] = data.apply(
        lambda row: row['order_seq_num'] + abs(row['prev_orders']) if row['prev_orders'] < 0 else row['order_seq_num'],
        axis=1)
    data, customer_min_max = get_customer_min_max_data(data, amount_indicator, customer_indicator)
    data = pivoting_orders_sequence(data, customer_indicator, amount_indicator)
    features = list(range(1, order_count))
    return data, features, order_count, customer_min_max, max_date


def order_count_decision(data, order_count, customer_indicator, directory):
    """
    This allows us to calculate an optimum number of order counts.
    Order count is also the feature number of the purchase amount model.

    !!! Caution !!!
        - Order count must be inserted into the test_parameters.yaml in order not to allow for changing later on prediction.
        - Once the model is built with calculated order_count it must be predicted with the same order count.

    !!! Why we need an order count of a decision? !!!
        - it is a crucial parameter for the purchase amount model.
        - The purchase amount is a 1 Dimensional Conv NN. It works with kernels and it is sizes are related to feature size.
        - At the purchase amount model features are sequential orders.
        - For instance if we assign order count as 5, user_1, user_2, user_3, user_4 have 100, 101, 300, 2 orders.
            The data set will be;
                - user_1: 95th, 96th, 97th, 98th, 99th, 100th  orders
                - user_2: 96th, 97th, 98th, 99th, q00th, 101st  orders
                - user_3: 295th, 296th, 297th, 298th, 299th, 300th  orders
                - user_4: only have 2 orders first 9 orders will be 0 and this will affect the model process.
        - That is why it is crucial to have a minimum 0 assigned order as user_4
          However, it is also a crucial point  to get as much previous order count for make kernel size larger,

    !!! How does it work? !!!
        - At least the last 5 orders are accepted for the model. Lower than 5 orders will not be fully Dimensional Conv.NN.
          It will work as Feed Forward NN because of the lack of kernel size.
        - Iterate from 5 to max order count for any customer at the data set.
        - Finds zero assign number of orders
        - Calculate expected total numbers with given order count; order_count * unique_customers
        - Calculate ratio: zero assign number of orders / expected total numbers with given order count
        - Collect the order count which has ratio less than 0.05
        - Assign as order count form max collected order count (ratio < 0.05)


    :param data: raw data with order_sequential per customer
    :param order_count: if order_count is assigned when the platform is triggered.
    :param customer_indicator: customer columns on data frame
    :param directory: need to check order count has been decided before on test_parameters.yaml
    :return: decided order_count
    """
    if order_count is None:
        params = check_for_existing_parameters(directory, 'purchase_amount')
        if params is None:
            total_orders = []
            max_order = max(data['max_order'])
            data = data.drop('max_order', axis=1)
            for rc in range(5, max_order):
                _data = pd.merge(data,
                                 data.groupby(customer_indicator)['order_seq_num'].max().reset_index().rename(
                                     columns={"order_seq_num": "max_order"}),
                                 on=customer_indicator, how='left')
                _data['prev_orders'] = _data['max_order'] - rc
                _data = _data.query("order_seq_num > prev_orders")
                _total_unique_customers = len(_data[customer_indicator].unique())
                _total_orders = len(_data)
                total_orders.append({"order_count": rc,
                                     "total_orders": _total_orders,
                                     "total_data_point": _total_unique_customers * rc,
                                     "zero_orders": (_total_unique_customers * rc) - _total_orders,
                                     "ratio": _total_orders / (_total_unique_customers * rc),
                                     'unique_customers': _total_unique_customers})
            total_orders = pd.DataFrame(total_orders)
            df = total_orders.query("ratio >= @accepted_ratio_of_actual_order")
            df = df.sort_values(by='ratio', ascending=False)
            df = df.sort_values(by='order_count', ascending=False)
            try:
              order_count = list(df['order_count'])[0]
            except Exception as e:
              print(e)
              order_count = list(total_orders['order_count'])[0]
        else:
            order_count = params['feature_count']
    return order_count


def order_count_normalization(data, time_indicator):
    """
    Min-Max Normalization is applied for order_count per day.
    :param data: data-frame withou order_count columns
    :param time_indicator: time_indicator
    :return: data-frame daily order_count (newcomers), min_max value of order count (data-frame)
    """
    # order_cont per day
    data['order_count'] = 1
    data = data.groupby(time_indicator).agg({"order_count": "sum"}).reset_index()
    # min-max normalization for order count per day
    min_max_columns = ["min_order_count", "max_order_count"]
    for i in min_max_columns:
        data[i] = min(data["order_count"]) if i.split("_")[0] == 'min' else max(data["order_count"])
    data["order_count"] = data.apply(lambda row: min_max_norm(row["order_count"],
                                                              row['min_order_count'],
                                                              row['max_order_count']), axis=1)
    min_max = pd.DataFrame([{i: list(data[i])[0] for i in min_max_columns}])
    data = data.drop(min_max_columns,
                     axis=1).fillna(0).sort_values(by=time_indicator,
                                                   ascending=True)
    return data, min_max


def find_newcomers_with_order_count(data, directory, order_count, customer_indicator, time_indicator):
    """
    By using feature_order_count value which is found
    while purchase amount model s processed is used in order to decide newcomer users
    :param data: raw data
    :param directory: path where test_parameters.yaml is stored with tuned Model Parameters
    :param customer_indicator: customer_indicator
    :param time_indicator: time_indicator
    :return: list of users (newcomers)
    """
    if order_count is None:
        order_count = int(read_yaml(directory, "test_parameters.yaml")['purchase_amount']['feature_count'])
    new_comers = data.groupby(customer_indicator).agg({time_indicator: "count"}).reset_index().rename(
        columns={time_indicator: "order_count"}).query("order_count <= @order_count")
    new_comers = list(new_comers[customer_indicator].unique())
    return new_comers


def min_max_norm(value, _min, _max):
    if abs(_max - _min) != 0:
        return (value - _min) / abs(_max - _min)
    else: return 0


def get_customer_min_max_data(data, feature, customer_indicator):
    data['user_max'], data['user_min'] = data[feature], data[feature]
    users_min_max = data.groupby(customer_indicator).agg({"user_max": "max", "user_min": "min"}).reset_index()
    data = pd.merge(data.drop(["user_max", "user_min"], axis=1), users_min_max, on=customer_indicator, how='left')
    data[feature + '_norm'] = data.apply(lambda row: min_max_norm(row[feature],
                                                                  row['user_min'],
                                                                  row['user_max']), axis=1)
    return data, users_min_max


def pivoting_orders_sequence(data, customer_indicator, feature):
    data = pd.DataFrame(np.array(data.pivot_table(index=customer_indicator,
                                                  columns="order_seq_num",
                                                  aggfunc={feature + "_norm": "first"}
                                                  ).reset_index())).rename(columns={0: customer_indicator})
    data = data.fillna(0)
    return data


def calculate_time_diff(date, prev_date, time_period):
    date = datetime.datetime.strptime(str(date)[0:10], '%Y-%m-%d')
    prev_date = datetime.datetime.strptime(str(prev_date)[0:10], '%Y-%m-%d')
    return abs((date - prev_date).total_seconds()) / 60 / 60 / 24


def sampling(sample, sample_size):
    if len(sample) <= sample_size:
        return sample
    else:
        return random.sample(sample, sample_size)


def get_sample_size(ratio, sample):
    return int(ratio * len(sample))


def random_data_split(data, ratio):
    index = range(len(data))
    train_index = random.sample(index, int(len(data) * ratio))
    test_index = list(set(index) - set(train_index))
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    return train, test


def reshape_data(model_data, features, y, prediction):
    if not prediction:
        model_data['x_train'] = model_data['train'][features].values.reshape(
            len(model_data['train']), len(features), 1)
        model_data['y_train'] = model_data['train'][y].values
        model_data['x_test'] = model_data['test'][features].values.reshape(
            len(model_data['test']),len(features), 1)
        model_data['y_test'] = model_data['test'][y].values
    else:
        model_data['prediction_x'] = model_data['prediction'][features].values.reshape(
            len(model_data['prediction']), len(features), 1)
    return model_data


def data_for_customer_prediction(data, prediction_data, params):
    if len(prediction_data) != 0:
        data = pd.concat([data[['time_diff_norm']], prediction_data[['time_diff_norm']]])
    x = pd.DataFrame(np.repeat(data[['time_diff_norm']].values, repeats=params['lag'], axis=1))
    shift_day = int(params['lahead'] / params['lag'])
    if params['lahead'] > 1:
        for i, c in enumerate(x.columns):
            x[c] = x[c].shift(i * shift_day)  # every each same days of shifted
    to_drop = max((params['tsteps'] - 1), (params['lahead'] - 1))
    if len(x.iloc[to_drop:]) < params['lag']:
        additional_historic_data = x.iloc[0:abs(params['lag'] - len(x.iloc[to_drop:]))]
        x = pd.concat([additional_historic_data, x.iloc[to_drop:]])
        return reshape_3(x.values)
    else:
        return reshape_3(x.iloc[to_drop:].values)


def check_for_next_prediction(data, model_num):
    columns = sorted([int(col) for col in data.columns], reverse=False)[-model_num:]
    data_for_pred = pd.DataFrame([list(data[col])[0] for col in columns])
    data_for_pred = data_for_pred.values.reshape(1, model_num, 1)
    return data_for_pred


def add_predicted_values_as_column(data, pred):
    max_num = max(list(data.columns))
    data[max_num + 1] = pred
    return data


def get_prediction(data, number, model_num, model):
    for num in range(0, int(number) + 1):
        _pred_data = check_for_next_prediction(data, model_num)
        _pred = model.predict(_pred_data)[0]
        data = add_predicted_values_as_column(data, _pred)
    return data


def get_predicted_data_readable_form(user, prediction, removing_columns, _user_min, _user_max, customer_indicator):
    removing_cols = list(range(removing_columns + 1))
    predictions = [{customer_indicator: user,
                    "user_min": _user_min,
                    "user_max": _user_max,
                    "pred_order_seq": col - removing_columns,
                    "prediction": list(prediction[col])[0]} for col in prediction.columns if col not in removing_cols]
    predictions = pd.DataFrame(predictions)
    predictions['prediction_values'] = predictions.apply(
        lambda row: ((row['user_max'] - row['user_min']) * row['prediction']) + row['user_min'], axis=1)
    return predictions


def merging_predicted_date_to_result_data(results,
                                          feuture_orders,
                                          customer_indicator,
                                          time_indicator,
                                          amount_indicator):
    results = results.rename(columns={"prediction_values": amount_indicator, 'pred_order_seq': 'order_seq_num'})
    results = pd.merge(results,
                       feuture_orders[[customer_indicator, 'order_seq_num', time_indicator]],
                       on=[customer_indicator, 'order_seq_num'], how='left')
    results['data_type'] = 'prediction'
    data_columns = [customer_indicator, 'order_seq_num', time_indicator, amount_indicator, 'data_type']
    return results[data_columns]


def merge_0_result_at_time_period(data,
                                  max_date,
                                  time_period,
                                  customer_indicator,
                                  time_indicator,
                                  amount_indicator):
    last_time_period_date = max_date + datetime.timedelta(days=convert_time_preiod_to_days(time_period))
    result = pd.DataFrame(data[customer_indicator].unique()).rename(columns={0: customer_indicator})
    result['order_seq_num'] = 1
    result[amount_indicator] = 0
    result['data_type'] = 'prediction'
    result[time_indicator] = last_time_period_date
    return result


reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], 1))
reshape_2 = lambda x: x.reshape((x.shape[0], 1))


def split_data(Y, X, params):
    split_size = int(params['split_ratio'] * len(X))
    x_train = reshape_3(X.iloc[:split_size].values)
    y_train = reshape_2(Y.iloc[:split_size].values)
    x_test = reshape_3(X.iloc[split_size:].values)
    y_test = reshape_2(Y.iloc[split_size:].values)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


def drop_calculation(df, parameters, is_prediction=False):
    data_count = len(df)
    to_drop = max((parameters['tsteps'] - 1), (parameters['lahead'] - 1))
    df = df.iloc[to_drop:]
    if not is_prediction:
        if df.shape[0] > parameters['batch_size']:
            to_drop = df.shape[0] % parameters['batch_size']
        if to_drop > 0:
            df = df.iloc[:-1 * to_drop]
    return df


def arrange__data_for_model(df, f, parameters, is_prediction=False):
    try:
        y = df[f].rolling(window=parameters['tsteps'], center=False).mean()
    except Exception as e:
        print(df[f].head())

    x = pd.DataFrame(np.repeat(df[f].values, repeats=parameters['lag'], axis=1))
    shift_day = int(parameters['lahead'] / parameters['lag'])
    if parameters['lahead'] > 1:
        for i, c in enumerate(x.columns):
            x[c] = x[c].shift(i * shift_day)  # every each same days of shifted
    x = drop_calculation(x, parameters, is_prediction=is_prediction)
    y = drop_calculation(y, parameters, is_prediction=is_prediction)
    return split_data(y, x, parameters) if not is_prediction else reshape_3(x.values)


def convert_time_preiod_to_days(time_period):
    if time_period == 'month':
        return 30
    if time_period == 'week':
        return 7
    if time_period == '2*week':
        return 14
    if time_period == '2*month':
        return 60
    if time_period == 'quarter':
        return 90
    if time_period == '6*month':
        return 180


def check_model_exists(path, model_name, time_period):
    current_date = datetime.datetime.strptime(get_current_day(replace=False), "%Y-%m-%d")
    _prev_model_day_diff = 0
    day_range_for_model_training = convert_time_preiod_to_days(time_period)
    prev_model_date = None
    time_diff = 10000000000
    for m in listdir(dirname(join(path, ""))):
        if "_".join(m.split("_")[:-2]) == model_name:
            _date_str = m.split("_")[-2]
            _date_str = "-".join([_date_str[0:4], _date_str[4:6], _date_str[6:]])
            _date = datetime.datetime.strptime(_date_str, "%Y-%m-%d")
            _time_diff = abs((_date - current_date).total_seconds()) / 60 / 60 / 24
            if _time_diff < day_range_for_model_training:
                if m.split("_")[-1].split(".")[0] == time_period:
                    if _time_diff < time_diff:
                        prev_model_date = _date_str.replace("-", "")
                        time_diff = _time_diff
    if prev_model_date is not None:
        print("previous last model trained :", prev_model_date, ", for ", model_name)
    return prev_model_date


def model_path(directory, model_name, date, time_period):
    return join(directory, model_name + "_" + date + "_" + time_period.replace(" ", "") + ".json")


def weights_path(directory, model_name, date, time_period):
    return join(directory, model_name + "_" + date + "_" + time_period.replace(" ", "") + ".h5")


def get_tuning_params(parameter_tuning, params):
    """
      activation: relu_tanh
      batch_size:
        - 5120
        - 10240
        - 20480
      epochs:  1000*40000
      units:
        - 8
        - 16
        - 32
        - 64

      drop_out_ratio': 0.1*0.5

    :param parameter_tuning:
    :param params:
    :param job:
    :return:
    """
    arrays = []
    hyper_params = {}
    for p in parameter_tuning:
        if type(parameter_tuning[p]) in [dict, list]:
            hyper_params[p] = parameter_tuning[p]
        if type(parameter_tuning[p]) == str:
            if "*" in parameter_tuning[p]:
                # e.g. 0.1*0.5 return [0.1, 0.2, ..., 0.5] or 0.1*0.5*0.05 return [0.1, 0.15, ..., 0.5]
                _splits = parameter_tuning[p].split("*")
                if len(_splits) == 2:
                    hyper_params[p] = np.arange(float(_splits[0]), float(_splits[1]), float(_splits[0])).tolist()
                if len(_splits) == 3:
                    hyper_params[p] = np.arange(float(_splits[0]), float(_splits[1]), float(_splits[2])).tolist()
                hyper_params[p] = [str(c) for c in hyper_params[p]]
            else:  # e.g. relu_tanh or relu
                hyper_params[p] = parameter_tuning[p].split("_")
    for p in hyper_params:
        if p not in ['activation', 'loss']:
            if p not in ['kernel_size', 'pool_size', 'max_pooling_unit', 'lstm_units',
                         'num_layers', 'units', 'batch_size']:
                hyper_params[p] = [float(c) for c in hyper_params[p]]
            else:
                if p != 'num_layers':
                    hyper_params[p] = [int(c) for c in hyper_params[p]]
                else:
                    hyper_params[p] = {c: int(hyper_params[p][c]) for c in hyper_params[p]}

    for p in params:
        if p not in list(hyper_params.keys()):
             hyper_params[p] = params[p]
    return hyper_params


def detect_prev_file(files, parsed_date, time_period):
    detected_file = None
    if len(parsed_date) != 0:
        current_date = min([parse(i.split("_")[2]) for i in files])
        date = current_date - datetime.timedelta(days=convert_time_preiod_to_days(time_period))
        detected_file = None
        for f in files:
            f_split = f.split("_")
            if f_split[1] == time_period:
                if parse(f_split[3].split(".")[0]) >= date:
                    date = parse(f_split[3].split(".")[0])
                    detected_file = f
    return detected_file


def get_results(directory, time_period):
    results = pd.DataFrame()
    result_files = [f for f in listdir(dirname(join(directory, ""))) if f.split("_")[0] == "results"]
    parsed_date = [parse(i.split("_")[2]) for i in result_files]
    detected_file = detect_prev_file(result_files, parsed_date, time_period)
    if detected_file is not None:
        results = pd.read_csv(join(directory, detected_file))
    return results


def check_for_previous_predicted_clv_results(results,
                                             path,
                                             time_period,
                                             time_indicator,
                                             customer_indicator,
                                             ):
    prev_result = get_results(path, time_period)
    if len(prev_result) != 0 and len(results) != 0:
        prev_result['same_order'] = True
        prev_result[time_indicator] = prev_result[time_indicator].apply(lambda x: str(x))
        results[time_indicator] = results[time_indicator].apply(lambda x: str(x))
        prev_result = pd.merge(prev_result,
                               results[[customer_indicator, time_indicator]],
                               on=[customer_indicator, time_indicator], how='left')
        prev_result = prev_result.query("same_order != same_order").drop('same_order', axis=1)
    results = pd.concat([prev_result, results])
    return results


def batch_size_hp_ranges(client_sample_sizes, num_of_customers, hyper_params):
    """
    For the computational cost and model of bias/variance problem related to other parameters of the NN.
    !!! optimum_batch_size: !!!
    client_sample_sizes are number of customers orders.
    The most frequent order count will be the business of the average of order count per user.
    batch size must be the  most frequent order count in order to
    make the predictive model more representative for customers.
    This will only become more affective when batch_size is assigned as The most frequent order count.
    That is why optimum_batch_size is The most frequent order count.

    additional to batch_size hyper parameters, optimum_batch_size and number of unique customers must be tunned.
    :param client_sample_sizes: number of row count per user
    :param num_of_customers: number ıf unique customer count
    :return: range of batch sizes, average_customer_batch
    """
    (unique, counts) = np.unique(client_sample_sizes, return_counts=True)
    optimum_batch_size = int(sorted(zip(counts, unique))[-1][1])
    average_customer_batch = int(num_of_customers - (num_of_customers % optimum_batch_size))
    hyper_params['batch_sizes'] = sorted(hyper_params['batch_sizes'] +
                                         [optimum_batch_size, average_customer_batch], reverse=True)
    return hyper_params, average_customer_batch


class OptimumLagDecision:
    def __init__(self, data, customer_indicator, time_indicator, params, directory):
        self.data = data
        self.customer_indicator = customer_indicator
        self.time_indicator = time_indicator
        self.directory = directory
        self.minimum_batch_size_count = params['batch_size']
        self.split_ratio = params['split_ratio']
        self.lag_range = list(range(1, 4))
        self.train, self.test, self.prediction = [], [], []
        self.model = None
        self.best_lag = params['lag']
        self.min_rmse = 1000000000000

    def collect_data(self):
        self.data['order_seq_num'] = self.data.sort_values(by=[self.customer_indicator,
                                                               self.time_indicator]
                                                           ).groupby([self.customer_indicator]).cumcount() + 1
        self.data = self.data.sort_values(by='order_seq_num', ascending=True).groupby('order_seq_num').agg(
            {"time_diff": "mean"}).reset_index()

    def split_data(self):
        self.train = list(
            self.data[self.data['order_seq_num'] < int(max(self.data['order_seq_num']) * self.split_ratio)][
                'time_diff'])
        self.test = list(
            self.data[self.data['order_seq_num'] >= int(max(self.data['order_seq_num']) * self.split_ratio)][
                'time_diff'])

    def build_arima_model(self, p):
        self.prediction = []
        for i in self.test:
            try:
                self.model = ARIMA(self.train, order=(p, 1, 1))
                self.model = self.model.fit()
                self.prediction.append(self.cal_residuals(i, self.model.forecast(steps=1)[0]))
            except Exception as e:
                print(e)
            self.train += [i]
        return sqrt(sum(self.prediction))

    def cal_residuals(self, actual, prediction):
        return pow(abs(actual - prediction), 2)

    def find_optimum_lag(self):
        if check_for_existing_parameters(self.directory, 'next_purchase') is None:
            self.collect_data()
            self.split_data()
            for lag in self.lag_range:
                _rmse = self.build_arima_model(lag)
                print("lag :", lag, "RMSE :", _rmse)
                if self.min_rmse > _rmse:
                    self.min_rmse = _rmse
                    self.best_lag = lag
            print("optimum lag :", self.best_lag)



