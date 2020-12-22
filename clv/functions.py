import numpy as np
import pandas as pd
from dateutil.parser import parse

try:
    from data_access import GetData
    from utils import *
    from configs import time_dimensions, day_of_year, time_indicator_accept_threshold, s_size_ratio
except Exception as e:
    from .data_access import GetData
    from .utils import *
    from .configs import time_dimensions, day_of_year, time_indicator_accept_threshold, s_size_ratio


def data_manipulation_np(date, time_indicator, order_count,
                         data_source, data_query_path, feature, customer_indicator):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=feature, date=date)
    data_process.data_execute()
    print("data size :", len(data_process.data))
    data = data_process.data
    data[time_indicator] = data[time_indicator].apply(lambda x: convert_str_to_day(x))
    data['last_days'] = data.sort_values(by=['user_id', 'days'], ascending=True).groupby("user_id")['days'].shift(1)
    data = data.query("last_days == last_days")
    data = pd.merge(data, data.rename(columns={"last_days": "last_days_2"}).groupby(
                                            customer_indicator)['last_days_2'].max(),
                    on=customer_indicator, how='left')
    data['last_recency'] = data.apply(
        lambda row: 1 if row['last_days'] == row['last_days'] and row['last_days_2'] == row[time_indicator] else 0,
        axis=1)
    data['time_diff'] = data.apply(lambda row: calculate_time_diff(row['last_days'], row[time_indicator]), axis=1)
    data, customer_min_max = get_customer_min_max_data(data, 'time_diff', customer_indicator)
    features = list(range(order_count))
    return data, 'time_diff_norm', customer_min_max


def data_manipulation(date, time_indicator, order_count, data_source, data_query_path, feature, customer_indicator):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=feature, date=date)
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
    data['prev_orders'] = data['max_order'] - order_count
    data = data.query("order_seq_num > prev_orders")
    data['order_seq_num'] = data.sort_values(by=[customer_indicator, time_indicator]).groupby(
        [customer_indicator]).cumcount() + 1
    data['order_seq_num'] = data.apply(
        lambda row: row['order_seq_num'] + abs(row['prev_orders']) if row['prev_orders'] < 0 else row['order_seq_num'],
        axis=1)
    data, customer_min_max = get_customer_min_max_data(data, feature, customer_indicator)
    data = pivoting_orders_sequence(data, customer_indicator, feature)
    features = list(range(1, order_count))
    return data, features, order_count, customer_min_max, max_date


def order_count_decision(data, order_count, customer_indicator):
    if order_count is None:
        total_orders = []
        for rc in range(5, max(data['max_max_order'])):
            data = pd.merge(data,
                            data.groupby(customer_indicator)['order_seq_num'].max().reset_index().rename(
                                columns={"order_seq_num": "max_order"}),
                            on=customer_indicator, how='left')
            data['prev_orders'] = data['max_order'] - order_count
            total_orders.append({"order_count": rc,
                                 "total_orders": sum([abs(prev_order) for prev_order in list(data['prev_orders'])])})
        order_count = list(pd.DataFrame(total_orders).sort_values(by='total_orders', ascending=True)['order_count'])[0]
    data = data.query("order_seq_num > prev_orders")
    data['prev_orders'] = data['max_order'] - order_count


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


def calculate_time_diff(date, prev_date):
    date = datetime.datetime.strptime(str(date)[0:19], '%Y-%m-%d %H:%M:%S')
    prev_date = datetime.datetime.strptime(str(prev_date)[0:19], '%Y-%m-%d %H:%M:%S')
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


def data_for_customer_prediction(data, params):
    x = pd.DataFrame(np.repeat(data[['time_diff_norm']].values, repeats=params['lag'], axis=1))
    shift_day = int(params['lahead'] / params['lag'])
    if params['lahead'] > 1:
        for i, c in enumerate(x.columns):
            x[c] = x[c].shift(i * shift_day)  # every each same days of shifted
    to_drop = max((params['tsteps'] - 1), (params['lahead'] - 1))
    return x, data, to_drop# reshape_3(x[to_drop:to_drop + 1].values)


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
    for num in range(0, number + 1):
        _pred_data = check_for_next_prediction(data, model_num)
        _pred = model.predict(_pred_data)[0]
        data = add_predicted_values_as_column(data, _pred)
    return data


def get_predicted_data_readable_form(user, prediction, removing_columns, norm_data):
    removing_cols = list(range(removing_columns + 1))
    predictions = [{"user_id": user,
                    "user_min": list(norm_data['user_min'])[0],
                    "user_max": list(norm_data['user_max'])[0],
                    "pred_order_seq": col - removing_columns,
                    "prediction": list(prediction[col])[0]} for col in prediction.columns if col not in removing_cols]
    predictions = pd.DataFrame(predictions)
    predictions['prediction_values'] = predictions.apply(
        lambda row: ((row['user_max'] - row['user_min']) * row['prediction']) + row['user_min'], axis=1)
    return predictions


def merging_predicted_date_to_result_date(results, feuture_orders,
                                           customer_indicator, time_indicator, amount_indicator):
    results = results.rename(columns={"prediction_values": amount_indicator, 'pred_order_seq': 'order_seq_num'})
    results = pd.merge(results,
                       feuture_orders[[customer_indicator, 'order_seq_num', time_indicator]],
                       on=[customer_indicator, 'order_seq_num'], how='left')
    results['data_type'] = 'prediction'
    data_columns = [customer_indicator, 'order_seq_num', time_indicator, amount_indicator, 'data_type']
    return results[data_columns]


reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], 1))
reshape_2 = lambda x: x.reshape((x.shape[0], 1))


def split_data(Y, X, params):
    x_train = reshape_3(X.values)
    y_train = reshape_2(Y.values)
    x_test = reshape_3(X.values)
    y_test = reshape_2(Y.values)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


def drop_calculation(df, parameters, is_prediction=False):
    data_count = len(df)
    to_drop = max((parameters['tsteps'] - 1), (parameters['lahead'] - 1))
    df = df[to_drop:]
    if not is_prediction:
        to_drop = df.shape[0] % parameters['batch_size']
        if to_drop > 0:
            df = df[:-1 * to_drop]
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
    if time_period == 'day':
        return 1
    if time_period == 'year':
        return 365
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


def check_model_exists(path, model_name, time_period):
    current_date = datetime.datetime.strptime(get_current_day(replace=False), "%Y-%m-%d")
    _prev_model_day_diff = 0
    day_range_for_model_training = convert_time_preiod_to_days(time_period)
    prev_model = None
    for m in listdir(dirname(join(path, ""))):
        if "_".join(m.split("_")[:-2]) == model_name:
            _date_str = m.split("_")[-2]
            _date_str = "-".join([_date_str[0:4], _date_str[4:6], _date_str[6:]])
            _date = datetime.datetime.strptime(_date_str, "%Y-%m-%d")
            if abs((_date - current_date).total_seconds()) / 60 / 60 / 24 < day_range_for_model_training:
                if m.split("_")[-1].split(".")[0] == time_period:
                    prev_model = m
    return prev_model


def model_path(directory, model_name, time_period):
    return join(directory, model_name + "_" + get_current_day() + "_" + time_period.replace(" ", "") + ".json")


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


def get_results(directory, time_period):
    results = pd.DataFrame()
    result_files = [f for f in listdir(dirname(join(directory, ""))) if f.split("_")[0] == "results"]
    current_date = min([parse(i.split("_")[2]) for i in result_files])
    date = current_date - datetime.timedelta(days=convert_time_preiod_to_days(time_period))
    detected_file = None
    for f in result_files:
        f_split = f.split("_")
        if f_split[1] == time_period:
            if parse(f_split[3].split(".")[0]) >= date:
                date = parse(f_split[3].split(".")[0])
                detected_file = f
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