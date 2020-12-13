import numpy as np
import pandas as pd

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
    return data, features, order_count, customer_min_max


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
    if time_period == '2_weeks':
        return 14
    if time_period == '2_months':
        return 60
    if time_period == 'quarter':
        return 90
