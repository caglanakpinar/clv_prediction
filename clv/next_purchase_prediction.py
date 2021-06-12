import argparse
import warnings
from pandas import DataFrame, concat
import os
import shutil

from tensorflow.keras.models import model_from_json

try:
    from functions import *
    from configs import hyper_conf, accept_threshold_for_loss_diff, parameter_tuning_trials
    from data_access import *
except Exception as e:
    from .functions import *
    from .configs import hyper_conf, accept_threshold_for_loss_diff, parameter_tuning_trials
    from .data_access import *


def model_from_to_json(path=None, weights_path=None, model=None, is_writing=False, lr=None):
    if is_writing:
        model_json = model.to_json()
        with open(path, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(weights_path)
    else:
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        try:
            model.load_weights(weights_path)
        except Exception as e:
            model.load_weights(weights_path)
        return model


def updating_hyper_parameters_related_to_data():
    return None


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    return params


class NextPurchaseModelPrediction:
    def __init__(self,
                 temp_data_path,
                 max_date,
                 future_date,
                 time_period,
                 directory,
                 indicators
                 ):
        self.temp_data = pd.read_csv(temp_data_path)
        self.max_date = convert_str_to_day(max_date)
        self.future_date = convert_str_to_day(future_date)
        self.directory = directory
        self.customer_indicator, self.amount_indicator, self.time_indicator = indicators.split("*")
        self.customers = list(self.temp_data[self.customer_indicator].unique())
        self.params = hyper_conf('next_purchase') \
            if check_for_existing_parameters(self.directory,'next_purchase') is None else \
            check_for_existing_parameters(self.directory, 'next_purchase')
        self.time_period = time_period
        self.get_actual_value = lambda _min, _max, _value: ((_max - _min) * _value) + _min
        self.prev_model_date = check_model_exists(self.directory, "trained_next_purchase_model", self.time_period)
        self.model = model_from_to_json(path=model_path(self.directory,
                                                        "trained_next_purchase_model",
                                                        self.prev_model_date if self.prev_model_date is not None else get_current_day(),
                                                        self.time_period),
                                        weights_path=weights_path(self.directory,
                                                                  "trained_next_purchase_model",
                                                                  self.prev_model_date if self.prev_model_date is not None else get_current_day(),
                                                                  self.time_period), lr=self.params['lr'])

    def prediction_date_add(self, data, pred_data, pred):
        max_date = max(data[self.time_indicator]) if len(pred_data) == 0 else max(pred_data[self.time_indicator])
        return max_date + datetime.timedelta(days=int(round(pred)))

    def prediction_condition_1(self, data, _pred_actual):
        accept = False
        try:
            if self.max_date < self.prediction_date_add(data, pd.DataFrame(), (_pred_actual * 3) + 0.05):
                accept = True
        except: accept = False
        return accept

    def prediction_condition_2(self, _pred_actual, _pred):
        try: return True if int(round(_pred_actual)) != 0 and _pred > 0 else False
        except: return False

    def predicted_date_in_range_decision(self, end, _pred_data, _predicted_date, customer, _pred_actual, _pred):
        if _predicted_date < end:
            _pred_data = concat([_pred_data, DataFrame([{self.time_indicator: _predicted_date,
                                                         self.customer_indicator: customer,
                                                         'time_diff': _pred_actual,
                                                         'time_diff_norm': _pred}])])
        return _pred_data

    def calculate_prediction(self, data, _pred_data, user_min, user_max):
        x = data_for_customer_prediction(data, _pred_data, self.params)
        try: _pred = self.model.predict(x)[0][-1]
        except: _pred = 0
        _pred_actual = self.get_actual_value(_min=user_min, _max=user_max, _value=_pred)
        return _pred_actual, _pred

    def prediction_per_customer(self,  customer):
        warnings.simplefilter("ignore")
        data = self.temp_data[self.temp_data[self.customer_indicator] == customer]
        user_min, user_max = [list(data[metric])[0] for metric in ['user_min', 'user_max']]
        _pred_data, prediction_data[customer] = DataFrame(), DataFrame()
        _predicted_date = self.future_date - datetime.timedelta(days=1)
        _pred_actual, _pred = self.calculate_prediction(data, _pred_data, user_min, user_max)
        if self.prediction_condition_2(_pred_actual, _pred) and self.prediction_condition_1(data, _pred_actual):
            counter = 0
            while self.max_date < _predicted_date < self.future_date:
                try:
                    _pred_actual, _pred = self.calculate_prediction(data, _pred_data, user_min, user_max)
                    if self.prediction_condition_2(_pred_actual, _pred) and counter < 190:
                        counter += 1
                        _predicted_date = self.prediction_date_add(data, _pred_data, _pred_actual)
                        _pred_data = self.predicted_date_in_range_decision(
                            self.future_date, _pred_data, _predicted_date, customer, _pred_actual, _pred)
                    else: _predicted_date = self.future_date + datetime.timedelta(days=1)
                except Exception as e: _predicted_date = self.future_date + datetime.timedelta(days=1)

            if len(_pred_data) != 0:
                try: _pred_data = _pred_data[(_pred_data[self.time_indicator] >= self.max_date) &
                                             (_pred_data[self.time_indicator] < self.future_date) &
                                             (_pred_data[self.time_indicator] == _pred_data[self.time_indicator])]
                except Exception as e: print(e)
            prediction_data[customer] = _pred_data
        del data, user_max, user_min

    def prediction_execute(self):
        print("*"*5, "PREDICTION", 5*"*")
        print("number of users :", len(self.customers))
        self.temp_data[self.time_indicator] = self.temp_data[self.time_indicator] .apply(lambda x: convert_str_to_day(x))
        self.temp_data = self.temp_data.sort_values(by=[self.customer_indicator, self.time_indicator], ascending=True)
        global prediction_data
        prediction_data = {}
        execute_parallel_run(self.customers, self.prediction_per_customer, arguments=None, parallel=4)
        _result = []
        for c in self.customers:
            try: _result.append(prediction_data[c])
            except Exception as e: print(c)
        pd.concat(_result).to_csv(
            join(self.directory, "temp_next_purchase_results", str(self.customers[0] + "_data.csv")), index=False)


if __name__ == '__main__':
    """
                -P    temp_data_path
                -MD   max_date
                -FD   future_date
                -TP   time_period
                -D    directory
                -IND  customer_indicator, amount_indicator, time_indicator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--temp_data_path", type=str,
                        help="""
                                train, prediction
                        """,
                        )
    parser.add_argument("-C", "--customers", type=str,
                        help="""
                                user_1*user_2*user_4*user_5* ...
                        """,
                        )
    parser.add_argument("-MD", "--max_date", type=str,
                        help="""
                                2021-05-05 (string)
                        """,
                        )
    parser.add_argument("-FD", "--future_date", type=str,
                        help="""
                                2021-11-05 (string)
                        """,
                        )
    parser.add_argument("-D", "--directory", type=str,
                        help="""
                                    /../../..
                            """,
                        )
    parser.add_argument("-TP", "--time_period", type=str,
                        help="""
                                    week, day, 6*months, quarter, ..
                            """,
                        )

    parser.add_argument("-IND", "--indicators", type=str,
                        help="""
                                    customer_indicator*amount_indicator*time_indicator
                            """,
                        )

    arguments = parser.parse_args()
    args = {'temp_data_path': arguments.temp_data_path,
            'max_date': arguments.max_date,
            'future_date': arguments.future_date,
            'directory': arguments.directory,
            'time_period': arguments.time_period,
            'indicators': arguments.indicators}
    prediction = NextPurchaseModelPrediction(**args)
    prediction.prediction_execute()
    del prediction



