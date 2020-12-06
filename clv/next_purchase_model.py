from pandas import DataFrame, concat
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import traceback
from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import Ones
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

try:
    from functions import *
    from configs import conf, boostrap_ratio, iteration, hyper_conf
    from data_access import *
except Exception as e:
    from .functions import *
    from .configs import conf, boostrap_ratio, iteration, hyper_conf
    from .data_access import *


def model_from_to_json(path=None, model=None, is_writing=False):
    if is_writing:
        model_json = model.to_json()
        with open(path, "w") as json_file:
            json_file.write(model_json)
    else:
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return model_from_json(loaded_model_json)


def updating_hyper_parameters_related_to_data():
    return None


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
        if type(parameter_tuning[p]) == list:
            hyper_params[p] = parameter_tuning[p]
        if type(parameter_tuning[p]) == str:
            if "*" in parameter_tuning[p]:
                # e.g. 0.1*0.5 return [0.1, 0.2, ..., 0.5] or 0.1*0.5*0.05 return [0.1, 0.15, ..., 0.5]
                _splits = parameter_tuning[p].split("*")
                print(_splits)
                if len(_splits) == 2:
                    hyper_params[p] = np.arange(float(_splits[0]), float(_splits[1]), float(_splits[0])).tolist()
                if len(_splits) == 3:
                    hyper_params[p] = np.arange(float(_splits[0]), float(_splits[1]), float(_splits[2])).tolist()
            else:  # e.g. relu_tanh or relu
                hyper_params[p] = parameter_tuning[p].split("_")
    for p in params:
        if p not in list(hyper_params.keys()):
            hyper_params[p] = params[p]
    return hyper_params


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    return params


class TrainLSTM:
    def __init__(self,
                 date=None,
                 time_indicator=None,
                 order_count=None,
                 data_source=None,
                 data_query_path=None,
                 time_period=None,
                 directory=None,
                 customer_indicator=None,
                 amount_indicator=None):
        self.params = hyper_conf('next_purchase')
        print(self.params)
        ## TODO: hyper parameters of ranges must be updated related to data
        self.hyper_params = get_tuning_params(hyper_conf('next_purchase_hyper'), self.params)
        self.optimized_parameters = {}
        self._p = None
        self.order_count = order_count
        self.time_period = time_period
        self.data, self.features, self.c_min_max = data_manipulation_np(date=date,
                                                                        feature=amount_indicator,
                                                                        time_indicator=time_indicator,
                                                                        order_count=order_count,

                                                                        data_source=data_source,
                                                                        data_query_path=data_query_path,
                                                                        customer_indicator=customer_indicator)
        self.customer_indicator = customer_indicator
        self.time_indicator = time_indicator
        self.data_count = 0
        self.hp = HyperParameters()
        self.model_data = {}
        self.input, self.model = None, None
        self.model = None
        self.result = None, None
        self.residuals, self.anomaly = [], []
        self.directory = directory
        self.customers = list(self.data[customer_indicator].unique())
        self.results = DataFrame()
        self.get_actual_value = lambda _min, _max, _value: ((_max - _min) * _value) + _min
        self.max_date = max(self.data[self.time_indicator])
        self.future_date = self.max_date + datetime.timedelta(days=convert_time_preiod_to_days(self.time_period))
        self.model_data = {"x_train": None, "y_train": None, "x_test": None, "y_test": None}

    def get_model_data(self, customer):
        _data = self.data[(self.data[self.customer_indicator] == customer) & (self.data['last_recency'] == 0)]
        # try:
        data = arrange__data_for_model(df=_data, f=[self.features], parameters=self.params)
        if data['x_train'].shape[0] != 0:
            model_data[customer] = {}
            print(data)
            model_data[customer] = arrange__data_for_model(df=_data, f=[self.features], parameters=self.params)

        # except Exception as e:
        #    print(e)

    def data_preparation(self):
        global model_data
        model_data = {}
        execute_parallel_run(self.customers, self.get_model_data, arguments=None)
        print(model_data)
        for c in model_data:
            if self.model_data['x_train'] is not None:
                self.model_data['x_train'] = np.concatenate([self.model_data['x_train'], model_data[c]['x_train']])
                self.model_data['y_train'] = np.concatenate([self.model_data['y_train'], model_data[c]['y_train']])
                self.model_data['x_test'] = np.concatenate([self.model_data['x_test'], model_data[c]['x_test']])
                self.model_data['y_test'] = np.concatenate([self.model_data['y_test'], model_data[c]['y_test']])
            else:
                self.model_data = model_data[c]

    def build_parameter_tuning_model(self, hp):
        self.input = Input(shape=(self.model_data['x_train'].shape[1], 1))
        lstm = LSTM(hp.Choice('units', self.hyper_params['units']),
                    use_bias=False,
                    activation=hp.Choice('activation', self.hyper_params['activation']),
                    batch_size=hp.Choice('batch_size', self.hyper_params['batch_size']),
                    kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                             l2=hp.Choice('l2', self.hyper_params['l2'])),
                    bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                    activity_regularizer=l2(hp.Choice('l', self.hyper_params['l2']))
                    )(self.input)
        lstm = BatchNormalization()(lstm)
        lstm = Dense(1)(lstm)
        model = Model(inputs=self.input, outputs=lstm)
        model.compile(loss='mae',
                      optimizer=Adam(lr=hp.Choice('lr', self.hyper_params['lr'])),
                      metrics=['mae'])
        return model

    def build_model(self):
        self.input = Input(shape=(self.model_data['train_x']['x_train'].shape[1], 1))
        # LSTM layer
        lstm = LSTM(self.params['units'],
                    batch_size=self.params['batch_size'],
                    recurrent_initializer=Ones(),
                    kernel_initializer=Ones(),
                    use_bias=False,
                    recurrent_activation=self.params['activation'],
                    dropout=0.1
                    )(self.input)
        lstm = BatchNormalization()(lstm)
        lstm = Dense(1)(lstm)
        self.model = Model(inputs=self.input, outputs=lstm)
        self.model.compile(loss='mae', optimizer=Adam(lr=self.params['lr']), metrics=['mae'])

    def learning_process(self, save_model=True):
        self.model.fit(self.model_data['x_train'],
                       self.model_data['y_train'],
                       batch_size=self.params['batch_size'],
                       epochs=int(self.params['epochs']),
                       verbose=1,
                       validation_split=1 - self.params['split_ratio'],
                       shuffle=True)
        if save_model:
            model_from_to_json(path=join(self.directory, "trained_next_purchase_model"),
                               model=self.model,
                               is_writing=True)

    def train_execute(self):
        print("*"*5, " train model process ", "*"*5)
        self.data_preparation()
        self.parameter_tuning()
        self.learning_process()

    def prediction_per_customer(self, customer):
        _norm_data = self.c_min_max[self.c_min_max[self.customer_indicator] == customer].iloc[0:1]
        start = self.max_date
        end = self.future_date
        _max_date = end - datetime.timedelta(days=1)
        _pred_data = DataFrame()
        counter = 0
        print("yessss")
        while start < _max_date < end:
            # try:
            print("yessss")
            x, _data, to_drop = data_for_customer_prediction(
                self.data[self.data[self.customer_indicator] == customer], self.params)
            if len(reshape_3(x[to_drop:].values)) != 0:
                _pred = self.model.predict(reshape_3(x[to_drop:].values))[0][-1]
                _pred_actual = self.get_actual_value(_min=list(_norm_data['user_min'])[0],
                                                     _max=list(_norm_data['user_max'])[0],
                                                     _value=_pred)
                _predicted_date = max(_data[self.time_indicator]) + datetime.timedelta(minutes=int(_pred_actual * 3600))
                if counter > 2:
                    _max_date = _predicted_date
                if start < _predicted_date < end:
                    print("predicted date :", _predicted_date,
                          "|| predicted frequency :", int(_pred_actual * 3600))
                _pred_data = concat([_pred_data, DataFrame([{'created_date': _predicted_date,
                                                             'user_id': self.customer_indicator,
                                                             'time_diff': _pred_actual,
                                                             'time_diff_norm': _pred}])])
                counter += 1
            else:
                _max_date = end + datetime.timedelta(days=1)
            # except Exception as e:
            #    print(e)
        prediction_data[customer] = _pred_data

    def prediction_execute(self):
        print("*"*5, "PREDICTION", 5*"*")
        print("number of users :", len(self.customers))
        self.results = self.data[[self.time_indicator, 'time_diff', 'time_diff_norm']]
        global prediction_data
        prediction_data = {}
        execute_parallel_run(self.customers, self.prediction_per_customer, arguments=None)
        for c in prediction_data:
            self.results = concat([self.results, prediction_data[c]])
        self.results['max_date'], self.results['future_date'] = self.max_date, self.future_date
        self.results = self.results[(self.results[self.time_indicator] > self.results['max_date']) &
                                    (self.results[self.time_indicator] < self.results['future_date'])]

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'next_purchase_model') is None:
            tuner = RandomSearch(
                                 self.build_parameter_tuning_model,
                                 max_trials=5,
                                 hyperparameters=self.hp,
                                 allow_new_entries=True,
                                 objective='loss')
            tuner.search(x=self.model_data['x_train'],
                         y=self.model_data['y_train'],
                         epochs=5,
                         verbose=1,
                         validation_data=(self.model_data['x_test'], self.model_data['y_test']))
            self.model = tuner.get_best_models(num_models=10)[0]
            for p in tuner.get_best_hyperparameters()[0].values:
                if p in list(self.params.keys()):
                    self.params[p] = tuner.get_best_hyperparameters()[0].values[p]
            # self.params = tuner.get_best_hyperparameters()[0].values
            # self.params['epochs'] = hyper_conf('next_purchase')['epochs']
            # self.params['split_ratio'] = hyper_conf('next_purchase')['split_ratio']
            write_yaml(self.directory, "test_parameters.yaml", self.params, ignoring_aliases=True)
        else:
            self.params = check_for_existing_parameters(self.directory, 'purchase_amount')




