import warnings
from pandas import DataFrame, concat
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Ones
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

try:
    from functions import *
    from configs import hyper_conf, accept_threshold_for_loss_diff
    from data_access import *
except Exception as e:
    from .functions import *
    from .configs import hyper_conf, accept_threshold_for_loss_diff
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
        self.directory = directory
        self.customer_indicator = customer_indicator
        self.time_indicator = time_indicator
        self.params = hyper_conf('next_purchase')
        self.hyper_params = get_tuning_params(hyper_conf('next_purchase_hyper'), self.params)
        self.optimized_parameters = {}
        self._p = None
        self.order_count = order_count
        self.time_period = time_period
        self.data, self.features, self.c_min_max, self.params = data_manipulation_np(date=date,
                                                                                     feature=amount_indicator,
                                                                                     time_indicator=time_indicator,
                                                                                     data_source=data_source,
                                                                                     data_query_path=data_query_path,
                                                                                     customer_indicator=customer_indicator,
                                                                                     params=self.params,
                                                                                     directory=directory)

        self.hp = HyperParameters()
        self.model_data = {}
        self.input, self.model = None, None
        self.model = check_model_exists(self.directory, "trained_next_purchase_model", self.time_period)
        self.result = None, None
        self.residuals, self.anomaly = [], []
        self.customers = list(self.data[customer_indicator].unique())
        self.results = DataFrame()
        self.get_actual_value = lambda _min, _max, _value: ((_max - _min) * _value) + _min
        self.max_date = max(self.data[self.time_indicator])
        self.future_date = self.max_date + datetime.timedelta(days=convert_time_preiod_to_days(self.time_period))
        self.model_data = {"x_train": None, "y_train": None, "x_test": None, "y_test": None}
        self.client_sample_sizes = []

    def get_model_data(self, customer):
        _data = self.data[(self.data[self.customer_indicator] == customer) & (self.data['last_recency'] == 0)]
        try:
            data = arrange__data_for_model(df=_data, f=[self.features], parameters=self.params)
            if data['x_train'].shape[0] != 0:
                model_data[customer] = {}
                model_data[customer] = data
        except Exception as e:
            print(e)

    def data_preparation(self):
        global model_data
        model_data = {}
        print("number of customers :", len(self.customers))
        execute_parallel_run(self.customers, self.get_model_data, arguments=None, parallel=16)
        try_count = 0
        while try_count < 20:
            try:
                self.model_data = {"x_train": None, "y_train": None, "x_test": None, "y_test": None}
                self.client_sample_sizes = []
                for c in model_data:
                    if self.model_data['x_train'] is not None:
                        self.model_data['x_train'] = np.concatenate([self.model_data['x_train'], model_data[c]['x_train']])
                        self.model_data['y_train'] = np.concatenate([self.model_data['y_train'], model_data[c]['y_train']])
                        self.model_data['x_test'] = np.concatenate([self.model_data['x_test'], model_data[c]['x_test']])
                        self.model_data['y_test'] = np.concatenate([self.model_data['y_test'], model_data[c]['y_test']])
                        self.client_sample_sizes.append(model_data[c]['x_train'].shape[0])
                    else:
                        self.client_sample_sizes.append(model_data[c]['x_train'].shape[0])
                        self.model_data = model_data[c]
                try_count = 20
            except Exception as e:
                print(e)
                try_count += 1

    def build_parameter_tuning_model(self, hp):
        self.input = Input(shape=(self.model_data['x_train'].shape[1], 1))
        lstm = LSTM(int(hp.Choice('units', self.hyper_params['units'])),
                    bias_initializer=Ones(),
                    kernel_initializer=Ones(),
                    use_bias=False,
                    activation=hp.Choice('activation', self.hyper_params['activation']),
                    dropout=0.1
                    )(self.input)
        lstm = BatchNormalization()(lstm)
        lstm = Dense(1)(lstm)
        model = Model(inputs=self.input, outputs=lstm)
        model.compile(loss='mae',
                      optimizer=Adam(lr=hp.Choice('lr', self.hyper_params['lr'])),
                      metrics=['mae'])
        return model

    def build_model(self):
        self.input = Input(shape=(self.model_data['x_train'].shape[1], 1))
        # LSTM layer
        lstm = LSTM(self.params['units'],
                    batch_size=self.params['batch_size'],
                    bias_initializer=Ones(),
                    kernel_initializer=Ones(),
                    use_bias=False,
                    recurrent_activation=self.params['activation'],
                    dropout=0.1
                    )(self.input)
        lstm = BatchNormalization()(lstm)
        lstm = Dense(1)(lstm)
        self.model = Model(inputs=self.input, outputs=lstm)
        self.model.compile(loss='mae', optimizer=Adam(lr=self.params['lr']), metrics=['mae'])

    def learning_process(self, save_model=True, history=False):
        if history:
            history = self.model.fit(self.model_data['x_train'],
                                     self.model_data['y_train'],
                                     batch_size=self.params['batch_size'],
                                     epochs=int(self.params['epochs']),
                                     verbose=1,
                                     validation_split=1 - self.params['split_ratio'],
                                     shuffle=True)
        else:
            self.model.fit(self.model_data['x_train'],
                           self.model_data['y_train'],
                           batch_size=self.params['batch_size'],
                           epochs=int(self.params['epochs']),
                           verbose=1,
                           validation_split=1 - self.params['split_ratio'],
                           shuffle=True)
        if save_model:
            model_from_to_json(path=model_path(self.directory,
                                               "trained_next_purchase_model", self.time_period),
                               model=self.model,
                               is_writing=True)

        if history:
            return history

    def train_execute(self):
        print("*"*5, "Next purchase train model process ", "*"*5)
        if self.model is None:
            self.data_preparation()
            self.parameter_tuning()
            self.build_model()
            self.learning_process()
        else:
            self.model = model_from_to_json(path=join(self.directory, self.model))
            print(self.model)
            print("Previous model already exits in the given directory  '" + self.directory + "'.")

    def prediction_per_customer(self,  customer):
        warnings.simplefilter("ignore")
        _norm_data = self.c_min_max[self.c_min_max[self.customer_indicator] == customer].iloc[0:1]
        start = self.max_date
        end = self.future_date
        _max_date = end - datetime.timedelta(days=1)
        _pred_data = DataFrame()
        counter = 0
        prediction_data[customer] = _pred_data
        while start < _max_date < end:
            x, _data, to_drop = data_for_customer_prediction(
                self.data[self.data[self.customer_indicator] == customer], self.params)
            if len(reshape_3(x[to_drop:].values)) != 0:
                _pred = self.model.predict(reshape_3(x[to_drop:].values))[0][-1]
                if _pred != 0:  # don`t waste time if prediction is O
                    _pred_actual = self.get_actual_value(_min=list(_norm_data['user_min'])[0],
                                                         _max=list(_norm_data['user_max'])[0],
                                                         _value=_pred)
                    if self.time_period != 'hour':
                        _predicted_date = max(_data[self.time_indicator]) + datetime.timedelta(days=min(int(max(_pred_actual, 0)), 365 * 10))
                    else:
                        _predicted_date = max(_data[self.time_indicator]) + \
                                          datetime.timedelta(minutes=min(int(max(_pred_actual, 0) * 3600), 365 * 10))
                    if counter > 2:
                        _max_date = end + datetime.timedelta(days=1)
                    if start < _predicted_date < end:
                        _pred_data = concat([_pred_data, DataFrame([{self.time_indicator: _predicted_date,
                                                                     self.customer_indicator: customer,
                                                                     'time_diff': _pred_actual,
                                                                     'time_diff_norm': _pred}])])
                    counter += 1
                else:
                    _max_date = end + datetime.timedelta(days=1)
            else:
                _max_date = end + datetime.timedelta(days=1)
        prediction_data[customer] = _pred_data

    def prediction_execute(self):
        print("*"*5, "PREDICTION", 5*"*")
        print("number of users :", len(self.customers))
        if self.model is not None:
            self.model = model_from_to_json(path=join(self.directory, self.model))
        self.results = self.data[[self.time_indicator, 'time_diff', 'time_diff_norm', self.customer_indicator]]
        global prediction_data
        prediction_data = {}
        execute_parallel_run(self.customers, self.prediction_per_customer, arguments=None, parallel=8)
        print("merge predicted data ...")

        for c in self.customers:
            try:
                self.results = concat([self.results, prediction_data[c]])
            except Exception as e:
                time.sleep(2)
                print(c)
            # self.prediction_per_customer(c)
            # self.results = concat([self.results, prediction_data[c]])
        print("number of total predicted values")
        print(len(self.results))
        self.results['max_date'], self.results['future_date'] = self.max_date, self.future_date
        self.results = self.results[(self.results[self.time_indicator] > self.results['max_date']) &
                                    (self.results[self.time_indicator] < self.results['future_date'])]
        print(self.results.head())

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'next_purchase') is None:
            print(""*3, "batch size optimization", "*"*3)
            print("number of sample client size:", len(self.client_sample_sizes))
            print("average of sample client size:", np.mean(self.client_sample_sizes))
            self.params['batch_size'] = batch_size_optimization(self.client_sample_sizes, len(self.customers))
            tuner = RandomSearch(
                                 self.build_parameter_tuning_model,
                                 max_trials=5,
                                 hyperparameters=self.hp,
                                 allow_new_entries=True,
                                 objective='loss')
            tuner.search(x=self.model_data['x_train'],
                         y=self.model_data['y_train'],
                         epochs=5,
                         batch_size=self.params['batch_size'],
                         verbose=1,
                         validation_data=(self.model_data['x_test'], self.model_data['y_test']))
            for p in tuner.get_best_hyperparameters()[0].values:
                if p in list(self.params.keys()):
                    self.params[p] = tuner.get_best_hyperparameters()[0].values[p]
            counter = 0
            optimum_epoch_process_done = False
            while not optimum_epoch_process_done:
                self.params['epochs'] = int(self.hyper_params['epochs'][counter])
                self.build_model()
                _history = self.learning_process(save_model=False, history=True)
                if _history.history['loss'][-1] < accept_threshold_for_loss_diff:
                    optimum_epoch_process_done = True
                counter += 1
            shutil.rmtree(join(abspath(__file__).split("next_purchase_model.py")[0].split("clv")[0][:-1],
                               "clv_prediction", "untitled_project"))

            try:
                _params = read_yaml(self.directory, "test_parameters.yaml")
                _params['next_purchase'] = self.params
                write_yaml(self.directory, "test_parameters.yaml", _params, ignoring_aliases=True)
            except Exception as e:
                print(e)
                print("Non of parameter tuning for both Model has been observed.")
                write_yaml(self.directory, "test_parameters.yaml", {'next_purchase': self.params}, ignoring_aliases=True)
        else:
            self.params = check_for_existing_parameters(self.directory, 'next_purchase')




