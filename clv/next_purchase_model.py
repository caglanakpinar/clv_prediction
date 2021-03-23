import warnings
from pandas import DataFrame, concat
import os
import shutil
from itertools import product
import glob
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
    from configs import hyper_conf, accept_threshold_for_loss_diff, parameter_tuning_trials
    from data_access import *
except Exception as e:
    from .functions import *
    from .configs import hyper_conf, accept_threshold_for_loss_diff, parameter_tuning_trials
    from .data_access import *


def model_from_to_json(path=None, weights_path=None, model=None, is_writing=False):
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
        self.amount_indicator = amount_indicator
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
                                                                                     time_period=time_period,
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
        self.optimum_batch_size = 32

    def get_model_data(self, customer):
        _data = self.data[self.data[self.customer_indicator] == customer]
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
                        for _split in ['x_train', 'y_train', 'x_test', 'y_test']:
                            self.model_data[_split] = np.concatenate([self.model_data[_split], model_data[c][_split]])
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
                                     validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                                     shuffle=True)
        else:
            print("*"*5, "Fit Next Purchase Model", "*"*5)
            self.model.fit(self.model_data['x_train'],
                           self.model_data['y_train'],
                           batch_size=self.params['batch_size'],
                           epochs=int(self.params['epochs']),
                           verbose=1,
                           validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                           shuffle=True)
        if save_model:
            model_from_to_json(path=model_path(self.directory,
                                               "trained_next_purchase_model", self.time_period),
                               weights_path = weights_path(self.directory,
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
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_next_purchase_model", self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_next_purchase_model", self.time_period))
            print("Previous model already exits in the given directory  '" + self.directory + "'.")

    def prediction_date_add(self, data, pred_data, pred):
        max_date = max(data[self.time_indicator]) if len(pred_data) == 0 else max(pred_data[self.time_indicator])
        if self.time_period != 'hour':
            _predicted_date = max_date + datetime.timedelta(minutes=int(round(pred)))
        if self.time_period != 'day':
            _predicted_date = max_date + datetime.timedelta(hours=int(round(pred)))
        if self.time_period not in ['day', 'hour']:
            _predicted_date = max_date + datetime.timedelta(days=int(round(pred)))
        return _predicted_date

    def predicted_date_in_range_decision(self, end, _pred_data, _predicted_date, customer, _pred_actual, _pred):
        if _predicted_date < end:
            _pred_data = concat([_pred_data, DataFrame([{self.time_indicator: _predicted_date,
                                                         self.customer_indicator: customer,
                                                         'time_diff': _pred_actual,
                                                         'time_diff_norm': _pred}])])
        return _pred_data

    def calculate_prediction(self, data, _pred_data, user_min, user_max):
        try:
            x = data_for_customer_prediction(data, _pred_data, self.params)
            _pred = self.model.predict(x)[0][-1]
            _pred_actual = self.get_actual_value(_min=user_min, _max=user_max, _value=_pred)
        except Exception as e:
            print(e)
            _pred_actual, _pred = 0, 0
        return _pred_actual, _pred

    def prediction_per_customer(self,  customer):
        warnings.simplefilter("ignore")
        user_min = self.temp_data[customer]['user_min']
        user_max = self.temp_data[customer]['user_max']
        data = pd.DataFrame(self.temp_data[customer]['customer_transactions']
                            ).rename(columns={0: self.time_indicator,
                                              1: self.amount_indicator,
                                              2: 'time_diff_norm',
                                              3: 'time_diff'})
        _pred_data, prediction_data[customer] = DataFrame(), DataFrame()
        _predicted_date = self.future_date - datetime.timedelta(days=1)
        _pred_actual, _pred = self.calculate_prediction(data, _pred_data, user_min, user_max)
        if self.max_date < self.prediction_date_add(data, pd.DataFrame(), (_pred_actual * 3) + 0.05):
            while self.max_date < _predicted_date < self.future_date:
                try:
                    _pred_actual, _pred = self.calculate_prediction(data, _pred_data, user_min, user_max)
                    if int(round(_pred_actual)) != 0:
                        _predicted_date = self.prediction_date_add(data, _pred_data, _pred_actual)
                        _pred_data = self.predicted_date_in_range_decision(self.future_date,
                                                                           _pred_data,
                                                                           _predicted_date,
                                                                           customer,
                                                                           _pred_actual,
                                                                           _pred)
                    else:
                        _predicted_date = self.future_date + datetime.timedelta(days=1)
                except Exception as e:
                    _predicted_date = self.future_date + datetime.timedelta(days=1)

            if len(_pred_data) != 0:
                try:
                    _pred_data = _pred_data[(_pred_data[self.time_indicator] >= self.max_date) &
                                            (_pred_data[self.time_indicator] < self.future_date) &
                                            (_pred_data[self.time_indicator] == _pred_data[self.time_indicator])]
                except Exception as e:
                    print(e)
            prediction_data[customer] = _pred_data

    def create_prediction_data(self):
        self.temp_data = self.data.groupby(self.customer_indicator).agg(
            {self.time_indicator: lambda x: list(x),
             self.amount_indicator: lambda x: list(x),
             "time_diff_norm": lambda x: list(x), 'time_diff': lambda x: list(x)}).reset_index()
        self.temp_data['customer_transactions'] = self.temp_data.apply(
            lambda row: list(zip(row[self.time_indicator], row[self.amount_indicator],
                                 row['time_diff_norm'], row['time_diff'])), axis=1)
        self.temp_data = self.temp_data.drop([self.time_indicator, self.amount_indicator], axis=1)
        self.temp_data = pd.merge(self.temp_data, self.c_min_max, on=self.customer_indicator, how='left')
        self.temp_data = {i[1]: i[0] for i in zip(self.temp_data.set_index(self.customer_indicator).to_dict('results'),
                                                  list(self.temp_data[self.customer_indicator]))}

    def parallel_prediction(self):
        try:
            os.mkdir(join(self.directory, "temp_next_purchase_results"))
        except Exception as e:
            print(e)
            print("recreating 'temp_next_purchase_results' folder ...")
            shutil.rmtree(join(self.directory, "temp_next_purchase_results", ""))
            os.mkdir(join(self.directory, "temp_next_purchase_results"))

        iters = int(len(self.customers) / 256) + 1
        for i in range(iters):
            print("main iteration :", str(i), " / ", str(iters))
            _sample_customers = get_iter_sample(self.customers, i, iters, 256)
            global prediction_data
            prediction_data = {}
            execute_parallel_run(_sample_customers, self.prediction_per_customer, arguments=None, parallel=8)
            _result = pd.DataFrame()
            for c in _sample_customers:
                try:
                    _result = concat([_result, prediction_data[c]])
                except Exception as e:
                    print(c)
            _result.to_csv(join(self.directory, "temp_next_purchase_results", str(_sample_customers[0] + "_data.csv")),
                           index=False)

    def create_prediction_result_data(self):
        print("merge predicted data ...")
        _import_files = glob.glob(join(self.directory, "temp_next_purchase_results", "*.csv"))
        li = []
        for f in _import_files:
            try:
                _df = pd.read_csv(f, index_col=None, header=0)
                li.append(_df)
            except Exception as e:
                print(e)
        self.results = concat(li, axis=0, ignore_index=True)
        self.results[self.time_indicator] = self.results[self.time_indicator].apply(lambda x: convert_str_to_day(x))
        print(self.results.head())
        shutil.rmtree(join(self.directory, "temp_next_purchase_results", ""))
        self.results = self.results[(self.results[self.time_indicator] > self.max_date) &
                                    (self.results[self.time_indicator] < self.future_date)]
        print(self.results.head())
        print("number of predicted customers :", len(self.results[self.customer_indicator].unique()))

    def prediction_execute(self):
        print("*"*5, "PREDICTION", 5*"*")
        print("number of users :", len(self.customers))
        if self.model is not None:
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_next_purchase_model", self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_next_purchase_model", self.time_period))
        self.results = self.data[[self.time_indicator, 'time_diff', 'time_diff_norm', self.customer_indicator]]
        self.create_prediction_data()
        self.parallel_prediction()
        self.create_prediction_result_data()

    def initialize_keras_tuner(self):
        """
        Parameter tuning process is triggered via Keras-Turner Library.
        However, batch_size and epoch parameters of optimization are created individually.
        """

        self.hyper_params, self.optimum_batch_size = batch_size_hp_ranges(client_sample_sizes=self.client_sample_sizes,
                                                                          num_of_customers=len(self.customers),
                                                                          hyper_params=self.hyper_params)
        tuner = RandomSearch(
            self.build_parameter_tuning_model,
            max_trials=parameter_tuning_trials,
            hyperparameters=self.hp,
            allow_new_entries=True,
            objective='loss')
        tuner.search(x=self.model_data['x_train'],
                     y=self.model_data['y_train'],
                     epochs=5,
                     batch_size=self.optimum_batch_size,
                     verbose=1,
                     validation_data=(self.model_data['x_test'], self.model_data['y_test']))
        for p in tuner.get_best_hyperparameters()[0].values:
            if p in list(self.params.keys()):
                self.params[p] = tuner.get_best_hyperparameters()[0].values[p]

    def batch_size_and_epoch_tuning(self):
        """
        It finds the optimum batch_size and epoch.
        Each chosen epoch and batch_size of model created via model.
        Last epoch of loss is lower than 'accept_threshold_for_loss_diff' decision be finalized.
        Epoch will test ascending format (e.g. 4, 8, 16, ..., 1024) in order to minimize time consumption
        of both parameter tuning and model creation.
        Batch size will test descending format (e.g. 1024, 512, ...4) in order to minimize time consumption
        of both parameter tuning and model creation.
        """

        counter = 0
        optimum_epoch_process_done = False
        epoch_bs_combs = list(product(sorted(self.hyper_params['batch_sizes'], reverse=True),
                                      sorted(self.hyper_params['epochs'])
                                      ))
        loss_values = []
        while not optimum_epoch_process_done:
            self.params['epochs'] = int(epoch_bs_combs[counter][1])
            self.params['batch_size'] = int(epoch_bs_combs[counter][0])
            self.build_model()
            _history = self.learning_process(save_model=False, history=True)
            if _history.history['loss'][-1] < accept_threshold_for_loss_diff:
                optimum_epoch_process_done = True
            loss_values.append({"epochs": self.params['epochs'],
                                "batch_size": self.params['batch_size'],
                                "loss": _history.history['loss'][-1]})
            counter += 1
            if counter >= parameter_tuning_trials:
                optimum_epoch_process_done = True
                loss_values_df = pd.DataFrame(loss_values).sort_values(by='loss', ascending=True)
                self.params['epochs'] = list(loss_values_df['epochs'])[0]
                self.params['batch_size'] = list(loss_values_df['batch_size'])[0]

    def remove_keras_tuner_folder(self):
        """
        removing keras tuner file. while you need to update the parameters it will affect rerun the parameter tuning.
        It won`t start unless the folder has been removed.
        """

        try:
            shutil.rmtree(
                join(abspath(__file__).split("next_purchase_model.py")[0].split("clv")[0][:-1], "clv_prediction",
                     "untitled_project"))
        except Exception as e:
            print(" Parameter Tuning Keras Turner dummy files have already removed!!")

    def update_tuned_parameter_file(self):
        """
        tuned parameters are stored at 'test_parameters.yaml.' in given 'export_path' argument.
        """

        if check_for_existing_parameters(self.directory, 'purchase_amount') is not None:
            _params = read_yaml(self.directory, "test_parameters.yaml")
            _params['next_purchase'] = self.params
        else:
            print("Non of parameter tuning for both Model has been observed.")
            _params = {'next_purchase': self.params}
        write_yaml(self.directory, "test_parameters.yaml", _params, ignoring_aliases=True)

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'next_purchase') is None:
            self.initialize_keras_tuner()
            self.batch_size_and_epoch_tuning()
            self.remove_keras_tuner_folder()
            self.update_tuned_parameter_file()

        else:
            self.params = check_for_existing_parameters(self.directory, 'next_purchase')




