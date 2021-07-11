import warnings
import pandas as pd
import os
import shutil
from itertools import product
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
sess = tf.compat.v1.Session()

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


class TrainLSTMNewComers:
    def __init__(self,
                 date=None,
                 time_indicator=None,
                 order_count=None,
                 data_source=None,
                 data_query_path=None,
                 time_period=None,
                 directory=None,
                 engaged_customers_results=pd.DataFrame(),
                 customer_indicator=None,
                 amount_indicator=None):
        self.sess = tf.compat.v1.Session()
        self.directory = directory
        self.customer_indicator = customer_indicator
        self.time_indicator = time_indicator
        self.amount_indicator = amount_indicator
        self.params = hyper_conf('newcomers') \
            if check_for_existing_parameters(self.directory,'newcomers') is None else \
            check_for_existing_parameters(self.directory, 'newcomers')
        self.hyper_params = get_tuning_params(hyper_conf('newcomers_hyper'), self.params)
        self.optimized_parameters = {}
        self._p = None
        self.order_count = order_count
        self.time_period = time_period
        self.engaged_customers_results = engaged_customers_results
        self.data, self.features, \
        self.average_amount, self.min_max = data_manipulation_nc(date=date,
                                                                 order_count=order_count,
                                                                 amount_indicator=amount_indicator,
                                                                 time_indicator=time_indicator,
                                                                 data_source=data_source,
                                                                 data_query_path=data_query_path,
                                                                 customer_indicator=customer_indicator,
                                                                 directory=directory)

        self.hp = HyperParameters()
        self.model_data = {}
        self.input, self.model = None, None
        self.prev_model_date = check_model_exists(self.directory, "trained_newcomers_model", self.time_period)
        self.residuals, self.anomaly = [], []
        self.results = pd.DataFrame()
        self.get_actual_value = lambda _min, _max, _value: ((_max - _min) * _value) + _min if _value >= 0 else _min
        self.max_date = max(self.data[self.time_indicator])
        self.future_date = self.max_date + datetime.timedelta(days=convert_time_preiod_to_days(self.time_period))
        self.model_data = {"x_train": None, "y_train": None, "x_test": None, "y_test": None}
        self.client_sample_sizes = []
        self.optimum_batch_size = 8

    def data_preparation(self):
        self.model_data = arrange__data_for_model(df=self.data, f=[self.features], parameters=self.params)

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

    def init_tf(self):
        self.sess.close()
        import tensorflow as tf
        self.sess = tf.compat.v1.Session()

        from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.initializers import Ones
        from tensorflow.keras.models import Model
        from tensorflow.keras.models import model_from_json
        from kerastuner.tuners import RandomSearch
        from kerastuner.engine.hyperparameters import HyperParameters

    def build_model(self, prediction=False):
        if prediction:
            self.init_tf()
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

    def learning_process(self, save_model=True, history=False, show_epochs=True):
        verbose = 1 if show_epochs else 0
        if history:
            history = self.model.fit(self.model_data['x_train'],
                                     self.model_data['y_train'],
                                     batch_size=self.params['batch_size'],
                                     epochs=int(self.params['epochs']),
                                     verbose=1,  # verbose = 1 if there if history = True
                                     validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                                     shuffle=True)
        else:
            if save_model:
                print("*"*5, "Fit Newcomers CLV Model", "*"*5)
            self.model.fit(self.model_data['x_train'],
                           self.model_data['y_train'],
                           batch_size=self.params['batch_size'],
                           epochs=int(self.params['epochs']),
                           verbose=verbose,
                           validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                           shuffle=True)
        if save_model:
            model_from_to_json(path=model_path(self.directory,
                                               "trained_newcomers_model",
                                               get_current_day(),
                                               self.time_period),
                               weights_path=weights_path(self.directory,
                                                         "trained_newcomers_model",
                                                         get_current_day(),
                                                         self.time_period),
                               model=self.model,
                               is_writing=True)

        if history:
            return history

    def train_execute(self):
        print("*"*5, "Newcomer CLV Prediction train model process ", "*"*5)
        if self.prev_model_date is None:
            self.data_preparation()
            self.parameter_tuning()
            self.build_model()
            self.learning_process()
        else:
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_newcomers_model",
                                                            self.prev_model_date,
                                                            self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_newcomers_model",
                                                                      self.prev_model_date,
                                                                      self.time_period))
            print("Previous model already exits in arrange__data_for_model the given directory  '" + self.directory + "'.")

    def prediction_execute(self):
        print("*"*5, "PREDICTION", 5*"*")
        if self.model is None:
            _model_date = self.prev_model_date if self.prev_model_date is not None else get_current_day()
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_newcomers_model",
                                                            _model_date,
                                                            self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_newcomers_model",
                                                                      _model_date,
                                                                      self.time_period))
        # daily calculations, day by day
        while self.max_date < self.future_date:
            print("date :", self.max_date)
            self.model_data = arrange__data_for_model(self.data, [self.features], self.params)
            self.build_model(prediction=True)
            self.learning_process(save_model=False, history=False, show_epochs=False)
            x = arrange__data_for_model(self.data, [self.features], self.params, is_prediction=True)
            _pred = pd.DataFrame([{self.time_indicator: self.max_date, "order_count": self.model.predict(x)[0][-1]}])
            self.data, self.results = pd.concat([self.data, _pred]), pd.concat([self.results, _pred])
            self.max_date += datetime.timedelta(days=1)
            del self.model_data, x, self.model
        for i in ['min_' + self.features, 'max_' + self.features]:
            self.results[i] = self.min_max[i]
        self.results[self.features] = self.results.apply(
            lambda row: self.get_actual_value(_min=row['min_' + self.features],
                                              _max=row['max_' + self.features],
                                              _value=row[self.features]), axis=1)
        self.results[self.amount_indicator] = self.results[self.features] * self.average_amount
        self.results[self.customer_indicator] = "newcomers"
        self.results['data_type'] = "prediction"
        self.results = self.results[['data_type', self.customer_indicator, self.time_indicator, self.amount_indicator]]
        print("result file : ", get_result_data_path(self.directory, self.time_period, self.max_date))
        pd.concat([self.results, self.engaged_customers_results]).to_csv(
            get_result_data_path(self.directory, self.time_period, self.max_date), index=False)

    def initialize_keras_tuner(self):
        """
        Parameter tuning process is triggered via Keras-Turner Library.
        However, batch_size and epoch parameters of optimization are created individually.
        """
        kwargs = {'directory': self.directory}
        tuner = RandomSearch(
            self.build_parameter_tuning_model,
            max_trials=parameter_tuning_trials,
            hyperparameters=self.hp,
            allow_new_entries=True,
            objective='loss', **kwargs)
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
            shutil.rmtree(join(self.directory, "untitled_project"))
        except Exception as e:
            print(" Parameter Tuning Keras Turner dummy files have already removed!!")

    def update_tuned_parameter_file(self):
        """
        tuned parameters are stored at 'test_parameters.yaml.' in given 'export_path' argument.
        """

        # existing model for 'purchase_amount' that means p. tuning was applied for both models.
        if check_for_existing_parameters(self.directory, 'purchase_amount') is not None:
            _params = read_yaml(self.directory, "test_parameters.yaml")
            _params['newcomers'] = self.params
        else:
            print("None of parameter tuning for both Model has been observed.")
            _params = {'newcomers': self.params}
        write_yaml(self.directory, "test_parameters.yaml", _params, ignoring_aliases=True)

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'newcomers') is None:
            self.initialize_keras_tuner()
            self.batch_size_and_epoch_tuning()
            self.remove_keras_tuner_folder()
            self.update_tuned_parameter_file()

        else:
            self.params = check_for_existing_parameters(self.directory, 'newcomers')




