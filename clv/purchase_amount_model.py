from pandas import DataFrame, concat
import os
from itertools import product
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import shutil
import glob
from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
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
    from .utils import get_current_day


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
            model.compile(loss='mae', optimizer=Adam(lr=lr), metrics=['mae'])
        return model


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    return params


def get_order_freq(predicted_orders, customer_indicator, time_indicator):
    if len(predicted_orders) != 0:
        predicted_orders['order_seq_num'] = predicted_orders.sort_values(
            by=[customer_indicator, time_indicator]).groupby([customer_indicator]).cumcount() + 1
    return predicted_orders


def get_number_of_orders(predicted_orders, customer_indicator):
    number_of_orders = None
    if len(predicted_orders) != 0:
        number_of_orders = predicted_orders.groupby(customer_indicator).agg({"order_seq_num": "max"}).reset_index()
    return number_of_orders


class TrainConv1Dimension:
    def __init__(self,
                 date=None,
                 time_indicator=None,
                 order_count=None,
                 data_source=None,
                 data_query_path=None,
                 time_period=None,
                 directory=None,
                 customer_indicator=None,
                 predicted_orders=None,
                 amount_indicator=None):
        self.directory = directory
        self.time_indicator = time_indicator
        self.customer_indicator = customer_indicator
        self.order_count = order_count
        self.time_period = time_period
        self.amount_indicator = amount_indicator
        self.params = hyper_conf('purchase_amount') \
            if check_for_existing_parameters(self.directory, 'purchase_amount') is None else \
            check_for_existing_parameters(self.directory, 'purchase_amount')
        self.hyper_params = get_tuning_params(hyper_conf('purchase_amount_hyper'), self.params)
        self.optimized_parameters = {}
        self._p = None
        self.data, self.features, self.y, self.c_min_max, \
        self.max_date = data_manipulation(
                                           date=date,
                                           amount_indicator=amount_indicator,
                                           time_indicator=time_indicator,
                                           order_count=order_count,
                                           data_source=data_source,
                                           data_query_path=data_query_path,
                                           customer_indicator=customer_indicator,
                                           directory=self.directory)
        self.params['feature_count'] = self.y
        self.hp = HyperParameters()
        self.input, self.model = None, None
        self.prev_model_date = check_model_exists(self.directory, "trained_purchase_amount_model", self.time_period)
        self.model_data = {}
        self.result = None
        self.residuals, self.anomaly = [], []
        self.customers = list(self.data[customer_indicator].unique())
        self.predicted_orders = get_order_freq(predicted_orders, customer_indicator, time_indicator)
        self.num_of_future_orders = get_number_of_orders(self.predicted_orders, self.customer_indicator)
        self.results = DataFrame()

    def train_test_split(self):
        self.model_data['train'], self.model_data['test'] = random_data_split(self.data, float(self.params['split_ratio']))

    def data_preparation(self, is_prediction=False):
        if not is_prediction:
            self.train_test_split()
        else:
            self.model_data['prediction'] = self.data
        self.model_data = reshape_data(self.model_data, self.features, self.y, is_prediction)

    def build_parameter_tuning_model(self, hp):
        self.input = Input(shape=(self.model_data['x_train'].shape[1], 1,))
        ### conv 1D layer
        conv = Conv1D(filters=hp.Choice('filters', self.hyper_params['filters']),
                      kernel_size=hp.Choice('kernel_size', self.hyper_params['kernel_size']),
                      padding='same',
                      activation=hp.Choice('activation', self.hyper_params['activation']),
                      kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                               l2=hp.Choice('l2', self.hyper_params['l2'])),
                      bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                      activity_regularizer=l2(hp.Choice('l2', self.hyper_params['l2']))
                      )(self.input)
        conv = BatchNormalization()(conv)
        # LSTM layer
        conv = MaxPooling1D(hp.Choice('max_pooling_unit', self.hyper_params['max_pooling_unit']))(conv)
        conv = Dropout(hp.Choice('drop_out_ratio', self.hyper_params['drop_out_ratio']))(conv)
        conv = LSTM(hp.Choice('lstm_units', self.hyper_params['lstm_units']),
                    use_bias=False,
                    activation=hp.Choice('activation', self.hyper_params['activation']),
                    kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                             l2=hp.Choice('l2', self.hyper_params['l2'])),
                    bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                    activity_regularizer=l2(hp.Choice('l2', self.hyper_params['l2']))
                    )(conv)
        conv = BatchNormalization()(conv)
        conv = Flatten()(conv)
        # layers after flattened layers
        for i in range(hp.Int('num_layers',
                              self.hyper_params['num_layers']['min'],
                              self.hyper_params['num_layers']['max'])):
            conv = Dense(hp.Choice('units', self.hyper_params['units']),
                         hp.Choice('activation', self.hyper_params['activation']),
                         kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                                  l2=hp.Choice('l2', self.hyper_params['l2'])),
                         bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                         activity_regularizer=l2(hp.Choice('l2', self.hyper_params['l2']))
                         )(conv)
            conv = BatchNormalization()(conv)
        output = Dense(1,
                       activation=hp.Choice('activation', self.hyper_params['activation']),
                       kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                                l2=hp.Choice('l2', self.hyper_params['l2'])),
                       bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                       activity_regularizer=l2(hp.Choice('l2', self.hyper_params['l2']))
                       )(conv)
        model = Model(inputs=self.input, outputs=output)
        model.compile(loss=self.params['loss'],
                      optimizer=Adam(lr=hp.Choice('lr', self.hyper_params['lr'])),
                      metrics=[hp.Choice('loss', self.hyper_params['loss'])])
        return model

    def build_model(self):
        print(self.params)
        self.input = Input(shape=(self.model_data['x_train'].shape[1], 1,))
        # conv 1D layer
        conv = Conv1D(filters=self.params['filters'],
                      kernel_size=self.params['kernel_size'],
                      padding='same',
                      activation=self.params['activation'],
                      kernel_regularizer=l1_l2(l1=self.params['l1'],
                                               l2=self.params['l2']),
                      bias_regularizer=l2(self.params['l2']),
                      activity_regularizer=l2(self.params['l2'])
                      )(self.input)
        conv = BatchNormalization()(conv)
        # LSTM layer
        conv = MaxPooling1D(self.params['max_pooling_unit'])(conv)
        conv = Dropout(self.params['drop_out_ratio'])(conv)
        conv = LSTM(self.params['lstm_units'],
                    use_bias=False,
                    activation=self.params['activation'],
                    kernel_regularizer=l1_l2(l1=self.params['l1'],
                                             l2=self.params['l2']),
                    bias_regularizer=l2(self.params['l2']),
                    activity_regularizer=l2(self.params['l2'])
                    )(conv)
        conv = BatchNormalization()(conv)
        conv = Flatten()(conv)
        # layers after flattened layers
        for i in range(self.params['num_layers']):
            conv = Dense(self.params['units'],
                         self.params['activation'],
                         kernel_regularizer=l1_l2(l1=self.params['l1'],
                                                  l2=self.params['l2']),
                         bias_regularizer=l2(self.params['l2']),
                         activity_regularizer=l2(self.params['l2'])
                         )(conv)
            conv = BatchNormalization()(conv)
        output = Dense(1,
                       activation=self.params['activation'],
                       kernel_regularizer=l1_l2(l1=self.params['l1'],
                                                l2=self.params['l2']),
                       bias_regularizer=l2(self.params['l2']),
                       activity_regularizer=l2(self.params['l2'])
                       )(conv)
        self.model = Model(inputs=self.input, outputs=output)
        self.model.compile(loss=self.params['loss'],
                           optimizer=Adam(lr=self.params['lr']),
                           metrics=[self.params['loss']])

    def learning_process(self, save_model=True, history=True):
        if history:
            history = self.model.fit(self.model_data['x_train'],
                                     self.model_data['y_train'],
                                     batch_size=int(self.params['batch_size']),
                                     epochs=int(self.params['epochs']),
                                     verbose=1,
                                     validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                                     shuffle=True)
        else:
            print("*" * 5, "Fit Purchase Amount Model", "*" * 5)
            self.model.fit(self.model_data['x_train'],
                           self.model_data['y_train'],
                           batch_size=int(self.params['batch_size']),
                           epochs=int(self.params['epochs']),
                           verbose=1,
                           validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                           shuffle=True)

        if save_model:
            model_from_to_json(path=model_path(self.directory,
                                               "trained_purchase_amount_model", get_current_day(), self.time_period),
                               weights_path=weights_path(self.directory,
                                                         "trained_purchase_amount_model",
                                                         get_current_day(), self.time_period),
                               model=self.model,
                               is_writing=True)

        if history:
            return history

    def train_execute(self):
        print("*" * 5, " Purchase Amount train model process ", "*" * 5)
        print(self.model)
        if self.prev_model_date is None:  # previous accepted date for the model file is not found
            self.data_preparation()
            self.parameter_tuning()
            self.build_model()
            self.learning_process()
        else:
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_purchase_amount_model",
                                                            self.prev_model_date,
                                                            self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_purchase_amount_model",
                                                                      self.prev_model_date,
                                                                      self.time_period), lr=self.params['lr'])
            print("Previous model already exits in the given directory  '" + self.directory + "'.")

    def prediction_per_customer(self, customer):
        _prediction_data = pd.DataFrame(prediction_data[customer]['data'])
        _number = prediction_data[customer]['number']
        _user_min = prediction_data[customer]['user_min']
        _user_max = prediction_data[customer]['user_max']
        _prediction = get_prediction(_prediction_data,
                                     _number,
                                     self.model.input.shape[1],
                                     self.model)
        prediction_data[customer] = {}
        prediction_data[customer]['prediction'] = get_predicted_data_readable_form(customer,
                                                                                   _prediction,
                                                                                   self.model.input.shape[1] + 1,
                                                                                   _user_min,
                                                                                   _user_max,
                                                                                   self.customer_indicator)

    def rearrange_prediction_data(self):
        self.model_data['prediction_data'] = pd.merge(self.model_data['prediction_data'],
                                                      self.num_of_future_orders[[self.customer_indicator, 'order_seq_num']],
                                                      on=self.customer_indicator, how='left')
        self.model_data['prediction_data'] = self.model_data['prediction_data'].query("order_seq_num == order_seq_num")
        self.model_data['prediction_data'] = self.model_data['prediction_data'].sort_values(by=self.customer_indicator)
        self.customers = list(self.model_data['prediction_data'][self.model_data['prediction_data'].isin(
            list(self.num_of_future_orders[self.customer_indicator].unique()))][self.customer_indicator].unique())

    def get_user_of_historic_data(self, sample_customers):
        _sample_p_data = self.model_data['prediction_data'][
            self.model_data['prediction_data'][self.customer_indicator].isin(sample_customers)]
        _c_min_max = np.array(self.c_min_max.sort_values(by=self.customer_indicator)[['user_min', 'user_max']]).tolist()
        _numbers = list(_sample_p_data['order_seq_num'])
        _historic_data = _sample_p_data.drop([self.customer_indicator, 'order_seq_num'], axis=1).to_dict('results')
        _values = list(zip(sample_customers, _historic_data, _numbers, _c_min_max))
        return {u[0]: {"data": [u[1]],
                       "number": u[2],
                       'user_min': u[3][0],
                       'user_max': u[3][1]} for u in _values}

    def parallel_prediction(self):
        try:
            os.mkdir(join(self.directory, "temp_purchase_amount_results", ""))
        except Exception as e:
            print(e)
            print("recreating 'temp_purchase_amount_results' folder ...")
            shutil.rmtree(join(self.directory, "temp_purchase_amount_results", ""))
            os.mkdir(join(self.directory, "temp_purchase_amount_results", ""))

        iters = int(len(self.customers) / 128) + 1
        print("number of iters :", iters)
        for i in range(iters):
            print("main iteration :", str(i), " / ", str(iters))
            _sample_customers = get_iter_sample(self.customers, i, iters, 128)
            global prediction_data
            prediction_data = self.get_user_of_historic_data(_sample_customers)
            execute_parallel_run(_sample_customers, self.prediction_per_customer, arguments=None, parallel=8)
            _result = pd.DataFrame()
            for c in _sample_customers:
                try:
                    _result = concat([_result, prediction_data[c]['prediction']])
                except Exception as e:
                    print(c)
            _result.to_csv(join(self.directory, "temp_purchase_amount_results", str(_sample_customers[0] + "_data.csv")),
                           index=False)

    def create_prediction_result_data(self):
        print("merge predicted data ...")
        _import_files = glob.glob(join(self.directory, "temp_purchase_amount_results", "*.csv"))
        li = []
        for f in _import_files:
            try:
                _df = pd.read_csv(f, index_col=None, header=0)
                li.append(_df)
            except Exception as e:
                print(e)
        self.results = concat(li, axis=0, ignore_index=True)
        try:
            shutil.rmtree(join(self.directory, "temp_purchase_amount_results", ""))
        except Exception as e:
            print(e)
        print(self.results.head())

    def prediction_execute(self):
        print("number of users :", len(self.customers))
        self.model_data['prediction_data'] = self.data[self.features + [self.customer_indicator]]
        if self.model is None:
            _model_date = self.prev_model_date if self.prev_model_date is not None else get_current_day()
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_purchase_amount_model", _model_date, self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_purchase_amount_model",
                                                                      _model_date, self.time_period),
                                            lr=self.params['lr'])

        if self.num_of_future_orders is not None:
            self.rearrange_prediction_data()
            self.parallel_prediction()
            self.create_prediction_result_data()
        else:
            self.results = merge_0_result_at_time_period(self.data,
                                                         self.max_date,
                                                         self.time_period,
                                                         self.customer_indicator,
                                                         self.time_indicator,
                                                         self.amount_indicator)
        self.results = merging_predicted_date_to_result_data(self.results,
                                                             self.predicted_orders,
                                                             self.customer_indicator,
                                                             self.time_indicator,
                                                             self.amount_indicator)
        self.results = check_for_previous_predicted_clv_results(self.results,
                                                                self.directory,
                                                                self.time_period,
                                                                self.time_indicator,
                                                                self.customer_indicator,
                                                                )

    def initialize_keras_tuner(self):
        """
        Parameter tuning process is triggered via Keras-Turner Library.
        However, batch_size and epoch parameters of optimization are created individually.
        :return:
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

        try:
            _params = read_yaml(self.directory, "test_parameters.yaml")
        except Exception as e:
            print(e)
            _params = None
        _params['purchase_amount'] = self.params if _params is not None else {'purchase_amount': self.params}
        write_yaml(self.directory, "test_parameters.yaml", _params, ignoring_aliases=True)

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'purchase_amount') is None:
            self.initialize_keras_tuner()
            self.batch_size_and_epoch_tuning()
            self.remove_keras_tuner_folder()
            self.update_tuned_parameter_file()
        else:
            self.params = check_for_existing_parameters(self.directory, 'purchase_amount')

