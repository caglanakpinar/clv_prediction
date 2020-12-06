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


def get_tuning_params(parameter_tuning, params, job):
    arrays = []
    for p in params:
        if p not in list(parameter_tuning.keys()):
            arrays.append([params[p]])
        else:
            arrays.append(
                          np.arange(float(parameter_tuning[p].split("*")[0]),
                                    float(parameter_tuning[p].split("*")[1]),
                                    float(parameter_tuning[p].split("*")[0])).tolist()
            )
    comb_arrays = list(product(*arrays))
    if job != 'parameter_tuning':
        return random.sample(comb_arrays, int(len(comb_arrays)*0.5))
    else:
        return comb_arrays


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    return params


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
                 num_of_future_orders=None,
                 amount_indicator=None):
        self.params = hyper_conf('purchase_amount')
        ## TODO: hyper parametrs of ranges must be updated related to data
        self.hyper_params = hyper_conf('purchase_amount_hyper')
        self.optimized_parameters = {}
        self._p = None
        self.order_count = order_count
        self.time_period = time_period
        self.data, self.features, self.y, self.c_min_max = data_manipulation(
                                                                             date=date,
                                                                             feature=amount_indicator,
                                                                             time_indicator=time_indicator,
                                                                             order_count=order_count,
                                                                             data_source=data_source,
                                                                             data_query_path=data_query_path,
                                                                             customer_indicator=customer_indicator)
        self.time_indicator = time_indicator
        self.customer_indicator = customer_indicator
        self.data_count = 0
        self.hp = HyperParameters()
        self.model_data = {}
        self.input, self.model = None, None
        self.model = None
        self.result = None, None
        self.residuals, self.anomaly = [], []
        self.directory = directory
        self.customers = list(self.data[customer_indicator].unique())
        self.num_of_future_orders = num_of_future_orders
        self.results = DataFrame()

    def train_test_split(self):
        self.model_data['train'], self.model_data['test'] = random_data_split(self.data, self.params['split_ratio'])

    def data_preparation(self, is_prediction):
        if not is_prediction:
            self.train_test_split()
        else:
            self.model_data['prediction'] = self.data
        self.model_data = reshape_data(self.model_data, self.features, self.y, is_prediction)

    def build_parameter_tuning_model(self, hp):
        self.input = Input(shape=(self.model_data['train_x'].shape[1], 1,))
        ### conv 1D layer
        conv = Conv1D(filters=hp.Choice('filters', self.hyper_params['filters']),
                      kernel_size=hp.Choice('kernel_size', self.hyper_params['kernel_size']),
                      padding='same',
                      activation=hp.Choice('activation', self.hyper_params['activation']),
                      kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                               l2=hp.Choice('l2', self.hyper_params['l2'])),
                      bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                      activity_regularizer=l2(hp.Choice('l', self.hyper_params['l2']))
                      )(self.input)
        conv = BatchNormalization()(conv)
        ### LSTM layer
        conv = MaxPooling1D(hp.Choice('max_pooling_unit', self.hyper_params['max_pooling_unit']))(conv)
        conv = Dropout(hp.Choice('drop_out_ratio', self.hyper_params['drop_out_ratio']))(conv)
        conv = LSTM(hp.Choice('lstm_units', self.hyper_params['lstm_units']),
                    use_bias=False,
                    activation=hp.Choice('activation', self.hyper_params['activation']),
                    kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                             l2=hp.Choice('l2', self.hyper_params['l2'])),
                    bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                    activity_regularizer=l2(hp.Choice('l', self.hyper_params['l2']))
                    )(conv)
        conv = BatchNormalization()(conv)
        conv = Flatten()(conv)
        ### layers after flattened layers
        for i in range(hp.Int('num_layers',
                              self.hyper_params['num_layers']['min'],
                              self.hyper_params['num_layers']['max'])):
            conv = Dense(hp.Choice('units', self.hyper_params['units']),
                         hp.Choice('activation', self.hyper_params['activation']),
                         kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                                  l2=hp.Choice('l2', self.hyper_params['l2'])),
                         bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                         activity_regularizer=l2(hp.Choice('l', self.hyper_params['l2']))
                         )(conv)
            conv = BatchNormalization()(conv)
        output = Dense(1,
                       activation=hp.Choice('activation', self.hyper_params['activation']),
                       kernel_regularizer=l1_l2(l1=hp.Choice('l1', self.hyper_params['l1']),
                                                l2=hp.Choice('l2', self.hyper_params['l2'])),
                       bias_regularizer=l2(hp.Choice('l2', self.hyper_params['l2'])),
                       activity_regularizer=l2(hp.Choice('l', self.hyper_params['l2']))
                       )(conv)
        model = Model(inputs=self.input, outputs=output)
        model.compile(loss=self.params['loss'],
                      optimizer=Adam(lr=hp.Choice('lr', self.hyper_params['lr'])),
                      metrics=[self.params['loss']])
        return model

    def build_model(self):
        self.input = Input(shape=(self.model_data['train_x'].shape[1], 1,))
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

    def learning_process(self, save_model=True):
        self.model.fit(self.model_data['x_train'],
                       self.model_data['y_train'],
                       batch_size=self.params['batch_size'],
                       epochs=self.params['epochs'],
                       verbose=0,
                       validation_data=(self.model_data['x_test'], self.model_data['y_test']),
                       shuffle=False)
        if save_model:
            model_from_to_json(path=join(self.directory, "trained_purchase_amount_model"),
                               model=self.model,
                               is_writing=True)

    def train_execute(self):
        self.data_preparation()
        self.parameter_tuning()
        self.learning_process()

    def prediction_execute(self):
        print("number of users :", len(self.customers))
        self.model_data['prediction_data'] = self.data[self.features + ["user_id"]]
        for u in self.num_of_future_orders.to_dict('results'):
            _number, _user = u['order_seq_num'], u['user_id']
            _prediction_data = self.model_data['prediction_data'].query("user_id == @_user").drop('user_id', axis=1)
            _prediction = get_prediction(_prediction_data,
                                         _number,
                                         self.model.input.shape[1],
                                         self.model)
            prediction = get_predicted_data_readable_form(_user,
                                                          _prediction,
                                                          self.model.input.shape[1] + 1,
                                                          self.c_min_max.query("user_id == @_user"))
            print("lifetime value :")
            print(self.results.head())

    def parameter_tuning(self):
        if check_for_existing_parameters(self.directory, 'purchase_amount') is None:
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
            self.params = tuner.get_best_hyperparameters()[0].values
            write_yaml(self.directory, "test_parameters.yaml", self.params, ignoring_aliases=True)
        else:
            self.params = check_for_existing_parameters(self.directory, 'purchase_amount')




