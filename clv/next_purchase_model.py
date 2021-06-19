import warnings
from pandas import DataFrame, concat
import os
import shutil
from itertools import product
import subprocess
from psutil import virtual_memory


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
    from utils import abspath_for_sample_data
except Exception as e:
    from .functions import *
    from .configs import hyper_conf, accept_threshold_for_loss_diff, parameter_tuning_trials
    from .data_access import *
    from .utils import abspath_for_sample_data


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
        self.params = hyper_conf('next_purchase') \
            if check_for_existing_parameters(self.directory,'next_purchase') is None else \
            check_for_existing_parameters(self.directory, 'next_purchase')
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
        self.prev_model_date = check_model_exists(self.directory, "trained_next_purchase_model", self.time_period)
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
                                               "trained_next_purchase_model",
                                               get_current_day(),
                                               self.time_period),
                               weights_path=weights_path(self.directory,
                                                         "trained_next_purchase_model",
                                                         get_current_day(),
                                                         self.time_period),
                               model=self.model,
                               is_writing=True)

        if history:
            return history

    def train_execute(self):
        print("*"*5, "Next purchase train model process ", "*"*5)
        if self.prev_model_date is None:
            self.data_preparation()
            self.parameter_tuning()
            self.build_model()
            self.learning_process()
        else:
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_next_purchase_model",
                                                            self.prev_model_date,
                                                            self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_next_purchase_model",
                                                                      self.prev_model_date,
                                                                      self.time_period), lr=self.params['lr'])
            print("Previous model already exits in the given directory  '" + self.directory + "'.")

    def create_prediction_data(self, customers, _directory):
        """
        import data with .csv format to the 'temp_next_purchase_inputs'. with given batch.
        Example of directory and data:
            data:
                customer_indicator | amount_indicator | time_indicator
                ------------------------------------------------------
                   cust_1          |   20.3           |  2021-06-05
                   .......         |   .....          |   .....
                   cust_2          |   30.3           |  2021-06-08
                   .......         |   .....          |   .....
                   cust_3          |   30.3           |  2021-06-10


            directory : <self.directory>/temp_next_purchase_inputs/prediction_inputs_cust_1.csv
        """
        self.temp_data = self.data[self.data[self.customer_indicator].isin(customers)]
        self.temp_data.to_csv(_directory, index=False)

    def parallel_prediction(self):
        """
        *** Parallel Prediction Process ***
            1. Create input folder: 'temp_next_purchase_inputs'
            2. Create result folder for the prediction: 'temp_next_purchase_results'
            3. split users (split number of user count related customer_indicator parameter) and run each batch sequentially.
            3. Parallel process is running on 'next_purchase_prediction.py'. Check related .py file for details.
        """
        try: # create input folder
            os.mkdir(join(self.directory, "temp_next_purchase_results"))
        except Exception as e:
            print(e)
            print("recreating 'temp_next_purchase_results' folder ...")
            shutil.rmtree(join(self.directory, "temp_next_purchase_results", ""))
            os.mkdir(join(self.directory, "temp_next_purchase_results"))


        try: # create output folder for the results
            os.mkdir(join(self.directory, "temp_next_purchase_inputs"))
        except Exception as e:
            print(e)
            print("recreating 'temp_next_purchase_inputs' folder ...")
            shutil.rmtree(join(self.directory, "temp_next_purchase_inputs", ""))
            os.mkdir(join(self.directory, "temp_next_purchase_inputs"))
        # check cpu count and available memory in order to optimize batch sizes
        cpus = cpu_count() * int((virtual_memory().total / 1000000000) * 4)
        communicates = [i for i in range(cpus) if i % cpu_count() == 0] + [cpus - 1]
        _sample_size = int(len(self.customers) / cpus) + 1
        _sub_processes = []
        for i in range(cpus):
            print("main iteration :", str(i), " / ", str(cpus))
            _sample_customers = get_iter_sample(self.customers, i, cpus, _sample_size)
            if len(_sample_customers) != 0:
                _directory = join(self.directory,
                                  "temp_next_purchase_inputs",
                                  "prediction_inputs_" + _sample_customers[0] + ".csv")
                self.create_prediction_data(_sample_customers, _directory)
                cmd = """python {0}/next_purchase_prediction.py -P {1} -MD {2} -FD {3} -TP {4} -D {5} -IND {6}
                """.format(abspath_for_sample_data(), _directory, str(self.max_date)[0:10],
                           str(self.future_date)[0:10], self.time_period, self.directory,
                           "*".join([self.customer_indicator, self.amount_indicator, self.time_indicator]))
                _p = subprocess.Popen(cmd, shell=True)
                _sub_processes.append(_p)
                if i != 0 and i in communicates:
                    [p.communicate() for p in _sub_processes]
                    print("done!")
                    _sub_processes = []

    def create_prediction_result_data(self):
        """
        after the parallel process prediction, results are stored in the folder 'temp_next_purchase_results'.
        Now, it is time to merge them
        """
        print("merge predicted data ...")
        _import_files = glob.glob(join(self.directory, "temp_next_purchase_results", "*.csv"))
        li = []
        for f in _import_files: # merge all result files
            try:
                _df = pd.read_csv(f, index_col=None, header=0)
                li.append(_df)
            except Exception as e:
                print(e)
        self.results = concat(li, axis=0, ignore_index=True)
        self.results[self.time_indicator] = self.results[self.time_indicator].apply(lambda x: convert_str_to_day(x))
        # remove temp folders
        shutil.rmtree(join(self.directory, "temp_next_purchase_results", ""))
        shutil.rmtree(join(self.directory, "temp_next_purchase_inputs", ""))
        # filter the dates related to selected time period
        self.results = self.results[(self.results[self.time_indicator] > self.max_date) &
                                    (self.results[self.time_indicator] < self.future_date)]
        print(self.results.head())
        print("number of predicted customers :", len(self.results[self.customer_indicator].unique()))

    def prediction_execute(self):
        """
        This process is running sequentially for each customer and predicts next order of timestamp
        """
        print("*"*5, "PREDICTION", 5*"*")
        print("number of users :", len(self.customers))
        if self.model is None:
            _model_date = self.prev_model_date if self.prev_model_date is not None else get_current_day()
            # import trained model from temp. directory folder
            self.model = model_from_to_json(path=model_path(self.directory,
                                                            "trained_next_purchase_model",
                                                            _model_date,
                                                            self.time_period),
                                            weights_path=weights_path(self.directory,
                                                                      "trained_next_purchase_model",
                                                                      _model_date,
                                                                      self.time_period), lr=self.params['lr'])
        self.results = self.data[[self.time_indicator, 'time_diff', 'time_diff_norm', self.customer_indicator]]
        self.parallel_prediction() # this process will be triggered via next_purchase_prediction.py
        self.create_prediction_result_data() # merge all result data

    def initialize_keras_tuner(self):
        """
        Parameter tuning process is triggered via Keras-Turner Library.
        However, batch_size and epoch parameters of optimization are created individually.
        """

        self.hyper_params, self.optimum_batch_size = batch_size_hp_ranges(client_sample_sizes=self.client_sample_sizes,
                                                                          num_of_customers=len(self.customers),
                                                                          hyper_params=self.hyper_params)
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
            _params['next_purchase'] = self.params
        else:
            print("None of parameter tuning for both Model has been observed.")
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





