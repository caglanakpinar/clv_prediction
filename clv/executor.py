from os.path import join
import threading
import pandas as pd

try:
    from main import main
    from data_access import GetData
    from utils import get_folder_path, write_yaml, read_yaml
    from configs import conf
    from dashboard import create_dahboard
    from functions import check_for_previous_predicted_clv_results
except Exception as e:
    from .main import main
    from .data_access import GetData
    from .utils import get_folder_path, write_yaml, read_yaml
    from .configs import conf
    from .dashboard import create_dashboard
    from .functions import check_for_previous_predicted_clv_results, check_model_exists
    from .functions import check_for_previous_predicted_clv_results


class CLV:
    """
    job                 :  Train, Prediction. train process is related to creating a model
    order_count         :  it allows us to create feature set of purchase amount model. if it is not assigned (it is not a
                           required argument in order to initialize the clv prediction), platform handles for decide
                           optimum order_count.
     customer_indicator :  This parameter indicates which column represents a unique customer identifier on given data.
     amount_indicator   :  This parameter indicates which column represents purchase value (integer, float ..) on the given data.
     time_indicator     :  This parameter indicates which column represents order checkout date with date format (timestamp)
                           (YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY-MM-DD) on given data.
     date               :  This allows us to query the data with a date filter. This removes data that occurs after the given date.
     data_source        :  The location where the data is stored or the query (check data source for details).
     data_query_path    :  Type of data source to import data to the platform (optional Ms SQL, PostgreSQL,
                           AWS RedShift, Google BigQuery, csv, json, pickle).
     connector          :  if there is a connection parameters as user, pasword, host port, this allows us to assign it
                           as dictionary format (e.g {"user":  , "pw":  *}).
     export_path        :  Export path where the outputs are stored. created models (.json format),
                           tuned parameters (test_parameters.yaml).
                           result data with predicted values per user per predicted order
                           (.csv format) are willing to store at given path.
     time_period        :  A period of time which is willing to predict.
                           Supported time periods 6*month, quarter, month, week, 2*week, '2*month'(Required).
    """
    def __init__(self,
                 customer_indicator=None,
                 amount_indicator=None,
                 time_indicator=None,
                 job=None,
                 date=None,
                 order_count=None,
                 data_source=None,
                 data_query_path=None,
                 time_schedule=None,
                 time_period=None,
                 export_path=None,
                 connector=None):

        self.job = job.lower() if job is not None else job
        self.time_period = time_period.lower() if data_source is not None else data_source
        self.amount_indicator = amount_indicator
        self.order_count = order_count
        self.customer_indicator = customer_indicator
        self.data_source = data_source.lower() if data_source is not None else data_source
        self.data_query_path = data_query_path
        self.data_query_path_raw = data_query_path
        self.time_indicator = time_indicator
        self.time_schedule = time_schedule.lower() if time_schedule is not None else time_schedule
        self.export_path = export_path
        self.connector = connector
        self.result_columns = [customer_indicator, 'order_seq_num', time_indicator, amount_indicator, 'data_type']
        self.raw_data = pd.DataFrame()
        self.results = pd.DataFrame()
        self.sorting_columns = [self.customer_indicator, self.time_indicator]
        self.arguments = {"job": job,
                          "time_period": time_period,
                          "order_count": order_count,
                          "date": date,
                          "customer_indicator": customer_indicator,
                          "amount_indicator": amount_indicator,
                          "data_source": data_source,
                          "data_query_path": data_query_path,
                          "time_indicator": time_indicator,
                          "export_path": export_path}
        self.arg_terminal = {"job": "J",
                             "time_period": "TP",
                             "order_count": "OC",
                             "date": "D",
                             "customer_indicator": "CI",
                             "amount_indicator": "AI",
                             "data_source": "DS",
                             "data_query_path": "DQP",
                             "time_indicator": "TI",
                             "export_path": "EP"}
        self.schedule_arg = "TS"
        self.args_str = ""
        self.mandetory_arguments = ["data_source", "data_query_path", "customer_indicator",
                                    "amount_indicator", "time_indicator", "export_path"]
        self.clv_predicted = None
        self.path = get_folder_path()
        self.model_count = 3

    def query_string_change(self):
        """
        When query with SQL syntax, it is more accurate to sent query string as argument with coverting spaces with "+"
        character.
        """
        if self.data_source in ['mysql', 'postgresql', 'awsredshift', 'googlebigquery']:
            self.data_query_path = self.data_query_path.replace("\r", " ").replace("\n", " ").replace(" ", "+")

    def get_connector(self):
        """
        Connection checks for given data_source, data_path, connection parameters.
        This will be stored at configs.yaml at docs folder.
        If it can read the data well, returns True
        If data is directly imported from any file with .csv, .json and .yaml format, returns True
        :return: True or False
        """
        config = conf('config')
        try:
            if self.data_source not in ["csv", "json"]:
                for i in config['db_connection']:
                    if i != 'data_source':
                        config['db_connection'][i] = self.connector[i]
                    else:
                        config['db_connection']['data_source'] = self.data_source
            write_yaml(join(self.path, "docs"), "configs.yaml", config, ignoring_aliases=False)
            source = GetData(data_source=self.data_source,
                             data_query_path=self.data_query_path,
                             time_indicator=self.time_indicator,
                             feature=self.amount_indicator)
            source.get_connection()
            return True
        except Exception as e:
            print(e)
            if self.data_source not in ["csv", "json"]:
                for i in config['db_connection']:
                    if i is not 'data_source':
                        config['db_connection'][i] = None
                    else:
                        config['db_connection']['data_source'] = self.data_source
            write_yaml(join(self.path, "docs"), "configs.yaml", config, ignoring_aliases=False)
            return False

    def check_for_mandetory_arguments(self):
        """
        checking for mandetory arguments before initializing for model train or prediction process.
        :return: True, False
        """
        for arg in self.arg_terminal:
            if arg in self.mandetory_arguments:
                return False if self.arguments[arg] is None else True

    def check_for_time_period(self):
        """
        When clv_prediction is triggered, valid time period must be checked.
        Here are the valid time periods that platform is supporting;
            - "day", "year", "month", "week", "2*week", '6*month',  "quarter"
        """
        if self.time_period is None:
            return True
        else:
            if self.time_period in ["month", "week", "2*week", '6*month',  "quarter", '2*month']:
                return True
            else: return False

    def checking_for_prediction_process(self):
        """
        When the platform is triggered for prediction, first, trained model for related time_period and time
        must be check. If the current trained models are not valid for this prediction process, it won`t be started.
            Checking models;
                - trained_next_purchase_model
                - trained_purchase_amount_model
                - trained_newcomers_model

            if all models are stored into the export_path, prediction process are able to be calculated.
        """
        accept = False
        if self.job == 'prediction':
            models = [check_model_exists(self.export_path, m, self.time_period)
                      for m in ["trained_next_purchase_model", "trained_purchase_amount_model",
                                "trained_newcomers_model"]]
            if len([1 for model in models if len(model) != 0]) == self.model_count:
                accept = True
        else:
            accept = True
        return accept

    def clv_prediction(self):
        """
        This the process where all has begun.
        Train process is triggered with running clv_prediction.
        Prediction process is triggered with running clv_prediction.
        There are conditions that must be acceoted in order to initialized the process;
            - connection .... Done (get_connector)
            - time period check ... Done (check_for_time_period)
            - required arguments .. Done (check_for_mandetory_arguments)
            - Check for trained model exits if it is the prediction process .. Done(checking_for_prediction_process)
            Sending arguments to main function at main.py which handles the model,
            prediction Deep Learning and other staff
        """
        self.query_string_change()
        if self.get_connector():
            if self.check_for_time_period():
                if self.check_for_mandetory_arguments():
                    if self.checking_for_prediction_process():
                        self.clv_predicted = main(**self.arguments)
                    else:
                        print("Execution : *** Prediction")
                        print("None of trained model has been detected!. Please run for Train process!")
                        print("if these are trained model with .json format please check any of these models of ",
                              "the time period ('month', 'week', ..) matchs with given time preiod.",
                              " Given time period is :", self.time_period)
                        print("argument; job: 'train'")
                else:
                    print("check for the required paramters to initialize CLV Prediction:")
                    print(" - ".join(self.mandetory_arguments))
            else:
                print("optional time periods are :")
                print("month", "week", "2*week", '2*month', "quarter", '6*month')
        else:
            print("pls check for data source connection / path / query.")

    def check_for_raw_data(self):
        """
        if model has been initialized before triggering for raw data,
        raw data can be collected from clv_predicted.next_purchase.data
        However, order_seq_num must be created.
        """
        if self.clv_predicted is not None:
            self.raw_data = self.clv_predicted['next_purchase']['data']
            self.raw_data['data_type'] = 'actual'
            self.raw_data['order_seq_num'] = self.raw_data.sort_values(by=self.sorting_columns).groupby(
                [self.customer_indicator]).cumcount() + 1

    def check_for_result_data(self):
        """
        if model has been initialized before triggering for result data,
        result data can be collected from clv_predicted.purchase_amount.results
        """
        if self.clv_predicted is not None:
            self.results = pd.concat([self.clv_predicted['purchase_amount']['results'],
                                      self.clv_predicted['newcomers']['results']])

    def check_for_result_data_from_previous_progresses(self):
        """
        If model has not been triggered but there is predicted result_Data.csv file in 'export_path'
        it can be collected from the given path by merging with previous available result_data.csv files.
        """
        if self.clv_predicted is None or len(self.results) == 0:
            self.results = check_for_previous_predicted_clv_results(self.results,
                                                                    self.export_path,
                                                                    self.time_period,
                                                                    self.time_indicator,
                                                                    self.customer_indicator,
                                                                    )

    def get_result_data(self):
        """
        When model prediction has been done, this allows us to get the predicted data with merge to the raw data.
        If model has not been initialized (clv_prediction has not been run yet),
        it directly collects the data from last predicted result_data.csv
        with given arguments (time_period, time_indicator, export_path, customer_indicator)
        :return: data frame
        """
        self.check_for_raw_data()
        self.check_for_result_data()
        self.check_for_result_data_from_previous_progresses()
        return self.results if self.clv_predicted is None else pd.concat([self.raw_data, self.results])

    def show_dashboard(self):
        """
        if you are running dashboard, make sure you have assigned export_path.
        """
        process = threading.Thread(target=create_dashboard,
                                   kwargs={"customer_indicator": self.customer_indicator,
                                           "amount_indicator": self.amount_indicator,
                                           "directory": self.export_path,
                                           "time_indicator": self.time_indicator,
                                           "time_period": self.time_period,
                                           "data_query_path": self.data_query_path_raw,
                                           "data_source": self.data_source})
        process.daemon = True
        process.start()

    def create_api_for_runnig_realtime_customer_value_predicion(self):
        """
        Real time api for prediction values for individual customer or customer list
        :return:
        """

