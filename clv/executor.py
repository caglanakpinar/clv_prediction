from os.path import join
import threading
import pandas as pd

try:
    from main import main
    from data_access import GetData
    from utils import get_folder_path, write_yaml, read_yaml
    from configs import conf
    from scheduler_service import create_job
    from dashboard import create_dahboard
    from functions import check_for_previous_predicted_clv_results
except Exception as e:
    from .main import main
    from .data_access import GetData
    from .utils import get_folder_path, write_yaml, read_yaml
    from .configs import conf
    from .scheduler_service import create_job
    from .dashboard import create_dashboard
    from .functions import check_for_previous_predicted_clv_results, check_model_exists

class CLV:
    """
    order_count:        column of the data which represents  A - B Test of groups.
                        It  is a column name from the data.
                        AB test runs as control  - active group name related to columns of unique values.
                        This column has to 2 unique values which shows us the test groups
    customer_indicator:

    amount_indicator:

    job:                train, prediction


    date:

    data_source:        AWS RedShift, BigQuery, PostgreSQL, csv, json files can be connected to system
                        E.g.
                        {"data_source": ..., "db": ..., "password": ..., "port": ..., "server": ..., "user": ...}

    data_query_path:    if there is file for data importing;
                            must be the path (e.g /../.../ab_test_raw_data.csv)
                        if there is ac- connection such as PostgreSQL / BigQuery
                            query must at the format "SELECT++++*+++FROM++ab_test_table_+++"

    time_indicator      This can only be applied with date. It can be hour, day, week, week_part, quarter, year, month.
                        Individually time indicator checks the date part is significantly
                        a individual group for data set or not.
                        If it is uses time_indicator as a  group

   time_schedule:      When AB Test need to be scheduled, only need to be assign here 'Hourly', 'Monthly',
                        'Weekly', 'Mondays', ... , Sundays.

    export_path        exporting the results data to. only path is enough for importing data with .csv format.

    time_period:
    """
    def __init__(self,
                 customer_indicator,
                 amount_indicator,
                 job=None,
                 date=None,
                 order_count=None,
                 data_source=None,
                 data_query_path=None,
                 time_schedule=None,
                 time_period=None,
                 time_indicator=None,
                 export_path=None,
                 connector=None):

        self.job = job
        self.time_period = time_period
        self.amount_indicator = amount_indicator
        self.order_count = order_count
        self.customer_indicator = customer_indicator
        self.data_source = data_source
        self.data_query_path = data_query_path
        self.data_query_path_raw = data_query_path
        self.time_indicator = time_indicator
        self.time_schedule = time_schedule
        self.export_path = export_path
        self.connector = connector
        self.result_columns = [customer_indicator, 'order_seq_num', time_indicator, amount_indicator, 'data_type']
        self.raw_data = pd.DataFrame()
        self.results = pd.DataFrame()
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

    def check_for_mandetory_arguments(self):
        for arg in self.arg_terminal:
            if arg in self.mandetory_arguments:
                return False if self.arguments[arg] is None else True

    def get_connector(self):
        """
       query_string_change İf data
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

    def get_connector(self):
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

    def check_for_time_period(self):
        if self.time_period is None:
            return True
        else:
            if self.time_period in ["day", "year", "month", "week", "2*week", '2*month', "quarter"]:
                return True
            else: return False

    def query_string_change(self):
        if self.data_source in ['mysql', 'postgresql', 'awsredshift', 'googlebigquery']:
            self.data_query_path = self.data_query_path.replace("\r", " ").replace("\n", " ").replace(" ", "+")

    def checking_for_prediction_process(self):
        if self.job == 'prediction':
            model_files = check_model_exists(self.export_path, "trained_next_purchase_model", self.time_period)
            if model_files is not None:
                if len(check_model_exists(self.export_path, "trained_next_purchase_model", self.time_period)) != 0:
                    return True
                else: return False
            else: return False
        else: return True

    def clv_prediction(self):
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
                print("day", "year", "month", "week", "2*week", '2*month', "hour", "quarter")
        else:
            print("pls check for data source connection / path / query.")

    def get_result_data(self):
        if self.clv_predicted is not None:
            self.raw_data = self.clv_predicted['next_purchase'].data
            self.raw_data['data_type'] = 'actual'
            self.raw_data['order_seq_num'] = self.raw_data.sort_values(
                by=[self.customer_indicator, self.time_indicator]).groupby([self.customer_indicator]).cumcount() + 1
            self.raw_data = self.raw_data[self.result_columns]
            self.results = self.clv_predicted['purchase_amount'].results
            if len(self.results) == 0:
                self.results = check_for_previous_predicted_clv_results(self.results,
                                                                        self.export_path,
                                                                        self.time_period,
                                                                        self.time_indicator,
                                                                        self.customer_indicator,
                                                                        )

            return pd.concat([self.raw_data, self.results])
        else: return None

    def check_for_time_schedule(self):
        if self.time_schedule is None:
            return True
        else:
            if self.time_schedule in ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays",
                                      "Saturdays", "Sundays", "Daily", "hour", "week"]:
                return True
            else: return False

    def create_schedule_file(self):
        write_yaml(self.directory, 'schedule.yaml',
                   {'ab_test_arguments': self.arguments, 'time_period': self.time_schedule})

    def schedule_clv_prediction(self):
        if self.get_connector():
            if self.check_for_time_schedule():
                if self.check_for_mandetory_arguments():
                    self.create_schedule_file()
                    process = threading.Thread(target=create_job, kwargs={'ab_test_arguments': self.arguments,
                                                                          'time_period': self.time_schedule})
                    process.daemon = True
                    process.start()
                else:
                    print("check for the required parameters to initialize A/B Test:")
                    print(" - ".join(self.mandetory_arguments))

            else:
                print("optional schedule time periods are :")
                print("Mondays - .. - Sundays", "Daily", "week", "hour")
        else:
            print("pls check for data source connection / path / query.")

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

    def create_api_for_runig_realtime_customer_value_predicion(self):
        """
        Real time api for prediction values for individual customer or customer list
        :return:
        """

