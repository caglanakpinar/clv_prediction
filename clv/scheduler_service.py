import schedule
import datetime
import time
import argparse
from os.path import abspath, join


try:
    from main import main
    from utils import date_part, convert_date, read_yaml, write_yaml
except Exception as e:
    from .main import main
    from .utils import date_part, convert_date, read_yaml, write_yaml

global args
global time_schedule
global iteration


def get_schedule(time_period):
    if time_period not in ['minute', 'hour', 'week']:
        return {'Mondays': schedule.every().monday,
                'Tuesdays': schedule.every().tuesday,
                'Wednesdays': schedule.every().wednesday,
                'Thursdays': schedule.every().thursday,
                'Fridays': schedule.every().friday,
                'Saturdays': schedule.every().saturday,
                'Sundays': schedule.every().sunday,
                'Daily': schedule.every().day
                }[time_period]
    if time_period == 'week':
        return schedule.every().week
    if time_period == 'hour':
        print("initial time :", str(datetime.datetime.now())[11:16])
        return schedule.every(1).hours.at(str(datetime.datetime.now())[11:16])


def update_date():
    date = convert_date(args['date'])
    if iteration != 0:
        if time_schedule == 'hour':
            date = date + datetime.timedelta(hours=iteration)
        if time_schedule == 'Daily':
            date = date + datetime.timedelta(days=iteration)
        if time_schedule == 'week':
            date = date + datetime.timedelta(days=int(7 * iteration))
    args['date'] = str(date)[0:19]
    print("AB Test Date :", args['date'])


def run_ab_test():
    update_date(args)
    main(**args)
    iteration += 1


def create_job(ab_test_arguments, time_period):
    iteration = 0
    time_schedule = time_period
    args = ab_test_arguments
    _sch = get_schedule(time_period)
    _sch.do(run_ab_test)
    while True:
        _sch.run_pending()
        time.sleep(1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-TS", "--time_schedule", type=str,
                        help="""
                        column of the data which represents  A - B Test of groups. 
                        It  is a column name from the data.
                        AB test runs as control  - active group name related to columns of unique values.
                        This column has to 2 unique values which shows us the test groups
                        """)
    parser.add_argument("-J", "--job", type=str,
                        help="""

                        """,
                        )
    parser.add_argument("-OC", "--order_count", type=str,
                        help="""

                        """,
                        )
    parser.add_argument("-CI", "--customer_indicator", type=str,
                        help="""

                        """,
                        )
    parser.add_argument("-AI", "--amount_indicator", type=str,
                        help="""

                        """,
                        )
    parser.add_argument("-DS", "--data_source", type=str,
                        help="""
                        AWS RedShift, BigQuery, PostgreSQL, csv, json files can be connected to system
                        """,
                        required=True)
    parser.add_argument("-DQP", "--data_query_path", type=str,
                        help="""
                        if there is file for data importing;
                            must be the path (e.g /../.../ab_test_raw_data.csv)
                        if there is ac- connection such as PostgreSQL / BigQuery
                            query must at the format "SELECT++++*+++FROM++ab_test_table_+++"
                        """,
                        required=True)
    parser.add_argument("-TI", "--time_indicator", type=str,
                        help="""
                        This can only be applied with date. It can be hour, day, week, week_part, quarter, year, month.
                        Individually time indicator checks the date part is significantly 
                        a individual group for data set or not.
                        If it is uses time_indicator as a  group
                        """,
                        )
    parser.add_argument("-EP", "--export_path", type=str,
                        help="""
                        This shows us to the time period if AB Test is running sequentially
                        """,
                        )
    arguments = parser.parse_args()
    clv_arguments = {'order_count': arguments.order_count,
                     'customer_indicator': arguments.customer_indicator,
                     'amount_indicator': arguments.amount_indicator,
                     'data_source': arguments.data_source,
                     'data_query_path': arguments.data_query_path,
                     'time_indicator': arguments.time_indicator,
                     'export_path': arguments.export_path}

    ab_test_arguments = {'test_groups': arguments.test_groups,
                         'groups': arguments.groups, 'date': arguments.date,
                         'feature': arguments.feature, 'data_source': arguments.data_source,
                         'data_query_path': arguments.data_query_path, 'time_period': arguments.time_period,
                         "export_path": arguments.export_path}

    create_job(ab_test_arguments, arguments.time_schedule)