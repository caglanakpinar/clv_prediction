import schedule
import datetime
import time
import argparse


try:
    from main import main
    from utils import convert_date, read_yaml, write_yaml
except Exception as e:
    from .main import main
    from .utils import convert_date, read_yaml, write_yaml


def convert_to_day_hour(x):
    return datetime.datetime.strptime(str(x)[0:13], "%Y-%m-%d %H")


def convert_to_day(x):
    return datetime.datetime.strptime(str(x)[0:10], "%Y-%m-%d")


def get_schedule(time_period):
    if time_period not in ['minute', 'hour', 'week']:
        return {'mondays': schedule.every().monday,
                'tuesdays': schedule.every().tuesday,
                'wednesdays': schedule.every().wednesday,
                'thursdays': schedule.every().thursday,
                'fridays': schedule.every().friday,
                'saturdays': schedule.every().saturday,
                'sundays': schedule.every().sunday,
                'day': schedule.every().day
                }[time_period]
    if time_period == 'week':
        return schedule.every().week
    if time_period == 'hour':
        print("initial time :", str(datetime.datetime.now())[11:16])
        return schedule.every(1).hours.at(str(datetime.datetime.now())[11:16])


def update_date(args):
    iteration, time_schedule = args['iteration'], args['time_schedule']
    if time_schedule == 'hour':
        if len(str(args['arguments']['date'])) == 10:
            args['arguments']['date'] = args['arguments']['date'] + ' 00'
        date = convert_to_day_hour(args['arguments']['date'])
    else:
        date = convert_to_day(args['arguments']['date'])
    if iteration != 0:
        if time_schedule == 'hour':
            date = date + datetime.timedelta(hours=iteration)
        if time_schedule == 'Daily':
            date = date + datetime.timedelta(days=iteration)
        if time_schedule == 'week':
            date = date + datetime.timedelta(days=int(7 * iteration))
    args['arguments']['date'] = str(date)[0:13]
    write_yaml(args['arguments']['export_path'], 'schedule_' + args['arguments']['job'] + '.yaml',
               {'arguments': args['arguments'],
                'time_schedule': args['time_schedule'],
                'iteration': args['iteration'] + 1
                })
    print("CLV Prediction Date :", args['arguments']['date'])


def run_ab_test(directory, job):
    arguments = read_yaml(directory, 'schedule_' + job + '.yaml')
    update_date(arguments)
    main(**arguments['arguments'])


def check_for_first_running(args):
    """
    before initialize the periodically run, it must be trained.
    If there are the trained model stored in the given directory, process will be skipped.
    args: arguments,  in order to initialize the train or prediction process.
    :return: None
    """
    if args['job'] == 'train':
        main(**args)


def decision_of_time_sleep(time_schedule):
    if time_schedule == 'hour':
        return 60
    if time_schedule == 'day':
        return 600
    if time_schedule == 'week':
        return 3600
    if time_schedule == 'month':
        return 18000
    if time_schedule not in ['hour', 'day', 'week', 'month']:
        return 1000


def create_job(arguments, time_schedule):
    time_schedule = time_schedule
    args = arguments
    check_for_first_running(args)
    _sch = get_schedule(time_schedule)
    _sch.do(run_ab_test, arguments['export_path'], arguments['job'])
    print(_sch)
    time_sec_wait = decision_of_time_sleep(time_schedule)
    while True:
        schedule.run_pending()
        time.sleep(time_sec_wait)
        print("waiting ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-TS", "--time_schedule", type=str,
                        help="""
                        
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
    clv_arguments = {'test_groups': arguments.test_groups,
                         'groups': arguments.groups, 'date': arguments.date,
                         'feature': arguments.feature, 'data_source': arguments.data_source,
                         'data_query_path': arguments.data_query_path, 'time_period': arguments.time_period,
                         "export_path": arguments.export_path}

    create_job(clv_arguments, arguments.time_schedule)