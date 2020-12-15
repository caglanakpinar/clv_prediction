import sys
from os.path import abspath
import argparse

try:
    from next_purchase_model import TrainLSTM
    from purchase_amount_model import TrainConv1Dimension
except Exception as e:
    from .next_purchase_model import TrainLSTM
    from .purchase_amount_model import TrainConv1Dimension


def main(job='train',
         order_count=None,
         customer_indicator=None,
         amount_indicator=None,
         data_source=None,
         data_query_path=None,
         time_period=None,
         time_indicator=None,
         export_path=None,
         date=None):
    print("received :", {"job": job,
                         'order_count': order_count,
                         'customer_indicator': customer_indicator,
                         'amount_indicator': amount_indicator,
                         'data_source': data_source,
                         'data_query_path': data_query_path,
                         'time_period': time_period,
                         'time_indicator': time_indicator,
                         'export_path': export_path}
          )
    next_purchase = TrainLSTM(
                              date=date,
                              time_indicator=time_indicator,
                              order_count=order_count,
                              data_source=data_source,
                              data_query_path=data_query_path,
                              time_period=time_period,
                              directory=export_path,
                              customer_indicator=customer_indicator,
                              amount_indicator=amount_indicator)
    if job == 'train':
        next_purchase.train_execute()
    purchase_amount = TrainConv1Dimension(
                                          date=date,
                                          time_indicator=time_indicator,
                                          order_count=order_count,
                                          data_source=data_source,
                                          data_query_path=data_query_path,
                                          time_period=time_period,
                                          directory=export_path,
                                          customer_indicator=customer_indicator,
                                          predicted_orders=next_purchase.results,
                                          amount_indicator=amount_indicator)
    if job == 'prediction':
        purchase_amount.prediction_execute()
    if job == 'train':
        purchase_amount.train_execute()
    return next_purchase, purchase_amount


if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-J", "--job", type=str,
                        help="""
                                train, prediction
                        """,
                        )
    parser.add_argument("-OC", "--order_count", type=str,
                        help="""
                                number of previous order count. It has been using for prediction next order frequency
                        """,
                        )
    parser.add_argument("-CI", "--customer_indicator", type=str,
                        help="""identifier of the customer (id)
    
                        """,
                        )
    parser.add_argument("-AI", "--amount_indicator", type=str,
                        help="""
                                data column n-amount (preferred numeric)
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
    parser.add_argument("-TP", "--time_period", type=str,
                        help="""
                        This shows us to the time period
                        """,
                        )
    parser.add_argument("-EP", "--export_path", type=str,
                        help="""
                        Exporting path of the results set. Csv file of exporting.
                        """,
                        )

    arguments = parser.parse_args()
    args = {'job': arguments.job, 'order_count': arguments.order_count,
            'customer_indicator': arguments.customer_indicator,
            'amount_indicator': arguments.amount_indicator,
            'data_source': arguments.data_source,
            'data_query_path': arguments.data_query_path,
            'time_period': arguments.time_period,
            'time_indicator': arguments.time_indicator,
            'export_path': arguments.export_path}
    print(args)
    main(**args)
