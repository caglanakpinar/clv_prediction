import argparse

try:
    from next_purchase_model import TrainLSTM
    from purchase_amount_model import TrainConv1Dimension
    from newcomers import TrainLSTMNewComers
except Exception as e:
    from .next_purchase_model import TrainLSTM
    from .purchase_amount_model import TrainConv1Dimension
    from .newcomers import TrainLSTMNewComers


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
    """
    This is how we run the model prediction and train processes.
    Depending on the job it is triggered for train and prediction process individually.
    job, customer_indicator, amount_indicator, data_source, data_query_path, time_period, export_path are required.
    if **order_count** is null, it is detecting for optimum order count.
    the date is crucial for data query. (< date)

    :param job: train or prediction
    :param order_count: number of order for purchase amount model
    :param customer_indicator: customer column at data
    :param amount_indicator: amount column at data
    :param data_source: postgres, .csv, .json, awsredshift, bigquery.
    :param data_query_path: query or file_name (with whole path).
    :param time_period: a period of time which is willing to predict.
    :param time_indicator: time columns at data set
    :param export_path: where data, model,tuned_parameters, schedule arguments are stored
    :param date: given date of previous data (< date)
    :return: class TrainLSTM (Next Purchase Model) & class TrainConv1Dimension (Purchase Amount Model)
    """
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

    # engaged customers of CLV models train and prediction process
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
    if job == 'prediction':
        next_purchase.prediction_execute()
    if job == 'train_prediction':
        next_purchase.train_execute()
        next_purchase.prediction_execute()

    predicted_orders = next_purchase.results
    raw_data = next_purchase.data
    del next_purchase

    purchase_amount = TrainConv1Dimension(
        date=date,
        time_indicator=time_indicator,
        order_count=order_count,
        data_source=data_source,
        data_query_path=data_query_path,
        time_period=time_period,
        directory=export_path,
        customer_indicator=customer_indicator,
        predicted_orders=predicted_orders,
        amount_indicator=amount_indicator)

    if job == 'train':
        purchase_amount.train_execute()
    if job == 'prediction':
        purchase_amount.prediction_execute()
    if job == 'train_prediction':
        purchase_amount.train_execute()
        purchase_amount.prediction_execute()

    engaged_customers_results = purchase_amount.results
    del purchase_amount

    # newcomers of CLV models train and prediction process
    newcomers = TrainLSTMNewComers(
        date=date,
        time_indicator=time_indicator,
        order_count=order_count,
        data_source=data_source,
        data_query_path=data_query_path,
        time_period=time_period,
        directory=export_path,
        customer_indicator=customer_indicator,
        engaged_customers_results=engaged_customers_results,
        amount_indicator=amount_indicator)

    if job == 'train':
        newcomers.train_execute()
    if job == 'prediction':
        newcomers.prediction_execute()
    if job == 'train_prediction':
        newcomers.train_execute()
        newcomers.prediction_execute()

    newcomers_clv = newcomers.results
    del newcomers

    return {"next_purchase": {'data': raw_data},
            "purchase_amount": {'results': engaged_customers_results},
            "newcomers": {'results': newcomers_clv}}


if __name__ == '__main__':
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
                        This can only be applied with date. It can be week, week_part, quarter, year, month.
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
