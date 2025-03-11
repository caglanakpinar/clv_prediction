# CLV Train Models and Prediction

## Train  - `job = 'train'`

**Next Purchase Model**, **Purchase Amount Model** and **NewComer Model** of the train process are progressed via tensorflow - Keras.
It is a LSTM NN.
Trained model stored at `export_path` with **.json** format.
**.json** trained file has a file name with `time_period`, name of the model, trained date (current date).
e.g; trained_purchase_amount_model_20210101_month.json


Before initialize the training process previously-stored model are checked which have been stored at `export_path`
    The most recent trained must be picked. Model name and `time_period`  also must be matched.
    e.g; recent model: trained_purchase_amount_model_20210101_month.json, model name: purchase_amount, time_period: month,
      current date 2020-01-30. This model trained 29 days before which is accepted range (accepted range 0 - 30 (one month)).


## Train-Prediction Proces `job = 'train_prediction'`

Each model process is trained, then they are predicted sequentially. 
At the end 3 models have been generalized, 
3 models of parameters tuning have been applied and 3 models of predictions are calculated.


## Running CLV Prediction
        customer_indicator = "user_id"
        amount_indicator = "transaction_value"
        time_indicator = "days"
        time_period = 'month'
        job = "train" # prediction or train_prediction
        date = '2021-01-01'
        order_count = 15
        data_source = "postgresql"
        data_query_path="""
                    select user_id,
                           transaction_value,
                           days
                    from purchases
        """
        export_path =  './data'
        connector = {"db": "c****s",
                     "password": "******",
                     "port": "5**3",
                     "server": "127.0.0.1",
                     "user": "*******"}

        from clv.executor import CLV
        clv = CLV(customer_indicator=customer_indicator,
                  amount_indicator=amount_indicator,
                  job=job,
                  date=date,
                  order_count=order_count,
                  data_source=data_source,
                  data_query_path=data_query_path,
                  time_period=time_period,
                  time_indicator=time_indicator,
                  export_path=export_path,
                  connector=connector)
        clv.clv_prediction()

## Collecting Prediction Result Data

Once, prediction process has been initialized (`job: 'prediction'` or `'train_prediction'`), 
It can be collected via `get_result_data`.
This data will be represented with raw data per customer of next purchase orders.

        from clv.executor import CLV
        clv = CLV(customer_indicator=customer_indicator,
                  amount_indicator=amount_indicator,
                  date=date,
                  order_count=order_count,
                  data_source=data_source,
                  data_query_path=data_query_path,
                  time_period=time_period,
                  time_indicator=time_indicator,
                  export_path=export_path,
                  connector=connector)
        results = clv.get_result_data()


| customers |data_type   |time_indicator  |amount_indicator  |
| ---------:|-----------:|---------------:|-----------------:|
| user_1    | actual     | 2021-01-07     | 10,4             |
| user_1    | actual     | 2021-01-14     | 15,4             |
| user_1    | actual     | 2021-01-28     | 20,4             |
| user_1    | prediction | 2021-02-05     | 25,4             |
| user_1    | prediction | 2021-02-06     | 30,8             |
| user_2    | prediction | 2021-02-05     | 8,7              |
| user_3    | prediction | 2021-02-05     | 29,2             |
| user_4    | prediction | 2021-02-05     | 1,4              |
| user_4    | prediction | 2021-02-06     | 18,6             |
| newcomers | prediction | 2021-02-05     | 12,6             |
| newcomers | prediction | 2021-02-06     | 12,6             |
