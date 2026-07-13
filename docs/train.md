# CLV Train Models and Prediction

## Train  - `job = 'train'`

The **Next Purchase Model**, **Purchase Amount Model**, and **NewComer Model** train processes are all run via TensorFlow/Keras.
Each is an LSTM NN.
Trained models are stored at `export_path` in **.keras** format.
Each **.keras** file name includes the `time_period`, the model's name, and the trained date (current date).
e.g. `trained_purchase_amount_model_20210101_month.keras`


Before initializing the training process, previously-stored models at `export_path` are checked.
    The most recently trained model is picked; its name and `time_period` must match.
    e.g. recent model: `trained_purchase_amount_model_20210101_month.keras`, model name: `purchase_amount`, time_period: `month`,
      current date 2020-01-30. This model was trained 29 days ago, which falls within the accepted range (0-30 days for one month).


## Train-Prediction Process `job = 'train_prediction'`

Each model is trained, then all three are predicted sequentially. 
By the end, all 3 models have been trained, 
hyperparameter tuning has been applied to all 3, and all 3 models' predictions have been calculated.


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

Once the prediction process has been initialized (`job: 'prediction'` or `'train_prediction'`), 
results can be collected via `get_result_data`.
This returns the raw data plus each customer's predicted next purchase orders.

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
