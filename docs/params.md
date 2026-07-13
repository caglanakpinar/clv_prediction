# CLV Prediction Parameters

## job
Train, Prediction, Train & Prediction. When `job = 'train'`,
the steps are Next Purchase Model training and Purchase Amount Model training.
Each model's hyperparameter tuning process is run before the model itself is trained.
Once hyperparameter tuning is complete, the tuned network parameters 
are stored in **test_parameters.yaml** inside `export_path`.
When a model is run repeatedly (or periodically), it is checked whether 
it has already been built during the current `time_period`.
If there are stored models in `export_path`, the latest model 
is imported and the process moves on without rebuilding the model.
When `job='prediction'`, the next purchase per customer is predicted first, then 
the purchase amount is predicted based on the next purchase prediction.
When `job='train_prediction'`, the framework first runs `job='train'`, then `job='prediction'`.

## order_count

It allows us to create a feature set of the purchase amount model.
(Check ***Why do we need order count as a feature at Purchase Amount Model?*** for details).
if it is not assigned (it is not a required argument in order to initialize the clv prediction), 
the platform handles it to decide the optimum order count. Order Count also affects the detection of NewComers.

## customer_indicator

This parameter indicates which column represents a unique customer identifier on given data.

## amount_indicator

This parameter indicates which column represents purchase value (integer, float ..) on the given data.

## time_indicator

This parameter indicates which column represents order checkout date with date 
format (timestamp) (YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY-MM-DD) on given data.

## date

This allows us to query the data with a date filter. This removes data that occurs after the given date.
If the date is not assigned there will be no date filtering. 
date arguments are filtering related to time_indicator column, make sure it is querying with the accurate format.
If clv prediction is running with schedule service, periodically given date is updated and filter with an updated given date.
If the date is not assigned when clv prediction is scheduling, the date will be the current date.

## data_source

The location where the data is stored or the query (check data source for details).

## data_query_path

Type of data source to import data to the platform (optional Ms SQL, PostgreSQL, AWS RedShift, 
Google BigQuery, csv, json, pickle).

## connector

If there are connection parameters such as user, password, host, port, 
this allows us to assign them as a dictionary (e.g. {"user": ***, "pw": ****, "db": ****}).

## export_path

The export path where outputs are stored: created models (`.keras` format),
tuned parameters (`test_parameters.yaml`), schedule service arguments (**schedule_service.yaml**), 
and result data with predicted values per user per predicted order 
(`.csv` format) are all stored at the given path. When prediction is initialized, the Next Purchase Model creates the folder 
`temp_next_purchase_results`, and the
Purchase Amount Model creates the folder `temp_purchase_amount_results`, in order to store results as `.csv` files.

## time_period

A period of time which is willing to predict. 
**Supported time periods month, week, '2*week', quarter, '6*month' (Required)**.
(by default, it is `time_period='week'`)

## time_schedule

A period of time which handles for running clv_prediction train or prediction process periodically. 
**Supported schedule periods day, year, month, week, 2*week.**
