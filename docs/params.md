# CLV Prediction Parameters

### job
Train, Prediction, Train&Prediction. train process is related to creating a model;
the next steps are going to be the Next Purchase Model and Purchase Amount Model.
Each model of the hyper parameter tuning process will be initialized before models have been initialized.
Once, the hyperparameter has been progressed tuned network parameters 
are stored in **test_parameters.yaml** where it is in ***export_path***.
When a model has been run repeatedly (or periodically), the model has been checked whether 
it has been already built during the ***time_period.
If there are stored models in ***export_path***, the latest model 
is imported and move on to the next process without a run for building the model.
When the ***prediction** process is triggered, first, the next purchase per customer is predicted then, 
the purchase amount is predicted related to the next purchase prediction.

### order_count

It allows us to create a feature set of the purchase amount model.
(Check ***Why do we need order count as a feature at Purchase Amount Model?*** for details).
if it is not assigned (it is not a required argument in order to initialize the clv prediction), 
the platform handles it to decide the optimum order count. Order Count also affects the detection of NewComers.

### customer_indicator

This parameter indicates which column represents a unique customer identifier on given data.

### amount_indicator

This parameter indicates which column represents purchase value (integer, float ..) on the given data.

### time_indicator

This parameter indicates which column represents order checkout date with date 
format (timestamp) (YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY-MM-DD) on given data.

### date

This allows us to query the data with a date filter. This removes data that occurs after the given date.
If the date is not assigned there will be no date filtering. 
date arguments are filtering related to time_indicator column, make sure it is querying with the accurate format.
If clv prediction is running with schedule service, periodically given date is updated and filter with an updated given date.
If the date is not assigned when clv prediction is scheduling, the date will be the current date.

### data_source

The location where the data is stored or the query (check data source for details).

### data_query_path

Type of data source to import data to the platform (optional Ms SQL, PostgreSQL, AWS RedShift, 
Google BigQuery, csv, json, pickle).

### connector

if there is a connection parameters as user, password, host port, 
this allows us to assign it as dictionary format (e.g {"user": ***, "pw": ****}).

### export_path

Export path where the outputs are stored. created models (.json format),
tuned parameters (test_parameters.yaml), schedule service arguments (schedule_service.yaml), 
result data with predicted values per user per predicted order 
(.csv format) are willing to store at given path. When prediction is initialized, Nex Purchase Model will create folder 
`temp_next_purchase_results` and
Purchase Amount Model will create folder 'temp_purchase_amount_results' in order to import results as .csv format

### time_period

A period of time which is willing to predict. Supported time periods month, week, '2*week', quarter, '6*month' (Required).

### time_schedule

A period of time which handles for running clv_prediction train or prediction process periodically. 
Supported schedule periods day, year, month, week, 2*week.
