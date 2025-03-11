# Data Preprocess

Each model has unique aggregation in order to prepare data to create model.

## 1. Next Purchase Model Data Preparation

Time difference of each order per customer is calculated as day difference of orders per user.
Normalized each time difference values related to Min-Max Normalization Method. Each customer of Min-Max Normalization individually.
number of lag parameter is tuned by using ARIMA time series model.
Regarding of *lahead* parameter of LSTM model, data set is shaped per customer.
Iteratively each customer of data set is collected individually. the process is parallelized according to CPU count.
When model data has been prepared per customer, 
it is split according to split ratio into the train and test data set (train_x, train_y, test_x, test_y).

| customers     | lag 3  |lag 2  |lag 1  |y      |
| -------------:| ------:|------:|------:|------:|
| user_1        | 0,4    |0,8    |1,7    |1,2    |
| user_1        | 0,8    |1,7    |1,2    |1,4    |
| user_1        | 1,7    |1,2    |1,4    |1,6    |
| user_1        | 1,2    |1,4    |1,6    |2,8    |
| user_1        | 1,4    |1,6    |2,8    |2,9    |
| user_1        | 1,6    |2,8    |2,9    |3,0    |
| user_2        | 5,4    |5,8    |5,7    |5,2    |
| user_2        | 5,8    |5,7    |5,2    |5,4    |
| user_2        | 5,7    |5,2    |5,4    |5,6    |
| user_2        | 5,2    |5,4    |5,6    |5,8    |
| user_2        | 5,4    |5,6    |5,8    |5,9    |
| user_2        | 5,6    |5,8    |5,9    |5,0    |


## 2. Purchase Amount Model Data Preparation
`order_count` parameter refers us for the model of the feature count. 
previous orders of purchase amounts of each customer is collected from fow data.
Regarding of `lahead` parameter of LSTM model, data set is shaped per customer.
Iteratively each customer of data set is collected individually. the process is parallelized according to CPU count.
When model data has been prepared per customer, 
it is split according to split ratio into the train and test data set (train_x, train_y, test_x, test_y).

### Why do we need order count as a feature at Purchase Amount Model?

Order count is also the feature number of the purchase amount model.

#### Caution

Order count must be inserted into the `test_parameters.yaml` in order not to allow for changing later on prediction. 
Once the model is built with calculated `order_count` it must be predicted with the same order count.

### Why we need an order count of a decision?

It is a crucial parameter for the purchase amount model. 
The purchase amount is a 1 Dimensional Conv NN. It works with kernel sizes and they are related to feature size.
At the purchase amount model features are sequential orders. 
For instance if we assign order count as 5, time_period.user_1, user_2, user_3, user_4 have 100, 101, 300, 2 orders.
The data set will be;

- `user_1: 95th, 96th, 97th, 98th, 99th, 100th  orders`

- `user_2: 96th, 97th, 98th, 99th, q00th, 101st  orders`

- `user_3: 295th, 296th, 297th, 298th, 299th, 300th  orders`

- `user_4: only have 2 orders first 9 orders will be 0 and this will affect the model process`

It is now crucial to have a minimum 0 assigned order as user_4
However, it is also a crucial point to get as much previous order count for make kernel size larger.
The order count must be optimized even sending to the platform as an argument. 
If this argument is not using, the platform handles for deciding order_count***.

| customers     | Last 5  |Last 4  |Last 3  |Last 2  |Last Order (y)  |
| -------------:| -------:|-------:|-------:|-------:|---------------:|
| user_1        | 10,4    | 13,4   | 18,4   | 11,4   | 15,4           |
| user_2        | 50,8    | 52,8   | 54,8   | 56,8   | 58,8           |
| user_3        | 30,7    | 25,7   | 15,7   | 10,7   | 8,7            |
| user_4        | 20,2    | 23.5   | 26,2   | 27,2   | 29,2           |
| user_5        | 1,4     | 1,4    | 1,4    | 1,4    | 1,4            |
| user_6        | 12,6    | 30,6   | 12,6   | 30,6   | 12,6           |


## 3. NewComers Model Data Preparation

`order_count` argument will be used in order to assign users weather NewComer pr not. 
THis parameter can be while initializing the platform as below;

```
    from clv.executor import CLV
    order_count = 3  # users who have < 3 orders will be new comers
    clv = CLV(customer_indicator=customer_indicator,
              ...
              order_count: int | None, 
              ....
 
    )
```

If `order_count=None`, `feature_count` in `test_parameters.yaml` will be used.


### Why do we need order count as a feature at NewComers CLV Model?

It is a crucial parameter for NewComers Model;

-   Users who have an order count less than `order_count` are not included in Combined of Next Purchase - Purchase Amount Models.

-   NewComers are individually predicted according to a dependent value is `order_count`.

Main concept of Newcomers is for predicting orders count daily.
Feature value is going to be total number of order count for all NewComers.
Normalized each order count values related to Min-Max Normalization Method per day.
Regarding of `lahead` parameter of LSTM model, data set is shaped just like below;

|       days | lag 3  |lag 2  |lag 1  |y  (total order count of Newcomers)    |
|-----------:| ------:|------:|------:|--------------------------------------:|
| 2021-05-01 | 25     |5      |10     |20                                     |
| 2021-05-02 | 5      |10     |20     |30                                     |
| 2021-05-03 | 10     |20     |30     |40                                     |
| 2021-05-04 | 20     |30     |40     |60                                     |
| 2021-05-05 | 30     |40     |60     |70                                     |
| 2021-05-06 | 40     |60     |70     |90                                     |
| 2021-05-07 | 60     |70     |90     |100                                    |


When model data has been prepared per day just like above.
it is split according to split ratio into the train and test data set (train_x, train_y, test_x, test_y).
LSTM model will be implemented for the data just like above.

Prediction process is calculated sequentially per day. Each day model has been regenerated (store .json format) 
with updated coefficient matrix (stored as .h5 format).
LSTM allows us to predict next step values regarding your lags. While we are predicting further prediction,
models of coefficients must be updated and the previous prediction values must be merged just like actual values.
But, just like the recent prediction and model, the Tuned parameters are will also be used for further future days of predictions.
