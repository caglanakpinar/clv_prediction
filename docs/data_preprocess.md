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
The `order_count` parameter refers to the model's feature count. 
Previous purchase amounts for each customer's orders are collected from the raw data.
Depending on the `lahead` parameter of the LSTM model, the data set is shaped per customer.
Each customer's data set is collected individually, and the process is parallelized according to CPU count.
Once model data has been prepared per customer, 
it is split according to the split ratio into train and test data sets (train_x, train_y, test_x, test_y).

### Why do we need order count as a feature at Purchase Amount Model?

Order count is also the feature count of the purchase amount model.

#### Caution

The order count must be stored in `test_parameters.yaml` so it can't change on later predictions. 
Once the model is built with a calculated `order_count`, predictions must use the same order count.

### Why do we need an order count decision?

It's a crucial parameter for the purchase amount model. 
The purchase amount model is a 1-Dimensional Conv NN. It works with kernel sizes, which are related to the feature size.
In the purchase amount model, features are sequential orders. 
For instance, if we assign order count as 5, and user_1, user_2, user_3, user_4 have 100, 101, 300, 2 orders respectively for a given time_period, the data set will be:

- `user_1: 95th, 96th, 97th, 98th, 99th, 100th  orders`

- `user_2: 96th, 97th, 98th, 99th, 100th, 101st  orders`

- `user_3: 295th, 296th, 297th, 298th, 299th, 300th  orders`

- `user_4: only have 2 orders first 9 orders will be 0 and this will affect the model process`

It's important to assign a minimum of 0 for missing orders, as with user_4.
However, it's also important to use as much previous order history as possible, to make the kernel size larger.
The order count should be optimized even when it's sent to the platform as an argument. 
If this argument isn't provided, the platform handles deciding the order count.

| customers     | Last 5  |Last 4  |Last 3  |Last 2  |Last Order (y)  |
| -------------:| -------:|-------:|-------:|-------:|---------------:|
| user_1        | 10,4    | 13,4   | 18,4   | 11,4   | 15,4           |
| user_2        | 50,8    | 52,8   | 54,8   | 56,8   | 58,8           |
| user_3        | 30,7    | 25,7   | 15,7   | 10,7   | 8,7            |
| user_4        | 20,2    | 23.5   | 26,2   | 27,2   | 29,2           |
| user_5        | 1,4     | 1,4    | 1,4    | 1,4    | 1,4            |
| user_6        | 12,6    | 30,6   | 12,6   | 30,6   | 12,6           |


## 3. NewComers Model Data Preparation

The `order_count` argument is used to determine whether a user is a NewComer or not. 
This parameter can be set while initializing the platform, as shown below:

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

It's a crucial parameter for the NewComers Model:

-   Users who have an order count less than `order_count` are not included in the combined Next Purchase / Purchase Amount Models.

-   NewComers are individually predicted based on the `order_count` value.

The main concept behind NewComers is predicting the daily order count.
The feature value is the total order count across all NewComers.
Each day's order count is normalized using the Min-Max Normalization Method.
Depending on the `lahead` parameter of the LSTM model, the data set is shaped just like below:

|       days | lag 3  |lag 2  |lag 1  |y  (total order count of Newcomers)    |
|-----------:| ------:|------:|------:|--------------------------------------:|
| 2021-05-01 | 25     |5      |10     |20                                     |
| 2021-05-02 | 5      |10     |20     |30                                     |
| 2021-05-03 | 10     |20     |30     |40                                     |
| 2021-05-04 | 20     |30     |40     |60                                     |
| 2021-05-05 | 30     |40     |60     |70                                     |
| 2021-05-06 | 40     |60     |70     |90                                     |
| 2021-05-07 | 60     |70     |90     |100                                    |


Once model data has been prepared per day as shown above,
it is split according to the split ratio into train and test data sets (train_x, train_y, test_x, test_y).
The LSTM model is then trained on this data.

The prediction process is calculated sequentially per day. Each day, the model is regenerated (stored in `.keras` format) 
with an updated coefficient matrix.
LSTM lets us predict next-step values based on your lags. While predicting further into the future,
the model's coefficients must be updated, and the previous prediction values must be merged in just like actual values.
The tuned parameters from the most recent prediction and model are also reused for further future-day predictions.
