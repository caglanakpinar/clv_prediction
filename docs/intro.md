# welcome to clv-prediction 101

Welcome to clv-prediction world. 
This platform allows you to run customers lifetime prediction for future relationship with a customer.
You can connect to any data source and run CLV without touching any code all you need to pass arguments like user ID etc.

## what is this all about?

This is about to build a pipeline which is starting from fetching data,
build model for users next purchases/sessions in platform,
make prediction per customer based on built model from previous step.
It also treats individually for new users who have no purchase or sessions background on historical data.
Finally, when all CLV train/prediction process have been completed, a dashboard will be available to visualize.

## Why do need `clv-prediciton`?

recent years, companies needed to see their feature and relation with their customers. 
This will help them to act proactively for the upcoming trends of users engagements.
For instance, Users might return churn in feature, so clv-prediction will give overview churn users.
Here are the benefits to use clv-prediction:
 - Allows you to predict your business of customers values individually.
 - Predicts customers of next purchase dates.
 - Predicts customers of next purchase amounts.
 - Predicts newcomers of next purchase amounts.
 - Dashboard for visualization predicted values.

## Step by Step Instruction to Use

there are 2 sections;
 - execute model train
 - run dashboard

## How to run clv and dashboard

```
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
    clv.show_dashboard()
```

## How it works?

### Main Concept

This framework we generate 2 main predictive model per customer. 
First, Next Purchase (Frequency) Model will be trained. 
This model will help us to predict the day of nex purchases per customer
Second, Customer Value Model will be trained. 
THis model will help us to predict what will be the amount of next purchases per customer.
There will be customers can not be predicted by those models above because of lack historical informations. 
Those customers are NewComers.
This platform allows us to predict NewComers' total lifetime values as well.


### Prediction of Next Purchase (Frequency) per Customer Model

Each customer of historical purchases of date differences is calculated.
    There will be accepted patterns related to customers ' behaviors.
    Some Users might have a pattern of every Monday.
    Some will have Mondays -Wednesdays- Fridays.
    There must be an individual predictive model for each customer, and this must be the Time Series model per each customer of historical frequency.
    However, it is not an efficient way and there will be a computational cost here. In that case, Deep Learning can handle this problem with LSTM NN (check next_purchase_model.py).
    There must be a model that each customer of frequency values are able to be predicted.

### Prediction Of Customer Value (Value) per Customer Model

Customer future values of prediction are also crucial to reach the final CLV calculation.
    Once frequency values are calculated per customer, by historical users' of purchase values can be predicted via using Deep Learning.
    At this process, there is a built-in network (check purchase_amount.py) which is created by using 1 Dimensional Convolutional LSTM NN.

### Prediction Of NewComers CLV Model

Newcomers are not likely predictable as Engaged users. 
They probably not have stabilized transactions pattern or they will not have a fitted train model unless they have enough transactions.
    At this point, rather than predicting the value of each transaction, predicting the amount of transaction will be more convenient.
    By using the historical total purchases per time period (daily), the next time period of total purchase count is able to be predicted.
    Assuming that Purchase Amount of Newcomers are Normal Distributed (Hypothesis Test).
    In that case, purchase Amount prediction per newcomer is going to be the Mean of Purchase Amounts.

### Combining Of Next Purchase Model & Purchase Amount Prediction Model & NewComers Prediction Model

Without predicting the frequency of users, we can not be sure when the customer will have a purchase.
    So, by using the next purchase model, customers of future purchase dates have to be predicted.
    Before predicting a date, the algorithm makes sure the predicted future order of dates is in **selected time period**.

***last purchased date from raw data < predicted purchase date < last purchased date from raw data + time period***

This time period must be assigned when the process is initialized. 
The time period will have a range between the last transaction date of the dataset and the last transaction date + time period.
It can be detected the users' purchases of dates and the next process will be predicting each purchase of values by using the Purchase Amount model.

After combining Of Next Purchase Model & Purchase Amount Prediction Model is done, NewComers of Predictions are merging the results.


## CLV Prediction Process Pipeline

![Untitled](https://user-images.githubusercontent.com/26736844/118328794-da34e000-b50e-11eb-8a7f-3a10373f8461.png)
