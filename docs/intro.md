# welcome to clv-prediction 101

Welcome to the clv-prediction world. 
This platform allows you to predict customer lifetime value for your future relationship with a customer.
You can connect to any data source and run CLV without touching any code &mdash; all you need to do is pass arguments like user ID, etc.

## what is this all about?

This is about building a pipeline that starts by fetching data,
builds a model for users' next purchases/sessions on the platform,
and makes a prediction per customer based on the model built in the previous step.
It also treats new users individually when they have no purchase or session history in the historical data.
Finally, once the whole CLV train/prediction process has completed, a dashboard will be available to visualize the results.

## Why do we need `clv-prediction`?

In recent years, companies have needed more visibility into their customers' future and their relationship with them. 
This helps them act proactively on upcoming trends in user engagement.
For instance, users might churn in the future, so clv-prediction will give an overview of churned users.
Here are the benefits to use clv-prediction:
 - Allows you to predict your business of customers values individually.
 - Predicts customers of next purchase dates.
 - Predicts customers of next purchase amounts.
 - Predicts newcomers of next purchase amounts.
 - Dashboard for visualization predicted values.

## Step by Step Instruction to Use

There are 2 sections:
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

This framework generates 2 main predictive models per customer. 
First, the Next Purchase (Frequency) Model is trained. 
This model helps predict the day of the next purchase per customer.
Second, the Customer Value Model is trained. 
This model helps predict what the amount of the next purchase will be per customer.
There will be customers who cannot be predicted by the models above because of a lack of historical information. 
Those customers are NewComers.
This platform allows us to predict NewComers' total lifetime values as well.


### Prediction of Next Purchase (Frequency) per Customer Model

The date difference between each customer's historical purchases is calculated.
    There will be accepted patterns related to customers' behaviors.
    Some users might have a pattern of every Monday.
    Some will have Mondays, Wednesdays, and Fridays.
    There should be an individual predictive model for each customer &mdash; a time series model based on each customer's historical frequency.
    However, this is not efficient and comes with a high computational cost. Deep Learning can handle this problem instead, using an LSTM NN (check next_purchase_model.py).
    This gives us a single model that can predict frequency values for every customer.

### Prediction Of Customer Value (Value) per Customer Model

Predicting customers' future values is also crucial for the final CLV calculation.
    Once frequency values are calculated per customer, historical purchase values can be used to predict future purchase values via Deep Learning.
    This process uses a built-in network (check purchase_amount_model.py) built using a 1-Dimensional Convolutional LSTM NN.

### Prediction Of NewComers CLV Model

Newcomers are not as easily predictable as engaged customers. 
They likely don't have a stabilized transaction pattern, and there won't be a well-fitted trained model unless they have enough transactions.
    At this point, rather than predicting the value of each transaction, predicting the total transaction amount is more convenient.
    By using the historical total purchases per time period (daily), the total purchase count for the next time period can be predicted.
    Assuming that the purchase amount of newcomers is normally distributed (hypothesis test),
    the purchase amount prediction per newcomer is the mean of purchase amounts.

### Combining Of Next Purchase Model & Purchase Amount Prediction Model & NewComers Prediction Model

Without predicting the frequency of users, we can't be sure when a customer will make a purchase.
    So the next purchase model is used to predict each customer's future purchase dates.
    Before predicting a date, the algorithm makes sure the predicted future order date falls within the **selected time period**.

***last purchased date from raw data < predicted purchase date < last purchased date from raw data + time period***

This time period must be assigned when the process is initialized. 
The time period will have a range between the last transaction date of the dataset and the last transaction date + time period.
Once the users' purchase dates are detected, the next step predicts each purchase's value using the Purchase Amount model.

After combining the Next Purchase Model and Purchase Amount Prediction Model, the NewComers predictions are merged into the results.


## CLV Prediction Process Pipeline

![Untitled](https://user-images.githubusercontent.com/26736844/118328794-da34e000-b50e-11eb-8a7f-3a10373f8461.png)
