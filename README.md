# Customer Lifetime Value Prediction

---------------------------

 -- [![PyPI version](https://badge.fury.io/py/abtest.svg)](https://badge.fury.io/py/abtest)
-- [![GitHub license](https://img.shields.io/github/license/caglanakpinar/abtp)](https://github.com/caglanakpinar/abtp/blob/master/LICENSE)

----------------------------

##### Key Features

##### How it works?

- **Main Concept**
    
    Main concept of customer value prediction is related to calculate *Return Rate*, *Churn Rate* (per customer or genereal Ratio), 
    then formulize the ratio with Value of each customers per purchase. 
    Depending on the churn rate, the total business value some how can be predicted and it will not be affted much by using this technique. 
    This technique, rather than predicting each customer of value, it gives general idea of about the business what will be the total revenue with the customer.
    How about is it changing the metology, rather than using general churn rate which is applied for each customer, predicting customers of selected future time period of possible order dates by using their hitorical transactions? If we can predict the exact date of each customer by historic time difference of each customer, we are able to predict future value of each order per customer.
    
    
- **Prediction Of next Purchase (Frequency) per Customer Model**

    Each customer of historical purchases of date differencases are calculated. thre will be excapted patterns related to customers of behaviors. Some Users might have a pattern of every Mondays. Some will have Mondays -Wednesdays- Fridays. there must be indşvşdual predictive model each custmer, and this must be the Time Series model per each customer of historical frequency. However, it is efficient way and there are much computatşonal cost at there. In that case, Deep Learning can handle this problem with LSTM NN (check next_purchase_model.py). There must be model that each customer of frequency values must be predicted.
    
- **Prediction Of Customer Value (Value) per Customer Model**

    Customer future values of prediction  is also the crucial to reach final CLV calculation. Once frequency values are calculated per customer, by historical users' of purchase values van be predicted by using Deep Learning. At this prcess there is a built-in network (check purchase_amount.py) which is creted by using 1 Dimensional Convolutional LSTM NN. 
    
- **Combining Of Next Purchase Model & Purchase Amount Prediction Model**

    Without predicting frequency of users, we can not be sure when the customer will have purchase. 
    So, by using next purchase model, each customer of future purchase date has to be predicted. 
    Before predicting date, the algorithm make sure the predicted future order of dates are in **selected time period**. 
    This time period must be assigned when the process is initialized. Time period will have range between last transaction date of dataset and last transaction date + time period.
    It can be detected the users' purchases of dates and next process will be predicting each purchase of values by using Puschase Amount model.
    

- **CLV Prediction Process Pipeline**

![draw_clv_prediction_process](https://user-images.githubusercontent.com/26736844/102719986-5c273100-4302-11eb-97ef-c86153336473.png)



    
    
    
    




    