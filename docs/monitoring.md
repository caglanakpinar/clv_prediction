# Dashboard for CLV Prediction

Here are examples of dashboard

![Screen Recording 2021-01-05 at 09 46 54 PM](https://user-images.githubusercontent.com/26736844/103687181-e9c07d00-4fa0-11eb-8e58-b9372c7e1542.gif)

![Screen Recording 2021-01-05 at 10 00 13 PM](https://user-images.githubusercontent.com/26736844/103687609-9ef33500-4fa1-11eb-814c-ac5488309fa4.gif)


## How does it work?

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

## Dashboard of Components

### 1. CLV Prediction Time Line

Based on the result_data.csv file, all previously calculated results are combined and shown in the line chart.

<img width="1641" alt="Screen Shot 2021-01-05 at 22 22 31" src="https://user-images.githubusercontent.com/26736844/103690845-5ee28100-4fa6-11eb-9f38-f44a94791cc8.png">

### 2. Churn Customers Of Purchase TimeLine

Based on the selected date in the **CLV Prediction Time Line**, 
customers who purchased before the selected date but never had an order in the prediction time period are detected.
These are the churn customers for the selected date.

<img width="373" alt="Screen Shot 2021-01-05 at 22 37 41" src="https://user-images.githubusercontent.com/26736844/103691245-eaf4a880-4fa6-11eb-808d-9a13f12db05f.png">

### 3. Newcomer Customers Of Purchase TimeLine

Based on the selected date in the **CLV Prediction Time Line**, 
customers who are newcomers at the selected date and haven't purchased before it are detected.
These are the newcomer customers for the selected date.

<img width="329" alt="Screen Shot 2021-01-05 at 22 38 19" src="https://user-images.githubusercontent.com/26736844/103691225-e4fec780-4fa6-11eb-839f-18567e14f9de.png">

### 4. Top 100 the Least Engaged Customers Of Sum Values per month


Customers who have fewer purchase amounts than others, based on the sum/mean of purchase amounts in the timeline.
These customers can be selected individually via the **Worst Customer List** filter.

<img width="400" alt="Screen Shot 2021-01-05 at 22 37 03" src="https://user-images.githubusercontent.com/26736844/103691256-f0ea8980-4fa6-11eb-82a4-9edf3866c768.png">

### 5. Top 100 the Most Engaged Customers Of Sum Values per month

Customers who have more purchase amounts than others, based on the sum/mean of purchase amounts in the timeline.
These customers can be selected individually via the **Top Customer List** filter.

<img width="389" alt="Screen Shot 2021-01-05 at 22 36 11" src="https://user-images.githubusercontent.com/26736844/103690943-7d487c80-4fa6-11eb-9981-4ee890fac404.png">

## 6. Churn Rate and Newcomer Rate per month

These pie charts show the Newcomer and Churn Rate of the business, based on the selected date in the **CLV Prediction Timeline**.

<img width="761" alt="Screen Shot 2021-01-05 at 23 53 11" src="https://user-images.githubusercontent.com/26736844/103697747-678c8480-4fb1-11eb-8474-923cbde1b6fe.png">
