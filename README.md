# Customer Lifetime Value Prediction

---------------------------

[![PyPI version](https://badge.fury.io/py/clv-prediction.svg)](https://badge.fury.io/py/clv-prediction)
[![GitHub license](https://img.shields.io/github/license/caglanakpinar/clv_prediction)](https://github.com/caglanakpinar/clv_prediction/blob/master/LICENSE)

----------------------------

[CLV Prediction Documents](https://caglanakpinar.github.io/clv_prediciton/)



This framework we generate 2 main predictive model per customer. 
First, Next Purchase (Frequency) Model will be trained. 
This model will help us to predict the day of nex purchases per customer
Second, Customer Value Model will be trained. 
THis model will help us to predict what will be the amount of next purchases per customer.
There will be customers can not be predicted by those models above because of lack historical informations. 
Those customers are NewComers.
This platform allows us to predict NewComers' total lifetime values as well.

## Installation

Tool can be used any other package by install it via pypi or git command

```bash
poetry add clv_prediction
```
OR

```bash
poetry add git+https://github.com/caglanakpinar/clv_prediction.git
```

## Project layout

    clv/
        docs/   
            - configs.yaml
            - test_parameters.yaml
        confgis.py
        dashboard.py
        data_access.py
        executor.py
        functions.py
        main.py
        newcomers.py
        next_purchase_model.py
        next_purchase_prediction.py
        purchase_amount_model.py
        utils.py
