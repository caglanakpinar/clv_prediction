# Customer Lifetime Value Prediction

---------------------------

[![PyPI version](https://badge.fury.io/py/clv-prediction.svg)](https://badge.fury.io/py/clv-prediction)
[![GitHub license](https://img.shields.io/github/license/caglanakpinar/clv_prediction)](https://github.com/caglanakpinar/clv_prediction/blob/master/LICENSE)

----------------------------

This framework generates 2 main predictive models per customer. 
First, the Next Purchase (Frequency) Model is trained. 
This model helps predict the day of the next purchase per customer.
Second, the Customer Value Model is trained. 
This model helps predict what the amount of the next purchase will be per customer.
There will be customers who cannot be predicted by the models above because of a lack of historical information. 
Those customers are NewComers.
This platform allows us to predict NewComers' total lifetime values as well.

## Installation

This tool can be installed like any other package, via pypi or git:

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
        configs.py
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
