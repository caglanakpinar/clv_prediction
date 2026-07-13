# Customer Lifetime Value Prediction

---------------------------

[![PyPI version](https://badge.fury.io/py/clv-prediction.svg)](https://badge.fury.io/py/clv-prediction)
[![GitHub license](https://img.shields.io/github/license/caglanakpinar/clv_prediction)](https://github.com/caglanakpinar/clv_prediction/blob/master/LICENSE)

----------------------------

[CLV Prediction Documents](https://caglanakpinar.github.io/clv_prediction/)

This framework generates 2 main predictive models per customer.
First, the Next Purchase (Frequency) Model is trained.
This model helps predict the day of the next purchase per customer.
Second, the Customer Value Model is trained.
This model helps predict what the amount of the next purchase will be per customer.
There will be customers who cannot be predicted by the models above because of a lack of historical information.
Those customers are NewComers.
This platform allows us to predict NewComers' total lifetime values as well.

## How It Works

Three deep learning models (Keras/TensorFlow, LSTM & 1D-Conv LSTM) work together, in this order:

1. **Next Purchase Model** &mdash; predicts *when* each customer's next order will happen.
2. **Purchase Amount Model** &mdash; predicts *how much* each of those predicted future orders will be worth.
3. **NewComers Model** &mdash; separately predicts total lifetime value for customers with too little history for the two models above.

The predicted purchase amounts (engaged customers) and predicted NewComer values are combined into a single
result set covering the whole customer base.

![CLV Prediction Pipeline](https://user-images.githubusercontent.com/26736844/118328794-da34e000-b50e-11eb-8a7f-3a10373f8461.png)

See [how to run](https://caglanakpinar.github.io/clv_prediciton/intro/) for the full walkthrough.

## Benchmark Results

Benchmarked against a naive baseline (avg order value x purchase frequency x horizon) on synthetic transaction
data (381 customers, 1yr train / 30d holdout). Full methodology and numbers: [docs/benchmark.md](docs/benchmark.md).

| metric | baseline | CLV (default) | CLV (tuned) |
|---|---:|---:|---:|
| MAE (per customer) | 139.50 | 213.89 | 199.87 |
| RMSE (per customer) | 232.53 | 1,105.77 | 965.22 |
| **Portfolio error** (total predicted vs. actual) | **319.4%** | **8.1%** | **19.5%** |

The naive baseline can look competitive per-customer, but the trained pipeline is dramatically more accurate at
the portfolio/aggregate level &mdash; the level most CLV decisions (budgeting, cohort value) are actually made at.
Hyperparameter tuning narrows the per-customer gap further, though it isn't guaranteed to improve the aggregate
number since tuning optimizes each model's own loss, not portfolio-level total error.

## Dashboard

Once training/prediction is done, `clv.show_dashboard()` launches an interactive Dash app to explore results:
a CLV prediction timeline, churn & newcomer breakdowns, top/worst customer lists, and churn/newcomer rate charts.

![CLV Dashboard](https://user-images.githubusercontent.com/26736844/103687181-e9c07d00-4fa0-11eb-8e58-b9372c7e1542.gif)

<img width="1641" alt="CLV Prediction Time Line" src="https://user-images.githubusercontent.com/26736844/103690845-5ee28100-4fa6-11eb-9f38-f44a94791cc8.png">

Full walkthrough with every dashboard component: [docs/monitoring.md](docs/monitoring.md).

## Installation

This tool can be installed like any other package, via pypi or git:

```bash
poetry add clv_prediction
```
OR

```bash
poetry add git+https://github.com/caglanakpinar/clv_prediction.git
```

## Quick Start

```python
from clv.executor import CLV

clv = CLV(
    customer_indicator="user_id",
    amount_indicator="transaction_value",
    time_indicator="days",
    job="train_prediction",          # "train", "prediction", or "train_prediction"
    date="2021-01-01",               # optional cutoff filter date
    order_count=15,                  # optional; None -> auto-detected
    data_source="csv",               # or postgresql/mysql/awsredshift/googlebigquery/json/parquet
    time_period="month",             # "week", "2*week", "month", "2*month", "quarter", "6*month"
    data_query_path="./data.csv",
    export_path="./data",
    connector=None,                  # dict of db creds, only needed for non-file sources
)
clv.clv_prediction()                 # trains/predicts all 3 sub-models
results = clv.get_result_data()      # actual + predicted rows per customer
```

## Documentation

| Page | Covers |
|---|---|
| [Getting Started](https://caglanakpinar.github.io/clv_prediciton/) | Overview & installation |
| [How to Run](https://caglanakpinar.github.io/clv_prediciton/intro/) | Concepts, architecture, pipeline |
| [Configurations](https://caglanakpinar.github.io/clv_prediciton/params/) | Every `CLV(...)` argument explained |
| [Data Access](https://caglanakpinar.github.io/clv_prediciton/data_access/) | Supported data sources & connection examples |
| [Pre-process](https://caglanakpinar.github.io/clv_prediciton/data_preprocess/) | Feature engineering per model |
| [Train](https://caglanakpinar.github.io/clv_prediciton/train/) | `train` / `prediction` / `train_prediction` jobs |
| [Hyperparameters](https://caglanakpinar.github.io/clv_prediciton/tune/) | Tuning approach |
| [Benchmark](https://caglanakpinar.github.io/clv_prediciton/benchmark/) | CLV model vs. naive baseline |
| [Monitoring](https://caglanakpinar.github.io/clv_prediciton/monitoring/) | Dashboard walkthrough |

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
    benchmark/
        generate_data.py
        baseline.py
        metrics.py
        run_benchmark.py
