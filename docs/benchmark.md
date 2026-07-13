# Benchmark

This page reports a reproducible benchmark of the CLV prediction pipeline (`clv.executor.CLV`) against a naive
heuristic baseline, run on synthetic transaction data. The benchmark tooling lives in the
[`benchmark/`](https://github.com/caglanakpinar/clv_prediction/tree/main/benchmark) directory and can be re-run with:

    poetry run python benchmark/run_benchmark.py

## Methodology

1. **Sample data** &mdash; `benchmark/generate_data.py` generates synthetic transactions for 400 customers across
   4 segments (`champion`, `regular`, `occasional`, `one_time`) so that the next-purchase, purchase-amount, and
   newcomers models are all exercised. Amounts are lognormal per customer; order dates follow segment-specific
   exponential inter-purchase gaps. A fixed seed (`42`) makes the run reproducible.
2. **Train / holdout split** &mdash; 1 year of history (`2024-01-01` &rarr; `2024-12-31`) is used for training; the
   following 30 days (`time_period="month"`) are held out as ground truth. Actual per-customer spend in the holdout
   window is the target the benchmark evaluates against.
3. **Baseline** &mdash; `benchmark/baseline.py` computes a standard naive CLV estimate per customer using only the
   training window: `avg_order_value x (order_count / customer_span_days) x horizon_days`. Single-order customers
   fall back to the dataset-wide average purchase rate (cold-start).
4. **CLV model** &mdash; the training CSV and cutoff date are handed to `clv.executor.CLV(job="train_prediction", ...)`,
   which trains and predicts all three models (next-purchase, purchase-amount, newcomers) exactly as a real user
   would call it. The benchmark runs this twice:
      - **default** &mdash; hyperparameter tuning is skipped (the package's own bundled default parameters are used
        directly); each model fits once at its documented default epoch count. Fast and reproducible.
      - **tuned** &mdash; real keras-tuner `RandomSearch` hyperparameter tuning runs for each model, exactly as it
        would for a user who leaves tuning on. Much slower, but reflects the best-case out-of-the-box training path.
5. **Evaluation** &mdash; `benchmark/metrics.py` implements MAE, RMSE, and SMAPE (symmetric, to handle customers with
   zero holdout spend) by hand, plus a portfolio-level total-spend comparison. The baseline and both model runs are
   scored against the same actual holdout totals.

## Dataset

| | |
|---|---|
| customers | 381 (of 400 requested; a few segment/date combinations produced no in-window orders) |
| transactions | 1,782 |
| training window | 2024-01-01 &rarr; 2024-12-31 (365 days) |
| holdout window | 2024-12-31 &rarr; 2025-01-30 (30 days) |
| time_period | `month` |

## Results

| metric | baseline | CLV (default) | CLV (tuned) |
|---|---:|---:|---:|
| MAE (per customer) | 139.50 | 213.89 | 199.87 |
| RMSE (per customer) | 232.53 | 1,105.77 | 965.22 |
| SMAPE (per customer) | 86.1% | 99.2% | 98.3% |
| **Portfolio error** (total predicted vs. total actual) | **319.4%** | **8.1%** | **19.5%** |

| | total actual | total predicted |
|---|---:|---:|
| baseline | 13,355.63 | 56,017.09 |
| CLV (default) | 13,355.63 | 12,275.63 |
| CLV (tuned) | 13,355.63 | 10,746.82 |

## Interpretation

The two approaches win on different axes, and it's worth being honest about both:

- **Per-customer point accuracy (MAE/RMSE/SMAPE): the naive baseline wins**, though hyperparameter tuning narrows
  the gap (tuned MAE/RMSE/SMAPE all improve over default). The CLV model's per-customer predictions still have
  higher average error and higher variance than the baseline's &mdash; a synthetic dataset this size gives the deep
  models relatively little signal per customer, and a handful of large individual misses drive RMSE up sharply.
- **Portfolio-level accuracy (aggregate total spend): the CLV model wins decisively either way.** The baseline's
  formulaic extrapolation (average order value x observed purchase rate x horizon) is highly sensitive to short
  observation windows &mdash; a customer with two orders a few days apart implies an implausibly high purchase rate
  once extrapolated, and these overshoots compound across the customer base to a 319% total error. Both CLV runs
  land within 20% of the actual total; interestingly, the untuned default run (8.1% error) landed closer to the
  actual total than the tuned run (19.5%) on this particular seed &mdash; tuning optimizes each model's own loss
  function, not portfolio-level total error, so it isn't guaranteed to improve the aggregate number even as it
  improves per-customer accuracy.

For CLV use cases that are ultimately about aggregate decisions &mdash; budgeting, cohort-level value, marketing
spend allocation &mdash; the portfolio-level result is usually the one that matters, and it's where the trained
pipeline clearly outperforms the naive heuristic regardless of tuning. For use cases that depend on precise
per-customer point estimates, tuning helps but this benchmark suggests the model still needs more data before it
reliably beats a simple heuristic at that granularity.
