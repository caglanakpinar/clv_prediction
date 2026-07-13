"""End-to-end benchmark: synthetic data -> naive baseline vs. clv package models.

Runs the CLV pipeline twice: once using the package's bundled default
hyperparameters (tuning skipped, fast), and once with real keras-tuner
hyperparameter search enabled (slow, but faithful to the best-case training
path a user would get by leaving order_count/params untouched).

Usage:
    poetry run python benchmark/run_benchmark.py
"""

import os
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.baseline import compute_baseline_predictions  # noqa: E402
from benchmark.generate_data import generate_sample_transactions  # noqa: E402
from benchmark.metrics import evaluate_predictions  # noqa: E402

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BENCHMARK_DIR, "artifacts")
DATA_CSV = os.path.join(ARTIFACTS_DIR, "sample_transactions.csv")

CUSTOMER_INDICATOR = "customer_id"
TIME_INDICATOR = "order_date"
AMOUNT_INDICATOR = "amount"
TIME_PERIOD = "month"
HORIZON_DAYS = 30
TRAIN_DAYS = 365
START_DATE = "2024-01-01"
SEED = 42


def write_default_params_for_export_path(export_path):
    """Pre-populate export_path/test_parameters.yaml with the package's bundled
    default params, in the flat shape check_for_existing_parameters expects, so
    every model skips keras-tuner hyperparameter search and fits once."""
    clv_docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clv", "docs"
    )
    with open(os.path.join(clv_docs_dir, "test_parameters.yaml")) as f:
        bundled = yaml.full_load(f)

    flat = {name: bundled["models"][name]["params"] for name in ("next_purchase", "purchase_amount", "newcomers")}
    with open(os.path.join(export_path, "test_parameters.yaml"), "w") as f:
        yaml.dump(flat, f, default_flow_style=False)


def run_clv_pipeline(export_path, data_csv, cutoff_str, tune):
    """Train + predict via clv.executor.CLV. If tune=False, pre-populate
    test_parameters.yaml so every model skips keras-tuner search and fits
    once at the package defaults. If tune=True, leave export_path empty so
    each model runs its real RandomSearch hyperparameter tuning first."""
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    os.makedirs(export_path, exist_ok=True)

    if not tune:
        write_default_params_for_export_path(export_path)

    from clv.executor import CLV

    clv = CLV(
        customer_indicator=CUSTOMER_INDICATOR,
        amount_indicator=AMOUNT_INDICATOR,
        time_indicator=TIME_INDICATOR,
        job="train_prediction",
        date=cutoff_str,
        order_count=None,
        data_source="csv",
        data_query_path=data_csv,
        time_period=TIME_PERIOD,
        export_path=export_path,
        connector=None,
    )
    clv.clv_prediction()
    results = clv.get_result_data()

    predicted_rows = results[results["data_type"] == "prediction"]
    model_df = (
        predicted_rows.groupby(CUSTOMER_INDICATOR)[AMOUNT_INDICATOR]
        .sum()
        .reset_index()
        .rename(columns={AMOUNT_INDICATOR: "predicted_clv"})
    )
    return model_df


def print_comparison(baseline_scores, scored_models):
    """scored_models: list of (label, scores_dict)."""
    header = f"{'metric':<20}{'baseline':>15}" + "".join(f"{label:>18}" for label, _ in scored_models)
    print(header)
    for metric in ("mae", "rmse", "smape", "portfolio_error_pct"):
        row = f"{metric:<20}{baseline_scores[metric]:>15.3f}"
        for _, scores in scored_models:
            row += f"{scores[metric]:>18.3f}"
        print(row)
    print(f"total actual holdout spend: {baseline_scores['total_actual']:.2f}")
    print(f"total baseline predicted:   {baseline_scores['total_predicted']:.2f}")
    for label, scores in scored_models:
        print(f"total {label} predicted:{' ' * max(1, 12 - len(label))}{scores['total_predicted']:.2f}")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("1/6 generating sample data ...")
    data = generate_sample_transactions(
        start_date=START_DATE, train_days=TRAIN_DAYS, holdout_days=HORIZON_DAYS, seed=SEED
    )
    data.to_csv(DATA_CSV, index=False)
    cutoff = pd.Timestamp(START_DATE) + pd.Timedelta(days=TRAIN_DAYS)
    holdout_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    print(f"    {len(data)} rows, {data[CUSTOMER_INDICATOR].nunique()} customers, cutoff={cutoff_str}")

    dates = pd.to_datetime(data[TIME_INDICATOR])
    train_df = data[dates < cutoff].copy()
    holdout_df = data[(dates >= cutoff) & (dates < holdout_end)].copy()

    print("2/6 computing ground truth holdout spend per customer ...")
    actual_df = (
        holdout_df.groupby(CUSTOMER_INDICATOR)[AMOUNT_INDICATOR]
        .sum()
        .reset_index()
        .rename(columns={AMOUNT_INDICATOR: "actual_clv"})
    )

    print("3/6 computing naive baseline predictions ...")
    baseline_df = compute_baseline_predictions(
        train_df, CUSTOMER_INDICATOR, TIME_INDICATOR, AMOUNT_INDICATOR, HORIZON_DAYS
    )

    print("4/6 training + predicting with clv.executor.CLV (default params, tuning skipped) ...")
    default_model_df = run_clv_pipeline(
        os.path.join(ARTIFACTS_DIR, "export_default"), DATA_CSV, cutoff_str, tune=False
    )

    print("5/6 training + predicting with clv.executor.CLV (hyperparameter tuning enabled) ...")
    tuned_model_df = run_clv_pipeline(
        os.path.join(ARTIFACTS_DIR, "export_tuned"), DATA_CSV, cutoff_str, tune=True
    )

    print("6/6 evaluating baseline vs. both models ...")
    baseline_scores = evaluate_predictions(
        actual_df, baseline_df, CUSTOMER_INDICATOR, "actual_clv", "predicted_clv"
    )
    default_scores = evaluate_predictions(
        actual_df, default_model_df, CUSTOMER_INDICATOR, "actual_clv", "predicted_clv"
    )
    tuned_scores = evaluate_predictions(
        actual_df, tuned_model_df, CUSTOMER_INDICATOR, "actual_clv", "predicted_clv"
    )

    print("\n=== Benchmark Results ===")
    print_comparison(baseline_scores, [("clv default", default_scores), ("clv tuned", tuned_scores)])

    return {
        "n_customers": int(data[CUSTOMER_INDICATOR].nunique()),
        "n_rows": int(len(data)),
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(holdout_df)),
        "cutoff": cutoff_str,
        "horizon_days": HORIZON_DAYS,
        "time_period": TIME_PERIOD,
        "baseline": baseline_scores,
        "model_default": default_scores,
        "model_tuned": tuned_scores,
    }


if __name__ == "__main__":
    main()
