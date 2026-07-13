"""Hand-rolled evaluation metrics (no scikit-learn dependency in this project)."""

import numpy as np


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual, predicted):
    denom = np.abs(actual) + np.abs(predicted)
    ratio = np.where(denom == 0, 0.0, np.abs(actual - predicted) / denom)
    return float(np.mean(ratio) * 100)


def evaluate_predictions(actual_df, predicted_df, customer_indicator, actual_col, predicted_col):
    """Outer-join actual vs. predicted per-customer values and score them.

    Customers missing from either side are treated as 0 for that side (a model
    that fails to predict for a customer is penalized, not excluded).
    """
    merged = actual_df.merge(predicted_df, on=customer_indicator, how="outer")
    merged[actual_col] = merged[actual_col].fillna(0.0)
    merged[predicted_col] = merged[predicted_col].fillna(0.0)

    actual = merged[actual_col].to_numpy(dtype=float)
    predicted = merged[predicted_col].to_numpy(dtype=float)

    total_actual = float(actual.sum())
    total_predicted = float(predicted.sum())
    portfolio_error_pct = (
        abs(total_predicted - total_actual) / total_actual * 100 if total_actual != 0 else 0.0
    )

    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "total_actual": total_actual,
        "total_predicted": total_predicted,
        "portfolio_error_pct": portfolio_error_pct,
        "n_customers": int(len(merged)),
    }
