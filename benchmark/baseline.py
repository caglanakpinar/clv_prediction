"""Naive baseline CLV prediction: average order value x purchase frequency x horizon."""

import pandas as pd


def compute_baseline_predictions(
    train_df,
    customer_indicator,
    time_indicator,
    amount_indicator,
    horizon_days,
):
    """Per-customer baseline CLV = avg_order_value * (order_count / span_days) * horizon_days.

    Customers with a single order (span_days == 0) fall back to the dataset-wide
    average purchase rate among customers with >= 2 orders, since a per-day rate
    can't be derived from one data point.
    """
    df = train_df.copy()
    df[time_indicator] = pd.to_datetime(df[time_indicator])

    grouped = df.groupby(customer_indicator)[time_indicator].agg(["min", "max", "count"])
    grouped["span_days"] = (grouped["max"] - grouped["min"]).dt.days
    avg_amount = df.groupby(customer_indicator)[amount_indicator].mean()

    multi_order = grouped[grouped["span_days"] > 0]
    fallback_rate = (
        (multi_order["count"] / multi_order["span_days"]).mean()
        if len(multi_order) > 0
        else 1.0 / horizon_days
    )

    purchase_rate = grouped.apply(
        lambda row: row["count"] / row["span_days"] if row["span_days"] > 0 else fallback_rate,
        axis=1,
    )

    baseline_clv = avg_amount * purchase_rate * horizon_days

    result = baseline_clv.reset_index()
    result.columns = [customer_indicator, "predicted_clv"]
    return result
