"""Synthetic transaction data generator used to benchmark the clv package."""

import numpy as np
import pandas as pd

SEGMENTS = {
    "champion": {"share": 0.15, "order_count": (8, 20), "mean_amount": (80, 160), "mean_gap_days": 18},
    "regular": {"share": 0.50, "order_count": (4, 9), "mean_amount": (40, 90), "mean_gap_days": 35},
    "occasional": {"share": 0.20, "order_count": (2, 4), "mean_amount": (25, 70), "mean_gap_days": 60},
    "one_time": {"share": 0.15, "order_count": (1, 1), "mean_amount": (20, 60), "mean_gap_days": 0},
}


def _customer_segment(rng):
    names = list(SEGMENTS.keys())
    probs = [SEGMENTS[n]["share"] for n in names]
    return rng.choice(names, p=probs)


def _order_dates_for_customer(rng, segment, join_day, period_days):
    spec = SEGMENTS[segment]
    if segment == "one_time":
        order_day = rng.integers(low=int(period_days * 0.7), high=period_days)
        return [order_day]

    order_count = rng.integers(spec["order_count"][0], spec["order_count"][1] + 1)
    gaps = rng.exponential(scale=spec["mean_gap_days"], size=order_count)
    days = join_day + np.cumsum(gaps)
    days = days[days < period_days]
    return days.astype(int).tolist()


def generate_sample_transactions(
    n_customers=400,
    start_date="2024-01-01",
    train_days=365,
    holdout_days=30,
    seed=42,
):
    """Generate a synthetic long-format transaction table.

    Columns: customer_id, order_date (YYYY-MM-DD), amount.
    Orders span [start_date, start_date + train_days + holdout_days). A caller can
    treat `start_date + train_days` as the train/holdout cutoff.
    """
    rng = np.random.default_rng(seed)
    period_days = train_days + holdout_days
    base_date = pd.Timestamp(start_date)

    rows = []
    for i in range(n_customers):
        customer_id = f"user_{i:05d}"
        segment = _customer_segment(rng)
        join_day = rng.integers(0, train_days)
        order_days = _order_dates_for_customer(rng, segment, join_day, period_days)
        if not order_days:
            continue

        mean_amount = rng.uniform(*SEGMENTS[segment]["mean_amount"])
        for day in order_days:
            noise = rng.lognormal(mean=0.0, sigma=0.25)
            amount = round(float(mean_amount * noise), 2)
            order_date = (base_date + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")
            rows.append((customer_id, order_date, amount))

    data = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    return data.sort_values(["customer_id", "order_date"]).reset_index(drop=True)


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_sample_transactions()
    df.to_csv(os.path.join(out_dir, "sample_transactions.csv"), index=False)
    print(f"wrote {len(df)} rows for {df['customer_id'].nunique()} customers")
