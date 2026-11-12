"""Features for predicting what a customer adds in month m.

Rule: a feature for month m may use customer attributes from month m (the bank knows
them at the start of the month) and product ownership from months before m only.
Product ownership in month m is the answer, so it never enters a feature.
tests/test_leakage.py enforces this.

Feature groups (the same families the 2nd-place Kaggle solution used):
    customer      age, seniority, income (+ missing flag), segment, province, channel, ...
    lags          ownership of each product 1..L months ago
    history       per product: times added and dropped in the last H months
    totals        products owned last month, total adds and drops, months of history
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prodrec.data.store import MonthStore
from prodrec.schema import CATEGORICAL, NUMERIC, PRODUCT_COLS, SHORT


def build_features(store: MonthStore, month: str, customers: pd.Index | None = None,
                   lags: int = 5, history_months: int = 5) -> pd.DataFrame:
    snap = store.load(month)
    idx = snap.index if customers is None else customers
    attrs = snap.reindex(idx)
    cols: dict[str, object] = {}
    for col in NUMERIC:
        cols[col] = attrs[col].astype("float32").to_numpy()
    cols["renta_missing"] = attrs["renta"].isna().astype("int8").to_numpy()
    for col in CATEGORICAL:
        cols[col] = attrs[col].astype("object").to_numpy()

    months = store.months()
    i = months.index(month)
    prev_months = [months[i - k] for k in range(1, max(lags, history_months) + 1) if i - k >= 0]

    owned = {}  # lag -> DataFrame (customers x products), NaN where the customer did not exist yet
    for k, m in enumerate(prev_months, start=1):
        owned[k] = store.products(m).reindex(idx).astype("float32")

    nan = np.full(len(idx), np.nan, dtype="float32")
    for k in range(1, lags + 1):
        lag = owned.get(k)
        for col in PRODUCT_COLS:
            cols[f"{SHORT[col]}_lag{k}"] = lag[col].to_numpy() if lag is not None else nan

    n_hist = min(history_months, len(prev_months))
    adds = np.zeros((len(idx), len(PRODUCT_COLS)), dtype="float32")
    drops = np.zeros_like(adds)
    for k in range(1, n_hist):
        newer, older = owned[k].fillna(0).to_numpy(), owned[k + 1].fillna(0).to_numpy()
        adds += (newer == 1) & (older == 0)
        drops += (newer == 0) & (older == 1)
    for j, col in enumerate(PRODUCT_COLS):
        cols[f"{SHORT[col]}_adds"] = adds[:, j]
        cols[f"{SHORT[col]}_drops"] = drops[:, j]

    cols["n_owned_lag1"] = owned[1].sum(axis=1, min_count=1).to_numpy("float32") if 1 in owned else nan
    cols["n_adds"] = adds.sum(axis=1)
    cols["n_drops"] = drops.sum(axis=1)
    cols["months_of_history"] = (np.column_stack([o.notna().any(axis=1).to_numpy() for o in owned.values()])
                                 .sum(axis=1).astype("float32") if owned else np.zeros(len(idx), "float32"))
    return pd.DataFrame(cols, index=idx)


def owned_last_month(store: MonthStore, month: str, customers: pd.Index) -> np.ndarray:
    """0/1 array (customers x 24): products owned in the month before `month`."""
    prev = store.previous(month)
    if prev is None:
        return np.zeros((len(customers), len(PRODUCT_COLS)), dtype=bool)
    return store.products(prev).reindex(customers).fillna(0).to_numpy().astype(bool)


class CategoryEncoder:
    """Fix the category list on training data so train and test use the same codes."""

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        self.categories = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, cats in self.categories.items():
            df[c] = pd.Categorical(df[c].astype("object").where(df[c].notna(), None), categories=cats)
        return df
