"""What did each customer add this month?

A product is "added" in month m if the customer owns it in m and did not own it in m-1.
A customer missing from m-1 (new customer) counts as owning nothing in m-1, the same
rule Kaggle used for the June 2016 test set.
"""

from __future__ import annotations

import pandas as pd

from prodrec.data.store import MonthStore
from prodrec.schema import PRODUCT_COLS


def added_matrix(store: MonthStore, month: str) -> pd.DataFrame:
    """0/1 matrix (customers in `month` x 24 products) of products added in `month`."""
    prev = store.previous(month)
    if prev is None:
        raise ValueError(f"{month} is the first month; nothing to compare against")
    now = store.products(month)
    before = store.products(prev).reindex(now.index).fillna(0).astype("int8")
    return ((now == 1) & (before == 0)).astype("int8")


def added_lists(store: MonthStore, month: str) -> pd.Series:
    """customer id -> list of added product columns (only customers who added something)."""
    m = added_matrix(store, month)
    m = m[m.sum(axis=1) > 0]
    cols = pd.Index(PRODUCT_COLS)
    return pd.Series([list(cols[row.astype(bool)]) for row in m.to_numpy()], index=m.index, name="added")
