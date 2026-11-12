"""Ranking metrics. MAP@7 is the Kaggle competition metric.

AP@k for one customer: walk down the ranked list; at each position i that is a hit,
add precision@i; divide by min(number of actual products, k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apk(actual: list, predicted: list, k: int = 7) -> float:
    if not actual:
        return 0.0
    predicted = predicted[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual: list[list], predicted: list[list], k: int = 7) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])) if actual else 0.0


def ranking_report(actual: pd.Series, ranked: pd.Series, k: int = 7) -> dict:
    """actual / ranked: customer id -> list of products (same index)."""
    ranked = ranked.reindex(actual.index)
    a, r = actual.tolist(), ranked.tolist()
    return {
        "customers": len(a),
        f"map@{k}": mapk(a, r, k),
        "precision@1": float(np.mean([x[0] in y if x else 0.0 for x, y in zip(r, a)])),
        f"recall@{k}": float(np.mean([len(set(x[:k]) & set(y)) / len(y) for x, y in zip(r, a)])),
    }


def per_product_recall(actual: pd.Series, ranked: pd.Series, k: int = 7) -> pd.DataFrame:
    """For each product: how often an addition of it was in the customer's top k."""
    ranked = ranked.reindex(actual.index)
    rows = {}
    for a, r in zip(actual, ranked):
        top = set(r[:k])
        for p in a:
            n, hit = rows.get(p, (0, 0))
            rows[p] = (n + 1, hit + (p in top))
    df = pd.DataFrame([(p, n, h, h / n) for p, (n, h) in rows.items()],
                      columns=["product", "additions", f"in_top{k}", f"recall@{k}"])
    return df.sort_values("additions", ascending=False).reset_index(drop=True)


def bootstrap_diff(ap_a: np.ndarray, ap_b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of MAP(a) - MAP(b) over the same customers.

    Resample customers with replacement, recompute the difference each time, and take
    the 2.5 and 97.5 percentiles. If the interval contains 0, the data cannot tell the
    two models apart."""
    rng = np.random.default_rng(seed)
    d = np.asarray(ap_a) - np.asarray(ap_b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5))}
