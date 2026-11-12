"""Recommenders. All of them return, for each customer, the products they do not own
yet, best first. Recommending something the customer already has is a wasted slot.

    popularity          most-added products in the training months, same list for everyone
    segment_popularity  most-added products per (segment, age band), falls back to popularity
    LGBMRecommender     LightGBM multiclass model: one training row per (customer, added
                        product); predicted class probabilities give the ranking
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prodrec.features import CategoryEncoder
from prodrec.schema import CATEGORICAL, PRODUCT_COLS

P_INDEX = {p: i for i, p in enumerate(PRODUCT_COLS)}


def rank_from_scores(scores: np.ndarray, owned: np.ndarray, index: pd.Index, k: int = 7) -> pd.Series:
    """scores: customers x 24. Owned products are pushed to the bottom, then top k kept."""
    s = np.where(owned, -np.inf, scores)
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    cols = np.array(PRODUCT_COLS)
    return pd.Series([list(cols[row][np.isfinite(s[i, row])]) for i, row in enumerate(top)], index=index)


# ---------------- baselines ----------------

def popularity_scores(train_added: pd.DataFrame) -> np.ndarray:
    """train_added: rows = (customer, month), 0/1 per product. Returns share of adds per product."""
    counts = train_added[PRODUCT_COLS].sum().to_numpy(dtype=float)
    return counts / counts.sum()


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [-1, 25, 40, 60, 200], labels=["<=25", "26-40", "41-60", ">60"]).astype(str)


def segment_popularity_scores(train_added: pd.DataFrame, train_feats: pd.DataFrame,
                              test_feats: pd.DataFrame, min_adds: int = 50) -> np.ndarray:
    glob = popularity_scores(train_added)
    key_tr = train_feats["segmento"].astype(str) + "|" + age_band(train_feats["age"])
    key_te = test_feats["segmento"].astype(str) + "|" + age_band(test_feats["age"])
    grouped = train_added[PRODUCT_COLS].groupby(key_tr.to_numpy()).sum()
    table = {}
    for key, row in grouped.iterrows():
        total = row.sum()
        table[key] = row.to_numpy(float) / total if total >= min_adds else glob
    return np.vstack([table.get(k, glob) for k in key_te])


# ---------------- LightGBM ----------------

class LGBMRecommender:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def _lgb_params(self) -> dict:
        p = self.params
        return {"objective": "multiclass", "num_class": len(PRODUCT_COLS), "metric": "multi_logloss",
                "learning_rate": p.get("learning_rate", 0.05), "num_leaves": p.get("num_leaves", 63),
                "min_data_in_leaf": p.get("min_data_in_leaf", 50),
                "feature_fraction": p.get("feature_fraction", 0.8),
                "bagging_fraction": p.get("bagging_fraction", 0.8), "bagging_freq": p.get("bagging_freq", 1),
                "lambda_l2": p.get("lambda_l2", 1.0), "seed": p.get("seed", 42), "verbose": -1,
                "num_threads": p.get("num_threads", 0)}

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None) -> "LGBMRecommender":
        """With a validation set: find the number of trees by early stopping, then refit
        on train + validation with that number. The evaluation month is never used here."""
        self.encoder = CategoryEncoder().fit(X if X_val is None else pd.concat([X, X_val]))
        self.features = list(X.columns)
        rounds = self.params.get("num_boost_round", 400)
        if X_val is not None:
            dtr = lgb.Dataset(self.encoder.transform(X), y, categorical_feature=CATEGORICAL, free_raw_data=False)
            dva = lgb.Dataset(self.encoder.transform(X_val), y_val, reference=dtr, categorical_feature=CATEGORICAL)
            probe = lgb.train(self._lgb_params(), dtr, rounds, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(self.params.get("early_stopping_rounds", 30), verbose=False)])
            self.best_iteration = probe.best_iteration or rounds
            X, y = pd.concat([X, X_val]), np.concatenate([y, y_val])
        else:
            self.best_iteration = rounds
        dall = lgb.Dataset(self.encoder.transform(X), y, categorical_feature=CATEGORICAL)
        self.booster = lgb.train(self._lgb_params(), dall, self.best_iteration)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.encoder.transform(X[self.features]))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions to each class score (TreeSHAP, built into LightGBM).
        Shape: customers x 24 products x (features + 1 bias)."""
        raw = self.booster.predict(self.encoder.transform(X[self.features]), pred_contrib=True)
        return raw.reshape(len(X), len(PRODUCT_COLS), len(self.features) + 1)

    def importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance("gain")
        df = pd.DataFrame({"feature": self.booster.feature_name(), "gain": gain})
        df["share"] = df["gain"] / df["gain"].sum()
        return df.sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(folder / "model.txt"))
        (folder / "meta.json").write_text(json.dumps({
            "features": self.features, "categories": self.encoder.categories,
            "best_iteration": self.best_iteration, "products": PRODUCT_COLS, "params": self.params,
        }, indent=1))

    @classmethod
    def load(cls, folder: Path) -> "LGBMRecommender":
        meta = json.loads((folder / "meta.json").read_text())
        m = cls(meta["params"])
        m.booster = lgb.Booster(model_file=str(folder / "model.txt"))
        m.features, m.best_iteration = meta["features"], meta["best_iteration"]
        m.encoder = CategoryEncoder()
        m.encoder.categories = meta["categories"]
        return m
