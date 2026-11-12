"""Steps: prepare -> train -> evaluate (+ experiments). dataset = "real" or "demo"."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from prodrec.config import Config
from prodrec.data.store import MonthStore
from prodrec.features import build_features, owned_last_month
from prodrec.labels import added_lists, added_matrix
from prodrec.metrics import apk, bootstrap_diff, per_product_recall, ranking_report
from prodrec.models import (P_INDEX, LGBMRecommender, popularity_scores, rank_from_scores,
                            segment_popularity_scores)
from prodrec.schema import PRODUCT_COLS, PRODUCTS, SHORT


def prepare(cfg: Config, dataset: str) -> None:
    if dataset == "demo":
        from prodrec.data.synthetic import generate
        generate(cfg)
    else:
        from prodrec.data.kaggle import convert_csv
        convert_csv(cfg)


def _feat(cfg: Config, store: MonthStore, month: str, customers=None) -> pd.DataFrame:
    f = cfg["features"]
    return build_features(store, month, customers, f["lags"], f["history_months"])


def training_rows(cfg: Config, store: MonthStore, months: list[str]):
    """One row per (customer, month, added product). Returns X, y, month label per row,
    and the 0/1 added matrix per (customer, month) for the popularity baselines."""
    Xs, ys, ms, added_all, feats_all = [], [], [], [], []
    for m in months:
        added = added_matrix(store, m)
        added = added[added.sum(axis=1) > 0]
        feats = _feat(cfg, store, m, added.index)
        rows, cls = np.nonzero(added.to_numpy())
        Xs.append(feats.iloc[rows])
        ys.append(cls)
        ms.append(np.full(len(rows), m))
        added_all.append(added.assign(month=m))
        feats_all.append(feats)
        print(f"  {m}: {len(added):,} customers added {len(rows):,} products")
    return (pd.concat(Xs), np.concatenate(ys), np.concatenate(ms),
            pd.concat(added_all), pd.concat(feats_all))


def train(cfg: Config, dataset: str, train_months: list[str] | None = None, save: bool = True):
    store = MonthStore(cfg, dataset)
    months = train_months or cfg["split"]["train_months"]
    print(f"[train] building rows for {len(months)} months")
    X, y, m, added, feats = training_rows(cfg, store, months)
    params = cfg["model"]
    if len(months) > 1:
        # early stopping on the latest training month, never on the evaluation month
        val = m == months[-1]
    else:
        rng = np.random.default_rng(params["seed"])
        cust = X.index.unique()
        held = set(rng.choice(cust, size=max(1, len(cust) // 5), replace=False))
        val = X.index.isin(held)
    t0 = time.time()
    model = LGBMRecommender(params).fit(X[~val], y[~val], X[val], y[val])
    print(f"[train] {len(X):,} rows, {X.shape[1]} features, {model.best_iteration} trees, "
          f"{time.time() - t0:.0f}s")
    if save:
        folder = cfg.path("models") / dataset / "lgbm"
        model.save(folder)
        added.to_parquet(folder / "train_added.parquet")
        feats[["segmento", "age"]].reset_index().to_parquet(folder / "train_profile.parquet")
        print(f"[train] saved to {folder}")
    return model, added, feats


def score_all(cfg: Config, store: MonthStore, month: str, model: LGBMRecommender,
              train_added: pd.DataFrame, train_feats: pd.DataFrame, customers: pd.Index):
    feats = _feat(cfg, store, month, customers)
    owned = owned_last_month(store, month, customers)
    k = cfg["evaluation"]["k"]
    scores = {
        "popularity": np.tile(popularity_scores(train_added), (len(customers), 1)),
        "segment_popularity": segment_popularity_scores(train_added, train_feats, feats),
        "lightgbm": model.predict_scores(feats),
    }
    ranked = {name: rank_from_scores(s, owned, customers, k) for name, s in scores.items()}
    return feats, ranked, scores


def evaluate(cfg: Config, dataset: str, model=None, train_added=None, train_feats=None,
             write: bool = True) -> dict:
    store = MonthStore(cfg, dataset)
    month = cfg["split"]["valid_month"]
    k = cfg["evaluation"]["k"]
    if model is None:
        folder = cfg.path("models") / dataset / "lgbm"
        model = LGBMRecommender.load(folder)
        train_added = pd.read_parquet(folder / "train_added.parquet")
        train_feats = pd.read_parquet(folder / "train_profile.parquet").set_index("ncodpers")
    actual = added_lists(store, month)
    n_all = len(store.load(month))
    feats, ranked, _ = score_all(cfg, store, month, model, train_added, train_feats, actual.index)

    rows = []
    for name, r in ranked.items():
        rep = ranking_report(actual, r, k)
        rep[f"kaggle_style_map@{k}"] = rep[f"map@{k}"] * len(actual) / n_all
        rows.append({"model": name, **rep})
    table = pd.DataFrame(rows).set_index("model")

    # seniority in months; customers with unknown seniority count as established
    new = feats["antiguedad"].fillna(999) < cfg["evaluation"]["new_customer_months"]
    cold = pd.DataFrame([
        {"model": name, "group": g, "customers": int(mask.sum()),
         f"map@{k}": ranking_report(actual[mask], r[mask], k)[f"map@{k}"] if mask.any() else np.nan}
        for name, r in ranked.items()
        for g, mask in (("new (< %d months)" % cfg["evaluation"]["new_customer_months"], new.to_numpy()),
                        ("established", ~new.to_numpy()))
    ])
    ap = {name: np.array([apk(a, p, k) for a, p in zip(actual, r.reindex(actual.index))])
          for name, r in ranked.items()}
    boot = pd.DataFrame([{"comparison": f"lightgbm - {other}", **bootstrap_diff(ap["lightgbm"], ap[other])}
                         for other in ("popularity", "segment_popularity")]).set_index("comparison")
    reac = reactivation_table(cfg, store, month, actual, ranked)
    per_prod = per_product_recall(actual, ranked["lightgbm"], k)
    per_prod.insert(1, "name", per_prod["product"].map(PRODUCTS))
    imp = model.importance()
    result = {"month": month, "customers_in_month": n_all, "customers_who_added": len(actual),
              "table": table, "bootstrap": boot, "reactivation": reac, "cold_start": cold, "per_product": per_prod, "importance": imp}
    if write:
        _write_report(cfg, dataset, result, k)
    return result


def reactivation_table(cfg: Config, store: MonthStore, month: str, actual: pd.Series,
                       ranked: dict) -> pd.DataFrame:
    """Split additions into re-activations (owned 2..H+1 months ago) and first-time
    products, and score each model on both. Many "new" products in bank data are
    products switched off and on again; a model can look good by predicting only those."""
    months = store.months()
    i = months.index(month)
    back = [months[i - k] for k in range(2, cfg["features"]["history_months"] + 2) if i - k >= 0]
    before = np.zeros((len(actual), len(PRODUCT_COLS)), dtype=bool)
    for m in back:
        before |= store.products(m).reindex(actual.index).fillna(0).to_numpy().astype(bool)
    rows = []
    for name, r in ranked.items():
        stats = {"re-activation": [], "first time": []}
        for row, (cust, prods) in enumerate(actual.items()):
            lst = r[cust]
            for p in prods:
                kind = "re-activation" if before[row, P_INDEX[p]] else "first time"
                pos = lst.index(p) + 1 if p in lst else None
                stats[kind].append(pos)
        for kind, pos in stats.items():
            if pos:
                rows.append({"model": name, "addition": kind, "count": len(pos),
                             "top1": float(np.mean([q == 1 for q in pos])),
                             "mrr": float(np.mean([1 / q if q else 0.0 for q in pos])),
                             f"recall@{cfg['evaluation']['k']}": float(np.mean([q is not None for q in pos]))})
    return pd.DataFrame(rows).set_index(["model", "addition"])


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def _write_report(cfg, dataset, res, k) -> None:
    out = cfg.path("reports") / dataset
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "models.csv")
    res["cold_start"].to_csv(out / "cold_start.csv", index=False)
    res["per_product"].to_csv(out / "per_product.csv", index=False)
    res["importance"].to_csv(out / "importance.csv", index=False)
    res["bootstrap"].to_csv(out / "bootstrap.csv")
    res["reactivation"].to_csv(out / "reactivation.csv")
    lines = [f"# Evaluation ({dataset})", ""]
    if dataset == "demo":
        lines += ["Synthetic data. These numbers check that the code works. They are not findings.", ""]
    lines += [f"Evaluation month: {res['month']}. {res['customers_who_added']:,} of "
              f"{res['customers_in_month']:,} customers added at least one product "
              f"({res['customers_who_added'] / res['customers_in_month']:.2%}).", "",
              "## Models (customers who added something)", _md(res["table"].round(4)), "",
              f"`kaggle_style_map@{k}` averages over all customers, counting those who added nothing as 0, "
              "like the competition leaderboard.", "",
              f"## Is the difference real? (paired bootstrap, 95 % interval of the MAP@{k} difference)",
              _md(res["bootstrap"].round(4)), "",
              "## Re-activations vs first-time products",
              "A re-activation is a product the customer owned at some point 2 to "
              f"{cfg['features']['history_months'] + 1} months before. MRR = mean of 1 / rank of the added product.", "",
              _md(res["reactivation"].round(3)), "",
              "## New vs established customers", _md(res["cold_start"].set_index(["model", "group"]).round(4)), "",
              "## LightGBM recall by product", _md(res["per_product"].round(3).set_index("product")), "",
              "## Top 15 features (gain share)", _md(res["importance"].head(15).set_index("feature").round(4)), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"[evaluate] full report: {out / 'report.md'}")


def experiments(cfg: Config, dataset: str) -> pd.DataFrame:
    """Same model, different training months, same evaluation month."""
    store = MonthStore(cfg, dataset)
    available = set(store.months())
    k = cfg["evaluation"]["k"]
    rows = []
    for name, months in cfg["experiments"].items():
        if not set(months) <= available:
            print(f"[experiments] skip {name}: months missing")
            continue
        print(f"[experiments] {name}: {months}")
        model, added, feats = train(cfg, dataset, months, save=False)
        res = evaluate(cfg, dataset, model, added, feats, write=False)
        t = res["table"]
        b = res["bootstrap"].loc["lightgbm - popularity"]
        rows.append({"experiment": name, "train_months": ", ".join(months),
                     f"lightgbm_map@{k}": t.loc["lightgbm", f"map@{k}"],
                     f"popularity_map@{k}": t.loc["popularity", f"map@{k}"],
                     "gain_vs_popularity": b["diff"],
                     "gain_95ci": f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"})
    df = pd.DataFrame(rows).set_index("experiment")
    out = cfg.path("reports") / dataset
    df.to_csv(out / "experiments.csv")
    print(_md(df.round(4)))
    return df


def explain_customer(cfg: Config, store: MonthStore, model: LGBMRecommender, month: str,
                     customer: int, top_products: int = 7, top_features: int = 5) -> dict:
    """Ranked products for one customer, each with the features that pushed it up most."""
    idx = pd.Index([customer], name="ncodpers")
    feats = _feat(cfg, store, month, idx)
    owned = owned_last_month(store, month, idx)[0]
    probs = model.predict_scores(feats)[0]
    contrib = model.contributions(feats)[0]  # products x (features + 1)
    order = [i for i in np.argsort(-probs) if not owned[i]][:top_products]
    recs = []
    for i in order:
        c = contrib[i, :-1]
        top = np.argsort(-c)[:top_features]
        recs.append({"product": PRODUCT_COLS[i], "name": PRODUCTS[PRODUCT_COLS[i]], "probability": float(probs[i]),
                     "reasons": [(model.features[j], _fmt(feats.iloc[0, j]), float(c[j])) for j in top if c[j] > 0]})
    return {"features": feats.iloc[0], "owned": [PRODUCT_COLS[i] for i in np.nonzero(owned)[0]], "recs": recs}


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return str(v)
