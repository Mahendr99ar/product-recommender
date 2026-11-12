"""Next-product recommender app.

    streamlit run app/streamlit_app.py

The hosted version runs on synthetic demo data: the Kaggle competition data is not
re-published here. Run locally with the real data to see real customers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from prodrec.config import load_config
from prodrec.data.store import MonthStore
from prodrec.labels import added_lists
from prodrec.models import LGBMRecommender
from prodrec.pipeline import explain_customer
from prodrec.schema import CATEGORICAL, NUMERIC, PRODUCTS, SHORT

st.set_page_config(page_title="Next-product recommender", layout="wide")
cfg = load_config()
SHORT_TO_NAME = {SHORT[c]: n for c, n in PRODUCTS.items()}
LABELS = NUMERIC | CATEGORICAL | {
    "renta_missing": "Income missing", "n_owned_lag1": "Products owned last month",
    "n_adds": "Products added, last 5 months", "n_drops": "Products dropped, last 5 months",
    "months_of_history": "Months of history",
}


def pretty(feature: str) -> str:
    if feature in LABELS:
        return LABELS[feature]
    base, _, kind = feature.rpartition("_")
    name = SHORT_TO_NAME.get(base, base)
    if kind.startswith("lag"):
        n = kind[3:]
        return f"{name} owned {n} month{'s' if n != '1' else ''} ago"
    if kind == "adds":
        return f"{name} added in last 5 months"
    if kind == "drops":
        return f"{name} dropped in last 5 months"
    return feature


available = [d for d in ("demo", "real")
             if (cfg.path("models") / d / "lgbm" / "model.txt").exists()
             and (cfg.path("processed") / d / "months").exists()]
if not available:
    st.error("No trained model found. Run: python -m prodrec.cli all --demo")
    st.stop()
dataset = st.sidebar.radio("Dataset", available,
                           help="demo = synthetic customers. real = Kaggle data, only on your own machine.")
if dataset == "demo":
    st.sidebar.caption("Synthetic customers. Nothing here describes real people or a real bank.")


@st.cache_resource
def load(ds: str):
    store = MonthStore(cfg, ds)
    model = LGBMRecommender.load(cfg.path("models") / ds / "lgbm")
    month = cfg["split"]["valid_month"]
    actual = added_lists(store, month)
    return store, model, month, actual


store, model, month, actual = load(dataset)
reports = cfg.path("reports") / dataset

st.title("Which product will this customer add next?")
st.caption(f"LightGBM ranking of 24 bank products, evaluated on {month}: customers are ranked using only "
           "what was known before that month, then compared with what they actually added.")

# ---- model comparison ----
if (reports / "models.csv").exists():
    table = pd.read_csv(reports / "models.csv", index_col=0)
    c1, c2, c3 = st.columns(3)
    for col, name in zip((c1, c2, c3), ("popularity", "segment_popularity", "lightgbm")):
        if name in table.index:
            col.metric(f"MAP@7: {name.replace('_', ' ')}", f"{table.loc[name, 'map@7']:.3f}",
                       f"{table.loc[name, 'map@7'] - table.loc['popularity', 'map@7']:+.3f} vs popularity"
                       if name != "popularity" else None)

st.divider()

# ---- one customer ----
st.subheader("One customer")
left, right = st.columns([1, 2])
with left:
    mode = st.radio("Pick from", ["customers who added something", "any customer"], horizontal=False)
    pool = actual.index if mode.startswith("customers who") else store.load(month).index
    if "cust" not in st.session_state or st.session_state.get("pool") != mode:
        st.session_state.cust, st.session_state.pool = int(pool[0]), mode
    if st.button("Random customer"):
        st.session_state.cust = int(np.random.default_rng().choice(pool))
    cust = st.number_input("Customer id", value=st.session_state.cust, step=1)
    if cust not in store.load(month).index:
        st.warning("That customer is not in this month.")
        st.stop()
    ex = explain_customer(cfg, store, model, month, int(cust))
    f = ex["features"]
    profile = {
        "Age": f["age"], "Seniority (months)": f["antiguedad"],
        "Income": "missing" if pd.isna(f["renta"]) else f"{f['renta']:,.0f}",
        "Segment": f["segmento"], "Province": f["nomprov"],
        "Active": "yes" if str(f["ind_actividad_cliente"]) == "1" else "no",
    }
    st.table(pd.DataFrame([(k, str(v)) for k, v in profile.items()], columns=["", "value"]).set_index(""))
    st.markdown("**Owned last month:** " + (", ".join(PRODUCTS[p] for p in ex["owned"]) or "nothing"))
    truth = actual.get(int(cust), [])
    st.markdown("**Actually added this month:** " + (", ".join(PRODUCTS[p] for p in truth) or "nothing"))

with right:
    rows = []
    for rank, r in enumerate(ex["recs"], 1):
        reasons = "; ".join(f"{pretty(feat)} = {val}" for feat, val, _ in r["reasons"][:3])
        rows.append({"rank": rank, "product": r["name"], "probability": r["probability"],
                     "added?": "yes" if r["product"] in truth else "", "main reasons": reasons})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.caption("Reasons are the features with the largest positive SHAP contribution to that product's "
               "score (TreeSHAP, computed by LightGBM). They explain the model, not the customer's intent.")

st.divider()

# ---- global views ----
g1, g2 = st.columns(2)
if (reports / "per_product.csv").exists():
    pp = pd.read_csv(reports / "per_product.csv")
    pp = pp[pp["additions"] >= 5]
    fig = px.bar(pp, x="recall@7", y="name", orientation="h", hover_data=["additions"],
                 title="How often each added product was in the top 7")
    fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=40, b=0))
    g1.plotly_chart(fig, width="stretch")
if (reports / "importance.csv").exists():
    imp = pd.read_csv(reports / "importance.csv").head(15)
    imp["feature"] = imp["feature"].map(pretty)
    fig = px.bar(imp, x="share", y="feature", orientation="h", title="Top 15 features (share of gain)")
    fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=40, b=0))
    g2.plotly_chart(fig, width="stretch")
if (reports / "experiments.csv").exists():
    st.subheader("Which training months help most?")
    st.dataframe(pd.read_csv(reports / "experiments.csv"), hide_index=True, width="stretch")
