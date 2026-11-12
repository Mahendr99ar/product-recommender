"""Labels, leakage and Kaggle CSV cleaning."""

import numpy as np
import pandas as pd

from prodrec.data.kaggle import clean_chunk
from prodrec.data.store import MonthStore
from prodrec.features import build_features
from prodrec.labels import added_lists, added_matrix
from prodrec.schema import CATEGORICAL, CUSTOMER, DATE, NUMERIC, PRODUCT_COLS


def test_added_means_owned_now_and_not_last_month(demo):
    store = MonthStore(demo, "demo")
    month = demo["split"]["valid_month"]
    now, before = store.products(month), store.products(store.previous(month))
    added = added_matrix(store, month)
    common = now.index.intersection(before.index)
    expected = ((now.loc[common] == 1) & (before.loc[common] == 0)).astype("int8")
    pd.testing.assert_frame_equal(added.loc[common], expected)
    # customers who did not exist last month: everything they own counts as added
    new = now.index.difference(before.index)
    if len(new):
        pd.testing.assert_frame_equal(added.loc[new], now.loc[new].astype("int8"))
    lists = added_lists(store, month)
    assert all(len(v) > 0 for v in lists)


def test_features_never_use_the_target_month_or_later(demo):
    """Change product ownership in the target month and after it: features must not move.
    Change it one month earlier: features must move. If this fails, every score is suspect."""
    store = MonthStore(demo, "demo")
    months = store.months()
    target = months[-2]
    base = build_features(store, target)

    for m in months[months.index(target):]:
        df = store.load(m).copy()
        df[PRODUCT_COLS] = 1 - df[PRODUCT_COLS]
        store._cache[m] = df
    after = build_features(store, target)
    pd.testing.assert_frame_equal(base, after)

    prev = months[months.index(target) - 1]
    df = store.load(prev).copy()
    df[PRODUCT_COLS] = 1 - df[PRODUCT_COLS]
    store._cache[prev] = df
    assert not build_features(store, target).equals(base)


def test_clean_chunk_handles_kaggle_quirks():
    raw = pd.DataFrame({
        DATE: ["2015-01-28", "2015-01-28"], CUSTOMER: ["1375586", " 1050611"],
        "age": [" 35", "NA"], "antiguedad": ["     6", "-999999"], "renta": ["87218.1", ""],
        **{c: ["1.0", "1"] for c in CATEGORICAL},
        **{p: ["0", "1"] for p in PRODUCT_COLS},
    })
    raw["ind_nomina_ult1"] = ["", "1"]
    out = clean_chunk(raw)
    assert out["age"].tolist()[0] == 35 and np.isnan(out["age"].tolist()[1])
    assert np.isnan(out["antiguedad"].iloc[1])
    assert np.isnan(out["renta"].iloc[1])
    assert out["ind_nuevo"].tolist() == ["1", "1"]          # "1.0" and "1" become the same category
    assert out["ind_nomina_ult1"].tolist() == [0, 1]       # missing product = not owned
    assert out[list(NUMERIC)].dtypes.eq("float32").all()


def test_convert_kaggle_style_csv(demo, tmp_path):
    """Write demo data as a Kaggle-format CSV (all 48 columns, strings, padding) and convert it."""
    import copy

    from prodrec.config import Config
    from prodrec.data.kaggle import convert_csv

    store = MonthStore(demo, "demo")
    frames = []
    for m in store.months()[:3]:
        df = store.load(m).reset_index()
        df.insert(0, DATE, m)
        frames.append(df)
    csv = pd.concat(frames)
    csv["age"] = csv["age"].map(lambda v: f"{v:5.0f}")             # Kaggle pads numbers with spaces
    csv["antiguedad"] = csv["antiguedad"].map(lambda v: f"{v:6.0f}")
    for extra in ("fecha_alta", "ult_fec_cli_1t", "conyuemp", "tipodom", "cod_prov"):
        csv[extra] = ""
    path = tmp_path / "train_ver2.csv"
    csv.to_csv(path, index=False)

    raw = copy.deepcopy(demo.raw)
    raw["data"]["chunk_rows"] = 5000                                 # several chunks per month
    cfg2 = Config(raw, root=tmp_path)
    months = convert_csv(cfg2, path)
    assert months == store.months()[:3]
    real = MonthStore(cfg2, "real")
    orig = store.load(months[1])
    got = real.load(months[1])
    assert len(got) == len(orig)
    pd.testing.assert_frame_equal(got[PRODUCT_COLS], orig[PRODUCT_COLS].astype("int8"), check_names=False)
    assert np.allclose(got["age"], orig["age"])
