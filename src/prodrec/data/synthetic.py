"""Synthetic bank customers with the same columns as the Kaggle data.

Used for the demo, the tests and the hosted app, so the Kaggle competition data never
has to leave your machine. The behaviour is invented but has the kind of structure the
real data has, so the models have something to find:
    - most customers add nothing in a given month (about 3 to 4 % add something)
    - a payroll account makes payroll and pension deposits likely next
    - richer and older customers add investment products
    - the "Taxes" product spikes in June (end of the Spanish tax year in the real data)
    - inactive customers rarely add anything

This data is fake. Never report results from it as findings about banking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prodrec.config import Config
from prodrec.data.store import MonthStore
from prodrec.schema import CUSTOMER, PRODUCT_COLS, SHORT

MONTHS = [d.strftime("%Y-%m-28") for d in pd.date_range("2015-01-01", "2016-05-01", freq="MS")]
P = {SHORT[c]: i for i, c in enumerate(PRODUCT_COLS)}

INITIAL = {"cco": 0.62, "recibo": 0.12, "ctop": 0.12, "ecue": 0.08, "cno": 0.08, "nomina": 0.05,
           "nom_pens": 0.06, "tjcr": 0.045, "dela": 0.045, "ctpp": 0.04, "reca": 0.05, "valo": 0.025,
           "ctma": 0.01, "fond": 0.018, "plan": 0.009, "pres": 0.003, "hip": 0.006, "viv": 0.004,
           "deco": 0.002, "deme": 0.002, "ctju": 0.0, "cder": 0.0005, "aval": 0.0002, "ahor": 0.0002}
ADD = {"cco": 0.006, "recibo": 0.010, "nomina": 0.003, "nom_pens": 0.003, "tjcr": 0.004, "cno": 0.003,
       "ecue": 0.003, "dela": 0.0012, "reca": 0.0006, "ctma": 0.0010, "valo": 0.0008, "fond": 0.0007,
       "ctop": 0.0005, "ctpp": 0.0004, "plan": 0.0003, "deco": 0.0002, "deme": 0.0002, "pres": 0.0002,
       "hip": 0.0001, "viv": 0.0001, "ctju": 0.0003, "cder": 0.00005, "aval": 0.00002, "ahor": 0.00002}
DROP = {k: 0.004 for k in P} | {"recibo": 0.02, "nomina": 0.015, "nom_pens": 0.015, "reca": 0.01, "tjcr": 0.01}

PROVINCES = ["MADRID", "BARCELONA", "VALENCIA", "SEVILLA", "CORUNA, A", "MALAGA", "MURCIA", "ALICANTE",
             "ZARAGOZA", "CADIZ", "ASTURIAS", "VALLADOLID", "PALMAS, LAS", "BADAJOZ", "BIZKAIA"]
CHANNELS = ["KHE", "KAT", "KFC", "KHQ", "KFA", "KHK", "KHM", "RED"]


def _customers(n: int, rng) -> pd.DataFrame:
    age = np.clip(np.where(rng.random(n) < 0.03, rng.integers(2, 18, n),
                           rng.gamma(6.0, 7.0, n) + 15), 2, 100).round()
    renta = rng.lognormal(11.3, 0.6, n).round(2)
    renta[rng.random(n) < 0.2] = np.nan
    segment = np.where(age < 26, "03 - UNIVERSITARIO",
                       np.where(np.nan_to_num(renta) > 180000, "01 - TOP", "02 - PARTICULARES"))
    return pd.DataFrame({
        CUSTOMER: np.arange(15000, 15000 + n),
        "age": age.astype("float32"),
        "renta": renta.astype("float32"),
        "segmento": segment,
        "sexo": rng.choice(["H", "V"], n),
        "nomprov": rng.choice(PROVINCES, n, p=np.r_[0.3, 0.12, np.full(13, 0.58 / 13)]),
        "canal_entrada": np.where(age < 26, "KHE", rng.choice(CHANNELS, n)),
        "ind_actividad_cliente": np.where(rng.random(n) < np.clip(0.25 + age / 120, 0, 0.85), "1", "0"),
        "ind_empleado": np.where(rng.random(n) < 0.002, "A", "N"),
        "pais_residencia": np.where(rng.random(n) < 0.005, "FR", "ES"),
        "indrel": "1",
        "indrel_1mes": "1",
        "tiprel_1mes": "A",
        "indresi": "S",
        "indext": np.where(rng.random(n) < 0.05, "S", "N"),
        "indfall": "N",
        # month index when the customer joined; most joined before the data starts
        "join": np.where(rng.random(n) < 0.8, -rng.integers(7, 200, n), rng.integers(1, len(MONTHS), n)),
    })


def _add_multipliers(c: pd.DataFrame, owned: np.ndarray, month: str) -> np.ndarray:
    n = len(c)
    m = np.ones((n, len(P)))
    age, rich = c["age"].to_numpy(), np.nan_to_num(c["renta"].to_numpy()) > 150000
    active = c["ind_actividad_cliente"].to_numpy() == "1"
    m[~active] *= 0.1
    has = lambda k: owned[:, P[k]] == 1
    m[has("cno"), P["nomina"]] *= 10
    m[has("cno"), P["nom_pens"]] *= 10
    m[has("nomina"), P["nom_pens"]] *= 4
    m[(age > 24) & (age < 62), P["cno"]] *= 3
    m[has("cco") | has("ctop"), P["recibo"]] *= 2.5
    m[rich, P["tjcr"]] *= 3
    m[rich | (age > 50), P["valo"]] *= 4
    m[rich | (age > 50), P["fond"]] *= 4
    m[(age > 40) & (age < 66), P["plan"]] *= 4
    m[age > 55, P["dela"]] *= 3
    m[age < 45, P["ecue"]] *= 2.5
    m[age >= 20, P["ctju"]] = 0
    m[age < 20, P["ctju"]] *= 30
    m[(age > 25) & (age < 45), P["hip"]] *= 5
    if month[5:7] == "06":
        m[:, P["reca"]] *= 25
        m[:, P["cco"]] *= 2
    return m


def generate(cfg: Config) -> list[str]:
    s = cfg["synthetic"]
    rng = np.random.default_rng(s["seed"])
    c = _customers(s["customers"], rng)
    init = np.array([INITIAL[SHORT[p]] for p in PRODUCT_COLS])
    owned = (rng.random((len(c), len(P))) < init).astype("int8")
    owned[:, P["ctju"]] = (c["age"].to_numpy() < 18) & (rng.random(len(c)) < 0.6)
    add = np.array([ADD[SHORT[p]] for p in PRODUCT_COLS])
    drop = np.array([DROP[SHORT[p]] for p in PRODUCT_COLS])

    store = MonthStore(cfg, "demo")
    for t, month in enumerate(MONTHS):
        if t > 0:
            p_add = np.clip(add * _add_multipliers(c, owned, month), 0, 0.5)
            new = (rng.random(owned.shape) < p_add) & (owned == 0)
            gone = (rng.random(owned.shape) < drop) & (owned == 1)
            owned = np.where(new, 1, np.where(gone, 0, owned)).astype("int8")
        present = c["join"].to_numpy() <= t
        joined_now = c["join"].to_numpy() == t
        owned[joined_now] = 0
        owned[joined_now, P["cco"]] = 1          # new customers start with a current account
        snap = c.loc[present].drop(columns=["join"]).copy()
        months_in = t - c.loc[present, "join"].to_numpy()
        snap["antiguedad"] = np.maximum(months_in, 0).astype("float32")
        snap["ind_nuevo"] = np.where(months_in < 6, "1", "0")
        snap["age"] = (snap["age"] + t // 12).astype("float32")
        for j, col in enumerate(PRODUCT_COLS):
            snap[col] = owned[present, j]
        store.write_part(month, snap.reset_index(drop=True), 0)
    print(f"[prepare] synthetic data: {len(c):,} customers, {len(MONTHS)} months -> {store.dir}")
    return MONTHS
