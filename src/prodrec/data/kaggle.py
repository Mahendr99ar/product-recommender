"""Convert Kaggle's train_ver2.csv (about 2.1 GB, 13.6 million rows) into monthly parquet.

Read in chunks so it fits in laptop memory, clean the messy columns once, and never
touch the CSV again. Known problems in the raw file:
    age and antiguedad are strings padded with spaces ("  35")
    antiguedad uses -999999 for unknown
    renta (income) is missing for about a fifth of rows
    ind_nomina_ult1 and ind_nom_pens_ult1 are missing in a few rows
    categorical columns mix numbers and strings ("1", "1.0", 1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prodrec.config import Config
from prodrec.data.store import MonthStore
from prodrec.schema import CATEGORICAL, CUSTOMER, DATE, NUMERIC, PRODUCT_COLS

USECOLS = [DATE, CUSTOMER, *NUMERIC, *CATEGORICAL, *PRODUCT_COLS]


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({DATE: df[DATE].astype(str).str.strip(),
                        CUSTOMER: pd.to_numeric(df[CUSTOMER], errors="coerce").astype("Int64")})
    for col in NUMERIC:
        out[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce").astype("float32")
    out.loc[out["antiguedad"] < 0, "antiguedad"] = np.nan
    out.loc[(out["age"] < 0) | (out["age"] > 120), "age"] = np.nan
    for col in CATEGORICAL:
        s = df[col].astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)          # "1.0" and "1" are the same value
        out[col] = s.replace({"nan": None, "": None, "NA": None})
    for col in PRODUCT_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
    out = out.dropna(subset=[CUSTOMER])
    out[CUSTOMER] = out[CUSTOMER].astype("int64")
    return out


def convert_csv(cfg: Config, csv_path: Path | None = None) -> list[str]:
    csv_path = Path(csv_path) if csv_path else cfg.path("raw") / cfg["data"]["csv_name"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download train_ver2.csv.zip from the Kaggle competition "
            "'santander-product-recommendation' (Data tab), unzip it into data/raw/.")
    store = MonthStore(cfg, "real")
    reader = pd.read_csv(csv_path, usecols=USECOLS, dtype=str, chunksize=cfg["data"]["chunk_rows"])
    n_rows = 0
    for i, chunk in enumerate(reader):
        clean = clean_chunk(chunk)
        for month, part in clean.groupby(DATE):
            store.write_part(month, part.drop(columns=[DATE]), i)
        n_rows += len(chunk)
        print(f"  chunk {i + 1}: {n_rows:,} rows read")
    months = store.months()
    print(f"[prepare] {n_rows:,} rows -> {len(months)} months ({months[0]} to {months[-1]}) in {store.dir}")
    return months
