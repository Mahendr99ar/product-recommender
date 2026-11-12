"""Monthly snapshots on disk: data/processed/<dataset>/months/<YYYY-MM-DD>/part-*.parquet.

One folder per month means a step that needs 6 months reads 6 folders, not the whole
2 GB file. Both the Kaggle data and the synthetic demo data use this layout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prodrec.config import Config
from prodrec.schema import CATEGORICAL, CUSTOMER, PRODUCT_COLS


class MonthStore:
    def __init__(self, cfg: Config, dataset: str):
        self.dir = cfg.path("processed") / dataset / "months"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}
        self._products: dict[str, pd.DataFrame] = {}

    def months(self) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p.is_dir() and any(p.glob("*.parquet")))

    def write_part(self, month: str, df: pd.DataFrame, part: int) -> None:
        folder = self.dir / month
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(folder / f"part-{part:04d}.parquet", index=False)

    def _read(self, month: str, columns: list[str] | None) -> pd.DataFrame:
        parts = sorted((self.dir / month).glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No data for month {month} in {self.dir}. Run `prepare` first.")
        df = pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)
        return df.drop_duplicates(CUSTOMER, keep="last").set_index(CUSTOMER).sort_index()

    def load(self, month: str) -> pd.DataFrame:
        """One month with all columns, indexed by customer id. Cached.

        Text columns become pandas categories: with about 930,000 customers a month in the
        Kaggle data, plain Python strings would need several GB for a few months."""
        if month not in self._cache:
            df = self._read(month, None)
            for c in CATEGORICAL:
                if c in df:
                    df[c] = df[c].astype("category")
            self._cache[month] = df
        return self._cache[month]

    def products(self, month: str) -> pd.DataFrame:
        """Only the 24 product columns (int8). Lag features need nothing else, so older
        months are never loaded in full."""
        if month in self._cache:
            return self._cache[month][PRODUCT_COLS]
        key = month + "#products"
        if key not in self._products:
            self._products[key] = self._read(month, [CUSTOMER, *PRODUCT_COLS]).astype("int8")
        return self._products[key]

    def previous(self, month: str, n: int = 1) -> str | None:
        ms = self.months()
        i = ms.index(month)
        return ms[i - n] if i - n >= 0 else None
