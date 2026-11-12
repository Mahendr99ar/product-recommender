"""Load configs/config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self, raw: dict[str, Any], root: Path = PROJECT_ROOT):
        self.raw = raw
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        p = self.root / self.raw["paths"][name]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
