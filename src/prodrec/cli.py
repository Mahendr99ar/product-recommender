"""Command line.

    python -m prodrec.cli all --demo        # synthetic data: prepare, train, evaluate, experiments
    python -m prodrec.cli prepare           # Kaggle CSV -> monthly parquet (several minutes)
    python -m prodrec.cli train
    python -m prodrec.cli evaluate
    python -m prodrec.cli experiments       # same model, different training months
"""

from __future__ import annotations

import argparse

from prodrec import pipeline
from prodrec.config import load_config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="prodrec")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "train", "evaluate", "experiments", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--demo", action="store_true", help="use synthetic data")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ds = "demo" if args.demo else "real"
    if args.cmd in ("prepare", "all"):
        pipeline.prepare(cfg, ds)
    if args.cmd in ("train", "all"):
        pipeline.train(cfg, ds)
    if args.cmd in ("evaluate", "all"):
        pipeline.evaluate(cfg, ds)
    if args.cmd in ("experiments", "all"):
        pipeline.experiments(cfg, ds)


if __name__ == "__main__":
    main()
