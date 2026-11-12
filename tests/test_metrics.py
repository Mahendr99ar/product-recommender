import numpy as np
import pandas as pd
import pytest

from prodrec.metrics import apk, mapk, per_product_recall
from prodrec.models import rank_from_scores
from prodrec.schema import PRODUCT_COLS


def test_apk_hand_computed():
    assert apk(["a"], ["a", "b"]) == 1.0
    assert apk(["a"], ["b", "a"]) == 0.5
    assert apk(["a", "b"], ["a", "x", "b"]) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert apk(["a"], ["b", "c"]) == 0.0
    assert apk([], ["a"]) == 0.0
    # only the first k positions count
    assert apk(["h"], list("abcdefgh"), k=7) == 0.0


def test_apk_ignores_repeated_predictions():
    assert apk(["a", "b"], ["a", "a", "b"]) == pytest.approx((1 + 2 / 3) / 2)


def test_mapk_is_mean_of_apk():
    assert mapk([["a"], ["b"]], [["a"], ["x", "b"]]) == pytest.approx(0.75)


def test_ranking_never_recommends_owned_products():
    scores = np.tile(np.arange(24, 0, -1, dtype=float), (2, 1))  # product 0 best
    owned = np.zeros((2, 24), dtype=bool)
    owned[0, :3] = True
    ranked = rank_from_scores(scores, owned, pd.Index([1, 2]), k=7)
    assert ranked[1][0] == PRODUCT_COLS[3]
    assert not set(ranked[1]) & set(PRODUCT_COLS[:3])
    assert ranked[2][0] == PRODUCT_COLS[0]
    assert len(ranked[1]) == 7


def test_ranking_when_almost_everything_is_owned():
    owned = np.ones((1, 24), dtype=bool)
    owned[0, 5] = False
    ranked = rank_from_scores(np.random.rand(1, 24), owned, pd.Index([9]), k=7)
    assert ranked[9] == [PRODUCT_COLS[5]]


def test_per_product_recall():
    actual = pd.Series([["a"], ["a", "b"]], index=[1, 2])
    ranked = pd.Series([["a"], ["x"]], index=[1, 2])
    pp = per_product_recall(actual, ranked).set_index("product")
    assert pp.loc["a", "recall@7"] == 0.5
    assert pp.loc["b", "recall@7"] == 0.0


def test_bootstrap_interval_contains_the_difference():
    from prodrec.metrics import bootstrap_diff
    rng = np.random.default_rng(1)
    a = rng.random(2000)
    b = a - 0.05 + rng.normal(0, 0.1, 2000)   # true gap 0.05, noisy per customer
    r = bootstrap_diff(a, b)
    assert r["ci_low"] < r["diff"] < r["ci_high"]
    assert r["ci_low"] > 0 and abs(r["diff"] - 0.05) < 0.01
    same = bootstrap_diff(a, a.copy())
    assert same["ci_low"] <= 0 <= same["ci_high"]
