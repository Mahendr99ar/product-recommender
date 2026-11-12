"""Train, evaluate and explain on small synthetic data."""

import numpy as np

from prodrec import pipeline
from prodrec.data.store import MonthStore
from prodrec.models import LGBMRecommender


def test_train_evaluate_explain(demo, tmp_path):
    model, added, feats = pipeline.train(demo, "demo")
    res = pipeline.evaluate(demo, "demo")
    t = res["table"]
    assert set(t.index) == {"popularity", "segment_popularity", "lightgbm"}
    assert t["map@7"].between(0, 1).all()
    # the model must at least match the baselines it is supposed to beat
    assert t.loc["lightgbm", "map@7"] >= 0.95 * t.loc["popularity", "map@7"]
    assert (demo.path("reports") / "demo" / "report.md").exists()

    # saved model reloads and predicts the same scores
    store = MonthStore(demo, "demo")
    month = demo["split"]["valid_month"]
    loaded = LGBMRecommender.load(demo.path("models") / "demo" / "lgbm")
    X = pipeline._feat(demo, store, month, store.load(month).index[:50])
    np.testing.assert_allclose(model.predict_scores(X), loaded.predict_scores(X), rtol=1e-6)

    # SHAP contributions add up to the raw class score
    contrib = loaded.contributions(X)
    raw = loaded.booster.predict(loaded.encoder.transform(X[loaded.features]), raw_score=True)
    np.testing.assert_allclose(contrib.sum(axis=2), raw, rtol=1e-4, atol=1e-4)

    cust = int(store.load(month).index[0])
    ex = pipeline.explain_customer(demo, store, loaded, month, cust)
    assert len(ex["recs"]) <= 7
    assert not {r["product"] for r in ex["recs"]} & set(ex["owned"])
