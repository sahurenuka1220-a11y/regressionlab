import numpy as np
import pytest

from regressionlab import LinearModel, grid_search, r2_score, train_test_split
from regressionlab.data import FEATURES, make_housing


def test_recovers_exact_line():
    x = np.arange(20, dtype=float)
    m = LinearModel().fit(x, 3 * x + 7)
    assert m.coef_[0] == pytest.approx(3, abs=1e-8)
    assert m.intercept_ == pytest.approx(7, abs=1e-6)


def test_legacy_c_dataset_matches_original_program():
    # Values printed by legacy/LinearRegression.c: y = 0.14x + 46.06, 1500 sq ft -> $263.53K
    x = [1400, 1600, 1700, 1875, 1100]
    y = [245, 312, 279, 308, 199]
    m = LinearModel().fit(x, y)
    assert m.coef_[0] == pytest.approx(0.14, abs=0.005)
    assert m.intercept_ == pytest.approx(46.06, abs=0.01)
    assert m.predict([1500])[0] == pytest.approx(263.53, abs=0.01)


def test_gd_matches_closed_form():
    df = make_housing(300)
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    a = LinearModel("closed_form").fit(X, y)
    b = LinearModel("gd").fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X), atol=0.5)
    assert b.loss_history_[-1] < b.loss_history_[0]


def test_ridge_shrinks_coefficients():
    df = make_housing(200)
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    plain = LinearModel(alpha=0).fit(X, y)
    ridge = LinearModel(alpha=5.0).fit(X, y)
    assert np.linalg.norm(ridge.coef_ * X.std(0)) < np.linalg.norm(plain.coef_ * X.std(0))


def test_polynomial_beats_linear_on_curve():
    x = np.linspace(-3, 3, 100)
    y = x ** 2
    assert LinearModel(degree=2).fit(x, y).score(x, y) > LinearModel().fit(x, y).score(x, y)


def test_serialisation_roundtrip():
    df = make_housing(100)
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    m = LinearModel(degree=2, alpha=0.1).fit(X, y)
    m2 = LinearModel.from_dict(m.to_dict())
    assert np.allclose(m.predict(X), m2.predict(X))


def test_good_generalisation_and_grid_search():
    df = make_housing(500)
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    best, results = grid_search(Xtr, ytr, alphas=(0, 0.1), degrees=(1, 2), k=3)
    m = LinearModel(alpha=best["alpha"], degree=best["degree"]).fit(Xtr, ytr)
    assert r2_score(yte, m.predict(Xte)) > 0.85
    assert len(results) == 4


def test_errors():
    with pytest.raises(RuntimeError):
        LinearModel().predict([[1.0]])
    with pytest.raises(ValueError):
        LinearModel(method="nope")
    with pytest.raises(ValueError):
        LinearModel().fit([[1], [2]], [1])
