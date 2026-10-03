"""Multiple / polynomial / ridge linear regression, implemented from scratch.

Two solvers are provided:
  * closed_form - normal equation (solved with least squares for stability)
  * gd          - batch gradient descent, records the loss curve
Features are standardised internally; coefficients are reported in the
original feature units so they stay interpretable.
"""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import r2_score, rmse

METHODS = ("closed_form", "gd")


def _as2d(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class LinearModel:
    def __init__(self, method="closed_form", alpha=0.0, degree=1,
                 lr=0.1, n_iter=5000, tol=1e-12):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.method, self.alpha, self.degree = method, float(alpha), int(degree)
        self.lr, self.n_iter, self.tol = lr, n_iter, tol
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def _expand(self, X):
        """Per-feature polynomial expansion: x, x^2, ..., x^degree."""
        return np.hstack([X ** d for d in range(1, self.degree + 1)])

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

    def fit(self, X, y):
        X, y = _as2d(X), np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.n_features_in_ = X.shape[1]
        Z = self._expand(X)
        mu, sigma = Z.mean(axis=0), Z.std(axis=0)
        sigma[sigma == 0] = 1.0
        Zs = (Z - mu) / sigma
        y_mean = y.mean()
        yc = y - y_mean
        n, p = Zs.shape

        self.loss_history_ = []
        if self.method == "closed_form":
            # Minimises (1/2n)||y - Zw||^2 + (alpha/2)||w||^2
            A = Zs.T @ Zs / n + self.alpha * np.eye(p)
            w = np.linalg.lstsq(A, Zs.T @ yc / n, rcond=None)[0]
        else:
            w = np.zeros(p)
            prev = np.inf
            for _ in range(self.n_iter):
                resid = yc - Zs @ w
                loss = 0.5 * np.mean(resid ** 2) + 0.5 * self.alpha * (w @ w)
                self.loss_history_.append(float(loss))
                if abs(prev - loss) < self.tol:
                    break
                prev = loss
                w += self.lr * (Zs.T @ resid / n - self.alpha * w)

        # convert back to original units
        self.coef_ = w / sigma
        self.intercept_ = float(y_mean - self.coef_ @ mu)
        return self

    def predict(self, X):
        self._check_fitted()
        X = _as2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return self._expand(X) @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def to_dict(self):
        self._check_fitted()
        return {"method": self.method, "alpha": self.alpha, "degree": self.degree,
                "n_features_in": self.n_features_in_,
                "coef": self.coef_.tolist(), "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d):
        m = cls(method=d["method"], alpha=d["alpha"], degree=d["degree"])
        m.n_features_in_ = d["n_features_in"]
        m.coef_ = np.array(d["coef"])
        m.intercept_ = d["intercept"]
        return m


def train_test_split(X, y, test_size=0.2, seed=0):
    X, y = _as2d(X), np.asarray(y, float).ravel()
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def kfold_scores(make_model, X, y, k=5, seed=0):
    """Return per-fold RMSE for a model factory."""
    X, y = _as2d(X), np.asarray(y, float).ravel()
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), k)
    scores = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model = make_model().fit(X[train_idx], y[train_idx])
        scores.append(rmse(y[test_idx], model.predict(X[test_idx])))
    return scores


def grid_search(X, y, alphas=(0.0, 0.01, 0.1, 1.0), degrees=(1, 2, 3), k=5, seed=0):
    """Cross-validated search over (alpha, degree). Lowest mean RMSE wins."""
    results = []
    for alpha, degree in itertools.product(alphas, degrees):
        s = kfold_scores(lambda: LinearModel(alpha=alpha, degree=degree), X, y, k, seed)
        results.append({"alpha": alpha, "degree": degree, "cv_rmse": float(np.mean(s))})
    results.sort(key=lambda r: r["cv_rmse"])
    return results[0], results
