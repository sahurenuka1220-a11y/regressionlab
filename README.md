# 🏠 RegressionLab

![CI](https://github.com/YOUR-USERNAME/regressionlab/actions/workflows/ci.yml/badge.svg)

Linear, polynomial and ridge regression **implemented from scratch with NumPy** — an upgrade of a
single-file C linear-regression exercise into a complete, tested, deployable project.

## What's inside

| Feature | Details |
|---|---|
| Multiple regression | Any number of features, not just one |
| Two solvers | Closed-form normal equation and batch gradient descent (with loss curve) |
| Ridge (L2) regularisation | `alpha` parameter |
| Polynomial features | `degree` parameter |
| Model selection | Train/test split, k-fold CV, grid search over alpha and degree |
| Metrics | R², RMSE, MAE |
| CLI | `regressionlab demo | train | predict` |
| Web app | Interactive Streamlit dashboard |
| Tests + CI | pytest, GitHub Actions on Python 3.10-3.12 |
| Legacy | Original C program kept in `legacy/` (bug fixed: it printed 1500 sq ft but predicted for 3000) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

regressionlab demo                                   # compare solvers
regressionlab train --degree 2 --alpha 0.1           # saves model.json
regressionlab predict --values 1500 3 10 5           # sqft, bedrooms, age, km to centre
streamlit run app.py                                 # web dashboard
pytest                                               # run tests
```

Train on your own data: `regressionlab train --csv data.csv --target price`
(numeric columns are used as features).

## Library usage

```python
from regressionlab import LinearModel, grid_search
model = LinearModel(method="gd", alpha=0.1, degree=2).fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## How it works

Features are standardised, then the model minimises `(1/2n)·||y − Zw||² + (α/2)·||w||²`,
either by solving the normal equations or by gradient descent. Coefficients are mapped back to
original units so they remain interpretable. See `regressionlab/models.py`.

## Deploy

See [DEPLOY.md](DEPLOY.md) for GitHub, Streamlit Community Cloud and Docker steps.

## License

MIT
