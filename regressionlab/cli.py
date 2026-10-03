import argparse
import json

from .data import FEATURES, load_csv, make_housing
from .metrics import mae, r2_score, rmse
from .models import LinearModel, grid_search, train_test_split


def _report(name, model, Xte, yte):
    p = model.predict(Xte)
    print(f"{name:<28} R2={r2_score(yte, p):.4f}  RMSE={rmse(yte, p):7.2f}  MAE={mae(yte, p):7.2f}")


def cmd_demo(_):
    df = make_housing()
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    print("Held-out test performance (synthetic housing data)\n")
    _report("Closed form (OLS)", LinearModel().fit(Xtr, ytr), Xte, yte)
    gd = LinearModel(method="gd").fit(Xtr, ytr)
    _report(f"Gradient descent ({len(gd.loss_history_)} it)", gd, Xte, yte)
    best, _all = grid_search(Xtr, ytr)
    print(f"\nCV best: alpha={best['alpha']}, degree={best['degree']} "
          f"(CV RMSE {best['cv_rmse']:.2f})")
    tuned = LinearModel(alpha=best["alpha"], degree=best["degree"]).fit(Xtr, ytr)
    _report("Tuned model", tuned, Xte, yte)


def cmd_train(a):
    if a.csv:
        X, y, feats = load_csv(a.csv, a.target)
    else:
        df = make_housing()
        X, y, feats = df[FEATURES].to_numpy(), df["price_k"].to_numpy(), FEATURES
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    model = LinearModel(a.method, a.alpha, a.degree).fit(Xtr, ytr)
    _report("Trained model", model, Xte, yte)
    with open(a.out, "w") as f:
        json.dump({"features": feats, "model": model.to_dict()}, f, indent=2)
    print(f"Saved model to {a.out}")


def cmd_predict(a):
    with open(a.model) as f:
        blob = json.load(f)
    model = LinearModel.from_dict(blob["model"])
    if len(a.values) != len(blob["features"]):
        raise SystemExit(f"Expected {len(blob['features'])} values: {blob['features']}")
    print(f"Prediction: {model.predict([a.values])[0]:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="regressionlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="compare solvers on synthetic data").set_defaults(fn=cmd_demo)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--csv")
    t.add_argument("--target", default="price_k")
    t.add_argument("--method", choices=["closed_form", "gd"], default="closed_form")
    t.add_argument("--alpha", type=float, default=0.0)
    t.add_argument("--degree", type=int, default=1)
    t.add_argument("--out", default="model.json")
    t.set_defaults(fn=cmd_train)

    q = sub.add_parser("predict", help="predict with a saved model")
    q.add_argument("--model", default="model.json")
    q.add_argument("--values", type=float, nargs="+", required=True)
    q.set_defaults(fn=cmd_predict)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
