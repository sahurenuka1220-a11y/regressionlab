import numpy as np
import pandas as pd

FEATURES = ["sqft", "bedrooms", "age_years", "dist_center_km"]


def make_housing(n=500, noise=15.0, seed=42):
    """Synthetic housing data. Price is in $1000s and mildly non-linear."""
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(600, 3500, n)
    bedrooms = np.clip(np.round(sqft / 900 + rng.normal(0, 0.6, n)), 1, 6)
    age = rng.uniform(0, 60, n)
    dist = rng.uniform(1, 30, n)
    price = (40 + 0.12 * sqft + 9 * bedrooms - 0.8 * age
             - 14 * np.sqrt(dist) + rng.normal(0, noise, n))
    return pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "age_years": age,
                         "dist_center_km": dist, "price_k": price})


def load_csv(path, target):
    df = pd.read_csv(path).dropna()
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    feats = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    return df[feats].to_numpy(float), df[target].to_numpy(float), feats
