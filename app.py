"""Streamlit dashboard:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from regressionlab import LinearModel, grid_search, mae, r2_score, rmse, train_test_split
from regressionlab.data import FEATURES, make_housing

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#1e3a8a,#3b82f6); color:white; padding:28px 32px;
       border-radius:16px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:2.1rem; color:white;}
.hero p {margin:6px 0 0; opacity:.92; font-size:1.05rem;}
.price-card {background:#f0fdf4; border:2px solid #22c55e; border-radius:16px;
             padding:26px; text-align:center; color:#14532d;}
.price-card .big {font-size:2.8rem; font-weight:800; line-height:1.2;}
.price-card .small {font-size:1rem; color:#166534;}
.tip {background:#eff6ff; border-left:5px solid #3b82f6; padding:12px 16px;
      border-radius:8px; color:#1e3a8a; margin:8px 0;}
</style>
<div class="hero"><h1>🏠 House Price Predictor</h1>
<p>Tell us about a house and get an instant price estimate in rupees, with a simple explanation.</p></div>
""", unsafe_allow_html=True)

LABELS = {"sqft": "Area (sq ft)", "bedrooms": "Bedrooms",
          "age_years": "House age (years)", "dist_center_km": "Distance from city centre (km)"}


def inr(x):
    """Format rupees the Indian way (lakh / crore)."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} crore"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.1f} lakh"
    return f"₹{x:,.0f}"


@st.cache_resource
def train(n, noise, method, alpha, degree):
    df = make_housing(n=n, noise=noise, seed=42)
    X, y = df[FEATURES].to_numpy(), df["price_k"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    return LinearModel(method, alpha, degree).fit(Xtr, ytr), Xtr, Xte, ytr, yte


tab_home, tab_acc, tab_adv = st.tabs(["🏡 Estimate my house", "📊 How accurate is it?", "⚙️ Advanced settings"])

# Advanced tab runs first so its settings are available to the other tabs
with tab_adv:
    st.info("You can ignore this tab. The defaults work well. It is here for people who like to experiment.")
    a1, a2 = st.columns(2)
    with a1:
        rate = st.number_input("Exchange rate (₹ per $1)", 50.0, 150.0, 88.0, 0.5,
                               help="The model learns in dollars internally. Set today's rate for accurate rupee values.")
        n = st.slider("Number of example houses", 100, 2000, 500, 50)
        noise = st.slider("Randomness in prices", 0.0, 50.0, 15.0)
    with a2:
        method = st.radio("How the model learns", ["closed_form", "gd"],
                          format_func=lambda m: "Direct formula (fast)" if m == "closed_form"
                          else "Step-by-step learning (gradient descent)")
        alpha = st.select_slider("Overfitting guard (ridge)", [0.0, 0.001, 0.01, 0.1, 1.0, 10.0])
        degree = st.slider("Curve flexibility (1 = straight line)", 1, 4, 1)

model, Xtr, Xte, ytr, yte = train(n, noise, method, alpha, degree)
pred_te = model.predict(Xte)
K = 1000 * rate  # model units ($ thousands) -> rupees

with tab_adv:
    if st.button("🔍 Find the best settings automatically"):
        best, results = grid_search(Xtr, ytr)
        st.success(f"Best combination: guard={best['alpha']}, flexibility={best['degree']}")
        st.dataframe(results)
    if method == "gd":
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(model.loss_history_)
        ax.set(xlabel="Learning steps", ylabel="Error", yscale="log", title="The model's error shrinking as it learns")
        st.pyplot(fig)

with tab_home:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("1. Describe the house")
        sqft = st.slider("📐 Area (sq ft)", 300, 6000, 1500, 50)
        beds = st.slider("🛏️ Bedrooms", 1, 8, 3)
        age = st.slider("🕰️ House age (years)", 0, 100, 10)
        dist = st.slider("📍 Distance from city centre (km)", 1, 50, 5)
    x = np.array([sqft, beds, age, dist], dtype=float)
    est = model.predict([x])[0] * K
    err = rmse(yte, pred_te) * K

    with right:
        st.subheader("2. Estimated price")
        st.markdown(
            f'<div class="price-card"><div class="small">Estimated market price</div>'
            f'<div class="big">{inr(est)}</div>'
            f'<div class="small">Likely range: {inr(max(est - err, 0))} to {inr(est + err)}</div></div>',
            unsafe_allow_html=True)
        st.caption(f"Exact estimate: ₹{est:,.0f}")

        st.markdown("**What is pushing the price up or down?**")
        mean_x = np.vstack([Xtr]).mean(axis=0)
        effects = []
        for i in range(4):
            alt = x.copy()
            alt[i] = mean_x[i]
            effects.append((model.predict([x])[0] - model.predict([alt])[0]) * K)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        names = [LABELS[f] for f in FEATURES]
        ax.barh(names, [e / 1e5 for e in effects], color=["#22c55e" if e >= 0 else "#ef4444" for e in effects])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Change in price vs an average house (₹ lakh)")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        biggest = int(np.argmax(np.abs(effects)))
        word = "raises" if effects[biggest] >= 0 else "lowers"
        st.markdown(f'<div class="tip">💡 The biggest factor here is <b>{LABELS[FEATURES[biggest]].lower()}</b>. '
                    f'It {word} the price by about {inr(abs(effects[biggest]))} compared with an average house.</div>',
                    unsafe_allow_html=True)

    st.caption("⚠️ This demo learns from computer-generated example houses, not real market data. "
               "Use it to learn how price prediction works, not to decide what to pay for a home.")

with tab_acc:
    r2 = r2_score(yte, pred_te)
    st.subheader("How good are the predictions?")
    st.write("We hid some houses from the model while it learned, then asked it to guess their prices. "
             "Here is how close the guesses were to the real prices.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy score", f"{r2 * 100:.0f}%",
              help="Share of the price differences between houses that the model explains. 100% is perfect.")
    c2.metric("Typical mistake", inr(rmse(yte, pred_te) * K), help="Roughly how far off a guess usually is.")
    c3.metric("Average mistake", inr(mae(yte, pred_te) * K), help="The average gap between guess and real price.")
    st.markdown(f'<div class="tip">In plain words: the model gets prices right to within about '
                f'<b>{inr(rmse(yte, pred_te) * K)}</b> most of the time.</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(yte * K / 1e5, pred_te * K / 1e5, alpha=0.6, color="#3b82f6")
        lo, hi = min(yte.min(), pred_te.min()) * K / 1e5, max(yte.max(), pred_te.max()) * K / 1e5
        ax.plot([lo, hi], [lo, hi], "r--", label="Perfect guess")
        ax.set(xlabel="Real price (₹ lakh)", ylabel="Model's guess (₹ lakh)", title="Guess vs reality")
        ax.legend()
        st.pyplot(fig)
        st.caption("The closer the dots are to the red line, the better the model.")
    with g2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist((yte - pred_te) * K / 1e5, bins=25, color="#8b5cf6")
        ax.set(xlabel="Mistake (₹ lakh)", ylabel="Number of houses", title="How big were the mistakes?")
        st.pyplot(fig)
        st.caption("A tall bar near zero means most guesses were very close.")
