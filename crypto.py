import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="CryptoSignal AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# DESIGN SYSTEM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: #04080f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,38,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,210,180,0.12) 0%, transparent 55%);
}

/* ── HERO ── */
.hero {
    padding: 52px 0 44px;
    position: relative;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,38,255,0.12);
    border: 1px solid rgba(99,38,255,0.35);
    border-radius: 100px;
    padding: 5px 16px 5px 10px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #a78bfa; margin-bottom: 22px;
}
.hero-tag-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #a78bfa;
    box-shadow: 0 0 8px #a78bfa;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.6rem, 5.5vw, 4.8rem);
    line-height: 1.0; letter-spacing: -0.02em;
    margin: 0 0 6px;
}
.hero-title-line1 { display: block;
    background: linear-gradient(90deg, #7c3aed 0%, #06b6d4 40%, #10b981 70%, #f59e0b 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: titleflow 8s linear infinite; }
.hero-title-line2 {
    display: block;
    background: linear-gradient(90deg, #7c3aed 0%, #06b6d4 40%, #10b981 70%, #f59e0b 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: titleflow 8s linear infinite;
}
@keyframes titleflow {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub {
    font-size: 1.0rem; color: #475569; margin-top: 16px;
    font-weight: 400; max-width: 540px;
}
.hero-sub b { color: #64748b; font-weight: 600; }
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(99,38,255,0.5) 0%, rgba(6,182,212,0.3) 50%, transparent 100%);
    margin: 32px 0;
}

/* ── STAT PILL ROW ── */
.stat-row { display: flex; gap: 12px; margin: 20px 0 32px; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px 20px;
    flex: 1; min-width: 120px;
    transition: border-color 0.2s, transform 0.15s;
}
.stat-pill:hover { border-color: rgba(99,38,255,0.4); transform: translateY(-1px); }
.stat-pill-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem; font-weight: 600; color: #f8fafc;
    line-height: 1;
}
.stat-pill-lbl { font-size: 0.68rem; color: #475569; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 5px; }

/* ── SECTION HEADERS ── */
.sec-wrap { margin: 44px 0 20px; }
.sec-chip {
    display: inline-block; background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.25); border-radius: 100px;
    padding: 2px 12px; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #06b6d4; margin-bottom: 8px;
}
.sec-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem; color: #f1f5f9; margin: 0;
    letter-spacing: -0.01em;
}
.sec-rule {
    height: 1px; margin: 14px 0 0;
    background: linear-gradient(90deg, rgba(6,182,212,0.3), transparent);
}

/* ── FEATURE BUTTONS ── */
.feat-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0 28px; }
.feat-btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 9px 18px; border-radius: 100px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.04em; cursor: default;
    transition: transform 0.15s, box-shadow 0.15s;
    white-space: nowrap;
}
.feat-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
.feat-btn-purple {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(99,38,255,0.1));
    border: 1px solid rgba(124,58,237,0.45); color: #a78bfa;
}
.feat-btn-cyan {
    background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(14,165,233,0.08));
    border: 1px solid rgba(6,182,212,0.4); color: #67e8f9;
}
.feat-btn-green {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7;
}
.feat-btn-orange {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(234,88,12,0.08));
    border: 1px solid rgba(245,158,11,0.35); color: #fcd34d;
}
.feat-btn-red {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.35); color: #fca5a5;
}

/* ── MODEL CARDS ── */
.model-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 22px 20px; text-align: center;
    transition: border-color 0.2s, transform 0.15s;
    height: 100%;
}
.model-card:hover { border-color: rgba(124,58,237,0.4); transform: translateY(-2px); }
.model-card.best {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.07));
    border-color: rgba(124,58,237,0.5);
    box-shadow: 0 0 30px rgba(124,58,237,0.15);
}
.mc-crown { font-size: 1.2rem; margin-bottom: 4px; }
.mc-name { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 10px; }
.mc-acc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem; font-weight: 600; line-height: 1;
}
.mc-auc { font-size: 0.72rem; color: #475569; margin-top: 6px; }

/* ── PREDICT SECTION ── */
.predict-zone {
    background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 24px; padding: 36px;
    margin: 8px 0;
}
.predict-zone-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem; color: #e2e8f0; margin-bottom: 4px;
}
.predict-zone-sub { font-size: 0.82rem; color: #475569; margin-bottom: 24px; }

/* ── PREDICT BUTTON ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #059669 0%, #0d9488 50%, #0891b2 100%);
    color: #fff; border: none; border-radius: 14px;
    padding: 20px 0; font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 8px 32px rgba(5,150,105,0.5);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(5,150,105,0.7);
    background: linear-gradient(135deg, #10b981 0%, #14b8a6 50%, #06b6d4 100%);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 16px rgba(5,150,105,0.4);
}

/* ── RESULT CARDS ── */
.res-up {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 50%, #022c22 100%);
    border: 1px solid rgba(16,185,129,0.5); border-radius: 20px;
    padding: 40px 28px; text-align: center;
    box-shadow: 0 16px 48px rgba(16,185,129,0.2);
    animation: popIn 0.45s cubic-bezier(0.34,1.56,0.64,1);
}
.res-down {
    background: linear-gradient(135deg, #1c0a0a 0%, #450a0a 50%, #1c0a0a 100%);
    border: 1px solid rgba(239,68,68,0.5); border-radius: 20px;
    padding: 40px 28px; text-align: center;
    box-shadow: 0 16px 48px rgba(239,68,68,0.2);
    animation: popIn 0.45s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes popIn {
    0%   { opacity:0; transform:scale(0.88) translateY(20px); }
    100% { opacity:1; transform:scale(1) translateY(0); }
}
.res-icon { font-size: 3.2rem; margin-bottom: 8px; }
.res-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #64748b; margin-bottom: 8px; }
.res-dir {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem; line-height: 1; margin-bottom: 10px;
}
.res-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; color: #64748b;
}
.res-conf b { color: #94a3b8; }
.res-model-tag {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px; padding: 4px 14px;
    font-size: 0.65rem; color: #475569; margin-top: 14px;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── SIDEBAR ── */
div[data-testid="stSidebar"] {
    background: #020508 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
div[data-testid="stSidebar"] * { color: #94a3b8; }
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #475569 !important; font-size: 0.75rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
}
div[data-testid="stSidebar"] h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #e2e8f0 !important; font-size: 1.3rem !important;
}
.sidebar-section {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: #7c3aed; margin: 18px 0 8px;
    padding-top: 14px; border-top: 1px solid rgba(124,58,237,0.2);
}

/* ── MISC ── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 0 !important; }
h1,h2,h3 { color: #f1f5f9 !important; }
p, li { color: #94a3b8; }
.stAlert { background: rgba(124,58,237,0.07) !important;
    border-color: rgba(124,58,237,0.25) !important; color: #94a3b8 !important; }
.stDataFrame { background: rgba(255,255,255,0.02) !important; }

/* plotly tooltips dark */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n_per = 700
    coins = {
        "BTC": ("Bitcoin",    45000, 2.0e10),
        "ETH": ("Ethereum",    2800, 1.2e10),
        "BNB": ("BNB",          350, 8.0e8),
        "SOL": ("Solana",        95, 3.0e8),
        "XRP": ("XRP",         0.55, 5.0e8),
    }
    rows = []
    for sym, (name, base, vol_m) in coins.items():
        price = base
        prices = []
        for _ in range(n_per):
            price *= np.exp(np.random.normal(0.0003, 0.025))
            prices.append(price)
        prices = np.array(prices)
        dates  = pd.date_range("2023-01-01", periods=n_per, freq="D")
        volume = vol_m * np.abs(np.random.lognormal(0, 0.5, n_per))

        sma20 = pd.Series(prices).rolling(20).mean().values
        sma50 = pd.Series(prices).rolling(50).mean().values
        delta = pd.Series(prices).diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).values
        ema12 = pd.Series(prices).ewm(span=12).mean()
        ema26 = pd.Series(prices).ewm(span=26).mean()
        macd  = (ema12 - ema26).values
        bb_m  = pd.Series(prices).rolling(20).mean()
        bb_s  = pd.Series(prices).rolling(20).std()
        bb_u  = (bb_m + 2*bb_s).values
        bb_l  = (bb_m - 2*bb_s).values
        atr   = pd.Series(prices).rolling(14).std().values
        vol30 = pd.Series(prices).pct_change().rolling(30).std().values * 100
        fg    = np.random.randint(0, 101, n_per).astype(float)
        fg_c  = pd.cut(fg, bins=[-1,24,44,54,74,100],
                       labels=["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"])

        for i in range(n_per):
            rows.append(dict(
                date=dates[i], symbol=sym, name=name,
                open=prices[i]*np.random.uniform(.99,1.01),
                high=prices[i]*np.random.uniform(1.0,1.03),
                low=prices[i]*np.random.uniform(.97,1.0),
                close=prices[i], volume=volume[i],
                market_cap=prices[i]*vol_m/50,
                sma_20=sma20[i], sma_50=sma50[i],
                rsi_14=rsi[i], macd=macd[i],
                bb_upper=bb_u[i], bb_lower=bb_l[i],
                atr=atr[i], volatility_30d=vol30[i],
                feargreed_index=fg[i], feargreed_class=str(fg_c[i]),
            ))
    df = pd.DataFrame(rows)
    df["target"] = (df.groupby("symbol")["close"].shift(-1).gt(df["close"]).astype(int))
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════
@st.cache_resource
def train_models(df):
    drop_cols = ["date", "name"]
    X = df.drop(drop_cols + ["target"], axis=1)
    y = df["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

    num_c = X.select_dtypes(include=np.number).columns.tolist()
    cat_c = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("n", num_pipe, num_c), ("c", cat_pipe, cat_c)])

    defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=120, random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=150, random_state=42),
    }

    res = {}
    for nm, mdl in defs.items():
        pipe = Pipeline([("prep", prep), ("model", mdl)])
        pipe.fit(X_tr, y_tr)
        pred  = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        rep   = classification_report(y_te, pred, output_dict=True)
        res[nm] = dict(
            pipeline=pipe, pred=pred, proba=proba,
            acc=accuracy_score(y_te, pred),
            auc=roc_auc_score(y_te, proba),
            report=rep, cm=confusion_matrix(y_te, pred),
        )

    best = max(res, key=lambda k: res[k]["auc"])
    return res, best, X_te, y_te, num_c, cat_c, X.columns.tolist()


# ══════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════
df = generate_dataset()
with st.spinner("⚡ Training 5 ML models on crypto data…"):
    results, best_name, X_test, y_test, num_cols, cat_cols, feat_cols = train_models(df)
best_pipe = results[best_name]["pipeline"]


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## ₿ CryptoSignal AI")
    st.markdown("<div style='font-size:0.75rem;color:#334155;margin-bottom:16px'>Configure your signal inputs below</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🪙 Coin Selection</div>', unsafe_allow_html=True)
    symbol = st.selectbox("Cryptocurrency", ["BTC","ETH","BNB","SOL","XRP"],
        format_func=lambda s: {"BTC":"₿ Bitcoin","ETH":"Ξ Ethereum","BNB":"◈ BNB",
                                "SOL":"◎ Solana","XRP":"✕ XRP"}[s])

    st.markdown('<div class="sidebar-section">📊 Price Data</div>', unsafe_allow_html=True)
    default_price = {"BTC":45000.0,"ETH":2800.0,"BNB":350.0,"SOL":95.0,"XRP":0.55}[symbol]
    close_p = st.number_input("Close Price ($)", value=default_price, step=default_price*0.01, format="%.4f")
    volume  = st.number_input("Volume ($)", value=2e10 if symbol=="BTC" else 1e9, step=1e8, format="%.2e")
    mktcap  = st.number_input("Market Cap ($)", value=close_p*2e7, step=1e9, format="%.2e")

    st.markdown('<div class="sidebar-section">📈 Technical Indicators</div>', unsafe_allow_html=True)
    rsi_val  = st.slider("RSI (14)", 0.0, 100.0, 52.0, 0.5)
    macd_val = st.slider("MACD", float(-close_p*0.05), float(close_p*0.05), 0.0)
    sma20    = st.number_input("SMA 20", value=round(close_p*0.98, 2))
    sma50    = st.number_input("SMA 50", value=round(close_p*0.95, 2))
    bb_upper = st.number_input("BB Upper", value=round(close_p*1.05, 2))
    bb_lower = st.number_input("BB Lower", value=round(close_p*0.95, 2))
    atr_val  = st.number_input("ATR", value=round(close_p*0.02, 2))
    vol30    = st.slider("30-Day Volatility (%)", 0.0, 30.0, 4.5, 0.1)

    st.markdown('<div class="sidebar-section">😱 Market Sentiment</div>', unsafe_allow_html=True)
    fg_idx   = st.slider("Fear & Greed Index", 0, 100, 55)
    fg_class = st.selectbox("Sentiment Class",
        ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"])

    st.markdown("---")
    predict_clicked = st.button("🔮  Predict Direction")


# ══════════════════════════════════════════
# HERO
# ══════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="hero-tag-dot"></span>Live ML Inference · 5 Models Trained</div>
    <div class="hero-title">
        <span class="hero-title-line1">CryptoSignal</span>
        <span class="hero-title-line2">Direction AI</span>
    </div>
    <p class="hero-sub">
        Predicts <b>next-day price direction</b> for BTC, ETH, BNB, SOL & XRP
        using ensemble machine learning on 3,500+ trading days of data.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# PREDICTION ZONE
# ══════════════════════════════════════════
input_df = pd.DataFrame([{
    "symbol": symbol,
    "open":  close_p * np.random.uniform(.99,1.01),
    "high":  close_p * 1.02, "low": close_p * 0.98, "close": close_p,
    "volume": volume, "market_cap": mktcap,
    "sma_20": sma20, "sma_50": sma50, "rsi_14": rsi_val, "macd": macd_val,
    "bb_upper": bb_upper, "bb_lower": bb_lower,
    "atr": atr_val, "volatility_30d": vol30,
    "feargreed_index": fg_idx, "feargreed_class": fg_class,
}])

pred_label = best_pipe.predict(input_df)[0]
pred_proba = best_pipe.predict_proba(input_df)[0]

if predict_clicked or "cpred" not in st.session_state:
    st.session_state.cpred  = pred_label
    st.session_state.cup_p  = pred_proba[1] * 100
    st.session_state.cdn_p  = pred_proba[0] * 100
    st.session_state.cconf  = pred_proba[pred_label] * 100
    st.session_state.csym   = symbol
    st.session_state.cclose = close_p
    st.session_state.crsi   = rsi_val
    st.session_state.cfg    = fg_idx
    st.session_state.cfgc   = fg_class

lbl       = st.session_state.cpred
up_p      = st.session_state.cup_p
dn_p      = st.session_state.cdn_p
confidence= st.session_state.cconf

st.markdown("""
<div class="sec-wrap">
  <div class="sec-chip">⚡ Live Signal Output</div>
  <div class="sec-heading">Prediction Result</div>
  <div class="sec-rule"></div>
</div>
""", unsafe_allow_html=True)

card_cls  = "res-up" if lbl == 1 else "res-down"
icon      = "📈" if lbl == 1 else "📉"
direction = "BULLISH · PRICE UP" if lbl == 1 else "BEARISH · PRICE DOWN"
dir_color = "#10b981" if lbl == 1 else "#ef4444"

col_l, col_c, col_r = st.columns([0.8, 2, 0.8])
with col_l:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:16px;padding:20px;text-align:center;height:100%;margin-top:4px">
        <div style="font-size:0.62rem;color:#334155;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px">Probability Split</div>
        <div style="font-size:1.6rem;font-weight:700;color:#10b981;font-family:'JetBrains Mono',monospace">{up_p:.1f}%</div>
        <div style="font-size:0.68rem;color:#475569;margin-bottom:16px">▲ UP</div>
        <div style="font-size:1.6rem;font-weight:700;color:#ef4444;font-family:'JetBrains Mono',monospace">{dn_p:.1f}%</div>
        <div style="font-size:0.68rem;color:#475569">▼ DOWN</div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="res-icon">{icon}</div>
        <div class="res-label">Tomorrow's Signal · {st.session_state.csym}</div>
        <div class="res-dir" style="color:{dir_color}">{direction}</div>
        <div class="res-conf">Confidence <b style="color:{dir_color}">{confidence:.1f}%</b>
            &nbsp;·&nbsp; Close $<b>{st.session_state.cclose:,.2f}</b>
        </div>
        <div class="res-model-tag">Model: {best_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:16px;padding:20px;text-align:center;height:100%;margin-top:4px">
        <div style="font-size:0.62rem;color:#334155;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px">Signal Inputs</div>
        <div style="font-size:0.78rem;color:#475569;line-height:1.9">
            RSI <b style="color:#94a3b8">{st.session_state.crsi:.1f}</b><br>
            F&amp;G <b style="color:#94a3b8">{st.session_state.cfg}</b><br>
            Sentiment <b style="color:#94a3b8">{st.session_state.cfgc.split()[0]}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:36px 0 20px">
    <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;
        background:linear-gradient(90deg,#7c3aed,#06b6d4,#10b981);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:6px">CryptoSignal AI</div>
    <div style="font-size:0.68rem;color:#1e293b;letter-spacing:0.08em">
        Random Forest · Gradient Boosting · Extra Trees · Decision Tree · Logistic Regression
    </div>
</div>
""", unsafe_allow_html=True)
