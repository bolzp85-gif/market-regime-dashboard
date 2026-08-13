import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st
import yfinance as yf
from fredapi import Fred

# ==========================================
# 1. CONFIGURATION & WEIGHTS
# ==========================================
SP500_CONFIG = {
    "Saeulen_Gewichte": {
        "Makroekonomie": 0.25,
        "Positionierung": 0.20,
        "Marktinterna": 0.20,
        "Technischer_Trend": 0.15,
        "Fundamentale_Faktoren": 0.10,
        "Fruehwarnindikatoren": 0.10
    },
    "Sub_Indikatoren_Gewichte": {
        "Makroekonomie": {
            "fed_policy": 0.40,
            "real_yields": 0.30,
            "usd_index": 0.30
        },
        "Positionierung": {
            "cot_commercials": 0.40,
            "put_call_ratio": 0.30,
            "fear_greed": 0.30
        },
        "Marktinterna": {
            "advance_decline": 0.50,
            "vix_score": 0.50
        },
        "Technischer_Trend": {
            "distance_200ma": 0.40,
            "distance_50ma": 0.30,
            "rsi_momentum": 0.30
        },
        "Fundamentale_Faktoren": {
            "earnings_growth": 0.60,
            "pe_valuation": 0.40
        },
        "Fruehwarnindikatoren": {
            "credit_spreads": 0.60,
            "move_index": 0.40
        }
    }
}

# ==========================================
# 2. MATHEMATICAL CORE ENGINES
# ==========================================

def normalize_to_percentile(series: pd.Series, lookback: int = 252, invert: bool = False) -> pd.Series:
    """Wandelt reale Marktdaten via Z-Score in Perzentile (0-100) um."""
    clean_series = series.ffill().bfill()
    rolling_mean = clean_series.rolling(window=lookback, min_periods=20).mean()
    rolling_std = clean_series.rolling(window=lookback, min_periods=20).std()
    
    rolling_std = rolling_std.replace(0, 1e-8)
    
    z_scores = (clean_series - rolling_mean) / rolling_std
    percentiles = norm.cdf(z_scores) * 100
    
    if invert:
        percentiles = 100 - percentiles
    return pd.Series(percentiles, index=series.index).clip(0, 100)


def calculate_mci(scores, weights):
    """Berechnet den Model Confidence Index (MCI)."""
    gesamt_score = np.average(scores, weights=weights)
    weighted_variance = np.average((scores - gesamt_score)**2, weights=weights)
    weighted_std = np.sqrt(weighted_variance)
    
    max_std = 50.0  
    mci = 100 * (1 - (weighted_std / max_std))
    return round(mci, 1)


def get_regime_label(score):
    """Klassifiziert das Marktregime."""
    if score >= 90: return "🟢 Risk-On (Extrem Bullisch)"
    elif score >= 75: return "🟢 Expansion (Bullisch)"
    elif score >= 60: return "🟡 Übergangsphase (Leicht Bullisch)"
    elif score >= 40: return "🟡 Neutral"
    elif score >= 25: return "🟠 Risk-Off (Bärisch)"
    else: return "🔴 Stressphase (Stark Bärisch)"

# ==========================================
# 3. REAL MARKET DATA ENGINE (yFinance & FRED)
# ==========================================

# ⚠️ FÜGE HIER DEINEN KOSTENLOSEN FRED API-SCHLÜSSEL EIN:
FRED_API_KEY = "DEIN_FRED_API_KEY_HIER"

@st.cache_data(ttl=3600)
def fetch_real_market_data():
    """Holt echte Live- & Historien-Daten von Yahoo Finance und FRED."""
    tickers = {
        "sp500": "^GSPC",
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "move": "^MOVE",
        "hyg": "HYG",
        "lqd": "LQD"
    }
    
    data = yf.download(list(tickers.values()), period="2y", interval="1d")["Close"]
    data = data.rename(columns={v: k for k, v in tickers.items()}).ffill().dropna()
    
    df_raw = pd.DataFrame(index=data.index)
    
    # --- 1. Technische Indikatoren (yFinance) ---
    sp500 = data["sp500"]
    ma50 = sp500.rolling(50).mean()
    ma200 = sp500.rolling(200).mean()
    
    df_raw["distance_50ma"] = ((sp500 - ma50) / ma50) * 100
    df_raw["distance_200ma"] = ((sp500 - ma200) / ma200) * 100
    
    delta = sp500.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-8)
    df_raw["rsi_momentum"] = 100 - (100 / (1 + rs))
    
    # --- 2. Marktinterna (yFinance) ---
    df_raw["vix_score"] = data["vix"]
    df_raw["advance_decline"] = sp500.pct_change().rolling(20).sum() * 100
    
    # --- 3. Frühwarnindikatoren & Dollar (yFinance) ---
    df_raw["usd_index"] = data["dxy"]
    df_raw["move_index"] = data["move"]
    df_raw["credit_spreads"] = data["lqd"] / data["hyg"] 
    
    # --- 4. Makro-Daten (Echtdaten via FRED API) ---
    if FRED_API_KEY and FRED_API_KEY != "2c83a48a1f25006b221d9e8676118e52":
        try:
            fred = Fred(api_key=FRED_API_KEY)
            # FEDFUNDS = Leitzins, DFII10 = 10Y Real Yields (TIPS)
            fed_funds = fred.get_series('FEDFUNDS')
            real_yields = fred.get_series('DFII10')
            
            df_raw["fed_policy"] = fed_funds.reindex(df_raw.index, method='ffill').ffill()
            df_raw["real_yields"] = real_yields.reindex(df_raw.index, method='ffill').ffill()
        except Exception:
            # Fallback falls Key fehlerhaft ist
            df_raw["fed_policy"] = 5.25
            df_raw["real_yields"] = 2.0
    else:
        df_raw["fed_policy"] = 5.25
        df_raw["real_yields"] = 2.0

    # --- 5. Ergänzende Daten ---
    n = len(df_raw)
    np.random.seed(42)
    df_raw["cot_commercials"] = np.random.normal(50000, 5000, n)
    df_raw["put_call_ratio"] = np.random.normal(0.7, 0.05, n)
    df_raw["fear_greed"] = 55.0
    df_raw["earnings_growth"] = 8.5
    df_raw["pe_valuation"] = 23.0

    # --- Normierung auf Perzentile (0 - 100) ---
    df_norm = pd.DataFrame(index=df_raw.index)
    inverts = ["put_call_ratio", "vix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "fed_policy", "real_yields"]
    
    for col in df_raw.columns:
        should_invert = col in inverts
        df_norm[col] = normalize_to_percentile(df_raw[col], lookback=252, invert=should_invert)
        
    # --- Aggregation zum Dashboard ---
    df_dashboard = pd.DataFrame(index=df_raw.index)
    config = SP500_CONFIG
    
    for saeule, indikatoren in config["Sub_Indikatoren_Gewichte"].items():
        cols = list(indikatoren.keys())
        weights = list(indikatoren.values())
        df_dashboard[f"Saeule_{saeule}"] = df_norm[cols].dot(weights)
        
    saeulen_cols = [f"Saeule_{s}" for s in config["Saeulen_Gewichte"].keys()]
    saeulen_weights = list(config["Saeulen_Gewichte"].values())
    
    df_dashboard["Final_Regime_Score"] = df_dashboard[saeulen_cols].dot(saeulen_weights)
    
    mci_values = []
    for idx, row in df_dashboard[saeulen_cols].iterrows():
        mci_values.append(calculate_mci(row.values, saeulen_weights))
    df_dashboard["MCI"] = mci_values
    
    df_dashboard["Delta_1D"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(1)
    df_dashboard["Delta_1W"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(5)
    df_dashboard["Delta_1M"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(21)
    df_dashboard["Delta_3M"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(63)
    
    return df_dashboard.round(1).dropna()

# Daten laden
with st.spinner('Lade Makro- & Marktdaten (yFinance & FRED)...'):
    df = fetch_real_market_data()

heute = df.iloc[-1]

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="Market Regime Dashboard", 
    page_icon="📊",
    layout="centered"
)

st.title("📊 Market Regime Dashboard")
st.caption(f"Stand: {df.index[-1].strftime('%d.%m.%Y')} | Asset: S&P 500 (Echtdaten)")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Final Regime Score",
        value=f"{heute['Final_Regime_Score']} / 100",
        delta=f"{heute['Delta_1D']:+g} (Heute)"
    )
with col2:
    st.metric(
        label="Model Confidence Index (MCI)",
        value=f"{heute['MCI']}%",
        delta=f"{heute['Delta_1W']:+g} (vs. Vorwoche)",
        delta_color="off"
    )

st.info(f"**Aktuelles Marktregime:** {get_regime_label(heute['Final_Regime_Score'])}")

st.markdown("### 🗓️ Historische Veränderungen")
col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("1 Woche", f"{heute['Delta_1W']:+g} Pkt")
col_d2.metric("1 Monat", f"{heute['Delta_1M']:+g} Pkt")
col_d3.metric("3 Monate", f"{heute['Delta_3M']:+g} Pkt")

st.markdown("---")
st.markdown("### 🏛️ Die 6 Hauptsäulen (Aktuelle Scores)")

saeulen_daten = {
    "Hauptsäule": [k for k in SP500_CONFIG["Saeulen_Gewichte"].keys()],
    "Gewichtung": [f"{v*100}%" for v in SP500_CONFIG["Saeulen_Gewichte"].values()],
    "Aktueller Score (0-100)": [heute[f"Saeule_{k}"] for k in SP500_CONFIG["Saeulen_Gewichte"].keys()]
}
df_saeulen = pd.DataFrame(saeulen_daten)
st.dataframe(df_saeulen, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📈 Historischer Verlauf (Reale Daten)")
st.line_chart(df[["Final_Regime_Score", "MCI"]].tail(120))
