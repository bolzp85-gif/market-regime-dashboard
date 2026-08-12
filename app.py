import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

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
    """Wandelt heterogene Daten via rollierendem Z-Score in Perzentile (0-100) um."""
    clean_series = series.ffill()
    rolling_mean = clean_series.rolling(window=lookback, min_periods=int(lookback * 0.5)).mean()
    rolling_std = clean_series.rolling(window=lookback, min_periods=int(lookback * 0.5)).std()
    
    z_scores = (clean_series - rolling_mean) / (rolling_std + 1e-8)
    percentiles = norm.cdf(z_scores) * 100
    
    if invert:
        percentiles = 100 - percentiles
    return pd.Series(percentiles, index=series.index).clip(0, 100)


def calculate_mci(scores, weights):
    """Berechnet den Model Confidence Index basierend auf der gewichteten Standardabweichung."""
    gesamt_score = np.average(scores, weights=weights)
    weighted_variance = np.average((scores - gesamt_score)**2, weights=weights)
    weighted_std = np.sqrt(weighted_variance)
    
    max_std = 50.0  # Maximal mögliche Streuung bei Extremwerten 0 und 100
    mci = 100 * (1 - (weighted_std / max_std))
    return round(mci, 1)


def get_regime_label(score):
    """Klassifiziert das Marktregime anhand der Punkteskala."""
    if score >= 90: return "🟢 Risk-On (Extrem Bullisch)"
    elif score >= 75: return "🟢 Expansion (Bullisch)"
    elif score >= 60: return "🟡 Übergangsphase (Leicht Bullisch)"
    elif score >= 40: return "🟡 Neutral"
    elif score >= 25: return "🟠 Risk-Off (Bärisch)"
    else: return "🔴 Stressphase (Stark Bärisch)"

# ==========================================
# 3. DATA SIMULATION & ENGINE
# ==========================================

@st.cache_data
def generate_historical_data():
    """Generiert synthetische, mathematisch logische Finanzdaten."""
    date_index = pd.date_range(start="2024-01-01", end="2026-08-12", freq="B")
    np.random.seed(42)
    n = len(date_index)
    
    raw_data = {
        "fed_policy": np.cumsum(np.random.normal(0, 0.1, n)) + 4.5,
        "real_yields": np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n) + 1.5,
        "usd_index": np.cumsum(np.random.normal(0, 0.2, n)) + 102,
        "cot_commercials": np.random.normal(50000, 10000, n),
        "put_call_ratio": np.random.normal(0.6, 0.1, n),
        "fear_greed": np.clip(np.convolve(np.random.uniform(10, 90, n), np.ones(5)/5, mode='same'), 0, 100),
        "advance_decline": np.cumsum(np.random.normal(10, 50, n)),
        "vix_score": np.random.lognormal(mean=2.8, sigma=0.2, size=n),
        "distance_200ma": np.random.normal(5, 3, n),
        "distance_50ma": np.random.normal(2, 1.5, n),
        "rsi_momentum": np.clip(np.random.normal(55, 10, n), 0, 100),
        "earnings_growth": np.random.normal(8, 2, n),
        "pe_valuation": np.random.normal(22, 2, n),
        "credit_spreads": np.random.lognormal(mean=0.2, sigma=0.1, size=n),
        "move_index": np.random.normal(100, 15, n)
    }
    
    df_raw = pd.DataFrame(raw_data, index=date_index)
    df_norm = pd.DataFrame(index=date_index)
    
    inverts = ["put_call_ratio", "vix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "real_yields"]
    for col in df_raw.columns:
        should_invert = col in inverts
        df_norm[col] = normalize_to_percentile(df_raw[col], lookback=252, invert=should_invert)
        
    df_dashboard = pd.DataFrame(index=date_index)
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
df = generate_historical_data()
heute = df.iloc[-1]

# ==========================================
# 4. STREAMLIT UI (DESKTOP & MOBILE RESPONSIVE)
# ==========================================
st.set_page_config(
    page_title="Market Regime Dashboard", 
    page_icon="📊",
    layout="centered"
)

st.title("📊 Market Regime Dashboard")
st.caption(f"Stand: {df.index[-1].strftime('%d.%m.%Y')} | Asset: S&P 500")
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
st.markdown("### 📈 Historischer Verlauf (Score vs. Konfidenz)")
st.line_chart(df[["Final_Regime_Score", "MCI"]].tail(120))
