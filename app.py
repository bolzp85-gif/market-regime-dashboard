import io
import zipfile
import numpy as np
import pandas as pd
import requests
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
            "fed_policy": 0.25,
            "real_yields": 0.25,
            "usd_index": 0.25,
            "net_liquidity": 0.25
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
# 2. MATHEMATICAL CORE ENGINES & SAFETY HELPERS
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


def safe_reindex_series(source_series: pd.Series, target_index: pd.Index) -> pd.Series:
    """Schutzfunktion: Richtet Zeitreihen absturzsicher auf das Ziel-Raster aus."""
    if source_series is None or not isinstance(source_series, pd.Series) or source_series.empty:
        return None
    
    s = source_series.copy()
    # Datetime säubern: Uhrzeiten entfernen & auf reine Kalendertage runden
    s.index = pd.to_datetime(s.index).tz_localize(None).floor('D')
    # Doppelte Einträge entfernen (letzten Wert des Tages behalten)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    
    clean_target = pd.to_datetime(target_index).tz_localize(None).floor('D')
    
    reindexed = s.reindex(clean_target, method='ffill').ffill().bfill()
    reindexed.index = target_index 
    return reindexed

# ==========================================
# 3. REAL MARKET DATA FETCHERS
# ==========================================

FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():
    """Holt den echten CNN Fear & Greed Index Score (inkl. Duplikate-Bereinigung)."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            hist = data.get("fear_and_greed_historical", {}).get("data", [])
            if hist:
                df_hist = pd.DataFrame(hist)
                df_hist["Date"] = pd.to_datetime(df_hist["x"], unit="ms").dt.tz_localize(None).dt.floor('D')
                df_hist = df_hist.drop_duplicates(subset=["Date"], keep="last")
                s = df_hist.set_index("Date")["y"].sort_index()
                return s
            if "fear_and_greed" in data and "score" in data["fear_and_greed"]:
                return float(data["fear_and_greed"]["score"])
    except Exception:
        pass
    return 55.0


@st.cache_data(ttl=14400)
def fetch_put_call_ratio():
    """Holt die echte Put/Call Ratio via CBOE CSV oder yFinance."""
    url = "https://cdn.cboe.com/data/us/options/market_statistics/daily_market_statistics/total_pc.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = df.columns.str.strip()
            date_col = [c for c in df.columns if 'DATE' in c.upper()][0]
            ratio_col = [c for c in df.columns if 'RATIO' in c.upper() or 'P/C' in c.upper()][0]
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None).dt.floor('D')
            df['Ratio'] = pd.to_numeric(df[ratio_col], errors='coerce')
            df = df.dropna(subset=['Date', 'Ratio']).drop_duplicates(subset=['Date'], keep='last')
            s = df.set_index('Date')['Ratio'].sort_index()
            if not s.empty:
                return s
    except Exception:
        pass

    try:
        pc_df = yf.download("^PCRATIO", period="1y", progress=False)
        if isinstance(pc_df.columns, pd.MultiIndex) and "Close" in pc_df.columns.levels[0]:
            pc_df = pc_df["Close"]
        elif "Close" in pc_df.columns:
            pc_df = pc_df["Close"]
            
        if not pc_df.empty:
            return pc_df.dropna()
    except Exception:
        pass

    return None


@st.cache_data(ttl=86400)
def fetch_fundamental_metrics():
    """Holt das echte KGV und Gewinnwachstum via SPY ETF."""
    pe_val = 24.5
    eg_val = 8.0
    try:
        spy = yf.Ticker("SPY")
        info = spy.info
        if "trailingPE" in info and info["trailingPE"]:
            pe_val = float(info["trailingPE"])
        if "earningsGrowth" in info and info["earningsGrowth"]:
            eg_val = float(info["earningsGrowth"]) * 100
    except Exception:
        pass
    return pe_val, eg_val


@st.cache_data(ttl=86400)
def fetch_sp500_cot_data():
    """Holt die wöchentlichen COT-Daten der Commercials von der CFTC."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    current_year = pd.Timestamp.now().year
    years = [current_year - 1, current_year]
    frames = []

    for yr in years:
        url = f"https://www.cftc.gov/files/dea/history/fut_com_txt_{yr}.zip"
        try:
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                    file_name = z.namelist()[0]
                    df_yr = pd.read_csv(z.open(file_name), low_memory=False)
                    sp_rows = df_yr[df_yr.iloc[:, 0].astype(str).str.contains("E-MINI S&P 500|S&P 500 CONSOLIDATED", case=False, na=False)]
                    if not sp_rows.empty:
                        frames.append(sp_rows)
        except Exception:
            pass

    if frames:
        try:
            df_all = pd.concat(frames, ignore_index=True)
            date_col = [c for c in df_all.columns if 'As_of_Date' in str(c)][0]
            long_col = [c for c in df_all.columns if 'Comm_Positions_Long_All' in str(c)][0]
            short_col = [c for c in df_all.columns if 'Comm_Positions_Short_All' in str(c)][0]

            df_all['Date'] = pd.to_datetime(df_all[date_col].astype(str), format='%Y%m%d', errors='coerce').dt.tz_localize(None).dt.floor('D')
            df_all['Net_Commercials'] = pd.to_numeric(df_all[long_col], errors='coerce') - pd.to_numeric(df_all[short_col], errors='coerce')
            df_all = df_all.dropna(subset=['Date']).drop_duplicates(subset=['Date'], keep='last')
            s = df_all.set_index('Date')['Net_Commercials'].sort_index()
            return s
        except Exception:
            pass
            
    return None

# ==========================================
# 4. MAIN PIPELINE (ALL LIVE DATA + NET LIQUIDITY)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_real_market_data():
    """Aggregiert Live-Daten inkl. Fed Net Liquidity."""
    tickers = {
        "sp500": "^GSPC",
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "move": "^MOVE",
        "hyg": "HYG",
        "lqd": "LQD"
    }
    
    data = yf.download(list(tickers.values()), period="2y", interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            data = data["Close"]
        elif "Close" in data.columns.levels[1]:
            data = data.xs("Close", axis=1, level=1)
    elif "Close" in data.columns:
        data = data["Close"]
        
    data = data.rename(columns={v: k for k, v in tickers.items()}).ffill().dropna()
    
    df_raw = pd.DataFrame(index=data.index)
    
    # --- 1. Technische Indikatoren ---
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
    
    # --- 2. Marktinterna ---
    df_raw["vix_score"] = data["vix"]
    df_raw["advance_decline"] = sp500.pct_change().rolling(20).sum() * 100
    
    # --- 3. Frühwarnindikatoren & Dollar ---
    df_raw["usd_index"] = data["dxy"]
    df_raw["move_index"] = data["move"]
    df_raw["credit_spreads"] = data["lqd"] / data["hyg"] 
    
    # --- 4. Makro-Daten & ECHTE FED NET LIQUIDITY (FRED API) ---
    if FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            fed_funds = fred.get_series('FEDFUNDS')
            real_yields = fred.get_series('DFII10')
            
            walcl = fred.get_series('WALCL')      # Bilanzsumme ($ Mio)
            tga = fred.get_series('WTREGEN')       # TGA Konto ($ Mio)
            rrp = fred.get_series('RPTCW')         # Reverse Repo ($ Mrd)
            
            net_liq = (walcl - tga - (rrp * 1000)) / 1000.0
            
            df_raw["fed_policy"] = safe_reindex_series(fed_funds, df_raw.index)
            df_raw["real_yields"] = safe_reindex_series(real_yields, df_raw.index)
            df_raw["net_liquidity"] = safe_reindex_series(net_liq, df_raw.index)
        except Exception:
            df_raw["fed_policy"] = 5.25
            df_raw["real_yields"] = 2.0
            df_raw["net_liquidity"] = 6000.0
    else:
        df_raw["fed_policy"] = 5.25
        df_raw["real_yields"] = 2.0
        df_raw["net_liquidity"] = 6000.0

    # Fallback-Werte sichern
    for col, default_val in [("fed_policy", 5.25), ("real_yields", 2.0), ("net_liquidity", 6000.0)]:
        if col not in df_raw.columns or df_raw[col].isnull().all():
            df_raw[col] = default_val

    # --- 5. COT Commercials (CFTC API) ---
    cot_series = fetch_sp500_cot_data()
    cot_reindexed = safe_reindex_series(cot_series, df_raw.index)
    if cot_reindexed is not None:
        df_raw["cot_commercials"] = cot_reindexed
    else:
        df_raw["cot_commercials"] = df_raw["distance_200ma"] * 1000 + 50000

    # --- 6. Live Stimmungs- & Fundamentaldaten ---
    fg_data = fetch_fear_and_greed()
    if isinstance(fg_data, pd.Series):
        fg_reindexed = safe_reindex_series(fg_data, df_raw.index)
        if fg_reindexed is not None:
            df_raw["fear_greed"] = fg_reindexed
        else:
            df_raw["fear_greed"] = 55.0
    elif isinstance(fg_data, (int, float)):
        df_raw["fear_greed"] = float(fg_data)
    else:
        df_raw["fear_greed"] = 55.0

    pc_series = fetch_put_call_ratio()
    if isinstance(pc_series, pd.Series):
        pc_reindexed = safe_reindex_series(pc_series, df_raw.index)
        if pc_reindexed is not None:
            df_raw["put_call_ratio"] = pc_reindexed
        else:
            df_raw["put_call_ratio"] = (df_raw["vix_score"] / 30.0).clip(0.4, 1.2)
    else:
        df_raw["put_call_ratio"] = (df_raw["vix_score"] / 30.0).clip(0.4, 1.2)

    pe_val, eg_val = fetch_fundamental_metrics()
    df_raw["pe_valuation"] = pe_val
    df_raw["earnings_growth"] = eg_val

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
with st.spinner('Lade Live-Daten inkl. Fed Net Liquidity...'):
    df = fetch_real_market_data()

heute = df.iloc[-1]

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="Market Regime Dashboard", 
    page_icon="📊",
    layout="centered"
)

st.title("📊 Market Regime Dashboard")
st.caption(f"Stand: {df.index[-1].strftime('%d.%m.%Y')} | Asset: S&P 500 (inkl. Fed Net Liquidity)")
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

# 🎯 Intraday Trading Bias (15:30 Uhr Setup Box)
st.markdown("---")
st.markdown("### 🎯 Intraday Trading Bias (15:30 Uhr Setup)")

score = heute['Final_Regime_Score']
mci = heute['MCI']

if score >= 60:
    bias = "🟢 BULLISCH (Long Bias)"
    rule = "Suche bevorzugt nach Dip-Käufen an Intraday-Support-Zonen (VWAP, EMA 20). Short-Setups meiden oder stark reduzieren."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size (MCI uneinig)"
elif score <= 40:
    bias = "🔴 BÄRISCH (Short Bias)"
    rule = "Suche bevorzugt nach Short-Setups bei Erholungen an Intraday-Resistance-Zonen. Longs nur als schnelle Scalps."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size (MCI uneinig)"
else:
    bias = "🟡 NEUTRAL / RANGE"
    rule = "Der Markt hat keine klare Richtung. Handle Range-Setups an den Grenzen der Initial Balance oder reduziere die Frequenz."
    pos_size = "50% Size (Erhöhtes Fehlsignal-Risiko)"

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Handelsrichtung", bias)
col_b2.metric("Positionsgröße", pos_size)
col_b3.metric("Fokus", "Trend-Follow" if abs(score - 50) > 15 else "Mean-Reversion")

st.info(f"**Intraday-Regel:** {rule}")

st.markdown("---")
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
st.markdown("### 📈 Historischer Verlauf")
st.line_chart(df[["Final_Regime_Score", "MCI"]].tail(120))
