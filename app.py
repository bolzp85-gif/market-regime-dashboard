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
# 0. STREAMLIT CONFIG (Muss ganz oben stehen!)
# ==========================================
st.set_page_config(page_title="Market Regime Dashboard", page_icon="📊", layout="centered")

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

LOOKBACK_CONFIG = {
    "fed_policy": 1260,
    "real_yields": 756,
    "net_liquidity": 756,
    "credit_spreads": 756,
    "usd_index": 504
}

# ==========================================
# 2. MATHEMATICAL CORE ENGINES & SAFETY HELPERS
# ==========================================

def normalize_to_percentile(series: pd.Series, lookback: int = 252, invert: bool = False) -> pd.Series:
    clean_series = series.ffill().bfill()
    
    if clean_series.isna().all():
        return pd.Series(50.0, index=series.index)

    rolling_mean = clean_series.rolling(window=lookback, min_periods=20).mean()
    rolling_std = clean_series.rolling(window=lookback, min_periods=20).std()
    
    rolling_std = rolling_std.replace(0, 1e-8)
    
    z_scores = (clean_series - rolling_mean) / rolling_std
    percentiles = norm.cdf(z_scores) * 100
    
    if invert:
        percentiles = 100 - percentiles
    return pd.Series(percentiles, index=series.index).clip(0, 100).ffill().bfill()


def calculate_mci(scores, weights):
    gesamt_score = np.average(scores, weights=weights)
    weighted_variance = np.average((scores - gesamt_score)**2, weights=weights)
    weighted_std = np.sqrt(weighted_variance)
    
    max_std = 50.0  
    mci = 100 * (1 - (weighted_std / max_std))
    return round(float(np.clip(mci, 0.0, 100.0)), 1)


def get_regime_label(score):
    if score >= 90: return "🟢 Risk-On (Extrem Bullisch)"
    elif score >= 75: return "🟢 Expansion (Bullisch)"
    elif score >= 60: return "🟡 Übergangsphase (Leicht Bullisch)"
    elif score >= 40: return "🟡 Neutral"
    elif score >= 25: return "🟠 Risk-Off (Bärisch)"
    else: return "🔴 Stressphase (Stark Bärisch)"


def safe_reindex_series(source_series: pd.Series, target_index: pd.Index) -> pd.Series:
    if source_series is None or not isinstance(source_series, pd.Series) or source_series.empty:
        return None
    
    s = source_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).floor('D')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    
    clean_target = pd.to_datetime(target_index).tz_localize(None).floor('D')
    
    reindexed = s.reindex(clean_target, method='ffill').ffill().bfill()
    reindexed.index = target_index 
    return reindexed

# ==========================================
# 3. REAL MARKET DATA FETCHERS
# ==========================================

FRED_API_KEY = ""
try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    pass


@st.cache_data(ttl=14400)
def fetch_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0"}
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
                return s, True
    except Exception:
        pass
    return 55.0, False


@st.cache_data(ttl=14400)
def fetch_put_call_ratio():
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
                return s, True
    except Exception:
        pass
    return None, False


@st.cache_data(ttl=86400)
def fetch_fundamental_metrics():
    pe_val, eg_val = 24.5, 8.0
    try:
        spy = yf.Ticker("SPY")
        info = spy.info
        if "trailingPE" in info and info["trailingPE"]:
            pe_val = float(info["trailingPE"])
        if "earningsGrowth" in info and info["earningsGrowth"]:
            eg_val = float(info["earningsGrowth"]) * 100
        return pe_val, eg_val, True
    except Exception:
        pass
    return pe_val, eg_val, False


@st.cache_data(ttl=86400)
def fetch_sp500_cot_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
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
                    sp_rows = df_yr[df_yr.iloc[:, 0].astype(str).str.contains("E-MINI S&P 500", case=False, na=False)]
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
            return df_all.set_index('Date')['Net_Commercials'].sort_index(), True
        except Exception:
            pass
    return None, False

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

@st.cache_data(ttl=3600)
def fetch_real_market_data():
    feed_status = {}
    
    tickers = {
        "sp500": "^GSPC", "vix": "^VIX", "dxy": "DX-Y.NYB",
        "move": "^MOVE", "hyg": "HYG", "lqd": "LQD"
    }
    
    data = yf.download(list(tickers.values()), period="5y", interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]: data = data["Close"]
        elif "Close" in data.columns.levels[1]: data = data.xs("Close", axis=1, level=1)
    elif "Close" in data.columns: data = data["Close"]
        
    data = data.rename(columns={v: k for k, v in tickers.items()}).ffill().bfill().dropna(how='all')
    feed_status["yFinance (Preis & Tech)"] = not data.empty
    
    if data.empty or "sp500" not in data.columns:
        return pd.DataFrame(), feed_status
    
    df_raw = pd.DataFrame(index=data.index)
    sp500 = data["sp500"]
    
    df_raw["distance_50ma"] = ((sp500 - sp500.rolling(50).mean()) / sp500.rolling(50).mean()) * 100
    df_raw["distance_200ma"] = ((sp500 - sp500.rolling(200).mean()) / sp500.rolling(200).mean()) * 100
    
    delta = sp500.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df_raw["rsi_momentum"] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-8))))
    
    df_raw["vix_score"] = data["vix"] if "vix" in data.columns else 20.0
    df_raw["usd_index"] = data["dxy"] if "dxy" in data.columns else 100.0
    df_raw["move_index"] = data["move"] if "move" in data.columns else 100.0
    df_raw["credit_spreads"] = data["lqd"] / data["hyg"] if "lqd" in data.columns and "hyg" in data.columns else 1.0
    df_raw["advance_decline"] = sp500.pct_change().rolling(20).sum() * 100
    
    if FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            walcl_s = safe_reindex_series(fred.get_series('WALCL'), df_raw.index)
            tga_s = safe_reindex_series(fred.get_series('WTREGEN'), df_raw.index)
            rrp_s = safe_reindex_series(fred.get_series('RPTCW'), df_raw.index)
            
            if walcl_s is not None and tga_s is not None and rrp_s is not None:
                df_raw["net_liquidity"] = (walcl_s - tga_s - (rrp_s * 1000.0)) / 1000.0
            else:
                df_raw["net_liquidity"] = 6000.0

            df_raw["fed_policy"] = safe_reindex_series(fred.get_series('FEDFUNDS'), df_raw.index)
            df_raw["real_yields"] = safe_reindex_series(fred.get_series('DFII10'), df_raw.index)
            feed_status["FRED API (Makro & Fed)"] = True
        except Exception:
            df_raw["fed_policy"], df_raw["real_yields"], df_raw["net_liquidity"] = 5.25, 2.0, 6000.0
            feed_status["FRED API (Makro & Fed)"] = False
    else:
        df_raw["fed_policy"], df_raw["real_yields"], df_raw["net_liquidity"] = 5.25, 2.0, 6000.0
        feed_status["FRED API (Makro & Fed)"] = False

    cot_data, cot_live = fetch_sp500_cot_data()
    feed_status["CFTC CoT (Commercials)"] = cot_live
    cot_reindexed = safe_reindex_series(cot_data, df_raw.index)
    df_raw["cot_commercials"] = cot_reindexed if cot_reindexed is not None else -df_raw["distance_200ma"] * 1000

    fg_data, fg_live = fetch_fear_and_greed()
    feed_status["CNN Fear & Greed"] = fg_live
    if isinstance(fg_data, pd.Series):
        fg_reindexed = safe_reindex_series(fg_data, df_raw.index)
        df_raw["fear_greed"] = fg_reindexed if fg_reindexed is not None else 55.0
    else:
        df_raw["fear_greed"] = float(fg_data)

    pc_data, pc_live = fetch_put_call_ratio()
    feed_status["CBOE Put/Call Ratio"] = pc_live
    pc_reindexed = safe_reindex_series(pc_data, df_raw.index)
    df_raw["put_call_ratio"] = pc_reindexed if pc_reindexed is not None else (df_raw["vix_score"] / 30.0).clip(0.4, 1.2)

    pe_val, eg_val, fund_live = fetch_fundamental_metrics()
    feed_status["Fundamentaldaten (KGV)"] = fund_live
    df_raw["pe_valuation"] = pe_val
    df_raw["earnings_growth"] = eg_val

    df_norm = pd.DataFrame(index=df_raw.index)
    inverts = ["put_call_ratio", "vix_score", "pe_valuation", "credit_spreads", 
               "move_index", "usd_index", "fed_policy", "real_yields", "cot_commercials"]
    
    for col in df_raw.columns:
        should_invert = col in inverts
        lb = LOOKBACK_CONFIG.get(col, 252)
        df_norm[col] = normalize_to_percentile(df_raw[col], lookback=lb, invert=should_invert)
        
    df_dashboard = pd.DataFrame(index=df_raw.index)
    config = SP500_CONFIG
    
    for saeule, indikatoren in config["Sub_Indikatoren_Gewichte"].items():
        cols = list(indikatoren.keys())
        weights = list(indikatoren.values())
        df_dashboard[f"Saeule_{saeule}"] = df_norm[cols].dot(weights)
        
    saeulen_cols = [f"Saeule_{s}" for s in config["Saeulen_Gewichte"].keys()]
    saeulen_weights = list(config["Saeulen_Gewichte"].values())
    
    df_dashboard["Final_Regime_Score"] = df_dashboard[saeulen_cols].dot(saeulen_weights)
    df_dashboard["MCI"] = [calculate_mci(row.values, saeulen_weights) for idx, row in df_dashboard[saeulen_cols].iterrows()]
    df_dashboard["Delta_1D"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(1)
    df_dashboard["Delta_1W"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(5)
    df_dashboard["Delta_1M"] = df_dashboard["Final_Regime_Score"] - df_dashboard["Final_Regime_Score"].shift(21)
    
    # 🛡️ FIX: Statt hartem .dropna() (was bei einer einzelnen Lücke alles löscht),
    # füllen wir verbleibende NaN-Ränder und behalten die historischen Daten stabil.
    df_dashboard = df_dashboard.ffill().bfill()
    
    return df_dashboard.round(1), feed_status

# ==========================================
# 5. STREAMLIT UI & SIDEBAR
# ==========================================

with st.spinner('Lade Live-Daten inkl. Fed Net Liquidity...'):
    df, feed_status = fetch_real_market_data()

if df.empty:
    st.error("⚠️ Marktdaten konnten nicht aggregiert werden. Bitte überprüfe die APIs und lade die Seite neu.")
    st.stop()

heute = df.iloc[-1]

with st.sidebar:
    st.title("⚙️ System Status")
    st.markdown(f"**Letztes Update:** {df.index[-1].strftime('%d.%m.%Y')}")
    st.markdown("---")
    
    st.markdown("### 📡 API Live-Feed Monitor")
    all_live = all(feed_status.values())
    
    if all_live:
        st.success("🟢 Alle Quellen LIVE")
    else:
        st.warning("🟡 Fallback-Daten aktiv")
        
    st.markdown("---")
    for source, is_live in feed_status.items():
        if is_live:
            st.markdown(f"🟢 **{source}**")
        else:
            st.markdown(f"⚠️ **{source}** *(Offline/Fallback)*")

    st.markdown("---")
    st.info("Tipp: Die Makro-Säule berechnet Zinszyklen dynamisch auf Basis von bis zu 5 Jahren (1260 Handelstage).")

st.title("📊 Market Regime Dashboard")
st.caption(f"Stand: {df.index[-1].strftime('%d.%m.%Y')} | Asset: S&P 500")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Final Regime Score", value=f"{heute['Final_Regime_Score']} / 100", delta=f"{heute['Delta_1D']:+g} (Heute)")
with col2:
    st.metric(label="Model Confidence Index (MCI)", value=f"{heute['MCI']}%", delta=f"{heute['Delta_1W']:+g} (vs. Vorwoche)", delta_color="off")

st.info(f"**Aktuelles Marktregime:** {get_regime_label(heute['Final_Regime_Score'])}")

st.markdown("---")
st.markdown("### 🎯 Intraday Trading Bias (15:30 Uhr Framework)")
score, mci = heute['Final_Regime_Score'], heute['MCI']

if score >= 60:
    bias, rule = "🟢 BULLISCH (Long Bias)", "Suche bevorzugt nach Dip-Käufen an Intraday-Support-Zonen (VWAP, EMA 10/20/50/100). Short-Setups meiden."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size (MCI uneinig)"
elif score <= 40:
    bias, rule = "🔴 BÄRISCH (Short Bias)", "Suche bevorzugt nach Short-Setups bei Erholungen an Intraday-Resistance-Zonen. Longs nur als Gegen-Trend Scalps."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size (MCI uneinig)"
else:
    bias, rule = "🟡 NEUTRAL / RANGE", "Der Markt hat keine klare Richtung. Handle Range-Setups an den Grenzen der Initial Balance oder Vortages-Extrema."
    pos_size = "50% Size (Risiko)"

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Handelsrichtung", bias)
col_b2.metric("Positionsgröße", pos_size)
col_b3.metric("Fokus", "Trend-Follow" if abs(score - 50) > 15 else "Mean-Reversion")
st.info(f"**Übergeordnete Regel:** {rule}")

# ==========================================
# 6. INTRADAY EXECUTION CHECKLIST
# ==========================================
st.markdown("---")
st.markdown("### 📋 Intraday Execution Checkliste (Vor-Börse 15:30 Uhr)")

with st.expander("🔍 Vor-Börsen-Check aufklappen & durchgehen", expanded=True):
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("**1. Kalender, Events & Saisonalität:**")
        st.checkbox("Keine High-Impact News (CPI, NFP, FOMC) in den nächsten 30 Minuten.")
        st.checkbox("Wochentags-Saisonalität berücksichtigt (z.B. Montag-Range vs. Di/Mi Trendtag).")
        st.checkbox("Hexensabbat / Triple Witching Verfallstag geprüft?")

    with col_c2:
        st.markdown("**2. Technisches Chart-Setup (5m / 15m):**")
        st.checkbox("Vortages-Hoch (PWH), Vortages-Tief (PWL) & Schlusskurs (PDC) im Chart markiert.")
        st.checkbox("Kursausrichtung zu EMAs (10, 20, 50, 100) im Einklang mit Regime-Bias?")
        st.checkbox("Stochastik-Signal im überkauften/überverkauften Bereich als Trigger genutzt?")
        st.checkbox("Stop-Loss mit ATR (z.B. 1.5x ATR) berechnet?")

st.markdown("---")
st.markdown("### 🏛️ Die 6 Hauptsäulen (Aktuelle Scores)")
saeulen_daten = {
    "Hauptsäule": list(SP500_CONFIG["Saeulen_Gewichte"].keys()),
    "Gewichtung": [f"{v*100:.0f}%" for v in SP500_CONFIG["Saeulen_Gewichte"].values()],
    "Aktueller Score (0-100)": [heute[f"Saeule_{k}"] for k in SP500_CONFIG["Saeulen_Gewichte"].keys()]
}
st.dataframe(pd.DataFrame(saeulen_daten), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📈 Historischer Verlauf")
st.line_chart(df[["Final_Regime_Score", "MCI"]].tail(120))
