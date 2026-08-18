import io
import zipfile
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import streamlit as st
import yfinance as yf
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytrends.request import TrendReq

# ==========================================
# 0. STREAMLIT CONFIG
# ==========================================
st.set_page_config(page_title="Multi-Asset Regime Dashboard", page_icon="📊", layout="centered")

# ==========================================
# 1. ASSET CONFIGURATIONS & WEIGHTS
# ==========================================
ASSET_CONFIGS = {
    "S&P 500": {
        "ticker": "^GSPC",
        "volatility_ticker": "^VIX",
        "cot_code": "E-MINI S&P 500",
        "invert_inverts": ["vix_score", "vvix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "fed_policy", "real_yields"],
        "Saeulen_Gewichte": {
            "Makroekonomie": 0.25, "Positionierung": 0.20, "Marktinterna": 0.20,
            "Technischer_Trend": 0.15, "Fundamentale_Faktoren": 0.10, "Fruehwarnindikatoren": 0.10
        },
        "Sub_Gewichte": {
            "Positionierung": {"cot_commercials": 0.50, "fear_greed": 0.50},
            "Marktinterna": {"advance_decline": 0.50, "vix_score": 0.50},
            "Fundamentale_Faktoren": {"pe_valuation": 1.0}
        }
    },
    "Nasdaq 100": {
        "ticker": "NQ=F",  
        "volatility_ticker": "^VXN",
        "cot_code": "NASDAQ-100",  
        "invert_inverts": ["vix_score", "vvix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "fed_policy", "real_yields"],
        "Saeulen_Gewichte": {
            "Makroekonomie": 0.25, "Positionierung": 0.20, "Marktinterna": 0.20,
            "Technischer_Trend": 0.15, "Fundamentale_Faktoren": 0.10, "Fruehwarnindikatoren": 0.10
        },
        "Sub_Gewichte": {
            "Positionierung": {"cot_commercials": 0.50, "fear_greed": 0.50},
            "Marktinterna": {"advance_decline": 0.50, "vix_score": 0.50},
            "Fundamentale_Faktoren": {"pe_valuation": 1.0}
        }
    },
    "Gold (XAU/USD)": {
        "ticker": "GC=F",
        "volatility_ticker": "^GVZ",
        "cot_code": "GOLD",
        "invert_inverts": ["vix_score", "vvix_score", "usd_index", "real_yields", "fed_policy"],
        "Saeulen_Gewichte": {
            "Makroekonomie": 0.35, "Positionierung": 0.25, "Marktinterna": 0.15,
            "Technischer_Trend": 0.15, "Fundamentale_Faktoren": 0.0, "Fruehwarnindikatoren": 0.10
        },
        "Sub_Gewichte": {
            "Positionierung": {"cot_commercials": 0.80, "fear_greed": 0.20},
            "Marktinterna": {"obv_momentum": 0.50, "vix_score": 0.50},
            "Fundamentale_Faktoren": {} 
        }
    },
    "WTI Crude Oil": {
        "ticker": "CL=F",
        "volatility_ticker": "^OVX",
        "cot_code": "CRUDE OIL",
        "invert_inverts": ["vix_score", "vvix_score", "usd_index", "inventories"],
        "Saeulen_Gewichte": {
            "Makroekonomie": 0.30, "Positionierung": 0.25, "Marktinterna": 0.15,
            "Technischer_Trend": 0.20, "Fundamentale_Faktoren": 0.10, "Fruehwarnindikatoren": 0.0
        },
        "Sub_Gewichte": {
            "Positionierung": {"cot_commercials": 0.80, "fear_greed": 0.20},
            "Marktinterna": {"obv_momentum": 0.50, "vix_score": 0.50},
            "Fundamentale_Faktoren": {"inventories": 1.0} 
        }
    }
}

SUB_WEIGHTS_BASE = {
    "Makroekonomie": {"fed_policy": 0.25, "real_yields": 0.25, "usd_index": 0.25, "net_liquidity": 0.25},
    "Technischer_Trend": {"distance_200ma": 0.40, "distance_50ma": 0.30, "rsi_momentum": 0.30},
    "Fruehwarnindikatoren": {"credit_spreads": 0.40, "move_index": 0.30, "vvix_score": 0.30} # VVIX Integriert
}

LOOKBACK_CONFIG = {
    "fed_policy": 1260, "real_yields": 756, "net_liquidity": 756,
    "credit_spreads": 756, "usd_index": 504, "vvix_score": 252
}

VOLA_THRESHOLDS = {
    "S&P 500": 30.0,
    "Gold (XAU/USD)": 25.0,
    "WTI Crude Oil": 45.0
}

TREND_KEYWORD_MAP = {
    "S&P 500": {"geo": "US", "lang": "en-US", "bull": ["buy stocks", "buy the dip"], "bear": ["stock market crash", "recession"]},
    "Nasdaq 100": {"geo": "US", "lang": "en-US", "bull": ["tech stocks", "buy the dip"], "bear": ["market crash", "tech bubble"]},
    "Gold (XAU/USD)": {"geo": "DE", "lang": "de-DE", "bull": ["Gold kaufen", "Goldmünzen"], "bear": ["Gold verkaufen", "Altgold"]},
    "WTI Crude Oil": {"geo": "DE", "lang": "de-DE", "bull": ["Heizöl kaufen", "Spritpreise"], "bear": ["Ölpreis crash", "Öl verkaufen"]}
}

@st.cache_data(ttl=21600)
def fetch_google_trends_sentiment(asset_name: str):
    cfg = TREND_KEYWORD_MAP.get(asset_name, TREND_KEYWORD_MAP["S&P 500"])
    try:
        pytrends = TrendReq(hl=cfg["lang"], tz=360)
        pytrends.build_payload(cfg["bull"] + cfg["bear"], timeframe='today 3-m', geo=cfg["geo"])
        df_trends = pytrends.interest_over_time()
        
        if df_trends.empty: return 50.0, 0.0, False
        if 'isPartial' in df_trends.columns: df_trends = df_trends.drop(columns=['isPartial'])

        def calc_z(series):
            mean = series.rolling(21, min_periods=5).mean()
            std = series.rolling(21, min_periods=5).std().replace(0, 1e-8)
            return (series - mean) / std

        valid_bull = [kw for kw in cfg["bull"] if kw in df_trends.columns]
        valid_bear = [kw for kw in cfg["bear"] if kw in df_trends.columns]
        if not valid_bull or not valid_bear: return 50.0, 0.0, False

        z_bull = sum(calc_z(df_trends[kw]) for kw in valid_bull) / len(valid_bull)
        z_bear = sum(calc_z(df_trends[kw]) for kw in valid_bear) / len(valid_bear)
        latest_spread = float((z_bull - z_bear).dropna().iloc[-1])
        return round(float(np.clip(50.0 - (latest_spread * 15.0), 0.0, 100.0)), 1), round(latest_spread, 2), True
    except Exception:
        return 50.0, 0.0, False

# ==========================================
# MATHEMATICAL CORE ENGINES & HELPERS
# ==========================================

def strip_timezone(datetime_index_or_series):
    dt = pd.to_datetime(datetime_index_or_series)
    if hasattr(dt, 'dt'): return dt.dt.tz_convert(None) if dt.dt.tz is not None else dt
    else: return dt.tz_convert(None) if dt.tz is not None else dt

def normalize_to_percentile(series: pd.Series, lookback: int = 252, invert: bool = False) -> pd.Series:
    clean_series = series.ffill().bfill()
    if clean_series.isna().all(): return pd.Series(50.0, index=series.index)
    rolling_mean = clean_series.rolling(window=lookback, min_periods=20).mean()
    rolling_std = clean_series.rolling(window=lookback, min_periods=20).std().replace(0, 1e-8)
    z_scores = (clean_series - rolling_mean) / rolling_std
    percentiles = norm.cdf(z_scores) * 100
    if invert: percentiles = 100 - percentiles
    return pd.Series(percentiles, index=series.index).clip(0, 100).ffill().bfill()

def calculate_mci(scores, weights):
    gesamt_score = np.average(scores, weights=weights)
    weighted_variance = np.average((scores - gesamt_score)**2, weights=weights)
    return round(float(np.clip(100 * (1 - (np.sqrt(weighted_variance) / 50.0)), 0.0, 100.0)), 1)

def get_regime_label(score):
    if score >= 90: return "🟢 Risk-On (Extrem Bullisch)"
    elif score >= 75: return "🟢 Expansion (Bullisch)"
    elif score >= 60: return "🟡 Übergangsphase (Leicht Bullisch)"
    elif score >= 40: return "🟡 Neutral"
    elif score >= 25: return "🟠 Risk-Off (Bärisch)"
    else: return "🔴 Stressphase (Stark Bärisch)"

def safe_reindex_series(source_series: pd.Series, target_index: pd.Index) -> pd.Series:
    if source_series is None or source_series.empty: return None
    s = source_series.copy()
    s.index = strip_timezone(s.index).floor('D')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s.reindex(strip_timezone(target_index).floor('D'), method='ffill').ffill().bfill().set_axis(target_index)

# ==========================================
# EXTERNAL DATA FETCHERS (FRED, CNN, CFTC)
# ==========================================

FRED_API_KEY = ""
try:
    if "FRED_API_KEY" in st.secrets: FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception: pass

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():
    try:
        res = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", 
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if res.status_code == 200:
            hist = res.json().get("fear_and_greed_historical", {}).get("data", [])
            if hist:
                df = pd.DataFrame(hist)
                df["Date"] = strip_timezone(pd.to_datetime(df["x"], unit="ms")).dt.floor('D')
                return df.drop_duplicates(subset=["Date"], keep="last").set_index("Date")["y"].sort_index(), True
    except Exception: pass
    return 55.0, False

@st.cache_data(ttl=86400)
def fetch_cot_data(asset_search_string):
    frames = []
    for yr in [pd.Timestamp.now().year - 1, pd.Timestamp.now().year]:
        try:
            res = requests.get(f"https://www.cftc.gov/files/dea/history/fut_com_txt_{yr}.zip", headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            if res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                    rows = df[df.iloc[:, 0].astype(str).str.contains(asset_search_string, case=False, na=False)]
                    if not rows.empty: frames.append(rows)
        except Exception: pass
    if frames:
        try:
            df_all = pd.concat(frames, ignore_index=True)
            date_col = [c for c in df_all.columns if 'As_of_Date' in str(c)][0]
            df_all['Date'] = strip_timezone(pd.to_datetime(df_all[date_col].astype(str), format='%Y%m%d', errors='coerce')).dt.floor('D')
            df_all['Net_Commercials'] = pd.to_numeric(df_all[[c for c in df_all.columns if 'Comm_Positions_Long_All' in str(c)][0]], errors='coerce') - pd.to_numeric(df_all[[c for c in df_all.columns if 'Comm_Positions_Short_All' in str(c)][0]], errors='coerce')
            return df_all.dropna(subset=['Date']).drop_duplicates(subset=['Date'], keep='last').set_index('Date')['Net_Commercials'].sort_index(), True
        except Exception: pass
    return None, False

# ==========================================
# MAIN PIPELINE
# ==========================================

@st.cache_data(ttl=3600)
def fetch_multi_asset_data(selected_asset):
    cfg = ASSET_CONFIGS[selected_asset]
    feed_status = {}
    
    # NEU: VVIX Ticker hinzugefügt
    tickers = {
        "asset": cfg["ticker"], "vix": cfg["volatility_ticker"],
        "dxy": "DX=F", "move": "^MOVE", "hyg": "HYG", "lqd": "LQD", "vvix": "^VVIX"
    }
    
    data = yf.download(list(tickers.values()), period="5y", interval="1d")
    
    # Sicheres Entpacken der MultiIndex Yahoo Finance Daten (inkl. High/Low für ATR)
    def get_price_component(df, comp):
        if isinstance(df.columns, pd.MultiIndex):
            try: return df[comp]
            except KeyError: return df.xs(comp, level=1, axis=1)
        return df if comp == "Close" else None

    close_data = get_price_component(data, "Close")
    high_data = get_price_component(data, "High")
    low_data = get_price_component(data, "Low")
    vol_data = get_price_component(data, "Volume")
    
    if close_data is None or close_data.empty:
        return pd.DataFrame(), feed_status

    close_data = close_data.rename(columns={v: k for k, v in tickers.items()}).ffill().bfill().dropna(how='all')
    high_data = high_data.rename(columns={v: k for k, v in tickers.items()}).ffill().bfill() if high_data is not None else close_data
    low_data = low_data.rename(columns={v: k for k, v in tickers.items()}).ffill().bfill() if low_data is not None else close_data
    
    feed_status["yFinance (Preis, Vol. & ATR)"] = not close_data.empty
    if "asset" not in close_data.columns: return pd.DataFrame(), feed_status
    
    df_raw = pd.DataFrame(index=close_data.index)
    price = close_data["asset"]
    
    # ATR 14 Berechnung (True Range)
    tr1 = high_data["asset"] - low_data["asset"]
    tr2 = (high_data["asset"] - price.shift(1)).abs()
    tr3 = (low_data["asset"] - price.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_raw["atr_14"] = true_range.rolling(14).mean()
    df_raw["daily_range"] = tr1 # Die rein heutige Spanne H-L
    
    # Volumen
    asset_volume = pd.Series(1000, index=close_data.index)
    has_vol = False
    if vol_data is not None:
        vol_renamed = vol_data.rename(columns={v: k for k, v in tickers.items()}) if isinstance(vol_data, pd.DataFrame) else vol_data
        if "asset" in vol_renamed.columns:
            asset_volume = vol_renamed["asset"].ffill().bfill()
            has_vol = True
    feed_status["Volumen/Orderflow Feed"] = has_vol
    
    # Technische Indikatoren
    df_raw["distance_50ma"] = ((price - price.rolling(50).mean()) / price.rolling(50).mean()) * 100
    df_raw["distance_200ma"] = ((price - price.rolling(200).mean()) / price.rolling(200).mean()) * 100
    
    delta = price.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df_raw["rsi_momentum"] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-8))))
    
    # Makro / Vola Indikatoren inkl. VVIX
    df_raw["vix_score"] = close_data["vix"] if "vix" in close_data.columns else 20.0
    df_raw["vvix_score"] = close_data["vvix"] if "vvix" in close_data.columns else 100.0
    df_raw["usd_index"] = close_data["dxy"] if "dxy" in close_data.columns else 100.0
    df_raw["move_index"] = close_data["move"] if "move" in close_data.columns else 100.0
    df_raw["credit_spreads"] = close_data["lqd"] / close_data["hyg"] if "lqd" in close_data.columns and "hyg" in close_data.columns else 1.0
    
    df_raw["advance_decline"] = price.pct_change().rolling(20).sum() * 100
    
    obv = pd.Series(np.select([delta > 0, delta < 0], [asset_volume, -asset_volume], default=0), index=price.index).cumsum()
    obv_ema = obv.ewm(span=50).mean()
    df_raw["obv_momentum"] = ((obv - obv_ema) / obv_ema.abs().replace(0, 1e-8)) * 100

    if selected_asset == "S&P 500": df_raw["pe_valuation"] = 24.5 

    if FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            walcl_s, tga_s, rrp_s = [safe_reindex_series(fred.get_series(x), df_raw.index) for x in ['WALCL', 'WTREGEN', 'RPTCW']]
            df_raw["net_liquidity"] = (walcl_s - tga_s - (rrp_s * 1000.0)) / 1000.0 if walcl_s is not None else 6000.0
            df_raw["fed_policy"] = safe_reindex_series(fred.get_series('FEDFUNDS'), df_raw.index)
            df_raw["real_yields"] = safe_reindex_series(fred.get_series('DFII10'), df_raw.index)
            if selected_asset == "WTI Crude Oil":
                inv_s = safe_reindex_series(fred.get_series('WCESTUS1'), df_raw.index)
                df_raw["inventories"] = inv_s if inv_s is not None else 500000.0
            feed_status["FRED API (Makro & Fed)"] = True
        except Exception:
            df_raw["fed_policy"], df_raw["real_yields"], df_raw["net_liquidity"] = 5.25, 2.0, 6000.0
            if selected_asset == "WTI Crude Oil": df_raw["inventories"] = 500000.0
            feed_status["FRED API (Makro & Fed)"] = False
    else:
        df_raw["fed_policy"], df_raw["real_yields"], df_raw["net_liquidity"] = 5.25, 2.0, 6000.0
        if selected_asset == "WTI Crude Oil": df_raw["inventories"] = 500000.0
        feed_status["FRED API (Makro & Fed)"] = False

    cot_data, cot_live = fetch_cot_data(cfg["cot_code"])
    feed_status[f"CFTC CoT ({cfg['cot_code']})"] = cot_live
    cot_reindexed = safe_reindex_series(cot_data, df_raw.index)
    df_raw["cot_commercials"] = cot_reindexed if cot_reindexed is not None else -df_raw["distance_200ma"] * 1000

    fg_data, fg_live = fetch_fear_and_greed()
    feed_status["CNN Fear & Greed"] = fg_live
    df_raw["fear_greed"] = safe_reindex_series(fg_data, df_raw.index) if isinstance(fg_data, pd.Series) else float(fg_data)

    df_norm = pd.DataFrame(index=df_raw.index)
    for col in df_raw.columns:
        if col in ["atr_14", "daily_range"]: continue # Werden nicht normiert
        df_norm[col] = normalize_to_percentile(df_raw[col], lookback=LOOKBACK_CONFIG.get(col, 252), invert=(col in cfg["invert_inverts"]))
        
    df_dashboard = pd.DataFrame(index=df_raw.index)
    df_dashboard["Raw_Volatility"] = df_raw["vix_score"]
    df_dashboard["ATR_14"] = df_raw["atr_14"]
    df_dashboard["Daily_Range"] = df_raw["daily_range"]
    df_dashboard["Asset_Price"] = price.ffill().bfill()
    
    active_sub_weights = {**SUB_WEIGHTS_BASE, **cfg["Sub_Gewichte"]}
    
    for saeule, indikatoren in active_sub_weights.items():
        cols = [c for c in indikatoren.keys() if c in df_norm.columns]
        weights = [indikatoren[c] for c in cols]
        if cols and sum(weights) > 0:
            df_dashboard[f"Saeule_{saeule}"] = df_norm[cols].dot(np.array(weights) / np.sum(weights))
        
    saeulen_cols = [f"Saeule_{s}" for s in cfg["Saeulen_Gewichte"].keys() if f"Saeule_{s}" in df_dashboard.columns]
    saeulen_weights = [cfg["Saeulen_Gewichte"][s.replace("Saeule_", "")] for s in saeulen_cols]
    
    if sum(saeulen_weights) > 0:
        saeulen_weights_norm = np.array(saeulen_weights) / np.sum(saeulen_weights)
        df_dashboard["Final_Regime_Score"] = df_dashboard[saeulen_cols].dot(saeulen_weights_norm).round(1)
        df_dashboard["MCI"] = [calculate_mci(df_dashboard[saeulen_cols].iloc[i].values, saeulen_weights_norm) for i in range(len(df_dashboard))]
    else:
        df_dashboard["Final_Regime_Score"], df_dashboard["MCI"] = 50.0, 0.0
        
    return df_dashboard.dropna(subset=["Final_Regime_Score"]), feed_status

# ==========================================
# STREAMLIT UI START
# ==========================================
with st.sidebar:
    st.title("⚙️ Multi-Asset Selector")
    selected_asset = st.selectbox("🎯 Asset auswählen", list(ASSET_CONFIGS.keys()), index=0)
    st.markdown("---")
    st.markdown("### 📡 API Live-Feed Monitor")

with st.spinner(f"Lade quantitative Daten für {selected_asset}..."):
    df_dash, feed_status = fetch_multi_asset_data(selected_asset)

with st.sidebar:
    for source, is_live in feed_status.items():
        st.markdown(f"{'🟢' if is_live else '⚠️'} **{source}**")

if df_dash.empty:
    st.error("⚠️ Marktdaten konnten nicht geladen werden.")
    st.stop()

heute = df_dash.iloc[-1]
delta_1d = heute['Final_Regime_Score'] - df_dash['Final_Regime_Score'].iloc[-2] if len(df_dash) > 1 else 0.0
delta_1w = heute['MCI'] - df_dash['MCI'].iloc[-6] if len(df_dash) > 5 else 0.0

st.title("📊 Quant Regime Dashboard")
st.caption(f"Asset: **{selected_asset}** | Stand: {df_dash.index[-1].strftime('%d.%m.%Y')}")
st.markdown("---")

col1, col2 = st.columns(2)
with col1: st.metric(label="Final Regime Score", value=f"{heute['Final_Regime_Score']} / 100", delta=f"{delta_1d:+.1f} (Heute)")
with col2: st.metric(label="Model Confidence Index (MCI)", value=f"{heute['MCI']}%", delta=f"{delta_1w:+.1f} (vs. Vorwoche)", delta_color="off")
st.info(f"**Aktuelles Marktregime ({selected_asset}):** {get_regime_label(heute['Final_Regime_Score'])}")

current_vola = heute.get("Raw_Volatility", 20.0)
vola_limit = VOLA_THRESHOLDS.get(selected_asset, 30.0)
if current_vola >= vola_limit:
    st.error(f"🚨 **VOLATILITÄTS-ALARM:** {ASSET_CONFIGS[selected_asset]['volatility_ticker']} bei **{current_vola:.2f}**. Extremes Slippage-Risiko! **Max 25% Positionsgröße!**")
elif current_vola >= vola_limit * 0.8:
    st.warning(f"⚠️ **Erhöhte Volatilität:** {ASSET_CONFIGS[selected_asset]['volatility_ticker']} bei **{current_vola:.2f}**.")

# --- INTRADAY TRADING BIAS ---
st.markdown("---")
st.markdown("### 🎯 Intraday Trading Bias")
score, mci = heute['Final_Regime_Score'], heute['MCI']

if score >= 60: bias, rule, pos_size = "🟢 BULLISCH", f"Dip-Käufe an Support-Zonen bei {selected_asset}.", "100% Size" if mci >= 70 else "75% Size"
elif score <= 40: bias, rule, pos_size = "🔴 BÄRISCH", f"Short-Setups an Resistance-Zonen bei {selected_asset}.", "100% Size" if mci >= 70 else "75% Size"
else: bias, rule, pos_size = "🟡 NEUTRAL / RANGE", "Handle Vor-Börsen-Extrema oder Range-Rotationen.", "50% Size"

if current_vola >= vola_limit: pos_size = "FLAT / Max 25% Size"

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Handelsrichtung", bias)
col_b2.metric("Positionsgröße", pos_size)
col_b3.metric("Fokus", "Trend-Follow" if abs(score - 50) > 15 else "Mean-Reversion")
st.info(f"**Regel:** {rule}")

# ==========================================
# NEU: ATR ERSCHÖPFUNGS-MODUL
# ==========================================
st.markdown("---")
st.markdown("### 📏 ATR-Erschöpfung (Tages-Range vs. 14D-ATR)")

atr_val = heute.get("ATR_14", 1.0)
range_val = heute.get("Daily_Range", 0.0)
atr_pct = (range_val / atr_val * 100) if atr_val > 0 else 0.0

col_atr1, col_atr2, col_atr3 = st.columns(3)
col_atr1.metric("Ø Schwankungsbreite (14D-ATR)", f"{atr_val:.1f} Pkt.")
col_atr2.metric("Heutige Spanne (High-Low)", f"{range_val:.1f} Pkt.")
col_atr3.metric("Ausschöpfung (Intraday)", f"{atr_pct:.1f}%", delta=f"{100-atr_pct:.1f}% Rest" if atr_pct<100 else "Überdehnt", delta_color="normal" if atr_pct<100 else "inverse")

if atr_pct >= 100:
    st.error("🔴 **ATR Erschöpft:** Die heutige Handelsspanne hat die typische Tagesreichweite überschritten. Breakout-Trades haben ein miserables CRV. Fokus ausschließlich auf Mean-Reversion Setups!")
elif atr_pct >= 80:
    st.warning("🟡 **ATR Warnung:** Der Markt hat über 80% seiner typischen Reichweite abgefahren. Hohe Gefahr von Intraday-Erschöpfung.")
else:
    st.success("🟢 **ATR Intakt:** Ausreichend Bewegungspotenzial für Trendfolge-Moves und Breakouts vorhanden.")
    
st.progress(min(atr_pct / 100.0, 1.0))

# --- GOOGLE TRENDS ---
st.markdown("---")
st.subheader("🌐 Retail Sentiment (Google Trends)")
contra_score, net_spread, trends_live = fetch_google_trends_sentiment(selected_asset)
col_gt1, col_gt2, col_gt3 = st.columns(3)
with col_gt1: st.metric("Google Retail Score", f"{contra_score} / 100", f"Spread: {net_spread:+.2f} σ", delta_color="inverse")
with col_gt2:
    if contra_score >= 65: st.success("🟢 Panik (Boden-Chance)")
    elif contra_score <= 35: st.error("🔴 Gier (Top-Gefahr)")
    else: st.info("🟡 Rauschen (Neutral)")
with col_gt3:
    cfg_gt = TREND_KEYWORD_MAP.get(selected_asset, TREND_KEYWORD_MAP["S&P 500"])
    st.markdown(f"**Bull:** {', '.join(cfg_gt['bull'])}<br>**Bear:** {', '.join(cfg_gt['bear'])}", unsafe_allow_html=True)

# --- TREIBER ANALYSE ---
st.markdown("---")
st.subheader("🔍 Treiber-Analyse (Die 6 Säulen)")

saeulen_details = {
    "Makroekonomie": {"funktion": "Zinsumfeld, Zentralbank-Liquidität & Dollar-Stärke.", "links": [("FRED Liquidity", "https://fred.stlouisfed.org")]},
    "Positionierung": {"funktion": "Institutional Commercial Positionierung & Retail-Sentiment.", "links": [("CoT Report", "https://www.cftc.gov")]},
    "Marktinterna": {"funktion": "Marktbreite, Volumen-Strom und implizite Volatilität.", "links": [("VIX", "https://finance.yahoo.com/quote/%5EVIX")]},
    "Technischer_Trend": {"funktion": "Gleitende Durchschnitte, RSI & Trendstruktur.", "links": []},
    "Fundamentale_Faktoren": {"funktion": "Bewertungen & Rohstoff-Lagerbestände.", "links": []},
    "Fruehwarnindikatoren": {"funktion": "Kreditrisiko (High-Yield Spreads), Bond-Stress (MOVE) & Tail-Risk Hedging (VVIX).", "links": [("Yahoo VVIX", "https://finance.yahoo.com/quote/%5EVVIX")]}
}

cols = st.columns(3)
for i, s_name in enumerate([c for c in df_dash.columns if c.startswith("Saeule_")]):
    val, raw_name = heute[s_name], s_name.replace("Saeule_", "")
    with cols[i % 3]:
        st.metric(label=f"{raw_name.replace('_', ' ')} {'🟢' if val>60 else '🔴' if val<40 else '🟡'}", value=f"{val:.1f}")
        if raw_name in saeulen_details:
            with st.expander("Details"):
                st.markdown(f"**Funktion:** {saeulen_details[raw_name]['funktion']}")
                for l, u in saeulen_details[raw_name]['links']: st.markdown(f"• [{l}]({u})")

# --- CHECKLISTE ---
st.markdown("---")
st.subheader("⚡ Intraday Execution Checkliste")

wochentag_index = pd.Timestamp.now(tz='Europe/Berlin').weekday()
ist_hexensabbat = (pd.Timestamp.now().month in [3, 6, 9, 12] and wochentag_index == 4 and 15 <= pd.Timestamp.now().day <= 21)

col_c1, col_c2 = st.columns(2)
with col_c1:
    c1 = st.checkbox("Trendkonformität (Marktstruktur intakt)", value=bool(heute.get("Saeule_Technischer_Trend", 50) > 55))
    c2 = st.checkbox("Kein akuter Bond-/Vola-Stress (MOVE/VVIX stabil)", value=bool(heute.get("Saeule_Fruehwarnindikatoren", 50) > 35))
    c3 = st.checkbox("Makro-Umgebung stützt die Richtung", value=bool(heute.get("Saeule_Makroekonomie", 50) > 50))
    c4 = st.checkbox("Statistisches Tagesprofil beachtet", value=True)
with col_c2:
    c5 = st.checkbox("Keine News in den nächsten 60 Minuten", value=True)
    c6 = st.checkbox("Kein Hexensabbat / Optionen-Verfall", value=not ist_hexensabbat)
    c7 = st.checkbox("CRV > 1:2 zum nächsten Ziel", value=True)
    # NEU: ATR Filter in Checkliste
    c8 = st.checkbox(f"Tages-ATR noch nicht erschöpft (< 85%)", value=bool(atr_pct < 85))

erfuellte_kriterien = sum([c1, c2, c3, c4, c5, c6, c7, c8])
st.progress(erfuellte_kriterien / 8.0)
st.caption(f"✅ **{erfuellte_kriterien} von 8 Kriterien erfüllt**")

if erfuellte_kriterien == 8 and score > 55: st.success("🟢 **EXECUTION GO**: Alle Filter grün.")
elif score < 40 or atr_pct >= 100: st.error("🔴 **STOP / KEIN TRADE**: Defense-Regime oder ATR ausgereizt.")
else: st.warning("🟡 **CAUTION**: Gemischte Signale. Reduzierte Size.")

# --- CHART ---
st.markdown("---")
st.subheader("📈 Regime-Historie & Preis")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_dash.index[-252:], y=df_dash["Final_Regime_Score"].iloc[-252:], name="Score", fill='tozeroy'), secondary_y=False)
fig.add_trace(go.Scatter(x=df_dash.index[-252:], y=df_dash["Asset_Price"].iloc[-252:], name="Preis", line=dict(color='green')), secondary_y=True)
fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
