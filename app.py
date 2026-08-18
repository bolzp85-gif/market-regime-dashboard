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
        "invert_inverts": ["vix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "fed_policy", "real_yields"],
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
        "invert_inverts": ["vix_score", "pe_valuation", "credit_spreads", "move_index", "usd_index", "fed_policy", "real_yields"],
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
        "invert_inverts": ["vix_score", "usd_index", "real_yields", "fed_policy"],
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
        "invert_inverts": ["vix_score", "usd_index", "inventories"],
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
    "Fruehwarnindikatoren": {"credit_spreads": 0.60, "move_index": 0.40}
}

LOOKBACK_CONFIG = {
    "fed_policy": 1260, "real_yields": 756, "net_liquidity": 756,
    "credit_spreads": 756, "usd_index": 504
}

VOLA_THRESHOLDS = {
    "S&P 500": 30.0,
    "Gold (XAU/USD)": 25.0,
    "WTI Crude Oil": 45.0
}

# ==========================================
# 2. GOOGLE TRENDS NET-SENTIMENT SPREAD ENGINE
# ==========================================

# Asset-spezifische Konfiguration: Region & Keyword-Sets (Max 4 Keywords pro Asset)
TREND_KEYWORD_MAP = {
    "S&P 500": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["buy stocks", "buy the dip"],        
        "bear": ["stock market crash", "recession"]  
    },
    "Nasdaq 100": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["tech stocks", "buy the dip"],       
        "bear": ["market crash", "tech bubble"]       
    },
    "Gold (XAU/USD)": {
        "geo": "DE",
        "lang": "de-DE",
        "bull": ["Gold kaufen", "Goldmünzen"],        
        "bear": ["Gold verkaufen", "Altgold"]         
    },
    "WTI Crude Oil": {
        "geo": "DE",
        "lang": "de-DE",
        "bull": ["Heizöl kaufen", "Spritpreise"],     
        "bear": ["Ölpreis crash", "Öl verkaufen"]     
    }
}

@st.cache_data(ttl=21600)  # 6-Stunden-Cache (Schützt vor Google IP-Sperren)
def fetch_google_trends_sentiment(asset_name: str):
    """
    Holt dynamisch US- oder DE-Trends-Daten, berechnet den aggregierten Z-Score
    und liefert den skalierten Kontraindikator-Score (0-100).
    """
    # Standard-Fallback auf S&P 500, falls Asset nicht in Map
    cfg = TREND_KEYWORD_MAP.get(asset_name, TREND_KEYWORD_MAP["S&P 500"])
    
    geo_loc = cfg["geo"]
    lang_loc = cfg["lang"]
    bull_kws = cfg["bull"]
    bear_kws = cfg["bear"]
    all_kws = bull_kws + bear_kws

    try:
        pytrends = TrendReq(hl=lang_loc, tz=360)
        # timeframe='today 3-m' erzwingt tägliche Datenpunkte (wichtig für die Reaktionszeit!)
        pytrends.build_payload(all_kws, timeframe='today 3-m', geo=geo_loc)
        df_trends = pytrends.interest_over_time()
        
        if df_trends.empty:
            return 50.0, 0.0, False

        # Google's temporäre 'isPartial'-Spalte entfernen, um Berechnungsfehler zu vermeiden
        if 'isPartial' in df_trends.columns:
            df_trends = df_trends.drop(columns=['isPartial'])

        # Z-Score Berechnung (Rolling 21 Tage = ca. 1 Handelsmonat)
        def calc_z(series):
            mean = series.rolling(21, min_periods=5).mean()
            std = series.rolling(21, min_periods=5).std().replace(0, 1e-8)
            return (series - mean) / std

        valid_bull = [kw for kw in bull_kws if kw in df_trends.columns]
        valid_bear = [kw for kw in bear_kws if kw in df_trends.columns]

        if not valid_bull or not valid_bear:
            return 50.0, 0.0, False

        # Gemittelte Z-Scores der jeweiligen Wortgruppen
        z_bull = sum(calc_z(df_trends[kw]) for kw in valid_bull) / len(valid_bull)
        z_bear = sum(calc_z(df_trends[kw]) for kw in valid_bear) / len(valid_bear)

        net_spread = z_bull - z_bear
        latest_spread = float(net_spread.dropna().iloc[-1])

        # Score-Berechnung: Invertiert! Hoher Spread (Gier) = Niedriger Score (Top-Gefahr)
        contrarian_score = float(np.clip(50.0 - (latest_spread * 15.0), 0.0, 100.0))

        return round(contrarian_score, 1), round(latest_spread, 2), True

    except Exception as e:
        # Fallback bei Verbindungsproblemen zu Google
        return 50.0, 0.0, False

# ==========================================
# 2.1 MATHEMATICAL CORE ENGINES & HELPERS
# ==========================================

def strip_timezone(datetime_index_or_series):
    """Sicheres Entfernen von Zeitzonen unabhängig von der Pandas-Version."""
    dt = pd.to_datetime(datetime_index_or_series)
    if hasattr(dt, 'dt'):
        if dt.dt.tz is not None:
            return dt.dt.tz_convert(None)
        return dt
    else:
        if dt.tz is not None:
            return dt.tz_convert(None)
        return dt

def normalize_to_percentile(series: pd.Series, lookback: int = 252, invert: bool = False) -> pd.Series:
    clean_series = series.ffill().bfill()
    if clean_series.isna().all():
        return pd.Series(50.0, index=series.index)

    rolling_mean = clean_series.rolling(window=lookback, min_periods=20).mean()
    rolling_std = clean_series.rolling(window=lookback, min_periods=20).std().replace(0, 1e-8)
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
    s.index = strip_timezone(s.index).floor('D')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    
    clean_target = strip_timezone(target_index).floor('D')
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://edition.cnn.com/"
    }
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            hist = data.get("fear_and_greed_historical", {}).get("data", [])
            if hist:
                df_hist = pd.DataFrame(hist)
                df_hist["Date"] = strip_timezone(pd.to_datetime(df_hist["x"], unit="ms")).dt.floor('D')
                df_hist = df_hist.drop_duplicates(subset=["Date"], keep="last")
                return df_hist.set_index("Date")["y"].sort_index(), True
    except Exception:
        pass
    return 55.0, False

@st.cache_data(ttl=86400)
def fetch_cot_data(asset_search_string):
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
                    rows = df_yr[df_yr.iloc[:, 0].astype(str).str.contains(asset_search_string, case=False, na=False)]
                    if not rows.empty:
                        frames.append(rows)
        except Exception:
            pass

    if frames:
        try:
            df_all = pd.concat(frames, ignore_index=True)
            date_col = [c for c in df_all.columns if 'As_of_Date' in str(c)][0]
            long_col = [c for c in df_all.columns if 'Comm_Positions_Long_All' in str(c)][0]
            short_col = [c for c in df_all.columns if 'Comm_Positions_Short_All' in str(c)][0]

            dates = strip_timezone(pd.to_datetime(df_all[date_col].astype(str), format='%Y%m%d', errors='coerce'))
            df_all['Date'] = dates.dt.floor('D')
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
def fetch_multi_asset_data(selected_asset):
    cfg = ASSET_CONFIGS[selected_asset]
    feed_status = {}
    
    tickers = {
        "asset": cfg["ticker"], "vix": cfg["volatility_ticker"],
        "dxy": "DX=F", "move": "^MOVE", "hyg": "HYG", "lqd": "LQD"
    }
    
    data = yf.download(list(tickers.values()), period="5y", interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            close_data = data["Close"]
            vol_data = data["Volume"] if "Volume" in data.columns.levels[0] else None
        elif "Close" in data.columns.levels[1]:
            close_data = data.xs("Close", axis=1, level=1)
            vol_data = data.xs("Volume", axis=1, level=1) if "Volume" in data.columns.levels[1] else None
        else:
            close_data = data
            vol_data = None
    else:
        close_data = data
        vol_data = None
        
    close_data = close_data.rename(columns={v: k for k, v in tickers.items()}).ffill().bfill().dropna(how='all')
    feed_status["yFinance (Preis & Tech)"] = not close_data.empty
    
    if close_data.empty or "asset" not in close_data.columns:
        return pd.DataFrame(), feed_status
    
    df_raw = pd.DataFrame(index=close_data.index)
    price = close_data["asset"]
    
    has_vol = False
    if vol_data is not None:
        if isinstance(vol_data, pd.DataFrame) and cfg["ticker"] in vol_data.columns:
            asset_volume = vol_data[cfg["ticker"]].ffill().bfill()
            has_vol = True
        elif isinstance(vol_data, pd.Series):
            asset_volume = vol_data.ffill().bfill()
            has_vol = True
            
    if not has_vol:
        asset_volume = pd.Series(1000, index=close_data.index)
        
    feed_status["Volumen/Orderflow Feed"] = has_vol
    
    df_raw["distance_50ma"] = ((price - price.rolling(50).mean()) / price.rolling(50).mean()) * 100
    df_raw["distance_200ma"] = ((price - price.rolling(200).mean()) / price.rolling(200).mean()) * 100
    
    delta = price.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df_raw["rsi_momentum"] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-8))))
    
    df_raw["vix_score"] = close_data["vix"] if "vix" in close_data.columns else 20.0
    df_raw["usd_index"] = close_data["dxy"] if "dxy" in close_data.columns else 100.0
    df_raw["move_index"] = close_data["move"] if "move" in close_data.columns else 100.0
    df_raw["credit_spreads"] = close_data["lqd"] / close_data["hyg"] if "lqd" in close_data.columns and "hyg" in close_data.columns else 1.0
    
    df_raw["advance_decline"] = price.pct_change().rolling(20).sum() * 100
    
    conditions = [delta > 0, delta < 0]
    choices = [asset_volume, -asset_volume]
    obv_daily = np.select(conditions, choices, default=0)
    obv = pd.Series(obv_daily, index=price.index).cumsum()
    obv_ema = obv.ewm(span=50).mean()
    df_raw["obv_momentum"] = ((obv - obv_ema) / obv_ema.abs().replace(0, 1e-8)) * 100

    if selected_asset == "S&P 500":
        df_raw["pe_valuation"] = 24.5 

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
            
            if selected_asset == "WTI Crude Oil":
                inv_s = safe_reindex_series(fred.get_series('WCESTUS1'), df_raw.index)
                df_raw["inventories"] = inv_s if inv_s is not None else 500000.0
                
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
    if isinstance(fg_data, pd.Series):
        df_raw["fear_greed"] = safe_reindex_series(fg_data, df_raw.index)
    else:
        df_raw["fear_greed"] = float(fg_data)

    df_norm = pd.DataFrame(index=df_raw.index)
    inverts = cfg["invert_inverts"]
    
    for col in df_raw.columns:
        should_invert = col in inverts
        lb = LOOKBACK_CONFIG.get(col, 252)
        df_norm[col] = normalize_to_percentile(df_raw[col], lookback=lb, invert=should_invert)
        
    df_dashboard = pd.DataFrame(index=df_raw.index)
    df_dashboard["Raw_Volatility"] = df_raw["vix_score"]
    
    active_sub_weights = {**SUB_WEIGHTS_BASE, **cfg["Sub_Gewichte"]}
    
    for saeule, indikatoren in active_sub_weights.items():
        cols = [c for c in indikatoren.keys() if c in df_norm.columns]
        weights = [indikatoren[c] for c in cols]
        if cols and sum(weights) > 0:
            weights_norm = np.array(weights) / np.sum(weights)
            df_dashboard[f"Saeule_{saeule}"] = df_norm[cols].dot(weights_norm)
        
    saeulen_cols = [f"Saeule_{s}" for s in cfg["Saeulen_Gewichte"].keys() if f"Saeule_{s}" in df_dashboard.columns]
    saeulen_weights = [cfg["Saeulen_Gewichte"][s.replace("Saeule_", "")] for s in saeulen_cols]
    
    if sum(saeulen_weights) > 0:
        saeulen_weights_norm = np.array(saeulen_weights) / np.sum(saeulen_weights)
        df_dashboard["Final_Regime_Score"] = df_dashboard[saeulen_cols].dot(saeulen_weights_norm).round(1)
        
        mci_list = []
        for i in range(len(df_dashboard)):
            row_scores = df_dashboard[saeulen_cols].iloc[i].values
            mci_list.append(calculate_mci(row_scores, saeulen_weights_norm))
        df_dashboard["MCI"] = mci_list
    else:
        df_dashboard["Final_Regime_Score"] = 50.0
        df_dashboard["MCI"] = 0.0
        
    df_dashboard["Asset_Price"] = price.ffill().bfill()
    return df_dashboard.dropna(subset=["Final_Regime_Score"]), feed_status

# ==========================================
# 5. STREAMLIT UI & SIDEBAR
# ==========================================

with st.sidebar:
    st.title("⚙️ Multi-Asset Selector")
    selected_asset = st.selectbox("🎯 Asset auswählen", list(ASSET_CONFIGS.keys()), index=0)
    st.markdown("---")
    st.markdown("### 📡 API Live-Feed Monitor")

with st.spinner(f"Lade quantitative Daten für {selected_asset}..."):
    df_dash, feed_status = fetch_multi_asset_data(selected_asset)

# Render Sidebar Live Feeds (Ergänzt)
with st.sidebar:
    for source, is_live in feed_status.items():
        if is_live: 
            st.markdown(f"🟢 **{source}**")
        else: 
            st.markdown(f"⚠️ **{source}** *(Fallback)*")

if df_dash.empty:
    st.error("⚠️ Marktdaten konnten nicht geladen werden. Yahoo Finance ist möglicherweise nicht erreichbar.")
    st.stop()

heute = df_dash.iloc[-1]

# Deltas berechnen
try:
    heute['Delta_1D'] = df_dash['Final_Regime_Score'].iloc[-1] - df_dash['Final_Regime_Score'].iloc[-2]
except IndexError:
    heute['Delta_1D'] = 0.0
    
try:
    heute['Delta_1W'] = df_dash['MCI'].iloc[-1] - df_dash['MCI'].iloc[-6]
except IndexError:
    heute['Delta_1W'] = 0.0

st.title("📊 Quant Regime Dashboard")
st.caption(f"Asset: **{selected_asset}** | Stand: {df_dash.index[-1].strftime('%d.%m.%Y')}")
st.markdown("---")

# --- HERO METRICS ---
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Final Regime Score", value=f"{heute['Final_Regime_Score']} / 100", delta=f"{heute['Delta_1D']:+.1f} (Heute)")
with col2:
    st.metric(label="Model Confidence Index (MCI)", value=f"{heute['MCI']}%", delta=f"{heute['Delta_1W']:+.1f} (vs. Vorwoche)", delta_color="off")

st.info(f"**Aktuelles Marktregime ({selected_asset}):** {get_regime_label(heute['Final_Regime_Score'])}")

# --- VOLATILITÄTS-WARNSYSTEM (Ergänzt & Ausführlich) ---
current_vola = heute.get("Raw_Volatility", 20.0)
vola_limit = VOLA_THRESHOLDS.get(selected_asset, 30.0)
vola_ticker = ASSET_CONFIGS[selected_asset]["volatility_ticker"]

if current_vola >= vola_limit:
    st.error(f"🚨 **VOLATILITÄTS-ALARM:** Der {vola_ticker} notiert kritisch hoch bei **{current_vola:.2f}** (Grenzwert: {vola_limit}). Extremes Slippage-Risiko! **Regel: Kein Handel oder maximal 25% Positionsgröße!**")
elif current_vola >= vola_limit * 0.8:
    st.warning(f"⚠️ **Erhöhte Volatilität:** Der {vola_ticker} steht bei **{current_vola:.2f}**. Achte auf saubere Entries und verkleinere ggf. die Positionsgröße.")

# --- INTRADAY TRADING BIAS (Ergänzt) ---
st.markdown("---")
st.markdown("### 🎯 Intraday Trading Bias")
score, mci = heute['Final_Regime_Score'], heute['MCI']

if score >= 60:
    bias, rule = "🟢 BULLISCH (Long Bias)", f"Suche bevorzugt nach Dip-Käufen bei {selected_asset} an dynamischen Support-Zonen (VWAP / EMAs)."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size"
elif score <= 40:
    bias, rule = "🔴 BÄRISCH (Short Bias)", f"Suche bevorzugt nach Short-Setups bei {selected_asset} an Resistance-Zonen."
    pos_size = "100% Standardsize" if mci >= 70 else "75% Size"
else:
    bias, rule = "🟡 NEUTRAL / RANGE", "Keine klare Trendrichtung. Handle Vor-Börsen-Extrema oder Range-Rotationen."
    pos_size = "50% Size (Risiko minimieren)"

if current_vola >= vola_limit:
    pos_size = "FLAT / Max 25% Size"

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Handelsrichtung", bias)
col_b2.metric("Positionsgröße", pos_size)
col_b3.metric("Fokus", "Trend-Follow" if abs(score - 50) > 15 else "Mean-Reversion")
st.info(f"**Übergeordnete Regel:** {rule}")

# --- TREIBER-ANALYSE MIT LIVE-LINKS & ROHWERTEN ---
st.markdown("---")
st.subheader("🔍 Treiber-Analyse (Die 6 Säulen)")

saeulen_details = {
    "Makroekonomie": {
        "quelle": "FRED API & Yahoo Finance",
        "funktion": "Zinsumfeld, Zentralbank-Liquidität & Dollar-Stärke.",
        "links": [
            ("FRED: Fed Total Assets (WALCL)", "https://fred.stlouisfed.org/series/WALCL"),
            ("FRED: TGA Account (WTREGEN)", "https://fred.stlouisfed.org/series/WTREGEN"),
            ("FRED: Reverse Repo (RRPONTSYD)", "https://fred.stlouisfed.org/series/RRPONTSYD"),
            ("FRED: 10Y Real Yields (DFII10)", "https://fred.stlouisfed.org/series/DFII10"),
            ("FRED: Fed Funds Rate", "https://fred.stlouisfed.org/series/FEDFUNDS"),
            ("Yahoo: US Dollar Index (DXY)", "https://finance.yahoo.com/quote/DX-Y.NYB")
        ]
    },
    "Positionierung": {
        "quelle": "CFTC CoT-Report & CNN Fear & Greed",
        "funktion": "Institutional Commercial Positionierung & Retail-Sentiment.",
        "links": [
            ("CFTC: CoT Reports Main", "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"),
            ("CNN: Fear & Greed Index", "https://edition.cnn.com/markets/fear-and-greed")
        ]
    },
    "Marktinterna": {
        "quelle": "Yahoo Finance",
        "funktion": "Marktbreite, Volumen-Strom und implizite Volatilität.",
        "links": [
            ("Yahoo: VIX (S&P Volatilität)", "https://finance.yahoo.com/quote/%5EVIX"),
            ("Yahoo: VXN (Nasdaq Volatilität)", "https://finance.yahoo.com/quote/%5EVXN"),
            ("Yahoo: GVZ (Gold Volatilität)", "https://finance.yahoo.com/quote/%5EGVZ"),
            ("Yahoo: OVX (Öl Volatilität)", "https://finance.yahoo.com/quote/%5EOVX")
        ]
    },
    "Technischer_Trend": {
        "quelle": "Yahoo Finance (Preisdaten)",
        "funktion": "Gleitende Durchschnitte, RSI & Trendstruktur.",
        "links": [
            ("Yahoo: Chart & Technicals", f"https://finance.yahoo.com/quote/{ASSET_CONFIGS[selected_asset]['ticker']}")
        ]
    },
    "Fundamentale_Faktoren": {
        "quelle": "FRED API, Multpl & WSJ",
        "funktion": "Bewertungs-KGVs & physikalische Rohstoff-Lagerbestände.",
        "links": [
            ("FRED: Crude Oil Stocks", "https://fred.stlouisfed.org/series/WCESTUS1"),
            ("Multpl: S&P 500 PE Ratio", "https://www.multpl.com/s-p-500-pe-ratio"),
            ("WSJ: P/E Ratios (inkl. Nasdaq)", "https://www.wsj.com/market-data/stocks/peyields")
        ]
    },
    "Fruehwarnindikatoren": {
        "quelle": "Yahoo Finance",
        "funktion": "Kreditrisiko (High-Yield Spreads) & Anleihen-Stress.",
        "links": [
            ("Yahoo: HYG High Yield ETF", "https://finance.yahoo.com/quote/HYG"),
            ("Yahoo: LQD Investment Grade ETF", "https://finance.yahoo.com/quote/LQD"),
            ("Yahoo: MOVE Index (Bond Vola)", "https://finance.yahoo.com/quote/%5EMOVE")
        ]
    }
}

cols = st.columns(3)
saeulen = [c for c in df_dash.columns if c.startswith("Saeule_")]

for i, s_name in enumerate(saeulen):
    val = heute[s_name]
    raw_name = s_name.replace("Saeule_", "")
    label = raw_name.replace("_", " ")
    emoji = "🟢" if val > 60 else "🔴" if val < 40 else "🟡"
    gewichtung = ASSET_CONFIGS[selected_asset]["Saeulen_Gewichte"].get(raw_name, 0) * 100
    
    with cols[i % 3]:
        st.metric(label=f"{label} {emoji}", value=f"{val:.1f}")
        
        if raw_name in saeulen_details:
            details = saeulen_details[raw_name]
            with st.expander("Details, Daten & Links"):
                st.markdown(f"**⚖️ Gewichtung:** {gewichtung:.0f}%")
                st.markdown(f"**⚙️ Funktion:** {details['funktion']}")
                
                # Externe Klick-Links einfügen
                st.markdown("**🔗 Live-Datenquellen öffnen:**")
                for link_title, url in details["links"]:
                    st.markdown(f"• [{link_title}]({url})")
        
        st.markdown("<br>", unsafe_allow_html=True)

# --- INTRADAY EXECUTION CHECKLISTE (MAXIMALE VERSION) ---
st.markdown("---")
st.subheader("⚡ Intraday Execution Checkliste & Filter")

# 1. Sichere Datenextraktion & Typen-Casting
score_gesamt = float(heute.get("Final_Regime_Score", 50))
trend_wert = float(heute.get("Saeule_Technischer_Trend", 50))
vola_wert = float(heute.get("Saeule_Fruehwarnindikatoren", 50))
makro_wert = float(heute.get("Saeule_Makroekonomie", 50))

# 2. Boolesche Logik für die Vorbelegung der Checkboxen
trend_intakt = bool(trend_wert > 55)
kein_bond_stress = bool(vola_wert > 35)
makro_tailwind = bool(makro_wert > 50)

# 3. Dynamische Kalender-Logik (Tagesprofil & Hexensabbat)
heute_datum = pd.Timestamp.now(tz='Europe/Berlin')
wochentag_index = heute_datum.weekday()

# Hexensabbat-Check (3. Freitag in März, Juni, September, Dezember)
ist_hexensabbat = (
    heute_datum.month in [3, 6, 9, 12] and 
    wochentag_index == 4 and 
    15 <= heute_datum.day <= 21
)
# Wenn heute Hexensabbat ist, ist der Haken standardmäßig RAUS (False) zur Warnung
opex_default = not ist_hexensabbat 

wochentag_profile = {
    0: "Montag: Preisfindung & Weekly Initial Balance (Erhöhte Gefahr von False Breakouts & Fake-Moves)",
    1: "Dienstag: Trendetablierung (Statistisch hohe Wahrscheinlichkeit für die Bildung des finalen Wochenhochs/-tiefs)",
    2: "Mittwoch: Trendfortsetzung oder Mid-Week Reversal (Oft Liquiditätsabgriffe & Richtungswechsel)",
    3: "Donnerstag: Momentum & Volatilität (Hohe Wahrscheinlichkeit für starke Trendfortsetzung oder schnelle Reversals)",
    4: "Freitag: Wochenschluss & Profit-Taking (Vorsicht vor erratischen Moves ab US-Mittag, Weekend-Risk & Optionsverfall)"
}
heutiges_profil = wochentag_profile.get(wochentag_index, "Wochenende: Märkte geschlossen")

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("#### 1. Strukturelle Filter")
    c1_val = st.checkbox("Trendkonformität (Preis über relevanten EMAs / Marktstruktur intakt)", value=trend_intakt, key="chk_trend_det")
    c2_val = st.checkbox("Anleihen- & Kreditmärkte stabil (Kein akuter Bond-Stress via MOVE/HYG)", value=kein_bond_stress, key="chk_bond_det")
    c3_val = st.checkbox(f"Makro-Umgebung im Rücken (Liquidität & Zinsen stützen die Richtung - Score: {makro_wert:.0f})", value=makro_tailwind, key="chk_makro_det")
    c4_val = st.checkbox(f"Statistisches Tagesprofil beachtet ({heutiges_profil})", value=True, key="chk_day_profile")

with col_c2:
    st.markdown("#### 2. Timing & Risikomanagement")
    c5_val = st.checkbox("Keine High-Impact News (CPI, FOMC, NFP) in den nächsten 60 Minuten", value=True, key="chk_news_det")
    # Dynamischer Opex-Filter
    c6_val = st.checkbox("Kein Hexensabbat / Ketten-Verfall (Dritter Freitag in März, Juni, Sept., Dez. – Extreme Pinning- & Volatilitätsrisiken beachten)", value=opex_default, key="chk_opex_det")
    c7_val = st.checkbox("CRV (Chance-Risiko-Verhältnis) von mindestens 1:2 zum nächsten charttechnischen Ziel", value=True, key="chk_crv_det")
    c8_val = st.checkbox("US-Eröffnung / Initial Balance abgewartet (Kein Trade direkt um 15:30 Uhr)", value=True, key="chk_time_det")

# 4. Visuelles Feedback: Fortschrittsbalken
st.markdown("<br>", unsafe_allow_html=True)
erfuellte_kriterien = sum([c1_val, c2_val, c3_val, c4_val, c5_val, c6_val, c7_val, c8_val])

st.progress(erfuellte_kriterien / 8.0)
st.caption(f"✅ **{erfuellte_kriterien} von 8 Kriterien erfüllt**")

# 5. Live-Auswertung
alle_kriterien_erfuellt = (erfuellte_kriterien == 8)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Signalausgabe (Konsistenz: Score muss über 55 liegen für einen klaren GO)
if alle_kriterien_erfuellt and score_gesamt > 55:
    st.success("🟢 **EXECUTION FREIGABE (GO)**: Alle Filter grün. Setup entspricht dem definierten Market Regime und den Risikoparametern.")
elif score_gesamt < 40:
    st.error("🔴 **STOP / KEIN TRADE**: Das Marktregime steht auf Defense. Kapitalerhalt hat höchste Priorität.")
else:
    st.warning("🟡 **CAUTION / WARNUNG**: Gemischte Signale. Execution nur mit reduzierter Positionsgröße oder an exakten charttechnischen Extrempunkten.")

# --- HISTORICAL PLOTLY CHART ---
st.markdown("---")
st.subheader("📈 Regime-Historie & Asset Preis (Letzte 12 Monate)")
df_plot = df_dash.tail(252).copy()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=df_plot.index, y=df_plot["Final_Regime_Score"], name="Regime Score (0-100)", 
               fill='tozeroy', marker_color='rgba(0,100,255,0.2)', line_color='blue'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=df_plot.index, y=df_plot["Asset_Price"], name=f"{selected_asset} Preis", 
               line=dict(color='green', width=2)),
    secondary_y=True,
)

fig.update_yaxes(title_text="Regime Score", range=[0, 100], secondary_y=False)
fig.update_yaxes(title_text="Asset Preis", secondary_y=True)
fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# --- SYSTEM & API STATUS FOOTER ---
st.markdown("---")
with st.expander("📡 System & API Status Details"):
    st.write("Live-Verbindungsstatus zu den externen Datenquellen:")
    status_cols = st.columns(2)
    
    for i, (feed, status) in enumerate(feed_status.items()):
        icon = "✅ Verbunden" if status else "⚠️ Fallback aktiv / Offline"
        color = "green" if status else "orange"
        status_cols[i % 2].markdown(f"**{feed}:** :{color}[{icon}]")
        
    st.caption("Das Dashboard nutzt intelligente Fallbacks. Ein Ausfall einzelner APIs (wie CNN oder CFTC) führt nicht zum Absturz, sondern aktiviert statistische Proxies.")
