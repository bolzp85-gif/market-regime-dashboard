import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

import streamlit as st
import yfinance as yf
from fredapi import Fred

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Google Trends: prefer the maintained TrendSpy client; keep pytrends as fallback.
try:
    from trendspy import Trends as TrendSpy
    TRENDSPY_AVAILABLE = True
except ImportError:
    TRENDSPY_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Asset Regime Dashboard v5",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# 1. ASSET CONFIGURATIONS & WEIGHTS
# ============================================================

ASSET_CONFIGS = {

    "S&P 500": {
        "ticker": "^GSPC",
        "volatility_ticker": "^VIX",
        "cot_code": "E-MINI S&P 500",
        "cot_market_code": "13874A",
        "options_proxy": "SPY",
        "options_pc_ticker": "SPY",

        "invert_inverts": [
            "vix_score",
            "pe_valuation",
            "credit_spreads",
            "move_index",
            "usd_index",
            "fed_policy",
            "real_yields"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": .25,
            "Positionierung": .15,
            "Marktinterna": .20,
            "Technischer_Trend": .20,
            "Fundamentale_Faktoren": .10,
            "Fruehwarnindikatoren": .10
        },

        "Sub_Gewichte": {
            "Positionierung": {
                "cot_noncommercials": .50,
                "fear_greed": .50,
                "options_put_call": 0.0
            },
            "Marktinterna": {
                "market_momentum": .50,
                "vix_score": .50
            },
            "Fundamentale_Faktoren": {
                "pe_valuation": 1.0
            }
        }
    },

    "Nasdaq 100": {
        "ticker": "NQ=F",
        "volatility_ticker": "^VXN",
        "cot_code": "NASDAQ-100",
        "cot_market_code": "209742",
        "options_proxy": "QQQ",
        "options_pc_ticker": "QQQ",

        "invert_inverts": [
            "vix_score",
            "pe_valuation",
            "credit_spreads",
            "move_index",
            "usd_index",
            "fed_policy",
            "real_yields"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": .25,
            "Positionierung": .15,
            "Marktinterna": .20,
            "Technischer_Trend": .20,
            "Fundamentale_Faktoren": .10,
            "Fruehwarnindikatoren": .10
        },

        "Sub_Gewichte": {
            "Positionierung": {
                "cot_noncommercials": .50,
                "fear_greed": .50,
                "options_put_call": 0.0
            },
            "Marktinterna": {
                "market_momentum": .50,
                "vix_score": .50
            },
            "Fundamentale_Faktoren": {
                "pe_valuation": 1.0
            }
        }
    },

    "Gold (XAU/USD)": {
        "ticker": "GC=F",
        "volatility_ticker": "^GVZ",
        "cot_code": "GOLD",
        "cot_market_code": "088691",
        "options_proxy": "GLD",
        "options_pc_ticker": "GLD",

        "invert_inverts": [
            "vix_score",
            "usd_index",
            "real_yields",
            "fed_policy"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": .35,
            "Positionierung": .25,
            "Marktinterna": .15,
            "Technischer_Trend": .15,
            "Fundamentale_Faktoren": 0.0,
            "Fruehwarnindikatoren": .10
        },

        "Sub_Gewichte": {
            "Positionierung": {
                "cot_noncommercials": .80,
                "fear_greed": .20,
                "options_put_call": 0.0
            },
            "Marktinterna": {
                "obv_momentum": .50,
                "vix_score": .50
            },
            "Fundamentale_Faktoren": {}
        }
    },

    "WTI Crude Oil": {
        "ticker": "CL=F",
        "volatility_ticker": "^OVX",
        "cot_code": "CRUDE OIL",
        "cot_market_code": "067651",
        "options_proxy": "USO",
        "options_pc_ticker": "USO",

        "invert_inverts": [
            "vix_score",
            "usd_index",
            "inventories"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": .30,
            "Positionierung": .25,
            "Marktinterna": .15,
            "Technischer_Trend": .20,
            "Fundamentale_Faktoren": .10,
            "Fruehwarnindikatoren": 0.0
        },

        "Sub_Gewichte": {
            "Positionierung": {
                "cot_noncommercials": .80,
                "fear_greed": .20,
                "options_put_call": 0.0
            },
            "Marktinterna": {
                "obv_momentum": .50,
                "vix_score": .50
            },
            "Fundamentale_Faktoren": {
                "inventories": 1.0
            }
        }
    },

    "EUR/USD": {
        "ticker": "EURUSD=X",
        "volatility_ticker": "^EVZ",
        "cot_code": "EURO FX",
        "cot_market_code": "099741",
        "options_proxy": "FXE",
        "options_pc_ticker": "FXE",

        "invert_inverts": [
            "vix_score",
            "fed_policy",
            "real_yields",
            "usd_index"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": .35,
            "Positionierung": .20,
            "Marktinterna": .15,
            "Technischer_Trend": .20,
            "Fundamentale_Faktoren": 0.0,
            "Fruehwarnindikatoren": .10
        },

        "Sub_Gewichte": {
            "Positionierung": {
                "cot_noncommercials": .70,
                "fear_greed": .30,
                "options_put_call": 0.0
            },
            "Marktinterna": {
                "market_momentum": .50,
                "vix_score": .50
            },
            "Fundamentale_Faktoren": {}
        }
    }
}


# ============================================================
# 1A. ASSET-REGELN & BEZUGSQUELLEN
# ============================================================

ASSET_RULES = {

    "S&P 500": {
        "profil": "US-Aktienindex / Large Caps",
        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Net Liquidity",
            "Positionierung: CFTC Non-Commercials + CNN Fear & Greed",
            "Marktinterna: 20-Tage-Momentum + VIX",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: historisches S&P-500-KGV als Bewertungsfilter",
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE"
        ],
        "quellen": [
            "Yahoo Finance: ^GSPC, ^VIX, DX-Y.NYB, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders (Non-Commercial Net)",
            "CNN: Fear & Greed",
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS",
            "Multpl: S&P-500-KGV"
        ]
    },

    "Nasdaq 100": {
        "profil": "US-Technologie-/Growth-Index",
        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Net Liquidity",
            "Positionierung: CFTC Non-Commercials + CNN Fear & Greed",
            "Marktinterna: 20-Tage-Momentum + VXN",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: derzeit neutraler Bewertungsblock",
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE"
        ],
        "quellen": [
            "Yahoo Finance: NQ=F, ^VXN, DX-Y.NYB, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders (Non-Commercial Net)",
            "CNN: Fear & Greed",
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS"
        ]
    },

    "Gold (XAU/USD)": {
        "profil": "Gold / Edelmetall",
        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Net Liquidity",
            "Positionierung: CFTC Non-Commercials mit höherem Gewicht",
            "Marktinterna: OBV-Momentum + GVZ",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamentale Säule: bewusst mit 0 % gewichtet",
            "Frühwarnung: Kreditproxy + MOVE"
        ],
        "quellen": [
            "Yahoo Finance: GC=F, ^GVZ, DX-Y.NYB, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders (Non-Commercial Net)",
            "FRED: Realrenditen, Fed Funds, Liquidität",
            "Yahoo Finance / ETF-Optionen: GLD"
        ]
    },

    "WTI Crude Oil": {
        "profil": "WTI-Rohöl",
        "regeln": [
            "Makro: USD und Liquiditäts-/Zinsumfeld",
            "Positionierung: CFTC Non-Commercials mit höherem Gewicht",
            "Marktinterna: OBV-Momentum + OVX",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: US-Rohöllagerbestände (WCESTUS1)",
            "Frühwarnindikatoren: in diesem Asset mit 0 % gewichtet"
        ],
        "quellen": [
            "Yahoo Finance: CL=F, ^OVX, DX-Y.NYB, HYG, LQD",
            "CFTC: Commitment of Traders (Non-Commercial Net)",
            "FRED: WCESTUS1 und Makrodaten",
            "Yahoo Finance / ETF-Optionen: USO"
        ]
    },

    "EUR/USD": {
        "profil": "Devisenpaar Euro gegen US-Dollar",
        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Liquidität",
            "Positionierung: CFTC EURO FX Non-Commercials",
            "Marktinterna: 20-Tage-Momentum + EVZ",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamentale Säule: bewusst mit 0 % gewichtet",
            "Frühwarnung: Kreditproxy + MOVE"
        ],
        "quellen": [
            "Yahoo Finance: EURUSD=X, ^EVZ, DX-Y.NYB, ^MOVE, HYG, LQD",
            "CFTC: EURO FX Commitment of Traders",
            "FRED: Fed Funds, Realrenditen und Liquiditätsdaten",
            "Yahoo Finance / ETF-Optionen: FXE"
        ]
    }
}


# ============================================================
# 2. BASIS-GEWICHTUNGEN / LOOKBACKS
# ============================================================

SUB_WEIGHTS_BASE = {

    "Makroökonomie": {
        "fed_policy": .20,
        "real_yields": .30,
        "usd_index": .20,
        "net_liquidity": .30
    },

    "Technischer_Trend": {
        "distance_200ma": .35,
        "distance_50ma": .35,
        "rsi_momentum": .30
    },

    "Fruehwarnindikatoren": {
        "credit_spreads": .60,
        "move_index": .40
    }
}


LOOKBACK_CONFIG = {
    "fed_policy": 1260,
    "real_yields": 756,
    "net_liquidity": 756,
    "credit_spreads": 756,
    "usd_index": 504,
    "inventories": 756,
    "pe_valuation": 756
}


VOLA_THRESHOLDS = {
    "S&P 500": 30.0,
    "Nasdaq 100": 35.0,
    "Gold (XAU/USD)": 25.0,
    "WTI Crude Oil": 45.0,
    "EUR/USD": 15.0
}


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
    },

    "EUR/USD": {
        "geo": "DE",
        "lang": "de-DE",
        "bull": ["Euro kaufen", "EUR USD kaufen"],
        "bear": ["Euro verkaufen", "EUR USD verkaufen"]
    }
}


# ============================================================
# 3. GOOGLE TRENDS
# ============================================================

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_google_trends_sentiment(asset_name):
    cfg = TREND_KEYWORD_MAP.get(asset_name, TREND_KEYWORD_MAP["S&P 500"])
    kws = list(dict.fromkeys(cfg["bull"] + cfg["bear"]))

    if not kws:
        return 50.0, 0.0, False

    d = None

    if TRENDSPY_AVAILABLE:
        try:
            tr = TrendSpy()
            d = tr.interest_over_time(
                kws,
                timeframe="today 3-m",
                geo=cfg["geo"]
            )
        except Exception:
            d = None

    if d is None and PYTRENDS_AVAILABLE:
        try:
            p = TrendReq(
                hl=cfg["lang"],
                tz=360,
                retries=1,
                backoff_factor=0.5,
                timeout=(10, 30)
            )
            p.build_payload(
                kws,
                timeframe="today 3-m",
                geo=cfg["geo"]
            )
            d = p.interest_over_time()
        except Exception:
            d = None

    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return 50.0, 0.0, False

    try:
        d = d.copy()
        if "isPartial" in d.columns:
            d = d.drop(columns="isPartial")
        if "is_partial" in d.columns:
            d = d.drop(columns="is_partial")

        bull_cols = [x for x in cfg["bull"] if x in d.columns]
        bear_cols = [x for x in cfg["bear"] if x in d.columns]
        if not bull_cols or not bear_cols:
            return 50.0, 0.0, False

        def z(s):
            s = pd.to_numeric(s, errors="coerce")
            m = s.rolling(21, min_periods=5).mean()
            sd = s.rolling(21, min_periods=5).std()
            sd = sd.replace(0, np.nan)
            return (s - m) / sd

        bull = sum(z(d[x]) for x in bull_cols) / len(bull_cols)
        bear = sum(z(d[x]) for x in bear_cols) / len(bear_cols)
        spread = (bull - bear).replace([np.inf, -np.inf], np.nan).dropna()
        if spread.empty:
            return 50.0, 0.0, False

        latest = float(spread.iloc[-1])
        score = float(np.clip(50 - latest * 15, 0, 100))
        return round(score, 1), round(latest, 2), True
    except Exception:
        return 50.0, 0.0, False


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def strip_timezone(x):
    dt = pd.to_datetime(x, errors="coerce")

    if isinstance(dt, pd.Series):
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(None)
        return dt

    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is not None:
            return dt.tz_convert(None)
        return dt

    return dt


def normalize_to_percentile(
    series,
    lookback=252,
    invert=False
):
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=float)

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    if s.dropna().empty:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float
        )

    s = s.ffill().bfill()

    m = s.rolling(
        lookback,
        min_periods=max(20, min(60, lookback // 4))
    ).mean()

    sd = s.rolling(
        lookback,
        min_periods=max(20, min(60, lookback // 4))
    ).std()

    sd = sd.replace(0, np.nan)

    z = (s - m) / sd

    out = pd.Series(
        norm.cdf(z) * 100,
        index=series.index,
        dtype=float
    )

    if invert:
        out = 100 - out

    return (
        out
        .replace([np.inf, -np.inf], np.nan)
        .clip(0, 100)
    )


def calculate_mci(scores, weights, coverage=None):
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)

    if s.shape != w.shape:
        return 0.0

    valid = (
        np.isfinite(s)
        & np.isfinite(w)
        & (w > 0)
    )

    if coverage is not None:
        c = np.asarray(coverage, dtype=float)
        if c.shape != s.shape:
            return 0.0
        c = np.clip(c, 0, 100)
        valid &= np.isfinite(c) & (c > 0)
    else:
        c = np.full_like(s, 100.0)

    s = s[valid]
    w = w[valid]
    c = c[valid]

    if len(s) == 0 or w.sum() <= 0:
        return 0.0

    effective_w = w * (c / 100.0)

    if effective_w.sum() <= 0:
        return 0.0

    effective_w = effective_w / effective_w.sum()

    mean = np.average(s, weights=effective_w)
    sd = np.sqrt(
        np.average(
            (s - mean) ** 2,
            weights=effective_w
        )
    )

    if len(s) < 2:
        return 50.0

    if np.isclose(sd, 0.0, atol=1e-9):
        consistency = 50.0 if np.isclose(mean, 50.0) else 100.0
    else:
        consistency = 100 * (1 - sd / 50)

    return round(
        float(np.clip(consistency, 0, 100)),
        1
    )


def get_regime_label(score):
    if score >= 90:
        return "🟢 Risk-On (Extrem Bullisch)"
    if score >= 75:
        return "🟢 Expansion (Bullisch)"
    if score >= 60:
        return "🟡 Übergangsphase (Leicht Bullisch)"
    if score >= 40:
        return "🟡 Neutral"
    if score >= 25:
        return "🟠 Risk-Off (Bärisch)"
    return "🔴 Stressphase (Stark Bärisch)"


def safe_reindex_series(source, target):
    if not isinstance(source, pd.Series) or source.empty:
        return None

    s = source.copy()

    s.index = (
        strip_timezone(s.index)
        .floor("D")
    )

    s = (
        s[
            ~s.index.duplicated(keep="last")
        ]
        .sort_index()
    )

    t = (
        strip_timezone(target)
        .floor("D")
    )

    r = (
        s.reindex(
            t,
            method="ffill"
        )
        .ffill()
        .bfill()
    )

    r.index = target

    return r


def extract_yfinance_field(data, field):
    if data is None or data.empty:
        return None

    if not isinstance(data.columns, pd.MultiIndex):
        if field in data.columns:
            x = data[field]
            return x if isinstance(x, pd.Series) else None
        return None

    level0 = data.columns.get_level_values(0)
    level1 = data.columns.get_level_values(1)

    if field in level0:
        x = data[field]
        return x if isinstance(x, pd.DataFrame) else None

    if field in level1:
        x = data.xs(
            field,
            axis=1,
            level=1
        )
        return x if isinstance(x, pd.DataFrame) else None

    return None


def flatten_yfinance_columns(frame):
    if frame is None:
        return None

    x = frame.copy()

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(-1)

    return x


def add_feed_status(status, name, ok):
    status[name] = bool(ok)


def neutralize_missing_columns(df, columns):
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan


def _find_cftc_column(df, candidates):
    """Helper to resolve column name variations in CFTC API response."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_pillar_score(
    norm_df,
    raw_df,
    weights,
    pillar_name
):
    """Vektorisierte Säulen- und Coverage-Berechnung via Matrixmultiplikation."""
    cols = [c for c in weights if c in norm_df.columns]

    if not cols:
        return (
            pd.Series(50.0, index=raw_df.index),
            pd.Series(0.0, index=raw_df.index)
        )

    w_series = pd.Series({c: weights[c] for c in cols}, dtype=float)
    total_weight = float(w_series.sum())

    if total_weight <= 0:
        return (
            pd.Series(50.0, index=raw_df.index),
            pd.Series(0.0, index=raw_df.index)
        )

    sub_df = norm_df[cols].astype(float)
    valid_mask = sub_df.notna()

    avail_weight = valid_mask.dot(w_series)
    weighted_sum = sub_df.fillna(0.0).dot(w_series)

    score = weighted_sum.div(avail_weight.replace(0, np.nan)).fillna(50.0).clip(0, 100)
    coverage = (avail_weight / total_weight * 100).fillna(0.0).clip(0, 100)

    return score, coverage


def calculate_model_confidence(
    feed_status,
    selected_asset,
    df
):
    checks = [
        bool(feed_status.get("yFinance (Preis & Tech)", False)),
        bool(feed_status.get("FRED API (Makro & Fed)", False)),
        bool(any("CFTC COT" in k and v for k, v in feed_status.items()))
    ]

    supplemental = [
        feed_status.get("CNN Fear & Greed", False),
        feed_status.get(
            f"Options Put/Call ({ASSET_CONFIGS[selected_asset]['options_pc_ticker']})",
            False
        )
    ]

    base = (sum(checks) / len(checks)) * 75.0
    supplement = (sum(supplemental) / len(supplemental)) * 25.0

    latest_coverage = 0.0
    if df is not None and not df.empty and 'Model_Data_Coverage' in df.columns:
        latest_coverage = float(
            pd.to_numeric(df['Model_Data_Coverage'].iloc[-1], errors='coerce')
        )
        if not np.isfinite(latest_coverage):
            latest_coverage = 0.0

    confidence = round(
        float((base + supplement) * (latest_coverage / 100.0)),
        1
    )

    if confidence >= 85:
        label = "🟢 Hoch"
    elif confidence >= 65:
        label = "🟡 Mittel"
    else:
        label = "🔴 Niedrig"

    return confidence, label


# ============================================================
# 5. CNN FEAR & GREED
# ============================================================

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/125 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://edition.cnn.com/"
            },
            timeout=12
        )
        r.raise_for_status()

        payload = r.json()
        historical = payload.get("fear_and_greed_historical", {}).get("data", [])
        d = pd.DataFrame(historical)

        if d.empty or not {"x", "y"}.issubset(d.columns):
            return pd.Series(dtype=float), False

        d["Date"] = (
            pd.to_datetime(d["x"], unit="ms", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )
        d["y"] = pd.to_numeric(d["y"], errors="coerce")

        d = (
            d.dropna(subset=["Date", "y"])
            .drop_duplicates("Date", keep="last")
            .sort_values("Date")
        )

        if d.empty:
            return pd.Series(dtype=float), False

        return d.set_index("Date")["y"], True

    except Exception:
        return pd.Series(dtype=float), False


# ============================================================
# 6. CFTC COT – OFFIZIELLE SOCRATA API + FRESHNESS-GATE
# ============================================================

CFTC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; QuantRegimeDashboard/5.1)",
    "Accept": "application/json,text/plain,*/*"
}

CFTC_LEGACY_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_MAX_AGE_DAYS = 12


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cot_data(asset_search_string, market_code=None):
    code = str(market_code or "").strip()
    if not code:
        return pd.Series(dtype=float), False, "No CFTC market code"

    params = {
        "cftc_contract_market_code": code,
        "$limit": 300,
        "$order": "report_date_as_yyyy_mm_dd DESC",
    }

    try:
        r = requests.get(
            CFTC_LEGACY_API,
            params=params,
            headers=CFTC_HEADERS,
            timeout=20
        )
        r.raise_for_status()
        payload = r.json()
        d = pd.DataFrame(payload)

        if d.empty:
            return pd.Series(dtype=float), False, "CFTC API returned no rows"

        date_col = _find_cftc_column(
            d,
            ["report_date_as_yyyy_mm_dd", "Report_Date_as_YYYY_MM_DD"]
        )
        long_col = _find_cftc_column(
            d,
            ["noncomm_positions_long_all", "NonComm_Positions_Long_All"]
        )
        short_col = _find_cftc_column(
            d,
            ["noncomm_positions_short_all", "NonComm_Positions_Short_All"]
        )

        if date_col is None or long_col is None or short_col is None:
            return pd.Series(dtype=float), False, "CFTC Non-Commercial fields changed"

        d["Date"] = pd.to_datetime(d[date_col], errors="coerce")
        d["Net_NonCommercials"] = (
            pd.to_numeric(d[long_col], errors="coerce")
            - pd.to_numeric(d[short_col], errors="coerce")
        )

        d = (
            d.dropna(subset=["Date", "Net_NonCommercials"])
             .drop_duplicates("Date", keep="first")
             .sort_values("Date")
        )

        if d.empty:
            return pd.Series(dtype=float), False, "No valid CFTC observations"

        latest_date = d["Date"].max()
        today = pd.Timestamp.now(tz="UTC").floor("D").tz_localize(None)
        age_days = int((today - latest_date.normalize()).days)

        if age_days < 0:
            return (
                pd.Series(dtype=float),
                False,
                f"CFTC report date is in the future ({latest_date:%Y-%m-%d})"
            )

        if age_days > CFTC_MAX_AGE_DAYS:
            return (
                pd.Series(dtype=float),
                False,
                f"CFTC stale: {age_days} days old (max {CFTC_MAX_AGE_DAYS})"
            )

        return (
            d.set_index("Date")["Net_NonCommercials"],
            True,
            f"CFTC API / {code} / {latest_date:%Y-%m-%d} / age {age_days}d"
        )

    except Exception as exc:
        return pd.Series(dtype=float), False, f"CFTC API error: {str(exc)[:90]}"


# ============================================================
# 7. OPTIONS PUT/CALL – AGGREGIERTER ETF-PROXY
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_put_call(ticker, max_expiries=3):
    ticker = str(ticker).strip().upper()
    if not ticker:
        return np.nan, False, "No option ticker"

    url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.yahoo.com/"
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
        result = (payload.get("optionChain", {}).get("result") or [])
        if not result:
            return np.nan, False, "Yahoo option chain empty"

        root = result[0]
        expiries = root.get("expirationDates") or []
        if not expiries:
            return np.nan, False, "No option expiries"

        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        future = []
        for ts in expiries:
            try:
                dt = pd.to_datetime(int(ts), unit="s").normalize()
                if dt >= now:
                    future.append((dt, int(ts)))
            except Exception:
                continue
        future.sort(key=lambda x: x[0])
        selected = future[:max_expiries]
        if not selected:
            return np.nan, False, "No future option expiries"

        total_put = 0.0
        total_call = 0.0
        used = 0

        for _, ts in selected:
            try:
                rr = requests.get(
                    f"{url}?date={ts}",
                    headers=headers,
                    timeout=20
                )
                rr.raise_for_status()
                res = (rr.json().get("optionChain", {}).get("result") or [])
                if not res:
                    continue
                opts = res[0].get("options") or []
                if not opts:
                    continue
                chain = opts[0]

                puts = pd.DataFrame(chain.get("puts") or [])
                calls = pd.DataFrame(chain.get("calls") or [])
                if puts.empty and calls.empty:
                    continue

                pv = pd.to_numeric(puts.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                cv = pd.to_numeric(calls.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                if pv > 0 or cv > 0:
                    total_put += float(pv)
                    total_call += float(cv)
                    used += 1
            except Exception:
                continue

        if used == 0:
            return np.nan, False, "No usable option volume"
        if total_call <= 0:
            return np.nan, False, "No call volume"

        ratio = total_put / total_call
        if not np.isfinite(ratio) or ratio < 0:
            return np.nan, False, "Invalid P/C ratio"

        return float(ratio), True, f"Yahoo Options {ticker} ({used} Expiries)"

    except Exception as exc:
        return np.nan, False, f"Yahoo options error: {str(exc)[:90]}"


# ============================================================
# 8. FRED API KEY
# ============================================================

FRED_API_KEY = ""

try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    pass


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Multi-Asset Selector")

    selected_asset = st.selectbox(
        "🎯 Asset auswählen",
        list(ASSET_CONFIGS),
        index=0
    )

    st.markdown("---")
    st.markdown("### 📡 API Live-Feed Monitor")


# ============================================================
# 10A. S&P 500 PE HISTORIE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_pe_history():
    url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; QuantRegimeDashboard/5.0)"}

    try:
        tables = pd.read_html(
            requests.get(url, headers=headers, timeout=12).text
        )

        for table in tables:
            if table.empty:
                continue

            cols = [str(c).strip().lower() for c in table.columns]
            date_col = next((table.columns[i] for i, c in enumerate(cols) if "date" in c), None)
            value_col = next((table.columns[i] for i, c in enumerate(cols) if c == "value"), None)

            if date_col is None or value_col is None:
                continue

            d = pd.DataFrame({
                "Date": pd.to_datetime(table[date_col], errors="coerce"),
                "PE": pd.to_numeric(
                    table[value_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                    errors="coerce"
                )
            })

            d = (
                d.dropna(subset=["Date", "PE"])
                .query("PE > 0 and PE < 200")
                .drop_duplicates("Date", keep="last")
                .sort_values("Date")
            )

            if len(d) >= 12:
                return d.set_index("Date")["PE"], True

    except Exception:
        pass

    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()

        pattern = re.compile(
            r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}).*?(\d+(?:\.\d+)?)",
            re.S
        )

        rows = []
        for date_text, value_text in pattern.findall(r.text):
            try:
                dt = pd.to_datetime(date_text, errors="coerce")
                value = float(value_text)
                if pd.notna(dt) and 0 < value < 200:
                    rows.append((dt, value))
            except Exception:
                continue

        if not rows:
            return pd.Series(dtype=float), False

        d = pd.DataFrame(rows, columns=["Date", "PE"])
        d = d.drop_duplicates("Date", keep="last").sort_values("Date")

        return d.set_index("Date")["PE"], True

    except Exception:
        return pd.Series(dtype=float), False


ALL_YF_TICKERS = tuple(sorted(set(
    [cfg['ticker'] for cfg in ASSET_CONFIGS.values()]
    + [cfg['volatility_ticker'] for cfg in ASSET_CONFIGS.values()]
    + ['DX-Y.NYB', '^MOVE', 'HYG', 'LQD']
)))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_market_bundle():
    try:
        data = yf.download(
            list(ALL_YF_TICKERS),
            period='5y',
            interval='1d',
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by='column'
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    return data


# ============================================================
# 10B. MULTI-ASSET DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multi_asset_data(selected_asset):
    cfg = ASSET_CONFIGS[selected_asset]
    status = {}

    tickers = {
        'asset': cfg['ticker'],
        'volatility': cfg['volatility_ticker'],
        'dxy': 'DX-Y.NYB',
        'move': '^MOVE',
        'hyg': 'HYG',
        'lqd': 'LQD'
    }

    data = fetch_yfinance_market_bundle()
    if data is None or data.empty:
        return pd.DataFrame(), {'yFinance (Preis & Tech)': False}

    close = extract_yfinance_field(data, "Close")
    if close is None:
        return pd.DataFrame(), {"yFinance (Preis & Tech)": False}

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = flatten_yfinance_columns(close)
    close = close.rename(columns={ticker: name for name, ticker in tickers.items()})
    close = close.apply(pd.to_numeric, errors="coerce").sort_index()

    asset_ok = "asset" in close.columns and not close["asset"].dropna().empty
    add_feed_status(status, "yFinance (Preis & Tech)", asset_ok)

    if not asset_ok:
        return pd.DataFrame(), status

    price = close["asset"].dropna()
    df = pd.DataFrame(index=price.index)

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    vol = extract_yfinance_field(data, "Volume")
    has_volume = False

    if isinstance(vol, pd.Series):
        asset_vol = pd.to_numeric(vol, errors="coerce").reindex(price.index)
        has_volume = asset_vol.notna().any()
    elif isinstance(vol, pd.DataFrame):
        vol = flatten_yfinance_columns(vol)
        vol = vol.rename(columns={ticker: name for name, ticker in tickers.items()})
        if "asset" in vol.columns:
            asset_vol = pd.to_numeric(vol["asset"], errors="coerce").reindex(price.index)
            has_volume = asset_vol.notna().any()

    if not has_volume:
        asset_vol = pd.Series(np.nan, index=price.index, dtype=float)

    add_feed_status(status, "Volumen / Orderflow Feed", has_volume)

    # --------------------------------------------------------
    # TECHNISCHER TREND
    # --------------------------------------------------------

    ma50 = price.rolling(50, min_periods=50).mean()
    ma200 = price.rolling(200, min_periods=200).mean()

    df["distance_50ma"] = ((price - ma50) / ma50.replace(0, np.nan)) * 100
    df["distance_200ma"] = ((price - ma200) / ma200.replace(0, np.nan)) * 100

    # --------------------------------------------------------
    # RSI 14 – Wilder-style EWM
    # --------------------------------------------------------

    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)

    df["rsi_momentum"] = (100 - (100 / (1 + rs))).clip(0, 100)

    # --------------------------------------------------------
    # VOLATILITY / USD / MOVE
    # --------------------------------------------------------

    if "volatility" in close.columns:
        df["vix_score"] = pd.to_numeric(close["volatility"], errors="coerce").reindex(df.index).ffill().bfill()
        vola_ok = df["vix_score"].notna().any()
    else:
        df["vix_score"] = np.nan
        vola_ok = False

    if "dxy" in close.columns:
        df["usd_index"] = pd.to_numeric(close["dxy"], errors="coerce").reindex(df.index).ffill().bfill()
        dxy_ok = df["usd_index"].notna().any()
    else:
        df["usd_index"] = np.nan
        dxy_ok = False

    if "move" in close.columns:
        df["move_index"] = pd.to_numeric(close["move"], errors="coerce").reindex(df.index).ffill().bfill()
        move_ok = df["move_index"].notna().any()
    else:
        df["move_index"] = np.nan
        move_ok = False

    add_feed_status(status, f"{cfg['volatility_ticker']} Volatilität", vola_ok)
    add_feed_status(status, "Yahoo Finance USD Index", dxy_ok)
    add_feed_status(status, "Yahoo Finance MOVE", move_ok)

    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------

    if "lqd" in close.columns and "hyg" in close.columns:
        lqd = pd.to_numeric(close["lqd"], errors="coerce").reindex(df.index).ffill().bfill()
        hyg = pd.to_numeric(close["hyg"], errors="coerce").reindex(df.index).ffill().bfill()
        df["credit_spreads"] = lqd / hyg.replace(0, np.nan)
        credit_ok = df["credit_spreads"].notna().any()
    else:
        df["credit_spreads"] = np.nan
        credit_ok = False

    add_feed_status(status, "LQD/HYG Kreditproxy", credit_ok)

    # --------------------------------------------------------
    # MARKET MOMENTUM
    # --------------------------------------------------------

    df["market_momentum"] = price.pct_change().rolling(20, min_periods=10).sum() * 100

    # --------------------------------------------------------
    # OBV MOMENTUM
    # --------------------------------------------------------

    if has_volume:
        signed_volume = np.where(delta > 0, asset_vol, np.where(delta < 0, -asset_vol, 0.0))
        obv = pd.Series(signed_volume, index=price.index).cumsum()
        obv_ema = obv.ewm(span=50, adjust=False).mean()
        df['obv_momentum'] = ((obv - obv_ema) / obv_ema.abs().replace(0, np.nan)) * 100
    else:
        df['obv_momentum'] = np.nan

    # --------------------------------------------------------
    # FRED
    # --------------------------------------------------------

    fred_ok = False

    if FRED_API_KEY:
        try:
            f = Fred(api_key=FRED_API_KEY)
            wal = safe_reindex_series(f.get_series("WALCL"), df.index)
            tga = safe_reindex_series(f.get_series("WTREGEN"), df.index)
            rrp = safe_reindex_series(f.get_series("RRPONTSYD"), df.index)
            fed = safe_reindex_series(f.get_series("FEDFUNDS"), df.index)
            real_yield = safe_reindex_series(f.get_series("DFII10"), df.index)

            if wal is not None and tga is not None and rrp is not None:
                FRED_MILLIONS_TO_BILLIONS = 1.0 / 1000.0
                df['net_liquidity'] = (
                    wal * FRED_MILLIONS_TO_BILLIONS
                    - tga * FRED_MILLIONS_TO_BILLIONS
                    - rrp
                )
            else:
                df["net_liquidity"] = np.nan

            df["fed_policy"] = fed if fed is not None else np.nan
            df["real_yields"] = real_yield if real_yield is not None else np.nan

            if selected_asset == "WTI Crude Oil":
                inventories = safe_reindex_series(f.get_series("WCESTUS1"), df.index)
                df["inventories"] = inventories if inventories is not None else np.nan

            fred_ok = (
                df["fed_policy"].notna().any()
                and df["real_yields"].notna().any()
                and df["net_liquidity"].notna().any()
            )

        except Exception:
            fred_ok = False

    add_feed_status(status, "FRED API (Makro & Fed)", fred_ok)

    neutral_cols = ["fed_policy", "real_yields", "net_liquidity"]
    if selected_asset == "WTI Crude Oil":
        neutral_cols.append("inventories")

    neutralize_missing_columns(df, neutral_cols)

    # --------------------------------------------------------
    # CFTC COT
    # --------------------------------------------------------

    cot, cot_ok, cot_source = fetch_cot_data(cfg["cot_code"], cfg.get("cot_market_code"))
    add_feed_status(status, f"CFTC COT ({cfg['cot_code']})", cot_ok)

    if cot is not None and not cot.empty:
        df["cot_noncommercials"] = safe_reindex_series(cot, df.index)
    else:
        df["cot_noncommercials"] = np.nan

    # --------------------------------------------------------
    # CNN FEAR & GREED
    # --------------------------------------------------------

    fg, fg_ok = fetch_fear_and_greed()
    add_feed_status(status, "CNN Fear & Greed", fg_ok)

    if isinstance(fg, pd.Series) and not fg.empty:
        df["fear_greed"] = safe_reindex_series(fg, df.index)
    else:
        df["fear_greed"] = np.nan

    # --------------------------------------------------------
    # OPTIONS PUT/CALL PROXY
    # --------------------------------------------------------

    opt_pc, opt_ok, opt_source = fetch_option_put_call(cfg["options_pc_ticker"])
    df["options_put_call"] = float(opt_pc) if np.isfinite(opt_pc) else np.nan
    add_feed_status(status, f"Options Put/Call ({cfg['options_pc_ticker']})", opt_ok)

    # --------------------------------------------------------
    # S&P 500 HISTORICAL PE
    # --------------------------------------------------------

    if selected_asset == "S&P 500":
        pe_series, pe_ok = fetch_sp500_pe_history()
        add_feed_status(status, "Bewertungsdaten (S&P 500 PE)", pe_ok)
        if pe_series is not None and not pe_series.empty:
            df["pe_valuation"] = safe_reindex_series(pe_series, df.index)
        else:
            df["pe_valuation"] = np.nan
    else:
        df["pe_valuation"] = np.nan

    # --------------------------------------------------------
    # NORMALISIERUNG
    # --------------------------------------------------------

    norm_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        norm_df[col] = normalize_to_percentile(
            df[col],
            LOOKBACK_CONFIG.get(col, 252),
            col in cfg["invert_inverts"]
        )

    # --------------------------------------------------------
    # SÄULEN
    # --------------------------------------------------------

    active = {k: v.copy() for k, v in SUB_WEIGHTS_BASE.items()}
    for cat, weights in cfg["Sub_Gewichte"].items():
        active[cat] = weights

    dash = pd.DataFrame(index=df.index)
    dash["Raw_Volatility"] = df["vix_score"]

    pillar_coverages = {}
    for pillar, weights in active.items():
        score_series, coverage_series = build_pillar_score(norm_df, df, weights, pillar)
        dash[f"Saeule_{pillar}"] = score_series
        dash[f"Coverage_{pillar}"] = coverage_series
        pillar_coverages[pillar] = coverage_series

    if not cot_ok and "Saeule_Positionierung" in dash.columns:
        dash["Saeule_Positionierung"] = np.nan
        dash["Coverage_Positionierung"] = 0.0
        pillar_coverages["Positionierung"] = dash["Coverage_Positionierung"]

    # --------------------------------------------------------
    # FINAL REGIME SCORE
    # --------------------------------------------------------

    pillar_cols = []
    pillar_weights = []

    for pillar, weight in cfg["Saeulen_Gewichte"].items():
        col = f"Saeule_{pillar}"
        if col in dash.columns and weight > 0:
            pillar_cols.append(col)
            pillar_weights.append(float(weight))

    w = np.asarray(pillar_weights, dtype=float)
    if len(w) == 0 or w.sum() <= 0:
        return pd.DataFrame(), status

    w = w / w.sum()

    pillar_matrix = dash[pillar_cols].astype(float)
    valid_matrix = pillar_matrix.notna()
    effective_weights = valid_matrix.mul(w, axis=1)
    weight_sum = effective_weights.sum(axis=1)

    weighted_sum = pillar_matrix.fillna(0.0).mul(w, axis=1).sum(axis=1)
    dash["Final_Regime_Score"] = (
        weighted_sum.div(weight_sum.replace(0, np.nan))
        .clip(0, 100)
        .round(1)
    )

    active_pillars = [
        pillar for pillar, weight in cfg['Saeulen_Gewichte'].items()
        if weight > 0 and f'Saeule_{pillar}' in dash.columns
    ]

    dash['MCI'] = [
        calculate_mci(
            dash[pillar_cols].iloc[i].values,
            w,
            np.asarray([
                dash.at[dash.index[i], f'Coverage_{pillar}']
                for pillar in active_pillars
            ], dtype=float)
        )
        for i in range(len(dash))
    ]

    coverage_cols = [
        f"Coverage_{p}" for p in cfg["Saeulen_Gewichte"]
        if f"Coverage_{p}" in dash.columns and cfg["Saeulen_Gewichte"].get(p, 0) > 0
    ]

    if coverage_cols:
        dash["Model_Data_Coverage"] = dash[coverage_cols].mean(axis=1).clip(0, 100).round(1)
    else:
        dash["Model_Data_Coverage"] = 0.0

    dash["Asset_Price"] = price.reindex(dash.index).ffill().bfill()
    dash["Options_Put_Call"] = opt_pc

    return dash.dropna(subset=["Final_Regime_Score"]), status


# ============================================================
# 11. DATEN LADEN
# ============================================================

with st.spinner(f"Lade quantitative Daten für {selected_asset}..."):
    df_dash, feed_status = fetch_multi_asset_data(selected_asset)


with st.sidebar:
    for source, live in feed_status.items():
        st.markdown(
            f"{'🟢' if live else '⚠️'} "
            f"**{source}**"
            f"{' *(Fallback / Offline)*' if not live else ''}"
        )


if df_dash.empty:
    st.error("⚠️ Marktdaten konnten nicht geladen werden.")
    st.stop()


# ============================================================
# 13. MODELLQUALITÄT
# ============================================================

model_confidence, confidence_label = calculate_model_confidence(
    feed_status, selected_asset, df_dash
)

today = df_dash.iloc[-1].copy()

today["Delta_1D"] = (
    df_dash["Final_Regime_Score"].iloc[-1] - df_dash["Final_Regime_Score"].iloc[-2]
    if len(df_dash) >= 2 else 0.0
)

today["Delta_1W"] = (
    df_dash["MCI"].iloc[-1] - df_dash["MCI"].iloc[-6]
    if len(df_dash) >= 6 else 0.0
)


# ============================================================
# 14. HEADER
# ============================================================

st.title("📊 Quant Regime Dashboard")
st.caption(f"Asset: **{selected_asset}** | Stand: {df_dash.index[-1].strftime('%d.%m.%Y')}")
st.markdown("---")


# ============================================================
# 15. FINAL SCORE + MCI + DATA QUALITY
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Final Regime Score",
        f"{today.Final_Regime_Score} / 100",
        f"{today.Delta_1D:+.1f} (Heute)"
    )

with c2:
    st.metric(
        "Model Consistency Index",
        f"{today.MCI}%",
        f"{today.Delta_1W:+.1f} (vs. Vorwoche)",
        delta_color="off"
    )

with c3:
    st.metric(
        "Model Data Coverage",
        f"{today.Model_Data_Coverage:.0f}%",
        confidence_label
    )


st.caption(
    "Der MCI misst die Übereinstimmung der sechs Modell-Säulen. "
    "Bei fehlenden Daten werden neutrale 50er-Werte nicht als "
    "echte Modellkonsistenz gewertet. Die Data Coverage zeigt, "
    "wie viel der aktiv gewichteten Indikator-Gewichte tatsächlich "
    "mit Daten befüllt sind."
)

if model_confidence >= 85:
    st.success(f"📡 **Datenqualität: {confidence_label}** ({model_confidence:.0f}/100)")
elif model_confidence >= 65:
    st.warning(f"📡 **Datenqualität: {confidence_label}** ({model_confidence:.0f}/100)")
else:
    st.error(
        f"📡 **Datenqualität: {confidence_label}** ({model_confidence:.0f}/100) – "
        "Execution-Gate wird eingeschränkt."
    )


st.info(
    f"**Aktuelles Marktregime ({selected_asset}):** "
    f"{get_regime_label(today.Final_Regime_Score)}"
)


# ============================================================
# 16. PUT/CALL RATIO
# ============================================================

st.markdown("---")
st.subheader("📊 Put/Call Ratio – zusätzlicher Positionierungsfilter")

opt_pc = float(today.get("Options_Put_Call", np.nan))

if np.isfinite(opt_pc):
    if opt_pc >= 1.20:
        pc_interpretation = "🟢 Stark Put-lastig"
        pc_bias = "Kontraindikativ bullisch"
    elif opt_pc >= 1.00:
        pc_interpretation = "🟢 Eher Put-lastig"
        pc_bias = "Leicht bullisch"
    elif opt_pc >= 0.80:
        pc_interpretation = "🟡 Neutral"
        pc_bias = "Neutral"
    elif opt_pc >= 0.60:
        pc_interpretation = "🟠 Eher Call-lastig"
        pc_bias = "Leicht bärisch"
    else:
        pc_interpretation = "🔴 Stark Call-lastig"
        pc_bias = "Kontraindikativ bärisch"
else:
    pc_interpretation = "⚪ Keine Daten"
    pc_bias = "Nicht verfügbar"


pc1, pc2, pc3 = st.columns(3)

with pc1:
    st.metric(
        f"Options Put/Call ({ASSET_CONFIGS[selected_asset]['options_proxy']})",
        f"{opt_pc:.2f}" if np.isfinite(opt_pc) else "n/a"
    )

with pc2:
    st.metric("P/C Interpretation", pc_interpretation)

with pc3:
    st.metric("Kontra-Signal", pc_bias)

st.caption(
    "Der PCR wird ausschließlich als zusätzlicher Positionierungs-/Sentimentfilter "
    "angezeigt und verändert den Final Regime Score nicht."
)

with st.expander("ℹ️ PCR-Methodik"):
    st.markdown(
        """
**Put/Call Ratio = Put-Volumen ÷ Call-Volumen**

Die aktuelle Version aggregiert die nächsten bis zu drei
verfügbaren Optionslaufzeiten des jeweiligen ETF-Proxys.

• **> 1,20:** deutlich Put-lastig → kann konträr bullisch sein
• **1,00–1,20:** leicht Put-lastig
• **0,80–1,00:** neutral
• **0,60–0,80:** eher Call-lastig
• **< 0,60:** stark Call-lastig
"""
    )

st.warning(
    "⚠️ Futures selbst besitzen keine Put/Call-Ratio. "
    "ES → SPY, NQ → QQQ, Gold → GLD, WTI → USO und "
    "EUR/USD → FXE sind ausdrücklich Options-Proxys."
)


# ============================================================
# 17. VOLATILITÄTS-ALARM
# ============================================================

current_vola = float(today.get("Raw_Volatility", np.nan))
limit = VOLA_THRESHOLDS.get(selected_asset, 30.0)
vt = ASSET_CONFIGS[selected_asset]["volatility_ticker"]

if np.isfinite(current_vola):
    if current_vola >= limit:
        st.error(f"🚨 **VOLATILITÄTS-ALARM:** {vt} bei **{current_vola:.2f}** (Grenzwert {limit:.1f}).")
    elif current_vola >= limit * .8:
        st.warning(f"⚠️ **Erhöhte Volatilität:** {vt} bei **{current_vola:.2f}**.")
else:
    st.warning(f"⚠️ {vt} aktuell nicht verfügbar.")


# ============================================================
# 18. INTRADAY TRADING BIAS
# ============================================================

st.markdown("---")
st.markdown("### 🎯 Intraday Trading Bias")

score = float(today.Final_Regime_Score)
mci = float(today.MCI)

if score >= 60:
    bias = "🟢 BULLISCH (Long Bias)"
    rule = f"Bevorzugt Long-Setups bei {selected_asset} suchen."
    pos = "100% Standardsize" if mci >= 70 else "75% Size" if mci >= 50 else "50% Size"

elif score <= 40:
    bias = "🔴 BÄRISCH (Short Bias)"
    rule = f"Bevorzugt Short-Setups bei {selected_asset} suchen."
    pos = "100% Standardsize" if mci >= 70 else "75% Size" if mci >= 50 else "50% Size"

else:
    bias = "🟡 NEUTRAL / RANGE"
    rule = "Keine klare Trendrichtung. Nur selektive Setups."
    pos = "50% Size"

if np.isfinite(current_vola) and current_vola >= limit:
    pos = "FLAT / Max 25% Size"

if model_confidence < 65:
    pos = "FLAT / Max 25% Size"

b1, b2, b3 = st.columns(3)

with b1:
    st.metric("Handelsrichtung", bias)

with b2:
    st.metric("Positionsgröße", pos)

with b3:
    st.metric("Fokus", "Trend-Follow" if abs(score - 50) > 15 else "Mean-Reversion")

st.info(f"**Übergeordnete Regel:** {rule}")


# ============================================================
# 19. GOOGLE RETAIL SENTIMENT
# ============================================================

st.markdown("---")
st.subheader("🌐 Retail Sentiment (Google Trends)")

contra, spread, trends_live = fetch_google_trends_sentiment(selected_asset)
add_feed_status(feed_status, 'Google Trends Retail Sentiment', trends_live)

g1, g2, g3 = st.columns(3)

with g1:
    st.metric(
        "Google Retail Score (0-100)",
        f"{contra} / 100",
        f"Net Spread: {spread:+.2f} σ",
        delta_color="inverse"
    )

with g2:
    if contra >= 65:
        st.success("🟢 Panik-Ausschlag: kontraindikativ potenziell positiv.")
    elif contra <= 35:
        st.error("🔴 Gier-Ausschlag: mögliche Überhitzung.")
    else:
        st.info("🟡 Ausgeglichenes Sentiment.")

with g3:
    cfg = TREND_KEYWORD_MAP[selected_asset]
    st.markdown(
        f"""
**🔍 Getrackte Parameter**
* **Region:** `{cfg['geo']}`
* **Euphorie:** {', '.join(repr(x) for x in cfg['bull'])}
* **Panik:** {', '.join(repr(x) for x in cfg['bear'])}
* **Status:** {'🟢 Live' if trends_live else '🔴 Offline/Fallback'}
"""
    )


# ============================================================
# 20. SECHS SÄULEN – DETAILS & QUELLEN
# ============================================================

saeulen_details = {
    "Makroökonomie": {
        "quelle": "FRED API & Yahoo Finance",
        "funktion": "Zinsumfeld, Zentralbank-Liquidität und Dollar-Stärke.",
        "links": [
            ("FRED: Fed Total Assets (WALCL)", "https://fred.stlouisfed.org/series/WALCL"),
            ("FRED: TGA Account (WTREGEN)", "https://fred.stlouisfed.org/series/WTREGEN"),
            ("FRED: Reverse Repo (RRPONTSYD)", "https://fred.stlouisfed.org/series/RRPONTSYD"),
            ("FRED: 10Y Real Yields (DFII10)", "https://fred.stlouisfed.org/series/DFII10"),
            ("FRED: Fed Funds Rate (FEDFUNDS)", "https://fred.stlouisfed.org/series/FEDFUNDS"),
            ("Yahoo: US Dollar Index (DX-Y.NYB)", "https://finance.yahoo.com/quote/DX-Y.NYB")
        ]
    },
    "Positionierung": {
        "quelle": "CFTC COT, CNN Fear & Greed & Options-P/C-Proxy",
        "funktion": "Institutionelle Positionierung, Sentiment und Optionspositionierung.",
        "links": [
            ("CFTC: Commitment of Traders", "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"),
            ("CNN: Fear & Greed Index", "https://edition.cnn.com/markets/fear-and-greed")
        ]
    },
    "Marktinterna": {
        "quelle": "Yahoo Finance",
        "funktion": "Preis-Momentum, Marktvolatilität und Risikoappetit.",
        "links": [
            ("Yahoo: VIX", "https://finance.yahoo.com/quote/%5EVIX"),
            ("Yahoo: VXN", "https://finance.yahoo.com/quote/%5EVXN")
        ]
    },
    "Technischer_Trend": {
        "quelle": "Yahoo Finance",
        "funktion": "200-Tage-Trend, 50-Tage-Trend und RSI-Momentum.",
        "links": [
            ("Yahoo Chart", f"https://finance.yahoo.com/quote/{ASSET_CONFIGS[selected_asset]['ticker']}")
        ]
    },
    "Fundamentale_Faktoren": {
        "quelle": "Multpl / FRED",
        "funktion": "Bewertung bzw. Rohstoff-Lagerbestände.",
        "links": [
            ("Multpl: S&P 500 PE Ratio", "https://www.multpl.com/s-p-500-pe-ratio")
        ]
    },
    "Fruehwarnindikatoren": {
        "quelle": "Yahoo Finance",
        "funktion": "Kreditmarkt-Proxy und Anleihenvolatilität.",
        "links": [
            ("Yahoo: HYG ETF", "https://finance.yahoo.com/quote/HYG"),
            ("Yahoo: LQD ETF", "https://finance.yahoo.com/quote/LQD")
        ]
    }
}


# ============================================================
# 21. TREIBER-ANALYSE
# ============================================================

st.markdown("---")
st.subheader("🔍 Treiber-Analyse (Die 6 Säulen)")

cols = st.columns(3)
saeulen = [c for c in df_dash.columns if c.startswith("Saeule_")]

for i, s in enumerate(saeulen):
    val = float(today.get(s, 50))
    raw = s.replace("Saeule_", "")
    label = raw.replace("_", " ")

    emoji = "🟢" if val > 60 else "🔴" if val < 40 else "🟡"
    weight = ASSET_CONFIGS[selected_asset]["Saeulen_Gewichte"].get(raw, 0) * 100
    coverage = float(today.get(f"Coverage_{raw}", 0))

    with cols[i % 3]:
        st.metric(f"{label} {emoji}", f"{val:.1f}")
        st.caption(f"Gewicht: {weight:.0f}% | Datenabdeckung: {coverage:.0f}%")

        if raw in saeulen_details:
            d = saeulen_details[raw]
            with st.expander("Details, Daten & Links"):
                st.markdown(f"**⚖️ Gewichtung:** {weight:.0f}%")
                st.markdown(f"**📡 Quelle:** {d['quelle']}")
                st.markdown(f"**⚙️ Funktion:** {d['funktion']}")
                st.markdown("**🔗 Live-Datenquellen:**")
                for title, url in d["links"]:
                    st.markdown(f"• [{title}]({url})")


# ============================================================
# 22. INTRADAY EXECUTION CHECKLISTE
# ============================================================

st.markdown("---")
st.subheader("⚡ Intraday Execution Checkliste & Filter")

trend = float(today.get("Saeule_Technischer_Trend", 50))
early = float(today.get("Saeule_Fruehwarnindikatoren", 50))
macro = float(today.get("Saeule_Makroökonomie", 50))

trend_ok = trend > 55
bond_ok = early > 35
macro_ok = macro > 50

now = pd.Timestamp.now(tz="Europe/Berlin")
wd = now.weekday()

hexensabbat = (
    now.month in [3, 6, 9, 12]
    and wd == 4
    and 15 <= now.day <= 21
)

profile = {
    0: "Montag: Preisfindung & Weekly Initial Balance",
    1: "Dienstag: Trendetablierung",
    2: "Mittwoch: Trendfortsetzung oder Mid-Week Reversal",
    3: "Donnerstag: Momentum & Volatilität",
    4: "Freitag: Wochenschluss & Profit-Taking"
}.get(wd, "Wochenende: Märkte geschlossen")

a, b = st.columns(2)

with a:
    st.markdown("#### 1. Strukturelle Filter")
    x1 = st.checkbox("Trendkonformität (Marktstruktur / GMs intakt)", value=trend_ok, key="chk_trend_det")
    x2 = st.checkbox("Anleihen- & Kreditmärkte stabil", value=bond_ok, key="chk_bond_det")
    x3 = st.checkbox(f"Makro-Umgebung im Rücken (Score: {macro:.0f})", value=macro_ok, key="chk_makro_det")
    x4 = st.checkbox(f"Statistisches Tagesprofil ({profile})", value=True, key="chk_day_profile")

with b:
    st.markdown("#### 2. Timing & Risikomanagement")
    x5 = st.checkbox("Keine High-Impact News in den nächsten 60 Min", value=True, key="chk_news_det")
    x6 = st.checkbox("Kein Hexensabbat / Ketten-Verfall", value=not hexensabbat, key="chk_opex_det")
    x7 = st.checkbox("CRV mindestens 1:2 zum Ziel", value=True, key="chk_crv_det")
    x8 = st.checkbox("US-Eröffnung / Initial Balance abgewartet", value=True, key="chk_time_det")

count = sum([x1, x2, x3, x4, x5, x6, x7, x8])
st.progress(count / 8)
st.caption(f"✅ **{count} von 8 Kriterien erfüllt**")

execution_data_ok = (model_confidence >= 65 and today.Model_Data_Coverage >= 60)

if count == 8 and score > 55 and execution_data_ok:
    st.success("🟢 **EXECUTION FREIGABE (GO):** Long-Bias erfüllt.")
elif count == 8 and score < 45 and execution_data_ok:
    st.error("🔴 **EXECUTION FREIGABE (SHORT):** Short-Bias erfüllt.")
elif not execution_data_ok:
    st.error("🛑 **DATA QUALITY GATE:** Keine Freigabe wegen fehlender Daten.")
elif score < 40:
    st.error("🔴 **STOP / KEIN TRADE:** Marktregime auf Defense.")
else:
    st.warning("🟡 **CAUTION / WARNUNG:** Gemischte Signale.")


# ============================================================
# 23. REGIME-HISTORIE
# ============================================================

st.markdown("---")
st.subheader("📈 Regime-Historie & Asset Preis (Letzte 12 Monate)")

plot = df_dash.tail(252)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=plot.index,
        y=plot.Final_Regime_Score,
        name="Regime Score (0-100)",
        fill="tozeroy"
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=plot.index,
        y=plot.Asset_Price,
        name=f"{selected_asset} Preis",
        line=dict(width=2)
    ),
    secondary_y=True
)

fig.update_yaxes(title_text="Regime Score", range=[0, 100], secondary_y=False)
fig.update_yaxes(title_text="Asset Preis", secondary_y=True)

fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 24. ASSET-REGELN & STATUS
# ============================================================

st.markdown("---")

with st.expander("📚 Asset-Regeln & Bezugsquellen"):
    asset_rule_cols = st.columns(2)
    for i, (asset_name, rule_cfg) in enumerate(ASSET_RULES.items()):
        with asset_rule_cols[i % 2]:
            st.markdown(f"### 🎯 {asset_name} – {rule_cfg['profil']}")
            for rule in rule_cfg["regeln"]:
                st.markdown(f"• {rule}")

with st.expander("⚖️ Aktuelle Modellgewichtungen"):
    weights = ASSET_CONFIGS[selected_asset]["Saeulen_Gewichte"]
    st.dataframe(
        pd.DataFrame({
            "Säule": list(weights),
            "Gewichtung": [f"{v * 100:.0f}%" for v in weights.values()]
        }),
        hide_index=True,
        use_container_width=True
    )

with st.expander("📡 System & API Status Details"):
    sc = st.columns(2)
    for i, (feed, status_value) in enumerate(feed_status.items()):
        sc[i % 2].markdown(
            f"**{feed}:** {'✅ Verbunden' if status_value else '⚠️ Fallback aktiv / Offline'}"
        )

st.markdown("---")
st.caption(
    "⚠️ Modellhinweis: Der Final Regime Score ist ein quantitatives Entscheidungs- und "
    "Regimefilter-Modell und keine Anlageberatung."
)
