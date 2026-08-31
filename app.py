import re
from html import unescape

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
            "vvix_score",
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
            },
            "Fruehwarnindikatoren": {
                "credit_spreads": .50,
                "move_index": .30,
                "vvix_score": .20
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
            "vvix_score",
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
            },
            "Fruehwarnindikatoren": {
                "credit_spreads": .50,
                "move_index": .30,
                "vvix_score": .20
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
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE + VVIX"
        ],
        "quellen": [
            "Yahoo Finance: ^GSPC, ^VIX, ^VVIX, DX-Y.NYB, ^MOVE, HYG, LQD",
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
            "Fundamental: Nasdaq-100-Bewertungsproxy über historische QQQ Trailing-P/E-Daten",
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE + VVIX"
        ],
        "quellen": [
            "Yahoo Finance: NQ=F, ^VXN, ^VVIX, DX-Y.NYB, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders (Non-Commercial Net)",
            "CNN: Fear & Greed",
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS",
            "Yahoo Finance: QQQ historische Trailing-P/E-Bewertung"
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
            "CFTC: EURO FX Commitment of Traders (Non-Commercial Net)",
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
    "vvix_score": 504,
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
    cfg = TREND_KEYWORD_MAP.get(
        asset_name,
        TREND_KEYWORD_MAP["S&P 500"]
    )

    kws = list(
        dict.fromkeys(
            cfg["bull"]
            + cfg["bear"]
        )
    )

    if not kws:
        return (
            50.0,
            0.0,
            False,
            None
        )

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

    if (
        d is None
        and PYTRENDS_AVAILABLE
    ):
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

            d = (
                p.interest_over_time()
            )

        except Exception:
            d = None

    if (
        d is None
        or not isinstance(
            d,
            pd.DataFrame
        )
        or d.empty
    ):
        return (
            50.0,
            0.0,
            False,
            None
        )

    try:
        d = d.copy()

        if "isPartial" in d.columns:
            d = d.drop(
                columns="isPartial"
            )

        if "is_partial" in d.columns:
            d = d.drop(
                columns="is_partial"
            )

        data_date = latest_valid_index_date(
            d
        )

        bull_cols = [
            x
            for x in cfg["bull"]
            if x in d.columns
        ]

        bear_cols = [
            x
            for x in cfg["bear"]
            if x in d.columns
        ]

        if (
            not bull_cols
            or not bear_cols
        ):
            return (
                50.0,
                0.0,
                False,
                data_date
            )

        def z(s):
            s = pd.to_numeric(
                s,
                errors="coerce"
            )

            m = s.rolling(
                21,
                min_periods=5
            ).mean()

            sd = (
                s.rolling(
                    21,
                    min_periods=5
                )
                .std()
                .replace(
                    0,
                    np.nan
                )
            )

            return (
                (s - m)
                / sd
            )

        bull = (
            sum(
                z(d[x])
                for x in bull_cols
            )
            / len(bull_cols)
        )

        bear = (
            sum(
                z(d[x])
                for x in bear_cols
            )
            / len(bear_cols)
        )

        spread = (
            (bull - bear)
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if spread.empty:
            return (
                50.0,
                0.0,
                False,
                data_date
            )

        latest = float(
            spread.iloc[-1]
        )

        score = float(
            np.clip(
                50
                - latest * 15,
                0,
                100
            )
        )

        return (
            round(score, 1),
            round(latest, 2),
            True,
            data_date
        )

    except Exception:
        return (
            50.0,
            0.0,
            False,
            None
        )


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


def normalize_to_percentile(series, lookback=252, invert=False):
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=float)

    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    if s.dropna().empty:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float
        )

    s = s.ffill().bfill()

    min_periods = max(
        20,
        min(60, lookback // 4)
    )

    m = s.rolling(
        lookback,
        min_periods=min_periods
    ).mean()

    sd = (
        s.rolling(
            lookback,
            min_periods=min_periods
        )
        .std()
        .replace(0, np.nan)
    )

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

    mean = np.average(
        s,
        weights=effective_w
    )

    sd = np.sqrt(
        np.average(
            (s - mean) ** 2,
            weights=effective_w
        )
    )

    if len(s) < 2:
        return 50.0

    if np.isclose(sd, 0.0, atol=1e-9):
        consistency = (
            50.0
            if np.isclose(mean, 50.0)
            else 100.0
        )
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


def latest_valid_index_date(obj):
    """
    Liefert das letzte tatsächliche Beobachtungsdatum eines
    Series/DataFrame-Objekts, ohne Forward-Fill künstlich als
    neue Datenaktualisierung zu interpretieren.
    """
    if obj is None:
        return None

    try:
        if isinstance(obj, pd.Series):
            clean = obj.dropna()

            if clean.empty:
                return None

            idx = clean.index

        elif isinstance(obj, pd.DataFrame):
            clean = obj.dropna(
                how="all"
            )

            if clean.empty:
                return None

            idx = clean.index

        else:
            return None

        dt = pd.to_datetime(
            idx,
            errors="coerce"
        )

        if isinstance(
            dt,
            pd.DatetimeIndex
        ):
            dt = dt[
                ~dt.isna()
            ]

            if len(dt) == 0:
                return None

            result = pd.Timestamp(
                dt.max()
            )

        else:
            result = pd.Timestamp(
                dt
            )

        if result.tzinfo is not None:
            result = result.tz_convert(
                None
            )

        return result.normalize()

    except Exception:
        return None


def latest_common_date(frame, columns):
    """
    Letztes Datum, an dem alle angegebenen Spalten gleichzeitig
    echte Werte besitzen.
    """
    if (
        frame is None
        or not isinstance(
            frame,
            pd.DataFrame
        )
        or frame.empty
    ):
        return None

    if not all(
        c in frame.columns
        for c in columns
    ):
        return None

    try:
        clean = (
            frame[columns]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .dropna(
                how="any"
            )
        )

        return latest_valid_index_date(
            clean
        )

    except Exception:
        return None


def oldest_available_date(dates):
    """
    Konservativer Datenstand für kombinierte Feeds:
    Gibt nur dann ein Datum zurück, wenn alle benötigten
    Teilreihen ein Datum besitzen, und nimmt dann das älteste.
    """
    cleaned = []

    for value in dates:
        if value is None:
            return None

        try:
            ts = pd.Timestamp(
                value
            )

            if ts.tzinfo is not None:
                ts = ts.tz_convert(
                    None
                )

            cleaned.append(
                ts.normalize()
            )

        except Exception:
            return None

    if not cleaned:
        return None

    return min(
        cleaned
    )


def format_feed_date(value):
    if value is None:
        return "Stand: unbekannt"

    try:
        ts = pd.Timestamp(
            value
        )

        if ts.tzinfo is not None:
            ts = ts.tz_convert(
                None
            )

        return (
            "Stand: "
            + ts.strftime(
                "%d.%m.%Y"
            )
        )

    except Exception:
        return "Stand: unbekannt"


def neutralize_missing_columns(df, columns):
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan


def _find_cftc_column(df, candidates):
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
    """
    Fehlende Komponenten werden nicht als echte 50er-Werte gezählt.
    Verfügbare Gewichte werden innerhalb der Säule dynamisch
    renormalisiert. Die Coverage zeigt, wie viel des ursprünglichen
    Säulengewichts tatsächlich mit Daten befüllt ist.
    """
    cols = [
        c for c in weights
        if c in norm_df.columns
    ]

    if not cols:
        return (
            pd.Series(50.0, index=raw_df.index),
            pd.Series(0.0, index=raw_df.index)
        )

    w_series = pd.Series(
        {
            c: weights[c]
            for c in cols
        },
        dtype=float
    )

    total_weight = float(
        w_series.sum()
    )

    if total_weight <= 0:
        return (
            pd.Series(50.0, index=raw_df.index),
            pd.Series(0.0, index=raw_df.index)
        )

    sub_df = (
        norm_df[cols]
        .astype(float)
    )

    valid_mask = sub_df.notna()

    avail_weight = valid_mask.dot(
        w_series
    )

    weighted_sum = (
        sub_df
        .fillna(0.0)
        .dot(w_series)
    )

    score = (
        weighted_sum
        .div(
            avail_weight.replace(
                0,
                np.nan
            )
        )
        .fillna(50.0)
        .clip(0, 100)
    )

    coverage = (
        avail_weight
        .div(total_weight)
        .mul(100.0)
        .fillna(0.0)
        .clip(0, 100)
    )

    return score, coverage


def calculate_model_confidence(
    feed_status,
    selected_asset,
    df
):
    checks = [
        bool(
            feed_status.get(
                "yFinance (Preis & Tech)",
                False
            )
        ),
        bool(
            feed_status.get(
                "FRED API (Makro & Fed)",
                False
            )
        ),
        bool(
            any(
                "CFTC COT" in k
                and v
                for k, v in feed_status.items()
            )
        )
    ]

    # CNN Fear & Greed is part of the active Positionierung pillar.
    # The Options Put/Call ratio is intentionally display-only (0% model
    # weight) and therefore must NOT reduce model confidence or indirectly
    # block execution when Yahoo Options is unavailable.
    sentiment_live = bool(
        feed_status.get(
            "CNN Fear & Greed",
            False
        )
    )

    base = (
        sum(checks)
        / len(checks)
    ) * 75.0

    supplement = (
        25.0
        if sentiment_live
        else 0.0
    )

    latest_coverage = 0.0

    if (
        df is not None
        and not df.empty
        and "Model_Data_Coverage" in df.columns
    ):
        latest_coverage = float(
            pd.to_numeric(
                df["Model_Data_Coverage"].iloc[-1],
                errors="coerce"
            )
        )

        if not np.isfinite(latest_coverage):
            latest_coverage = 0.0

    confidence = round(
        float(
            (base + supplement)
            * (latest_coverage / 100.0)
        ),
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
    url = (
        "https://production.dataviz.cnn.io/"
        "index/fearandgreed/graphdata"
    )

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

        historical = (
            payload
            .get("fear_and_greed_historical", {})
            .get("data", [])
        )

        d = pd.DataFrame(
            historical
        )

        if (
            d.empty
            or not {"x", "y"}.issubset(d.columns)
        ):
            return pd.Series(dtype=float), False

        d["Date"] = (
            pd.to_datetime(
                d["x"],
                unit="ms",
                errors="coerce"
            )
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        d["y"] = pd.to_numeric(
            d["y"],
            errors="coerce"
        )

        d = (
            d.dropna(
                subset=["Date", "y"]
            )
            .drop_duplicates(
                "Date",
                keep="last"
            )
            .sort_values("Date")
        )

        if d.empty:
            return pd.Series(dtype=float), False

        return (
            d.set_index("Date")["y"],
            True
        )

    except Exception:
        return pd.Series(dtype=float), False


# ============================================================
# 6. CFTC COT – OFFIZIELLE SOCRATA API + FRESHNESS-GATE
# ============================================================

CFTC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; QuantRegimeDashboard/5.1)",
    "Accept": "application/json,text/plain,*/*"
}

CFTC_LEGACY_API = (
    "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
)

CFTC_MAX_AGE_DAYS = 12


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cot_data(asset_search_string, market_code=None):
    """
    CFTC Legacy Futures-Only über die offizielle Socrata API.

    Verwendet:
    Non-Commercial Long - Non-Commercial Short.

    Nur Daten mit maximal 12 Tagen Alter gelten als live.

    Rückgabe:
    series, live_status, source_text, last_report_date
    """
    code = str(
        market_code or ""
    ).strip()

    if not code:
        return (
            pd.Series(dtype=float),
            False,
            "No CFTC market code",
            None
        )

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

        d = pd.DataFrame(
            payload
        )

        if d.empty:
            return (
                pd.Series(dtype=float),
                False,
                "CFTC API returned no rows",
                None
            )

        date_col = _find_cftc_column(
            d,
            [
                "report_date_as_yyyy_mm_dd",
                "Report_Date_as_YYYY_MM_DD"
            ]
        )

        long_col = _find_cftc_column(
            d,
            [
                "noncomm_positions_long_all",
                "NonComm_Positions_Long_All"
            ]
        )

        short_col = _find_cftc_column(
            d,
            [
                "noncomm_positions_short_all",
                "NonComm_Positions_Short_All"
            ]
        )

        if (
            date_col is None
            or long_col is None
            or short_col is None
        ):
            return (
                pd.Series(dtype=float),
                False,
                "CFTC Non-Commercial fields changed",
                None
            )

        d["Date"] = pd.to_datetime(
            d[date_col],
            errors="coerce"
        )

        d["Net_NonCommercials"] = (
            pd.to_numeric(
                d[long_col],
                errors="coerce"
            )
            -
            pd.to_numeric(
                d[short_col],
                errors="coerce"
            )
        )

        d = (
            d.dropna(
                subset=[
                    "Date",
                    "Net_NonCommercials"
                ]
            )
            .drop_duplicates(
                "Date",
                keep="first"
            )
            .sort_values(
                "Date"
            )
        )

        if d.empty:
            return (
                pd.Series(dtype=float),
                False,
                "No valid CFTC observations",
                None
            )

        latest_date = (
            pd.Timestamp(
                d["Date"].max()
            )
            .tz_localize(None)
            .normalize()
        )

        today_utc = (
            pd.Timestamp.now(
                tz="UTC"
            )
            .floor("D")
            .tz_localize(None)
        )

        age_days = int(
            (
                today_utc
                - latest_date
            ).days
        )

        if age_days < 0:
            return (
                pd.Series(dtype=float),
                False,
                (
                    "CFTC report date is in the future "
                    f"({latest_date:%Y-%m-%d})"
                ),
                latest_date
            )

        if age_days > CFTC_MAX_AGE_DAYS:
            return (
                pd.Series(dtype=float),
                False,
                (
                    f"CFTC stale: {age_days} days old "
                    f"(max {CFTC_MAX_AGE_DAYS})"
                ),
                latest_date
            )

        return (
            d.set_index(
                "Date"
            )[
                "Net_NonCommercials"
            ],
            True,
            (
                f"CFTC API / {code} / "
                f"{latest_date:%Y-%m-%d} / "
                f"age {age_days}d"
            ),
            latest_date
        )

    except Exception as exc:
        return (
            pd.Series(dtype=float),
            False,
            f"CFTC API error: {str(exc)[:90]}",
            None
        )


# ============================================================
# 7. OPTIONS PUT/CALL – YFINANCE AUTH + RAW FALLBACK
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_put_call(ticker, max_expiries=3):
    """
    Put/Call-Volumenverhältnis für ETF-Optionsproxies.

    Abrufreihenfolge:
    1) yfinance: mehrere Expiries über Ticker.options
    2) yfinance: option_chain() ohne Datum
       -> funktioniert auch dann, wenn die separate Expiry-Liste scheitert
    3) direkter Yahoo-Endpunkt query2
    4) direkter Yahoo-Endpunkt query1

    Rückgabe:
    ratio, live_status, source_text, last_option_trade_date
    """
    ticker = str(
        ticker
    ).strip().upper()

    if not ticker:
        return (
            np.nan,
            False,
            "No option ticker",
            None
        )

    now = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .tz_localize(None)
        .normalize()
    )

    def extract_chain_ratio(
        puts,
        calls,
        raw_unix=False
    ):
        if not isinstance(
            puts,
            pd.DataFrame
        ):
            puts = pd.DataFrame()

        if not isinstance(
            calls,
            pd.DataFrame
        ):
            calls = pd.DataFrame()

        if (
            puts.empty
            and calls.empty
        ):
            return (
                np.nan,
                None
            )

        pv = pd.to_numeric(
            puts.get(
                "volume",
                pd.Series(
                    dtype=float
                )
            ),
            errors="coerce"
        ).fillna(
            0
        ).sum()

        cv = pd.to_numeric(
            calls.get(
                "volume",
                pd.Series(
                    dtype=float
                )
            ),
            errors="coerce"
        ).fillna(
            0
        ).sum()

        trade_dates = []

        for frame in (
            puts,
            calls
        ):
            if (
                frame.empty
                or "lastTradeDate"
                not in frame.columns
            ):
                continue

            try:
                if raw_unix:
                    dt = pd.to_datetime(
                        frame[
                            "lastTradeDate"
                        ],
                        unit="s",
                        errors="coerce",
                        utc=True
                    )
                else:
                    dt = pd.to_datetime(
                        frame[
                            "lastTradeDate"
                        ],
                        errors="coerce",
                        utc=True
                    )

                dt = dt.dropna()

                if not dt.empty:
                    latest = pd.Timestamp(
                        dt.max()
                    ).tz_convert(
                        None
                    ).normalize()

                    trade_dates.append(
                        latest
                    )

            except Exception:
                continue

        last_trade_date = (
            max(trade_dates)
            if trade_dates
            else None
        )

        if cv <= 0:
            return (
                np.nan,
                last_trade_date
            )

        ratio = (
            float(pv)
            / float(cv)
        )

        if (
            not np.isfinite(
                ratio
            )
            or ratio < 0
        ):
            return (
                np.nan,
                last_trade_date
            )

        return (
            float(ratio),
            last_trade_date
        )

    errors = []

    # --------------------------------------------------------
    # PRIMARY A: yfinance multiple expiries
    # --------------------------------------------------------

    try:
        t = yf.Ticker(
            ticker
        )

        expiries = list(
            t.options
            or []
        )

        future = []

        for expiry in expiries:
            try:
                dt = pd.to_datetime(
                    expiry,
                    errors="coerce"
                )

                if pd.notna(dt):
                    dt = (
                        pd.Timestamp(
                            dt
                        )
                        .tz_localize(
                            None
                        )
                        .normalize()
                    )

                    if dt >= now:
                        future.append(
                            (
                                dt,
                                str(expiry)
                            )
                        )

            except Exception:
                continue

        future.sort(
            key=lambda x: x[0]
        )

        selected = future[
            :max_expiries
        ]

        total_put = 0.0
        total_call = 0.0
        used = 0
        trade_dates = []

        for _, expiry in selected:
            try:
                chain = (
                    t.option_chain(
                        expiry
                    )
                )

                puts = getattr(
                    chain,
                    "puts",
                    pd.DataFrame()
                )

                calls = getattr(
                    chain,
                    "calls",
                    pd.DataFrame()
                )

                pv = pd.to_numeric(
                    puts.get(
                        "volume",
                        pd.Series(
                            dtype=float
                        )
                    ),
                    errors="coerce"
                ).fillna(
                    0
                ).sum()

                cv = pd.to_numeric(
                    calls.get(
                        "volume",
                        pd.Series(
                            dtype=float
                        )
                    ),
                    errors="coerce"
                ).fillna(
                    0
                ).sum()

                _, chain_date = (
                    extract_chain_ratio(
                        puts,
                        calls
                    )
                )

                if chain_date is not None:
                    trade_dates.append(
                        chain_date
                    )

                if (
                    pv > 0
                    or cv > 0
                ):
                    total_put += float(
                        pv
                    )

                    total_call += float(
                        cv
                    )

                    used += 1

            except Exception as exc:
                errors.append(
                    (
                        f"yfinance expiry "
                        f"{expiry}: "
                        f"{str(exc)[:80]}"
                    )
                )

        if (
            used > 0
            and total_call > 0
        ):
            ratio = (
                total_put
                / total_call
            )

            if (
                np.isfinite(
                    ratio
                )
                and ratio >= 0
            ):
                return (
                    float(ratio),
                    True,
                    (
                        f"yfinance {ticker} "
                        f"({used} Expiries)"
                    ),
                    (
                        max(trade_dates)
                        if trade_dates
                        else None
                    )
                )

        errors.append(
            "yfinance expiry list produced no usable volume"
        )

    except Exception as exc:
        errors.append(
            (
                "yfinance Ticker.options failed: "
                f"{str(exc)[:100]}"
            )
        )

    # --------------------------------------------------------
    # PRIMARY B: yfinance default option_chain()
    # --------------------------------------------------------
    #
    # Important: option_chain(date=None) is a separate path.
    # It avoids making successful retrieval dependent on t.options.

    try:
        t_default = yf.Ticker(
            ticker
        )

        chain = (
            t_default.option_chain()
        )

        puts = getattr(
            chain,
            "puts",
            pd.DataFrame()
        )

        calls = getattr(
            chain,
            "calls",
            pd.DataFrame()
        )

        ratio, trade_date = (
            extract_chain_ratio(
                puts,
                calls
            )
        )

        if np.isfinite(
            ratio
        ):
            return (
                float(ratio),
                True,
                (
                    f"yfinance {ticker} "
                    "default expiry"
                ),
                trade_date
            )

        errors.append(
            "yfinance default option_chain returned no usable call volume"
        )

    except Exception as exc:
        errors.append(
            (
                "yfinance default option_chain failed: "
                f"{str(exc)[:100]}"
            )
        )

    # --------------------------------------------------------
    # FALLBACK: direct Yahoo endpoints
    # --------------------------------------------------------

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131 Safari/537.36"
        ),
        "Accept": (
            "application/json,"
            "text/plain,*/*"
        ),
        "Referer": (
            "https://finance.yahoo.com/"
        )
    }

    for yahoo_host in (
        "query2.finance.yahoo.com",
        "query1.finance.yahoo.com"
    ):
        url = (
            f"https://{yahoo_host}/"
            f"v7/finance/options/{ticker}"
        )

        try:
            r = requests.get(
                url,
                headers=headers,
                timeout=20
            )

            r.raise_for_status()

            root_result = (
                r.json()
                .get(
                    "optionChain",
                    {}
                )
                .get(
                    "result"
                )
                or []
            )

            if not root_result:
                errors.append(
                    f"{yahoo_host}: empty option chain"
                )
                continue

            expiries = (
                root_result[0]
                .get(
                    "expirationDates"
                )
                or []
            )

            future = []

            for ts in expiries:
                try:
                    dt = pd.to_datetime(
                        int(ts),
                        unit="s"
                    ).normalize()

                    if dt >= now:
                        future.append(
                            (
                                dt,
                                int(ts)
                            )
                        )

                except Exception:
                    continue

            future.sort(
                key=lambda x: x[0]
            )

            selected = future[
                :max_expiries
            ]

            total_put = 0.0
            total_call = 0.0
            used = 0
            trade_dates = []

            for _, ts in selected:
                try:
                    rr = requests.get(
                        f"{url}?date={ts}",
                        headers=headers,
                        timeout=20
                    )

                    rr.raise_for_status()

                    res = (
                        rr.json()
                        .get(
                            "optionChain",
                            {}
                        )
                        .get(
                            "result"
                        )
                        or []
                    )

                    if not res:
                        continue

                    opts = (
                        res[0]
                        .get(
                            "options"
                        )
                        or []
                    )

                    if not opts:
                        continue

                    chain = (
                        opts[0]
                    )

                    puts = pd.DataFrame(
                        chain.get(
                            "puts"
                        )
                        or []
                    )

                    calls = pd.DataFrame(
                        chain.get(
                            "calls"
                        )
                        or []
                    )

                    pv = pd.to_numeric(
                        puts.get(
                            "volume",
                            pd.Series(
                                dtype=float
                            )
                        ),
                        errors="coerce"
                    ).fillna(
                        0
                    ).sum()

                    cv = pd.to_numeric(
                        calls.get(
                            "volume",
                            pd.Series(
                                dtype=float
                            )
                        ),
                        errors="coerce"
                    ).fillna(
                        0
                    ).sum()

                    _, chain_date = (
                        extract_chain_ratio(
                            puts,
                            calls,
                            raw_unix=True
                        )
                    )

                    if chain_date is not None:
                        trade_dates.append(
                            chain_date
                        )

                    if (
                        pv > 0
                        or cv > 0
                    ):
                        total_put += float(
                            pv
                        )

                        total_call += float(
                            cv
                        )

                        used += 1

                except Exception as exc:
                    errors.append(
                        (
                            f"{yahoo_host} expiry "
                            f"{ts}: "
                            f"{str(exc)[:70]}"
                        )
                    )

            if (
                used > 0
                and total_call > 0
            ):
                ratio = (
                    total_put
                    / total_call
                )

                if (
                    np.isfinite(
                        ratio
                    )
                    and ratio >= 0
                ):
                    return (
                        float(ratio),
                        True,
                        (
                            f"{yahoo_host} "
                            f"{ticker} "
                            f"({used} Expiries)"
                        ),
                        (
                            max(trade_dates)
                            if trade_dates
                            else None
                        )
                    )

            errors.append(
                f"{yahoo_host}: no usable option volume"
            )

        except Exception as exc:
            errors.append(
                (
                    f"{yahoo_host}: "
                    f"{str(exc)[:100]}"
                )
            )

    detail = " | ".join(
        errors[-6:]
    )

    return (
        np.nan,
        False,
        (
            "Options offline: "
            f"{detail[:600]}"
        ),
        None
    )


# ============================================================
# 8. FRED API KEY
# ============================================================

FRED_API_KEY = ""

try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = (
            st.secrets[
                "FRED_API_KEY"
            ]
        )
except Exception:
    pass


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:
    st.title(
        "⚙️ Multi-Asset Selector"
    )

    selected_asset = st.selectbox(
        "🎯 Asset auswählen",
        list(ASSET_CONFIGS),
        index=0
    )

    st.markdown("---")
    st.markdown(
        "### 📡 API Live-Feed Monitor"
    )


# ============================================================
# 10A. BEWERTUNGSDATEN – S&P 500 & NASDAQ 100
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yfinance_pe_history(proxy_ticker):
    """
    Historische monatliche Trailing-P/E-Daten über yfinance.

    yfinance >= 1.3 stellt get_valuation_measures() bereit.
    Mit freq="monthly" und periods=None werden alle verfügbaren
    Monatswerte angefordert.
    """
    try:
        t = yf.Ticker(
            str(proxy_ticker)
            .strip()
            .upper()
        )

        if not hasattr(
            t,
            "get_valuation_measures"
        ):
            return (
                pd.Series(dtype=float),
                False
            )

        vm = t.get_valuation_measures(
            freq="monthly",
            periods=None
        )

        if (
            vm is None
            or not isinstance(
                vm,
                pd.DataFrame
            )
            or vm.empty
            or "Trailing P/E"
            not in vm.index
        ):
            return (
                pd.Series(dtype=float),
                False
            )

        row = vm.loc[
            "Trailing P/E"
        ]

        records = []

        for col, value in row.items():
            if str(col) == "Current":
                continue

            dt = pd.to_datetime(
                col,
                errors="coerce"
            )

            val = pd.to_numeric(
                value,
                errors="coerce"
            )

            if (
                pd.notna(dt)
                and np.isfinite(val)
                and 0 < float(val) < 300
            ):
                records.append(
                    (
                        pd.Timestamp(dt)
                        .tz_localize(None),
                        float(val)
                    )
                )

        if len(records) < 12:
            return (
                pd.Series(dtype=float),
                False
            )

        d = pd.DataFrame(
            records,
            columns=[
                "Date",
                "PE"
            ]
        )

        d = (
            d.drop_duplicates(
                "Date",
                keep="last"
            )
            .sort_values("Date")
        )

        return (
            d.set_index("Date")[
                "PE"
            ],
            True
        )

    except Exception:
        return (
            pd.Series(dtype=float),
            False
        )


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_pe_history():
    """
    Historisches S&P-500-KGV.

    Priorität:
    1) Multpl S&P 500 PE Ratio by Month
       -> tatsächlicher Index-Bewertungsfeed
    2) pandas.read_html() auf dieselbe Tabelle
    3) yfinance SPY valuation history nur als Fallback

    Rückgabe:
    series, ok, source_text
    """
    url = (
        "https://www.multpl.com/"
        "s-p-500-pe-ratio/table/by-month"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": (
            "en-US,en;q=0.9"
        )
    }

    multpl_error = None

    # --------------------------------------------------------
    # PRIMARY: dependency-free HTML table parser
    # --------------------------------------------------------

    try:
        r = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        r.raise_for_status()

        row_blocks = re.findall(
            r"<tr\b[^>]*>(.*?)</tr>",
            r.text,
            flags=re.I | re.S
        )

        rows = []

        for row_html in row_blocks:
            cells = re.findall(
                r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
                row_html,
                flags=re.I | re.S
            )

            if len(cells) < 2:
                continue

            cleaned = []

            for cell in cells:
                value = re.sub(
                    r"<[^>]+>",
                    " ",
                    cell
                )

                value = unescape(
                    value
                )

                value = re.sub(
                    r"\s+",
                    " ",
                    value
                ).strip()

                cleaned.append(
                    value
                )

            dt = pd.to_datetime(
                cleaned[0],
                errors="coerce"
            )

            if pd.isna(dt):
                continue

            pe_value = None

            # Do not parse the date cell again. Search the remaining
            # cells from right to left so "Estimate 29.79" resolves
            # to the actual P/E value, not to a year or footnote.
            for cell_text in reversed(
                cleaned[1:]
            ):
                matches = re.findall(
                    r"(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)",
                    cell_text
                )

                for candidate in reversed(
                    matches
                ):
                    try:
                        number = float(
                            candidate
                        )

                        if 0 < number < 200:
                            pe_value = number
                            break

                    except Exception:
                        continue

                if pe_value is not None:
                    break

            if pe_value is not None:
                rows.append(
                    (
                        pd.Timestamp(dt)
                        .tz_localize(None),
                        pe_value
                    )
                )

        if rows:
            d = pd.DataFrame(
                rows,
                columns=[
                    "Date",
                    "PE"
                ]
            )

            d = (
                d.dropna(
                    subset=[
                        "Date",
                        "PE"
                    ]
                )
                .drop_duplicates(
                    "Date",
                    keep="last"
                )
                .sort_values(
                    "Date"
                )
            )

            if len(d) >= 12:
                return (
                    d.set_index(
                        "Date"
                    )[
                        "PE"
                    ],
                    True,
                    "Multpl S&P 500 PE / HTML parser"
                )

        multpl_error = (
            "Multpl HTML parser returned "
            "fewer than 12 observations"
        )

    except Exception as exc:
        multpl_error = (
            "Multpl request/parser error: "
            f"{str(exc)[:120]}"
        )

    # --------------------------------------------------------
    # SECONDARY: pandas HTML parser
    # --------------------------------------------------------

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        tables = pd.read_html(
            response.text
        )

        for table in tables:
            if table.empty:
                continue

            cols = [
                str(c)
                .strip()
                .lower()
                for c in table.columns
            ]

            date_col = next(
                (
                    table.columns[i]
                    for i, c
                    in enumerate(cols)
                    if "date" in c
                ),
                None
            )

            value_col = next(
                (
                    table.columns[i]
                    for i, c
                    in enumerate(cols)
                    if c == "value"
                ),
                None
            )

            if (
                date_col is None
                or value_col is None
            ):
                continue

            d = pd.DataFrame(
                {
                    "Date": pd.to_datetime(
                        table[
                            date_col
                        ],
                        errors="coerce"
                    ),
                    "PE": pd.to_numeric(
                        table[
                            value_col
                        ]
                        .astype(str)
                        .str.replace(
                            r"[^0-9.\-]",
                            "",
                            regex=True
                        ),
                        errors="coerce"
                    )
                }
            )

            d = (
                d.dropna(
                    subset=[
                        "Date",
                        "PE"
                    ]
                )
                .query(
                    "PE > 0 and PE < 200"
                )
                .drop_duplicates(
                    "Date",
                    keep="last"
                )
                .sort_values(
                    "Date"
                )
            )

            if len(d) >= 12:
                return (
                    d.set_index(
                        "Date"
                    )[
                        "PE"
                    ],
                    True,
                    "Multpl S&P 500 PE / pandas.read_html"
                )

    except Exception as exc:
        multpl_error = (
            f"{multpl_error}; "
            "read_html: "
            f"{str(exc)[:100]}"
        )

    # --------------------------------------------------------
    # LAST FALLBACK: SPY valuation history
    # --------------------------------------------------------

    yf_pe, yf_ok = (
        fetch_yfinance_pe_history(
            "SPY"
        )
    )

    if (
        yf_ok
        and yf_pe is not None
        and not yf_pe.empty
    ):
        return (
            yf_pe,
            True,
            (
                "yfinance SPY Trailing P/E fallback; "
                f"Multpl failed: {multpl_error}"
            )
        )

    return (
        pd.Series(
            dtype=float
        ),
        False,
        (
            "S&P 500 valuation offline; "
            f"{multpl_error}; "
            "SPY valuation fallback unavailable"
        )
    )


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq100_pe_history():
    """
    Nasdaq-100-Bewertungsproxy über QQQ Trailing-P/E-Historie.
    """
    pe, ok = (
        fetch_yfinance_pe_history(
            "QQQ"
        )
    )

    if (
        pe is not None
        and not pe.empty
        and ok
    ):
        return (
            pe,
            True,
            "yfinance QQQ Trailing P/E"
        )

    return (
        pd.Series(dtype=float),
        False,
        "Nasdaq valuation unavailable"
    )


# Common Yahoo Finance universe. Download once per cache window so
# switching the sidebar asset does not trigger a fresh 5-year download.
ALL_YF_TICKERS = tuple(
    sorted(
        set(
            [cfg["ticker"] for cfg in ASSET_CONFIGS.values()]
            + [cfg["volatility_ticker"] for cfg in ASSET_CONFIGS.values()]
            + ["DX-Y.NYB", "^MOVE", "^VVIX", "HYG", "LQD"]
        )
    )
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_market_bundle():
    try:
        data = yf.download(
            list(ALL_YF_TICKERS),
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column"
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
    cfg = ASSET_CONFIGS[
        selected_asset
    ]

    status = {}
    feed_dates = {}
    feed_notes = {}

    tickers = {
        "asset": cfg["ticker"],
        "volatility": cfg["volatility_ticker"],
        "dxy": "DX-Y.NYB",
        "move": "^MOVE",
        "vvix": "^VVIX",
        "hyg": "HYG",
        "lqd": "LQD"
    }

    data = (
        fetch_yfinance_market_bundle()
    )

    if data is None or data.empty:
        add_feed_status(
            status,
            "yFinance (Preis & Tech)",
            False
        )

        feed_dates[
            "yFinance (Preis & Tech)"
        ] = None

        return (
            pd.DataFrame(),
            status,
            feed_dates,
            feed_notes
        )

    close = extract_yfinance_field(
        data,
        "Close"
    )

    if close is None:
        add_feed_status(
            status,
            "yFinance (Preis & Tech)",
            False
        )

        feed_dates[
            "yFinance (Preis & Tech)"
        ] = None

        return (
            pd.DataFrame(),
            status,
            feed_dates,
            feed_notes
        )

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = flatten_yfinance_columns(
        close
    )

    close = close.rename(
        columns={
            ticker: name
            for name, ticker
            in tickers.items()
        }
    )

    close = (
        close
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .sort_index()
    )

    asset_ok = (
        "asset" in close.columns
        and not close[
            "asset"
        ].dropna().empty
    )

    add_feed_status(
        status,
        "yFinance (Preis & Tech)",
        asset_ok
    )

    feed_dates[
        "yFinance (Preis & Tech)"
    ] = latest_valid_index_date(
        close["asset"]
        if "asset" in close.columns
        else None
    )

    if not asset_ok:
        return (
            pd.DataFrame(),
            status,
            feed_dates,
            feed_notes
        )

    price = (
        close["asset"]
        .dropna()
    )

    df = pd.DataFrame(
        index=price.index
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    vol = extract_yfinance_field(
        data,
        "Volume"
    )

    has_volume = False

    if isinstance(vol, pd.Series):
        asset_vol = (
            pd.to_numeric(
                vol,
                errors="coerce"
            )
            .reindex(
                price.index
            )
        )

        has_volume = (
            asset_vol
            .notna()
            .any()
        )

    elif isinstance(vol, pd.DataFrame):
        vol = flatten_yfinance_columns(
            vol
        )

        vol = vol.rename(
            columns={
                ticker: name
                for name, ticker
                in tickers.items()
            }
        )

        if "asset" in vol.columns:
            asset_vol = (
                pd.to_numeric(
                    vol["asset"],
                    errors="coerce"
                )
                .reindex(
                    price.index
                )
            )

            has_volume = (
                asset_vol
                .notna()
                .any()
            )

    if not has_volume:
        asset_vol = pd.Series(
            np.nan,
            index=price.index,
            dtype=float
        )

    add_feed_status(
        status,
        "Volumen / Orderflow Feed",
        has_volume
    )

    feed_dates[
        "Volumen / Orderflow Feed"
    ] = (
        latest_valid_index_date(
            asset_vol
        )
        if has_volume
        else None
    )

    # --------------------------------------------------------
    # TECHNISCHER TREND
    # --------------------------------------------------------

    ma50 = price.rolling(
        50,
        min_periods=50
    ).mean()

    ma200 = price.rolling(
        200,
        min_periods=200
    ).mean()

    df["distance_50ma"] = (
        (
            price
            - ma50
        )
        /
        ma50.replace(
            0,
            np.nan
        )
        * 100
    )

    df["distance_200ma"] = (
        (
            price
            - ma200
        )
        /
        ma200.replace(
            0,
            np.nan
        )
        * 100
    )

    # --------------------------------------------------------
    # RSI 14 – Wilder-style EWM
    # --------------------------------------------------------

    delta = price.diff()

    gain = (
        delta
        .clip(lower=0)
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        gain
        /
        loss.replace(
            0,
            np.nan
        )
    )

    df["rsi_momentum"] = (
        100
        -
        100 / (
            1 + rs
        )
    ).clip(
        0,
        100
    )

    # --------------------------------------------------------
    # VOLATILITY / USD / MOVE
    # --------------------------------------------------------

    if "volatility" in close.columns:
        df["vix_score"] = (
            pd.to_numeric(
                close["volatility"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        vola_ok = (
            df["vix_score"]
            .notna()
            .any()
        )

    else:
        df["vix_score"] = np.nan
        vola_ok = False

    if "dxy" in close.columns:
        df["usd_index"] = (
            pd.to_numeric(
                close["dxy"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        dxy_ok = (
            df["usd_index"]
            .notna()
            .any()
        )

    else:
        df["usd_index"] = np.nan
        dxy_ok = False

    if "move" in close.columns:
        df["move_index"] = (
            pd.to_numeric(
                close["move"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        move_ok = (
            df["move_index"]
            .notna()
            .any()
        )

    else:
        df["move_index"] = np.nan
        move_ok = False

    if "vvix" in close.columns:
        df["vvix_score"] = (
            pd.to_numeric(
                close["vvix"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        vvix_ok = (
            df["vvix_score"]
            .notna()
            .any()
        )

    else:
        df["vvix_score"] = np.nan
        vvix_ok = False

    add_feed_status(
        status,
        f"{cfg['volatility_ticker']} Volatilität",
        vola_ok
    )

    feed_dates[
        f"{cfg['volatility_ticker']} Volatilität"
    ] = (
        latest_valid_index_date(
            close["volatility"]
        )
        if "volatility" in close.columns
        else None
    )

    add_feed_status(
        status,
        "Yahoo Finance USD Index",
        dxy_ok
    )

    feed_dates[
        "Yahoo Finance USD Index"
    ] = (
        latest_valid_index_date(
            close["dxy"]
        )
        if "dxy" in close.columns
        else None
    )

    add_feed_status(
        status,
        "Yahoo Finance MOVE",
        move_ok
    )

    feed_dates[
        "Yahoo Finance MOVE"
    ] = (
        latest_valid_index_date(
            close["move"]
        )
        if "move" in close.columns
        else None
    )

    if selected_asset in {
        "S&P 500",
        "Nasdaq 100"
    }:
        add_feed_status(
            status,
            "Yahoo Finance VVIX",
            vvix_ok
        )

        feed_dates[
            "Yahoo Finance VVIX"
        ] = (
            latest_valid_index_date(
                close["vvix"]
            )
            if "vvix" in close.columns
            else None
        )

        feed_notes[
            "Yahoo Finance VVIX"
        ] = (
            "Cboe VIX of VIX Index (^VVIX) · "
            "20% der Frühwarnsäule"
        )

    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------

    if (
        "lqd" in close.columns
        and "hyg" in close.columns
    ):
        lqd = (
            pd.to_numeric(
                close["lqd"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        hyg = (
            pd.to_numeric(
                close["hyg"],
                errors="coerce"
            )
            .reindex(df.index)
            .ffill()
            .bfill()
        )

        df["credit_spreads"] = (
            lqd
            /
            hyg.replace(
                0,
                np.nan
            )
        )

        credit_ok = (
            df["credit_spreads"]
            .notna()
            .any()
        )

    else:
        df["credit_spreads"] = np.nan
        credit_ok = False

    add_feed_status(
        status,
        "LQD/HYG Kreditproxy",
        credit_ok
    )

    feed_dates[
        "LQD/HYG Kreditproxy"
    ] = latest_common_date(
        close,
        [
            "lqd",
            "hyg"
        ]
    )

    # --------------------------------------------------------
    # MARKET MOMENTUM
    # --------------------------------------------------------

    df["market_momentum"] = (
        price
        .pct_change()
        .rolling(
            20,
            min_periods=10
        )
        .sum()
        * 100
    )

    # --------------------------------------------------------
    # OBV MOMENTUM
    # --------------------------------------------------------

    if has_volume:
        signed_volume = np.where(
            delta > 0,
            asset_vol,
            np.where(
                delta < 0,
                -asset_vol,
                0.0
            )
        )

        obv = pd.Series(
            signed_volume,
            index=price.index
        ).cumsum()

        obv_ema = (
            obv
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
        )

        df["obv_momentum"] = (
            (
                obv
                - obv_ema
            )
            /
            obv_ema
            .abs()
            .replace(
                0,
                np.nan
            )
            * 100
        )

    else:
        df["obv_momentum"] = np.nan

    # --------------------------------------------------------
    # FRED
    # --------------------------------------------------------

    fred_ok = False
    fred_data_date = None
    inventories_ok = False
    inventories_date = None

    if FRED_API_KEY:
        try:
            f = Fred(
                api_key=FRED_API_KEY
            )

            wal_raw = f.get_series("WALCL")
            tga_raw = f.get_series("WTREGEN")
            rrp_raw = f.get_series("RRPONTSYD")
            fed_raw = f.get_series("FEDFUNDS")
            real_yield_raw = f.get_series("DFII10")

            wal = safe_reindex_series(wal_raw, df.index)
            tga = safe_reindex_series(tga_raw, df.index)
            rrp = safe_reindex_series(rrp_raw, df.index)
            fed = safe_reindex_series(fed_raw, df.index)
            real_yield = safe_reindex_series(real_yield_raw, df.index)

            fred_data_date = oldest_available_date(
                [
                    latest_valid_index_date(wal_raw),
                    latest_valid_index_date(tga_raw),
                    latest_valid_index_date(rrp_raw),
                    latest_valid_index_date(fed_raw),
                    latest_valid_index_date(real_yield_raw),
                ]
            )

            if (
                wal is not None
                and tga is not None
                and rrp is not None
            ):
                # WALCL / WTREGEN = million USD, RRPONTSYD = billion USD.
                FRED_MILLIONS_TO_BILLIONS = 1.0 / 1000.0
                df["net_liquidity"] = (
                    wal * FRED_MILLIONS_TO_BILLIONS
                    - tga * FRED_MILLIONS_TO_BILLIONS
                    - rrp
                )
            else:
                df["net_liquidity"] = np.nan

            df["fed_policy"] = fed if fed is not None else np.nan
            df["real_yields"] = (
                real_yield if real_yield is not None else np.nan
            )

            fred_ok = (
                df["fed_policy"].notna().any()
                and df["real_yields"].notna().any()
                and df["net_liquidity"].notna().any()
            )

        except Exception:
            fred_ok = False
            fred_data_date = None

    add_feed_status(
        status,
        "FRED API (Makro & Fed)",
        fred_ok
    )

    feed_dates[
        "FRED API (Makro & Fed)"
    ] = fred_data_date

    # WTI inventories are a separate fundamental input. A failure here must
    # not turn an otherwise healthy FRED macro feed red.
    if selected_asset == "WTI Crude Oil":
        if FRED_API_KEY:
            try:
                f_inventory = Fred(
                    api_key=FRED_API_KEY
                )
                inventories_raw = f_inventory.get_series(
                    "WCESTUS1"
                )
                inventories = safe_reindex_series(
                    inventories_raw,
                    df.index
                )
                df["inventories"] = (
                    inventories
                    if inventories is not None
                    else np.nan
                )
                inventories_ok = (
                    isinstance(inventories, pd.Series)
                    and inventories.notna().any()
                )
                inventories_date = latest_valid_index_date(
                    inventories_raw
                )
            except Exception:
                df["inventories"] = np.nan
                inventories_ok = False
                inventories_date = None
        else:
            df["inventories"] = np.nan

        inventory_feed = "FRED WTI Inventories (WCESTUS1)"
        add_feed_status(
            status,
            inventory_feed,
            inventories_ok
        )
        feed_dates[inventory_feed] = inventories_date
        feed_notes[inventory_feed] = (
            "Weekly U.S. Ending Stocks of Crude Oil excluding SPR"
        )

    neutral_cols = [
        "fed_policy",
        "real_yields",
        "net_liquidity"
    ]

    if selected_asset == "WTI Crude Oil":
        neutral_cols.append(
            "inventories"
        )

    neutralize_missing_columns(
        df,
        neutral_cols
    )

    # --------------------------------------------------------
    # CFTC COT
    # --------------------------------------------------------

    cot, cot_ok, cot_source, cot_date = (
        fetch_cot_data(
            cfg["cot_code"],
            cfg.get(
                "cot_market_code"
            )
        )
    )

    cot_feed_name = (
        f"CFTC COT ({cfg['cot_code']})"
    )

    add_feed_status(
        status,
        cot_feed_name,
        cot_ok
    )

    feed_dates[
        cot_feed_name
    ] = cot_date

    if (
        cot is not None
        and not cot.empty
    ):
        df["cot_noncommercials"] = (
            safe_reindex_series(
                cot,
                df.index
            )
        )

    else:
        df["cot_noncommercials"] = np.nan

    # --------------------------------------------------------
    # CNN FEAR & GREED
    # --------------------------------------------------------

    fg, fg_ok = (
        fetch_fear_and_greed()
    )

    add_feed_status(
        status,
        "CNN Fear & Greed",
        fg_ok
    )

    feed_dates[
        "CNN Fear & Greed"
    ] = latest_valid_index_date(
        fg
    )

    if (
        isinstance(fg, pd.Series)
        and not fg.empty
    ):
        df["fear_greed"] = (
            safe_reindex_series(
                fg,
                df.index
            )
        )

    else:
        df["fear_greed"] = np.nan

    # --------------------------------------------------------
    # OPTIONS PUT/CALL PROXY
    # --------------------------------------------------------

    (
        opt_pc,
        opt_ok,
        opt_source,
        opt_data_date
    ) = fetch_option_put_call(
        cfg[
            "options_pc_ticker"
        ]
    )

    df["options_put_call"] = (
        float(opt_pc)
        if np.isfinite(opt_pc)
        else np.nan
    )

    options_feed_name = (
        "Options Put/Call "
        f"({cfg['options_pc_ticker']})"
    )

    add_feed_status(
        status,
        options_feed_name,
        opt_ok
    )

    feed_dates[
        options_feed_name
    ] = opt_data_date

    feed_notes[
        options_feed_name
    ] = opt_source

    # --------------------------------------------------------
    # FUNDAMENTALE BEWERTUNGSDATEN
    # --------------------------------------------------------

    if selected_asset == "S&P 500":
        (
            pe_series,
            pe_ok,
            pe_source
        ) = fetch_sp500_pe_history()

        add_feed_status(
            status,
            "Bewertungsdaten (S&P 500 PE)",
            pe_ok
        )

        feed_dates[
            "Bewertungsdaten (S&P 500 PE)"
        ] = latest_valid_index_date(
            pe_series
        )

        feed_notes[
            "Bewertungsdaten (S&P 500 PE)"
        ] = pe_source

        if (
            pe_series is not None
            and not pe_series.empty
        ):
            df["pe_valuation"] = (
                safe_reindex_series(
                    pe_series,
                    df.index
                )
            )

        else:
            df["pe_valuation"] = np.nan

    elif selected_asset == "Nasdaq 100":
        (
            pe_series,
            pe_ok,
            pe_source
        ) = fetch_nasdaq100_pe_history()

        add_feed_status(
            status,
            "Bewertungsdaten (Nasdaq 100 / QQQ PE)",
            pe_ok
        )

        feed_dates[
            "Bewertungsdaten (Nasdaq 100 / QQQ PE)"
        ] = latest_valid_index_date(
            pe_series
        )

        feed_notes[
            "Bewertungsdaten (Nasdaq 100 / QQQ PE)"
        ] = pe_source

        if (
            pe_series is not None
            and not pe_series.empty
        ):
            df["pe_valuation"] = (
                safe_reindex_series(
                    pe_series,
                    df.index
                )
            )

        else:
            df["pe_valuation"] = np.nan

    else:
        df["pe_valuation"] = np.nan

    # --------------------------------------------------------
    # NORMALISIERUNG
    # --------------------------------------------------------

    norm_df = pd.DataFrame(
        index=df.index
    )

    for col in df.columns:
        norm_df[col] = (
            normalize_to_percentile(
                df[col],
                LOOKBACK_CONFIG.get(
                    col,
                    252
                ),
                col in cfg[
                    "invert_inverts"
                ]
            )
        )

    # --------------------------------------------------------
    # SÄULEN
    # --------------------------------------------------------

    active = {
        k: v.copy()
        for k, v
        in SUB_WEIGHTS_BASE.items()
    }

    for cat, weights in cfg[
        "Sub_Gewichte"
    ].items():
        active[cat] = weights

    dash = pd.DataFrame(
        index=df.index
    )

    dash["Raw_Volatility"] = (
        df["vix_score"]
    )

    for pillar, weights in active.items():
        score_series, coverage_series = (
            build_pillar_score(
                norm_df,
                df,
                weights,
                pillar
            )
        )

        dash[
            f"Saeule_{pillar}"
        ] = score_series

        dash[
            f"Coverage_{pillar}"
        ] = coverage_series

    # WICHTIG:
    # Kein Hard-Nulling der gesamten Positionierungs-Säule bei
    # CFTC-Ausfall. CNN Fear & Greed kann weiterhin valide sein.
    # build_pillar_score() reduziert stattdessen nur die Coverage
    # und renormalisiert die verbleibenden verfügbaren Komponenten.

    # --------------------------------------------------------
    # FINAL REGIME SCORE – COVERAGE-AWARE
    # --------------------------------------------------------

    active_pillars = [
        pillar
        for pillar, weight
        in cfg[
            "Saeulen_Gewichte"
        ].items()
        if (
            weight > 0
            and
            f"Saeule_{pillar}"
            in dash.columns
            and
            f"Coverage_{pillar}"
            in dash.columns
        )
    ]

    if not active_pillars:
        return (
            pd.DataFrame(),
            status,
            feed_dates,
            feed_notes
        )

    pillar_cols = [
        f"Saeule_{pillar}"
        for pillar
        in active_pillars
    ]

    base_weights = pd.Series(
        {
            f"Saeule_{pillar}": float(
                cfg[
                    "Saeulen_Gewichte"
                ][pillar]
            )
            for pillar
            in active_pillars
        },
        dtype=float
    )

    if base_weights.sum() <= 0:
        return (
            pd.DataFrame(),
            status,
            feed_dates,
            feed_notes
        )

    base_weights = (
        base_weights
        / base_weights.sum()
    )

    pillar_matrix = (
        dash[pillar_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    coverage_matrix = pd.DataFrame(
        {
            f"Saeule_{pillar}": (
                pd.to_numeric(
                    dash[
                        f"Coverage_{pillar}"
                    ],
                    errors="coerce"
                )
                .fillna(0.0)
                .clip(0, 100)
                / 100.0
            )
            for pillar
            in active_pillars
        },
        index=dash.index
    )

    # Effektives Gewicht:
    # Modellgewicht × tatsächliche Datenabdeckung.
    effective_weights = (
        coverage_matrix
        .mul(
            base_weights,
            axis=1
        )
    )

    effective_weights = (
        effective_weights
        .where(
            pillar_matrix.notna(),
            0.0
        )
    )

    effective_weight_sum = (
        effective_weights
        .sum(axis=1)
    )

    weighted_sum = (
        pillar_matrix
        .fillna(0.0)
        .mul(
            effective_weights
        )
        .sum(axis=1)
    )

    dash["Final_Regime_Score"] = (
        weighted_sum
        .div(
            effective_weight_sum
            .replace(
                0,
                np.nan
            )
        )
        .clip(0, 100)
        .round(1)
    )

    # --------------------------------------------------------
    # MODEL CONSISTENCY INDEX
    # --------------------------------------------------------

    mci_weights = np.asarray(
        [
            base_weights[
                f"Saeule_{pillar}"
            ]
            for pillar
            in active_pillars
        ],
        dtype=float
    )

    dash["MCI"] = [
        calculate_mci(
            np.asarray(
                [
                    dash.at[
                        dash.index[i],
                        f"Saeule_{pillar}"
                    ]
                    for pillar
                    in active_pillars
                ],
                dtype=float
            ),
            mci_weights,
            np.asarray(
                [
                    dash.at[
                        dash.index[i],
                        f"Coverage_{pillar}"
                    ]
                    for pillar
                    in active_pillars
                ],
                dtype=float
            )
        )
        for i
        in range(len(dash))
    ]

    # --------------------------------------------------------
    # MODEL DATA COVERAGE – GEWICHTET
    # --------------------------------------------------------

    dash["Model_Data_Coverage"] = (
        coverage_matrix
        .mul(
            base_weights,
            axis=1
        )
        .sum(axis=1)
        .mul(100.0)
        .clip(0, 100)
        .round(1)
    )

    dash["Asset_Price"] = (
        price
        .reindex(
            dash.index
        )
        .ffill()
        .bfill()
    )

    dash["Options_Put_Call"] = (
        opt_pc
    )

    return (
        dash.dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        status,
        feed_dates,
        feed_notes
    )


# ============================================================
# 11. DATEN LADEN
# ============================================================

with st.spinner(
    f"Lade quantitative Daten für {selected_asset}..."
):
    (
        df_dash,
        feed_status,
        feed_dates,
        feed_notes
    ) = fetch_multi_asset_data(
        selected_asset
    )


with st.sidebar:
    for source, live in feed_status.items():
        date_label = format_feed_date(
            feed_dates.get(
                source
            )
        )

        fallback_label = (
            " <small><i>Fallback / Offline</i></small>"
            if not live
            else ""
        )

        st.markdown(
            (
                f"{'🟢' if live else '⚠️'} "
                f"<b>{source}</b> "
                f"<small>({date_label})</small>"
                f"{fallback_label}"
            ),
            unsafe_allow_html=True
        )


if df_dash.empty:
    st.error(
        "⚠️ Marktdaten konnten nicht geladen werden."
    )
    st.stop()


# ============================================================
# 13. MODELLQUALITÄT
# ============================================================

model_confidence, confidence_label = (
    calculate_model_confidence(
        feed_status,
        selected_asset,
        df_dash
    )
)

today = (
    df_dash.iloc[-1]
    .copy()
)

today["Delta_1D"] = (
    df_dash[
        "Final_Regime_Score"
    ].iloc[-1]
    -
    df_dash[
        "Final_Regime_Score"
    ].iloc[-2]
    if len(df_dash) >= 2
    else 0.0
)

today["Delta_1W"] = (
    df_dash["MCI"].iloc[-1]
    -
    df_dash["MCI"].iloc[-6]
    if len(df_dash) >= 6
    else 0.0
)


# ============================================================
# 14. HEADER
# ============================================================

st.title(
    "📊 Quant Regime Dashboard"
)

st.caption(
    f"Asset: **{selected_asset}** | "
    f"Stand: {df_dash.index[-1].strftime('%d.%m.%Y')}"
)

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
    "Fehlende Komponenten werden nicht als echte neutrale 50er-Werte "
    "gewertet. Der Final Regime Score berücksichtigt neben den "
    "fachlichen Säulengewichten auch die tatsächliche Datenabdeckung."
)

if model_confidence >= 85:
    st.success(
        f"📡 **Datenqualität: {confidence_label}** "
        f"({model_confidence:.0f}/100)"
    )

elif model_confidence >= 65:
    st.warning(
        f"📡 **Datenqualität: {confidence_label}** "
        f"({model_confidence:.0f}/100)"
    )

else:
    st.error(
        f"📡 **Datenqualität: {confidence_label}** "
        f"({model_confidence:.0f}/100) – "
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

st.subheader(
    "📊 Put/Call Ratio – zusätzlicher Positionierungsfilter"
)

opt_pc = float(
    today.get(
        "Options_Put_Call",
        np.nan
    )
)

if np.isfinite(opt_pc):
    if opt_pc >= 1.20:
        pc_interpretation = (
            "🟢 Stark Put-lastig"
        )
        pc_bias = (
            "Kontraindikativ bullisch"
        )

    elif opt_pc >= 1.00:
        pc_interpretation = (
            "🟢 Eher Put-lastig"
        )
        pc_bias = (
            "Leicht bullisch"
        )

    elif opt_pc >= 0.80:
        pc_interpretation = (
            "🟡 Neutral"
        )
        pc_bias = "Neutral"

    elif opt_pc >= 0.60:
        pc_interpretation = (
            "🟠 Eher Call-lastig"
        )
        pc_bias = (
            "Leicht bärisch"
        )

    else:
        pc_interpretation = (
            "🔴 Stark Call-lastig"
        )
        pc_bias = (
            "Kontraindikativ bärisch"
        )

else:
    pc_interpretation = (
        "⚪ Keine Daten"
    )

    pc_bias = (
        "Nicht verfügbar"
    )


pc1, pc2, pc3 = (
    st.columns(3)
)

with pc1:
    st.metric(
        (
            "Options Put/Call "
            f"({ASSET_CONFIGS[selected_asset]['options_proxy']})"
        ),
        (
            f"{opt_pc:.2f}"
            if np.isfinite(opt_pc)
            else "n/a"
        )
    )

with pc2:
    st.metric(
        "P/C Interpretation",
        pc_interpretation
    )

with pc3:
    st.metric(
        "Kontra-Signal",
        pc_bias
    )

st.caption(
    "Der PCR wird ausschließlich als zusätzlicher "
    "Positionierungs-/Sentimentfilter angezeigt und "
    "verändert den Final Regime Score nicht."
)

with st.expander(
    "ℹ️ PCR-Methodik"
):
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

Der PCR ist kein eigenständiges Einstiegssignal.
"""
    )

st.warning(
    "⚠️ Futures selbst besitzen keine Put/Call-Ratio. "
    "ES → SPY, NQ → QQQ, Gold → GLD, WTI → USO und "
    "EUR/USD → FXE sind ausdrücklich Options-Proxys."
)

st.markdown(
    "🔗 **Externe PCR-Referenz:** "
    "[Tools-for-Trader – Market Monitor (täglich aktualisiert)]"
    "(https://tools-for-trader.de/market-monitor-taeglich-aktualisiert/)"
)


# ============================================================
# 17. VOLATILITÄTS-ALARM
# ============================================================

current_vola = float(
    today.get(
        "Raw_Volatility",
        np.nan
    )
)

limit = VOLA_THRESHOLDS.get(
    selected_asset,
    30.0
)

vt = (
    ASSET_CONFIGS[
        selected_asset
    ][
        "volatility_ticker"
    ]
)

if np.isfinite(current_vola):
    if current_vola >= limit:
        st.error(
            f"🚨 **VOLATILITÄTS-ALARM:** "
            f"{vt} bei **{current_vola:.2f}** "
            f"(Grenzwert {limit:.1f})."
        )

    elif current_vola >= limit * .8:
        st.warning(
            f"⚠️ **Erhöhte Volatilität:** "
            f"{vt} bei **{current_vola:.2f}**."
        )

else:
    st.warning(
        f"⚠️ {vt} aktuell nicht verfügbar."
    )


# ============================================================
# 18. INTRADAY TRADING BIAS
# ============================================================

st.markdown("---")
st.markdown(
    "### 🎯 Intraday Trading Bias"
)

score = float(
    today.Final_Regime_Score
)

mci = float(
    today.MCI
)

if score >= 60:
    bias = (
        "🟢 BULLISCH (Long Bias)"
    )

    rule = (
        f"Bevorzugt Long-Setups bei {selected_asset} suchen."
    )

    pos = (
        "100% Standardsize"
        if mci >= 70
        else
        "75% Size"
        if mci >= 50
        else
        "50% Size"
    )

elif score <= 40:
    bias = (
        "🔴 BÄRISCH (Short Bias)"
    )

    rule = (
        f"Bevorzugt Short-Setups bei {selected_asset} suchen."
    )

    pos = (
        "100% Standardsize"
        if mci >= 70
        else
        "75% Size"
        if mci >= 50
        else
        "50% Size"
    )

else:
    bias = (
        "🟡 NEUTRAL / RANGE"
    )

    rule = (
        "Keine klare Trendrichtung. "
        "Nur selektive Setups."
    )

    pos = (
        "50% Size"
    )


if (
    np.isfinite(current_vola)
    and current_vola >= limit
):
    pos = (
        "FLAT / Max 25% Size"
    )

if model_confidence < 65:
    pos = (
        "FLAT / Max 25% Size"
    )

b1, b2, b3 = (
    st.columns(3)
)

with b1:
    st.metric(
        "Handelsrichtung",
        bias
    )

with b2:
    st.metric(
        "Positionsgröße",
        pos
    )

with b3:
    st.metric(
        "Fokus",
        (
            "Trend-Follow"
            if abs(score - 50) > 15
            else "Mean-Reversion"
        )
    )

st.info(
    f"**Übergeordnete Regel:** {rule}"
)


# ============================================================
# 19. GOOGLE RETAIL SENTIMENT
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Retail Sentiment (Google Trends)"
)

st.caption(
    "Unabhängiger Kontraindikator auf Basis "
    "des Suchverhaltens von Privatanlegern."
)

(
    contra,
    spread,
    trends_live,
    trends_data_date
) = fetch_google_trends_sentiment(
    selected_asset
)

add_feed_status(
    feed_status,
    "Google Trends Retail Sentiment",
    trends_live
)

feed_dates[
    "Google Trends Retail Sentiment"
] = trends_data_date

feed_notes[
    "Google Trends Retail Sentiment"
] = (
    "TrendSpy / pytrends"
    if trends_live
    else "Google Trends unavailable"
)

with st.sidebar:
    trends_date_label = (
        format_feed_date(
            trends_data_date
        )
    )

    trends_fallback_label = (
        " <small><i>Fallback / Offline</i></small>"
        if not trends_live
        else ""
    )

    st.markdown(
        (
            f"{'🟢' if trends_live else '⚠️'} "
            "<b>Google Trends Retail Sentiment</b> "
            f"<small>({trends_date_label})</small>"
            f"{trends_fallback_label}"
        ),
        unsafe_allow_html=True
    )

g1, g2, g3 = (
    st.columns(3)
)

with g1:
    st.metric(
        "Google Retail Score (0-100)",
        f"{contra} / 100",
        f"Net Spread: {spread:+.2f} σ",
        delta_color="inverse"
    )

with g2:
    if contra >= 65:
        st.success(
            "🟢 Panik-Ausschlag: "
            "kontraindikativ potenziell positiv."
        )

    elif contra <= 35:
        st.error(
            "🔴 Gier-Ausschlag: "
            "mögliche Überhitzung."
        )

    else:
        st.info(
            "🟡 Ausgeglichenes Sentiment."
        )

with g3:
    cfg = TREND_KEYWORD_MAP[
        selected_asset
    ]

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
        "quelle": (
            "FRED API & Yahoo Finance"
        ),
        "funktion": (
            "Zinsumfeld, Zentralbank-Liquidität "
            "und Dollar-Stärke."
        ),
        "links": [
            (
                "FRED: Fed Total Assets (WALCL)",
                "https://fred.stlouisfed.org/series/WALCL"
            ),
            (
                "FRED: TGA Account (WTREGEN)",
                "https://fred.stlouisfed.org/series/WTREGEN"
            ),
            (
                "FRED: Reverse Repo (RRPONTSYD)",
                "https://fred.stlouisfed.org/series/RRPONTSYD"
            ),
            (
                "FRED: 10Y Real Yields (DFII10)",
                "https://fred.stlouisfed.org/series/DFII10"
            ),
            (
                "FRED: Fed Funds Rate (FEDFUNDS)",
                "https://fred.stlouisfed.org/series/FEDFUNDS"
            ),
            (
                "Yahoo: US Dollar Index (DX-Y.NYB)",
                "https://finance.yahoo.com/quote/DX-Y.NYB"
            )
        ]
    },

    "Positionierung": {
        "quelle": (
            "CFTC COT Non-Commercials & CNN Fear & Greed"
        ),
        "funktion": (
            "Spekulative institutionelle Positionierung und "
            "CNN-Sentiment. Die Options-P/C-Ratio wird separat "
            "angezeigt und hat 0 % Modellgewicht."
        ),
        "links": [
            (
                "CFTC: Commitment of Traders",
                "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
            ),
            (
                "CFTC: Futures Only / Combined",
                "https://publicreporting.cftc.gov/stories/s/r4w3-av2u"
            ),
            (
                "CNN: Fear & Greed Index",
                "https://edition.cnn.com/markets/fear-and-greed"
            ),
            (
                "Yahoo Finance Options",
                "https://finance.yahoo.com/"
            ),
            (
                "Tools-for-Trader: Market Monitor / Put-Call-Ratio",
                "https://tools-for-trader.de/market-monitor-taeglich-aktualisiert/"
            )
        ]
    },

    "Marktinterna": {
        "quelle": (
            "Yahoo Finance"
        ),
        "funktion": (
            "Preis-Momentum, Marktvolatilität "
            "und Risikoappetit."
        ),
        "links": [
            (
                "Yahoo: VIX",
                "https://finance.yahoo.com/quote/%5EVIX"
            ),
            (
                "Yahoo: VXN",
                "https://finance.yahoo.com/quote/%5EVXN"
            ),
            (
                "Yahoo: GVZ",
                "https://finance.yahoo.com/quote/%5EGVZ"
            ),
            (
                "Yahoo: OVX",
                "https://finance.yahoo.com/quote/%5EOVX"
            ),
            (
                "Yahoo: EVZ",
                "https://finance.yahoo.com/quote/%5EEVZ"
            )
        ]
    },

    "Technischer_Trend": {
        "quelle": (
            "Yahoo Finance"
        ),
        "funktion": (
            "200-Tage-Trend, 50-Tage-Trend "
            "und RSI-Momentum."
        ),
        "links": [
            (
                "Yahoo: Chart & Technicals",
                (
                    "https://finance.yahoo.com/quote/"
                    f"{ASSET_CONFIGS[selected_asset]['ticker']}"
                )
            )
        ]
    },

    "Fundamentale_Faktoren": {
        "quelle": (
            "yfinance Valuation Measures / Multpl / FRED"
        ),
        "funktion": (
            "Historische Aktienindex-Bewertung bzw. Rohstoff-Lagerbestände. "
            "S&P 500 nutzt SPY/Multpl, Nasdaq 100 QQQ-Valuation-History, "
            "WTI die US-Rohöllagerbestände. Für Assets ohne aktive "
            "Fundamental-Säule ist die Gewichtung 0 %."
        ),
        "links": [
            (
                "Multpl: S&P 500 PE Ratio",
                "https://www.multpl.com/s-p-500-pe-ratio"
            ),
            (
                "Yahoo Finance: QQQ Key Statistics",
                "https://finance.yahoo.com/quote/QQQ/key-statistics/"
            ),
            (
                "FRED: Crude Oil Stocks",
                "https://fred.stlouisfed.org/series/WCESTUS1"
            )
        ]
    },

    "Fruehwarnindikatoren": {
        "quelle": (
            "Yahoo Finance / Cboe-Indexdaten"
        ),
        "funktion": (
            "Kreditmarkt-Proxy und Anleihenvolatilität; "
            "bei S&P 500/Nasdaq zusätzlich VVIX als "
            "Volatilität-der-Volatilität-Frühwarnsignal."
        ),
        "links": [
            (
                "Yahoo: HYG High Yield ETF",
                "https://finance.yahoo.com/quote/HYG"
            ),
            (
                "Yahoo: LQD Investment Grade ETF",
                "https://finance.yahoo.com/quote/LQD"
            ),
            (
                "Yahoo: MOVE Index",
                "https://finance.yahoo.com/quote/%5EMOVE"
            ),
            (
                "Yahoo: VVIX Index",
                "https://finance.yahoo.com/quote/%5EVVIX"
            ),
            (
                "Cboe: VVIX / Historical Volatility Indices",
                "https://www.cboe.com/tradable-products/vix/vix-historical-data"
            )
        ]
    }
}


# ============================================================
# 21. TREIBER-ANALYSE
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Treiber-Analyse (Die 6 Säulen)"
)

cols = st.columns(3)

saeulen = [
    c
    for c in df_dash.columns
    if c.startswith(
        "Saeule_"
    )
]

for i, s in enumerate(
    saeulen
):
    val = float(
        today.get(
            s,
            50
        )
    )

    raw = s.replace(
        "Saeule_",
        ""
    )

    label = raw.replace(
        "_",
        " "
    )

    emoji = (
        "🟢"
        if val > 60
        else
        "🔴"
        if val < 40
        else
        "🟡"
    )

    weight = (
        ASSET_CONFIGS[
            selected_asset
        ][
            "Saeulen_Gewichte"
        ].get(
            raw,
            0
        )
        * 100
    )

    coverage = float(
        today.get(
            f"Coverage_{raw}",
            0
        )
    )

    with cols[
        i % 3
    ]:
        if weight <= 0:
            st.metric(
                f"{label} ⚪",
                "Nicht aktiv"
            )

            st.caption(
                "Gewicht: 0% | "
                "Für dieses Asset bewusst nicht im Regime-Score aktiv."
            )

        else:
            st.metric(
                f"{label} {emoji}",
                f"{val:.1f}"
            )

            st.caption(
                f"Gewicht: {weight:.0f}% | "
                f"Datenabdeckung: {coverage:.0f}%"
            )

        if (
            raw == "Fundamentale_Faktoren"
            and weight > 0
        ):
            fundamental_feed = (
                "Bewertungsdaten (S&P 500 PE)"
                if selected_asset == "S&P 500"
                else
                "Bewertungsdaten (Nasdaq 100 / QQQ PE)"
                if selected_asset == "Nasdaq 100"
                else
                "FRED API (Makro & Fed)"
                if selected_asset == "WTI Crude Oil"
                else None
            )

            if fundamental_feed is not None:
                fundamental_live = (
                    feed_status.get(
                        fundamental_feed,
                        False
                    )
                )

                fundamental_note = (
                    feed_notes.get(
                        fundamental_feed,
                        ""
                    )
                )

                if fundamental_live:
                    st.caption(
                        "Fundamental-Feed aktiv. "
                        "Ein niedriger Säulenwert ist dann ein "
                        "bearishes Bewertungssignal und kein Datenfehler."
                    )
                else:
                    st.caption(
                        "⚠️ Fundamental-Feed derzeit nicht verfügbar."
                    )

                if fundamental_note:
                    st.caption(
                        f"Quelle/Diagnose: {fundamental_note}"
                    )

        if raw in saeulen_details:
            d = saeulen_details[
                raw
            ]

            with st.expander(
                "Details, Daten & Links"
            ):
                st.markdown(
                    f"**⚖️ Gewichtung:** "
                    f"{weight:.0f}%"
                )

                st.markdown(
                    f"**📡 Quelle:** "
                    f"{d['quelle']}"
                )

                st.markdown(
                    f"**⚙️ Funktion:** "
                    f"{d['funktion']}"
                )

                st.markdown(
                    "**🔗 Live-Datenquellen:**"
                )

                for title, url in d[
                    "links"
                ]:
                    st.markdown(
                        f"• [{title}]({url})"
                    )


# ============================================================
# 22. INTRADAY EXECUTION CHECKLISTE
# ============================================================

st.markdown("---")

st.subheader(
    "⚡ Intraday Execution Checkliste & Filter"
)

trend = float(
    today.get(
        "Saeule_Technischer_Trend",
        50
    )
)

early = float(
    today.get(
        "Saeule_Fruehwarnindikatoren",
        50
    )
)

macro = float(
    today.get(
        "Saeule_Makroökonomie",
        50
    )
)

trend_ok = (
    trend > 55
)

bond_ok = (
    early > 35
)

macro_ok = (
    macro > 50
)

now = pd.Timestamp.now(
    tz="Europe/Berlin"
)

wd = now.weekday()

hexensabbat = (
    now.month
    in [3, 6, 9, 12]
    and wd == 4
    and 15 <= now.day <= 21
)

profile = {
    0: (
        "Montag: Preisfindung & "
        "Weekly Initial Balance"
    ),
    1: (
        "Dienstag: Trendetablierung"
    ),
    2: (
        "Mittwoch: Trendfortsetzung "
        "oder Mid-Week Reversal"
    ),
    3: (
        "Donnerstag: Momentum & Volatilität"
    ),
    4: (
        "Freitag: Wochenschluss & Profit-Taking"
    )
}.get(
    wd,
    "Wochenende: Märkte geschlossen"
)

a, b = (
    st.columns(2)
)

with a:
    st.markdown(
        "#### 1. Strukturelle Filter"
    )

    x1 = st.checkbox(
        (
            "Trendkonformität "
            "(Marktstruktur / gleitende Durchschnitte intakt)"
        ),
        value=trend_ok,
        key="chk_trend_det"
    )

    x2 = st.checkbox(
        (
            "Anleihen- & Kreditmärkte stabil "
            "(kein akuter Stress)"
        ),
        value=bond_ok,
        key="chk_bond_det"
    )

    x3 = st.checkbox(
        (
            "Makro-Umgebung im Rücken "
            f"(Score: {macro:.0f})"
        ),
        value=macro_ok,
        key="chk_makro_det"
    )

    x4 = st.checkbox(
        (
            "Statistisches Tagesprofil beachtet "
            f"({profile})"
        ),
        value=True,
        key="chk_day_profile"
    )

with b:
    st.markdown(
        "#### 2. Timing & Risikomanagement"
    )

    x5 = st.checkbox(
        (
            "Keine High-Impact News "
            "(CPI, FOMC, NFP) "
            "in den nächsten 60 Minuten"
        ),
        value=True,
        key="chk_news_det"
    )

    x6 = st.checkbox(
        (
            "Kein Hexensabbat / Ketten-Verfall"
        ),
        value=not hexensabbat,
        key="chk_opex_det"
    )

    x7 = st.checkbox(
        (
            "CRV mindestens 1:2 zum nächsten "
            "charttechnischen Ziel"
        ),
        value=True,
        key="chk_crv_det"
    )

    x8 = st.checkbox(
        (
            "US-Eröffnung / Initial Balance abgewartet"
        ),
        value=True,
        key="chk_time_det"
    )

count = sum(
    [
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8
    ]
)

st.progress(
    count / 8
)

st.caption(
    f"✅ **{count} von 8 Kriterien erfüllt**"
)

execution_data_ok = (
    model_confidence >= 65
    and
    today.Model_Data_Coverage >= 60
)

if (
    count == 8
    and score > 55
    and execution_data_ok
):
    st.success(
        "🟢 **EXECUTION FREIGABE (GO):** "
        "Alle Filter erfüllt, Datenqualität ausreichend "
        "und bullischer Long-Bias."
    )

elif (
    count == 8
    and score < 45
    and execution_data_ok
):
    st.error(
        "🔴 **EXECUTION FREIGABE (SHORT):** "
        "Alle Filter erfüllt, Datenqualität ausreichend "
        "und bärischer Bias."
    )

elif not execution_data_ok:
    st.error(
        "🛑 **DATA QUALITY GATE:** "
        "Keine Execution-Freigabe, weil zu viele "
        "Live-Komponenten fehlen."
    )

elif score < 40:
    st.error(
        "🔴 **STOP / KEIN TRADE:** "
        "Marktregime auf Defense."
    )

else:
    st.warning(
        "🟡 **CAUTION / WARNUNG:** "
        "Gemischte Signale."
    )


# ============================================================
# 23. REGIME-HISTORIE
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Regime-Historie & Asset Preis "
    "(Letzte 12 Monate)"
)

plot = (
    df_dash.tail(252)
)

fig = make_subplots(
    specs=[
        [
            {
                "secondary_y": True
            }
        ]
    ]
)

fig.add_trace(
    go.Scatter(
        x=plot.index,
        y=plot[
            "Final_Regime_Score"
        ],
        name=(
            "Regime Score (0-100)"
        ),
        fill="tozeroy"
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=plot.index,
        y=plot[
            "Asset_Price"
        ],
        name=(
            f"{selected_asset} Preis"
        ),
        line=dict(
            width=2
        )
    ),
    secondary_y=True
)

fig.update_yaxes(
    title_text=(
        "Regime Score"
    ),
    range=[
        0,
        100
    ],
    secondary_y=False
)

fig.update_yaxes(
    title_text=(
        "Asset Preis"
    ),
    secondary_y=True
)

fig.update_layout(
    height=400,
    margin=dict(
        l=0,
        r=0,
        t=30,
        b=0
    ),
    hovermode=(
        "x unified"
    ),
    paper_bgcolor=(
        "rgba(0,0,0,0)"
    ),
    plot_bgcolor=(
        "rgba(0,0,0,0)"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 24. ASSET-REGELN
# ============================================================

st.markdown("---")

with st.expander(
    "📚 Asset-Regeln & Bezugsquellen"
):
    st.caption(
        "Das Dashboard verbindet sechs Säulen aus "
        "Makroökonomie, Positionierung, Marktinterna, "
        "technischem Trend, fundamentalen Faktoren "
        "und Frühwarnindikatoren."
    )

    asset_rule_cols = (
        st.columns(2)
    )

    for i, (
        asset_name,
        rule_cfg
    ) in enumerate(
        ASSET_RULES.items()
    ):
        with asset_rule_cols[
            i % 2
        ]:
            st.markdown(
                f"### 🎯 {asset_name} – "
                f"{rule_cfg['profil']}"
            )

            st.markdown(
                "**Modellregeln:**"
            )

            for rule in rule_cfg[
                "regeln"
            ]:
                st.markdown(
                    f"• {rule}"
                )

            st.markdown(
                "**Bezugsquellen:**"
            )

            for source in rule_cfg[
                "quellen"
            ]:
                st.markdown(
                    f"• {source}"
                )

            st.markdown("---")


# ============================================================
# 25. GEWICHTUNGEN
# ============================================================

with st.expander(
    "⚖️ Aktuelle Modellgewichtungen"
):
    weights = (
        ASSET_CONFIGS[
            selected_asset
        ][
            "Saeulen_Gewichte"
        ]
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Säule": list(
                    weights
                ),
                "Gewichtung": [
                    f"{v * 100:.0f}%"
                    for v
                    in weights.values()
                ]
            }
        ),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Die Gewichtungen sind fachlich begründete "
        "Startgewichte und nicht empirisch "
        "backtest-optimiert."
    )


# ============================================================
# 26. SYSTEM & API STATUS
# ============================================================

with st.expander(
    "📡 System & API Status Details"
):
    st.write(
        "Live-Verbindungsstatus zu den externen Datenquellen:"
    )

    sc = (
        st.columns(2)
    )

    for i, (
        feed,
        status_value
    ) in enumerate(
        feed_status.items()
    ):
        date_label = format_feed_date(
            feed_dates.get(
                feed
            )
        )

        note = (
            feed_notes.get(
                feed,
                ""
            )
        )

        note_html = (
            f"<br><small>{note}</small>"
            if note
            else ""
        )

        sc[
            i % 2
        ].markdown(
            (
                f"<b>{feed}</b> "
                f"<small>({date_label})</small>: "
                f"{'✅ Verbunden' if status_value else '⚠️ Fallback aktiv / Offline'}"
                f"{note_html}"
            ),
            unsafe_allow_html=True
        )

    st.caption(
        "Das Datum hinter jeder Quelle bezeichnet nach Möglichkeit "
        "den tatsächlichen letzten Datenstand bzw. Handelstag und nicht "
        "bloß den Zeitpunkt des API-Abrufs. Bei kombinierten FRED-Daten "
        "wird konservativ der älteste notwendige Teil-Datenstand gezeigt. "
        "Fallback-/Offline-Komponenten erhalten kein eigenes Modellgewicht; "
        "fehlende Daten reduzieren Model Data Coverage und Model Confidence."
    )


# ============================================================
# 27. ABSCHLUSS-HINWEIS
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Modellhinweis: Der Final Regime Score ist ein "
    "quantitatives Entscheidungs- und Regimefilter-Modell "
    "und keine Anlageberatung. Der LQD/HYG-Wert ist ein "
    "Kreditmarkt-Proxy und kein tatsächlicher Credit Spread. "
    "Die Put/Call-Komponente ist separat ausgewiesen. "
    "Futures selbst besitzen keine Put/Call-Ratio."
)
