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


# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Asset Regime Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 0A. ZENTRALE MODELLSCHWELLEN
# ============================================================

BULL_THRESHOLD = 60.0
BEAR_THRESHOLD = 40.0

STRONG_BULL_THRESHOLD = 75.0
STRONG_BEAR_THRESHOLD = 25.0

MCI_HIGH = 70.0
MCI_MEDIUM = 50.0

HIGH_IMPACT_VOLA_FACTOR = 1.0
ELEVATED_VOLA_FACTOR = 0.8


# ============================================================
# 1. ASSET CONFIGURATIONS & WEIGHTS
# ============================================================

ASSET_CONFIGS = {

    "S&P 500": {
        "ticker": "^GSPC",
        "volatility_ticker": "^VIX",
        "cot_code": "E-MINI S&P 500",

        "options_proxy": "SPY",
        "futures_pc_ticker": "ES=F",
        "options_pc_ticker": "SPY",

        "invert_inverts": [
            "vix_score",
            "pe_valuation",
            "credit_proxy",
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
                "cot_commercials": .50,
                "fear_greed": .50,
                "futures_put_call": .0,
                "options_put_call": .0
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

        "options_proxy": "QQQ",
        "futures_pc_ticker": "NQ=F",
        "options_pc_ticker": "QQQ",

        "invert_inverts": [
            "vix_score",
            "pe_valuation",
            "credit_proxy",
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
                "cot_commercials": .50,
                "fear_greed": .50,
                "futures_put_call": .0,
                "options_put_call": .0
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

        "options_proxy": "GLD",
        "futures_pc_ticker": "GC=F",
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
                "cot_commercials": .80,
                "fear_greed": .20,
                "futures_put_call": .0,
                "options_put_call": .0
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

        "options_proxy": "USO",
        "futures_pc_ticker": "CL=F",
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
                "cot_commercials": .80,
                "fear_greed": .20,
                "futures_put_call": .0,
                "options_put_call": .0
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

        "options_proxy": "FXE",
        "futures_pc_ticker": "6E=F",
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
                "cot_commercials": .70,
                "fear_greed": .30,
                "futures_put_call": .0,
                "options_put_call": .0
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
            "Positionierung: CFTC Commercials + CNN Fear & Greed",
            "Marktinterna: 20-Tage-Momentum + VIX",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: S&P-500-KGV als Bewertungsfilter",
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE"
        ],

        "quellen": [
            "Yahoo Finance: ^GSPC, ^VIX, DX=F, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders",
            "CNN: Fear & Greed",
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS",
            "Yahoo Finance / Multpl / WSJ: Bewertungsdaten"
        ]
    },


    "Nasdaq 100": {
        "profil": "US-Technologie-/Growth-Index",

        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Net Liquidity",
            "Positionierung: CFTC Commercials + CNN Fear & Greed",
            "Marktinterna: 20-Tage-Momentum + VXN",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: Bewertungsfilter",
            "Frühwarnung: LQD/HYG-Kreditproxy + MOVE"
        ],

        "quellen": [
            "Yahoo Finance: NQ=F, ^VXN, DX=F, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders",
            "CNN: Fear & Greed",
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS"
        ]
    },


    "Gold (XAU/USD)": {
        "profil": "Gold / Edelmetall",

        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Net Liquidity",
            "Positionierung: CFTC Commercials mit höherem Gewicht",
            "Marktinterna: OBV-Momentum + GVZ",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamentale Säule: bewusst mit 0 % gewichtet",
            "Frühwarnung: Kreditproxy + MOVE"
        ],

        "quellen": [
            "Yahoo Finance: GC=F, ^GVZ, DX=F, ^MOVE, HYG, LQD",
            "CFTC: Commitment of Traders",
            "FRED: Realrenditen, Fed Funds, Liquidität",
            "Yahoo Finance / ETF-Optionen: GLD als Optionsproxy"
        ]
    },


    "WTI Crude Oil": {
        "profil": "WTI-Rohöl",

        "regeln": [
            "Makro: insbesondere USD und Liquiditäts-/Zinsumfeld",
            "Positionierung: CFTC Commercials mit höherem Gewicht",
            "Marktinterna: OBV-Momentum + OVX",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamental: US-Rohöllagerbestände (WCESTUS1)",
            "Frühwarnindikatoren: in diesem Asset mit 0 % gewichtet"
        ],

        "quellen": [
            "Yahoo Finance: CL=F, ^OVX, DX=F, HYG, LQD",
            "CFTC: Commitment of Traders",
            "FRED: WCESTUS1 und Makrodaten",
            "Yahoo Finance / ETF-Optionen: USO als Optionsproxy"
        ]
    },


    "EUR/USD": {
        "profil": "Devisenpaar Euro gegen US-Dollar",

        "regeln": [
            "Makro: Fed-Politik, Realrenditen, USD und Liquidität",
            "Positionierung: CFTC EURO FX Commercials",
            "Marktinterna: 20-Tage-Momentum + EVZ",
            "Trend: Abstand zur 50-/200-Tage-Linie + RSI",
            "Fundamentale Säule: bewusst mit 0 % gewichtet",
            "Frühwarnung: Kreditproxy + MOVE"
        ],

        "quellen": [
            "Yahoo Finance: EURUSD=X, ^EVZ, DX=F, ^MOVE, HYG, LQD",
            "CFTC: EURO FX Commitment of Traders",
            "FRED: Fed Funds, Realrenditen und Liquiditätsdaten",
            "Yahoo Finance / ETF-Optionen: FXE als Optionsproxy"
        ]
    }
}


# ============================================================
# 2. BASIS-GEWICHTUNGEN
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
        "credit_proxy": .60,
        "move_index": .40
    }
}


LOOKBACK_CONFIG = {

    "fed_policy": 1260,
    "real_yields": 756,
    "net_liquidity": 756,
    "credit_proxy": 756,
    "usd_index": 504,
    "inventories": 756,

    "cot_commercials": 252,
    "fear_greed": 252,
    "market_momentum": 252,
    "obv_momentum": 252,
    "vix_score": 252,
    "move_index": 252
}


VOLA_THRESHOLDS = {

    "S&P 500": 30.0,
    "Nasdaq 100": 35.0,
    "Gold (XAU/USD)": 25.0,
    "WTI Crude Oil": 45.0,
    "EUR/USD": 15.0
}


# ============================================================
# 3. GOOGLE TRENDS
# ============================================================

TREND_KEYWORD_MAP = {

    "S&P 500": {
        "geo": "US",
        "lang": "en-US",

        "bull": [
            "buy stocks",
            "buy the dip"
        ],

        "bear": [
            "stock market crash",
            "recession"
        ]
    },


    "Nasdaq 100": {
        "geo": "US",
        "lang": "en-US",

        "bull": [
            "tech stocks",
            "buy the dip"
        ],

        "bear": [
            "market crash",
            "tech bubble"
        ]
    },


    "Gold (XAU/USD)": {
        "geo": "DE",
        "lang": "de-DE",

        "bull": [
            "Gold kaufen",
            "Goldmünzen"
        ],

        "bear": [
            "Gold verkaufen",
            "Altgold"
        ]
    },


    "WTI Crude Oil": {
        "geo": "DE",
        "lang": "de-DE",

        "bull": [
            "Heizöl kaufen",
            "Spritpreise"
        ],

        "bear": [
            "Ölpreis crash",
            "Öl verkaufen"
        ]
    },


    "EUR/USD": {
        "geo": "DE",
        "lang": "de-DE",

        "bull": [
            "Euro kaufen",
            "EUR USD kaufen"
        ],

        "bear": [
            "Euro verkaufen",
            "EUR USD verkaufen"
        ]
    }
}


@st.cache_data(ttl=21600)
def fetch_google_trends_sentiment(asset_name):

    cfg = TREND_KEYWORD_MAP.get(
        asset_name,
        TREND_KEYWORD_MAP["S&P 500"]
    )

    try:

        p = TrendReq(
            hl=cfg["lang"],
            tz=360,
            retries=2,
            backoff_factor=.2
        )

        kws = cfg["bull"] + cfg["bear"]

        p.build_payload(
            kws,
            timeframe="today 3-m",
            geo=cfg["geo"]
        )

        d = p.interest_over_time()

        if d.empty:
            return np.nan, np.nan, False

        if "isPartial" in d:
            d = d.drop(
                columns="isPartial"
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
                /
                sd
            )

        vb = [
            x for x in cfg["bull"]
            if x in d.columns
        ]

        vr = [
            x for x in cfg["bear"]
            if x in d.columns
        ]

        if not vb or not vr:
            return np.nan, np.nan, False

        spread = (
            sum(
                z(d[x])
                for x in vb
            )
            / len(vb)
            -
            sum(
                z(d[x])
                for x in vr
            )
            / len(vr)
        )

        spread = (
            spread
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if spread.empty:
            return np.nan, np.nan, False

        latest = float(
            spread.iloc[-1]
        )

        score = float(
            np.clip(
                50 - latest * 15,
                0,
                100
            )
        )

        return (
            round(score, 1),
            round(latest, 2),
            True
        )

    except Exception:

        return np.nan, np.nan, False


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def strip_timezone(x):

    dt = pd.to_datetime(
        x,
        errors="coerce"
    )

    if isinstance(
        dt,
        pd.Series
    ):

        if getattr(
            dt.dt,
            "tz",
            None
        ) is not None:

            return dt.dt.tz_convert(
                None
            )

        return dt

    if isinstance(
        dt,
        pd.DatetimeIndex
    ):

        if dt.tz is not None:

            return dt.tz_convert(
                None
            )

        return dt

    return dt


def normalize_to_percentile(
    series,
    lookback=252,
    invert=False
):

    if series is None:

        return None

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    if s.isna().all():

        return pd.Series(
            np.nan,
            index=series.index
        )

    s = (
        s.ffill()
        .bfill()
    )

    m = s.rolling(
        lookback,
        min_periods=20
    ).mean()

    sd = (
        s.rolling(
            lookback,
            min_periods=20
        )
        .std()
        .replace(
            0,
            np.nan
        )
    )

    z = (
        (s - m)
        /
        sd
    )

    out = pd.Series(
        norm.cdf(z) * 100,
        index=s.index
    )

    if invert:
        out = 100 - out

    return (
        out
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .clip(
            0,
            100
        )
    )


def valuation_score_from_pe(pe):

    """
    Bewertungs-Score:
    niedrigeres KGV = günstiger = höherer Score.

    Die Grenzen sind bewusst transparent und
    nicht backtest-optimiert.
    """

    if not np.isfinite(pe):
        return np.nan

    pe = float(pe)

    if pe <= 15:
        return 95.0

    if pe <= 18:
        return 85.0

    if pe <= 21:
        return 75.0

    if pe <= 24:
        return 60.0

    if pe <= 27:
        return 50.0

    if pe <= 30:
        return 40.0

    if pe <= 35:
        return 25.0

    return 10.0


def calculate_mci(
    scores,
    weights
):

    s = np.asarray(
        scores,
        float
    )

    w = np.asarray(
        weights,
        float
    )

    v = (
        np.isfinite(s)
        &
        np.isfinite(w)
        &
        (w > 0)
    )

    s = s[v]
    w = w[v]

    if (
        len(s) == 0
        or w.sum() <= 0
    ):

        return np.nan

    w = w / w.sum()

    mean = np.average(
        s,
        weights=w
    )

    sd = np.sqrt(
        np.average(
            (s - mean) ** 2,
            weights=w
        )
    )

    return round(
        float(
            np.clip(
                100 * (
                    1 - sd / 50
                ),
                0,
                100
            )
        ),
        1
    )


def get_regime_label(score):

    if not np.isfinite(score):
        return "⚪ Daten unvollständig"

    if score >= STRONG_BULL_THRESHOLD:
        return "🟢 Risk-On (Extrem Bullisch)"

    if score >= BULL_THRESHOLD:
        return "🟢 Expansion (Bullisch)"

    if score >= 50:
        return "🟡 Übergangsphase (Leicht Bullisch)"

    if score >= BEAR_THRESHOLD:
        return "🟡 Neutral"

    if score >= STRONG_BEAR_THRESHOLD:
        return "🟠 Risk-Off (Bärisch)"

    return "🔴 Stressphase (Stark Bärisch)"


def safe_reindex_series(
    source,
    target
):

    if (
        not isinstance(
            source,
            pd.Series
        )
        or source.empty
    ):

        return None

    s = source.copy()

    s.index = (
        strip_timezone(
            s.index
        )
        .floor("D")
    )

    s = (
        s[
            ~s.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    t = (
        strip_timezone(
            target
        )
        .floor("D")
    )

    r = (
        s.reindex(
            t,
            method="ffill"
        )
        .ffill()
    )

    r.index = target

    return r


def extract_yfinance_field(
    data,
    field
):

    if data is None:
        return None

    if not isinstance(
        data.columns,
        pd.MultiIndex
    ):

        if field in data.columns:

            x = data[field]

            if isinstance(
                x,
                pd.Series
            ):

                return x

        return None

    levels = [
        list(
            data.columns
            .get_level_values(i)
        )
        for i in range(
            data.columns.nlevels
        )
    ]

    if field in levels[0]:

        x = data[field]

        if isinstance(
            x,
            pd.DataFrame
        ):

            return x

    if field in levels[-1]:

        x = data.xs(
            field,
            axis=1,
            level=-1
        )

        if isinstance(
            x,
            pd.DataFrame
        ):

            return x

    return None


def weighted_row_score(
    frame,
    weights
):

    """
    Berechnet einen gewichteten Score je Zeile.

    Fehlende Indikatoren werden nicht als 50 behandelt.
    Stattdessen werden nur vorhandene Indikatoren
    verwendet und deren Gewichte neu normiert.
    """

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float
    )

    for idx in frame.index:

        values = []
        used_weights = []

        for col, weight in weights.items():

            if (
                col not in frame.columns
                or weight <= 0
            ):
                continue

            value = frame.at[
                idx,
                col
            ]

            if np.isfinite(value):

                values.append(
                    float(value)
                )

                used_weights.append(
                    float(weight)
                )

        if values:

            w = np.asarray(
                used_weights,
                float
            )

            w = w / w.sum()

            result.at[idx] = np.average(
                values,
                weights=w
            )

    return result


def weighted_final_score(
    frame,
    pillar_weights
):

    """
    Final Score mit dynamischer
    Renormalisierung bei fehlenden Säulen.
    """

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float
    )

    for idx in frame.index:

        values = []
        weights = []

        for pillar, weight in pillar_weights.items():

            if weight <= 0:
                continue

            col = (
                f"Saeule_{pillar}"
            )

            if col not in frame.columns:
                continue

            value = frame.at[
                idx,
                col
            ]

            if np.isfinite(value):

                values.append(
                    float(value)
                )

                weights.append(
                    float(weight)
                )

        if values:

            w = np.asarray(
                weights,
                float
            )

            w = w / w.sum()

            result.at[idx] = np.average(
                values,
                weights=w
            )

    return (
        result
        .clip(
            0,
            100
        )
        .round(1)
    )


# ============================================================
# 5. CNN FEAR & GREED
# ============================================================

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():

    try:

        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://edition.cnn.com/"
            },
            timeout=8
        )

        if r.status_code != 200:
            return None, False

        h = (
            r.json()
            .get(
                "fear_and_greed_historical",
                {}
            )
            .get(
                "data",
                []
            )
        )

        d = pd.DataFrame(h)

        if (
            d.empty
            or not {
                "x",
                "y"
            }.issubset(
                d.columns
            )
        ):

            return None, False

        d["Date"] = (
            strip_timezone(
                pd.to_datetime(
                    d.x,
                    unit="ms",
                    errors="coerce"
                )
            )
            .dt.floor("D")
        )

        d["y"] = pd.to_numeric(
            d.y,
            errors="coerce"
        )

        d = (
            d.dropna(
                subset=[
                    "Date",
                    "y"
                ]
            )
            .drop_duplicates(
                "Date",
                keep="last"
            )
        )

        return (
            d.set_index(
                "Date"
            ).y.sort_index(),
            True
        )

    except Exception:

        return None, False


# ============================================================
# 6. CFTC COT
# ============================================================

@st.cache_data(ttl=86400)
def fetch_cot_data(
    asset_search_string
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    year = pd.Timestamp.now().year

    frames = []

    for yr in [
        year - 1,
        year
    ]:

        url = (
            f"https://www.cftc.gov/files/dera/history/"
            f"fut_com_txt_{yr}.zip"
        )

        try:

            r = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            if r.status_code != 200:
                continue

            with zipfile.ZipFile(
                io.BytesIO(
                    r.content
                )
            ) as z:

                names = z.namelist()

                if not names:
                    continue

                d = pd.read_csv(
                    z.open(
                        names[0]
                    ),
                    low_memory=False
                )

                if d.empty:
                    continue

                market_column = d.columns[0]

                rows = d[
                    d[
                        market_column
                    ]
                    .astype(str)
                    .str.contains(
                        asset_search_string,
                        case=False,
                        na=False
                    )
                ]

                if not rows.empty:

                    frames.append(
                        rows
                    )

        except Exception:

            continue

    if not frames:

        return None, False

    try:

        d = pd.concat(
            frames,
            ignore_index=True
        )

        date_columns = [
            c for c in d.columns
            if "As_of_Date" in str(c)
        ]

        long_columns = [
            c for c in d.columns
            if "Comm_Positions_Long_All"
            in str(c)
        ]

        short_columns = [
            c for c in d.columns
            if "Comm_Positions_Short_All"
            in str(c)
        ]

        if (
            not date_columns
            or not long_columns
            or not short_columns
        ):

            return None, False

        d["Date"] = (
            strip_timezone(
                pd.to_datetime(
                    d[
                        date_columns[0]
                    ].astype(str),
                    errors="coerce"
                )
            )
            .dt.floor("D")
        )

        d["Net_Commercials"] = (

            pd.to_numeric(
                d[
                    long_columns[0]
                ],
                errors="coerce"
            )

            -

            pd.to_numeric(
                d[
                    short_columns[0]
                ],
                errors="coerce"
            )
        )

        d = (
            d.dropna(
                subset=[
                    "Date"
                ]
            )
            .drop_duplicates(
                "Date",
                keep="last"
            )
        )

        return (
            d.set_index(
                "Date"
            )
            .Net_Commercials
            .sort_index(),

            True
        )

    except Exception:

        return None, False


# ============================================================
# 7. OPTIONS PUT/CALL RATIO
# ============================================================

@st.cache_data(ttl=1800)
def fetch_option_put_call(
    ticker
):

    try:

        t = yf.Ticker(
            ticker
        )

        expiries = t.options

        if not expiries:

            return (
                np.nan,
                False,
                "Keine Optionslaufzeiten"
            )

        # Erste verfügbare Laufzeit.
        # Bewusst nur als einfacher Proxy.
        expiry = expiries[0]

        chain = t.option_chain(
            expiry
        )

        puts = chain.puts
        calls = chain.calls

        if puts.empty and calls.empty:

            return (
                np.nan,
                False,
                "Optionskette leer"
            )

        pv = (
            pd.to_numeric(
                puts.get(
                    "volume"
                ),
                errors="coerce"
            )
            .fillna(0)
            .sum()
            if not puts.empty
            else 0
        )

        cv = (
            pd.to_numeric(
                calls.get(
                    "volume"
                ),
                errors="coerce"
            )
            .fillna(0)
            .sum()
            if not calls.empty
            else 0
        )

        if cv <= 0:

            return (
                np.nan,
                False,
                "Kein Call-Volumen"
            )

        return (
            float(
                pv / cv
            ),
            True,
            f"{ticker} Optionen ({expiry})"
        )

    except Exception:

        return (
            np.nan,
            False,
            "Optionskette nicht verfügbar"
        )


# ============================================================
# 8. FRED API KEY CONFIG
# ============================================================

FRED_API_KEY = ""

try:

    if "FRED_API_KEY" in st.secrets:

        FRED_API_KEY = st.secrets[
            "FRED_API_KEY"
        ]

except Exception:

    pass


# ============================================================
# 9. S&P 500 KGV
# ============================================================

@st.cache_data(ttl=21600)
def fetch_sp500_pe():

    try:

        ticker = yf.Ticker(
            "^GSPC"
        )

        info = ticker.info

        pe = info.get(
            "trailingPE"
        )

        if pe is not None:

            pe = float(pe)

            if (
                np.isfinite(pe)
                and pe > 0
            ):

                return pe, True

    except Exception:

        pass

    return np.nan, False


# ============================================================
# 10. SIDEBAR CONTROL
# ============================================================

with st.sidebar:

    st.title(
        "⚙️ Multi-Asset Selector"
    )

    selected_asset = st.selectbox(
        "🎯 Asset auswählen",
        list(
            ASSET_CONFIGS
        ),
        index=0
    )

    st.markdown("---")

    st.markdown(
        "### 📡 API Live-Feed Monitor"
    )


# ============================================================
# 11. MULTI-ASSET DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600)
def fetch_multi_asset_data(
    selected_asset
):

    cfg = ASSET_CONFIGS[
        selected_asset
    ]

    status = {}

    tickers = {

        "asset":
            cfg["ticker"],

        "vix":
            cfg["volatility_ticker"],

        "dxy":
            "DX=F",

        "move":
            "^MOVE",

        "hyg":
            "HYG",

        "lqd":
            "LQD"
    }

    # --------------------------------------------------------
    # YFINANCE
    # --------------------------------------------------------

    try:

        data = yf.download(
            list(
                tickers.values()
            ),

            period="5y",
            interval="1d",

            auto_adjust=False,

            progress=False,
            threads=True
        )

    except Exception:

        return (
            pd.DataFrame(),

            {
                "yFinance (Preis & Tech)": False
            }
        )

    if data.empty:

        return (
            pd.DataFrame(),

            {
                "yFinance (Preis & Tech)": False
            }
        )

    close = extract_yfinance_field(
        data,
        "Close"
    )

    if close is None:

        return (
            pd.DataFrame(),

            {
                "yFinance (Preis & Tech)": False
            }
        )

    if isinstance(
        close,
        pd.Series
    ):

        close = close.to_frame()

    if isinstance(
        close.columns,
        pd.MultiIndex
    ):

        close.columns = (
            close.columns
            .get_level_values(-1)
        )

    close = (
        close
        .rename(
            columns={
                v: k
                for k, v
                in tickers.items()
            }
        )
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .sort_index()
    )

    status[
        "yFinance (Preis & Tech)"
    ] = bool(
        "asset" in close
        and not close[
            "asset"
        ].dropna().empty
    )

    if not status[
        "yFinance (Preis & Tech)"
    ]:

        return (
            pd.DataFrame(),
            status
        )

    price = (
        close[
            "asset"
        ]
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

    asset_vol = pd.Series(
        np.nan,
        index=price.index
    )

    if vol is not None:

        if isinstance(
            vol,
            pd.Series
        ):

            asset_vol = (
                vol
                .reindex(
                    price.index
                )
            )

            has_volume = (
                asset_vol.notna().any()
            )

        else:

            if isinstance(
                vol.columns,
                pd.MultiIndex
            ):

                vol.columns = (
                    vol.columns
                    .get_level_values(-1)
                )

            vol = vol.rename(
                columns={
                    v: k
                    for k, v
                    in tickers.items()
                }
            )

            if "asset" in vol:

                asset_vol = (
                    vol[
                        "asset"
                    ]
                    .reindex(
                        price.index
                    )
                )

                has_volume = (
                    asset_vol.notna().any()
                )

    status[
        "Volumen / Orderflow Feed"
    ] = bool(has_volume)

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

    df[
        "distance_50ma"
    ] = (
        (
            price - ma50
        )
        /
        ma50.replace(
            0,
            np.nan
        )
        * 100
    )

    df[
        "distance_200ma"
    ] = (
        (
            price - ma200
        )
        /
        ma200.replace(
            0,
            np.nan
        )
        * 100
    )

    # --------------------------------------------------------
    # ROBUSTER RSI
    # --------------------------------------------------------

    delta = price.diff()

    gain = (
        delta.clip(
            lower=0
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    loss = (
        -delta.clip(
            upper=0
        )
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

    rsi = (
        100
        -
        100
        /
        (
            1 + rs
        )
    )

    # Edge Cases:
    # Gewinnserie ohne Verluste -> RSI 100
    # Verlustserie ohne Gewinne -> RSI 0

    rsi = rsi.where(
        ~(
            (loss == 0)
            & (gain > 0)
        ),
        100
    )

    rsi = rsi.where(
        ~(
            (gain == 0)
            & (loss > 0)
        ),
        0
    )

    df[
        "rsi_momentum"
    ] = rsi.clip(
        0,
        100
    )

    # --------------------------------------------------------
    # VOLATILITÄT
    # --------------------------------------------------------

    if "vix" in close:

        df[
            "vix_score"
        ] = (
            close[
                "vix"
            ]
            .reindex(
                df.index
            )
            .ffill()
        )

    else:

        df[
            "vix_score"
        ] = np.nan

    status[
        f"{cfg['volatility_ticker']} Volatilität"
    ] = bool(
        df[
            "vix_score"
        ].notna().any()
    )

    # --------------------------------------------------------
    # USD
    # --------------------------------------------------------

    if "dxy" in close:

        df[
            "usd_index"
        ] = (
            close[
                "dxy"
            ]
            .reindex(
                df.index
            )
            .ffill()
        )

    else:

        df[
            "usd_index"
        ] = np.nan

    status[
        "Yahoo Finance USD Index"
    ] = bool(
        df[
            "usd_index"
        ].notna().any()
    )

    # --------------------------------------------------------
    # MOVE
    # --------------------------------------------------------

    if "move" in close:

        df[
            "move_index"
        ] = (
            close[
                "move"
            ]
            .reindex(
                df.index
            )
            .ffill()
        )

    else:

        df[
            "move_index"
        ] = np.nan

    status[
        "Yahoo Finance MOVE"
    ] = bool(
        df[
            "move_index"
        ].notna().any()
    )

    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------

    if (
        "lqd" in close
        and "hyg" in close
    ):

        lqd = (
            close[
                "lqd"
            ]
            .reindex(
                df.index
            )
            .ffill()
        )

        hyg = (
            close[
                "hyg"
            ]
            .reindex(
                df.index
            )
            .ffill()
        )

        df[
            "credit_proxy"
        ] = (
            lqd
            /
            hyg.replace(
                0,
                np.nan
            )
        )

    else:

        df[
            "credit_proxy"
        ] = np.nan

    status[
        "LQD/HYG Kreditproxy"
    ] = bool(
        df[
            "credit_proxy"
        ].notna().any()
    )

    # --------------------------------------------------------
    # MARKET MOMENTUM
    # --------------------------------------------------------

    df[
        "market_momentum"
    ] = (
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

        clean_volume = (
            pd.to_numeric(
                asset_vol,
                errors="coerce"
            )
            .fillna(0)
        )

        obv = pd.Series(
            np.where(
                delta > 0,
                clean_volume,

                np.where(
                    delta < 0,
                    -clean_volume,
                    0.0
                )
            ),

            index=price.index
        ).cumsum()

        ema = obv.ewm(
            span=50,
            adjust=False
        ).mean()

        df[
            "obv_momentum"
        ] = (
            (
                obv - ema
            )
            /
            ema.abs().replace(
                0,
                np.nan
            )
            * 100
        )

    else:

        df[
            "obv_momentum"
        ] = np.nan

    # ========================================================
    # S&P 500 / NASDAQ FUNDAMENTAL DATA
    # ========================================================

    pe_value = np.nan
    pe_ok = False

    if selected_asset in [
        "S&P 500",
        "Nasdaq 100"
    ]:

        pe_value, pe_ok = (
            fetch_sp500_pe()
        )

    if selected_asset == "S&P 500":

        df[
            "pe_valuation"
        ] = pe_value

    elif selected_asset == "Nasdaq 100":

        # Kein künstlicher Nasdaq-KGV-Wert.
        # Ohne belastbare Daten bleibt der Indikator leer.
        df[
            "pe_valuation"
        ] = np.nan

    status[
        "Bewertungsdaten"
    ] = bool(pe_ok)

    # ========================================================
    # FRED
    # ========================================================

    fred_ok = False

    if FRED_API_KEY:

        try:

            f = Fred(
                api_key=FRED_API_KEY
            )

            wal = safe_reindex_series(
                f.get_series(
                    "WALCL"
                ),
                df.index
            )

            tga = safe_reindex_series(
                f.get_series(
                    "WTREGEN"
                ),
                df.index
            )

            rrp = safe_reindex_series(
                f.get_series(
                    "RRPONTSYD"
                ),
                df.index
            )

            # Alle drei Serien sind in Millionen USD.
            # Daher direkte Subtraktion.
            # Anschließend Umrechnung in Milliarden USD.

            if (
                wal is not None
                and tga is not None
                and rrp is not None
            ):

                df[
                    "net_liquidity"
                ] = (
                    wal
                    - tga
                    - rrp
                ) / 1000.0

            else:

                df[
                    "net_liquidity"
                ] = np.nan

            df[
                "fed_policy"
            ] = safe_reindex_series(
                f.get_series(
                    "FEDFUNDS"
                ),
                df.index
            )

            df[
                "real_yields"
            ] = safe_reindex_series(
                f.get_series(
                    "DFII10"
                ),
                df.index
            )

            if selected_asset == "WTI Crude Oil":

                df[
                    "inventories"
                ] = safe_reindex_series(
                    f.get_series(
                        "WCESTUS1"
                    ),
                    df.index
                )

            fred_ok = True

        except Exception:

            fred_ok = False

    if not fred_ok:

        for c in [
            "fed_policy",
            "real_yields",
            "net_liquidity"
        ]:

            df[c] = np.nan

        if selected_asset == "WTI Crude Oil":

            df[
                "inventories"
            ] = np.nan

    status[
        "FRED API (Makro & Fed)"
    ] = fred_ok

    # ========================================================
    # CFTC COT
    # ========================================================

    cot, cot_ok = fetch_cot_data(
        cfg["cot_code"]
    )

    status[
        f"CFTC COT ({cfg['cot_code']})"
    ] = cot_ok

    if cot is not None:

        df[
            "cot_commercials"
        ] = safe_reindex_series(
            cot,
            df.index
        )

    else:

        df[
            "cot_commercials"
        ] = np.nan

    # ========================================================
    # CNN FEAR & GREED
    # ========================================================

    fg, fg_ok = (
        fetch_fear_and_greed()
    )

    status[
        "CNN Fear & Greed"
    ] = fg_ok

    if isinstance(
        fg,
        pd.Series
    ):

        df[
            "fear_greed"
        ] = safe_reindex_series(
            fg,
            df.index
        )

    else:

        df[
            "fear_greed"
        ] = np.nan

    # ========================================================
    # OPTIONS PUT/CALL
    # ========================================================

    opt_pc, opt_ok, opt_source = (
        fetch_option_put_call(
            cfg[
                "options_pc_ticker"
            ]
        )
    )

    df[
        "options_put_call"
    ] = (
        opt_pc
        if np.isfinite(
            opt_pc
        )
        else np.nan
    )

    status[
        "Options Put/Call Proxy "
        f"({cfg['options_pc_ticker']})"
    ] = opt_ok

    # ========================================================
    # NORMALISIERUNG
    # ========================================================

    norm = pd.DataFrame(
        index=df.index
    )

    inverts = cfg[
        "invert_inverts"
    ]

    for col in df.columns:

        # KGV wird NICHT als Zeitreihe
        # normalisiert, sondern direkt bewertet.
        if col == "pe_valuation":

            pe_score = (
                valuation_score_from_pe(
                    pe_value
                )
            )

            norm[col] = pe_score

            continue

        norm[col] = (
            normalize_to_percentile(
                df[col],

                LOOKBACK_CONFIG.get(
                    col,
                    252
                ),

                col in inverts
            )
        )

    # ========================================================
    # SÄULEN
    # ========================================================

    dash = pd.DataFrame(
        index=df.index
    )

    dash[
        "Raw_Volatility"
    ] = df[
        "vix_score"
    ]

    active = {
        k: v.copy()
        for k, v in
        SUB_WEIGHTS_BASE.items()
    }

    for cat, weights_cfg in cfg[
        "Sub_Gewichte"
    ].items():

        active[cat] = (
            weights_cfg.copy()
        )

    for pillar, inds in active.items():

        cols = [
            c for c in inds
            if (
                c in norm.columns
                and inds[c] > 0
            )
        ]

        if cols:

            pillar_frame = (
                norm[cols]
            )

            pillar_weights = {
                c: inds[c]
                for c in cols
            }

            dash[
                f"Saeule_{pillar}"
            ] = weighted_row_score(
                pillar_frame,
                pillar_weights
            )

        else:

            dash[
                f"Saeule_{pillar}"
            ] = np.nan

    # ========================================================
    # FINAL REGIME SCORE
    # ========================================================

    dash[
        "Final_Regime_Score"
    ] = weighted_final_score(
        dash,
        cfg[
            "Saeulen_Gewichte"
        ]
    )

    # ========================================================
    # MCI
    # ========================================================

    pillar_columns = []

    for pillar, weight in cfg[
        "Saeulen_Gewichte"
    ].items():

        if weight <= 0:
            continue

        column = (
            f"Saeule_{pillar}"
        )

        if column in dash.columns:

            pillar_columns.append(
                (
                    column,
                    weight
                )
            )

    mci_values = []

    for idx in dash.index:

        values = []
        weights = []

        for col, weight in pillar_columns:

            value = dash.at[
                idx,
                col
            ]

            if np.isfinite(value):

                values.append(
                    value
                )

                weights.append(
                    weight
                )

        if values:

            mci_values.append(
                calculate_mci(
                    values,
                    weights
                )
            )

        else:

            mci_values.append(
                np.nan
            )

    dash[
        "MCI"
    ] = mci_values

    # ========================================================
    # ASSET PRICE
    # ========================================================

    dash[
        "Asset_Price"
    ] = (
        price
        .reindex(
            dash.index
        )
        .ffill()
    )

    # ========================================================
    # AKTUELLE PCR-WERTE
    # ========================================================

    dash[
        "Options_Put_Call"
    ] = opt_pc

    # ========================================================
    # DATENQUALITÄT
    # ========================================================

    required_core = [
        "Final_Regime_Score",
        "MCI"
    ]

    dash[
        "Model_Data_Complete"
    ] = (
        dash[
            required_core
        ]
        .notna()
        .all(axis=1)
    )

    return (
        dash.dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        status
    )


# ============================================================
# 12. DATEN LADEN & SIDEBAR UPDATES
# ============================================================

with st.spinner(
    f"Lade quantitative Daten für {selected_asset}..."
):

    df_dash, feed_status = (
        fetch_multi_asset_data(
            selected_asset
        )
    )


with st.sidebar:

    for source, live in feed_status.items():

        st.markdown(
            f"{'🟢' if live else '⚠️'} "
            f"**{source}**"
            f"{' *(Fallback / Offline)*' if not live else ''}"
        )


if df_dash.empty:

    st.error(
        "⚠️ Marktdaten konnten nicht geladen werden."
    )

    st.stop()


# ============================================================
# 13. AKTUELLE WERTE
# ============================================================

heute = (
    df_dash
    .iloc[-1]
    .copy()
)

heute[
    "Delta_1D"
] = (
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

heute[
    "Delta_1W"
] = (
    df_dash[
        "MCI"
    ].iloc[-1]
    -
    df_dash[
        "MCI"
    ].iloc[-6]
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
    f"Stand: "
    f"{df_dash.index[-1].strftime('%d.%m.%Y')}"
)

st.markdown("---")


# ============================================================
# 15. FINAL SCORE + MCI
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Final Regime Score",
        f"{heute.Final_Regime_Score:.1f} / 100",
        f"{heute.Delta_1D:+.1f} (Heute)"
    )

with c2:

    mci_display = (
        f"{heute.MCI:.1f}%"
        if np.isfinite(
            heute.MCI
        )
        else "n/a"
    )

    st.metric(
        "Model Consistency Index",
        mci_display,
        (
            f"{heute.Delta_1W:+.1f} "
            "(vs. Vorwoche)"
            if np.isfinite(
                heute.MCI
            )
            else "n/a"
        ),
        delta_color="off"
    )


st.caption(
    "Der Model Consistency Index ist ein "
    "**proprietärer Konsistenzindex des Modells**. "
    "Er misst die Übereinstimmung der verfügbaren "
    "Modellsäulen und ist weder eine statistische "
    "Wahrscheinlichkeit noch eine Trefferquote."
)


st.info(
    f"**Aktuelles Marktregime ({selected_asset}):** "
    f"{get_regime_label(heute.Final_Regime_Score)}"
)


# ============================================================
# 16. PUT/CALL RATIO
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Options Put/Call Ratio – zusätzlicher Filter"
)

opt_pc = float(
    heute.get(
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

        pc_bias = (
            "Neutral"
        )

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


pc1, pc2, pc3 = st.columns(3)

with pc1:

    st.metric(
        "Options Put/Call Proxy",
        (
            f"{opt_pc:.2f}"
            if np.isfinite(opt_pc)
            else "n/a"
        )
    )

with pc2:

    st.metric(
        "Interpretation",
        pc_interpretation
    )

with pc3:

    st.metric(
        "Kontra-Signal",
        pc_bias
    )


st.caption(
    f"Proxy-Markt: "
    f"**{ASSET_CONFIGS[selected_asset]['options_pc_ticker']}**. "
    "Die Ratio basiert auf Optionsvolumen und ist kein "
    "direkter Futures-Put/Call-Wert."
)


with st.expander(
    "ℹ️ PCR-Interpretation"
):

    st.markdown(
        """
**Put/Call Ratio = Put-Volumen ÷ Call-Volumen**

• **> 1,20:** deutlich Put-lastig → kann konträr als bullisches Signal gewertet werden

• **1,00–1,20:** leicht Put-lastig → leicht konträr bullisch

• **0,80–1,00:** weitgehend neutral

• **0,60–0,80:** eher Call-lastig → leicht konträr bärisch

• **< 0,60:** stark Call-lastig → mögliche Überhitzung

Der PCR wird ausschließlich als zusätzlicher
Positionierungs-/Sentimentfilter verwendet und
verändert den Final Regime Score nicht.

Die Werte sind als Marktproxy zu verstehen und
nicht als isoliertes Einstiegssignal.
"""
    )


# ============================================================
# 17. VOLATILITÄTS-ALARM
# ============================================================

current_vola = float(
    heute.get(
        "Raw_Volatility",
        np.nan
    )
)

limit = VOLA_THRESHOLDS.get(
    selected_asset,
    30.0
)

vt = ASSET_CONFIGS[
    selected_asset
][
    "volatility_ticker"
]


if not np.isfinite(
    current_vola
):

    st.warning(
        f"⚠️ **Volatilitätsdaten nicht verfügbar:** "
        f"{vt}. Positionsgröße wird deshalb nicht "
        f"automatisch aus der Volatilität abgeleitet."
    )

elif current_vola >= (
    limit
    * HIGH_IMPACT_VOLA_FACTOR
):

    st.error(
        f"🚨 **VOLATILITÄTS-ALARM:** "
        f"{vt} bei **{current_vola:.2f}** "
        f"(Grenzwert {limit:.1f})."
    )

elif current_vola >= (
    limit
    * ELEVATED_VOLA_FACTOR
):

    st.warning(
        f"⚠️ **Erhöhte Volatilität:** "
        f"{vt} bei **{current_vola:.2f}**."
    )


# ============================================================
# 18. INTRADAY TRADING BIAS
# ============================================================

st.markdown("---")

st.markdown(
    "### 🎯 Intraday Trading Bias"
)

score = float(
    heute.Final_Regime_Score
)

mci = float(
    heute.MCI
) if np.isfinite(
    heute.MCI
) else np.nan


if score >= BULL_THRESHOLD:

    bias = (
        "🟢 BULLISCH (Long Bias)"
    )

    rule = (
        f"Bevorzugt Long-Setups "
        f"bei {selected_asset} suchen."
    )

    if np.isfinite(mci):

        if mci >= MCI_HIGH:
            pos = "100% Standardsize"

        elif mci >= MCI_MEDIUM:
            pos = "75% Size"

        else:
            pos = "50% Size"

    else:

        pos = "DATA UNAVAILABLE"

elif score <= BEAR_THRESHOLD:

    bias = (
        "🔴 BÄRISCH (Short Bias)"
    )

    rule = (
        f"Bevorzugt Short-Setups "
        f"bei {selected_asset} suchen."
    )

    if np.isfinite(mci):

        if mci >= MCI_HIGH:
            pos = "100% Standardsize"

        elif mci >= MCI_MEDIUM:
            pos = "75% Size"

        else:
            pos = "50% Size"

    else:

        pos = "DATA UNAVAILABLE"

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


# ------------------------------------------------------------
# VOLATILITÄT MUSS FÜR AUTOMATISCHES SIZING VORHANDEN SEIN
# ------------------------------------------------------------

if not np.isfinite(
    current_vola
):

    pos = (
        "DATA UNAVAILABLE / "
        "KEIN AUTOMATISCHES SIZING"
    )

elif current_vola >= limit:

    pos = (
        "FLAT / Max 25% Size"
    )


b1, b2, b3 = st.columns(3)

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
            if abs(
                score - 50
            ) > 15
            else
            "Mean-Reversion"
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
    "Experimenteller, unabhängiger Kontraindikator "
    "auf Basis des Suchverhaltens. "
    "Der Wert fließt bewusst NICHT in den Core Regime Score ein."
)


contra, spread, trends_live = (
    fetch_google_trends_sentiment(
        selected_asset
    )
)


g1, g2, g3 = st.columns(3)

with g1:

    st.metric(
        "Google Retail Score (0-100)",
        (
            f"{contra} / 100"
            if np.isfinite(
                contra
            )
            else "n/a"
        ),
        (
            f"Net Spread: {spread:+.2f} σ"
            if np.isfinite(
                spread
            )
            else "n/a"
        ),
        delta_color="inverse"
    )

with g2:

    if not np.isfinite(
        contra
    ):

        st.warning(
            "⚪ Google Trends aktuell nicht verfügbar."
        )

    elif contra >= 65:

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
**🔍 Getrackte Parameter:**

* **Region:** `{cfg['geo']}`

* **Euphorie:** {', '.join(repr(x) for x in cfg['bull'])}

* **Panik:** {', '.join(repr(x) for x in cfg['bear'])}

* **Status:** {'🟢 Live' if trends_live else '🔴 Offline'}
"""
    )


# ============================================================
# 20. SECHS SÄULEN – DETAILS & QUELLEN
# ============================================================

saeulen_details = {

    "Makroökonomie": {

        "quelle":
            "FRED API & Yahoo Finance",

        "funktion":
            "Zinsumfeld, Zentralbank-Liquidität "
            "und Dollar-Stärke.",

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
                "Yahoo: US Dollar Index",
                "https://finance.yahoo.com/quote/DX-Y.NYB"
            )
        ]
    },


    "Positionierung": {

        "quelle":
            "CFTC COT, CNN Fear & Greed "
            "& Options-Put/Call",

        "funktion":
            "Institutionelle Positionierung, "
            "Sentiment und Optionspositionierung.",

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
            )
        ]
    },


    "Marktinterna": {

        "quelle":
            "Yahoo Finance",

        "funktion":
            "Preis-Momentum, Marktvolatilität "
            "und Risikoappetit.",

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

        "quelle":
            "Yahoo Finance",

        "funktion":
            "200-Tage-Trend, 50-Tage-Trend "
            "und RSI-Momentum.",

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

        "quelle":
            "Yahoo Finance / FRED / Multpl / WSJ",

        "funktion":
            "Bewertung bzw. "
            "Rohstoff-Lagerbestände.",

        "links": [

            (
                "FRED: Crude Oil Stocks",
                "https://fred.stlouisfed.org/series/WCESTUS1"
            ),

            (
                "Multpl: S&P 500 PE Ratio",
                "https://www.multpl.com/s-p-500-pe-ratio"
            ),

            (
                "WSJ: P/E Ratios",
                "https://www.wsj.com/market-data/stocks/peyields"
            )
        ]
    },


    "Fruehwarnindikatoren": {

        "quelle":
            "Yahoo Finance",

        "funktion":
            "Kreditmarkt-Proxy und "
            "Anleihenvolatilität.",

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
    c for c in df_dash.columns
    if c.startswith(
        "Saeule_"
    )
]


for i, s in enumerate(
    saeulen
):

    val = float(
        heute.get(
            s,
            np.nan
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

    if not np.isfinite(val):

        emoji = "⚪"

        display_value = "n/a"

    else:

        emoji = (
            "🟢"
            if val > 60
            else
            "🔴"
            if val < 40
            else
            "🟡"
        )

        display_value = (
            f"{val:.1f}"
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

    with cols[
        i % 3
    ]:

        st.metric(
            f"{label} {emoji}",
            display_value
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

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


# ============================================================
# 22. INTRADAY EXECUTION CHECKLISTE
# ============================================================

st.markdown("---")

st.subheader(
    "⚡ Intraday Execution Checkliste & Filter"
)

trend = float(
    heute.get(
        "Saeule_Technischer_Trend",
        np.nan
    )
)

early = float(
    heute.get(
        "Saeule_Fruehwarnindikatoren",
        np.nan
    )
)

macro = float(
    heute.get(
        "Saeule_Makroökonomie",
        np.nan
    )
)


trend_ok = (
    np.isfinite(trend)
    and trend > 55
)

bond_ok = (
    np.isfinite(early)
    and early > 35
)

macro_ok = (
    np.isfinite(macro)
    and macro > 50
)


now = pd.Timestamp.now(
    tz="Europe/Berlin"
)

wd = now.weekday()

hexensabbat = (
    now.month in [
        3,
        6,
        9,
        12
    ]
    and wd == 4
    and 15 <= now.day <= 21
)


profile = {
    0:
        "Montag: Preisfindung & Weekly Initial Balance",

    1:
        "Dienstag: Trendetablierung",

    2:
        "Mittwoch: Trendfortsetzung oder Mid-Week Reversal",

    3:
        "Donnerstag: Momentum & Volatilität",

    4:
        "Freitag: Wochenschluss & Profit-Taking"

}.get(
    wd,
    "Wochenende: Märkte geschlossen"
)


a, b = st.columns(2)


with a:

    st.markdown(
        "#### 1. Strukturelle Filter"
    )

    x1 = st.checkbox(
        "Trendkonformität "
        "(Marktstruktur / gleitende Durchschnitte intakt)",

        value=trend_ok,

        key="chk_trend_det"
    )

    x2 = st.checkbox(
        "Anleihen- & Kreditmärkte stabil "
        "(kein akuter Stress)",

        value=bond_ok,

        key="chk_bond_det"
    )

    x3 = st.checkbox(
        f"Makro-Umgebung im Rücken "
        f"(Score: "
        f"{macro:.0f}"
        f")"
        if np.isfinite(macro)
        else
        "Makro-Umgebung im Rücken "
        "(Daten nicht vollständig)",

        value=macro_ok,

        key="chk_makro_det"
    )

    x4 = st.checkbox(
        f"Statistisches Tagesprofil beachtet "
        f"({profile})",

        value=True,

        key="chk_day_profile"
    )


with b:

    st.markdown(
        "#### 2. Timing & Risikomanagement"
    )

    x5 = st.checkbox(
        "Keine High-Impact News "
        "(CPI, FOMC, NFP) in den nächsten 60 Minuten",

        value=True,

        key="chk_news_det"
    )

    x6 = st.checkbox(
        "Kein Hexensabbat / Ketten-Verfall",

        value=not hexensabbat,

        key="chk_opex_det"
    )

    x7 = st.checkbox(
        "CRV mindestens 1:2 "
        "zum nächsten charttechnischen Ziel",

        value=True,

        key="chk_crv_det"
    )

    x8 = st.checkbox(
        "US-Eröffnung / Initial Balance abgewartet",

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


if (
    count == 8
    and score >= BULL_THRESHOLD
):

    st.success(
        "🟢 **EXECUTION FREIGABE (LONG):** "
        "Alle Filter erfüllt und bullischer Regime-Bias."
    )

elif (
    count == 8
    and score <= BEAR_THRESHOLD
):

    st.error(
        "🔴 **EXECUTION FREIGABE (SHORT):** "
        "Alle Filter erfüllt und bärischer Regime-Bias."
    )

elif score < BEAR_THRESHOLD:

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


plot = df_dash.tail(
    252
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

        name="Regime Score (0-100)",

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

        name=f"{selected_asset} Preis",

        line=dict(
            width=2
        )
    ),

    secondary_y=True
)


fig.update_yaxes(
    title_text="Regime Score",

    range=[
        0,
        100
    ],

    secondary_y=False
)


fig.update_yaxes(
    title_text="Asset Preis",

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

    hovermode="x unified",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)"
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
        "Das Dashboard verbindet für jedes ausgewählte Asset "
        "sechs Säulen aus Makroökonomie, Positionierung, "
        "Marktinterna, technischem Trend, fundamentalen Faktoren "
        "und Frühwarnindikatoren."
    )

    asset_rule_cols = st.columns(2)

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

            st.markdown(
                "---"
            )


# ============================================================
# 25. MODELLGEWICHTUNGEN
# ============================================================

with st.expander(
    "⚖️ Aktuelle Modellgewichtungen"
):

    weights = ASSET_CONFIGS[
        selected_asset
    ][
        "Saeulen_Gewichte"
    ]

    st.dataframe(
        pd.DataFrame(
            {
                "Säule":
                    list(weights),

                "Gewichtung":
                    [
                        f"{v * 100:.0f}%"
                        for v in weights.values()
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

    sc = st.columns(2)

    for i, (
        feed,
        status_value
    ) in enumerate(
        feed_status.items()
    ):

        sc[
            i % 2
        ].markdown(
            f"**{feed}:** "
            f"{'✅ Verbunden' if status_value else '⚠️ Offline / nicht verfügbar'}"
        )

    st.caption(
        "Nicht verfügbare Daten werden im Modell "
        "nicht künstlich als neutraler Wert 50 interpretiert. "
        "Stattdessen werden die betroffenen Gewichte "
        "dynamisch aus der Berechnung entfernt."
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
    "Die Put/Call-Komponente ist separat ausgewiesen und "
    "basiert auf dem jeweiligen ETF-Optionsproxy. "
    "Futures selbst besitzen keine Put/Call-Ratio. "
    "Fehlende Daten werden nicht als neutrale Signale "
    "behandelt. Die Modellgewichtungen sind nicht "
    "backtest-optimiert."
)