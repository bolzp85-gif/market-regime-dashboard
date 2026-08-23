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
# 0. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Multi-Asset Regime Dashboard",
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
        "options_proxy": "SPY",
        "futures_pc_ticker": "ES=F",
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
            "Multpl / WSJ: Bewertungsdaten"
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
            "FRED: WALCL, WTREGEN, RRPONTSYD, DFII10, FEDFUNDS",
            "Multpl / WSJ: Bewertungsdaten"
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
# ASSET-REGELN IM DASHBOARD
# ============================================================

st.markdown("---")
st.subheader("📚 Asset-Regeln & Bezugsquellen")

st.caption(
    "Das Dashboard verbindet für jedes ausgewählte Asset sechs Säulen "
    "aus Makroökonomie, Positionierung, Marktinterna, technischem Trend, "
    "fundamentalen Faktoren und Frühwarnindikatoren. Die folgenden Regeln "
    "beschreiben die verwendeten Datenquellen und deren Rolle im Modell."
)

asset_rule_cols = st.columns(2)

for i, (asset_name, rule_cfg) in enumerate(ASSET_RULES.items()):

    with asset_rule_cols[i % 2]:

        with st.expander(
            f"🎯 {asset_name} – {rule_cfg['profil']}"
        ):

            st.markdown("**Modellregeln:**")

            for rule in rule_cfg["regeln"]:
                st.markdown(f"• {rule}")

            st.markdown("**Bezugsquellen:**")

            for source in rule_cfg["quellen"]:
                st.markdown(f"• {source}")


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
    "inventories": 756
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


# ============================================================
# 3. GOOGLE TRENDS
# ============================================================

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
            return 50., 0., False

        if "isPartial" in d:
            d = d.drop(columns="isPartial")

        def z(s):

            s = pd.to_numeric(
                s,
                errors="coerce"
            )

            m = s.rolling(
                21,
                min_periods=5
            ).mean()

            sd = s.rolling(
                21,
                min_periods=5
            ).std().replace(0, np.nan)

            return (s - m) / sd

        vb = [
            x for x in cfg["bull"]
            if x in d
        ]

        vr = [
            x for x in cfg["bear"]
            if x in d
        ]

        if not vb or not vr:
            return 50., 0., False

        spread = (
            sum(z(d[x]) for x in vb) / len(vb)
            -
            sum(z(d[x]) for x in vr) / len(vr)
        )

        spread = (
            spread
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if spread.empty:
            return 50., 0., False

        latest = float(spread.iloc[-1])

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
        return 50., 0., False


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def strip_timezone(x):

    dt = pd.to_datetime(
        x,
        errors="coerce"
    )

    if isinstance(dt, pd.Series):

        return (
            dt.dt.tz_convert(None)
            if getattr(dt.dt, "tz", None) is not None
            else dt
        )

    if isinstance(dt, pd.DatetimeIndex):

        return (
            dt.tz_convert(None)
            if dt.tz is not None
            else dt
        )

    return dt


def normalize_to_percentile(
    series,
    lookback=252,
    invert=False
):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).ffill().bfill()

    if s.isna().all():
        return pd.Series(
            50.,
            index=series.index
        )

    m = s.rolling(
        lookback,
        min_periods=20
    ).mean()

    sd = s.rolling(
        lookback,
        min_periods=20
    ).std().replace(
        0,
        np.nan
    )

    z = (s - m) / sd

    out = pd.Series(
        norm.cdf(z) * 100,
        index=series.index
    )

    if invert:
        out = 100 - out

    return (
        out
        .replace([np.inf, -np.inf], np.nan)
        .clip(0, 100)
        .ffill()
        .bfill()
        .fillna(50.)
    )


def calculate_mci(scores, weights):

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
    )

    s = s[v]
    w = w[v]

    if len(s) == 0 or w.sum() <= 0:
        return 0.

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
                100 * (1 - sd / 50),
                0,
                100
            )
        ),
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

    if (
        not isinstance(source, pd.Series)
        or source.empty
    ):
        return None

    s = source.copy()

    s.index = (
        strip_timezone(s.index)
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

    if data is None:
        return None

    if not isinstance(
        data.columns,
        pd.MultiIndex
    ):

        return (
            data[field]
            if field in data.columns
            and isinstance(
                data[field],
                pd.Series
            )
            else None
        )

    if field in data.columns.get_level_values(0):

        x = data[field]

        return (
            x
            if isinstance(x, pd.DataFrame)
            else None
        )

    if field in data.columns.get_level_values(1):

        x = data.xs(
            field,
            axis=1,
            level=1
        )

        return (
            x
            if isinstance(x, pd.DataFrame)
            else None
        )

    return None


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
            return 55., False

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
            or not {"x", "y"}.issubset(d.columns)
        ):
            return 55., False

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

        d.y = pd.to_numeric(
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
            d.set_index("Date")
            .y
            .sort_index(),
            True
        )

    except Exception:
        return 55., False


# ============================================================
# 6. CFTC COT
# ============================================================

@st.cache_data(ttl=86400)
def fetch_cot_data(
    asset_search_string,
    combined=False
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
                io.BytesIO(r.content)
            ) as z:

                n = z.namelist()

                if not n:
                    continue

                d = pd.read_csv(
                    z.open(n[0]),
                    low_memory=False
                )

                mc = d.columns[0]

                rows = d[
                    d[mc]
                    .astype(str)
                    .str.contains(
                        asset_search_string,
                        case=False,
                        na=False
                    )
                ]

                if not rows.empty:
                    frames.append(rows)

        except Exception:
            continue

    if not frames:
        return None, False

    try:

        d = pd.concat(
            frames,
            ignore_index=True
        )

        dc = [
            c for c in d
            if "As_of_Date" in str(c)
        ]

        lc = [
            c for c in d
            if "Comm_Positions_Long_All" in str(c)
        ]

        sc = [
            c for c in d
            if "Comm_Positions_Short_All" in str(c)
        ]

        if (
            not dc
            or not lc
            or not sc
        ):
            return None, False

        d["Date"] = (
            strip_timezone(
                pd.to_datetime(
                    d[dc[0]].astype(str),
                    format="%Y%m%d",
                    errors="coerce"
                )
            )
            .dt.floor("D")
        )

        d["Net_Commercials"] = (
            pd.to_numeric(
                d[lc[0]],
                errors="coerce"
            )
            -
            pd.to_numeric(
                d[sc[0]],
                errors="coerce"
            )
        )

        d = (
            d.dropna(
                subset=["Date"]
            )
            .drop_duplicates(
                "Date",
                keep="last"
            )
        )

        return (
            d.set_index("Date")
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
def fetch_option_put_call(ticker):

    """
    Aggregierte Put/Call-Ratio:
    Put-Volumen / Call-Volumen.

    Es wird der erste verfügbare Optionsverfall
    des konfigurierten ETF-/Optionsproxys verwendet.

    Die Ratio wird ausschließlich als separater
    Positionierungsfilter verwendet.
    """

    try:

        t = yf.Ticker(ticker)

        expiries = t.options

        if not expiries:
            return (
                np.nan,
                False,
                "No option chain"
            )

        expiry = expiries[0]

        chain = t.option_chain(
            expiry
        )

        puts = chain.puts
        calls = chain.calls

        pv = (
            pd.to_numeric(
                puts.get("volume"),
                errors="coerce"
            )
            .fillna(0)
            .sum()
            if not puts.empty
            else 0
        )

        cv = (
            pd.to_numeric(
                calls.get("volume"),
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
                "No call volume"
            )

        return (
            float(pv / cv),
            True,
            f"{ticker} Optionen ({expiry})"
        )

    except Exception:
        return (
            np.nan,
            False,
            "Option chain unavailable"
        )


# ============================================================
# 8. FUTURES PUT/CALL PROXY
# ============================================================

@st.cache_data(ttl=1800)
def fetch_futures_put_call(proxy_ticker):

    """
    Futures selbst besitzen keine Put/Call-Ratio.

    Deshalb wird der konfigurierte Optionsproxy verwendet.

    Beispiele:
    ES -> SPY
    NQ -> QQQ
    Gold -> GLD
    WTI -> USO
    EUR/USD -> FXE
    """

    return fetch_option_put_call(
        proxy_ticker
    )


# ============================================================
# 9. FRED
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
# 10. SIDEBAR
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
# 11. MULTI-ASSET DATA
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
        "asset": cfg["ticker"],
        "vix": cfg["volatility_ticker"],
        "dxy": "DX=F",
        "move": "^MOVE",
        "hyg": "HYG",
        "lqd": "LQD"
    }

    try:

        data = yf.download(
            list(tickers.values()),
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
                for k, v in tickers.items()
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
        and not close.asset
        .dropna()
        .empty
    )

    if not status[
        "yFinance (Preis & Tech)"
    ]:

        return (
            pd.DataFrame(),
            status
        )

    price = close.asset.dropna()

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

    hasvol = False

    if vol is not None:

        if isinstance(
            vol,
            pd.Series
        ):

            asset_vol = (
                vol
                .reindex(price.index)
                .ffill()
                .bfill()
            )

            hasvol = True

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
                    for k, v in tickers.items()
                }
            )

            if "asset" in vol:

                asset_vol = (
                    vol.asset
                    .reindex(price.index)
                    .ffill()
                    .bfill()
                )

                hasvol = True

    if not hasvol:

        asset_vol = pd.Series(
            1000.,
            index=price.index
        )

    status[
        "Volumen / Orderflow Feed"
    ] = hasvol


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
        (price - ma50)
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
        (price - ma200)
        /
        ma200.replace(
            0,
            np.nan
        )
        * 100
    )


    # --------------------------------------------------------
    # RSI
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

    df[
        "rsi_momentum"
    ] = (
        100
        -
        100
        /
        (
            1
            +
            gain
            /
            loss.replace(
                0,
                np.nan
            )
        )
    ).clip(
        0,
        100
    )


    # --------------------------------------------------------
    # VOLATILITÄT / USD / MOVE
    # --------------------------------------------------------

    df["vix_score"] = (
        close.vix
        .reindex(df.index)
        .ffill()
        .bfill()
        if "vix" in close
        else 20.
    )

    df["usd_index"] = (
        close.dxy
        .reindex(df.index)
        .ffill()
        .bfill()
        if "dxy" in close
        else 100.
    )

    df["move_index"] = (
        close.move
        .reindex(df.index)
        .ffill()
        .bfill()
        if "move" in close
        else 100.
    )


    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------

    if (
        "lqd" in close
        and "hyg" in close
    ):

        df[
            "credit_spreads"
        ] = (
            close.lqd
            .reindex(df.index)
            .ffill()
            .bfill()
            /
            close.hyg
            .reindex(df.index)
            .ffill()
            .bfill()
            .replace(
                0,
                np.nan
            )
        )

    else:

        df[
            "credit_spreads"
        ] = 1.


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

    obv = pd.Series(
        np.where(
            delta > 0,
            asset_vol,
            np.where(
                delta < 0,
                -asset_vol,
                0.
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
        (obv - ema)
        /
        ema.abs().replace(
            0,
            np.nan
        )
        * 100
    )


    # --------------------------------------------------------
    # S&P 500 PE
    # --------------------------------------------------------

    if selected_asset == "S&P 500":

        df[
            "pe_valuation"
        ] = 24.5


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
                f.get_series("WALCL"),
                df.index
            )

            tga = safe_reindex_series(
                f.get_series("WTREGEN"),
                df.index
            )

            rrp = safe_reindex_series(
                f.get_series("RRPONTSYD"),
                df.index
            )

            df[
                "net_liquidity"
            ] = (
                (wal - tga - rrp * 1000)
                / 1000
                if (
                    wal is not None
                    and tga is not None
                    and rrp is not None
                )
                else np.nan
            )

            df[
                "fed_policy"
            ] = safe_reindex_series(
                f.get_series("FEDFUNDS"),
                df.index
            )

            df[
                "real_yields"
            ] = safe_reindex_series(
                f.get_series("DFII10"),
                df.index
            )

            if selected_asset == "WTI Crude Oil":

                df[
                    "inventories"
                ] = safe_reindex_series(
                    f.get_series("WCESTUS1"),
                    df.index
                )

            fred_ok = True

        except Exception:
            pass


    status[
        "FRED API (Makro & Fed)"
    ] = fred_ok


    if not fred_ok:

        for c in [
            "fed_policy",
            "real_yields",
            "net_liquidity"
        ]:

            df[c] = np.nan

        if selected_asset == "WTI Crude Oil":
            df["inventories"] = np.nan


    # ========================================================
    # CFTC COT
    # ========================================================

    cot, cot_ok = fetch_cot_data(
        cfg["cot_code"]
    )

    status[
        f"CFTC COT ({cfg['cot_code']})"
    ] = cot_ok

    df[
        "cot_commercials"
    ] = (
        safe_reindex_series(
            cot,
            df.index
        )
        if cot is not None
        else np.nan
    )


    # ========================================================
    # CNN FEAR & GREED
    # ========================================================

    fg, fg_ok = fetch_fear_and_greed()

    status[
        "CNN Fear & Greed"
    ] = fg_ok

    df[
        "fear_greed"
    ] = (
        safe_reindex_series(
            fg,
            df.index
        )
        if isinstance(
            fg,
            pd.Series
        )
        else float(fg)
    )


    # ========================================================
    # PUT/CALL DATEN
    # ========================================================
    #
    # WICHTIG:
    # Diese Daten werden NICHT in die Modellgewichtung
    # aufgenommen.
    #
    # options_put_call = tatsächliche Options-P/C-Ratio
    # des konfigurierten Optionsproxys.
    #
    # futures_put_call = Optionsproxy für den jeweiligen
    # Futures-Markt.
    #
    # Dadurch bleibt die bestehende Modellhistorie unverändert.
    # ========================================================

    opt_pc, opt_ok, opt_source = (
        fetch_option_put_call(
            cfg["options_pc_ticker"]
        )
    )

    fut_pc, fut_ok, fut_source = (
        fetch_futures_put_call(
            cfg["options_pc_ticker"]
        )
    )

    df[
        "options_put_call"
    ] = (
        opt_pc
        if np.isfinite(opt_pc)
        else np.nan
    )

    df[
        "futures_put_call"
    ] = (
        fut_pc
        if np.isfinite(fut_pc)
        else np.nan
    )

    status[
        f"Options Put/Call ({cfg['options_pc_ticker']})"
    ] = opt_ok

    status[
        f"Futures Put/Call Proxy ({cfg['futures_pc_ticker']})"
    ] = fut_ok


    # ========================================================
    # NORMALISIERUNG
    # ========================================================

    norm = pd.DataFrame(
        index=df.index
    )

    inverts = cfg[
        "invert_inverts"
    ]

    for col in df:

        norm[col] = normalize_to_percentile(
            df[col],
            LOOKBACK_CONFIG.get(
                col,
                252
            ),
            col in inverts
        )


    # ========================================================
    # SÄULEN
    # ========================================================

    dash = pd.DataFrame(
        index=df.index
    )

    dash[
        "Raw_Volatility"
    ] = df.vix_score


    active = {
        k: v.copy()
        for k, v in SUB_WEIGHTS_BASE.items()
    }

    for cat, w in cfg[
        "Sub_Gewichte"
    ].items():

        active[cat] = w


    for pillar, inds in active.items():

        cols = [
            c for c in inds
            if c in norm.columns
        ]

        ws = np.array(
            [
                inds[c]
                for c in cols
            ],
            float
        )

        if (
            cols
            and ws.sum() > 0
        ):

            dash[
                f"Saeule_{pillar}"
            ] = norm[cols].dot(
                ws / ws.sum()
            )

        else:

            dash[
                f"Saeule_{pillar}"
            ] = 50.


    # ========================================================
    # FINAL REGIME SCORE
    # ========================================================

    sc = []
    ws = []

    for pillar, w in cfg[
        "Saeulen_Gewichte"
    ].items():

        c = (
            f"Saeule_{pillar}"
        )

        if c in dash:

            sc.append(c)
            ws.append(w)

    w = np.array(
        ws,
        float
    )

    w = w / w.sum()

    dash[
        "Final_Regime_Score"
    ] = (
        dash[sc]
        .dot(w)
        .clip(0, 100)
        .round(1)
    )

    dash[
        "MCI"
    ] = [
        calculate_mci(
            dash[sc].iloc[i].values,
            w
        )
        for i in range(len(dash))
    ]

    dash[
        "Asset_Price"
    ] = (
        price
        .reindex(dash.index)
        .ffill()
        .bfill()
    )


    # ========================================================
    # AKTUELLE PCR-WERTE
    # ========================================================

    dash[
        "Options_Put_Call"
    ] = opt_pc

    dash[
        "Futures_Put_Call_Proxy"
    ] = fut_pc


    return (
        dash.dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        status
    )


# ============================================================
# 12. DATEN LADEN
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

heute = df_dash.iloc[-1].copy()

heute[
    "Delta_1D"
] = (
    df_dash.Final_Regime_Score.iloc[-1]
    -
    df_dash.Final_Regime_Score.iloc[-2]
    if len(df_dash) >= 2
    else 0.
)

heute[
    "Delta_1W"
] = (
    df_dash.MCI.iloc[-1]
    -
    df_dash.MCI.iloc[-6]
    if len(df_dash) >= 6
    else 0.
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
# 15. FINAL SCORE + MCI
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Final Regime Score",
        f"{heute.Final_Regime_Score} / 100",
        f"{heute.Delta_1D:+.1f} (Heute)"
    )

with c2:

    st.metric(
        "Model Consistency Index",
        f"{heute.MCI}%",
        f"{heute.Delta_1W:+.1f} (vs. Vorwoche)",
        delta_color="off"
    )


st.caption(
    "Der Model Consistency Index misst die Übereinstimmung "
    "der sechs Modell-Säulen. Er ist keine statistische "
    "Wahrscheinlichkeit."
)


st.info(
    f"**Aktuelles Marktregime ({selected_asset}):** "
    f"{get_regime_label(heute.Final_Regime_Score)}"
)


# ============================================================
# 16. PUT/CALL RATIO – ZUSÄTZLICHER FILTER
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Put/Call Ratio – zusätzlicher Positionierungsfilter"
)

opt_pc = float(
    heute.get(
        "Options_Put_Call",
        np.nan
    )
)

fut_pc = float(
    heute.get(
        "Futures_Put_Call_Proxy",
        np.nan
    )
)


# ------------------------------------------------------------
# PCR INTERPRETATION
# ------------------------------------------------------------

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


pc1, pc2, pc3, pc4 = st.columns(4)


with pc1:

    st.metric(
        "Options Put/Call",
        (
            f"{opt_pc:.2f}"
            if np.isfinite(opt_pc)
            else "n/a"
        )
    )


with pc2:

    st.metric(
        "Futures-P/C-Proxy",
        (
            f"{fut_pc:.2f}"
            if np.isfinite(fut_pc)
            else "n/a"
        )
    )


with pc3:

    st.metric(
        "P/C Interpretation",
        pc_interpretation
    )


with pc4:

    st.metric(
        "Kontra-Signal",
        pc_bias
    )


st.caption(
    "Der PCR wird ausschließlich als zusätzlicher "
    "Positionierungs-/Sentimentfilter angezeigt. "
    "Er verändert den Final Regime Score und die "
    "Säulengewichte nicht."
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

Der PCR sollte niemals isoliert als Einstiegssignal verwendet werden.
"""
    )


st.caption(
    "⚠️ Wichtig: Futures selbst besitzen keine Put/Call-Ratio. "
    "Der Futures-P/C-Wert ist daher ausdrücklich ein "
    "Options-Proxy zum jeweiligen Futures-Markt "
    "(ES → SPY, NQ → QQQ, Gold → GLD, WTI → USO, EUR/USD → FXE). "
    "Die CFTC-COT-Daten sind dagegen echte Futures bzw. "
    "Futures-and-Options-Combined-Positionsdaten und keine "
    "Put/Call-Ratio."
)


# ============================================================
# 17. VOLATILITÄTS-ALARM
# ============================================================

current_vola = float(
    heute.get(
        "Raw_Volatility",
        20
    )
)

limit = VOLA_THRESHOLDS.get(
    selected_asset,
    30
)

vt = ASSET_CONFIGS[
    selected_asset
]["volatility_ticker"]


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
)


if score >= 60:

    bias = (
        "🟢 BULLISCH (Long Bias)"
    )

    rule = (
        f"Bevorzugt Long-Setups "
        f"bei {selected_asset} suchen."
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
        f"Bevorzugt Short-Setups "
        f"bei {selected_asset} suchen."
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


if current_vola >= limit:

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
            if abs(score - 50) > 15
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
    "Unabhängiger Kontraindikator auf Basis "
    "des Suchverhaltens von Privatanlegern."
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
**🔍 Getrackte Parameter:**

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
                "Yahoo: US Dollar Index",
                "https://finance.yahoo.com/quote/DX-Y.NYB"
            )
        ]
    },

    "Positionierung": {
        "quelle": (
            "CFTC COT, CNN Fear & Greed "
            "& Put/Call-Daten"
        ),
        "funktion": (
            "Institutionelle Positionierung, "
            "Sentiment und Optionspositionierung."
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
            )
        ]
    },

    "Marktinterna": {
        "quelle": "Yahoo Finance",
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
        "quelle": "Yahoo Finance",
        "funktion": (
            "200-Tage-Trend, 50-Tage-Trend "
            "und RSI-Momentum."
        ),
        "links": [
            (
                "Yahoo: Chart & Technicals",
                f"https://finance.yahoo.com/quote/"
                f"{ASSET_CONFIGS[selected_asset]['ticker']}"
            )
        ]
    },

    "Fundamentale_Faktoren": {
        "quelle": "FRED, Multpl & WSJ",
        "funktion": (
            "Bewertung bzw. "
            "Rohstoff-Lagerbestände."
        ),
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
        "quelle": "Yahoo Finance",
        "funktion": (
            "Kreditmarkt-Proxy und "
            "Anleihenvolatilität."
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
    if c.startswith("Saeule_")
]


for i, s in enumerate(saeulen):

    val = float(
        heute.get(
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
        ) * 100
    )

    with cols[i % 3]:

        st.metric(
            f"{label} {emoji}",
            f"{val:.1f}"
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
        50
    )
)

early = float(
    heute.get(
        "Saeule_Fruehwarnindikatoren",
        50
    )
)

macro = float(
    heute.get(
        "Saeule_Makroökonomie",
        50
    )
)


trend_ok = trend > 55
bond_ok = early > 35
macro_ok = macro > 50


now = pd.Timestamp.now(
    tz="Europe/Berlin"
)

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
        f"(Score: {macro:.0f})",
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


count = sum([
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8
])


st.progress(
    count / 8
)

st.caption(
    f"✅ **{count} von 8 Kriterien erfüllt**"
)


if count == 8 and score > 55:

    st.success(
        "🟢 **EXECUTION FREIGABE (GO):** "
        "Alle Filter erfüllt und bullischer Long-Bias."
    )

elif count == 8 and score < 45:

    st.error(
        "🔴 **EXECUTION FREIGABE (SHORT):** "
        "Alle Filter erfüllt und ausreichend bärisch."
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


plot = df_dash.tail(252)


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


fig.update_yaxes(
    title_text="Regime Score",
    range=[0, 100],
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
# 24. AKTUELLE MODELLGEWICHTUNGEN
# ============================================================

st.markdown("---")

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
                "Säule": list(
                    weights
                ),
                "Gewichtung": [
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
# 25. SYSTEM & API STATUS
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

        sc[i % 2].markdown(
            f"**{feed}:** "
            f"{'✅ Verbunden' if status_value else '⚠️ Fallback aktiv / Offline'}"
        )

    st.caption(
        "Fallback-Werte sind nicht als Live-Daten "
        "zu interpretieren."
    )


# ============================================================
# 26. ABSCHLUSS-HINWEIS
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Modellhinweis: Der Final Regime Score ist ein "
    "quantitatives Entscheidungs- und Regimefilter-Modell "
    "und keine Anlageberatung. Der LQD/HYG-Wert ist ein "
    "Kreditmarkt-Proxy und kein tatsächlicher Credit Spread. "
    "Die Put/Call-Komponente ist separat ausgewiesen; "
    "Futures selbst besitzen keine Put/Call-Ratio."
)
