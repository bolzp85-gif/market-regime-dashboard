import io
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from scipy.stats import norm
from fredapi import Fred
from plotly.subplots import make_subplots
from pytrends.request import TrendReq


# ============================================================
# 0. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Quant Regime Dashboard V2",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 1. ASSET CONFIG
# ============================================================

ASSET_CONFIGS = {

    "S&P 500": {
        "ticker": "^GSPC",
        "proxy_ticker": "SPY",
        "volatility_ticker": "^VIX",
        "cot_code": "E-MINI S&P 500",

        "breadth_index": "S&P 500",

        "vol_percentile_limit": 90,

        "regime_weights": {
            "Makroökonomie": 0.25,
            "Positionierung": 0.20,
            "Marktinterna": 0.20,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.10,
        },

        "sub_weights": {
            "Positionierung": {
                "cot_percentile": 0.50,
                "fear_greed": 0.50,
            },
            "Marktinterna": {
                "breadth": 0.50,
                "vol_percentile": 0.50,
            },
            "Fundamentale_Faktoren": {
                "pe_valuation": 1.00,
            },
        },
    },

    "Nasdaq 100": {
        "ticker": "NQ=F",
        "proxy_ticker": "QQQ",
        "volatility_ticker": "^VXN",
        "cot_code": "NASDAQ-100",

        "breadth_index": "NASDAQ-100",

        "vol_percentile_limit": 90,

        "regime_weights": {
            "Makroökonomie": 0.25,
            "Positionierung": 0.20,
            "Marktinterna": 0.20,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.10,
        },

        "sub_weights": {
            "Positionierung": {
                "cot_percentile": 0.50,
                "fear_greed": 0.50,
            },
            "Marktinterna": {
                "breadth": 0.50,
                "vol_percentile": 0.50,
            },
            "Fundamentale_Faktoren": {},
        },
    },

    "Gold (XAU/USD)": {
        "ticker": "GC=F",
        "proxy_ticker": "GLD",
        "volatility_ticker": "^GVZ",
        "cot_code": "GOLD",

        "breadth_index": None,

        "vol_percentile_limit": 90,

        "regime_weights": {
            "Makroökonomie": 0.35,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.00,
            "Fruehwarnindikatoren": 0.10,
        },

        "sub_weights": {
            "Positionierung": {
                "cot_percentile": 0.80,
                "fear_greed": 0.20,
            },
            "Marktinterna": {
                "obv_momentum": 0.50,
                "vol_percentile": 0.50,
            },
            "Fundamentale_Faktoren": {},
        },
    },

    "WTI Crude Oil": {
        "ticker": "CL=F",
        "proxy_ticker": "USO",
        "volatility_ticker": "^OVX",
        "cot_code": "CRUDE OIL",

        "breadth_index": None,

        "vol_percentile_limit": 90,

        "regime_weights": {
            "Makroökonomie": 0.30,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.20,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.00,
        },

        "sub_weights": {
            "Positionierung": {
                "cot_percentile": 0.80,
                "fear_greed": 0.20,
            },
            "Marktinterna": {
                "obv_momentum": 0.50,
                "vol_percentile": 0.50,
            },
            "Fundamentale_Faktoren": {
                "inventories": 1.00,
            },
        },
    },
}


MACRO_WEIGHTS = {
    "fed_policy": 0.25,
    "real_yields": 0.25,
    "usd_index": 0.25,
    "net_liquidity": 0.25,
}

TECH_WEIGHTS = {
    "distance_200ma": 0.40,
    "distance_50ma": 0.30,
    "rsi_momentum": 0.30,
}

EARLY_WARNING_WEIGHTS = {
    "credit_stress": 0.60,
    "move_index": 0.40,
}


LOOKBACKS = {
    "fed_policy": 1260,
    "real_yields": 756,
    "usd_index": 504,
    "net_liquidity": 756,
    "credit_stress": 756,
    "move_index": 756,
    "pe_valuation": 2520,
    "inventories": 756,
    "cot_percentile": 756,
    "breadth": 756,
    "obv_momentum": 252,
    "vol_percentile": 756,
}


# ============================================================
# 2. FRED
# ============================================================

FRED_API_KEY = ""

try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    pass


# ============================================================
# 3. HELPERS
# ============================================================

def clean_datetime_index(index):
    idx = pd.to_datetime(index)

    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_convert(None)

        return idx.floor("D")

    return idx


def safe_series(series):
    if series is None:
        return None

    if isinstance(series, pd.DataFrame):
        if series.empty:
            return None
        series = series.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.dropna()

    if series.empty:
        return None

    series.index = clean_datetime_index(series.index)
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def align_to_index(series, target_index):
    series = safe_series(series)

    if series is None:
        return None

    target = clean_datetime_index(target_index)

    result = series.reindex(target, method="ffill")

    result.index = target_index

    return result


def percentile_score(
    series,
    lookback=252,
    invert=False,
    min_periods=60,
):
    """
    Rolling percentile auf Basis der historischen Verteilung.

    0 = extrem niedrig
    50 = neutral
    100 = extrem hoch
    """

    if series is None:
        return None

    s = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    def rolling_percentile(window):

        if len(window) < min_periods:
            return np.nan

        current = window.iloc[-1]
        historical = window.iloc[:-1].dropna()

        if len(historical) < min_periods - 1:
            return np.nan

        percentile = (historical <= current).mean() * 100

        return percentile

    result = s.rolling(
        lookback,
        min_periods=min_periods,
    ).apply(
        rolling_percentile,
        raw=False,
    )

    if invert:
        result = 100 - result

    return result.clip(0, 100)


def normalize_weighted_scores(df, weights):
    """
    Gewichtet nur tatsächlich verfügbare Komponenten.

    Fehlende Daten werden NICHT durch 50 ersetzt.
    """

    available = [
        c for c in weights
        if c in df.columns and df[c].notna().any()
    ]

    if not available:
        return pd.Series(np.nan, index=df.index), 0.0

    weight_values = np.array(
        [weights[c] for c in available],
        dtype=float,
    )

    weight_values /= weight_values.sum()

    values = df[available].copy()

    weighted = values.mul(
        weight_values,
        axis=1,
    ).sum(axis=1, min_count=1)

    coverage = (
        df[available]
        .notna()
        .mul(weight_values, axis=1)
        .sum(axis=1)
    )

    return weighted, coverage


def calculate_mci(row, weights, coverage):
    """
    MCI basiert auf:
    1. Konsistenz der vorhandenen Säulen
    2. Datenabdeckung

    Kein künstlicher Fallback.
    """

    available = [
        c for c in weights
        if c in row.index and pd.notna(row[c])
    ]

    if len(available) < 2:
        return np.nan

    w = np.array(
        [weights[c] for c in available],
        dtype=float,
    )

    w /= w.sum()

    values = np.array(
        [float(row[c]) for c in available]
    )

    mean = np.average(values, weights=w)

    variance = np.average(
        (values - mean) ** 2,
        weights=w,
    )

    std = np.sqrt(variance)

    # 35 Punkte Streuung entsprechen hier 0 % Konsistenz.
    consistency = np.clip(
        100 * (1 - std / 35),
        0,
        100,
    )

    data_quality = np.clip(
        coverage * 100,
        0,
        100,
    )

    return round(
        0.75 * consistency +
        0.25 * data_quality,
        1,
    )


def regime_label(score):

    if pd.isna(score):
        return "⚪ Keine ausreichenden Daten"

    if score >= 80:
        return "🟢 Starkes Risk-On"
    elif score >= 65:
        return "🟢 Risk-On"
    elif score >= 55:
        return "🟡 Leicht bullisch"
    elif score >= 45:
        return "🟡 Neutral"
    elif score >= 35:
        return "🟠 Risk-Off"
    else:
        return "🔴 Starkes Risk-Off"


def wilder_rsi(price, period=14):

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def flatten_yfinance_columns(df):

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # bevorzugt Close / Volume aus MultiIndex
    if "Close" in df.columns.get_level_values(0):
        return df

    if "Close" in df.columns.get_level_values(1):
        df = df.copy()
        df.columns = df.columns.swaplevel(0, 1)
        return df.sort_index(axis=1)

    return df


# ============================================================
# 4. CFTC COT
# ============================================================

@st.cache_data(ttl=86400)
def fetch_cot_data(search_string):

    current_year = pd.Timestamp.utcnow().year

    frames = []

    for year in [
        current_year - 3,
        current_year - 2,
        current_year - 1,
        current_year,
    ]:

        url = (
            "https://www.cftc.gov/files/dea/history/"
            f"fut_com_txt_{year}.zip"
        )

        try:

            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )

            if response.status_code != 200:
                continue

            with zipfile.ZipFile(
                io.BytesIO(response.content)
            ) as archive:

                name = archive.namelist()[0]

                df = pd.read_csv(
                    archive.open(name),
                    low_memory=False,
                )

            first_col = df.columns[0]

            mask = (
                df[first_col]
                .astype(str)
                .str.contains(
                    search_string,
                    case=False,
                    na=False,
                )
            )

            filtered = df.loc[mask].copy()

            if not filtered.empty:
                frames.append(filtered)

        except Exception:
            continue

    if not frames:
        return None, False

    try:

        df = pd.concat(
            frames,
            ignore_index=True,
        )

        date_col = next(
            c for c in df.columns
            if "As_of_Date" in str(c)
        )

        long_col = next(
            c for c in df.columns
            if "Comm_Positions_Long_All" in str(c)
        )

        short_col = next(
            c for c in df.columns
            if "Comm_Positions_Short_All" in str(c)
        )

        dates = pd.to_datetime(
            df[date_col].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

        net = (
            pd.to_numeric(
                df[long_col],
                errors="coerce",
            )
            -
            pd.to_numeric(
                df[short_col],
                errors="coerce",
            )
        )

        result = pd.DataFrame({
            "date": dates,
            "net_commercials": net,
        })

        result = result.dropna()

        result = result.drop_duplicates(
            "date",
            keep="last",
        )

        result = result.set_index("date")

        return (
            result["net_commercials"].sort_index(),
            True,
        )

    except Exception:
        return None, False


def cot_percentile(series, lookback=756):

    return percentile_score(
        series,
        lookback=lookback,
        invert=False,
        min_periods=60,
    )


# ============================================================
# 5. CNN FEAR & GREED
# ============================================================

@st.cache_data(ttl=14400)
def fetch_fear_greed():

    url = (
        "https://production.dataviz.cnn.io/"
        "index/fearandgreed/graphdata"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://edition.cnn.com/",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        hist = (
            data
            .get("fear_and_greed_historical", {})
            .get("data", [])
        )

        if not hist:
            return None, False

        df = pd.DataFrame(hist)

        df["Date"] = pd.to_datetime(
            df["x"],
            unit="ms",
        )

        df["Date"] = clean_datetime_index(
            df["Date"]
        )

        result = (
            df.drop_duplicates("Date")
            .set_index("Date")["y"]
            .sort_index()
        )

        return result, True

    except Exception:
        return None, False


# ============================================================
# 6. GOOGLE TRENDS
# ============================================================

TREND_KEYWORDS = {

    "S&P 500": {
        "geo": "US",
        "hl": "en-US",
        "bull": [
            "buy stocks",
            "buy the dip",
        ],
        "bear": [
            "stock market crash",
            "recession",
        ],
    },

    "Nasdaq 100": {
        "geo": "US",
        "hl": "en-US",
        "bull": [
            "tech stocks",
            "buy the dip",
        ],
        "bear": [
            "market crash",
            "tech bubble",
        ],
    },

    "Gold (XAU/USD)": {
        "geo": "DE",
        "hl": "de-DE",
        "bull": [
            "Gold kaufen",
            "Goldmünzen",
        ],
        "bear": [
            "Gold verkaufen",
            "Altgold",
        ],
    },

    "WTI Crude Oil": {
        "geo": "DE",
        "hl": "de-DE",
        "bull": [
            "Heizöl kaufen",
            "Spritpreise",
        ],
        "bear": [
            "Ölpreis crash",
            "Öl verkaufen",
        ],
    },
}


@st.cache_data(ttl=21600)
def fetch_google_trends(asset):

    cfg = TREND_KEYWORDS[asset]

    try:

        pytrends = TrendReq(
            hl=cfg["hl"],
            tz=360,
        )

        keywords = (
            cfg["bull"] +
            cfg["bear"]
        )

        pytrends.build_payload(
            keywords,
            timeframe="today 3-m",
            geo=cfg["geo"],
        )

        df = pytrends.interest_over_time()

        if df.empty:
            return None, None, False

        if "isPartial" in df.columns:
            df = df.drop(
                columns=["isPartial"]
            )

        def zscore(series):

            mean = (
                series
                .rolling(
                    21,
                    min_periods=10,
                )
                .mean()
            )

            std = (
                series
                .rolling(
                    21,
                    min_periods=10,
                )
                .std()
                .replace(0, np.nan)
            )

            return (series - mean) / std

        bull_cols = [
            x for x in cfg["bull"]
            if x in df.columns
        ]

        bear_cols = [
            x for x in cfg["bear"]
            if x in df.columns
        ]

        if not bull_cols or not bear_cols:
            return None, None, False

        bull = pd.concat(
            [zscore(df[c]) for c in bull_cols],
            axis=1,
        ).mean(axis=1)

        bear = pd.concat(
            [zscore(df[c]) for c in bear_cols],
            axis=1,
        ).mean(axis=1)

        spread = bull - bear

        latest = spread.dropna().iloc[-1]

        # Kontraindikator:
        # + Spread = Euphorie = niedriger Score
        # - Spread = Angst = höherer Score
        score = np.clip(
            50 - latest * 15,
            0,
            100,
        )

        return (
            round(float(score), 1),
            round(float(latest), 2),
            True,
        )

    except Exception:
        return None, None, False


# ============================================================
# 7. HISTORISCHES S&P 500 P/E
# ============================================================

@st.cache_data(ttl=86400)
def fetch_sp500_pe():

    url = (
        "https://www.multpl.com/"
        "s-p-500-pe-ratio/table/by-month"
    )

    try:

        tables = pd.read_html(url)

        if not tables:
            return None, False

        table = tables[0].copy()

        table.columns = [
            str(c).strip()
            for c in table.columns
        ]

        date_col = table.columns[0]
        value_col = table.columns[1]

        table["Date"] = pd.to_datetime(
            table[date_col],
            errors="coerce",
        )

        table["PE"] = (
            table[value_col]
            .astype(str)
            .str.replace(
                r"[^0-9.\-]",
                "",
                regex=True,
            )
        )

        table["PE"] = pd.to_numeric(
            table["PE"],
            errors="coerce",
        )

        result = (
            table
            .dropna(subset=["Date", "PE"])
            .set_index("Date")["PE"]
            .sort_index()
        )

        return result, True

    except Exception:
        return None, False


# ============================================================
# 8. INDEX COMPONENTS / MARKET BREADTH
# ============================================================

@st.cache_data(ttl=86400)
def fetch_index_components(index_name):

    url = {
        "S&P 500":
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",

        "NASDAQ-100":
            "https://en.wikipedia.org/wiki/Nasdaq-100",
    }.get(index_name)

    if not url:
        return None

    try:

        tables = pd.read_html(url)

        if index_name == "S&P 500":
            table = tables[0]
            tickers = table["Symbol"].tolist()

        else:
            table = next(
                t for t in tables
                if "Ticker" in t.columns
            )

            tickers = table["Ticker"].tolist()

        cleaned = []

        for ticker in tickers:

            ticker = str(ticker).strip()

            # Yahoo verwendet bei manchen Aktien "-"
            # statt ".".
            ticker = ticker.replace(".", "-")

            if re.match(
                r"^[A-Z0-9\-]+$",
                ticker,
            ):
                cleaned.append(ticker)

        return sorted(set(cleaned))

    except Exception:
        return None


@st.cache_data(ttl=86400)
def fetch_market_breadth(index_name):

    tickers = fetch_index_components(
        index_name
    )

    if not tickers:
        return None, False

    # Begrenzung auf tatsächlich vorhandene Komponenten.
    # Yahoo kann bei einzelnen Symbolen temporär Fehler liefern.
    try:

        data = yf.download(
            tickers,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if data.empty:
            return None, False

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" in data.columns.levels[0]:
                closes = data["Close"]

            elif "Close" in data.columns.levels[1]:
                closes = data.xs(
                    "Close",
                    axis=1,
                    level=1,
                )

            else:
                return None, False

        else:
            closes = data[["Close"]]

        closes = closes.apply(
            pd.to_numeric,
            errors="coerce",
        )

        daily_returns = closes.pct_change()

        advancing = (
            daily_returns > 0
        ).sum(axis=1)

        declining = (
            daily_returns < 0
        ).sum(axis=1)

        total = advancing + declining

        breadth = (
            (advancing - declining)
            / total.replace(0, np.nan)
        ) * 100

        # geglättete Breadth-Line
        breadth_ma = breadth.rolling(
            20,
            min_periods=10,
        ).mean()

        return breadth_ma, True

    except Exception:
        return None, False


# ============================================================
# 9. FRED
# ============================================================

@st.cache_data(ttl=21600)
def fetch_fred_data():

    if not FRED_API_KEY:
        return {}, False

    try:

        fred = Fred(
            api_key=FRED_API_KEY
        )

        data = {}

        series_map = {
            "walcl": "WALCL",
            "tga": "WTREGEN",
            "rrp": "RRPONTSYD",
            "fed_policy": "FEDFUNDS",
            "real_yields": "DFII10",
            "inventories": "WCESTUS1",
        }

        for name, series_id in series_map.items():

            try:

                series = fred.get_series(
                    series_id
                )

                series = safe_series(series)

                if series is not None:
                    data[name] = series

            except Exception:
                continue

        # Net Liquidity:
        #
        # WALCL  = Millionen USD
        # TGA    = Millionen USD
        # RRP    = Milliarden USD
        #
        # Deshalb:
        #
        # WALCL / 1000
        # TGA / 1000
        # RRP direkt
        #
        if all(
            x in data
            for x in [
                "walcl",
                "tga",
                "rrp",
            ]
        ):

            walcl = data["walcl"] / 1000.0
            tga = data["tga"] / 1000.0
            rrp = data["rrp"]

            net_liquidity = (
                walcl
                - tga
                - rrp
            )

            data["net_liquidity"] = (
                net_liquidity
            )

        return data, True

    except Exception:
        return {}, False


# ============================================================
# 10. MAIN DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600)
def build_dashboard(asset):

    cfg = ASSET_CONFIGS[asset]

    feed_status = {}

    tickers = {
        "asset": cfg["ticker"],
        "vol": cfg["volatility_ticker"],
        "dxy": "DX=F",
        "move": "^MOVE",
        "hyg": "HYG",
        "lqd": "LQD",
    }

    try:

        raw = yf.download(
            list(tickers.values()),
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    except Exception:
        return pd.DataFrame(), feed_status

    if raw.empty:
        return pd.DataFrame(), feed_status

    # -------------------------
    # Close
    # -------------------------

    if isinstance(raw.columns, pd.MultiIndex):

        if "Close" in raw.columns.levels[0]:

            close = raw["Close"]

            volume = (
                raw["Volume"]
                if "Volume" in raw.columns.levels[0]
                else None
            )

        elif "Close" in raw.columns.levels[1]:

            close = raw.xs(
                "Close",
                axis=1,
                level=1,
            )

            volume = (
                raw.xs(
                    "Volume",
                    axis=1,
                    level=1,
                )
                if "Volume" in raw.columns.levels[1]
                else None
            )

        else:
            return pd.DataFrame(), feed_status

    else:

        close = raw.copy()

        volume = None

    close = close.apply(
        pd.to_numeric,
        errors="coerce",
    )

    rename_map = {
        ticker: key
        for key, ticker in tickers.items()
    }

    close = close.rename(
        columns=rename_map
    )

    if "asset" not in close.columns:
        return pd.DataFrame(), feed_status

    price = close["asset"].dropna()

    if price.empty:
        return pd.DataFrame(), feed_status

    feed_status["Yahoo Finance"] = True

    # ========================================================
    # RAW DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        index=price.index
    )

    # ========================================================
    # TECHNICAL
    # ========================================================

    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()

    df["distance_50ma"] = (
        (price - ma50)
        / ma50
    ) * 100

    df["distance_200ma"] = (
        (price - ma200)
        / ma200
    ) * 100

    df["rsi_momentum"] = wilder_rsi(
        price,
        14,
    )

    # ========================================================
    # VOLUME / OBV
    # ========================================================

    if (
        volume is not None
        and isinstance(volume, pd.DataFrame)
        and cfg["ticker"] in volume.columns
    ):

        asset_volume = pd.to_numeric(
            volume[cfg["ticker"]],
            errors="coerce",
        )

        volume_available = True

    else:

        asset_volume = None
        volume_available = False

    feed_status["Volumen"] = (
        volume_available
    )

    if asset_volume is not None:

        delta = price.diff()

        signed_volume = np.where(
            delta > 0,
            asset_volume,
            np.where(
                delta < 0,
                -asset_volume,
                0,
            ),
        )

        obv = pd.Series(
            signed_volume,
            index=price.index,
        ).cumsum()

        obv_ema = obv.ewm(
            span=50,
            adjust=False,
        ).mean()

        df["obv_momentum"] = (
            (obv - obv_ema)
            / obv_ema.abs().replace(
                0,
                np.nan,
            )
        ) * 100

    # ========================================================
    # VOLATILITY
    # ========================================================

    if "vol" in close.columns:

        df["vol_raw"] = close["vol"]

        df["vol_percentile"] = percentile_score(
            df["vol_raw"],
            lookback=LOOKBACKS[
                "vol_percentile"
            ],
            invert=True,
        )

    # ========================================================
    # DXY
    # ========================================================

    if "dxy" in close.columns:

        df["usd_index"] = close["dxy"]

    # ========================================================
    # MOVE
    # ========================================================

    if "move" in close.columns:

        df["move_index"] = close["move"]

    # ========================================================
    # CREDIT STRESS PROXY
    # ========================================================

    if (
        "lqd" in close.columns
        and "hyg" in close.columns
    ):

        # steigendes LQD/HYG =
        # relative Stärke Investment Grade
        # gegenüber High Yield
        #
        # -> Stress
        df["credit_stress"] = (
            close["lqd"]
            / close["hyg"]
        )

    # ========================================================
    # FRED
    # ========================================================

    fred_data, fred_live = fetch_fred_data()

    feed_status["FRED"] = fred_live

    for key in [
        "fed_policy",
        "real_yields",
        "net_liquidity",
        "inventories",
    ]:

        if key in fred_data:

            df[key] = align_to_index(
                fred_data[key],
                df.index,
            )

    # ========================================================
    # COT
    # ========================================================

    cot, cot_live = fetch_cot_data(
        cfg["cot_code"]
    )

    feed_status["CFTC COT"] = cot_live

    if cot is not None:

        cot_aligned = align_to_index(
            cot,
            df.index,
        )

        if cot_aligned is not None:

            df["cot_percentile"] = cot_percentile(
                cot_aligned,
                LOOKBACKS[
                    "cot_percentile"
                ],
            )

    # ========================================================
    # FEAR & GREED
    # ========================================================

    fg, fg_live = fetch_fear_greed()

    feed_status["CNN Fear & Greed"] = fg_live

    if fg is not None:

        fg_aligned = align_to_index(
            fg,
            df.index,
        )

        if fg_aligned is not None:

            # Fear & Greed:
            # hohe Gier = schlechter für Kontraindikator
            df["fear_greed"] = (
                100 - fg_aligned
            )

    # ========================================================
    # MARKET BREADTH
    # ========================================================

    if cfg["breadth_index"]:

        breadth, breadth_live = (
            fetch_market_breadth(
                cfg["breadth_index"]
            )
        )

        feed_status["Market Breadth"] = (
            breadth_live
        )

        if breadth is not None:

            breadth_aligned = align_to_index(
                breadth,
                df.index,
            )

            if breadth_aligned is not None:

                # -100 ... +100
                # -> 0 ... 100
                df["breadth"] = (
                    breadth_aligned + 100
                ) / 2

    # ========================================================
    # S&P 500 P/E
    # ========================================================

    if asset == "S&P 500":

        pe, pe_live = fetch_sp500_pe()

        feed_status["S&P 500 P/E"] = (
            pe_live
        )

        if pe is not None:

            pe_aligned = align_to_index(
                pe,
                df.index,
            )

            if pe_aligned is not None:

                df["pe_valuation"] = (
                    pe_aligned
                )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    scores = pd.DataFrame(
        index=df.index
    )

    # -------------------------
    # Macro
    # -------------------------

    macro_raw = pd.DataFrame(
        index=df.index
    )

    if "fed_policy" in df:
        macro_raw["fed_policy"] = df[
            "fed_policy"
        ]

    if "real_yields" in df:
        macro_raw["real_yields"] = df[
            "real_yields"
        ]

    if "usd_index" in df:
        macro_raw["usd_index"] = df[
            "usd_index"
        ]

    if "net_liquidity" in df:
        macro_raw["net_liquidity"] = df[
            "net_liquidity"
        ]

    macro_scores = pd.DataFrame(
        index=df.index
    )

    # Niedrige Zinsen / Real Yields / DXY
    # sind für klassische Risk Assets
    # grundsätzlich günstiger.
    #
    # Bei Gold wird ebenfalls invertiert,
    # da höhere Real Yields/DXY tendenziell
    # Gegenwind darstellen.

    if "fed_policy" in macro_raw:
        macro_scores[
            "fed_policy"
        ] = percentile_score(
            macro_raw["fed_policy"],
            LOOKBACKS["fed_policy"],
            invert=True,
        )

    if "real_yields" in macro_raw:
        macro_scores[
            "real_yields"
        ] = percentile_score(
            macro_raw["real_yields"],
            LOOKBACKS["real_yields"],
            invert=True,
        )

    if "usd_index" in macro_raw:
        macro_scores[
            "usd_index"
        ] = percentile_score(
            macro_raw["usd_index"],
            LOOKBACKS["usd_index"],
            invert=True,
        )

    if "net_liquidity" in macro_raw:
        macro_scores[
            "net_liquidity"
        ] = percentile_score(
            macro_raw["net_liquidity"],
            LOOKBACKS["net_liquidity"],
            invert=False,
        )

    # Für Gold ist der Fed-/Dollar-/Yield-Effekt
    # ähnlich, aber nicht identisch. Wir lassen
    # deshalb die gleiche Richtung zunächst bestehen.

    scores[
        "Makroökonomie"
    ], macro_coverage = (
        normalize_weighted_scores(
            macro_scores,
            MACRO_WEIGHTS,
        )
    )

    # -------------------------
    # Positionierung
    # -------------------------

    positioning_scores = pd.DataFrame(
        index=df.index
    )

    if "cot_percentile" in df:
        positioning_scores[
            "cot_percentile"
        ] = df[
            "cot_percentile"
        ]

    if "fear_greed" in df:
        positioning_scores[
            "fear_greed"
        ] = df[
            "fear_greed"
        ]

    position_weights = cfg[
        "sub_weights"
    ].get(
        "Positionierung",
        {},
    )

    scores[
        "Positionierung"
    ], position_coverage = (
        normalize_weighted_scores(
            positioning_scores,
            position_weights,
        )
    )

    # -------------------------
    # Market Internals
    # -------------------------

    internals = pd.DataFrame(
        index=df.index
    )

    if "breadth" in df:
        internals[
            "breadth"
        ] = df[
            "breadth"
        ]

    if "obv_momentum" in df:
        internals[
            "obv_momentum"
        ] = percentile_score(
            df["obv_momentum"],
            LOOKBACKS[
                "obv_momentum"
            ],
        )

    if "vol_percentile" in df:
        internals[
            "vol_percentile"
        ] = df[
            "vol_percentile"
        ]

    internal_weights = cfg[
        "sub_weights"
    ].get(
        "Marktinterna",
        {},
    )

    scores[
        "Marktinterna"
    ], internal_coverage = (
        normalize_weighted_scores(
            internals,
            internal_weights,
        )
    )

    # -------------------------
    # Technical
    # -------------------------

    tech = pd.DataFrame(
        index=df.index
    )

    if "distance_200ma" in df:
        tech[
            "distance_200ma"
        ] = percentile_score(
            df["distance_200ma"],
            756,
        )

    if "distance_50ma" in df:
        tech[
            "distance_50ma"
        ] = percentile_score(
            df["distance_50ma"],
            504,
        )

    if "rsi_momentum" in df:
        tech[
            "rsi_momentum"
        ] = df[
            "rsi_momentum"
        ]

    scores[
        "Technischer_Trend"
    ], tech_coverage = (
        normalize_weighted_scores(
            tech,
            TECH_WEIGHTS,
        )
    )

    # -------------------------
    # Fundamentals
    # -------------------------

    fundamental = pd.DataFrame(
        index=df.index
    )

    if "pe_valuation" in df:

        fundamental[
            "pe_valuation"
        ] = percentile_score(
            df["pe_valuation"],
            LOOKBACKS[
                "pe_valuation"
            ],
            invert=True,
            min_periods=60,
        )

    if "inventories" in df:

        # Niedrige Lagerbestände =
        # grundsätzlich bullischer für Öl.
        fundamental[
            "inventories"
        ] = percentile_score(
            df["inventories"],
            LOOKBACKS[
                "inventories"
            ],
            invert=True,
        )

    fundamental_weights = cfg[
        "sub_weights"
    ].get(
        "Fundamentale_Faktoren",
        {},
    )

    scores[
        "Fundamentale_Faktoren"
    ], fundamental_coverage = (
        normalize_weighted_scores(
            fundamental,
            fundamental_weights,
        )
    )

    # -------------------------
    # Early Warning
    # -------------------------

    early = pd.DataFrame(
        index=df.index
    )

    if "credit_stress" in df:

        early[
            "credit_stress"
        ] = percentile_score(
            df["credit_stress"],
            LOOKBACKS[
                "credit_stress"
            ],
            invert=True,
        )

    if "move_index" in df:

        early[
            "move_index"
        ] = percentile_score(
            df["move_index"],
            LOOKBACKS[
                "move_index"
            ],
            invert=True,
        )

    scores[
        "Fruehwarnindikatoren"
    ], early_coverage = (
        normalize_weighted_scores(
            early,
            EARLY_WARNING_WEIGHTS,
        )
    )

    # ========================================================
    # REGIME SCORE
    # ========================================================

    regime_columns = [
        "Makroökonomie",
        "Positionierung",
        "Marktinterna",
        "Technischer_Trend",
        "Fundamentale_Faktoren",
        "Fruehwarnindikatoren",
    ]

    weighted_scores = pd.DataFrame(
        index=df.index
    )

    for column in regime_columns:

        if column in scores:

            weighted_scores[column] = (
                scores[column]
            )

    regime_weights = cfg[
        "regime_weights"
    ]

    final_score, regime_coverage = (
        normalize_weighted_scores(
            weighted_scores,
            regime_weights,
        )
    )

    # ========================================================
    # MCI
    # ========================================================

    mci = []

    for date in scores.index:

        row = scores.loc[date]

        available_weights = {
            k: v
            for k, v in regime_weights.items()
            if k in row.index
            and pd.notna(row[k])
        }

        coverage = (
            sum(
                available_weights.values()
            )
            /
            sum(regime_weights.values())
        )

        mci.append(
            calculate_mci(
                row,
                regime_weights,
                coverage,
            )
        )

    result = scores.copy()

    result["Final_Regime_Score"] = (
        final_score
    )

    result["MCI"] = mci

    result["Asset_Price"] = price

    result["Raw_Volatility"] = df.get(
        "vol_raw"
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    result["Data_Coverage"] = (
        regime_coverage * 100
    )

    # ========================================================
    # LOOK-AHEAD PROTECTION
    # ========================================================
    #
    # Das Tages-Regime wird erst am Folgetag
    # als handelbares Regime verwendet.
    #
    result["Tradable_Regime"] = (
        result["Final_Regime_Score"]
        .shift(1)
    )

    result["Tradable_MCI"] = (
        result["MCI"].shift(1)
    )

    return (
        result.dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        feed_status,
    )


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Quant Regime V2")

    selected_asset = st.selectbox(
        "Asset",
        list(ASSET_CONFIGS.keys()),
    )

    st.markdown("---")

    st.caption(
        "Version 2 trennt Regime, "
        "Sentiment und Intraday-Execution."
    )


# ============================================================
# 12. LOAD DATA
# ============================================================

with st.spinner(
    f"Lade quantitative Daten für {selected_asset}..."
):

    df_dash, feed_status = (
        build_dashboard(
            selected_asset
        )
    )


if df_dash.empty:

    st.error(
        "Keine ausreichenden Marktdaten verfügbar."
    )

    st.stop()


today = df_dash.iloc[-1]


# ============================================================
# 13. GOOGLE TRENDS
# ============================================================

trend_score, trend_spread, trend_live = (
    fetch_google_trends(
        selected_asset
    )
)


# ============================================================
# 14. TITLE
# ============================================================

st.title(
    "📊 Quant Regime Dashboard V2"
)

st.caption(
    f"Asset: **{selected_asset}** | "
    f"Datenstand: "
    f"{df_dash.index[-1].strftime('%d.%m.%Y')}"
)

st.markdown("---")


# ============================================================
# 15. CORE METRICS
# ============================================================

score = today["Final_Regime_Score"]
mci = today["MCI"]
coverage = today["Data_Coverage"]

previous_score = (
    df_dash["Final_Regime_Score"]
    .iloc[-2]
    if len(df_dash) > 1
    else score
)

delta_score = score - previous_score


c1, c2, c3 = st.columns(3)

c1.metric(
    "Regime Score",
    f"{score:.1f} / 100",
    f"{delta_score:+.1f}",
)

c2.metric(
    "Model Confidence",
    f"{mci:.1f}%" if pd.notna(mci) else "n/a",
)

c3.metric(
    "Datenabdeckung",
    f"{coverage:.0f}%",
)

st.info(
    f"**Aktuelles Regime:** "
    f"{regime_label(score)}"
)


# ============================================================
# 16. REGIME INTERPRETATION
# ============================================================

if score >= 65:

    regime_bias = "🟢 LONG BIAS"
    regime_text = (
        "Das übergeordnete Regime bevorzugt "
        "Long-Setups. Der Regime Score ist "
        "jedoch kein unmittelbares Entry-Signal."
    )

elif score <= 35:

    regime_bias = "🔴 SHORT BIAS"
    regime_text = (
        "Das übergeordnete Regime bevorzugt "
        "Short-Setups. Der Regime Score ist "
        "jedoch kein unmittelbares Entry-Signal."
    )

else:

    regime_bias = "🟡 NEUTRAL"
    regime_text = (
        "Kein ausreichender Regime-Vorteil. "
        "Intraday-Setups sollten selektiv "
        "gehandelt werden."
    )


st.subheader("🎯 Regime Bias")

b1, b2 = st.columns(2)

b1.metric(
    "Richtung",
    regime_bias,
)

b2.metric(
    "Tradable Regime",
    (
        f"{today['Tradable_Regime']:.1f}"
        if pd.notna(today["Tradable_Regime"])
        else "n/a"
    ),
)

st.info(regime_text)


# ============================================================
# 17. VOLATILITY FILTER
# ============================================================

if pd.notna(today.get("Raw_Volatility")):

    volatility = today["Raw_Volatility"]

    st.subheader(
        "⚠️ Volatilitätsfilter"
    )

    st.metric(
        ASSET_CONFIGS[
            selected_asset
        ]["volatility_ticker"],
        f"{volatility:.2f}",
    )

    # historisches Percentile
    volatility_series = df_dash[
        "Raw_Volatility"
    ].dropna()

    if len(volatility_series) >= 60:

        vol_percentile_now = (
            volatility_series
            .iloc[:-1]
            <= volatility
        ).mean() * 100

    else:

        vol_percentile_now = np.nan

    if (
        pd.notna(vol_percentile_now)
        and vol_percentile_now >= 95
    ):

        st.error(
            "🚨 Extremes Volatilitätsregime "
            "(oberstes historisches 5%-Segment). "
            "Positionsgröße deutlich reduzieren."
        )

    elif (
        pd.notna(vol_percentile_now)
        and vol_percentile_now >= 90
    ):

        st.warning(
            "⚠️ Erhöhte Volatilität. "
            "Entries selektiver wählen und "
            "Positionsgröße reduzieren."
        )

    else:

        st.success(
            "Volatilität aktuell nicht im "
            "historischen Extrembereich."
        )


# ============================================================
# 18. GOOGLE TRENDS
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Retail Sentiment – Google Trends"
)

st.caption(
    "Separater Kontraindikator. "
    "Er beeinflusst den Regime Score nicht direkt."
)

gt1, gt2, gt3 = st.columns(3)

if trend_live:

    gt1.metric(
        "Retail Contrarian Score",
        f"{trend_score:.1f} / 100",
    )

    gt2.metric(
        "Net Spread",
        f"{trend_spread:+.2f} σ",
    )

    if trend_score >= 65:

        gt3.success(
            "🟢 Angst-/Paniküberhang"
        )

    elif trend_score <= 35:

        gt3.error(
            "🔴 Euphorie-/Gierüberhang"
        )

    else:

        gt3.info(
            "🟡 Kein extremes Sentiment"
        )

else:

    gt1.metric(
        "Google Trends",
        "Offline",
    )

    gt2.info(
        "Kein Sentiment-Signal."
    )

    gt3.info(
        "Regime Score bleibt unverändert."
    )


# ============================================================
# 19. SIX PILLARS
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Regime-Treiber"
)

pillar_names = [
    "Makroökonomie",
    "Positionierung",
    "Marktinterna",
    "Technischer_Trend",
    "Fundamentale_Faktoren",
    "Fruehwarnindikatoren",
]

cols = st.columns(3)

for i, pillar in enumerate(pillar_names):

    value = today.get(
        pillar,
        np.nan,
    )

    weight = (
        ASSET_CONFIGS[
            selected_asset
        ]["regime_weights"]
        .get(pillar, 0)
        * 100
    )

    with cols[i % 3]:

        if pd.isna(value):

            st.metric(
                pillar.replace("_", " "),
                "n/a",
            )

            st.caption(
                f"Gewichtung: {weight:.0f}%"
            )

        else:

            emoji = (
                "🟢"
                if value >= 60
                else
                "🔴"
                if value <= 40
                else
                "🟡"
            )

            st.metric(
                f"{emoji} {pillar.replace('_', ' ')}",
                f"{value:.1f}",
            )

            st.caption(
                f"Gewichtung: {weight:.0f}%"
            )


# ============================================================
# 20. INTRADAY EXECUTION
# ============================================================

st.markdown("---")

st.subheader(
    "⚡ Intraday Execution Layer"
)

st.caption(
    "Der Regime Score bestimmt nur den "
    "Richtungsvorteil. Diese Ebene entscheidet "
    "über die konkrete Trade-Freigabe."
)


trend_score_technical = today.get(
    "Technischer_Trend",
    np.nan,
)

early_score = today.get(
    "Fruehwarnindikatoren",
    np.nan,
)

macro_score = today.get(
    "Makroökonomie",
    np.nan,
)


trend_ok = (
    pd.notna(trend_score_technical)
    and trend_score_technical >= 55
)

market_stable = (
    pd.notna(early_score)
    and early_score >= 45
)

macro_ok = (
    pd.notna(macro_score)
    and macro_score >= 50
)

mci_ok = (
    pd.notna(mci)
    and mci >= 60
)


# ============================================================
# DAY PROFILE
# ============================================================

berlin_now = pd.Timestamp.now(
    tz="Europe/Berlin"
)

weekday = berlin_now.weekday()

day_profile = {
    0:
        "Montag – erhöhte Gefahr von False Breakouts.",
    1:
        "Dienstag – häufig bessere Trendetablierung.",
    2:
        "Mittwoch – Trendfortsetzung oder Reversal.",
    3:
        "Donnerstag – häufig erhöhtes Momentum.",
    4:
        "Freitag – Profit-Taking und Weekend-Risk.",
}

profile_text = day_profile.get(
    weekday,
    "Wochenende",
)


# ============================================================
# OPEX
# ============================================================

is_opex = (
    berlin_now.month in [3, 6, 9, 12]
    and weekday == 4
    and 15 <= berlin_now.day <= 21
)


c1, c2 = st.columns(2)

with c1:

    st.markdown(
        "### 1. Regime / Struktur"
    )

    check_trend = st.checkbox(
        f"Technischer Trend bestätigt "
        f"(Score {trend_score_technical:.1f})"
        if pd.notna(trend_score_technical)
        else "Technischer Trend bestätigt",
        value=trend_ok,
    )

    check_market = st.checkbox(
        f"Makro-/Kreditumfeld akzeptabel "
        f"(Score {early_score:.1f})"
        if pd.notna(early_score)
        else "Makro-/Kreditumfeld akzeptabel",
        value=market_stable,
    )

    check_macro = st.checkbox(
        f"Makro-Rückenwind "
        f"(Score {macro_score:.1f})"
        if pd.notna(macro_score)
        else "Makro-Rückenwind",
        value=macro_ok,
    )

    check_mci = st.checkbox(
        f"Modell-Confidence ausreichend "
        f"(MCI {mci:.1f}%)"
        if pd.notna(mci)
        else "Modell-Confidence ausreichend",
        value=mci_ok,
    )


with c2:

    st.markdown(
        "### 2. Execution / Risiko"
    )

    check_news = st.checkbox(
        "Keine High-Impact-News "
        "innerhalb der nächsten 60 Minuten",
        value=False,
    )

    check_opex = st.checkbox(
        "Kein erhöhtes Options-/OPEX-Risiko",
        value=not is_opex,
    )

    check_crv = st.checkbox(
        "Mindestens 1:2 CRV zum nächsten Ziel",
        value=False,
    )

    check_timing = st.checkbox(
        "US-Eröffnung / Initial Balance "
        "ist bereits etabliert",
        value=False,
    )


# ============================================================
# 21. EXECUTION RESULT
# ============================================================

checks = [
    check_trend,
    check_market,
    check_macro,
    check_mci,
    check_news,
    check_opex,
    check_crv,
    check_timing,
]

fulfilled = sum(checks)

st.progress(
    fulfilled / len(checks)
)

st.caption(
    f"**{fulfilled} von {len(checks)} "
    f"Execution-Kriterien erfüllt.**"
)


# ============================================================
# 22. FINAL TRADE DECISION
# ============================================================

if (
    fulfilled == 8
    and score >= 65
):

    st.success(
        "🟢 LONG EXECUTION – "
        "Regime und Execution-Filter bestätigen "
        "einen Long-Bias."
    )

elif (
    fulfilled == 8
    and score <= 35
):

    st.error(
        "🔴 SHORT EXECUTION – "
        "Regime und Execution-Filter bestätigen "
        "einen Short-Bias."
    )

elif (
    score >= 65
    and fulfilled >= 6
):

    st.warning(
        "🟡 LONG BIAS – Regime ist bullisch, "
        "aber mindestens ein Execution-Filter fehlt."
    )

elif (
    score <= 35
    and fulfilled >= 6
):

    st.warning(
        "🟠 SHORT BIAS – Regime ist bärisch, "
        "aber mindestens ein Execution-Filter fehlt."
    )

else:

    st.info(
        "⚪ NO TRADE – kein ausreichender "
        "Vorteil für eine aktive Position."
    )


# ============================================================
# 23. DATA QUALITY
# ============================================================

st.markdown("---")

st.subheader(
    "📡 Datenqualität"
)

status_cols = st.columns(2)

for i, (
    source,
    status,
) in enumerate(feed_status.items()):

    if status:

        status_cols[
            i % 2
        ].success(
            f"🟢 {source}"
        )

    else:

        status_cols[
            i % 2
        ].warning(
            f"🟠 {source}: nicht verfügbar"
        )


# ============================================================
# 24. HISTORICAL REGIME CHART
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Regime-Historie"
)

plot_df = df_dash.tail(
    min(252, len(df_dash))
).copy()


fig = make_subplots(
    specs=[
        [{"secondary_y": True}]
    ]
)


fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df[
            "Final_Regime_Score"
        ],
        name="Regime Score",
        fill="tozeroy",
    ),
    secondary_y=False,
)


fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df[
            "MCI"
        ],
        name="MCI",
        line=dict(
            width=2,
        ),
    ),
    secondary_y=False,
)


fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df[
            "Asset_Price"
        ],
        name=selected_asset,
        line=dict(
            width=2,
        ),
    ),
    secondary_y=True,
)


fig.add_hline(
    y=65,
    line_dash="dash",
    secondary_y=False,
)

fig.add_hline(
    y=35,
    line_dash="dash",
    secondary_y=False,
)


fig.update_yaxes(
    title_text="Score",
    range=[0, 100],
    secondary_y=False,
)

fig.update_yaxes(
    title_text="Preis",
    secondary_y=True,
)

fig.update_layout(
    height=500,
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# 25. COMPONENT DIAGNOSTICS
# ============================================================

st.markdown("---")

with st.expander(
    "🔬 Quantitative Komponenten anzeigen"
):

    diagnostics = []

    for column in df_dash.columns:

        if column in [
            "Asset_Price",
            "Raw_Volatility",
        ]:
            continue

        diagnostics.append({
            "Komponente": column,
            "Aktuell": (
                round(
                    float(today[column]),
                    2,
                )
                if pd.notna(today[column])
                else None
            ),
        })

    diagnostic_df = pd.DataFrame(
        diagnostics
    )

    st.dataframe(
        diagnostic_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 26. METHODOLOGICAL NOTES
# ============================================================

st.markdown("---")

with st.expander(
    "ℹ️ Modellhinweise"
):

    st.markdown(
        """
### Was der Score bedeutet

Der **Regime Score** ist ein aggregierter
Marktregime-Indikator zwischen 0 und 100.

- 80–100: starkes Risk-On
- 65–80: Risk-On
- 55–65: leicht bullisch
- 45–55: neutral
- 35–45: Risk-Off
- 0–35: starkes Risk-Off

### Wichtig

Der Score ist **kein automatisches Kaufsignal**.

Das Tages-Regime wird um einen Tag verschoben,
bevor es als `Tradable Regime` verwendet wird.
Damit wird verhindert, dass das Modell den
End-of-Day-Wert desselben Tages für einen Trade
verwendet, der zeitlich davor stattgefunden hätte.

Google Trends wird bewusst separat dargestellt
und beeinflusst den Regime Score nicht direkt.

Fehlende Daten werden nicht künstlich mit 50,
einem festen Zinssatz oder einer festen
Liquidität ersetzt. Die betroffene Komponente
wird aus der Gewichtung entfernt und die
Datenabdeckung reduziert entsprechend den MCI.
"""
    )