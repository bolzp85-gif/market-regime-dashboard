
import io
import zipfile
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None


# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Asset Regime Dashboard",
    page_icon="📊",
    layout="centered",
)


# ============================================================
# 1. ASSET CONFIGURATIONS
# ============================================================

ASSET_CONFIGS = {
    "S&P 500": {
        "ticker": "^GSPC",
        "volatility_ticker": "^VIX",
        "cot_code": "E-MINI S&P 500",
        "volatility_mode": "index",
        "volatility_threshold": 30.0,
        "invert_inverts": [
            "vix_score", "pe_valuation", "credit_spreads",
            "move_index", "usd_index", "fed_policy", "real_yields",
        ],
        "Saeulen_Gewichte": {
            "Makroökonomie": 0.25,
            "Positionierung": 0.15,
            "Marktinterna": 0.20,
            "Technischer_Trend": 0.20,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.10,
        },
        "Sub_Gewichte": {
            "Positionierung": {
                "cot_commercials": 0.50,
                "fear_greed": 0.50,
            },
            "Marktinterna": {
                "market_momentum": 0.50,
                "vix_score": 0.50,
            },
            "Fundamentale_Faktoren": {
                "pe_valuation": 1.00,
            },
        },
    },

    "Nasdaq 100": {
        "ticker": "NQ=F",
        "volatility_ticker": "^VXN",
        "cot_code": "NASDAQ-100",
        "volatility_mode": "index",
        "volatility_threshold": 35.0,
        "invert_inverts": [
            "vix_score", "pe_valuation", "credit_spreads",
            "move_index", "usd_index", "fed_policy", "real_yields",
        ],
        "Saeulen_Gewichte": {
            "Makroökonomie": 0.25,
            "Positionierung": 0.15,
            "Marktinterna": 0.20,
            "Technischer_Trend": 0.20,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.10,
        },
        "Sub_Gewichte": {
            "Positionierung": {
                "cot_commercials": 0.50,
                "fear_greed": 0.50,
            },
            "Marktinterna": {
                "market_momentum": 0.50,
                "vix_score": 0.50,
            },
            "Fundamentale_Faktoren": {
                "pe_valuation": 1.00,
            },
        },
    },

    "Gold (XAU/USD)": {
        "ticker": "GC=F",
        "volatility_ticker": "^GVZ",
        "cot_code": "GOLD",
        "volatility_mode": "index",
        "volatility_threshold": 25.0,
        "invert_inverts": [
            "vix_score", "usd_index", "real_yields", "fed_policy",
        ],
        "Saeulen_Gewichte": {
            "Makroökonomie": 0.35,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.00,
            "Fruehwarnindikatoren": 0.10,
        },
        "Sub_Gewichte": {
            "Positionierung": {
                "cot_commercials": 1.00,
            },
            "Marktinterna": {
                "obv_momentum": 0.50,
                "vix_score": 0.50,
            },
            "Fundamentale_Faktoren": {},
        },
    },

    "WTI Crude Oil": {
        "ticker": "CL=F",
        "volatility_ticker": "^OVX",
        "cot_code": "CRUDE OIL",
        "volatility_mode": "index",
        "volatility_threshold": 45.0,
        "invert_inverts": [
            "vix_score", "usd_index", "inventories",
        ],
        "Saeulen_Gewichte": {
            "Makroökonomie": 0.30,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.20,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.00,
        },
        "Sub_Gewichte": {
            "Positionierung": {
                "cot_commercials": 1.00,
            },
            "Marktinterna": {
                "obv_momentum": 0.50,
                "vix_score": 0.50,
            },
            "Fundamentale_Faktoren": {
                "inventories": 1.00,
            },
        },
    },

    "EUR/USD": {
        "ticker": "EURUSD=X",
        "volatility_ticker": None,
        "cot_code": "EURO FX",
        "volatility_mode": "realized_fx",
        "volatility_threshold": 12.0,
        "invert_inverts": [
            "vix_score",
            "usd_index",
            "real_yields",
            "fed_policy",
            "rate_differential",
            "credit_spreads",
        ],
        "Saeulen_Gewichte": {
            "Makroökonomie": 0.40,
            "Positionierung": 0.20,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.00,
            "Fruehwarnindikatoren": 0.10,
        },
        "Sub_Gewichte": {
            "Makroökonomie": {
                "rate_differential": 0.40,
                "real_yields": 0.20,
                "usd_index": 0.15,
                "net_liquidity": 0.25,
            },
            "Positionierung": {
                "cot_commercials": 1.00,
            },
            "Marktinterna": {
                "market_momentum": 0.50,
                "vix_score": 0.50,
            },
            "Fundamentale_Faktoren": {},
        },
    },
}


# ============================================================
# 2. BASE SUB-WEIGHTS
# ============================================================

SUB_WEIGHTS_BASE = {
    "Makroökonomie": {
        "fed_policy": 0.20,
        "real_yields": 0.30,
        "usd_index": 0.20,
        "net_liquidity": 0.30,
    },
    "Technischer_Trend": {
        "distance_200ma": 0.35,
        "distance_50ma": 0.35,
        "rsi_momentum": 0.30,
    },
    "Fruehwarnindikatoren": {
        "credit_spreads": 0.60,
        "move_index": 0.40,
    },
}


LOOKBACK_CONFIG = {
    "fed_policy": 1260,
    "real_yields": 756,
    "net_liquidity": 756,
    "credit_spreads": 756,
    "usd_index": 504,
    "rate_differential": 756,
    "inventories": 756,
    "pe_valuation": 756,
}


# ============================================================
# 3. GOOGLE TRENDS CONFIG
# ============================================================

TREND_KEYWORD_MAP = {
    "S&P 500": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["buy stocks", "buy the dip"],
        "bear": ["stock market crash", "recession"],
    },
    "Nasdaq 100": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["tech stocks", "buy the dip"],
        "bear": ["market crash", "tech bubble"],
    },
    "Gold (XAU/USD)": {
        "geo": "DE",
        "lang": "de-DE",
        "bull": ["Gold kaufen", "Goldmünzen"],
        "bear": ["Gold verkaufen", "Altgold"],
    },
    "WTI Crude Oil": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["buy oil", "oil prices"],
        "bear": ["oil price crash", "sell oil"],
    },
    "EUR/USD": {
        "geo": "US",
        "lang": "en-US",
        "bull": ["buy euro", "EUR USD"],
        "bear": ["sell euro", "EUR USD forecast"],
    },
}


# ============================================================
# 4. HELPERS
# ============================================================

def strip_timezone(obj):
    dt = pd.to_datetime(obj)
    if hasattr(dt, "dt"):
        if dt.dt.tz is not None:
            return dt.dt.tz_convert(None)
        return dt
    if getattr(dt, "tz", None) is not None:
        return dt.tz_convert(None)
    return dt


def normalize_to_percentile(series, lookback=252, invert=False):
    clean = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    if clean.isna().all():
        return pd.Series(50.0, index=series.index)

    rolling_mean = clean.rolling(
        lookback, min_periods=min(20, lookback)
    ).mean()
    rolling_std = clean.rolling(
        lookback, min_periods=min(20, lookback)
    ).std()

    rolling_std = rolling_std.replace(0, np.nan)
    z = (clean - rolling_mean) / rolling_std
    percentile = pd.Series(norm.cdf(z) * 100, index=series.index)

    if invert:
        percentile = 100 - percentile

    return percentile.clip(0, 100).ffill().bfill().fillna(50.0)


def calculate_mci(scores, weights):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(scores) & np.isfinite(weights)
    scores = scores[valid]
    weights = weights[valid]

    if len(scores) == 0 or len(weights) == 0 or weights.sum() <= 0:
        return 0.0

    weights = weights / weights.sum()
    mean = np.average(scores, weights=weights)
    variance = np.average((scores - mean) ** 2, weights=weights)
    std = np.sqrt(variance)

    return round(float(np.clip(100 * (1 - std / 50), 0, 100)), 1)


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


def safe_reindex_series(source_series, target_index):
    if (
        source_series is None
        or not isinstance(source_series, pd.Series)
        or source_series.empty
    ):
        return None

    s = source_series.copy()
    s.index = strip_timezone(s.index).floor("D")
    s = s[~s.index.duplicated(keep="last")].sort_index()

    target = strip_timezone(target_index).floor("D")
    out = s.reindex(target, method="ffill").ffill().bfill()
    out.index = target_index
    return out


def get_close_and_volume(data):
    if isinstance(data.columns, pd.MultiIndex):
        levels = [
            list(data.columns.get_level_values(i))
            for i in range(data.columns.nlevels)
        ]

        if "Close" in levels[0]:
            close = data["Close"]
            volume = data["Volume"] if "Volume" in levels[0] else None
        elif "Close" in levels[1]:
            close = data.xs("Close", axis=1, level=1)
            volume = (
                data.xs("Volume", axis=1, level=1)
                if "Volume" in levels[1]
                else None
            )
        else:
            close, volume = data, None
    else:
        close = data.copy()
        volume = data["Volume"] if "Volume" in data.columns else None

    return close, volume


# ============================================================
# 5. GOOGLE TRENDS
# ============================================================

@st.cache_data(ttl=21600)
def fetch_google_trends_sentiment(asset_name):
    if TrendReq is None:
        return 50.0, 0.0, False

    cfg = TREND_KEYWORD_MAP[asset_name]
    all_kws = cfg["bull"] + cfg["bear"]

    try:
        pytrends = TrendReq(hl=cfg["lang"], tz=360)
        pytrends.build_payload(
            all_kws,
            timeframe="today 3-m",
            geo=cfg["geo"],
        )

        df = pytrends.interest_over_time()

        if df.empty:
            return 50.0, 0.0, False

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        def calc_z(series):
            mean = series.rolling(21, min_periods=5).mean()
            std = series.rolling(21, min_periods=5).std().replace(0, np.nan)
            return (series - mean) / std

        bull = [x for x in cfg["bull"] if x in df.columns]
        bear = [x for x in cfg["bear"] if x in df.columns]

        if not bull or not bear:
            return 50.0, 0.0, False

        z_bull = sum(calc_z(df[x]) for x in bull) / len(bull)
        z_bear = sum(calc_z(df[x]) for x in bear) / len(bear)
        spread = (z_bull - z_bear).dropna()

        if spread.empty:
            return 50.0, 0.0, False

        latest = float(spread.iloc[-1])
        score = float(np.clip(50 - latest * 15, 0, 100))

        return round(score, 1), round(latest, 2), True

    except Exception:
        return 50.0, 0.0, False


# ============================================================
# 6. CNN FEAR & GREED
# ============================================================

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():
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
        res = requests.get(url, headers=headers, timeout=8)

        if res.status_code != 200:
            return None, False

        data = res.json()
        hist = (
            data.get("fear_and_greed_historical", {})
            .get("data", [])
        )

        if not hist:
            return None, False

        df = pd.DataFrame(hist)
        df["Date"] = (
            strip_timezone(pd.to_datetime(df["x"], unit="ms"))
            .dt.floor("D")
        )

        df = df.drop_duplicates("Date", keep="last")

        return (
            df.set_index("Date")["y"].sort_index(),
            True,
        )

    except Exception:
        return None, False


# ============================================================
# 7. CFTC COT
# ============================================================

@st.cache_data(ttl=86400)
def fetch_cot_data(asset_search_string):
    headers = {"User-Agent": "Mozilla/5.0"}
    current_year = pd.Timestamp.now().year
    frames = []

    for year in [current_year - 1, current_year]:
        url = (
            "https://www.cftc.gov/files/dea/history/"
            f"fut_com_txt_{year}.zip"
        )

        try:
            res = requests.get(url, headers=headers, timeout=12)

            if res.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                names = z.namelist()
                if not names:
                    continue

                with z.open(names[0]) as f:
                    df_year = pd.read_csv(f, low_memory=False)

            first_col = df_year.columns[0]
            rows = df_year[
                df_year[first_col]
                .astype(str)
                .str.contains(asset_search_string, case=False, na=False)
            ]

            if not rows.empty:
                frames.append(rows)

        except Exception:
            continue

    if not frames:
        return None, False

    try:
        df = pd.concat(frames, ignore_index=True)

        date_cols = [
            c for c in df.columns if "As_of_Date" in str(c)
        ]
        long_cols = [
            c for c in df.columns
            if "Comm_Positions_Long_All" in str(c)
        ]
        short_cols = [
            c for c in df.columns
            if "Comm_Positions_Short_All" in str(c)
        ]

        if not date_cols or not long_cols or not short_cols:
            return None, False

        df["Date"] = strip_timezone(
            pd.to_datetime(
                df[date_cols[0]].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
        ).dt.floor("D")

        df["Net_Commercials"] = (
            pd.to_numeric(df[long_cols[0]], errors="coerce")
            - pd.to_numeric(df[short_cols[0]], errors="coerce")
        )

        df = (
            df.dropna(subset=["Date", "Net_Commercials"])
            .drop_duplicates("Date", keep="last")
            .sort_values("Date")
        )

        return df.set_index("Date")["Net_Commercials"], True

    except Exception:
        return None, False


# ============================================================
# 8. MAIN DATA PIPELINE
# ============================================================

@st.cache_data(ttl=3600)
def fetch_multi_asset_data(selected_asset, fred_api_key):
    cfg = ASSET_CONFIGS[selected_asset]
    feed_status = {}

    tickers = {
        "asset": cfg["ticker"],
        "dxy": "DX=F",
        "move": "^MOVE",
        "hyg": "HYG",
        "lqd": "LQD",
    }

    if cfg["volatility_mode"] == "index":
        tickers["volatility"] = cfg["volatility_ticker"]

    try:
        data = yf.download(
            list(tickers.values()),
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame(), {"yFinance (Preis & Tech)": False}

    if data.empty:
        return pd.DataFrame(), {"yFinance (Preis & Tech)": False}

    close_data, volume_data = get_close_and_volume(data)

    rename_map = {
        ticker: key for key, ticker in tickers.items()
    }

    close_data = close_data.rename(columns=rename_map)

    if isinstance(volume_data, pd.DataFrame):
        volume_data = volume_data.rename(columns=rename_map)

    close_data = (
        close_data.apply(pd.to_numeric, errors="coerce")
        .ffill()
        .bfill()
        .dropna(how="all")
    )

    feed_status["yFinance (Preis & Tech)"] = not close_data.empty

    if close_data.empty or "asset" not in close_data.columns:
        return pd.DataFrame(), feed_status

    price = close_data["asset"].copy()
    df_raw = pd.DataFrame(index=close_data.index)

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        isinstance(volume_data, pd.DataFrame)
        and "asset" in volume_data.columns
    ):
        asset_volume = volume_data["asset"].ffill().bfill()
        has_volume = True
    else:
        asset_volume = pd.Series(1000.0, index=price.index)
        has_volume = False

    feed_status["Volumen / Orderflow Feed"] = has_volume

    # --------------------------------------------------------
    # Technical trend
    # --------------------------------------------------------

    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()

    df_raw["distance_50ma"] = ((price - ma50) / ma50) * 100
    df_raw["distance_200ma"] = ((price - ma200) / ma200) * 100

    delta = price.diff()

    gain = (
        delta.where(delta > 0, 0)
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    df_raw["rsi_momentum"] = (
        100 - (100 / (1 + rs))
    ).fillna(50.0)

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    if cfg["volatility_mode"] == "index":
        if "volatility" in close_data.columns:
            df_raw["vix_score"] = close_data["volatility"]
        else:
            df_raw["vix_score"] = 20.0
            feed_status["Volatilitätsindex"] = False
    else:
        # EUR/USD: annualisierte 20-Tage-realized volatility.
        df_raw["vix_score"] = (
            price.pct_change()
            .rolling(20)
            .std()
            * np.sqrt(252)
            * 100
        )
        feed_status["EUR/USD Realisierte Volatilität"] = True

    # --------------------------------------------------------
    # Dollar
    # --------------------------------------------------------

    if "dxy" in close_data.columns:
        df_raw["usd_index"] = close_data["dxy"]
    else:
        df_raw["usd_index"] = 100.0

    # --------------------------------------------------------
    # MOVE
    # --------------------------------------------------------

    if "move" in close_data.columns:
        df_raw["move_index"] = close_data["move"]
    else:
        df_raw["move_index"] = 100.0

    # --------------------------------------------------------
    # Credit proxy
    # --------------------------------------------------------

    if "lqd" in close_data.columns and "hyg" in close_data.columns:
        df_raw["credit_spreads"] = (
            close_data["lqd"] / close_data["hyg"]
        )
    else:
        df_raw["credit_spreads"] = 1.0

    # --------------------------------------------------------
    # Market momentum
    # --------------------------------------------------------

    df_raw["market_momentum"] = (
        price.pct_change()
        .rolling(20)
        .sum()
        * 100
    )

    # --------------------------------------------------------
    # OBV momentum
    # --------------------------------------------------------

    obv_daily = np.select(
        [delta > 0, delta < 0],
        [asset_volume, -asset_volume],
        default=0,
    )

    obv = pd.Series(
        obv_daily,
        index=price.index,
    ).cumsum()

    obv_ema = obv.ewm(
        span=50,
        adjust=False,
    ).mean()

    df_raw["obv_momentum"] = (
        (obv - obv_ema)
        / obv_ema.abs().replace(0, np.nan)
    ) * 100

    # --------------------------------------------------------
    # Fundamental factor
    # --------------------------------------------------------
    # Keine künstliche konstante KGV-Zeitreihe.
    # Bis eine echte historische Bewertungsreihe angeschlossen
    # wird, bleibt dieser Faktor neutral = 50.

    df_raw["pe_valuation"] = 50.0

    # --------------------------------------------------------
    # FRED
    # --------------------------------------------------------

    fred_ok = False

    if fred_api_key and Fred is not None:
        try:
            fred = Fred(api_key=fred_api_key)

            walcl = safe_reindex_series(
                fred.get_series("WALCL"),
                df_raw.index,
            )

            tga = safe_reindex_series(
                fred.get_series("WTREGEN"),
                df_raw.index,
            )

            rrp = safe_reindex_series(
                fred.get_series("RRPONTSYD"),
                df_raw.index,
            )

            fedfunds = safe_reindex_series(
                fred.get_series("FEDFUNDS"),
                df_raw.index,
            )

            real_yield = safe_reindex_series(
                fred.get_series("DFII10"),
                df_raw.index,
            )

            if (
                walcl is not None
                and tga is not None
                and rrp is not None
            ):
                # WALCL/TGA: Mio. USD
                # RRPONTSYD: Mrd. USD
                # Ergebnis: Mrd. USD
                df_raw["net_liquidity"] = (
                    walcl
                    - tga
                    - (rrp * 1000.0)
                ) / 1000.0
            else:
                df_raw["net_liquidity"] = 6000.0

            df_raw["fed_policy"] = (
                fedfunds
                if fedfunds is not None
                else 5.25
            )

            df_raw["real_yields"] = (
                real_yield
                if real_yield is not None
                else 2.0
            )

            # EUR/USD: EZB-Einlagefazilität
            if selected_asset == "EUR/USD":
                ecb_rate = safe_reindex_series(
                    fred.get_series("ECBDFR"),
                    df_raw.index,
                )

                if ecb_rate is not None:
                    df_raw["ecb_policy"] = ecb_rate
                else:
                    df_raw["ecb_policy"] = 2.0

                df_raw["rate_differential"] = (
                    df_raw["fed_policy"]
                    - df_raw["ecb_policy"]
                )

            # WTI: US crude oil inventories
            if selected_asset == "WTI Crude Oil":
                inv = safe_reindex_series(
                    fred.get_series("WCESTUS1"),
                    df_raw.index,
                )

                df_raw["inventories"] = (
                    inv
                    if inv is not None
                    else 500000.0
                )

            fred_ok = True

        except Exception:
            df_raw["fed_policy"] = 5.25
            df_raw["real_yields"] = 2.0
            df_raw["net_liquidity"] = 6000.0

            if selected_asset == "EUR/USD":
                df_raw["ecb_policy"] = 2.0
                df_raw["rate_differential"] = (
                    df_raw["fed_policy"]
                    - df_raw["ecb_policy"]
                )

            if selected_asset == "WTI Crude Oil":
                df_raw["inventories"] = 500000.0

    else:
        df_raw["fed_policy"] = 5.25
        df_raw["real_yields"] = 2.0
        df_raw["net_liquidity"] = 6000.0

        if selected_asset == "EUR/USD":
            df_raw["ecb_policy"] = 2.0
            df_raw["rate_differential"] = (
                df_raw["fed_policy"]
                - df_raw["ecb_policy"]
            )

        if selected_asset == "WTI Crude Oil":
            df_raw["inventories"] = 500000.0

    feed_status["FRED API (Makro & Fed)"] = fred_ok

    # --------------------------------------------------------
    # COT
    # --------------------------------------------------------

    cot_data, cot_live = fetch_cot_data(
        cfg["cot_code"]
    )

    feed_status[
        f"CFTC COT ({cfg['cot_code']})"
    ] = cot_live

    cot = safe_reindex_series(
        cot_data,
        df_raw.index,
    )

    if cot is not None:
        df_raw["cot_commercials"] = cot
    else:
        # Neutraler Fallback statt künstlicher Kursabhängigkeit.
        df_raw["cot_commercials"] = 0.0

    # --------------------------------------------------------
    # CNN Fear & Greed
    # --------------------------------------------------------

    fg_data, fg_live = fetch_fear_and_greed()

    feed_status["CNN Fear & Greed"] = (
        fg_live
        if selected_asset in ["S&P 500", "Nasdaq 100"]
        else False
    )

    if selected_asset in ["S&P 500", "Nasdaq 100"]:
        if isinstance(fg_data, pd.Series):
            fg = safe_reindex_series(
                fg_data,
                df_raw.index,
            )

            df_raw["fear_greed"] = (
                fg
                if fg is not None
                else 55.0
            )
        else:
            df_raw["fear_greed"] = 55.0
    else:
        # Aktien-Fear-&-Greed wird bewusst nicht in
        # Gold, Öl oder EUR/USD verwendet.
        df_raw["fear_greed"] = 50.0

    # ========================================================
    # NORMALIZATION
    # ========================================================

    df_norm = pd.DataFrame(
        index=df_raw.index
    )

    inverts = cfg["invert_inverts"]

    for col in df_raw.columns:
        df_norm[col] = normalize_to_percentile(
            df_raw[col],
            lookback=LOOKBACK_CONFIG.get(
                col,
                252,
            ),
            invert=(col in inverts),
        )

    # ========================================================
    # DASHBOARD SCORES
    # ========================================================

    df_dashboard = pd.DataFrame(
        index=df_raw.index
    )

    df_dashboard["Raw_Volatility"] = (
        df_raw["vix_score"]
    )

    active_sub_weights = {
        category: dict(weights)
        for category, weights
        in SUB_WEIGHTS_BASE.items()
    }

    for category, weights in cfg[
        "Sub_Gewichte"
    ].items():
        active_sub_weights[category] = dict(
            weights
        )

    for saeule, indikatoren in (
        active_sub_weights.items()
    ):
        cols = [
            c for c in indikatoren
            if c in df_norm.columns
        ]

        weights = np.array(
            [
                indikatoren[c]
                for c in cols
            ],
            dtype=float,
        )

        if cols and weights.sum() > 0:
            weights = (
                weights
                / weights.sum()
            )

            df_dashboard[
                f"Saeule_{saeule}"
            ] = df_norm[
                cols
            ].dot(weights)

    # ========================================================
    # FINAL REGIME SCORE
    # ========================================================

    saeulen_cols = []
    saeulen_weights = []

    for saeule, weight in (
        cfg["Saeulen_Gewichte"].items()
    ):
        col = f"Saeule_{saeule}"

        if (
            col in df_dashboard.columns
            and weight > 0
        ):
            saeulen_cols.append(col)
            saeulen_weights.append(weight)

    if saeulen_cols:
        weights = np.array(
            saeulen_weights,
            dtype=float,
        )

        weights = (
            weights
            / weights.sum()
        )

        df_dashboard[
            "Final_Regime_Score"
        ] = (
            df_dashboard[
                saeulen_cols
            ]
            .dot(weights)
            .round(1)
        )

        mci_values = []

        for i in range(
            len(df_dashboard)
        ):
            mci_values.append(
                calculate_mci(
                    df_dashboard[
                        saeulen_cols
                    ].iloc[i].values,
                    weights,
                )
            )

        df_dashboard["MCI"] = mci_values

    else:
        df_dashboard[
            "Final_Regime_Score"
        ] = 50.0

        df_dashboard["MCI"] = 0.0

    df_dashboard[
        "Asset_Price"
    ] = price.ffill().bfill()

    return (
        df_dashboard.dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        feed_status,
    )


# ============================================================
# 9. SIDEBAR / API KEY
# ============================================================

with st.sidebar:
    st.title(
        "⚙️ Multi-Asset Selector"
    )

    selected_asset = st.selectbox(
        "🎯 Asset auswählen",
        list(ASSET_CONFIGS.keys()),
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "### 📡 API Live-Feed Monitor"
    )

FRED_API_KEY = ""

try:
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets[
            "FRED_API_KEY"
        ]
except Exception:
    pass


# ============================================================
# 10. LOAD DATA
# ============================================================

with st.spinner(
    f"Lade quantitative Daten für "
    f"{selected_asset}..."
):
    df_dash, feed_status = (
        fetch_multi_asset_data(
            selected_asset,
            FRED_API_KEY,
        )
    )


with st.sidebar:
    for source, is_live in (
        feed_status.items()
    ):
        if is_live:
            st.markdown(
                f"🟢 **{source}**"
            )
        else:
            st.markdown(
                f"⚠️ **{source}** "
                f"*(Fallback)*"
            )


if df_dash.empty:
    st.error(
        "⚠️ Marktdaten konnten nicht geladen "
        "werden. Bitte Yahoo Finance / "
        "Internetverbindung prüfen."
    )
    st.stop()


# ============================================================
# 11. CURRENT DATA / DELTAS
# ============================================================

heute = (
    df_dash.iloc[-1].copy()
)

heute["Delta_1D"] = (
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

heute["Delta_1W"] = (
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
# 12. TITLE
# ============================================================

st.title(
    "📊 Quant Regime Dashboard"
)

st.caption(
    f"Asset: **{selected_asset}** | "
    f"Stand: **"
    f"{df_dash.index[-1].strftime('%d.%m.%Y')}"
    f"**"
)

st.markdown("---")


# ============================================================
# 13. HERO METRICS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Final Regime Score",
        (
            f"{heute['Final_Regime_Score']:.1f}"
            f" / 100"
        ),
        (
            f"{heute['Delta_1D']:+.1f}"
            f" (Heute)"
        ),
    )

with col2:
    st.metric(
        "Model Consistency Index",
        (
            f"{heute['MCI']:.1f}%"
        ),
        (
            f"{heute['Delta_1W']:+.1f}"
            f" (vs. Vorwoche)"
        ),
        delta_color="off",
    )

st.caption(
    "Der MCI misst die Übereinstimmung "
    "der Modell-Säulen. Er ist keine "
    "statistische Wahrscheinlichkeit "
    "für steigende oder fallende Kurse."
)

st.info(
    f"**Aktuelles Marktregime ({selected_asset}):** "
    f"{get_regime_label(float(heute['Final_Regime_Score']))}"
)


# ============================================================
# 14. VOLATILITY
# ============================================================

current_vola = float(
    heute.get(
        "Raw_Volatility",
        20.0,
    )
)

vola_limit = (
    ASSET_CONFIGS[
        selected_asset
    ]["volatility_threshold"]
)

if (
    ASSET_CONFIGS[
        selected_asset
    ]["volatility_mode"]
    == "index"
):
    vola_label = (
        ASSET_CONFIGS[
            selected_asset
        ]["volatility_ticker"]
    )
else:
    vola_label = (
        "20T Realisierte Volatilität"
    )

if current_vola >= vola_limit:

    st.error(
        f"🚨 **VOLATILITÄTS-ALARM:** "
        f"{vola_label} = "
        f"**{current_vola:.2f}%** "
        f"(Grenzwert: "
        f"{vola_limit:.1f}%). "
        f"Positionsgröße reduzieren."
    )

elif current_vola >= vola_limit * 0.8:

    st.warning(
        f"⚠️ **Erhöhte Volatilität:** "
        f"{vola_label} = "
        f"**{current_vola:.2f}%**. "
        f"Entries selektiver wählen."
    )


# ============================================================
# 15. INTRADAY TRADING BIAS
# ============================================================

st.markdown("---")

st.markdown(
    "### 🎯 Intraday Trading Bias"
)

score = float(
    heute[
        "Final_Regime_Score"
    ]
)

mci = float(
    heute["MCI"]
)

if score >= 60:

    bias = (
        "🟢 BULLISCH (Long Bias)"
    )

    rule = (
        f"Bevorzugt nach Long-Setups "
        f"bei {selected_asset} suchen, "
        f"idealerweise an VWAP/EMA/Support."
    )

elif score <= 40:

    bias = (
        "🔴 BÄRISCH (Short Bias)"
    )

    rule = (
        f"Bevorzugt nach Short-Setups "
        f"bei {selected_asset} suchen, "
        f"idealerweise an Resistance."
    )

else:

    bias = (
        "🟡 NEUTRAL / RANGE"
    )

    rule = (
        "Keine klare Trendrichtung. "
        "Nur selektive Setups an "
        "klar definierten charttechnischen "
        "Extrempunkten."
    )


if mci >= 70:

    pos_size = (
        "100% Standardsize"
    )

elif mci >= 50:

    pos_size = (
        "75% Size"
    )

else:

    pos_size = (
        "50% Size – "
        "widersprüchliche Faktoren"
    )


if current_vola >= vola_limit:

    pos_size = (
        "FLAT / Max 25% Size"
    )


focus = (
    "Trend-Follow"
    if abs(score - 50) > 15
    else "Mean-Reversion"
)


b1, b2, b3 = st.columns(3)

with b1:
    st.metric(
        "Handelsrichtung",
        bias,
    )

with b2:
    st.metric(
        "Positionsgröße",
        pos_size,
    )

with b3:
    st.metric(
        "Fokus",
        focus,
    )

st.info(
    f"**Übergeordnete Regel:** "
    f"{rule}"
)


# ============================================================
# 16. GOOGLE TRENDS
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Retail Sentiment (Google Trends)"
)

st.caption(
    "Kontraindikator auf Basis des "
    "Suchverhaltens. Nicht als "
    "eigenständiges Handelssignal verwenden."
)

contra_score, net_spread, trends_live = (
    fetch_google_trends_sentiment(
        selected_asset
    )
)

gt1, gt2, gt3 = st.columns(3)

with gt1:

    st.metric(
        "Google Retail Score",
        (
            f"{contra_score:.1f}"
            f" / 100"
        ),
        (
            f"Net Spread: "
            f"{net_spread:+.2f} σ"
        ),
        delta_color="inverse",
    )

with gt2:

    if contra_score >= 65:

        st.success(
            "🟢 Angst-Überhang: "
            "kontraindikativ potenziell "
            "positiv."
        )

    elif contra_score <= 35:

        st.error(
            "🔴 Gier-Überhang: "
            "mögliche Überhitzung."
        )

    else:

        st.info(
            "🟡 Kein extremes "
            "Sentiment-Signal."
        )

with gt3:

    cfg = TREND_KEYWORD_MAP[
        selected_asset
    ]

    st.markdown(
        f"""
**🔍 Parameter**

* Region: `{cfg['geo']}`
* Euphorie: {", ".join(cfg['bull'])}
* Panik: {", ".join(cfg['bear'])}
* Status: {"🟢 Live" if trends_live else "🔴 Offline/Fallback"}
"""
    )


# ============================================================
# 17. DRIVER ANALYSIS
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Treiber-Analyse (6 Säulen)"
)

saeulen_details = {

    "Makroökonomie": {
        "quelle":
            "FRED API & Yahoo Finance",

        "funktion":
            "Zinsumfeld, Zentralbank-Liquidität, "
            "Dollar-Stärke und bei EUR/USD "
            "das US/EU-Zinsdifferenzial.",
    },

    "Positionierung": {
        "quelle":
            "CFTC COT / CNN Fear & Greed",

        "funktion":
            "CFTC-Positionierung. "
            "CNN Fear & Greed wird nur "
            "für Aktienindizes verwendet.",
    },

    "Marktinterna": {
        "quelle":
            "Yahoo Finance",

        "funktion":
            "Preis-Momentum, Volatilität "
            "und Risikoappetit.",
    },

    "Technischer_Trend": {
        "quelle":
            "Yahoo Finance",

        "funktion":
            "50-/200-Tage-Trend "
            "und RSI-Momentum.",
    },

    "Fundamentale_Faktoren": {
        "quelle":
            "FRED / Modell",

        "funktion":
            "Rohstoff-Lagerbestände bei WTI. "
            "S&P/Nasdaq-Bewertung bleibt "
            "aktuell neutral, weil keine "
            "historische KGV-Zeitreihe "
            "angeschlossen ist.",
    },

    "Fruehwarnindikatoren": {
        "quelle":
            "Yahoo Finance",

        "funktion":
            "Kreditmarkt-Proxy und "
            "Anleihenvolatilität.",
    },
}


cols = st.columns(3)

saeulen = [
    c
    for c in df_dash.columns
    if c.startswith("Saeule_")
]

for i, s_name in enumerate(
    saeulen
):

    val = float(
        heute[s_name]
    )

    raw_name = s_name.replace(
        "Saeule_",
        ""
    )

    label = raw_name.replace(
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
        ]["Saeulen_Gewichte"]
        .get(
            raw_name,
            0,
        )
        * 100
    )

    with cols[
        i % 3
    ]:

        st.metric(
            label=(
                f"{label} {emoji}"
            ),
            value=f"{val:.1f}",
        )

        details = saeulen_details.get(
            raw_name
        )

        if details:

            with st.expander(
                "Details"
            ):

                st.markdown(
                    f"**⚖️ Gewichtung:** "
                    f"{weight:.0f}%"
                )

                st.markdown(
                    f"**⚙️ Funktion:** "
                    f"{details['funktion']}"
                )

                st.markdown(
                    f"**📡 Quelle:** "
                    f"{details['quelle']}"
                )


# ============================================================
# 18. EXECUTION CHECKLIST
# ============================================================

st.markdown("---")

st.subheader(
    "⚡ Intraday Execution "
    "Checkliste & Filter"
)

score_gesamt = float(
    heute[
        "Final_Regime_Score"
    ]
)

trend_wert = float(
    heute.get(
        "Saeule_Technischer_Trend",
        50,
    )
)

fruehwarn_wert = float(
    heute.get(
        "Saeule_Fruehwarnindikatoren",
        50,
    )
)

makro_wert = float(
    heute.get(
        "Saeule_Makroekonomie",
        50,
    )
)

trend_intakt = (
    trend_wert > 55
)

kein_bond_stress = (
    fruehwarn_wert > 35
)

makro_tailwind = (
    makro_wert > 50
)

heute_datum = pd.Timestamp.now(
    tz="Europe/Berlin"
)

wochentag_index = (
    heute_datum.weekday()
)

ist_hexensabbat = (
    heute_datum.month in [
        3,
        6,
        9,
        12,
    ]
    and wochentag_index == 4
    and 15 <= heute_datum.day <= 21
)

opex_default = (
    not ist_hexensabbat
)

wochentag_profile = {

    0:
        "Montag: Preisfindung & "
        "Weekly Initial Balance",

    1:
        "Dienstag: Trendetablierung",

    2:
        "Mittwoch: Trendfortsetzung "
        "oder Mid-Week Reversal",

    3:
        "Donnerstag: Momentum "
        "& Volatilität",

    4:
        "Freitag: Wochenschluss "
        "& Profit-Taking",
}

heutiges_profil = (
    wochentag_profile[
        wochentag_index
    ]
)


c1, c2 = st.columns(2)

with c1:

    st.markdown(
        "#### 1. Strukturelle Filter"
    )

    c1_val = st.checkbox(
        "Trendkonformität "
        "(Marktstruktur / gleitende "
        "Durchschnitte intakt)",
        value=trend_intakt,
        key="chk_trend_det",
    )

    c2_val = st.checkbox(
        "Anleihen- & Kreditmärkte "
        "stabil (kein akuter Stress)",
        value=kein_bond_stress,
        key="chk_bond_det",
    )

    c3_val = st.checkbox(
        f"Makro-Umgebung im Rücken "
        f"(Score: {makro_wert:.0f})",
        value=makro_tailwind,
        key="chk_makro_det",
    )

    c4_val = st.checkbox(
        f"Statistisches Tagesprofil "
        f"beachtet ({heutiges_profil})",
        value=True,
        key="chk_day_profile",
    )

with c2:

    st.markdown(
        "#### 2. Timing & Risikomanagement"
    )

    c5_val = st.checkbox(
        "Keine High-Impact News in "
        "den nächsten 60 Minuten "
        "(MANUELL prüfen)",
        value=True,
        key="chk_news_det",
    )

    c6_val = st.checkbox(
        "Kein Hexensabbat / "
        "Ketten-Verfall",
        value=opex_default,
        key="chk_opex_det",
    )

    c7_val = st.checkbox(
        "CRV mindestens 1:2 zum "
        "nächsten charttechnischen Ziel",
        value=True,
        key="chk_crv_det",
    )

    c8_val = st.checkbox(
        "US-Eröffnung / Initial "
        "Balance abgewartet (MANUELL)",
        value=True,
        key="chk_time_det",
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True,
)

erfuellte_kriterien = sum(
    [
        c1_val,
        c2_val,
        c3_val,
        c4_val,
        c5_val,
        c6_val,
        c7_val,
        c8_val,
    ]
)

st.progress(
    erfuellte_kriterien / 8.0
)

st.caption(
    f"✅ **{erfuellte_kriterien} "
    f"von 8 Kriterien erfüllt**"
)

alle_kriterien_erfuellt = (
    erfuellte_kriterien == 8
)


if (
    alle_kriterien_erfuellt
    and score_gesamt > 55
):

    st.success(
        "🟢 **EXECUTION FREIGABE (GO):** "
        "Alle Filter erfüllt; "
        "Regime bullisch genug für "
        "Long-Bias."
    )

elif (
    alle_kriterien_erfuellt
    and score_gesamt < 45
):

    st.error(
        "🔴 **EXECUTION FREIGABE (SHORT):** "
        "Alle Filter erfüllt; "
        "Regime ausreichend bärisch."
    )

elif score_gesamt < 40:

    st.error(
        "🔴 **STOP / KEIN TRADE:** "
        "Marktregime steht auf Defense."
    )

else:

    st.warning(
        "🟡 **CAUTION / WARNUNG:** "
        "Gemischte Signale. "
        "Selektive Setups bzw. "
        "reduzierte Größe."
    )


# ============================================================
# 19. HISTORICAL CHART
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Regime-Historie & Asset-Preis "
    "(Letzte 12 Monate)"
)

df_plot = (
    df_dash
    .tail(252)
    .copy()
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

        x=df_plot.index,

        y=df_plot[
            "Final_Regime_Score"
        ],

        name="Regime Score",

        fill="tozeroy",

        line=dict(
            width=2
        ),
    ),

    secondary_y=False,
)

fig.add_trace(

    go.Scatter(

        x=df_plot.index,

        y=df_plot[
            "Asset_Price"
        ],

        name=f"{selected_asset} Preis",

        line=dict(
            width=2
        ),
    ),

    secondary_y=True,
)

fig.update_yaxes(
    title_text="Regime Score",
    range=[
        0,
        100,
    ],
    secondary_y=False,
)

fig.update_yaxes(
    title_text="Asset Preis",
    secondary_y=True,
)

fig.update_layout(
    height=400,
    margin=dict(
        l=0,
        r=0,
        t=30,
        b=0,
    ),
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# 20. MODEL WEIGHTS
# ============================================================

st.markdown("---")

with st.expander(
    "⚖️ Aktuelle Modellgewichtungen"
):

    st.markdown(
        f"### {selected_asset}"
    )

    weights = (
        ASSET_CONFIGS[
            selected_asset
        ]["Saeulen_Gewichte"]
    )

    weight_df = pd.DataFrame(
        {
            "Säule": list(
                weights.keys()
            ),
            "Gewichtung": [
                f"{v * 100:.0f}%"
                for v in weights.values()
            ],
        }
    )

    st.dataframe(
        weight_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Die Gewichtungen sind fachlich "
        "begründete Startwerte und nicht "
        "empirisch backtest-optimiert."
    )


# ============================================================
# 21. SYSTEM STATUS
# ============================================================

st.markdown("---")

with st.expander(
    "📡 System & API Status"
):

    status_cols = st.columns(2)

    for i, (
        feed,
        status
    ) in enumerate(
        feed_status.items()
    ):

        icon = (
            "✅ Verbunden"
            if status
            else
            "⚠️ Fallback / Offline"
        )

        status_cols[
            i % 2
        ].markdown(
            f"**{feed}:** {icon}"
        )

    st.caption(
        "Fallback-Werte verhindern einen "
        "Absturz einzelner Datenquellen. "
        "Sie sind ausdrücklich nicht als "
        "Live-Daten zu interpretieren."
    )


# ============================================================
# 22. MODEL DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Modellhinweis: Der Final Regime Score "
    "ist ein quantitatives Regimefilter-Modell "
    "und keine statistische Wahrscheinlichkeit. "
    "Insbesondere COT, Google Trends, "
    "Kreditproxy und technische Indikatoren "
    "sind keine kausalen Prognosen. "
    "Die Gewichtungen sind fachlich begründete "
    "Startwerte und müssen historisch "
    "backgetestet und validiert werden."
)
