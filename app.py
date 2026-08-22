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

st.set_page_config(page_title="Multi-Asset Regime Dashboard", page_icon="📊", layout="centered")

# ============================================================
# 1. ASSET CONFIGURATIONS & WEIGHTS
# ============================================================
# EUR/USD is deliberately included as a selectable Multi-Asset.
ASSET_CONFIGS = {
    "S&P 500": {
        "ticker":"^GSPC", "volatility_ticker":"^VIX", "cot_code":"E-MINI S&P 500",
        "options_proxy":"SPY", "futures_pc_ticker":"ES=F", "options_pc_ticker":"SPY",
        "invert_inverts":["vix_score","pe_valuation","credit_spreads","move_index","usd_index","fed_policy","real_yields"],
        "Saeulen_Gewichte":{"Makroökonomie":.25,"Positionierung":.15,"Marktinterna":.20,"Technischer_Trend":.20,"Fundamentale_Faktoren":.10,"Fruehwarnindikatoren":.10},
        "Sub_Gewichte":{"Positionierung":{"cot_commercials":.50,"fear_greed":.50,"futures_put_call":.0,"options_put_call":.0},"Marktinterna":{"market_momentum":.50,"vix_score":.50},"Fundamentale_Faktoren":{"pe_valuation":1.0}}
    },
    "Nasdaq 100": {
        "ticker":"NQ=F", "volatility_ticker":"^VXN", "cot_code":"NASDAQ-100",
        "options_proxy":"QQQ", "futures_pc_ticker":"NQ=F", "options_pc_ticker":"QQQ",
        "invert_inverts":["vix_score","pe_valuation","credit_spreads","move_index","usd_index","fed_policy","real_yields"],
        "Saeulen_Gewichte":{"Makroökonomie":.25,"Positionierung":.15,"Marktinterna":.20,"Technischer_Trend":.20,"Fundamentale_Faktoren":.10,"Fruehwarnindikatoren":.10},
        "Sub_Gewichte":{"Positionierung":{"cot_commercials":.50,"fear_greed":.50,"futures_put_call":.0,"options_put_call":.0},"Marktinterna":{"market_momentum":.50,"vix_score":.50},"Fundamentale_Faktoren":{"pe_valuation":1.0}}
    },
    "Gold (XAU/USD)": {
        "ticker":"GC=F", "volatility_ticker":"^GVZ", "cot_code":"GOLD",
        "options_proxy":"GLD", "futures_pc_ticker":"GC=F", "options_pc_ticker":"GLD",
        "invert_inverts":["vix_score","usd_index","real_yields","fed_policy"],
        "Saeulen_Gewichte":{"Makroökonomie":.35,"Positionierung":.25,"Marktinterna":.15,"Technischer_Trend":.15,"Fundamentale_Faktoren":0.0,"Fruehwarnindikatoren":.10},
        "Sub_Gewichte":{"Positionierung":{"cot_commercials":.80,"fear_greed":.20,"futures_put_call":.0,"options_put_call":.0},"Marktinterna":{"obv_momentum":.50,"vix_score":.50},"Fundamentale_Faktoren":{}}
    },
    "WTI Crude Oil": {
        "ticker":"CL=F", "volatility_ticker":"^OVX", "cot_code":"CRUDE OIL",
        "options_proxy":"USO", "futures_pc_ticker":"CL=F", "options_pc_ticker":"USO",
        "invert_inverts":["vix_score","usd_index","inventories"],
        "Saeulen_Gewichte":{"Makroökonomie":.30,"Positionierung":.25,"Marktinterna":.15,"Technischer_Trend":.20,"Fundamentale_Faktoren":.10,"Fruehwarnindikatoren":0.0},
        "Sub_Gewichte":{"Positionierung":{"cot_commercials":.80,"fear_greed":.20,"futures_put_call":.0,"options_put_call":.0},"Marktinterna":{"obv_momentum":.50,"vix_score":.50},"Fundamentale_Faktoren":{"inventories":1.0}}
    },
    "EUR/USD": {
        "ticker":"EURUSD=X", "volatility_ticker":"^EVZ", "cot_code":"EURO FX",
        "options_proxy":"FXE", "futures_pc_ticker":"6E=F", "options_pc_ticker":"FXE",
        "invert_inverts":["vix_score","fed_policy","real_yields","usd_index"],
        "Saeulen_Gewichte":{"Makroökonomie":.35,"Positionierung":.20,"Marktinterna":.15,"Technischer_Trend":.20,"Fundamentale_Faktoren":0.0,"Fruehwarnindikatoren":.10},
        "Sub_Gewichte":{"Positionierung":{"cot_commercials":.70,"fear_greed":.30,"futures_put_call":.0,"options_put_call":.0},"Marktinterna":{"market_momentum":.50,"vix_score":.50},"Fundamentale_Faktoren":{}}
    }
}

SUB_WEIGHTS_BASE = {
    "Makroökonomie":{"fed_policy":.20,"real_yields":.30,"usd_index":.20,"net_liquidity":.30},
    "Technischer_Trend":{"distance_200ma":.35,"distance_50ma":.35,"rsi_momentum":.30},
    "Fruehwarnindikatoren":{"credit_spreads":.60,"move_index":.40}
}
LOOKBACK_CONFIG={"fed_policy":1260,"real_yields":756,"net_liquidity":756,"credit_spreads":756,"usd_index":504,"inventories":756}
VOLA_THRESHOLDS={"S&P 500":30.0,"Nasdaq 100":35.0,"Gold (XAU/USD)":25.0,"WTI Crude Oil":45.0,"EUR/USD":15.0}
TREND_KEYWORD_MAP={
 "S&P 500":{"geo":"US","lang":"en-US","bull":["buy stocks","buy the dip"],"bear":["stock market crash","recession"]},
 "Nasdaq 100":{"geo":"US","lang":"en-US","bull":["tech stocks","buy the dip"],"bear":["market crash","tech bubble"]},
 "Gold (XAU/USD)":{"geo":"DE","lang":"de-DE","bull":["Gold kaufen","Goldmünzen"],"bear":["Gold verkaufen","Altgold"]},
 "WTI Crude Oil":{"geo":"DE","lang":"de-DE","bull":["Heizöl kaufen","Spritpreise"],"bear":["Ölpreis crash","Öl verkaufen"]},
 "EUR/USD":{"geo":"DE","lang":"de-DE","bull":["Euro kaufen","EUR USD kaufen"],"bear":["Euro verkaufen","EUR USD verkaufen"]}
}

@st.cache_data(ttl=21600)
def fetch_google_trends_sentiment(asset_name):
    cfg=TREND_KEYWORD_MAP.get(asset_name,TREND_KEYWORD_MAP["S&P 500"])
    try:
        p=TrendReq(hl=cfg["lang"],tz=360,retries=2,backoff_factor=.2)
        kws=cfg["bull"]+cfg["bear"]; p.build_payload(kws,timeframe="today 3-m",geo=cfg["geo"])
        d=p.interest_over_time()
        if d.empty:return 50.,0.,False
        if "isPartial" in d:d=d.drop(columns="isPartial")
        def z(s):
            s=pd.to_numeric(s,errors="coerce"); m=s.rolling(21,min_periods=5).mean(); sd=s.rolling(21,min_periods=5).std().replace(0,np.nan); return (s-m)/sd
        vb=[x for x in cfg["bull"] if x in d]; vr=[x for x in cfg["bear"] if x in d]
        if not vb or not vr:return 50.,0.,False
        spread=sum(z(d[x]) for x in vb)/len(vb)-sum(z(d[x]) for x in vr)/len(vr); spread=spread.replace([np.inf,-np.inf],np.nan).dropna()
        if spread.empty:return 50.,0.,False
        latest=float(spread.iloc[-1]); score=float(np.clip(50-latest*15,0,100)); return round(score,1),round(latest,2),True
    except Exception:return 50.,0.,False

def strip_timezone(x):
    dt=pd.to_datetime(x,errors="coerce")
    if isinstance(dt,pd.Series): return dt.dt.tz_convert(None) if getattr(dt.dt,"tz",None) is not None else dt
    if isinstance(dt,pd.DatetimeIndex): return dt.tz_convert(None) if dt.tz is not None else dt
    return dt

def normalize_to_percentile(series,lookback=252,invert=False):
    s=pd.to_numeric(series,errors="coerce").replace([np.inf,-np.inf],np.nan).ffill().bfill()
    if s.isna().all():return pd.Series(50.,index=series.index)
    m=s.rolling(lookback,min_periods=20).mean(); sd=s.rolling(lookback,min_periods=20).std().replace(0,np.nan); z=(s-m)/sd; out=pd.Series(norm.cdf(z)*100,index=series.index)
    if invert:out=100-out
    return out.replace([np.inf,-np.inf],np.nan).clip(0,100).ffill().bfill().fillna(50.)

def calculate_mci(scores,weights):
    s=np.asarray(scores,float); w=np.asarray(weights,float); v=np.isfinite(s)&np.isfinite(w); s=s[v];w=w[v]
    if len(s)==0 or w.sum()<=0:return 0.
    w=w/w.sum(); mean=np.average(s,weights=w); sd=np.sqrt(np.average((s-mean)**2,weights=w)); return round(float(np.clip(100*(1-sd/50),0,100)),1)

def get_regime_label(score):
    if score>=90:return "🟢 Risk-On (Extrem Bullisch)"
    if score>=75:return "🟢 Expansion (Bullisch)"
    if score>=60:return "🟡 Übergangsphase (Leicht Bullisch)"
    if score>=40:return "🟡 Neutral"
    if score>=25:return "🟠 Risk-Off (Bärisch)"
    return "🔴 Stressphase (Stark Bärisch)"

def safe_reindex_series(source,target):
    if not isinstance(source,pd.Series) or source.empty:return None
    s=source.copy(); s.index=strip_timezone(s.index).floor("D"); s=s[~s.index.duplicated(keep="last")].sort_index(); t=strip_timezone(target).floor("D"); r=s.reindex(t,method="ffill").ffill().bfill();r.index=target;return r

def extract_yfinance_field(data,field):
    if data is None:return None
    if not isinstance(data.columns,pd.MultiIndex):return data[field] if field in data.columns and isinstance(data[field],pd.Series) else None
    if field in data.columns.get_level_values(0):
        x=data[field]; return x if isinstance(x,pd.DataFrame) else None
    if field in data.columns.get_level_values(1):
        x=data.xs(field,axis=1,level=1); return x if isinstance(x,pd.DataFrame) else None
    return None

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():
    try:
        r=requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",headers={"User-Agent":"Mozilla/5.0","Accept":"application/json","Referer":"https://edition.cnn.com/"},timeout=8)
        if r.status_code!=200:return 55.,False
        h=r.json().get("fear_and_greed_historical",{}).get("data",[]); d=pd.DataFrame(h)
        if d.empty or not {"x","y"}.issubset(d.columns):return 55.,False
        d["Date"]=strip_timezone(pd.to_datetime(d.x,unit="ms",errors="coerce")).dt.floor("D");d.y=pd.to_numeric(d.y,errors="coerce");d=d.dropna(subset=["Date","y"]).drop_duplicates("Date",keep="last")
        return d.set_index("Date").y.sort_index(),True
    except Exception:return 55.,False

# CFTC: Futures-only COT is kept separately from Futures-and-Options Combined.
# This is intentionally NOT called a put/call ratio: COT contains long/short positions,
# not option puts/calls. The dashboard's "Futures P/C" is therefore based on a public
# market option chain proxy; when unavailable it remains neutral rather than fabricated.
@st.cache_data(ttl=86400)
def fetch_cot_data(asset_search_string,combined=False):
    headers={"User-Agent":"Mozilla/5.0"}; year=pd.Timestamp.now().year; frames=[]
    for yr in [year-1,year]:
        url=f"https://www.cftc.gov/files/dea/history/fut_com_txt_{yr}.zip"
        try:
            r=requests.get(url,headers=headers,timeout=10)
            if r.status_code!=200:continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                n=z.namelist();
                if not n:continue
                d=pd.read_csv(z.open(n[0]),low_memory=False); mc=d.columns[0]; rows=d[d[mc].astype(str).str.contains(asset_search_string,case=False,na=False)]
                if not rows.empty:frames.append(rows)
        except Exception:continue
    if not frames:return None,False
    try:
        d=pd.concat(frames,ignore_index=True); dc=[c for c in d if "As_of_Date" in str(c)]; lc=[c for c in d if "Comm_Positions_Long_All" in str(c)];sc=[c for c in d if "Comm_Positions_Short_All" in str(c)]
        if not dc or not lc or not sc:return None,False
        d["Date"]=strip_timezone(pd.to_datetime(d[dc[0]].astype(str),format="%Y%m%d",errors="coerce")).dt.floor("D");d["Net_Commercials"]=pd.to_numeric(d[lc[0]],errors="coerce")-pd.to_numeric(d[sc[0]],errors="coerce");d=d.dropna(subset=["Date"]).drop_duplicates("Date",keep="last")
        return d.set_index("Date").Net_Commercials.sort_index(),True
    except Exception:return None,False

@st.cache_data(ttl=1800)
def fetch_option_put_call(ticker):
    """Aggregated option-chain put/call ratio using put volume / call volume.
    Returns ratio, live flag, source text. yfinance generally provides equity/ETF chains;
    futures option chains are not consistently exposed, so the configured ETF proxy is used."""
    try:
        t=yf.Ticker(ticker); expiries=t.options
        if not expiries:return np.nan,False,"No option chain"
        expiry=expiries[0]; chain=t.option_chain(expiry)
        puts=chain.puts; calls=chain.calls
        pv=pd.to_numeric(puts.get("volume"),errors="coerce").fillna(0).sum() if not puts.empty else 0
        cv=pd.to_numeric(calls.get("volume"),errors="coerce").fillna(0).sum() if not calls.empty else 0
        if cv<=0:return np.nan,False,"No call volume"
        return float(pv/cv),True,f"{ticker} Optionen ({expiry})"
    except Exception:return np.nan,False,"Option chain unavailable"

@st.cache_data(ttl=1800)
def fetch_futures_put_call(proxy_ticker):
    # Futures themselves have no put/call ratio. We use the closest liquid options proxy
    # configured per asset and label it explicitly as a futures-market proxy.
    return fetch_option_put_call(proxy_ticker)

FRED_API_KEY=""
try:
    if "FRED_API_KEY" in st.secrets:FRED_API_KEY=st.secrets["FRED_API_KEY"]
except Exception:pass

with st.sidebar:
    st.title("⚙️ Multi-Asset Selector")
    selected_asset=st.selectbox("🎯 Asset auswählen",list(ASSET_CONFIGS),index=0)
    st.markdown("---");st.markdown("### 📡 API Live-Feed Monitor")

@st.cache_data(ttl=3600)
def fetch_multi_asset_data(selected_asset):
    cfg=ASSET_CONFIGS[selected_asset]; status={}
    tickers={"asset":cfg["ticker"],"vix":cfg["volatility_ticker"],"dxy":"DX=F","move":"^MOVE","hyg":"HYG","lqd":"LQD"}
    try:data=yf.download(list(tickers.values()),period="5y",interval="1d",auto_adjust=False,progress=False,threads=True)
    except Exception:return pd.DataFrame(),{"yFinance (Preis & Tech)":False}
    if data.empty:return pd.DataFrame(),{"yFinance (Preis & Tech)":False}
    close=extract_yfinance_field(data,"Close")
    if close is None:return pd.DataFrame(),{"yFinance (Preis & Tech)":False}
    if isinstance(close,pd.Series):close=close.to_frame()
    if isinstance(close.columns,pd.MultiIndex):close.columns=close.columns.get_level_values(-1)
    close=close.rename(columns={v:k for k,v in tickers.items()}).apply(pd.to_numeric,errors="coerce").sort_index();status["yFinance (Preis & Tech)"]=bool("asset" in close and not close.asset.dropna().empty)
    if not status["yFinance (Preis & Tech)"]:return pd.DataFrame(),status
    price=close.asset.dropna(); df=pd.DataFrame(index=price.index)
    vol=extract_yfinance_field(data,"Volume"); hasvol=False
    if vol is not None:
        if isinstance(vol,pd.Series):asset_vol=vol.reindex(price.index).ffill().bfill();hasvol=True
        else:
            if isinstance(vol.columns,pd.MultiIndex):vol.columns=vol.columns.get_level_values(-1)
            vol=vol.rename(columns={v:k for k,v in tickers.items()})
            if "asset" in vol:asset_vol=vol.asset.reindex(price.index).ffill().bfill();hasvol=True
    if not hasvol:asset_vol=pd.Series(1000.,index=price.index)
    status["Volumen / Orderflow Feed"]=hasvol
    ma50=price.rolling(50,min_periods=50).mean();ma200=price.rolling(200,min_periods=200).mean();df["distance_50ma"]=(price-ma50)/ma50.replace(0,np.nan)*100;df["distance_200ma"]=(price-ma200)/ma200.replace(0,np.nan)*100
    delta=price.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();loss=-delta.clip(upper=0).ewm(alpha=1/14,adjust=False).mean();df["rsi_momentum"]=(100-100/(1+gain/loss.replace(0,np.nan))).clip(0,100)
    df["vix_score"]=close.vix.reindex(df.index).ffill().bfill() if "vix" in close else 20.;df["usd_index"]=close.dxy.reindex(df.index).ffill().bfill() if "dxy" in close else 100.;df["move_index"]=close.move.reindex(df.index).ffill().bfill() if "move" in close else 100.
    if "lqd" in close and "hyg" in close:df["credit_spreads"]=close.lqd.reindex(df.index).ffill().bfill()/close.hyg.reindex(df.index).ffill().bfill().replace(0,np.nan)
    else:df["credit_spreads"]=1.
    df["market_momentum"]=price.pct_change().rolling(20,min_periods=10).sum()*100
    obv=pd.Series(np.where(delta>0,asset_vol,np.where(delta<0,-asset_vol,0.)),index=price.index).cumsum();ema=obv.ewm(span=50,adjust=False).mean();df["obv_momentum"]=(obv-ema)/ema.abs().replace(0,np.nan)*100
    if selected_asset=="S&P 500":df["pe_valuation"]=24.5
    fred_ok=False
    if FRED_API_KEY:
        try:
            f=Fred(api_key=FRED_API_KEY);wal=safe_reindex_series(f.get_series("WALCL"),df.index);tga=safe_reindex_series(f.get_series("WTREGEN"),df.index);rrp=safe_reindex_series(f.get_series("RRPONTSYD"),df.index)
            df["net_liquidity"]=((wal-tga-rrp*1000)/1000) if wal is not None and tga is not None and rrp is not None else np.nan
            df["fed_policy"]=safe_reindex_series(f.get_series("FEDFUNDS"),df.index);df["real_yields"]=safe_reindex_series(f.get_series("DFII10"),df.index)
            if selected_asset=="WTI Crude Oil":df["inventories"]=safe_reindex_series(f.get_series("WCESTUS1"),df.index)
            fred_ok=True
        except Exception:pass
    status["FRED API (Makro & Fed)"]=fred_ok
    if not fred_ok:
        for c in ["fed_policy","real_yields","net_liquidity"]:df[c]=np.nan
        if selected_asset=="WTI Crude Oil":df["inventories"]=np.nan
    cot,cot_ok=fetch_cot_data(cfg["cot_code"]);status[f"CFTC COT ({cfg['cot_code']})"]=cot_ok;df["cot_commercials"]=safe_reindex_series(cot,df.index) if cot is not None else np.nan
    fg,fg_ok=fetch_fear_and_greed();status["CNN Fear & Greed"]=fg_ok;df["fear_greed"]=safe_reindex_series(fg,df.index) if isinstance(fg,pd.Series) else float(fg)
    # Put/call feeds. These are displayed and normalized, but are NOT silently injected into
    # the pillar weights: default weight is 0.0 so adding the feed does not alter the model.
    opt_pc,opt_ok,opt_source=fetch_option_put_call(cfg["options_pc_ticker"]); fut_pc,fut_ok,fut_source=fetch_futures_put_call(cfg["options_pc_ticker"])
    df["options_put_call"]=opt_pc if np.isfinite(opt_pc) else np.nan;df["futures_put_call"]=fut_pc if np.isfinite(fut_pc) else np.nan
    status[f"Options Put/Call ({cfg['options_pc_ticker']})"]=opt_ok;status[f"Futures Put/Call Proxy ({cfg['futures_pc_ticker']})"]=fut_ok
    # same latest ratio across historical index; normalization therefore becomes neutral until
    # a real time series is available, avoiding fake historical data.
    norm=pd.DataFrame(index=df.index);inverts=cfg["invert_inverts"]
    for col in df:
        norm[col]=normalize_to_percentile(df[col],LOOKBACK_CONFIG.get(col,252),col in inverts)
    dash=pd.DataFrame(index=df.index);dash["Raw_Volatility"]=df.vix_score
    active={k:v.copy() for k,v in SUB_WEIGHTS_BASE.items()}
    for cat,w in cfg["Sub_Gewichte"].items():active[cat]=w
    for pillar,inds in active.items():
        cols=[c for c in inds if c in norm.columns]; ws=np.array([inds[c] for c in cols],float)
        dash[f"Saeule_{pillar}"]=norm[cols].dot(ws/ws.sum()) if cols and ws.sum()>0 else 50.
    sc=[];ws=[]
    for pillar,w in cfg["Saeulen_Gewichte"].items():
        c=f"Saeule_{pillar}"
        if c in dash:sc.append(c);ws.append(w)
    w=np.array(ws,float);w=w/w.sum();dash["Final_Regime_Score"]=dash[sc].dot(w).clip(0,100).round(1);dash["MCI"]=[calculate_mci(dash[sc].iloc[i].values,w) for i in range(len(dash))];dash["Asset_Price"]=price.reindex(dash.index).ffill().bfill()
    # Current P/C ratios are stored as separate dashboard fields for display.
    dash["Options_Put_Call"]=opt_pc;dash["Futures_Put_Call_Proxy"]=fut_pc
    return dash.dropna(subset=["Final_Regime_Score"]),status

with st.spinner(f"Lade quantitative Daten für {selected_asset}..."):
    df_dash,feed_status=fetch_multi_asset_data(selected_asset)
with st.sidebar:
    for source,live in feed_status.items():st.markdown(f"{'🟢' if live else '⚠️'} **{source}**{' *(Fallback / Offline)*' if not live else ''}")
if df_dash.empty:st.error("⚠️ Marktdaten konnten nicht geladen werden.");st.stop()
heute=df_dash.iloc[-1].copy();heute["Delta_1D"]=df_dash.Final_Regime_Score.iloc[-1]-df_dash.Final_Regime_Score.iloc[-2] if len(df_dash)>=2 else 0.;heute["Delta_1W"]=df_dash.MCI.iloc[-1]-df_dash.MCI.iloc[-6] if len(df_dash)>=6 else 0.

st.title("📊 Quant Regime Dashboard");st.caption(f"Asset: **{selected_asset}** | Stand: {df_dash.index[-1].strftime('%d.%m.%Y')}");st.markdown("---")
c1,c2=st.columns(2)
with c1:st.metric("Final Regime Score",f"{heute.Final_Regime_Score} / 100",f"{heute.Delta_1D:+.1f} (Heute)")
with c2:st.metric("Model Consistency Index",f"{heute.MCI}%",f"{heute.Delta_1W:+.1f} (vs. Vorwoche)",delta_color="off")
st.caption("Der Model Consistency Index misst die Übereinstimmung der sechs Modell-Säulen. Er ist keine statistische Wahrscheinlichkeit.")
st.info(f"**Aktuelles Marktregime ({selected_asset}):** {get_regime_label(heute.Final_Regime_Score)}")

# P/C panel: explicit distinction between real options-chain ratio and the futures-market proxy.
st.markdown("---");st.subheader("📊 Put/Call Ratio – Positionierungsfilter")
pc1,pc2,pc3=st.columns(3)
opt_pc=float(heute.get("Options_Put_Call",np.nan));fut_pc=float(heute.get("Futures_Put_Call_Proxy",np.nan))
with pc1:st.metric("Optionen Put/Call",f"{opt_pc:.2f}" if np.isfinite(opt_pc) else "n/a")
with pc2:st.metric("Futures Put/Call*",f"{fut_pc:.2f}" if np.isfinite(fut_pc) else "n/a")
with pc3:st.metric("Interpretation",("eher Put-lastig" if np.isfinite(opt_pc) and opt_pc>1 else "eher Call-lastig" if np.isfinite(opt_pc) else "keine Daten"))
st.caption("*Wichtig: Futures selbst haben keine Put/Call-Ratio. Der hier angezeigte Futures-Wert ist deshalb ausdrücklich ein Options-Proxy zum jeweiligen Futures-Markt (z. B. ES→SPY, NQ→QQQ, Gold→GLD, WTI→USO, EUR/USD→FXE). Die CFTC-COT-Daten sind dagegen echte Futures bzw. Futures-and-Options-Combined-Positionsdaten und keine Put/Call-Ratio. Die CFTC stellt Futures Only und Futures-and-Options Combined getrennt bereit. citeturn0search1turn0search11")

current_vola=float(heute.get("Raw_Volatility",20));limit=VOLA_THRESHOLDS.get(selected_asset,30);vt=ASSET_CONFIGS[selected_asset]["volatility_ticker"]
if current_vola>=limit:st.error(f"🚨 **VOLATILITÄTS-ALARM:** {vt} bei **{current_vola:.2f}** (Grenzwert {limit:.1f}).")
elif current_vola>=limit*.8:st.warning(f"⚠️ **Erhöhte Volatilität:** {vt} bei **{current_vola:.2f}**.")

st.markdown("---");st.markdown("### 🎯 Intraday Trading Bias");score=float(heute.Final_Regime_Score);mci=float(heute.MCI)
if score>=60:bias="🟢 BULLISCH (Long Bias)";rule=f"Bevorzugt Long-Setups bei {selected_asset} suchen.";pos="100% Standardsize" if mci>=70 else "75% Size" if mci>=50 else "50% Size"
elif score<=40:bias="🔴 BÄRISCH (Short Bias)";rule=f"Bevorzugt Short-Setups bei {selected_asset} suchen.";pos="100% Standardsize" if mci>=70 else "75% Size" if mci>=50 else "50% Size"
else:bias="🟡 NEUTRAL / RANGE";rule="Keine klare Trendrichtung. Nur selektive Setups.";pos="50% Size"
if current_vola>=limit:pos="FLAT / Max 25% Size"
b1,b2,b3=st.columns(3)
with b1:st.metric("Handelsrichtung",bias)
with b2:st.metric("Positionsgröße",pos)
with b3:st.metric("Fokus","Trend-Follow" if abs(score-50)>15 else "Mean-Reversion")
st.info(f"**Übergeordnete Regel:** {rule}")

st.markdown("---");st.subheader("🌐 Retail Sentiment (Google Trends)");st.caption("Unabhängiger Kontraindikator auf Basis des Suchverhaltens von Privatanlegern.")
contra,spread,trends_live=fetch_google_trends_sentiment(selected_asset);g1,g2,g3=st.columns(3)
with g1:st.metric("Google Retail Score (0-100)",f"{contra} / 100",f"Net Spread: {spread:+.2f} σ",delta_color="inverse")
with g2:
    if contra>=65:st.success("🟢 Panik-Ausschlag: kontraindikativ potenziell positiv.")
    elif contra<=35:st.error("🔴 Gier-Ausschlag: mögliche Überhitzung.")
    else:st.info("🟡 Ausgeglichenes Sentiment.")
with g3:
    cfg=TREND_KEYWORD_MAP[selected_asset];st.markdown(f"**🔍 Getrackte Parameter:**\n\n* **Region:** `{cfg['geo']}`\n* **Euphorie:** {', '.join(repr(x) for x in cfg['bull'])}\n* **Panik:** {', '.join(repr(x) for x in cfg['bear'])}\n* **Status:** {'🟢 Live' if trends_live else '🔴 Offline/Fallback'}")

# ============================================================
# SIX PILLARS – DETAILS & SOURCE LINKS
# ============================================================
saeulen_details={
 "Makroökonomie":{"quelle":"FRED API & Yahoo Finance","funktion":"Zinsumfeld, Zentralbank-Liquidität und Dollar-Stärke.","links":[("FRED: Fed Total Assets (WALCL)","https://fred.stlouisfed.org/series/WALCL"),("FRED: TGA Account (WTREGEN)","https://fred.stlouisfed.org/series/WTREGEN"),("FRED: Reverse Repo (RRPONTSYD)","https://fred.stlouisfed.org/series/RRPONTSYD"),("FRED: 10Y Real Yields (DFII10)","https://fred.stlouisfed.org/series/DFII10"),("FRED: Fed Funds Rate (FEDFUNDS)","https://fred.stlouisfed.org/series/FEDFUNDS"),("Yahoo: US Dollar Index","https://finance.yahoo.com/quote/DX-Y.NYB")]},
 "Positionierung":{"quelle":"CFTC COT, CNN Fear & Greed & Put/Call-Daten","funktion":"Institutionelle Positionierung, Sentiment und Optionspositionierung.","links":[("CFTC: Commitment of Traders","https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"),("CFTC: Futures Only / Combined","https://publicreporting.cftc.gov/stories/s/r4w3-av2u"),("CNN: Fear & Greed Index","https://edition.cnn.com/markets/fear-and-greed"),("Yahoo Finance Options","https://finance.yahoo.com/")]},
 "Marktinterna":{"quelle":"Yahoo Finance","funktion":"Preis-Momentum, Marktvolatilität und Risikoappetit.","links":[("Yahoo: VIX","https://finance.yahoo.com/quote/%5EVIX"),("Yahoo: VXN","https://finance.yahoo.com/quote/%5EVXN"),("Yahoo: GVZ","https://finance.yahoo.com/quote/%5EGVZ"),("Yahoo: OVX","https://finance.yahoo.com/quote/%5EOVX"),("Yahoo: EVZ","https://finance.yahoo.com/quote/%5EEVZ")]},
 "Technischer_Trend":{"quelle":"Yahoo Finance","funktion":"200-Tage-Trend, 50-Tage-Trend und RSI-Momentum.","links":[("Yahoo: Chart & Technicals",f"https://finance.yahoo.com/quote/{ASSET_CONFIGS[selected_asset]['ticker']}")]},
 "Fundamentale_Faktoren":{"quelle":"FRED, Multpl & WSJ","funktion":"Bewertung bzw. Rohstoff-Lagerbestände.","links":[("FRED: Crude Oil Stocks","https://fred.stlouisfed.org/series/WCESTUS1"),("Multpl: S&P 500 PE Ratio","https://www.multpl.com/s-p-500-pe-ratio"),("WSJ: P/E Ratios","https://www.wsj.com/market-data/stocks/peyields")]},
 "Fruehwarnindikatoren":{"quelle":"Yahoo Finance","funktion":"Kreditmarkt-Proxy und Anleihenvolatilität.","links":[("Yahoo: HYG High Yield ETF","https://finance.yahoo.com/quote/HYG"),("Yahoo: LQD Investment Grade ETF","https://finance.yahoo.com/quote/LQD"),("Yahoo: MOVE Index","https://finance.yahoo.com/quote/%5EMOVE")]}
}

st.markdown("---");st.subheader("🔍 Treiber-Analyse (Die 6 Säulen)");cols=st.columns(3);saeulen=[c for c in df_dash.columns if c.startswith("Saeule_")]
for i,s in enumerate(saeulen):
    val=float(heute.get(s,50));raw=s.replace("Saeule_","");label=raw.replace("_"," ");emoji="🟢" if val>60 else "🔴" if val<40 else "🟡";weight=ASSET_CONFIGS[selected_asset]["Saeulen_Gewichte"].get(raw,0)*100
    with cols[i%3]:
        st.metric(f"{label} {emoji}",f"{val:.1f}")
        if raw in saeulen_details:
            d=saeulen_details[raw]
            with st.expander("Details, Daten & Links"):
                st.markdown(f"**⚖️ Gewichtung:** {weight:.0f}%");st.markdown(f"**📡 Quelle:** {d['quelle']}");st.markdown(f"**⚙️ Funktion:** {d['funktion']}");st.markdown("**🔗 Live-Datenquellen:**")
                for title,url in d["links"]:st.markdown(f"• [{title}]({url})")
        st.markdown("<br>",unsafe_allow_html=True)

st.markdown("---");st.subheader("⚡ Intraday Execution Checkliste & Filter")
trend=float(heute.get("Saeule_Technischer_Trend",50));early=float(heute.get("Saeule_Fruehwarnindikatoren",50));macro=float(heute.get("Saeule_Makroökonomie",50));trend_ok=trend>55;bond_ok=early>35;macro_ok=macro>50
now=pd.Timestamp.now(tz="Europe/Berlin");wd=now.weekday();hexensabbat=now.month in [3,6,9,12] and wd==4 and 15<=now.day<=21;profile={0:"Montag: Preisfindung & Weekly Initial Balance",1:"Dienstag: Trendetablierung",2:"Mittwoch: Trendfortsetzung oder Mid-Week Reversal",3:"Donnerstag: Momentum & Volatilität",4:"Freitag: Wochenschluss & Profit-Taking"}.get(wd,"Wochenende: Märkte geschlossen")
a,b=st.columns(2)
with a:
    st.markdown("#### 1. Strukturelle Filter");x1=st.checkbox("Trendkonformität (Marktstruktur / gleitende Durchschnitte intakt)",value=trend_ok,key="chk_trend_det");x2=st.checkbox("Anleihen- & Kreditmärkte stabil (kein akuter Stress)",value=bond_ok,key="chk_bond_det");x3=st.checkbox(f"Makro-Umgebung im Rücken (Score: {macro:.0f})",value=macro_ok,key="chk_makro_det");x4=st.checkbox(f"Statistisches Tagesprofil beachtet ({profile})",value=True,key="chk_day_profile")
with b:
    st.markdown("#### 2. Timing & Risikomanagement");x5=st.checkbox("Keine High-Impact News (CPI, FOMC, NFP) in den nächsten 60 Minuten",value=True,key="chk_news_det");x6=st.checkbox("Kein Hexensabbat / Ketten-Verfall",value=not hexensabbat,key="chk_opex_det");x7=st.checkbox("CRV mindestens 1:2 zum nächsten charttechnischen Ziel",value=True,key="chk_crv_det");x8=st.checkbox("US-Eröffnung / Initial Balance abgewartet",value=True,key="chk_time_det")
count=sum([x1,x2,x3,x4,x5,x6,x7,x8]);st.progress(count/8);st.caption(f"✅ **{count} von 8 Kriterien erfüllt**")
if count==8 and score>55:st.success("🟢 **EXECUTION FREIGABE (GO):** Alle Filter erfüllt und bullischer Long-Bias.")
elif count==8 and score<45:st.error("🔴 **EXECUTION FREIGABE (SHORT):** Alle Filter erfüllt und ausreichend bärisch.")
elif score<40:st.error("🔴 **STOP / KEIN TRADE:** Marktregime auf Defense.")
else:st.warning("🟡 **CAUTION / WARNUNG:** Gemischte Signale.")

st.markdown("---");st.subheader("📈 Regime-Historie & Asset Preis (Letzte 12 Monate)");plot=df_dash.tail(252);fig=make_subplots(specs=[[{"secondary_y":True}]]);fig.add_trace(go.Scatter(x=plot.index,y=plot.Final_Regime_Score,name="Regime Score (0-100)",fill="tozeroy"),secondary_y=False);fig.add_trace(go.Scatter(x=plot.index,y=plot.Asset_Price,name=f"{selected_asset} Preis",line=dict(width=2)),secondary_y=True);fig.update_yaxes(title_text="Regime Score",range=[0,100],secondary_y=False);fig.update_yaxes(title_text="Asset Preis",secondary_y=True);fig.update_layout(height=400,margin=dict(l=0,r=0,t=30,b=0),hovermode="x unified",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)");st.plotly_chart(fig,use_container_width=True)

st.markdown("---")
with st.expander("⚖️ Aktuelle Modellgewichtungen"):
    weights=ASSET_CONFIGS[selected_asset]["Saeulen_Gewichte"];st.dataframe(pd.DataFrame({"Säule":list(weights),"Gewichtung":[f"{v*100:.0f}%" for v in weights.values()]}),hide_index=True,use_container_width=True);st.caption("Die Gewichtungen sind fachlich begründete Startgewichte und nicht empirisch backtest-optimiert.")
with st.expander("📡 System & API Status Details"):
    st.write("Live-Verbindungsstatus zu den externen Datenquellen:");sc=st.columns(2)
    for i,(feed,status) in enumerate(feed_status.items()):sc[i%2].markdown(f"**{feed}:** {'✅ Verbunden' if status else '⚠️ Fallback aktiv / Offline'}")
    st.caption("Fallback-Werte sind nicht als Live-Daten zu interpretieren.")
st.markdown("---");st.caption("⚠️ Modellhinweis: Der Final Regime Score ist ein quantitatives Entscheidungs- und Regimefilter-Modell und keine Anlageberatung. Der LQD/HYG-Wert ist ein Kreditmarkt-Proxy und kein tatsächlicher Credit Spread. Die Put/Call-Komponente ist separat ausgewiesen; Futures selbst besitzen keine Put/Call-Ratio.")

    # ========================================================
    # GOLD
    # ========================================================
    "Gold (XAU/USD)": {

        "ticker": "GC=F",
        "volatility_ticker": "^GVZ",
        "cot_code": "GOLD",

        "invert_inverts": [
            "vix_score",
            "usd_index",
            "real_yields",
            "fed_policy"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": 0.35,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.15,
            "Fundamentale_Faktoren": 0.00,
            "Fruehwarnindikatoren": 0.10
        },

        "Sub_Gewichte": {

            "Positionierung": {
                "cot_commercials": 0.80,
                "fear_greed": 0.20
            },

            "Marktinterna": {
                "obv_momentum": 0.50,
                "vix_score": 0.50
            },

            "Fundamentale_Faktoren": {}
        }
    },

    # ========================================================
    # WTI CRUDE OIL
    # ========================================================
    "WTI Crude Oil": {

        "ticker": "CL=F",
        "volatility_ticker": "^OVX",
        "cot_code": "CRUDE OIL",

        "invert_inverts": [
            "vix_score",
            "usd_index",
            "inventories"
        ],

        "Saeulen_Gewichte": {
            "Makroökonomie": 0.30,
            "Positionierung": 0.25,
            "Marktinterna": 0.15,
            "Technischer_Trend": 0.20,
            "Fundamentale_Faktoren": 0.10,
            "Fruehwarnindikatoren": 0.00
        },

        "Sub_Gewichte": {

            "Positionierung": {
                "cot_commercials": 0.80,
                "fear_greed": 0.20
            },

            "Marktinterna": {
                "obv_momentum": 0.50,
                "vix_score": 0.50
            },

            "Fundamentale_Faktoren": {
                "inventories": 1.00
            }
        }
    }
}

# ============================================================
# 2. BASE SUB-WEIGHTS
# ============================================================

SUB_WEIGHTS_BASE = {

    "Makroökonomie": {

        "fed_policy": 0.20,
        "real_yields": 0.30,
        "usd_index": 0.20,
        "net_liquidity": 0.30
    },

    "Technischer_Trend": {

        "distance_200ma": 0.35,
        "distance_50ma": 0.35,
        "rsi_momentum": 0.30
    },

    "Fruehwarnindikatoren": {

        "credit_spreads": 0.60,
        "move_index": 0.40
    }
}

# ============================================================
# 3. LOOKBACK CONFIG
# ============================================================

LOOKBACK_CONFIG = {

    "fed_policy": 1260,
    "real_yields": 756,
    "net_liquidity": 756,
    "credit_spreads": 756,
    "usd_index": 504,
    "inventories": 756
}

# ============================================================
# 4. VOLATILITY THRESHOLDS
# ============================================================

VOLA_THRESHOLDS = {

    "S&P 500": 30.0,
    "Nasdaq 100": 35.0,
    "Gold (XAU/USD)": 25.0,
    "WTI Crude Oil": 45.0
}

# ============================================================
# 5. GOOGLE TRENDS CONFIG
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
    }
}

# ============================================================
# 6. GOOGLE TRENDS SENTIMENT ENGINE
# ============================================================

@st.cache_data(ttl=21600)
def fetch_google_trends_sentiment(asset_name: str):

    cfg = TREND_KEYWORD_MAP.get(
        asset_name,
        TREND_KEYWORD_MAP["S&P 500"]
    )

    try:

        pytrends = TrendReq(
            hl=cfg["lang"],
            tz=360,
            retries=2,
            backoff_factor=0.2
        )

        all_kws = (
            cfg["bull"]
            + cfg["bear"]
        )

        pytrends.build_payload(
            all_kws,
            timeframe="today 3-m",
            geo=cfg["geo"]
        )

        df_trends = (
            pytrends
            .interest_over_time()
        )

        if df_trends.empty:
            return 50.0, 0.0, False

        if "isPartial" in df_trends.columns:

            df_trends = (
                df_trends
                .drop(
                    columns=["isPartial"]
                )
            )

        def calc_z(series):

            series = pd.to_numeric(
                series,
                errors="coerce"
            )

            rolling_mean = (
                series
                .rolling(
                    21,
                    min_periods=5
                )
                .mean()
            )

            rolling_std = (
                series
                .rolling(
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
                (series - rolling_mean)
                / rolling_std
            )

        valid_bull = [
            kw
            for kw in cfg["bull"]
            if kw in df_trends.columns
        ]

        valid_bear = [
            kw
            for kw in cfg["bear"]
            if kw in df_trends.columns
        ]

        if (
            not valid_bull
            or not valid_bear
        ):

            return 50.0, 0.0, False

        z_bull = sum(
            calc_z(
                df_trends[kw]
            )
            for kw in valid_bull
        ) / len(valid_bull)

        z_bear = sum(
            calc_z(
                df_trends[kw]
            )
            for kw in valid_bear
        ) / len(valid_bear)

        net_spread = (
            z_bull - z_bear
        )

        clean_spread = (
            net_spread
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if clean_spread.empty:

            return 50.0, 0.0, False

        latest_spread = float(
            clean_spread.iloc[-1]
        )

        contrarian_score = float(
            np.clip(
                50.0
                - latest_spread * 15.0,
                0.0,
                100.0
            )
        )

        return (
            round(
                contrarian_score,
                1
            ),
            round(
                latest_spread,
                2
            ),
            True
        )

    except Exception:

        return 50.0, 0.0, False

# ============================================================
# 7. TIMEZONE HELPER
# ============================================================

def strip_timezone(
    datetime_index_or_series
):

    dt = pd.to_datetime(
        datetime_index_or_series,
        errors="coerce"
    )

    if isinstance(
        dt,
        pd.Series
    ):

        if (
            hasattr(dt.dt, "tz")
            and dt.dt.tz is not None
        ):

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

# ============================================================
# 8. NORMALIZATION
# ============================================================

def normalize_to_percentile(
    series: pd.Series,
    lookback: int = 252,
    invert: bool = False
):

    clean_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    clean_series = (
        clean_series
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .ffill()
        .bfill()
    )

    if clean_series.isna().all():

        return pd.Series(
            50.0,
            index=series.index
        )

    rolling_mean = (
        clean_series
        .rolling(
            window=lookback,
            min_periods=20
        )
        .mean()
    )

    rolling_std = (
        clean_series
        .rolling(
            window=lookback,
            min_periods=20
        )
        .std()
        .replace(
            0,
            np.nan
        )
    )

    z_scores = (
        clean_series
        - rolling_mean
    ) / rolling_std

    percentiles = (
        pd.Series(
            norm.cdf(
                z_scores
            ) * 100,
            index=series.index
        )
    )

    if invert:

        percentiles = (
            100
            - percentiles
        )

    return (
        percentiles
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .clip(
            0,
            100
        )
        .ffill()
        .bfill()
        .fillna(50.0)
    )

# ============================================================
# 9. MODEL CONSISTENCY INDEX
# ============================================================

def calculate_mci(
    scores,
    weights
):

    scores = np.asarray(
        scores,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    valid = (
        np.isfinite(scores)
        & np.isfinite(weights)
    )

    scores = scores[valid]
    weights = weights[valid]

    if (
        len(scores) == 0
        or len(weights) == 0
        or np.sum(weights) <= 0
    ):

        return 0.0

    weights = (
        weights
        / np.sum(weights)
    )

    weighted_mean = np.average(
        scores,
        weights=weights
    )

    weighted_variance = np.average(
        (
            scores
            - weighted_mean
        ) ** 2,
        weights=weights
    )

    weighted_std = np.sqrt(
        weighted_variance
    )

    max_std = 50.0

    consistency = (
        100
        * (
            1
            - weighted_std
            / max_std
        )
    )

    return round(
        float(
            np.clip(
                consistency,
                0.0,
                100.0
            )
        ),
        1
    )

# ============================================================
# 10. REGIME LABEL
# ============================================================

def get_regime_label(
    score
):

    if score >= 90:

        return (
            "🟢 Risk-On "
            "(Extrem Bullisch)"
        )

    elif score >= 75:

        return (
            "🟢 Expansion "
            "(Bullisch)"
        )

    elif score >= 60:

        return (
            "🟡 Übergangsphase "
            "(Leicht Bullisch)"
        )

    elif score >= 40:

        return "🟡 Neutral"

    elif score >= 25:

        return (
            "🟠 Risk-Off "
            "(Bärisch)"
        )

    else:

        return (
            "🔴 Stressphase "
            "(Stark Bärisch)"
        )

# ============================================================
# 11. SAFE REINDEX
# ============================================================

def safe_reindex_series(
    source_series: pd.Series,
    target_index: pd.Index
):

    if (
        source_series is None
        or not isinstance(
            source_series,
            pd.Series
        )
        or source_series.empty
    ):

        return None

    s = source_series.copy()

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

    clean_target = (
        strip_timezone(
            target_index
        )
        .floor("D")
    )

    reindexed = (
        s
        .reindex(
            clean_target,
            method="ffill"
        )
        .ffill()
        .bfill()
    )

    reindexed.index = target_index

    return reindexed

# ============================================================
# 12. YFINANCE COLUMN HELPER
# ============================================================

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

            result = data[field]

            if isinstance(
                result,
                pd.Series
            ):

                return result

        return None

    level0 = (
        data.columns
        .get_level_values(0)
    )

    level1 = (
        data.columns
        .get_level_values(1)
    )

    if field in level0:

        result = data[field]

        return (
            result
            if isinstance(
                result,
                pd.DataFrame
            )
            else None
        )

    if field in level1:

        result = data.xs(
            field,
            axis=1,
            level=1
        )

        return (
            result
            if isinstance(
                result,
                pd.DataFrame
            )
            else None
        )

    return None

# ============================================================
# 13. ASSET SELECTION
# ============================================================

with st.sidebar:

    st.title(
        "⚙️ Multi-Asset Selector"
    )

    selected_asset = st.selectbox(
        "🎯 Asset auswählen",
        list(
            ASSET_CONFIGS.keys()
        ),
        index=0
    )

    st.markdown("---")

    st.markdown(
        "### 📡 API Live-Feed Monitor"
    )

# ============================================================
# 14. FRED API
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
# 15. CNN FEAR & GREED
# ============================================================

@st.cache_data(ttl=14400)
def fetch_fear_and_greed():

    url = (
        "https://production.dataviz.cnn.io/"
        "index/fearandgreed/graphdata"
    )

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36",

        "Accept":
            "application/json",

        "Referer":
            "https://edition.cnn.com/"
    }

    try:

        res = requests.get(
            url,
            headers=headers,
            timeout=8
        )

        if res.status_code != 200:

            return 55.0, False

        data = res.json()

        hist = (
            data
            .get(
                "fear_and_greed_historical",
                {}
            )
            .get(
                "data",
                []
            )
        )

        if not hist:

            return 55.0, False

        df_hist = pd.DataFrame(
            hist
        )

        if (
            "x" not in df_hist.columns
            or "y" not in df_hist.columns
        ):

            return 55.0, False

        df_hist["Date"] = (
            strip_timezone(
                pd.to_datetime(
                    df_hist["x"],
                    unit="ms",
                    errors="coerce"
                )
            )
            .dt
            .floor("D")
        )

        df_hist["y"] = pd.to_numeric(
            df_hist["y"],
            errors="coerce"
        )

        df_hist = (
            df_hist
            .dropna(
                subset=[
                    "Date",
                    "y"
                ]
            )
            .drop_duplicates(
                subset=["Date"],
                keep="last"
            )
        )

        return (
            df_hist
            .set_index("Date")[
                "y"
            ]
            .sort_index(),
            True
        )

    except Exception:

        return 55.0, False

# ============================================================
# 16. CFTC COT DATA
# ============================================================

@st.cache_data(ttl=86400)
def fetch_cot_data(
    asset_search_string
):

    headers = {
        "User-Agent":
            "Mozilla/5.0"
    }

    current_year = (
        pd.Timestamp.now().year
    )

    years = [
        current_year - 1,
        current_year
    ]

    frames = []

    for yr in years:

        url = (
            "https://www.cftc.gov/files/"
            "dea/history/"
            f"fut_com_txt_{yr}.zip"
        )

        try:

            res = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            if res.status_code != 200:
                continue

            with zipfile.ZipFile(
                io.BytesIO(
                    res.content
                )
            ) as z:

                file_names = (
                    z.namelist()
                )

                if not file_names:
                    continue

                file_name = (
                    file_names[0]
                )

                df_yr = pd.read_csv(
                    z.open(file_name),
                    low_memory=False
                )

                if df_yr.empty:
                    continue

                market_col = (
                    df_yr.columns[0]
                )

                rows = df_yr[
                    df_yr[
                        market_col
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

        df_all = pd.concat(
            frames,
            ignore_index=True
        )

        date_cols = [
            c
            for c in df_all.columns
            if "As_of_Date" in str(c)
        ]

        long_cols = [
            c
            for c in df_all.columns
            if (
                "Comm_Positions_Long_All"
                in str(c)
            )
        ]

        short_cols = [
            c
            for c in df_all.columns
            if (
                "Comm_Positions_Short_All"
                in str(c)
            )
        ]

        if (
            not date_cols
            or not long_cols
            or not short_cols
        ):

            return None, False

        date_col = date_cols[0]
        long_col = long_cols[0]
        short_col = short_cols[0]

        dates = (
            pd.to_datetime(
                df_all[
                    date_col
                ].astype(str),
                format="%Y%m%d",
                errors="coerce"
            )
        )

        df_all["Date"] = (
            strip_timezone(
                dates
            )
            .dt
            .floor("D")
        )

        long_values = pd.to_numeric(
            df_all[
                long_col
            ],
            errors="coerce"
        )

        short_values = pd.to_numeric(
            df_all[
                short_col
            ],
            errors="coerce"
        )

        df_all[
            "Net_Commercials"
        ] = (
            long_values
            - short_values
        )

        df_all = (
            df_all
            .dropna(
                subset=["Date"]
            )
            .drop_duplicates(
                subset=["Date"],
                keep="last"
            )
        )

        if df_all.empty:

            return None, False

        return (
            df_all
            .set_index("Date")[
                "Net_Commercials"
            ]
            .sort_index(),
            True
        )

    except Exception:

        return None, False

# ============================================================
# 17. MAIN DATA PIPELINE
# ============================================================

@st.cache_data(ttl=3600)
def fetch_multi_asset_data(
    selected_asset
):

    cfg = ASSET_CONFIGS[
        selected_asset
    ]

    feed_status = {}

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
                "yFinance (Preis & Tech)":
                    False
            }
        )

    if data.empty:

        return (
            pd.DataFrame(),
            {
                "yFinance (Preis & Tech)":
                    False
            }
        )

    # --------------------------------------------------------
    # CLOSE DATA
    # --------------------------------------------------------

    close_data = (
        extract_yfinance_field(
            data,
            "Close"
        )
    )

    if close_data is None:

        return (
            pd.DataFrame(),
            {
                "yFinance (Preis & Tech)":
                    False
            }
        )

    if isinstance(
        close_data,
        pd.Series
    ):

        close_data = (
            close_data
            .to_frame()
        )

    close_data = (
        close_data
        .copy()
    )

    # --------------------------------------------------------
    # FLATTEN / RENAME
    # --------------------------------------------------------

    if isinstance(
        close_data.columns,
        pd.MultiIndex
    ):

        close_data.columns = (
            close_data.columns
            .get_level_values(-1)
        )

    rename_map = {
        ticker: key
        for key, ticker
        in tickers.items()
    }

    close_data = (
        close_data
        .rename(
            columns=rename_map
        )
    )

    close_data = (
        close_data
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .sort_index()
    )

    feed_status[
        "yFinance (Preis & Tech)"
    ] = (
        "asset" in close_data.columns
        and not close_data["asset"]
        .dropna()
        .empty
    )

    if (
        close_data.empty
        or "asset"
        not in close_data.columns
    ):

        return (
            pd.DataFrame(),
            feed_status
        )

    price = (
        close_data[
            "asset"
        ]
        .dropna()
    )

    if price.empty:

        return (
            pd.DataFrame(),
            feed_status
        )

    df_raw = pd.DataFrame(
        index=price.index
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_data = (
        extract_yfinance_field(
            data,
            "Volume"
        )
    )

    has_vol = False

    if (
        volume_data is not None
        and not volume_data.empty
    ):

        if isinstance(
            volume_data,
            pd.Series
        ):

            asset_volume = (
                volume_data
                .reindex(
                    price.index
                )
                .ffill()
                .bfill()
            )

            has_vol = True

        else:

            if isinstance(
                volume_data.columns,
                pd.MultiIndex
            ):

                volume_data.columns = (
                    volume_data.columns
                    .get_level_values(-1)
                )

            volume_data = (
                volume_data
                .rename(
                    columns=rename_map
                )
            )

            if (
                "asset"
                in volume_data.columns
            ):

                asset_volume = (
                    volume_data[
                        "asset"
                    ]
                    .reindex(
                        price.index
                    )
                    .ffill()
                    .bfill()
                )

                has_vol = True

    if not has_vol:

        asset_volume = pd.Series(
            1000.0,
            index=price.index
        )

    feed_status[
        "Volumen / Orderflow Feed"
    ] = has_vol

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    ma50 = (
        price
        .rolling(
            50,
            min_periods=50
        )
        .mean()
    )

    ma200 = (
        price
        .rolling(
            200,
            min_periods=200
        )
        .mean()
    )

    df_raw[
        "distance_50ma"
    ] = (
        (
            price
            - ma50
        )
        / ma50.replace(
            0,
            np.nan
        )
    ) * 100

    df_raw[
        "distance_200ma"
    ] = (
        (
            price
            - ma200
        )
        / ma200.replace(
            0,
            np.nan
        )
    ) * 100

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = price.diff()

    gain = (
        delta
        .clip(
            lower=0
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    loss = (
        -delta
        .clip(
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
        / loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        - (
            100
            / (
                1 + rs
            )
        )
    )

    df_raw[
        "rsi_momentum"
    ] = rsi.clip(
        0,
        100
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    if "vix" in close_data.columns:

        df_raw[
            "vix_score"
        ] = (
            close_data[
                "vix"
            ]
            .reindex(
                df_raw.index
            )
            .ffill()
            .bfill()
        )

    else:

        df_raw[
            "vix_score"
        ] = 20.0

    # --------------------------------------------------------
    # DOLLAR
    # --------------------------------------------------------

    if "dxy" in close_data.columns:

        df_raw[
            "usd_index"
        ] = (
            close_data[
                "dxy"
            ]
            .reindex(
                df_raw.index
            )
            .ffill()
            .bfill()
        )

    else:

        df_raw[
            "usd_index"
        ] = 100.0

    # --------------------------------------------------------
    # MOVE
    # --------------------------------------------------------

    if "move" in close_data.columns:

        df_raw[
            "move_index"
        ] = (
            close_data[
                "move"
            ]
            .reindex(
                df_raw.index
            )
            .ffill()
            .bfill()
        )

    else:

        df_raw[
            "move_index"
        ] = 100.0

    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------
    #
    # Kein echter Credit Spread.
    #
    # LQD/HYG dient lediglich als Marktpreis-basierter
    # Proxy für die relative Stärke von Investment Grade
    # gegenüber High Yield.
    #
    # Höheres LQD/HYG-Verhältnis:
    # -> defensiveres Kreditmarktverhalten
    # -> wird deshalb im Modell invertiert.
    #

    if (
        "lqd" in close_data.columns
        and "hyg" in close_data.columns
    ):

        hyg = (
            close_data[
                "hyg"
            ]
            .reindex(
                df_raw.index
            )
            .ffill()
            .bfill()
        )

        lqd = (
            close_data[
                "lqd"
            ]
            .reindex(
                df_raw.index
            )
            .ffill()
            .bfill()
        )

        df_raw[
            "credit_spreads"
        ] = (
            lqd
            / hyg.replace(
                0,
                np.nan
            )
        )

    else:

        df_raw[
            "credit_spreads"
        ] = 1.0

    # --------------------------------------------------------
    # MARKET MOMENTUM
    # --------------------------------------------------------

    df_raw[
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

    obv_direction = np.where(
        delta > 0,
        asset_volume,
        np.where(
            delta < 0,
            -asset_volume,
            0.0
        )
    )

    obv = pd.Series(
        obv_direction,
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

    df_raw[
        "obv_momentum"
    ] = (
        (
            obv
            - obv_ema
        )
        /
        obv_ema.abs()
        .replace(
            0,
            np.nan
        )
    ) * 100

    # --------------------------------------------------------
    # S&P FUNDAMENTAL
    # --------------------------------------------------------

    if selected_asset == "S&P 500":

        # Bewusster Modell-Proxy.
        #
        # Kein Live-KGV und daher ausdrücklich nicht
        # als echte historische KGV-Zeitreihe zu verstehen.

        df_raw[
            "pe_valuation"
        ] = 24.5

    # --------------------------------------------------------
    # FRED DATA
    # --------------------------------------------------------

    fred_success = False

    if FRED_API_KEY:

        try:

            fred = Fred(
                api_key=FRED_API_KEY
            )

            walcl_s = (
                safe_reindex_series(
                    fred.get_series(
                        "WALCL"
                    ),
                    df_raw.index
                )
            )

            tga_s = (
                safe_reindex_series(
                    fred.get_series(
                        "WTREGEN"
                    ),
                    df_raw.index
                )
            )

            rrp_s = (
                safe_reindex_series(
                    fred.get_series(
                        "RRPONTSYD"
                    ),
                    df_raw.index
                )
            )

            if (
                walcl_s is not None
                and tga_s is not None
                and rrp_s is not None
            ):

                # WALCL und WTREGEN:
                # Millionen USD
                #
                # RRPONTSYD:
                # Milliarden USD
                #
                # Umrechnung auf Milliarden USD.

                df_raw[
                    "net_liquidity"
                ] = (
                    (
                        walcl_s
                        - tga_s
                        - (
                            rrp_s
                            * 1000.0
                        )
                    )
                    / 1000.0
                )

            else:

                df_raw[
                    "net_liquidity"
                ] = np.nan

            fed_series = (
                safe_reindex_series(
                    fred.get_series(
                        "FEDFUNDS"
                    ),
                    df_raw.index
                )
            )

            real_yield_series = (
                safe_reindex_series(
                    fred.get_series(
                        "DFII10"
                    ),
                    df_raw.index
                )
            )

            df_raw[
                "fed_policy"
            ] = fed_series

            df_raw[
                "real_yields"
            ] = real_yield_series

            if selected_asset == "WTI Crude Oil":

                inv_s = (
                    safe_reindex_series(
                        fred.get_series(
                            "WCESTUS1"
                        ),
                        df_raw.index
                    )
                )

                df_raw[
                    "inventories"
                ] = inv_s

            fred_success = True

        except Exception:

            fred_success = False

    if not fred_success:

        df_raw[
            "fed_policy"
        ] = np.nan

        df_raw[
            "real_yields"
        ] = np.nan

        df_raw[
            "net_liquidity"
        ] = np.nan

        if selected_asset == "WTI Crude Oil":

            df_raw[
                "inventories"
            ] = np.nan

    feed_status[
        "FRED API (Makro & Fed)"
    ] = fred_success

    # ========================================================
    # COT
    # ========================================================

    cot_data, cot_live = (
        fetch_cot_data(
            cfg["cot_code"]
        )
    )

    feed_status[
        f"CFTC COT ({cfg['cot_code']})"
    ] = cot_live

    cot_reindexed = (
        safe_reindex_series(
            cot_data,
            df_raw.index
        )
    )

    if cot_reindexed is not None:

        df_raw[
            "cot_commercials"
        ] = cot_reindexed

    else:

        # Kein künstlicher Ersatz durch technische Daten.
        # Neutraler Fallback wird erst bei der Normalisierung
        # auf Score 50 behandelt.

        df_raw[
            "cot_commercials"
        ] = np.nan

    # ========================================================
    # CNN FEAR & GREED
    # ========================================================

    fg_data, fg_live = (
        fetch_fear_and_greed()
    )

    feed_status[
        "CNN Fear & Greed"
    ] = fg_live

    if isinstance(
        fg_data,
        pd.Series
    ):

        fg_reindexed = (
            safe_reindex_series(
                fg_data,
                df_raw.index
            )
        )

        if fg_reindexed is not None:

            df_raw[
                "fear_greed"
            ] = fg_reindexed

        else:

            df_raw[
                "fear_greed"
            ] = np.nan

    else:

        df_raw[
            "fear_greed"
        ] = float(
            fg_data
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    df_norm = pd.DataFrame(
        index=df_raw.index
    )

    inverts = cfg[
        "invert_inverts"
    ]

    for col in df_raw.columns:

        should_invert = (
            col in inverts
        )

        lb = LOOKBACK_CONFIG.get(
            col,
            252
        )

        df_norm[col] = (
            normalize_to_percentile(
                df_raw[col],
                lookback=lb,
                invert=should_invert
            )
        )

    # ========================================================
    # DASHBOARD SCORES
    # ========================================================

    df_dashboard = pd.DataFrame(
        index=df_raw.index
    )

    df_dashboard[
        "Raw_Volatility"
    ] = df_raw[
        "vix_score"
    ]

    # --------------------------------------------------------
    # SUB-WEIGHTS
    # --------------------------------------------------------

    active_sub_weights = (
        SUB_WEIGHTS_BASE.copy()
    )

    for category, weights in (
        cfg[
            "Sub_Gewichte"
        ].items()
    ):

        active_sub_weights[
            category
        ] = weights

    # --------------------------------------------------------
    # SIX PILLARS
    # --------------------------------------------------------

    for saeule, indikatoren in (
        active_sub_weights.items()
    ):

        cols = [
            c
            for c in indikatoren.keys()
            if c in df_norm.columns
        ]

        weights = [
            indikatoren[c]
            for c in cols
        ]

        if (
            cols
            and sum(weights) > 0
        ):

            weights_norm = (
                np.array(
                    weights,
                    dtype=float
                )
                /
                np.sum(weights)
            )

            df_dashboard[
                f"Saeule_{saeule}"
            ] = (
                df_norm[
                    cols
                ]
                .dot(
                    weights_norm
                )
            )

        else:

            df_dashboard[
                f"Saeule_{saeule}"
            ] = 50.0

    # ========================================================
    # FINAL REGIME SCORE
    # ========================================================

    saeulen_cols = []

    saeulen_weights = []

    for saeule, weight in (
        cfg[
            "Saeulen_Gewichte"
        ].items()
    ):

        col_name = (
            f"Saeule_{saeule}"
        )

        if col_name in df_dashboard.columns:

            saeulen_cols.append(
                col_name
            )

            saeulen_weights.append(
                weight
            )

    if (
        saeulen_cols
        and sum(saeulen_weights) > 0
    ):

        saeulen_weights_norm = (
            np.array(
                saeulen_weights,
                dtype=float
            )
            /
            np.sum(
                saeulen_weights
            )
        )

        df_dashboard[
            "Final_Regime_Score"
        ] = (
            df_dashboard[
                saeulen_cols
            ]
            .dot(
                saeulen_weights_norm
            )
            .clip(
                0,
                100
            )
            .round(1)
        )

        # ----------------------------------------------------
        # MODEL CONSISTENCY
        # ----------------------------------------------------

        mci_list = []

        for i in range(
            len(df_dashboard)
        ):

            row_scores = (
                df_dashboard[
                    saeulen_cols
                ]
                .iloc[i]
                .values
            )

            mci_list.append(
                calculate_mci(
                    row_scores,
                    saeulen_weights_norm
                )
            )

        df_dashboard[
            "MCI"
        ] = mci_list

    else:

        df_dashboard[
            "Final_Regime_Score"
        ] = 50.0

        df_dashboard[
            "MCI"
        ] = 0.0

    # ========================================================
    # PRICE
    # ========================================================

    df_dashboard[
        "Asset_Price"
    ] = (
        price
        .reindex(
            df_dashboard.index
        )
        .ffill()
        .bfill()
    )

    return (
        df_dashboard
        .dropna(
            subset=[
                "Final_Regime_Score"
            ]
        ),
        feed_status
    )

# ============================================================
# 18. LOAD DATA
# ============================================================

with st.spinner(
    f"Lade quantitative Daten für "
    f"{selected_asset}..."
):

    df_dash, feed_status = (
        fetch_multi_asset_data(
            selected_asset
        )
    )

# ============================================================
# 19. SIDEBAR LIVE FEEDS
# ============================================================

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
                f"*(Fallback / Offline)*"
            )

# ============================================================
# 20. ERROR HANDLING
# ============================================================

if df_dash.empty:

    st.error(
        "⚠️ Marktdaten konnten nicht "
        "geladen werden. Yahoo Finance "
        "ist möglicherweise nicht erreichbar."
    )

    st.stop()

# ============================================================
# 21. CURRENT DATA
# ============================================================

heute = (
    df_dash
    .iloc[-1]
    .copy()
)

# ============================================================
# 22. DELTAS
# ============================================================

if len(df_dash) >= 2:

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
    )

else:

    heute[
        "Delta_1D"
    ] = 0.0

if len(df_dash) >= 6:

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
    )

else:

    heute[
        "Delta_1W"
    ] = 0.0

# ============================================================
# 23. MAIN TITLE
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
# 24. HERO METRICS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Final Regime Score",
        value=(
            f"{heute['Final_Regime_Score']} / 100"
        ),
        delta=(
            f"{heute['Delta_1D']:+.1f} "
            f"(Heute)"
        )
    )

with col2:

    st.metric(
        label="Model Consistency Index",
        value=(
            f"{heute['MCI']}%"
        ),
        delta=(
            f"{heute['Delta_1W']:+.1f} "
            f"(vs. Vorwoche)"
        ),
        delta_color="off"
    )

st.caption(
    "Der Model Consistency Index misst, "
    "wie stark die sechs Modell-Säulen "
    "untereinander übereinstimmen. "
    "Er ist keine statistische "
    "Wahrscheinlichkeit für steigende "
    "oder fallende Kurse."
)

# ============================================================
# 25. CURRENT REGIME
# ============================================================

st.info(
    f"**Aktuelles Marktregime "
    f"({selected_asset}):** "
    f"{get_regime_label("
    f"heute['Final_Regime_Score']"
    f")}"
)

# ============================================================
# 26. VOLATILITY WARNING
# ============================================================

current_vola = float(
    heute.get(
        "Raw_Volatility",
        20.0
    )
)

vola_limit = (
    VOLA_THRESHOLDS.get(
        selected_asset,
        30.0
    )
)

vola_ticker = (
    ASSET_CONFIGS[
        selected_asset
    ][
        "volatility_ticker"
    ]
)

if current_vola >= vola_limit:

    st.error(
        f"🚨 **VOLATILITÄTS-ALARM:** "
        f"Der {vola_ticker} notiert bei "
        f"**{current_vola:.2f}** "
        f"(Grenzwert: {vola_limit:.1f}). "
        f"Erhöhtes Risiko. "
        f"Positionsgröße reduzieren."
    )

elif current_vola >= (
    vola_limit * 0.8
):

    st.warning(
        f"⚠️ **Erhöhte Volatilität:** "
        f"Der {vola_ticker} steht bei "
        f"**{current_vola:.2f}**. "
        f"Entries selektiver wählen und "
        f"Positionsgröße gegebenenfalls "
        f"reduzieren."
    )

# ============================================================
# 27. INTRADAY TRADING BIAS
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
        f"idealerweise an dynamischen "
        f"Support-Zonen wie VWAP / EMAs."
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

elif score <= 40:

    bias = (
        "🔴 BÄRISCH (Short Bias)"
    )

    rule = (
        f"Bevorzugt nach Short-Setups "
        f"bei {selected_asset} suchen, "
        f"idealerweise an Resistance-Zonen."
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

    pos_size = (
        "50% Size"
    )

if current_vola >= vola_limit:

    pos_size = (
        "FLAT / Max 25% Size"
    )

col_b1, col_b2, col_b3 = (
    st.columns(3)
)

with col_b1:

    st.metric(
        "Handelsrichtung",
        bias
    )

with col_b2:

    st.metric(
        "Positionsgröße",
        pos_size
    )

with col_b3:

    st.metric(
        "Fokus",
        (
            "Trend-Follow"
            if abs(score - 50) > 15
            else "Mean-Reversion"
        )
    )

st.info(
    f"**Übergeordnete Regel:** "
    f"{rule}"
)

# ============================================================
# 28. GOOGLE TRENDS SENTIMENT
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Retail Sentiment (Google Trends)"
)

st.caption(
    "Unabhängiger Kontraindikator auf "
    "Basis des Suchverhaltens von "
    "Privatanlegern."
)

(
    contra_score,
    net_spread,
    trends_live
) = fetch_google_trends_sentiment(
    selected_asset
)

col_gt1, col_gt2, col_gt3 = (
    st.columns(3)
)

with col_gt1:

    st.metric(
        label="Google Retail Score (0-100)",
        value=(
            f"{contra_score} / 100"
        ),
        delta=(
            f"Net Spread: "
            f"{net_spread:+.2f} σ"
        ),
        delta_color="inverse"
    )

with col_gt2:

    if contra_score >= 65:

        st.success(
            "🟢 **Panik-Ausschlag:** "
            "Angst-Überhang im Retail-Segment. "
            "Kontraindikativ potenziell "
            "positiv."
        )

    elif contra_score <= 35:

        st.error(
            "🔴 **Gier-Ausschlag:** "
            "Retail-Suchvolumen signalisiert "
            "mögliche Überhitzung. "
            "Vorsicht vor Long-Positionen."
        )

    else:

        st.info(
            "🟡 **Ausgeglichenes Sentiment:** "
            "Kein extremes Signal."
        )

with col_gt3:

    cfg = TREND_KEYWORD_MAP.get(
        selected_asset,
        TREND_KEYWORD_MAP["S&P 500"]
    )

    bull_str = ", ".join(
        [
            f"'{k}'"
            for k in cfg["bull"]
        ]
    )

    bear_str = ", ".join(
        [
            f"'{k}'"
            for k in cfg["bear"]
        ]
    )

    st.markdown(
        f"""
**🔍 Getrackte Parameter:**

* **Region:** `{cfg['geo']}`
* **Euphorie:** {bull_str}
* **Panik:** {bear_str}
* **Status:** {
    '🟢 Live'
    if trends_live
    else
    '🔴 Offline/Fallback'
}
"""
    )

# ============================================================
# 29. DRIVER ANALYSIS – DETAILS & LINKS
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Treiber-Analyse "
    "(Die 6 Säulen)"
)

saeulen_details = {

    # ========================================================
    # MAKROÖKONOMIE
    # ========================================================

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

    # ========================================================
    # POSITIONIERUNG
    # ========================================================

    "Positionierung": {

        "quelle":
            "CFTC COT & CNN Fear & Greed",

        "funktion":
            "Institutionelle Commercial-Positionierung "
            "und allgemeines Marktsentiment.",

        "links": [

            (
                "CFTC: Commitment of Traders",
                "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
            ),

            (
                "CNN: Fear & Greed Index",
                "https://edition.cnn.com/markets/fear-and-greed"
            )
        ]
    },

    # ========================================================
    # MARKTINTERNA
    # ========================================================

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
            )
        ]
    },

    # ========================================================
    # TECHNISCHER TREND
    # ========================================================

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

    # ========================================================
    # FUNDAMENTALE FAKTOREN
    # ========================================================

    "Fundamentale_Faktoren": {

        "quelle":
            "FRED, Multpl & WSJ",

        "funktion":
            "Bewertung bzw. Rohstoff-Lagerbestände.",

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

    # ========================================================
    # FRÜHWARNINDIKATOREN
    # ========================================================

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
        heute.get(
            s_name,
            50.0
        )
    )

    raw_name = (
        s_name.replace(
            "Saeule_",
            ""
        )
    )

    label = (
        raw_name.replace(
            "_",
            " "
        )
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

    gewichtung = (
        ASSET_CONFIGS[
            selected_asset
        ][
            "Saeulen_Gewichte"
        ].get(
            raw_name,
            0
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
            value=f"{val:.1f}"
        )

        if raw_name in saeulen_details:

            details = (
                saeulen_details[
                    raw_name
                ]
            )

            with st.expander(
                "Details, Daten & Links"
            ):

                st.markdown(
                    f"**⚖️ Gewichtung:** "
                    f"{gewichtung:.0f}%"
                )

                st.markdown(
                    f"**📡 Quelle:** "
                    f"{details['quelle']}"
                )

                st.markdown(
                    f"**⚙️ Funktion:** "
                    f"{details['funktion']}"
                )

                st.markdown(
                    "**🔗 Live-Datenquellen:**"
                )

                for (
                    link_title,
                    url
                ) in details["links"]:

                    st.markdown(
                        f"• [{link_title}]({url})"
                    )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

# ============================================================
# 30. INTRADAY EXECUTION CHECKLIST
# ============================================================

st.markdown("---")

st.subheader(
    "⚡ Intraday Execution Checkliste & Filter"
)

score_gesamt = float(
    heute.get(
        "Final_Regime_Score",
        50.0
    )
)

trend_wert = float(
    heute.get(
        "Saeule_Technischer_Trend",
        50.0
    )
)

fruehwarn_wert = float(
    heute.get(
        "Saeule_Fruehwarnindikatoren",
        50.0
    )
)

# WICHTIG:
# Der Schlüssel entspricht exakt dem Dictionary-Key
# in ASSET_CONFIGS.
#
# Alter Fehler:
# "Saeule_Makroekonomie"
#
# Korrekt:
# "Saeule_Makroökonomie"

makro_wert = float(
    heute.get(
        "Saeule_Makroökonomie",
        50.0
    )
)

# ============================================================
# STRUCTURAL FILTERS
# ============================================================

trend_intakt = bool(
    trend_wert > 55
)

kein_bond_stress = bool(
    fruehwarn_wert > 35
)

makro_tailwind = bool(
    makro_wert > 50
)

# ============================================================
# DATE PROFILE
# ============================================================

heute_datum = pd.Timestamp.now(
    tz="Europe/Berlin"
)

wochentag_index = (
    heute_datum.weekday()
)

# ============================================================
# OPEX / HEXENSABBAT
# ============================================================

ist_hexensabbat = (

    heute_datum.month
    in [3, 6, 9, 12]

    and

    wochentag_index == 4

    and

    15 <= heute_datum.day <= 21
)

opex_default = (
    not ist_hexensabbat
)

wochentag_profile = {

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
}

heutiges_profil = (
    wochentag_profile.get(
        wochentag_index,
        "Wochenende: Märkte geschlossen"
    )
)

# ============================================================
# CHECKBOXES
# ============================================================

col_c1, col_c2 = (
    st.columns(2)
)

with col_c1:

    st.markdown(
        "#### 1. Strukturelle Filter"
    )

    c1_val = st.checkbox(
        "Trendkonformität "
        "(Marktstruktur / gleitende Durchschnitte intakt)",
        value=trend_intakt,
        key="chk_trend_det"
    )

    c2_val = st.checkbox(
        "Anleihen- & Kreditmärkte stabil "
        "(kein akuter Stress)",
        value=kein_bond_stress,
        key="chk_bond_det"
    )

    c3_val = st.checkbox(
        f"Makro-Umgebung im Rücken "
        f"(Score: {makro_wert:.0f})",
        value=makro_tailwind,
        key="chk_makro_det"
    )

    c4_val = st.checkbox(
        f"Statistisches Tagesprofil beachtet "
        f"({heutiges_profil})",
        value=True,
        key="chk_day_profile"
    )

with col_c2:

    st.markdown(
        "#### 2. Timing & Risikomanagement"
    )

    c5_val = st.checkbox(
        "Keine High-Impact News "
        "(CPI, FOMC, NFP) in den nächsten 60 Minuten",
        value=True,
        key="chk_news_det"
    )

    c6_val = st.checkbox(
        "Kein Hexensabbat / Ketten-Verfall "
        "(erhöhte Pinning- und Volatilitätsrisiken)",
        value=opex_default,
        key="chk_opex_det"
    )

    c7_val = st.checkbox(
        "CRV mindestens 1:2 "
        "zum nächsten charttechnischen Ziel",
        value=True,
        key="chk_crv_det"
    )

    c8_val = st.checkbox(
        "US-Eröffnung / Initial Balance abgewartet",
        value=True,
        key="chk_time_det"
    )

# ============================================================
# 31. CHECKLIST PROGRESS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
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
        c8_val
    ]
)

st.progress(
    erfuellte_kriterien / 8.0
)

st.caption(
    f"✅ **{erfuellte_kriterien} von 8 "
    f"Kriterien erfüllt**"
)

alle_kriterien_erfuellt = (
    erfuellte_kriterien == 8
)

# ============================================================
# 32. EXECUTION SIGNAL
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

if (
    alle_kriterien_erfuellt
    and score_gesamt > 55
):

    st.success(
        "🟢 **EXECUTION FREIGABE (GO):** "
        "Alle Filter sind erfüllt und das "
        "Marktregime ist bullisch genug für "
        "den definierten Long-Bias."
    )

elif (
    alle_kriterien_erfuellt
    and score_gesamt < 45
):

    st.error(
        "🔴 **EXECUTION FREIGABE (SHORT):** "
        "Alle Filter sind erfüllt und das "
        "Marktregime ist ausreichend bärisch."
    )

elif score_gesamt < 40:

    st.error(
        "🔴 **STOP / KEIN TRADE:** "
        "Das Marktregime steht auf Defense. "
        "Kapitalerhalt hat Priorität."
    )

else:

    st.warning(
        "🟡 **CAUTION / WARNUNG:** "
        "Gemischte Signale. "
        "Nur selektive Setups bzw. "
        "reduzierte Positionsgröße."
    )

# ============================================================
# 33. HISTORICAL CHART
# ============================================================

st.markdown("---")

st.subheader(
    "📈 Regime-Historie & Asset Preis "
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

        name="Regime Score (0-100)",

        fill="tozeroy",

        line_color="rgba(0,100,255,0.8)"
    ),

    secondary_y=False
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
# 34. MODEL WEIGHTS DISPLAY
# ============================================================

st.markdown("---")

with st.expander(
    "⚖️ Aktuelle Modellgewichtungen"
):

    st.markdown(
        f"### {selected_asset}"
    )

    weights = ASSET_CONFIGS[
        selected_asset
    ][
        "Saeulen_Gewichte"
    ]

    weight_df = pd.DataFrame(
        {
            "Säule": list(
                weights.keys()
            ),

            "Gewichtung": [
                f"{v * 100:.0f}%"
                for v
                in weights.values()
            ]
        }
    )

    st.dataframe(
        weight_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Die Gewichtungen sind fachlich "
        "begründete Startgewichte und "
        "nicht empirisch backtest-optimiert."
    )

# ============================================================
# 35. SYSTEM & API STATUS
# ============================================================

st.markdown("---")

with st.expander(
    "📡 System & API Status Details"
):

    st.write(
        "Live-Verbindungsstatus zu den "
        "externen Datenquellen:"
    )

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
            "⚠️ Fallback aktiv / Offline"
        )

        status_cols[
            i % 2
        ].markdown(
            f"**{feed}:** {icon}"
        )

    st.caption(
        "Das Dashboard verwendet neutrale "
        "Fallbacks bzw. historische Daten, "
        "wenn einzelne Datenquellen nicht "
        "verfügbar sind. Fallback-Werte sind "
        "nicht als Live-Daten zu interpretieren."
    )

# ============================================================
# 36. MODEL DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Modellhinweis: Der Final Regime Score "
    "ist ein quantitatives Entscheidungs- und "
    "Regimefilter-Modell. Er stellt keine "
    "statistische Wahrscheinlichkeit und "
    "keine Anlageberatung dar. Die verwendeten "
    "Gewichtungen sind fachlich begründete "
    "Startwerte und müssen für eine empirische "
    "Validierung historisch getestet werden. "
    "Insbesondere der LQD/HYG-Wert ist lediglich "
    "ein Kreditmarkt-Proxy und kein tatsächlicher "
    "Credit Spread."
)