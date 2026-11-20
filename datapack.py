"""夜のブリーフィング用データパックを作る（GitHub Actionsで毎日実行）。

- レートは yfinance（為替・ドル指数・米10年金利）、ゴールドは Twelve Data（キーがあれば）から取得
- 数字はすべてこのスクリプトが計算する。Claudeは docs/datapack.json を引用するだけ
- 出力: docs/datapack.json（最新）, docs/archive/YYYY-MM-DD.json（的中率の検証用）
"""
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))
CFG = json.load(open(os.path.join(os.path.dirname(__file__), "config.json"), encoding="utf-8"))

# ---------------------------------------------------------------- 取得
def fetch_yf_hourly(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period="60d", interval="1h", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_twelvedata_hourly(symbol: str, key: str) -> pd.DataFrame:
    url = ("https://api.twelvedata.com/time_series?symbol=" + urllib.request.quote(symbol)
           + "&interval=1h&outputsize=1500&timezone=UTC&apikey=" + key)
    data = json.load(urllib.request.urlopen(url, timeout=30))
    if "values" not in data:
        raise RuntimeError(f"twelvedata error: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close"]].astype(float)
    return df


def fetch_calendar() -> list:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))

# ---------------------------------------------------------------- 計算
def to_daily(h1: pd.DataFrame) -> pd.DataFrame:
    """1時間足→日足（UTC日付。日曜の数時間は月曜に含める）"""
    day = h1.index.normalize()
    day = day + pd.to_timedelta((h1.index.dayofweek == 6).astype(int), unit="D")
    g = h1.groupby(day)
    d = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(),
                      "Low": g["Low"].min(), "Close": g["Close"].last()})
    return d[d.index.dayofweek < 5]


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()   # MT5のATRに近いWilder平均


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def trend_score(c, e20, e75, e200) -> int:
    if c > e20 > e75 > e200:
        return 2
    if c < e20 < e75 < e200:
        return -2
    if e20 > e75:
        return 1
    if e20 < e75:
        return -1
    return 0


def round_step(name: str, price: float) -> float:
    if "XAU" in name:
        return 10.0
    if "JPY" in name:
        return 0.5
    return 0.005 if price < 50 else 0.5


def jpy_per_quote(quote: str, rates: dict) -> float | None:
    """決済通貨1単位が何円か"""
    if quote == "JPY":
        return 1.0
    usdjpy = rates.get("USDJPY")
    if quote == "USD":
        return usdjpy
    direct = rates.get(quote + "JPY")
    if direct:
        return direct
    inv = rates.get("USD" + quote)
    if inv and usdjpy:
        return usdjpy / inv
    return None


def lots_for_risk(sl_dist, contract, jpy_q, equity_jpy, risk_pct, step=0.01):
    if not sl_dist or sl_dist <= 0 or not jpy_q:
        return None
    loss_per_lot = sl_dist * contract * jpy_q
    lots = equity_jpy * risk_pct / 100 / loss_per_lot
    return math.floor(lots / step) * step


def analyze(name: str, h1: pd.DataFrame, rates: dict, now_utc: datetime) -> dict:
    d1 = to_daily(h1)
    h4 = h1.resample("4h", origin="epoch").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    price = float(h1["Close"].iloc[-1])
    last_bar = h1.index[-1].to_pydatetime()
    today = d1.iloc[-1]
    prev = d1.iloc[-2]
    wk = d1.resample("W-FRI").agg({"High": "max", "Low": "min"})
    prev_week = wk.iloc[-2] if d1.index[-1] <= wk.index[-1] else wk.iloc[-1]

    atr_d1 = float(atr(d1).iloc[-2])          # 確定した前日まで
    atr_h1 = float(atr(h1).iloc[-2])
    e = {f"ema{n}_{tf}": float(ema(df["Close"], n).iloc[-2])
         for tf, df in (("h1", h1), ("h4", h4)) for n in (20, 75, 200)}
    th1 = trend_score(price, e["ema20_h1"], e["ema75_h1"], e["ema200_h1"])
    th4 = trend_score(price, e["ema20_h4"], e["ema75_h4"], e["ema200_h4"])
    ts = th1 + th4

    P = (prev.High + prev.Low + prev.Close) / 3
    step = round_step(name, price)
    levels = {
        "prev_high": prev.High, "prev_low": prev.Low,
        "today_high": today.High, "today_low": today.Low,
        "prev_week_high": prev_week.High, "prev_week_low": prev_week.Low,
        "pivot": P, "r1": 2 * P - prev.Low, "s1": 2 * P - prev.High,
        "r2": P + (prev.High - prev.Low), "s2": P - (prev.High - prev.Low),
        "round_below": math.floor(price / step) * step,
        "round_above": math.floor(price / step) * step + step,
    }
    levels = {k: float(v) for k, v in levels.items()}
    below = sorted([(v, k) for k, v in levels.items() if v <= price], reverse=True)
    above = sorted([(v, k) for k, v in levels.items() if v >= price])

    inst = CFG["instruments"][name]
    jq = jpy_per_quote(inst["quote"], rates)
    eq, rp = CFG["equity_jpy"], CFG["risk_pct"]
    sl_buf = 0.5 * atr_h1

    long = short = None
    if below:
        sup, sup_name = below[0]
        lo, hi = sup, sup + 0.25 * atr_h1
        entry, sl = (lo + hi) / 2, lo - sl_buf
        tp = next(((v, k) for v, k in sorted((v, k) for k, v in levels.items()) if v >= hi + 0.5 * atr_h1), None)
        long = {"zone_lo": lo, "zone_hi": hi, "sl": sl, "base_level": sup_name,
                "tp": tp[0] if tp else None, "tp_level": tp[1] if tp else None,
                "rr": (tp[0] - entry) / (entry - sl) if tp else None,
                "dist_atr": max(0.0, (price - hi) / atr_h1),
                "lots": lots_for_risk(entry - sl, inst["contract"], jq, eq, rp)}
    if above:
        res, res_name = above[0]
        lo, hi = res - 0.25 * atr_h1, res
        entry, sl = (lo + hi) / 2, hi + sl_buf
        tp = next(((v, k) for v, k in sorted(((v, k) for k, v in levels.items()), reverse=True) if v <= lo - 0.5 * atr_h1), None)
        short = {"zone_lo": lo, "zone_hi": hi, "sl": sl, "base_level": res_name,
                 "tp": tp[0] if tp else None, "tp_level": tp[1] if tp else None,
                 "rr": (entry - tp[0]) / (sl - entry) if tp else None,
                 "dist_atr": max(0.0, (lo - price) / atr_h1),
                 "lots": lots_for_risk(sl - entry, inst["contract"], jq, eq, rp)}

    side, cand = "WAIT", None
    if ts >= 2 and long and long["rr"] and long["rr"] >= 1.5:
        side, cand = "LONG", long
    elif ts <= -2 and short and short["rr"] and short["rr"] >= 1.5:
        side, cand = "SHORT", short
    range_used = (today.High - today.Low) / atr_d1 * 100
    score = 0
    if cand:
        score = abs(ts) + 1 + (1 if cand["dist_atr"] <= 1.0 else 0) - (1 if range_used >= 90 else 0)

    stale_h = (now_utc - last_bar).total_seconds() / 3600
    return {
        "symbol": name, "source": inst["source"], "price": price,
        "last_bar_jst": last_bar.astimezone(JST).strftime("%Y-%m-%d %H:%M"), "stale": stale_h > 3,
        "atr_d1": atr_d1, "atr_h1": atr_h1, "range_used_pct": float(range_used),
        **e, "trend_h1": th1, "trend_h4": th4, "trend_score": ts,
        "levels": levels, "long": long, "short": short, "code_side": side, "score": score,
    }


def rnd(o, digits):
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: rnd(v, 2 if k in ("rr", "dist_atr", "lots", "range_used_pct") else digits) for k, v in o.items()}
    return o


def events_window(cal: list, now_utc: datetime, hours: int) -> list:
    out = []
    for ev in cal:
        if ev.get("impact") not in ("High", "Medium"):
            continue
        if ev.get("country") not in CFG["event_currencies"]:
            continue
        t = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        if now_utc - timedelta(hours=1) <= t <= now_utc + timedelta(hours=hours):
            out.append({"time_jst": t.astimezone(JST).strftime("%m-%d %H:%M"), "currency": ev["country"],
                        "impact": ev["impact"], "title": ev["title"],
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda x: x["time_jst"])


def main():
    now = datetime.now(timezone.utc)
    errors, data, rates = [], {}, {}
    td_key = os.environ.get("TWELVEDATA_API_KEY", "")
    # レート換算用に先に為替を取る
    for name, inst in CFG["instruments"].items():
        try:
            if inst["source"] == "twelvedata" and td_key:
                data[name] = fetch_twelvedata_hourly(inst["ticker"], td_key)
            elif inst["source"] == "twelvedata":
                # キー未設定時は先物で代用（現物と数ドル〜数十ドルずれる）
                inst["source"] = "yfinance:" + inst["fallback_ticker"] + "(先物で代用)"
                data[name] = fetch_yf_hourly(inst["fallback_ticker"])
            else:
                data[name] = fetch_yf_hourly(inst["ticker"])
            rates[name] = float(data[name]["Close"].iloc[-1])
        except Exception as ex:  # 1銘柄の失敗で全体を止めない
            errors.append(f"{name}: {ex}")
    pairs = []
    for name in CFG["instruments"]:
        if name in data:
            try:
                inst_digits = CFG["instruments"][name]["digits"]
                pairs.append(rnd(analyze(name, data[name], rates, now), inst_digits))
            except Exception as ex:
                errors.append(f"{name} analyze: {ex}")
    macro = {}
    for key, tk in CFG["macro"].items():
        try:
            d = to_daily(fetch_yf_hourly(tk))
            macro[key] = {"now": round(float(d["Close"].iloc[-1]), 3),
                          "prev_close": round(float(d["Close"].iloc[-2]), 3),
                          "chg": round(float(d["Close"].iloc[-1] - d["Close"].iloc[-2]), 3)}
        except Exception as ex:
            errors.append(f"{key}: {ex}")
    try:
        events = events_window(fetch_calendar(), now, CFG["event_hours"])
    except Exception as ex:
        events = None
        errors.append(f"calendar: {ex}")

    pairs.sort(key=lambda p: -p["score"])
    pack = {"generated_jst": now.astimezone(JST).strftime("%Y-%m-%d %H:%M"),
            "equity_jpy": CFG["equity_jpy"], "risk_pct": CFG["risk_pct"],
            "rules_version": CFG["rules_version"], "macro": macro, "events": events,
            "pairs": pairs, "errors": errors}
    os.makedirs("docs/archive", exist_ok=True)
    for path in ("docs/datapack.json", f"docs/archive/{now.astimezone(JST):%Y-%m-%d}.json"):
        json.dump(pack, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"pairs": len(pairs), "errors": errors}, ensure_ascii=False))
    if not pairs:
        sys.exit(1)


if __name__ == "__main__":
    main()
