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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))
CFG = json.load(open(os.path.join(os.path.dirname(__file__), "config.json"), encoding="utf-8"))

# ---------------------------------------------------------------- 取得
def fetch_yf_hourly(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period="365d", interval="1h", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_twelvedata_hourly(symbol: str, key: str) -> pd.DataFrame:
    url = ("https://api.twelvedata.com/time_series?symbol=" + urllib.request.quote(symbol)
           + "&interval=1h&outputsize=5000&timezone=UTC&apikey=" + key)
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
    """1時間足→日足。NYクローズ(17:00 NY＝冬7:00/夏6:00 JST)で区切る＝朝には前日が確定している"""
    ny = h1.index.tz_convert("America/New_York")
    day = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)
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


def pip_size(name: str) -> float:
    """表示用の1pip（ゴールドは1ドル単位で表示）"""
    if "XAU" in name:
        return 1.0
    return 0.01 if "JPY" in name else 0.0001


def pip_unit(name: str) -> str:
    return "ドル" if "XAU" in name else "pips"


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
        lmode = "now" if lo <= price <= hi else ("wait" if price > hi else "broken")
        long = {"zone_lo": lo, "zone_hi": hi, "sl": sl, "base_level": sup_name, "entry_mode": lmode,
                "dist_pips": max(0.0, (price - hi) / pip_size(name)),
                "tp": tp[0] if tp else None, "tp_level": tp[1] if tp else None,
                "rr": (tp[0] - entry) / (entry - sl) if tp else None,
                "dist_atr": max(0.0, (price - hi) / atr_h1),
                "lots": lots_for_risk(entry - sl, inst["contract"], jq, eq, rp)}
    if above:
        res, res_name = above[0]
        lo, hi = res - 0.25 * atr_h1, res
        entry, sl = (lo + hi) / 2, hi + sl_buf
        tp = next(((v, k) for v, k in sorted(((v, k) for k, v in levels.items()), reverse=True) if v <= lo - 0.5 * atr_h1), None)
        smode = "now" if lo <= price <= hi else ("wait" if price < lo else "broken")
        short = {"zone_lo": lo, "zone_hi": hi, "sl": sl, "base_level": res_name, "entry_mode": smode,
                 "dist_pips": max(0.0, (lo - price) / pip_size(name)),
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
    range_left = max(0.0, atr_d1 - (today.High - today.Low)) / pip_size(name)
    score = 0
    if cand:
        score = abs(ts) + 1 + (1 if cand["dist_atr"] <= 1.0 else 0) - (1 if range_used >= 90 else 0)

    stale_h = (now_utc - last_bar).total_seconds() / 3600
    return {
        "symbol": name, "digits": inst["digits"], "source": h1.attrs.get("source", inst["source"]), "price": price,
        "last_bar_jst": last_bar.astimezone(JST).strftime("%Y-%m-%d %H:%M"), "stale": stale_h > 3,
        "atr_d1": atr_d1, "atr_h1": atr_h1, "range_used_pct": float(range_used), "range_left": float(range_left),
        **e, "trend_h1": th1, "trend_h4": th4, "trend_score": ts,
        "levels": levels, "long": long, "short": short, "code_side": side, "score": score,
    }


def rnd(o, digits):
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: rnd(v, 0 if k == "range_used_pct" else 1 if k in ("range_left", "dist_pips") else 2 if k in ("rr", "dist_atr", "lots") else digits)
                for k, v in o.items()}
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
            out.append({"iso": t.isoformat(), "time_jst": t.astimezone(JST).strftime("%m-%d %H:%M"), "currency": ev["country"],
                        "impact": ev["impact"], "title": ev["title"],
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda x: x["iso"])


# ---------------------------------------------------------------- 引く線
LEVEL_JA = {"prev_high": "前日高値", "prev_low": "前日安値", "today_high": "今日の高値", "today_low": "今日の安値",
            "prev_week_high": "前週高値", "prev_week_low": "前週安値", "pivot": "ピボット", "r1": "R1", "r2": "R2",
            "s1": "S1", "s2": "S2", "round_below": "キリ番", "round_above": "キリ番"}


def swings(h4: pd.DataFrame, kind: str, n: int = 2, lookback: int = 90) -> list:
    """4時間足の山・谷（前後n本より高い/低い足）"""
    df = h4.iloc[-lookback:-1]
    vals = df["Low"] if kind == "low" else df["High"]
    out = []
    for i in range(n, len(df) - n):
        w = vals.iloc[i - n:i + n + 1]
        v = float(vals.iloc[i])
        if (kind == "low" and v == float(w.min())) or (kind == "high" and v == float(w.max())):
            out.append((df.index[i], v))
    return out


def trend_line(h1: pd.DataFrame, bias: str, now_utc: datetime) -> dict | None:
    """ロングは切り上がる谷2つ、ショートは切り下がる山2つを結ぶ。現在値が線を割っていれば出さない"""
    if bias not in ("LONG", "SHORT"):
        return None
    h4 = h1.resample("4h", origin="epoch").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    pts = swings(h4, "low" if bias == "LONG" else "high")
    if len(pts) < 2:
        return None
    p2 = pts[-1]
    p1 = next((p for p in reversed(pts[:-1]) if (p[1] < p2[1] if bias == "LONG" else p[1] > p2[1])), None)
    if not p1:
        return None
    hours = (p2[0] - p1[0]).total_seconds() / 3600
    slope = (p2[1] - p1[1]) / hours
    now_val = p2[1] + slope * (pd.Timestamp(now_utc) - p2[0]).total_seconds() / 3600
    price = float(h1["Close"].iloc[-1])
    if (bias == "LONG" and price < now_val) or (bias == "SHORT" and price > now_val):
        return None
    j = lambda t: t.tz_convert(JST).strftime("%m/%d %H:%M")
    return {"from_jst": j(p1[0]), "from_price": p1[1], "to_jst": j(p2[0]), "to_price": p2[1], "now": float(now_val)}


def key_lines(levels: dict, price: float, k: int = 2) -> dict:
    """現在値に近い水平線を上下k本ずつ（同じ価格は1本にまとめる）"""
    seen, above, below = set(), [], []
    for name, v in sorted(levels.items(), key=lambda kv: abs(kv[1] - price)):
        if name.startswith("today_"):   # 朝の時点では今日の高安はまだ意味がない
            continue
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        (above if v >= price else below).append({"name": LEVEL_JA.get(name, name), "price": v})
    return {"above": sorted(above[:k], key=lambda x: x["price"]), "below": sorted(below[:k], key=lambda x: -x["price"])}


# ---------------------------------------------------------------- 朝の方向
def morning_bias(h1: pd.DataFrame) -> dict:
    """前日までの確定足で今日の方向を決める（-5〜+5点）"""
    d1 = to_daily(h1)
    h4 = h1.resample("4h", origin="epoch").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    prev, prev2 = d1.iloc[-2], d1.iloc[-3]
    de20, de75 = float(ema(d1["Close"], 20).iloc[-2]), float(ema(d1["Close"], 75).iloc[-2])
    c = float(prev.Close)
    d_tr = 2 if c > de20 > de75 else -2 if c < de20 < de75 else (1 if de20 > de75 else -1 if de20 < de75 else 0)
    price = float(h1["Close"].iloc[-1])
    h4_tr = trend_score(price, *[float(ema(h4["Close"], n).iloc[-2]) for n in (20, 75, 200)])
    struct = 1 if c > prev2.High else -1 if c < prev2.Low else 0
    total = d_tr + h4_tr + struct
    bias = "LONG" if total >= 3 else "SHORT" if total <= -3 else "WAIT"
    invalid = float(prev.Low) if bias == "LONG" else float(prev.High) if bias == "SHORT" else None
    note = ""
    if bias == "LONG" and price < invalid or bias == "SHORT" and price > invalid:
        note = f"点数は{bias}だが朝の時点で前日の{'安値' if bias == 'LONG' else '高値'}を抜けている"
        bias, invalid = "WAIT", None
    strength = "強" if abs(total) >= 4 else "弱"
    return {"bias": bias, "strength": strength if bias != "WAIT" else "", "note": note, "price": price, "score": total, "d1_trend": d_tr, "h4_trend": h4_tr, "structure": struct,
            "invalid_level": invalid, "prev_day": {k: float(prev[k]) for k in ("Open", "High", "Low", "Close")},
            "ema20_d1": de20, "ema75_d1": de75}


def check_bias(m: dict, h1: pd.DataFrame, since_utc: datetime, trend_h1: int) -> str:
    """朝の方向が今も生きているか：無効 / 弱まった / 維持"""
    if m["bias"] == "WAIT":
        return "WAIT"
    closes = h1.loc[h1.index >= pd.Timestamp(since_utc)]["Close"]
    if m["bias"] == "LONG" and (closes < m["invalid_level"]).any():
        return "無効"
    if m["bias"] == "SHORT" and (closes > m["invalid_level"]).any():
        return "無効"
    if m["bias"] == "LONG" and trend_h1 < 0 or m["bias"] == "SHORT" and trend_h1 > 0:
        return "弱まった"
    return "維持"


def night_side(m: dict, status: str, a: dict) -> tuple[str, str]:
    """夜の推奨：朝の方向と同じ向きか見送りの2択"""
    if status in ("WAIT", "無効"):
        return "WAIT", "朝の方向なし" if status == "WAIT" else "朝の方向が無効化"
    side = m["bias"]
    th1 = a["trend_h1"]
    cand = a["long"] if side == "LONG" else a["short"]
    if (side == "LONG" and th1 < 1) or (side == "SHORT" and th1 > -1):
        return "WAIT", "1時間足が揃っていない"
    if not cand or not cand["rr"] or cand["rr"] < 1.5:
        return "WAIT", "RRが1.5未満"
    if cand.get("entry_mode") == "broken":
        return "WAIT", "価格がゾーンを抜けて損切りに近い（待つ）"
    # 最小ロット(0.01)でも1%リスクを超えるなら見送り（固定ルールを守るため）
    if not cand["lots"] or cand["lots"] < CFG.get("min_lot", 0.01):
        return "WAIT", "最小ロットでも1%リスクを超える（資金に対して損切り幅が大きい）"
    # 今日の残り値幅で利確まで届かないなら見送り
    left_price = a["range_left"] * pip_size(a["symbol"])
    need = abs(cand["tp"] - (cand["zone_lo"] + cand["zone_hi"]) / 2)
    if left_price < need:
        return "WAIT", "今日の残り値幅では利確まで届きにくい"
    return side, ""

# ---------------------------------------------------------------- 時間割・答え合わせ・重複警告
def session_times(trade_day: str) -> list:
    """NY株式オープン・ロンドンフィックスの日本時間（夏冬は自動）"""
    d = datetime.strptime(trade_day, "%Y-%m-%d")
    items = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ZoneInfo("America/New_York")), "NY株式市場オープン（値動きが出やすい）"),
             (datetime(d.year, d.month, d.day, 16, 0, tzinfo=ZoneInfo("Europe/London")), "ロンドンフィックス（急な値動きに注意）")]
    return [{"iso": t.astimezone(timezone.utc).isoformat(), "time_jst": t.astimezone(JST).strftime("%m-%d %H:%M"), "title": n}
            for t, n in items]


def build_schedule(events: list | None, trade_day: str) -> list:
    sch = [dict(x, kind="session") for x in session_times(trade_day)]
    for e in events or []:
        if e["impact"] != "High":
            continue
        t = datetime.fromisoformat(e["iso"]).astimezone(JST)
        sch.append({"iso": e["iso"], "time_jst": e["time_jst"], "kind": "event",
                    "title": f"{e['currency']} {EV_JA.get(e['title'], e['title'])}",
                    "no_entry": f"{t - timedelta(minutes=15):%H:%M}〜{t + timedelta(minutes=15):%H:%M}"})
    return sorted(sch, key=lambda x: x["iso"])


def bars_of_day(h1: pd.DataFrame, trade_day: str) -> pd.DataFrame:
    ny = h1.index.tz_convert("America/New_York")
    lab = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    return h1[lab == pd.Timestamp(trade_day)]


def review_day(state: dict, data: dict) -> dict:
    """前の取引日の答え合わせ：朝の方向（NYクローズまで）と夜の推奨（利確/損切りどちらが先か）"""
    day = state["trade_day"]
    morning, night = [], []
    for name, m in state["pairs"].items():
        if name not in data or m["bias"] == "WAIT":
            continue
        bars = bars_of_day(data[name], day)
        if bars.empty:
            continue
        move = float(bars["Close"].iloc[-1]) - m["price"]
        thr = 0.2 * float(atr(to_daily(data[name])).loc[:pd.Timestamp(day)].iloc[-2])
        sign = 1 if m["bias"] == "LONG" else -1
        res = "判定外" if abs(move) < thr else ("的中" if move * sign > 0 else "外れ")
        morning.append({"symbol": name, "bias": m["bias"], "result": res})
    for r in state.get("night_recs", []):
        if r["symbol"] not in data:
            continue
        bars = bars_of_day(data[r["symbol"]], day)
        bars = bars[bars.index > pd.Timestamp(r["time_utc"])]
        res, entered = "未到達", False
        for _, b in bars.iterrows():
            if not entered:
                entered = b.Low <= r["zone_hi"] if r["side"] == "LONG" else b.High >= r["zone_lo"]
                if not entered:
                    continue
            hit_sl = b.Low <= r["sl"] if r["side"] == "LONG" else b.High >= r["sl"]
            hit_tp = b.High >= r["tp"] if r["side"] == "LONG" else b.Low <= r["tp"]
            if hit_sl:          # 同じ足で両方なら損切り扱い（甘く見ない）
                res = "損切り"; break
            if hit_tp:
                res = "利確"; break
            res = "決着なし"
        night.append({"symbol": r["symbol"], "side": r["side"], "time_jst": r["time_jst"], "result": res})
    return {"trade_day": day, "morning": morning, "night": night}


CCY_JA = {"USD": "ドル", "JPY": "円", "EUR": "ユーロ", "GBP": "ポンド", "AUD": "豪ドル", "XAU": "金"}


def overlap_warnings(pairs: list) -> list:
    """同じ通貨を同じ向きに賭けている推奨の組み合わせ"""
    exp = {}
    for p in pairs:
        side = p["night"]["side"]
        if side == "WAIT":
            continue
        sg = 1 if side == "LONG" else -1
        for ccy, v in ((p["symbol"][:3], sg), (p["symbol"][3:], -sg)):
            exp.setdefault((ccy, v), []).append(p["symbol"])
    out = []
    for (ccy, v), syms in exp.items():
        if len(syms) >= 2 and ccy != "XAU":
            both = "どちらも" if len(syms) == 2 else "いずれも"
            one = "どちらか1つ" if len(syms) == 2 else "どれか1つ"
            out.append(f"{'・'.join(syms)}は{both}「{CCY_JA.get(ccy, ccy)}{'買い' if v > 0 else '売り'}」＝実質同じ賭け。持つなら{one}")
    return out

# ---------------------------------------------------------------- 文章（夜の毎時報告はスクリプトが書く）
EV_JA = {"Unemployment Claims": "新規失業保険申請件数", "Non-Farm Employment Change": "雇用統計(非農業部門)",
         "Unemployment Rate": "失業率", "CPI m/m": "消費者物価指数(前月比)", "Core CPI m/m": "コアCPI(前月比)",
         "Core PCE Price Index m/m": "コアPCE(前月比)", "Federal Funds Rate": "FOMC政策金利",
         "FOMC Statement": "FOMC声明", "FOMC Press Conference": "FOMC会見", "ISM Manufacturing PMI": "ISM製造業",
         "ISM Services PMI": "ISM非製造業", "Retail Sales m/m": "小売売上高", "Advance GDP q/q": "GDP速報",
         "JOLTS Job Openings": "JOLTS求人", "ADP Non-Farm Employment Change": "ADP雇用",
         "CB Consumer Confidence": "消費者信頼感", "Prelim UoM Consumer Sentiment": "ミシガン大学消費者信頼感",
         "PPI m/m": "生産者物価指数", "FOMC Meeting Minutes": "FOMC議事録"}
SIDE_JA = {"LONG": "ロング推奨", "SHORT": "ショート推奨", "WAIT": "見送り"}
BIAS_JA = {"LONG": "ロング", "SHORT": "ショート", "WAIT": "方向なし"}


def bias_label(m: dict, status: str = "") -> str:
    """ロング（強）／ショート（弱）など。夕方以降に弱まったら弱に下げる"""
    if m["bias"] == "WAIT":
        return "方向なし"
    st = "弱" if status == "弱まった" else m.get("strength", "")
    return f"{BIAS_JA[m['bias']]}（{st}）" if st else BIAS_JA[m["bias"]]


def fmt(p: dict, v) -> str:
    return "NA" if v is None else f"{v:.{p.get('digits', 5)}f}"


def schedule_lines(pack: dict, upcoming_only: bool) -> list:
    now = datetime.now(timezone.utc) - timedelta(minutes=15)
    items = [x for x in pack["schedule"] if not upcoming_only or datetime.fromisoformat(x["iso"]) >= now]
    if not items:
        return []
    out = ["［時間割］"]
    for x in items[:5]:
        extra = f"→ {x['no_entry']}は入らない" if x.get("no_entry") else ""
        out.append(f"  {x['time_jst'][6:]} {x['title']}{extra}")
    return out


def review_lines(rv: dict | None) -> list:
    if not rv:
        return []
    m, n = rv["morning"], rv["night"]
    cnt = lambda lst, k: sum(1 for x in lst if x["result"] == k)
    out = [f"［前日({rv['trade_day'][5:]})の答え合わせ］"]
    out.append(f"  朝の方向：的中{cnt(m, '的中')}・外れ{cnt(m, '外れ')}・判定外{cnt(m, '判定外')}" if m else "  朝の方向：方向ありのペアなし")
    if n:
        out.append(f"  夜の推奨{len(n)}件：利確{cnt(n, '利確')}・損切り{cnt(n, '損切り')}・未到達{cnt(n, '未到達')}・決着なし{cnt(n, '決着なし')}")
    return out


def pair_line(p: dict) -> str:
    n, m = p["night"], p["morning"]
    u = pip_unit(p["symbol"])
    left = f"残り値幅 約{p['range_left']}{u}（消化{p['range_used_pct']:.0f}%）"
    label = bias_label(m, n["status"])
    if n["side"] == "WAIT":
        why = n["reason"] if n["status"] == "WAIT" else f"{n['reason']}／朝:{label}→{n['status']}"
        return f"・{p['symbol']}：見送り（{why}）"
    c = p["long"] if n["side"] == "LONG" else p["short"]
    act = "ロング" if n["side"] == "LONG" else "ショート"
    if c["entry_mode"] == "now":
        how = f"今の価格で{act}OK（ゾーン内）"
    else:
        how = (f"{fmt(p, c['zone_hi'] if n['side'] == 'LONG' else c['zone_lo'])}まで"
               f"{'下がったら' if n['side'] == 'LONG' else '上がったら'}{act}（あと約{c['dist_pips']}{u}・今は待つ）")
    return (f"・{p['symbol']} {label}：{how}\n"
            f"  ゾーン {fmt(p, c['zone_lo'])}〜{fmt(p, c['zone_hi'])}／損切 {fmt(p, c['sl'])}／利確 {fmt(p, c['tp'])}"
            f"（RR {c['rr']}）／{c['lots']}lot／撤退ライン {fmt(p, m['invalid_level'])}／{left}")


def line_set(p: dict) -> dict:
    """そのペアで引くべき線の一覧 {キー: 表示文字列}"""
    m, ln = p["morning"], p["lines"]
    out = {}
    for x in ln["horizontal"]["above"] + ln["horizontal"]["below"]:
        out[f"h:{fmt(p, x['price'])}"] = f"水平線 {fmt(p, x['price'])}（{x['name']}）"
    if m.get("invalid_level") is not None:
        out[f"h:{fmt(p, m['invalid_level'])}"] = f"水平線 {fmt(p, m['invalid_level'])}（撤退ライン：1時間足で抜けたら今日の方向は撤回）"
    t = ln["trend"]
    if t:
        out[f"t:{t['from_jst']}|{t['to_jst']}"] = (f"トレンド線 {t['from_jst']} {fmt(p, t['from_price'])} → "
                                                   f"{t['to_jst']} {fmt(p, t['to_price'])}（今は{fmt(p, t['now'])}付近）")
    return out


def draw_lines(pack: dict) -> list:
    """朝だけ：前日から変わった線だけ『引く／消す』を短く指示する"""
    prev = pack.get("prev_lines") or {}
    out = []
    short = lambda v: v.split("（")[0].replace("水平線 ", "").replace("トレンド線 ", "トレンド線")
    for p in pack["pairs"]:
        m = p["morning"]
        cur = line_set(p) if m["bias"] != "WAIT" else {}
        old = prev.get(p["symbol"], {})
        if not cur and not old:
            continue
        head = f"・{p['symbol']} {bias_label(m)}"
        if not cur:
            out.append(f"{head}：昨日の線は全部消してOK")
            continue
        add = {k: v for k, v in cur.items() if k not in old}
        rem = [v for k, v in old.items() if k not in cur]
        if not add and not rem:
            out.append(f"{head}：前日から変更なし（引き直し不要）")
            continue
        out.append(head)
        if old and not set(old) & set(cur):
            out.append("  昨日の線は全部消して、次を引く")
        elif rem:
            out.append("  消す：" + "・".join(short(v) for v in rem))
        hz = [v.replace("水平線 ", "") for k, v in add.items() if k.startswith("h:") and "撤退" not in v]
        if hz:
            out.append("  引く（水平線）：" + "・".join(hz))
        out += [f"  引く：{v}" for v in add.values() if "撤退" in v or v.startswith("トレンド線")]
        c = p["long"] if m["bias"] == "LONG" else p["short"]
        if c:
            out.append(f"  ゾーンの目安：{fmt(p, c['zone_lo'])}〜{fmt(p, c['zone_hi'])}（引かなくてよい・夜の通知で更新）")
    return (["［スマホのMT5で引いておく線］"] + out) if out else []


def build_text(pack: dict) -> str:
    mode, pairs = pack["mode"], pack["pairs"]
    t = pack["generated_jst"][11:]
    lines = []
    if mode == "morning":
        lines.append(f"【朝 {t}　今日の方向】")
        for p in pairs:
            m = p["morning"]
            if m["bias"] == "WAIT":
                lines.append(f"・{p['symbol']}：方向なし（点数{m['score']:+d}{'／' + m['note'] if m.get('note') else ''}）")
            else:
                lines.append(f"・{p['symbol']}：{bias_label(m)}　点数{m['score']:+d}／撤退ライン {fmt(p, m['invalid_level'])}")
        lines += draw_lines(pack)
        lines += schedule_lines(pack, False) + review_lines(pack.get("review"))
    else:
        live = [p for p in pairs if p["night"]["status"] in ("維持", "弱まった")]
        recs = [p for p in pairs if p["night"]["side"] != "WAIT"]
        if mode == "evening":
            lines.append(f"【夕方 {t}　朝の方向の確認】")
            lines.append("結論：" + ("、".join(f"{p['symbol']} {bias_label(p['morning'], p['night']['status'])}・{p['night']['status']}" for p in live)
                                     + " が今夜の候補" if live else "朝の方向が全滅＝今夜は休み"))
        else:
            lines.append(f"【{t} 状況】")
            lines.append("結論：" + ("、".join(f"{p['symbol']} {bias_label(p['morning'], p['night']['status'])}" for p in recs) if recs
                                     else ("今は全ペア見送り（待つ）" if live else "全ペア見送り＝今夜は休み")))
        lines += [pair_line(p) for p in pairs]
        lines += ["⚠ " + w for w in pack.get("warnings", [])]
        lines += schedule_lines(pack, mode == "hourly")
    if pack["errors"]:
        lines.append("※取得エラー：" + " / ".join(pack["errors"]))
    lines.append("※水準は無料データ基準。発注前に自分のチャートで確認")
    return "\n".join(lines)


def push(title: str, text: str):
    topic = os.environ.get("NTFY_TOPIC", "")
    if not topic:
        return
    body = json.dumps({"topic": topic, "title": title, "message": text}).encode()
    urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/", data=body,
                           headers={"Content-Type": "application/json"}), timeout=30)

# ---------------------------------------------------------------- 実行
def pick_mode(now_jst: datetime) -> str:
    forced = os.environ.get("MODE", "")
    if forced:
        return forced
    h = now_jst.hour
    if 5 <= h < 12:
        return "morning"
    if 17 <= h < 20 and not (h == 19 and now_jst.minute >= 45):
        return "evening"
    return "hourly"


def load_data(errors: list) -> tuple[dict, dict]:
    data, rates = {}, {}
    td_key = os.environ.get("TWELVEDATA_API_KEY", "")
    for name, inst in CFG["instruments"].items():
        try:
            if inst["source"] == "twelvedata" and td_key:
                data[name] = fetch_twelvedata_hourly(inst["ticker"], td_key)
            elif inst["source"] == "twelvedata":
                # キー未設定時は先物で代用（現物と数ドル〜数十ドルずれる）
                data[name] = fetch_yf_hourly(inst["fallback_ticker"])
                data[name].attrs["source"] = inst["fallback_ticker"] + "(先物で代用)"
                errors.append(f"{name}: Twelve Dataキー未設定のため先物({inst['fallback_ticker']})で代用")
            else:
                data[name] = fetch_yf_hourly(inst["ticker"])
            rates[name] = float(data[name]["Close"].iloc[-1])
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    return data, rates


def previous_state(trade_day: str) -> dict | None:
    if not os.path.isdir("docs/state"):
        return None
    days = sorted(f[:-5] for f in os.listdir("docs/state") if f.endswith(".json") and f[:-5] < trade_day)
    return json.load(open(f"docs/state/{days[-1]}.json", encoding="utf-8")) if days else None


def main():
    now = datetime.now(timezone.utc)
    now_jst = now.astimezone(JST)
    mode = pick_mode(now_jst)
    trade_day = (now_jst - timedelta(hours=6)).strftime("%Y-%m-%d")   # 深夜1時台は前日扱い
    state_path = f"docs/state/{trade_day}.json"
    errors = []
    data, rates = load_data(errors)

    # 朝の方向：朝は新規作成、それ以外は保存済みを読む（無ければその場で作る）
    if mode != "morning" and os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))
    else:
        state = {"trade_day": trade_day, "created_jst": now_jst.strftime("%Y-%m-%d %H:%M"),
                 "created_utc": now.isoformat(), "pairs": {}, "night_recs": []}
        for name, df in data.items():
            try:
                state["pairs"][name] = rnd(morning_bias(df), CFG["instruments"][name]["digits"])
            except Exception as ex:
                errors.append(f"{name} morning: {ex}")
        if mode != "morning":
            errors.append("朝の方向が未作成だったため、いま作成")
    since = datetime.fromisoformat(state["created_utc"])

    review, prev = None, None
    if mode == "morning":
        prev = previous_state(trade_day)
        if prev:
            try:
                review = review_day(prev, data)
            except Exception as ex:
                errors.append(f"review: {ex}")

    pairs = []
    for name in CFG["instruments"]:
        if name not in data or name not in state["pairs"]:
            continue
        try:
            dg = CFG["instruments"][name]["digits"]
            a = rnd(analyze(name, data[name], rates, now), dg)
            m = state["pairs"][name]
            status = check_bias(m, data[name], since, a["trend_h1"])
            side, reason = night_side(m, status, a)
            a["morning"] = m
            a["night"] = {"status": status, "side": side, "reason": reason}
            a["lines"] = {"horizontal": key_lines(a["levels"], a["price"]),
                          "trend": trend_line(data[name], m["bias"], now)}
            a = rnd(a, dg)
            pairs.append(a)
        except Exception as ex:
            errors.append(f"{name} analyze: {ex}")

    # 夜の推奨を記録（翌朝の答え合わせ用。同じ内容の連続は1件にまとめる）
    if mode == "hourly":
        for p in pairs:
            side = p["night"]["side"]
            if side == "WAIT":
                continue
            c = p["long"] if side == "LONG" else p["short"]
            rec = {"symbol": p["symbol"], "side": side, "zone_lo": c["zone_lo"], "zone_hi": c["zone_hi"],
                   "sl": c["sl"], "tp": c["tp"]}
            last = next((r for r in reversed(state["night_recs"]) if r["symbol"] == p["symbol"]), None)
            if not last or any(last[k] != rec[k] for k in rec):
                state["night_recs"].append(dict(rec, time_utc=now.isoformat(), time_jst=now_jst.strftime("%H:%M")))
    if mode == "morning":
        state["lines"] = {p["symbol"]: line_set(p) for p in pairs if p["morning"]["bias"] != "WAIT"}
    os.makedirs("docs/state", exist_ok=True)
    json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

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
        events = events_window(fetch_calendar(), now, CFG["event_hours"] if mode != "morning" else 20)
    except Exception as ex:
        events = None
        errors.append(f"calendar: {ex}")

    order = {"LONG": 0, "SHORT": 0, "WAIT": 1}
    if mode == "morning":
        pairs.sort(key=lambda p: (p["morning"]["bias"] == "WAIT", -abs(p["morning"]["score"])))
    else:
        pairs.sort(key=lambda p: (order[p["night"]["side"]], p["night"]["status"] not in ("維持", "弱まった"),
                                  -abs(p["morning"]["score"])))
    pack = {"mode": mode, "trade_day": trade_day, "generated_jst": now_jst.strftime("%Y-%m-%d %H:%M"),
            "equity_jpy": CFG["equity_jpy"], "risk_pct": CFG["risk_pct"], "rules_version": CFG["rules_version"],
            "risk_per_trade_jpy": round(CFG["equity_jpy"] * CFG["risk_pct"] / 100),
            "daily_stop": {"losing_streak": CFG["daily_stop_streak"], "loss_jpy": round(CFG["equity_jpy"] * CFG["daily_stop_pct"] / 100)},
            "peak_stop_jpy": round(CFG["equity_jpy"] * 0.15),
            "macro": macro, "events": events, "schedule": build_schedule(events, trade_day),
            "warnings": overlap_warnings(pairs) if mode == "hourly" else [],
            "review": review, "prev_lines": (prev or {}).get("lines", {}), "pairs": pairs, "errors": errors}
    pack["text"] = build_text(pack)
    os.makedirs("docs/archive", exist_ok=True)
    for path in (f"docs/{mode}.json", "docs/datapack.json",
                 f"docs/archive/{trade_day}_{now_jst:%H%M}_{mode}.json"):
        json.dump(pack, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(pack["text"])
    if mode == "hourly":
        try:
            push(f"FX {now_jst:%H:%M}", pack["text"])
        except Exception as ex:
            print("push failed:", ex)
    if not pairs:
        sys.exit(1)


if __name__ == "__main__":
    main()
