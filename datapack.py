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
        "symbol": name, "digits": inst["digits"], "source": h1.attrs.get("source", inst["source"]), "price": price,
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
    return {"bias": bias, "note": note, "score": total, "d1_trend": d_tr, "h4_trend": h4_tr, "structure": struct,
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
    return side, ""

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


def hourly_text(pack: dict) -> str:
    now = pack["generated_jst"][11:]
    lines = [f"【{now} 状況】"]
    for p in pack["pairs"]:
        n = p["night"]
        head = f"{p['symbol']}：{SIDE_JA[n['side']]}"
        if n["side"] == "WAIT":
            why = n["reason"] if n["status"] == "WAIT" else f"{n['reason']}／朝:{BIAS_JA[p['morning']['bias']]}→{n['status']}"
            lines.append(f"・{head}（{why}）")
        else:
            c = p["long"] if n["side"] == "LONG" else p["short"]
            d = p.get("digits", 5)
            f = lambda v: f"{v:.{d}f}"
            lines.append(f"・{head} ゾーン {f(c['zone_lo'])}〜{f(c['zone_hi'])}／損切 {f(c['sl'])}／利確 {f(c['tp'])}"
                         f"（RR {c['rr']}）／{c['lots']}lot")
    nxt = [e for e in (pack["events"] or []) if e["impact"] == "High"]
    if nxt:
        lines.append("重要指標：" + "、".join(f"{e['time_jst'][6:]} {e['currency']} {EV_JA.get(e['title'], e['title'])}" for e in nxt[:3])
                     + "（前後15分は入らない）")
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


def main():
    now = datetime.now(timezone.utc)
    now_jst = now.astimezone(JST)
    mode = pick_mode(now_jst)
    trade_day = (now_jst - timedelta(hours=6)).strftime("%Y-%m-%d")   # 深夜1時台は前日扱い
    state_path = f"docs/state/{trade_day}.json"
    errors, data, rates = [], {}, {}
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

    # 朝の方向：朝は新規作成、それ以外は保存済みを読む（無ければその場で作る）
    if mode != "morning" and os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))
    else:
        state = {"created_jst": now_jst.strftime("%Y-%m-%d %H:%M"), "created_utc": now.isoformat(), "pairs": {}}
        for name, df in data.items():
            try:
                state["pairs"][name] = rnd(morning_bias(df), CFG["instruments"][name]["digits"])
            except Exception as ex:
                errors.append(f"{name} morning: {ex}")
        if mode != "morning":
            errors.append("朝の方向が未作成だったため、いま作成")
        os.makedirs("docs/state", exist_ok=True)
        json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    since = datetime.fromisoformat(state["created_utc"])

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
            pairs.append(a)
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
        events = events_window(fetch_calendar(), now, CFG["event_hours"] if mode != "morning" else 20)
    except Exception as ex:
        events = None
        errors.append(f"calendar: {ex}")

    order = {"LONG": 0, "SHORT": 0, "WAIT": 1}
    pairs.sort(key=lambda p: (order[p["night"]["side"]], -abs(p["morning"]["score"])))
    pack = {"mode": mode, "trade_day": trade_day, "generated_jst": now_jst.strftime("%Y-%m-%d %H:%M"),
            "equity_jpy": CFG["equity_jpy"], "risk_pct": CFG["risk_pct"], "rules_version": CFG["rules_version"],
            "macro": macro, "events": events, "pairs": pairs, "errors": errors}
    pack["text"] = hourly_text(pack)
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
