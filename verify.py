"""ブリーフィング本文の数字が、すべてデータパックに存在するか機械チェックする。
使い方: python verify.py briefing.md docs/datapack.json
時刻(20:30)・日付・パーセント記号付きの比率など、データパック由来でない数字は許可リストで除外する。
"""
import json, re, sys

def collect(o, out):
    if isinstance(o, dict):
        for v in o.values(): collect(v, out)
    elif isinstance(o, list):
        for v in o: collect(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.add(o)
    elif isinstance(o, str):
        for m in re.findall(r"-?\d+(?:\.\d+)?", o): out.add(float(m))

text = open(sys.argv[1], encoding="utf-8").read()
pack = json.load(open(sys.argv[2], encoding="utf-8"))
nums = set(); collect(pack, nums)
# 時刻・日付・見出し番号などを除去してから数字を拾う
clean = re.sub(r"\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}|\d{1,2}-\d{2}|\d{1,2}/\d{1,2}|^\s*\d+\.\s", " ", text, flags=re.M)
bad = []
for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", clean):
    s = m.group(); v = float(s)
    if abs(v) <= 20 and "." not in s:      # 小さい整数(件数・点数・分)は対象外
        continue
    if not any(abs(v - n) < 1e-9 for n in nums):
        bad.append(s)
if bad:
    print("NG: データパックにない数字:", sorted(set(bad))); sys.exit(1)
print("OK")
