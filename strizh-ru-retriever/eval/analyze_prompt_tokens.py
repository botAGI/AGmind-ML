#!/usr/bin/env python3
"""Разбор results-prompt-tokens-*.txt: связь TTFT с числом токенов промпта.

Числа в шапке лога и в статье считаются ЭТИМ скриптом, а не руками. Печатает:
  - подгонку по всем сырым замерам (то, что вынесено в статью);
  - подгонку по свёрнутым точкам (повторы усреднены) и без точки с высоким рычагом;
  - помодельные подгонки и расхождение прямых в наблюдаемом диапазоне;
  - попарные разности токенов и знаковый тест;
  - запросы, где обе модели дали одинаковый промпт (естественный контроль).

args: путь к results-prompt-tokens-*.txt
"""
import json, math, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results-prompt-tokens-strix-2026-07-28.txt"
blocks, cur = {}, None
for line in open(path, encoding="utf-8"):
    if line.startswith("=="): cur = line.strip("= \n").split()[0]
    elif line.startswith("{") and cur: blocks[cur] = json.loads(line)
(na, a), (nb, b) = list(blocks.items())[:2]

def ols(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    icept = my - slope*mx
    sst = sum((y-my)**2 for y in ys)
    ssr = sum((y-(icept+slope*x))**2 for x, y in zip(xs, ys))
    return icept, slope, 1 - ssr/sst

raw = [(r["prompt_n"], r["ttft_ms"]) for r in a["rows"] + b["rows"]]
i, s, r2 = ols(raw)
print(f"сырые {len(raw)} замеров:      TTFT = {i:.1f} + {s:.4f}*n, R2 = {r2:.3f}")

coll = {}
for n, y in raw: coll.setdefault(n, []).append(y)
pts = [(n, statistics.mean(v)) for n, v in coll.items()]
i, s, r2 = ols(pts)
print(f"свёрнутые {len(pts)} точек:      TTFT = {i:.1f} + {s:.4f}*n, R2 = {r2:.3f}")
lo = min(p[0] for p in pts)
i2, s2, r22 = ols([p for p in pts if p[0] != lo])
print(f"без точки {lo} ток. (рычаг): TTFT = {i2:.1f} + {s2:.4f}*n, R2 = {r22:.3f}")

fits = {}
for name, blk in ((na, a), (nb, b)):
    fits[name] = ols([(r["prompt_n"], r["ttft_ms"]) for r in blk["rows"]])
    print(f"{name:10}:               TTFT = {fits[name][0]:.1f} + {fits[name][1]:.4f}*n, R2 = {fits[name][2]:.3f}")
xs = [p[0] for p in pts]
gap = [abs((fits[na][0]+fits[na][1]*x) - (fits[nb][0]+fits[nb][1]*x)) for x in (min(xs), max(xs))]
print(f"расхождение помодельных прямых в диапазоне {min(xs)}-{max(xs)} ток.: {min(gap):.0f}-{max(gap):.0f} мс")

pa, pb = {}, {}
for r in a["rows"]: pa.setdefault(r["q"], r["prompt_n"])
for r in b["rows"]: pb.setdefault(r["q"], r["prompt_n"])
d = [pa[q]-pb[q] for q in pa]; nz = [x for x in d if x]; pos = sum(1 for x in nz if x > 0)
pv = min(1.0, 2*sum(math.comb(len(nz), k) for k in range(pos, len(nz)+1))/2**len(nz))
print(f"попарные разности токенов: {d}")
print(f"медиана {statistics.median(d):+.0f}, знаковый тест {pos}/{len(nz)} ненулевых, p = {pv:.2f}")

for q in pa:
    if pa[q] == pb[q]:
        ta = statistics.mean(r["ttft_ms"] for r in a["rows"] if r["q"] == q)
        tb = statistics.mean(r["ttft_ms"] for r in b["rows"] if r["q"] == q)
        print(f"одинаковый промпт ({pa[q]} ток.): TTFT {ta:.0f} против {tb:.0f} мс, разница {ta-tb:+.0f}")
