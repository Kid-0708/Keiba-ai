"""馬場状態の適性と血統の分析（人気から期待される成績と比べて、どれだけ上回ったか）
使い方: python pipeline/blood_analysis.py   → data/blood_analysis.json と画面への要約
・期待値：同じ人気・同じ頭数帯の馬の3着以内率（オッズに織り込まれた評価）
・「期待との差」がプラス＝人気以上に走った。予想に使えるのは、ここがプラス・マイナスに偏る要素だけ
・血統は、学習期間（古い7割）で父ごとの道悪の成績を出し、検証期間（新しい3割）の道悪のレースで当たるかを確かめる
"""
import json, os
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data")
r = pd.read_csv(f"{D}/csv/races.csv", dtype={"race_id": str})
e = pd.read_csv(f"{D}/csv/entries.csv", dtype={"race_id": str, "horse_id": str})
h = pd.read_csv(f"{D}/csv/horses.csv", dtype={"horse_id": str})
for c in ["sire", "dam_sire"]:
    if c not in h: h[c] = None
df = e.merge(r, on="race_id").merge(h[["horse_id", "sire", "dam_sire"]], on="horse_id", how="left")
df = df[df.finish.notna() & df.popularity.notna()].sort_values(["date", "race_id"]).reset_index(drop=True)
df["top3"] = (df.finish <= 3).astype(int)
df["heavy"] = df.track_condition.isin(["稍重", "重", "不良"])
df["fb"] = pd.cut(df.num_horses, [0, 10, 14, 18], labels=["s", "m", "l"]).astype(str)
base = df.groupby(["popularity", "fb"]).top3.mean()
df["exp3"] = [base.get((p, f), np.nan) for p, f in zip(df.popularity, df.fb)]
df["ex"] = df.top3 - df.exp3
df["fr"] = (df.finish - 1) / (df.num_horses - 1)
out = {"races": int(df.race_id.nunique()), "heavy_races": int(df[df.heavy].race_id.nunique())}


def table(g):
    t = g.agg(n=("ex", "size"), top3=("top3", "mean"), exp=("exp3", "mean"), ex=("ex", "mean"))
    t["se"] = np.sqrt(t.exp * (1 - t.exp) / t.n)
    return [dict(label=str(k), n=int(v.n), top3=round(v.top3 * 100, 1), exp=round(v.exp * 100, 1),
                 ex=round(v.ex * 100, 1), sig=bool(abs(v.ex) > 2 * v.se)) for k, v in t.iterrows()]


# 1) 本馬の道悪適性（そのレースより前の成績だけで計算）
hist, apt, hr = {}, [], []
for hid, hv, fr in zip(df.horse_id, df.heavy, df.fr):
    H = hist.get(hid, []); a = [x for c, x in H if c]; b = [x for c, x in H if not c]
    hr.append(len(a)); apt.append(np.mean(a) - np.mean(b) if len(a) >= 2 and len(b) >= 2 else np.nan)
    hist.setdefault(hid, []).append((hv, fr))
df["apt"], df["hv_runs"] = apt, hr
s = df[df.heavy & df.apt.notna()]
out["horse_apt"] = table(s.groupby(pd.cut(s.apt, [-1, -.15, -.05, .05, .15, 1],
                                          labels=["道悪でかなり好走", "やや好走", "差なし", "やや凡走", "かなり凡走"]), observed=True))
out["heavy_experience"] = table(df[df.heavy].groupby(pd.cut(df[df.heavy].hv_runs, [-1, 0, 1, 3, 99],
                                                             labels=["道悪は初めて", "1戦", "2〜3戦", "4戦以上"]), observed=True))
out["first_heavy_share"] = round(float((df[df.heavy].hv_runs <= 1).mean()) * 100, 1)

# 2) 芝・ダート別の、馬場による走破タイムの変化（勝ち馬・1000mあたり・良との差）
def sec(t):
    try:
        m, x = str(t).split(":"); return int(m) * 60 + float(x)
    except Exception:
        try: return float(t)
        except Exception: return np.nan
w = df[df.finish == 1].copy(); w["ts"] = w.time.map(sec) / w.distance * 1000
ref = w[w.track_condition == "良"].groupby(["surface", "course", "distance"]).ts.mean()
w["d"] = w.ts - [ref.get(k, np.nan) for k in zip(w.surface, w.course, w.distance)]
out["time_shift"] = {f"{s_}・{c}": round(float(v), 2) for (s_, c), v in w.groupby(["surface", "track_condition"]).d.mean().items() if c != "良"}

# 3) 血統（父・母の父）
known = df.sire.notna() & (df.sire != "不明")
out["sire_known_share"] = round(float(known.mean()) * 100, 1)
cut = df.date.iloc[int(len(df) * 0.7)]          # 日付順に並んでいるので7割の位置で区切る
def sire_eval(col, k=30):
    kn = df[df[col].notna() & (df[col] != "不明")]
    if kn.heavy.sum() < 2000: return None
    tr, te = kn[kn.date <= cut], kn[kn.date > cut]
    def score(d):   # 道悪での期待との差 − 良での期待との差（縮小推定）
        g = d.groupby([col, "heavy"]).ex.agg(["sum", "count"]).unstack(fill_value=0)
        hv = g[("sum", True)] / (g[("count", True)] + k); gd = g[("sum", False)] / (g[("count", False)] + k)
        return (hv - gd), g[("count", True)]
    sc, n = score(tr)
    t = te[te.heavy].copy(); t["sc"] = t[col].map(sc)
    t = t[t.sc.notna()]
    res = dict(test_runs=int(len(t)),
               bands=table(t.groupby(pd.qcut(t.sc, 3, labels=["道悪が苦手な血統", "平均的", "道悪が得意な血統"], duplicates="drop"), observed=True)) if len(t) > 300 else [])
    full, nf = score(kn)
    top = pd.DataFrame({"score": full, "n": nf})
    top = top[top.n >= 40].sort_values("score")
    res["best"] = [dict(name=i, ex=round(v.score * 100, 1), n=int(v.n)) for i, v in top.tail(10)[::-1].iterrows()]
    res["worst"] = [dict(name=i, ex=round(v.score * 100, 1), n=int(v.n)) for i, v in top.head(10).iterrows()]
    return res
out["sire"] = sire_eval("sire")
out["dam_sire"] = sire_eval("dam_sire")
json.dump(out, open(f"{D}/blood_analysis.json", "w"), ensure_ascii=False, indent=1)

print(f"レース {out['races']}（うち道悪 {out['heavy_races']}）／ 血統が分かっている出走 {out['sire_known_share']}%")
for k, t in [("本馬の過去の道悪成績 → 道悪での成績", "horse_apt"), ("道悪の経験 → 道悪での成績", "heavy_experience")]:
    print("\n##", k); [print(f"  {x['label']:<10} {x['n']:>6}頭  3着内 {x['top3']}%  期待 {x['exp']}%  差 {x['ex']:+}{' *' if x['sig'] else ''}") for x in out[t]]
print("\n## 馬場によるタイム変化（秒/1000m、良との差）", out["time_shift"])
for col in ["sire", "dam_sire"]:
    v = out[col]
    if not v: print(f"\n## {col}: データがまだ足りません"); continue
    print(f"\n## {col}: 学習期間で道悪が得意とされた血統は、検証期間の道悪で走ったか（{v['test_runs']}頭）")
    [print(f"  {x['label']:<10} {x['n']:>6}頭  3着内 {x['top3']}%  期待 {x['exp']}%  差 {x['ex']:+}{' *' if x['sig'] else ''}") for x in v["bands"]]
    print("  得意:", "、".join(f"{x['name']}({x['ex']:+})" for x in v["best"][:6]))
    print("  苦手:", "、".join(f"{x['name']}({x['ex']:+})" for x in v["worst"][:6]))
