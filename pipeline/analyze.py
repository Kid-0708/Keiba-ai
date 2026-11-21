"""成績の集計と分析：実際の記録（運用開始後）と検証期間（過去半年）について、券種別の的中率・回収率、
条件別の得意・不得意、確率の当てはまり、改善の履歴をまとめる → data/analysis.json"""
import itertools, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from tickets import tickets, marks, hit_return, BETS, BET_NAME

ROOT = os.path.join(os.path.dirname(__file__), "..")
C = os.path.join(ROOT, "data", "csv")
races = pd.read_csv(os.path.join(C, "races.csv"), dtype={"race_id": str})
ent = pd.read_csv(os.path.join(C, "entries.csv"), dtype={"race_id": str})
pp = os.path.join(C, "payouts.csv")
pay = pd.read_csv(pp, dtype={"race_id": str, "combo": str}) if os.path.exists(pp) else pd.DataFrame(columns=["race_id", "bet", "combo", "yen"])
PAY = {rid: {(b, c): y for b, c, y in zip(g.bet, g.combo, g.yen)} for rid, g in pay.groupby("race_id")}
FIN = {rid: dict(zip(g.umaban, g.finish)) for rid, g in ent[ent.finish.notna()].groupby("race_id")}


def evaluate(P, T=None):
    """P: 予想（race_id, umaban, p_win, odds）。T: 記録済みの買い目（なければ確率から作る）"""
    P = P[P.race_id.isin(PAY.keys()) & P.race_id.isin(FIN.keys())]
    rids = sorted(P.race_id.unique())
    if not rids: return None
    acc = {b: dict(n=0, hits=0, invest=0, ret=0) for b in BETS}
    hon = dict(n=0, win=0, top2=0, top3=0)
    per_race = []
    for rid, g in P.groupby("race_id"):
        u = list(g.umaban.astype(int)); p = list(g.p_win)
        if T is not None and rid in T: tk = T[rid]
        else: tk = tickets(u, p)
        fin = FIN[rid]; best = u[int(np.argmax(p))]; f = fin.get(best)
        hon["n"] += 1; hon["win"] += f == 1; hon["top2"] += (f or 99) <= 2; hon["top3"] += (f or 99) <= 3
        row = dict(race_id=rid, hon_fin=f)
        for b in BETS:
            lst = tk.get(b, [])
            r = hit_return(b, lst, PAY[rid])
            acc[b]["n"] += 1; acc[b]["invest"] += 100 * len(lst); acc[b]["ret"] += r; acc[b]["hits"] += r > 0
            row[b] = r - 100 * len(lst)
        per_race.append(row)
    bets = [dict(bet=b, name=BET_NAME[b], races=a["n"], hits=int(a["hits"]), hit_rate=round(a["hits"] / a["n"], 3),
                 invest=int(a["invest"]), ret=int(a["ret"]), roi=round(a["ret"] / a["invest"] * 100, 1) if a["invest"] else None)
            for b, a in acc.items()]
    h = dict(races=hon["n"], win=round(hon["win"] / hon["n"], 3), top2=round(hon["top2"] / hon["n"], 3), top3=round(hon["top3"] / hon["n"], 3))
    return dict(races=len(rids), honmei=h, bets=bets), pd.DataFrame(per_race)


def breakdown(P, per_race):
    """条件別：◎の勝率と単勝回収率（30レース以上ある条件だけ）"""
    R = races.set_index("race_id")
    df = per_race.merge(R, left_on="race_id", right_index=True, how="left")
    df["dist_band"] = pd.cut(df.distance, [0, 1400, 1800, 2200, 4000], labels=["短距離(〜1400m)", "マイル(〜1800m)", "中距離(〜2200m)", "長距離(2300m〜)"]).astype(str)
    df["field"] = pd.cut(df.num_horses, [0, 10, 14, 18], labels=["少頭数(〜10頭)", "中頭数(11〜14頭)", "多頭数(15頭〜)"]).astype(str)
    out = []
    for col, label in [("course", "競馬場"), ("surface", "芝・ダート"), ("dist_band", "距離"), ("grade", "クラス"), ("track_condition", "馬場"), ("field", "頭数")]:
        for v, g in df.groupby(col):
            if len(g) < 30 or v in ("nan", None): continue
            out.append(dict(kind=label, value=str(v), races=int(len(g)), win=round(float((g.hon_fin == 1).mean()), 3),
                            roi=round(float((g.tan + 100).sum() / (100 * len(g)) * 100), 1)))
    return out


def top5(P):
    """AIの上位5頭：順位ごとの3着以内率、1〜3着が全部上位5頭に入った割合、5頭を使った買い方の回収率"""
    P = P[P.race_id.isin(PAY.keys()) & P.race_id.isin(FIN.keys())]
    if not len(P): return None
    rank_hit = np.zeros(5); n = 0; all3 = 0
    strat = {k: dict(n=0, hits=0, invest=0, ret=0) for k in ["tribox", "triobox", "tri_1fix", "tri_3x5"]}
    pts = {"tribox": 60, "triobox": 10, "tri_1fix": 12, "tri_3x5": 36}
    for rid, g in P.groupby("race_id"):
        g = g.sort_values("p_win", ascending=False)
        if len(g) < 6: continue
        top = list(g.umaban.astype(int)[:5]); fin = FIN[rid]
        f3 = [u for u, f in fin.items() if f and f <= 3]
        order = [u for u, f in sorted(fin.items(), key=lambda x: x[1]) if f and f <= 3]
        if len(order) < 3: continue
        n += 1
        for i, u in enumerate(top): rank_hit[i] += u in f3
        inside = all(u in top for u in order); all3 += inside
        pay = PAY[rid]
        tri = pay.get(("tri", "-".join(map(str, order))), 0)
        trio = pay.get(("trio", "-".join(map(str, sorted(order)))), 0)
        hit = {"tribox": inside, "triobox": inside, "tri_1fix": inside and order[0] == top[0],
               "tri_3x5": inside and order[0] in top[:3]}
        ret = {"tribox": tri, "triobox": trio, "tri_1fix": tri, "tri_3x5": tri}
        for k in strat:
            strat[k]["n"] += 1; strat[k]["invest"] += 100 * pts[k]
            if hit[k]: strat[k]["hits"] += 1; strat[k]["ret"] += ret[k]
    if not n: return None
    names = {"tribox": "3連単 5頭BOX（60点）", "triobox": "3連複 5頭BOX（10点）",
             "tri_1fix": "3連単 1位固定→上位5頭（12点）", "tri_3x5": "3連単 上位3頭1着→上位5頭（36点）"}
    return dict(races=n, rank_top3=[round(float(x / n), 3) for x in rank_hit], all_in_top5=round(all3 / n, 3),
                strategies=[dict(name=names[k], points=pts[k], hit_rate=round(v["hits"] / v["n"], 3), invest=v["invest"], ret=v["ret"],
                                 roi=round(v["ret"] / v["invest"] * 100, 1), avg_pay=int(v["ret"] / v["hits"]) if v["hits"] else None)
                            for k, v in strat.items()])


def calib(P):
    P = P[P.finish.notna()] if "finish" in P else P
    P = P.assign(win=(P.finish == 1).astype(int))
    P["bin"] = pd.cut(P.p_win, [0, .05, .1, .2, .3, 1], labels=["〜5%", "5〜10%", "10〜20%", "20〜30%", "30%〜"])
    return [dict(bin=str(b), n=int(len(g)), pred=round(float(g.p_win.mean()), 3), actual=round(float(g.win.mean()), 3))
            for b, g in P.groupby("bin", observed=True)]


out = {}
# --- 検証期間（過去半年・学習に使っていない期間）
bp = os.path.join(ROOT, "data", "backtest_preds.csv")
if os.path.exists(bp):
    B = pd.read_csv(bp, dtype={"race_id": str})
    ev = evaluate(B)
    if ev:
        res, per = ev
        out["backtest"] = dict(**res, period=f"{B.date.min()} 〜 {B.date.max()}", breakdown=breakdown(B, per), calib=calib(B), top5=top5(B))
# --- 実際の記録（発走前に記録した予想）
lp = os.path.join(ROOT, "data", "predictions", "predictions.csv")
tp = os.path.join(ROOT, "data", "predictions", "tickets.csv")
if os.path.exists(lp):
    L = pd.read_csv(lp, dtype={"race_id": str})
    T = {}
    if os.path.exists(tp):
        for rid, g in pd.read_csv(tp, dtype={"race_id": str}).groupby("race_id"):
            T[rid] = {b: str(t).split("|") for b, t in zip(g.bet, g.tickets)}
    L["finish"] = [FIN.get(r, {}).get(u) for r, u in zip(L.race_id, L.umaban)]
    ev = evaluate(L, T)
    if ev:
        res, per = ev
        per = per.merge(L.drop_duplicates("race_id")[["race_id", "date"]], on="race_id")
        per["week"] = pd.to_datetime(per.date).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
        weekly = [dict(week=w, races=int(len(g)), win=round(float((g.hon_fin == 1).mean()), 3),
                       tan=int(g.tan.sum()), all_bets=int(sum(g[b].sum() for b in BETS))) for w, g in per.groupby("week")]
        out["live"] = dict(**res, period=f"{L.date.min()} 〜 {L.date.max()}", weekly=weekly, calib=calib(L), top5=top5(L))
# --- 改善の履歴
hp = os.path.join(ROOT, "data", "improve_history.csv")
if os.path.exists(hp):
    out["improve"] = pd.read_csv(hp).tail(10).to_dict("records")
json.dump(out, open(os.path.join(ROOT, "data", "analysis.json"), "w"), ensure_ascii=False, default=lambda o: None)
print("分析を出力:", list(out))


# ================= 回収率100%を目指す買い方の検証（前半で選び、後半で確かめる） =================
def strategy_search(B):
    B = B[B.race_id.isin(PAY.keys()) & B.race_id.isin(FIN.keys()) & B.odds.notna()].copy()
    if not len(B): return None
    inv = 1 / B.odds
    B["pmk"] = inv / inv.groupby(B.race_id).transform("sum")
    B["pm"] = B.p_model / B.groupby("race_id").p_model.transform("sum")
    B["ratio"] = B.pm / B.pmk
    B["rank"] = B.groupby("race_id").p_win.rank(ascending=False, method="first")
    dates = sorted(B.date.unique()); mid = dates[len(dates) // 2]
    races = {rid: g.sort_values("rank") for rid, g in B.groupby("race_id")}
    s2 = lambda a, b: "-".join(map(str, sorted([int(a), int(b)])))
    s3 = lambda *x: "-".join(map(str, sorted(map(int, x))))

    def run(rule):
        """rule(g) → [(bet, combo)] を100円ずつ"""
        res = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}          # bets, invest, return, hits(races)
        for rid, g in races.items():
            tk = rule(g)
            if not tk: continue
            half = 0 if g.date.iloc[0] < mid else 1
            pay = PAY[rid]; ret = sum(pay.get(t, 0) for t in tk)
            r = res[half]; r[0] += 1; r[1] += 100 * len(tk); r[2] += ret; r[3] += ret > 0
        return res

    rules = {}
    top = lambda g, k: list(g.umaban.astype(int)[:k])
    rules["ワイド 上位5頭BOX（10点）"] = lambda g: [("wide", s2(a, b)) for i, a in enumerate(top(g, 5)) for b in top(g, 5)[i + 1:]]
    rules["馬連 上位5頭BOX（10点）"] = lambda g: [("umaren", s2(a, b)) for i, a in enumerate(top(g, 5)) for b in top(g, 5)[i + 1:]]
    rules["ワイド 1位→2〜5位（4点）"] = lambda g: [("wide", s2(top(g, 1)[0], b)) for b in top(g, 5)[1:]]
    rules["3連複 1位軸→2〜5位（6点）"] = lambda g: [("trio", s3(top(g, 1)[0], a, b)) for i, a in enumerate(top(g, 5)[1:]) for b in top(g, 5)[i + 2:]]
    rules["3連複 上位5頭BOX（10点）"] = lambda g: [("trio", s3(*c)) for c in itertools.combinations(top(g, 5), 3)]
    rules["複勝 1位（1点）"] = lambda g: [("fuku", str(top(g, 1)[0]))]
    for rt in [1.3, 1.6, 2.0]:
        for pop in [4, 6, 8]:
            def val(g, rt=rt, pop=pop): return g[(g.ratio >= rt) & (g.popularity >= pop) & (g.pm >= 0.03)]
            rules[f"穴馬の単勝（評価がオッズの{rt}倍以上・{pop}番人気以下）"] = lambda g, v=val: [("tan", str(int(u))) for u in v(g).umaban]
            rules[f"穴馬の複勝（評価がオッズの{rt}倍以上・{pop}番人気以下）"] = lambda g, v=val: [("fuku", str(int(u))) for u in v(g).umaban]
            rules[f"ワイド 1位－穴馬（評価がオッズの{rt}倍以上・{pop}番人気以下）"] = lambda g, v=val: [("wide", s2(top(g, 1)[0], u)) for u in v(g).umaban if u != top(g, 1)[0]]
    for sr in [1, 2]:
        rules[f"タイム指数{sr}位以内なのに6番人気以下の複勝"] = lambda g, sr=sr: [("fuku", str(int(u))) for u in g[(g.spd_rank <= sr) & (g.popularity >= 6)].umaban]
        rules[f"タイム指数{sr}位以内なのに6番人気以下の単勝"] = lambda g, sr=sr: [("tan", str(int(u))) for u in g[(g.spd_rank <= sr) & (g.popularity >= 6)].umaban]

    out = []
    for name, rule in rules.items():
        r = run(rule)
        a, b = r[0], r[1]
        if a[0] < 40 or b[0] < 20: continue
        out.append(dict(name=name, races1=a[0], roi1=round(a[2] / a[1] * 100, 1), hit1=round(a[3] / a[0], 3),
                        races2=b[0], roi2=round(b[2] / b[1] * 100, 1), hit2=round(b[3] / b[0], 3),
                        roi_all=round((a[2] + b[2]) / (a[1] + b[1]) * 100, 1)))
    out.sort(key=lambda x: -x["roi1"])            # 前半の成績で順位を付ける（後半はまだ見ていない扱い）
    return dict(split=f"前半 {dates[0]}〜{mid}／後半 {mid}〜{dates[-1]}", results=out)


def top5_dist(B):
    B = B[B.race_id.isin(FIN.keys())]
    cnt = {0: 0, 1: 0, 2: 0, 3: 0}; n = 0
    for rid, g in B.groupby("race_id"):
        top = set(g.sort_values("p_win", ascending=False).umaban.astype(int)[:5])
        f3 = [u for u, f in FIN[rid].items() if f and f <= 3]
        if len(f3) < 3: continue
        cnt[min(3, sum(u in top for u in f3))] += 1; n += 1
    return dict(races=n, dist=[round(cnt[k] / n, 3) for k in range(4)]) if n else None


if os.path.exists(bp):
    B = pd.read_csv(bp, dtype={"race_id": str})
    out.setdefault("backtest", {})["search"] = strategy_search(B)
    out["backtest"]["top5_dist"] = top5_dist(B)
    json.dump(out, open(os.path.join(ROOT, "data", "analysis.json"), "w"), ensure_ascii=False, default=lambda o: None)
    print("買い方の検証を追加")
