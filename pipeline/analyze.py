"""成績の集計と分析：実際の記録（運用開始後）と検証期間（過去半年）について、券種別の的中率・回収率、
条件別の得意・不得意、確率の当てはまり、改善の履歴をまとめる → data/analysis.json"""
import json, os, sys
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
        out["backtest"] = dict(**res, period=f"{B.date.min()} 〜 {B.date.max()}", breakdown=breakdown(B, per), calib=calib(B))
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
        out["live"] = dict(**res, period=f"{L.date.min()} 〜 {L.date.max()}", weekly=weekly, calib=calib(L))
# --- 改善の履歴
hp = os.path.join(ROOT, "data", "improve_history.csv")
if os.path.exists(hp):
    out["improve"] = pd.read_csv(hp).tail(10).to_dict("records")
json.dump(out, open(os.path.join(ROOT, "data", "analysis.json"), "w"), ensure_ascii=False, default=lambda o: None)
print("分析を出力:", list(out))
