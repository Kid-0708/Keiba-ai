"""発走前の予想を記録する（更新のたびに上書きし、発走時刻を過ぎたらその時点の予想が「最終予想」として残る）"""
import datetime as dt, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from tickets import marks, tickets, place_probs, BETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
src = os.path.join(ROOT, "data", "app_data.json")
dst = os.path.join(ROOT, "data", "predictions", "predictions.csv")
tdst = os.path.join(ROOT, "data", "predictions", "tickets.csv")
os.makedirs(os.path.dirname(dst), exist_ok=True)
JST = dt.timezone(dt.timedelta(hours=9))
now = dt.datetime.now(JST)
d = json.load(open(src))

rows, trows, ids = [], [], set()
for r in d["upcoming"]:
    try:
        h, m = map(int, (r.get("post") or "23:59").split(":"))
        post = dt.datetime.fromisoformat(r["date"]).replace(hour=h, minute=m, tzinfo=JST)
    except Exception:
        continue
    if now >= post: continue                      # 発走後は記録を変えない
    cond = r["cond"]
    u = [x["umaban"] for x in r["horses"]]
    p = [x["sc"][cond]["p"] for x in r["horses"]]
    od = [x["odds"] for x in r["horses"]]
    mk = marks(u, p, od)
    t2, t3 = place_probs(u, p)
    for x, pp in zip(r["horses"], p):
        rows.append(dict(race_id=r["race_id"], date=r["date"], post=r["post"], snapshot=now.strftime("%Y-%m-%d %H:%M"),
                         umaban=x["umaban"], name=x["name"], p_win=round(pp, 4), p_top2=round(t2[x["umaban"]], 4),
                         p_top3=round(t3[x["umaban"]], 4), odds=x["odds"], mark=mk.get(x["umaban"], ""),
                         odds_known=all(o for o in od)))
    for bet, lst in tickets(u, p).items():
        trows.append(dict(race_id=r["race_id"], bet=bet, tickets="|".join(lst)))
    ids.add(r["race_id"])

def upsert(path, new):
    if not len(new): return
    new = pd.DataFrame(new); new["race_id"] = new.race_id.astype(str)
    old = pd.read_csv(path, dtype={"race_id": str}) if os.path.exists(path) else pd.DataFrame()
    if len(old): old = old[~old.race_id.isin(ids)]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)

upsert(dst, rows); upsert(tdst, trows)
print("予想を記録:", len(ids), "レース")
