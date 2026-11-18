"""予想結果（data/app_data.json）をアプリ用の docs/data.json に書き出す"""
import datetime as dt, json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "app_data.json")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "data.json")
d = json.load(open(src))
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
meta = dict(kind="meta", generated=d["generated"], updated=now.strftime("%Y-%m-%d %H:%M"),
            today=os.environ.get("TODAY") or now.date().isoformat(), ev_threshold=d["ev_threshold"], combo=d["combo"],
            groups=d["groups"], importance=d["importance"])
recs = [meta, dict(kind="backtest", **d["backtest"])]
ap = os.path.join(ROOT, "data", "analysis.json")
if os.path.exists(ap):
    recs.append(dict(kind="analysis", **json.load(open(ap))))
recs += [dict(kind="upcoming", **r) for r in d["upcoming"]] + [dict(kind="review", **r) for r in d["review"]]
with open(dst, "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in recs) + "\n]\n")
print("exported", dst, len(d["upcoming"]), "races upcoming,", len(d["review"]), "results")
