import json, datetime
d=json.load(open("app_data.json"))
meta=dict(kind="meta",generated=d["generated"],updated=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M"),
          today="2026-11-18",ev_threshold=d["ev_threshold"],combo=d["combo"],groups=d["groups"],importance=d["importance"])
recs=[meta,dict(kind="backtest",**d["backtest"])]
recs+=[dict(kind="upcoming",**r) for r in d["upcoming"]]+[dict(kind="review",**r) for r in d["review"]]
open("site/data.json","w").write("[\n"+",\n".join(json.dumps(r,ensure_ascii=False,separators=(",",":")) for r in recs)+"\n]\n")
print("ok", meta["updated"])
