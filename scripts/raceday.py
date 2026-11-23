"""今日まだ終わっていない中央のレースがあり、17:45（日本時間）前なら 0 を返す（＝更新を続ける）"""
import datetime as dt, sys
import pandas as pd
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
if not (6 <= now.hour and (now.hour, now.minute) < (17, 45)): sys.exit(1)
try:
    r = pd.read_csv("data/csv/races.csv", dtype={"race_id": str})
    e = pd.read_csv("data/csv/entries.csv", dtype={"race_id": str})
except Exception: sys.exit(1)
ids = set(r[r.date == now.date().isoformat()].race_id)
if not ids: sys.exit(1)
done = set(e[e.race_id.isin(ids) & e.finish.notna()].race_id)
sys.exit(0 if len(ids - done) > 0 else 1)
