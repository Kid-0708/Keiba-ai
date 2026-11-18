"""JRAの構造調査（3回目）：1開催分の全レース結果・全レース出馬表・単勝複勝オッズを1ページずつ保存"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import fetch
OUT = os.path.join(os.path.dirname(__file__), "..", "probe")
for path, cname, name in [("/JRADB/accessS.html", "pw01ses10052026050320261114/7B", "jra_results_all.html"),
                          ("/JRADB/accessD.html", "pw01des10052026050320261114/BF", "jra_entries_all.html"),
                          ("/JRADB/accessO.html", "pw151ou1005202605030120261114Z/C9", "jra_odds_tanfuku.html")]:
    html = fetch("https://www.jra.go.jp" + path, data={"cname": cname}, cache=False)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(html); print("saved", name, len(html))
