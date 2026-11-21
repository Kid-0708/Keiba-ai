"""出馬表メニューの調査：今週の開催がどう載っているかを保存"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import fetch
import jra
OUT = os.path.join(os.path.dirname(__file__), "..", "probe")
for path, cname, name in [("/JRADB/accessD.html", "pw01dli00/F3", "now_entries_menu.html"), ("/JRADB/accessS.html", "pw01sli00/AF", "now_results_menu.html")]:
    html = fetch("https://www.jra.go.jp" + path, data={"cname": cname}, cache=False)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(html)
    open(os.path.join(OUT, name + ".links.txt"), "w").write("\n".join(f"{p} {c}" for p, c in jra.links(html)))
    print(name, len(html))
