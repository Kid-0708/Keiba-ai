"""JRAの構造調査（2回目）：開催→レース一覧→レース結果、過去の開催一覧、出馬表・オッズの一覧を数ページだけ保存"""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import fetch

OUT = os.path.join(os.path.dirname(__file__), "..", "probe")
BASE = "https://www.jra.go.jp"
ACT = re.compile(r"doAction\('([^']+)',\s*'([^']+)'\)")


def get(path, cname, name):
    html = fetch(BASE + path, data={"cname": cname}, cache=False)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(html)
    print("saved", name, len(html))
    return html, ACT.findall(html)


# 1) ある開催のレース一覧 → 11Rのレース結果
_, acts = get("/JRADB/accessS.html", "pw01srl10052026050320261114/72", "jra_s_racelist.html")
races = [(p, c) for p, c in acts if c.startswith("pw01sde")]
open(os.path.join(OUT, "jra_s_racelist_links.txt"), "w").write("\n".join(f"{p} {c}" for p, c in acts))
if races:
    get(*races[min(10, len(races) - 1)], "jra_s_result.html")

# 2) 過去の開催を探すページ
_, acts = get("/JRADB/accessS.html", "pw01skl00999999/B3", "jra_s_archive.html")
open(os.path.join(OUT, "jra_s_archive_links.txt"), "w").write("\n".join(f"{p} {c}" for p, c in acts))

# 3) 出馬表とオッズの開催一覧（今週分が出ていれば1レースずつ）
for path, cname, tag, nxt in [("/JRADB/accessD.html", "pw01dli00/F3", "d", "pw01dde"), ("/JRADB/accessO.html", "pw15oli00/6D", "o", "pw15")]:
    html, acts = get(path, cname, f"jra_{tag}_menu.html")
    lists = [(p, c) for p, c in acts if c.startswith(("pw01drl", "pw15orl"))]
    if lists:
        _, acts2 = get(*lists[0], f"jra_{tag}_racelist.html")
        open(os.path.join(OUT, f"jra_{tag}_racelist_links.txt"), "w").write("\n".join(f"{p} {c}" for p, c in acts2))
        det = [(p, c) for p, c in acts2 if c.startswith(nxt) and c not in (cname,)]
        if det:
            get(*det[min(10, len(det) - 1)], f"jra_{tag}_detail.html")
print("done")
