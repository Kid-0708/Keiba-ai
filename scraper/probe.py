"""公式サイトの構造調査：robots.txt と、出馬表・オッズ・結果のページを少しだけ保存する（合計20ページ以内）"""
import os, re, sys
from urllib.parse import urljoin
sys.path.insert(0, os.path.dirname(__file__))
from common import fetch, allowed, _session

OUT = os.path.join(os.path.dirname(__file__), "..", "probe")
os.makedirs(OUT, exist_ok=True)
LIMIT = 20
saved = []


def save(name, text):
    p = os.path.join(OUT, name)
    open(p, "w", encoding="utf-8").write(text)
    saved.append(name)
    print("saved", name, len(text))


def robots(base, name):
    try:
        r = _session.get(base + "/robots.txt", timeout=20)
        save(name, f"status={r.status_code}\n{r.text}")
    except Exception as e:
        save(name, f"error={e}")


# ---- 1) robots.txt
robots("https://www.keiba.go.jp", "nar_robots.txt")
robots("https://www.jra.go.jp", "jra_robots.txt")

# ---- 2) 地方競馬（keiba.go.jp）：トップ → 本日のレース一覧 → 出馬表・オッズ・結果を各1〜2ページ
try:
    top = fetch("https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/TopPage", cache=False)
    save("nar_top.html", top)
    links = sorted(set(urljoin("https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/TopPage", h)
                       for h in re.findall(r'href="([^"]+)"', top)))
    save("nar_top_links.txt", "\n".join(links))
    want = ["RaceList", "DebaTable", "Odds", "RaceMarkTable", "Result"]
    picked = []
    for w in want:
        picked += [l for l in links if w in l][:2]
    for l in picked:
        if len(saved) >= LIMIT: break
        if not allowed(l): continue
        html = fetch(l, cache=False)
        save("nar_" + re.sub(r"[^A-Za-z0-9]+", "_", l.split("KeibaWeb/")[-1])[:80] + ".html", html)
        # レース一覧から先のリンクも1段だけたどる
        if "RaceList" in l:
            sub = sorted(set(urljoin(l, h) for h in re.findall(r'href="([^"]+)"', html)))
            save("nar_racelist_links.txt", "\n".join(sub))
            for w in ["DebaTable", "OddsTanFuku", "RaceMarkTable"]:
                for s in [x for x in sub if w in x][:1]:
                    if len(saved) >= LIMIT: break
                    save("nar_" + w + ".html", fetch(s, cache=False))
except Exception as e:
    save("nar_error.txt", repr(e))

# ---- 3) 中央競馬（JRA）：トップと出馬表メニュー（JRAはフォーム送信でページが切り替わる）
try:
    top = fetch("https://www.jra.go.jp/", cache=False)
    save("jra_top.html", top)
    for name, cname in [("jra_entries_menu.html", "pw01dli00/F3"), ("jra_odds_menu.html", "pw15oli00/6D"),
                        ("jra_results_menu.html", "pw01sli00/AF")]:
        if len(saved) >= LIMIT: break
        try:
            save(name, fetch("https://www.jra.go.jp/JRADB/accessD.html" if "dli" in cname else
                             "https://www.jra.go.jp/JRADB/accessO.html" if "oli" in cname else
                             "https://www.jra.go.jp/JRADB/accessS.html", data={"cname": cname}, cache=False))
        except Exception as e:
            save(name + ".error.txt", repr(e))
except Exception as e:
    save("jra_error.txt", repr(e))

print("done", len(saved), "files")
