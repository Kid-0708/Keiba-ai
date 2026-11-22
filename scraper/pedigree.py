"""血統（父・母・母の父）を公式サイトの競走馬ページから集める
使い方:
  python scraper/pedigree.py upcoming 8      ← 今週出走する馬だけ（最大8分）。予想の更新のたびに実行
  python scraper/pedigree.py all 330         ← 過去の出走馬も含めて（最大330分）。平日の夜に実行
競走馬ページの場所（cname）は、出馬表・レース結果のページにある馬名のリンクから集める。
"""
import os, re, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import jra, store
from bs4 import BeautifulSoup

mode = sys.argv[1] if len(sys.argv) > 1 else "upcoming"
budget = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 8 * 60
since = sys.argv[3] if len(sys.argv) > 3 else "2024-11"
t0 = time.time()
left = lambda: budget - (time.time() - t0)
HLINK = re.compile(r"(pw01dud\d{2}(\d{10})/[0-9A-F]{2})")
PROBE = os.path.join(os.path.dirname(__file__), "..", "probe", "jra_horse_sample.html")


def harvest(html, out):
    for cname, hid in HLINK.findall(html or ""):
        out.setdefault(hid, cname)


def clean(s):
    s = re.sub(r"[\s　]+", " ", s or "").strip(" ：:")
    s = re.sub(r"[（(].*$", "", s).strip()           # 「（米）」などの産地表記を外す
    return s or None


def parse_pedigree(html):
    """競走馬ページ → {sire, dam, dam_sire}。ページの作りが変わっても読めるよう、いくつかの方法を順に試す"""
    soup = BeautifulSoup(html, "lxml")
    out = {}
    # 1) 「父」「母」「母の父」という見出しの、すぐ次の要素
    for lab, key in [("父", "sire"), ("母", "dam"), ("母の父", "dam_sire"), ("母父", "dam_sire")]:
        if out.get(key): continue
        for el in soup.find_all(["th", "dt", "span", "div", "td", "p"]):
            if el.get_text(strip=True).rstrip("：:") == lab:
                nx = el.find_next_sibling()
                v = clean(nx.get_text(" ")) if nx else None
                if v and v != lab: out[key] = v; break
    # 2) 「父：〇〇」のような書き方
    text = soup.get_text(" ")
    pats = {"sire": r"(?<![の母])父\s*[：:]\s*([^\s　/／]+)", "dam": r"(?<!の)母\s*[：:]\s*([^\s　/／]+)", "dam_sire": r"母の?父\s*[：:]\s*([^\s　/／]+)"}
    for key, pat in pats.items():
        if not out.get(key):
            m = re.search(pat, text)
            if m: out[key] = clean(m.group(1))
    return out


# ---- 1) 競走馬ページの場所を集める
h = store.load_horses()
r, e = store.load()
for c in store.HORSE_EXTRA:
    if c not in h: h[c] = None
links = {k: v for k, v in zip(h.horse_id, h.cname) if isinstance(v, str)}
n0 = len(links)
res, ent = jra.recent_meetings()
today = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 9 * 3600))
for drl, _ in ent:                                  # 今週の出馬表
    mk = jra.meeting_key(drl)
    if mk and mk[4] >= today and left() > 60:
        try: harvest(jra.entries_of_meeting(drl), links)
        except Exception as ex: print("出馬表の取得に失敗:", ex)
if mode == "all":                                   # 過去のレース結果（開催ごとのページ）
    have_meet = set(e[e.horse_id.isin(links)].race_id.astype(str).str[:10])
    stop = False
    for ym, meetings in jra.archive_months():
        if ym < since or stop: break
        for srl in meetings:
            if left() < budget * 0.5: stop = True; break       # 時間の半分までを場所集めに使う
            mk = jra.meeting_key(srl)
            if not mk or "".join(mk[:4]) in have_meet: continue
            try: harvest(jra.results_of_meeting(srl), links)
            except Exception as ex: print("結果ページの取得に失敗:", srl, ex)
print(f"競走馬ページの場所: {n0} → {len(links)} 頭")
h["cname"] = h.horse_id.map(links).fillna(h.cname)

# ---- 2) 血統が分かっていない馬のページを読む（今週走る馬 → 最近走った馬の順）
up_ids = set(e[e.finish.isna()].horse_id)
last_run = e.merge(r[["race_id", "date"]], on="race_id").groupby("horse_id").date.max()
need = h[(h.sire.isna() | (h.sire == "不明")) & h.cname.notna()].copy()
if mode == "upcoming": need = need[need.horse_id.isin(up_ids)]
need["pri"] = need.horse_id.isin(up_ids).astype(int)
need["last"] = need.horse_id.map(last_run).fillna("")
need = need.sort_values(["pri", "last"], ascending=False)
print("血統を調べる馬:", len(need), "頭")
got = fail = 0
for i, (idx, row) in enumerate(need.iterrows()):
    if left() < 20: print("時間切れのため中断（次回はここから再開）"); break
    try:
        html = jra.page("/JRADB/accessU.html", row.cname)
    except Exception as ex:
        print("競走馬ページの取得に失敗:", row.horse_id, ex)
        if "拒否" in str(ex): break
        continue
    ped = parse_pedigree(html)
    if ped.get("sire"):
        for k, v in ped.items(): h.loc[idx, k] = v
        got += 1
    else:
        fail += 1
    if i == 0 or (fail and got == 0):
        os.makedirs(os.path.dirname(PROBE), exist_ok=True)
        open(PROBE, "w", encoding="utf-8").write(html)        # ページの作りを確認できるように1件保存
    if fail >= 5 and got == 0:
        print("血統を読み取れません。ページの作りが想定と違うため中止します（probe/jra_horse_sample.html を確認）")
        break
    if got and got % 50 == 0:
        store.save_horses(h); print(f"  {got} 頭 保存")
store.save_horses(h)
known = (h.sire.notna() & (h.sire != "不明")).sum()
print(f"血統を取得: {got} 頭（失敗 {fail}）／ 判明している馬 {known} / {len(h)} 頭")
