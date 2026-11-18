"""JRA公式サイトのページを読み取る（レース結果・出馬表・単勝オッズ）とページのたどり方"""
import re
from bs4 import BeautifulSoup
from common import fetch

BASE = "https://www.jra.go.jp"
ACT = re.compile(r"doAction\('([^']+)',\s*'([^']+)'\)")
HREF = re.compile(r"(/JRADB/access\w+\.html)\?CNAME=([\w/]+)")
COURSE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
DIRS = {"東京": "左", "新潟": "左", "中京": "左"}  # それ以外は右回り
WAKU = re.compile(r"枠(\d)")


def links(html):
    """ページ内の移動先（パス, cname）をすべて取り出す"""
    return ACT.findall(html) + HREF.findall(html)


def page(path, cname):
    return fetch(BASE + path, data={"cname": cname})


def txt(el):
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el else ""


def num(s, typ=float):
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return typ(m.group()) if m else None


def race_id_from(cname):
    """pw01sde1005202605030120261114/87 → 202605050301（年・場・回・日・R）"""
    m = re.search(r"(?:sde|dde|ou)10(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})", cname)
    if not m: return None, None
    jo, yy, kai, nichi, r, ymd = m.groups()
    return f"{yy}{jo}{kai}{nichi}{r}", f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


def grade_of(cls_text, name_html, name):
    g = None
    alt = " ".join(i.get("alt", "") for i in name_html.find_all("img")) if name_html else ""
    for k, v in [("GⅠ", "G1"), ("GⅡ", "G2"), ("GⅢ", "G3"), ("G1", "G1"), ("G2", "G2"), ("G3", "G3")]:
        if k in alt: g = v; break
    if g: return g
    for k in ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]:
        if k in cls_text: return k
    if "新馬" in name: return "新馬"
    return "オープン"


def parse_header(cap):
    """表のキャプション（レース情報）を読む"""
    d = {}
    date_line = txt(cap.select_one(".cell.date"))
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_line)
    d["date"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None
    m = re.search(r"\d+回(\S+?)\d+日", date_line)
    d["course"] = m.group(1) if m else None
    t = txt(cap.select_one(".cell.time"))
    m = re.search(r"(\d{1,2})時(\d{2})分", t)
    d["post_time"] = f"{int(m.group(1))}:{m.group(2)}" if m else None
    w = cap.select_one("li.weather .txt")
    d["weather"] = txt(w) or None
    cond = None
    for li in cap.select(".cell.baba li"):
        if "weather" in (li.get("class") or []): continue
        cond = txt(li.select_one(".txt")) or cond
    d["track_condition"] = cond
    rn = cap.select_one(".race_number img")
    d["race_no"] = num(rn.get("alt", "") if rn else "", int)
    name_el = cap.select_one(".race_name") or cap.select_one("h2")
    d["race_name"] = txt(name_el)
    cat = txt(cap.select_one(".cell.category"))
    cls = txt(cap.select_one(".cell.class"))
    d["jump"] = "障害" in cat or "障害" in cls or "ジャンプ" in d["race_name"]
    d["grade"] = grade_of(cls, cap.select_one("h2"), d["race_name"])
    course = txt(cap.select_one(".cell.course"))
    d["distance"] = num(course.replace(",", ""), int)
    det = txt(cap.select_one(".cell.course .detail"))
    d["surface"] = "芝" if "芝" in det else ("ダート" if "ダート" in det else None)
    d["direction"] = "左" if "左" in det else ("右" if "右" in det else DIRS.get(d["course"], "右"))
    return d


def horse_cells(tr):
    c = {td.get("class")[0] if td.get("class") else "": td for td in tr.find_all("td")}
    a = c.get("horse").find("a") if c.get("horse") else None
    hid = None
    if a and a.get("href"):
        m = re.search(r"pw01dud\d*?(\d{10})/", a["href"])
        hid = m.group(1) if m else None
    wk = c.get("waku").find("img") if c.get("waku") else None
    sexage = txt(c.get("age"))
    hw = txt(c.get("h_weight"))
    m = re.search(r"(\d{3})\s*(?:kg)?\s*\(([+\-]?\d+|前計不|計不)?\)", hw.replace(" ", ""))
    jk = c.get("jockey").find("a") if c.get("jockey") else None
    jid = None
    if jk is not None:
        m2 = re.search(r"pw04kmk(\d+)", str(jk))
        jid = "J" + m2.group(1) if m2 else None
    return dict(
        horse_id=hid or ("N" + txt(c.get("horse"))), horse_name=txt(c.get("horse")),
        waku=num(wk.get("alt", "") if wk else "", int), umaban=num(txt(c.get("num")), int),
        sex=sexage[:1] if sexage else None, age=num(sexage, int),
        jockey_id=jid or ("J" + txt(c.get("jockey"))), jockey_name=txt(c.get("jockey")).replace(" ", ""),
        weight_carried=num(txt(c.get("weight"))),
        horse_weight=int(m.group(1)) if m else None,
        weight_change=(int(m.group(2)) if m and m.group(2) and re.match(r"[+\-]?\d+$", m.group(2)) else None),
        trainer=txt(c.get("trainer")).replace(" ", ""), _cells=c)


def parse_results_all(html):
    """「全てのレースを表示」のレース結果ページ → (races, entries)"""
    soup = BeautifulSoup(html, "lxml")
    races, entries = [], []
    for unit in soup.select("div.race_result_unit"):
        t = unit.find("table")
        cap = t.find("caption")
        h = parse_header(cap)
        if h["jump"] or not h["surface"]: continue
        odds_link = [c for p, c in links(str(cap)) if c.startswith("pw151ou")]
        rid, _ = race_id_from(odds_link[0]) if odds_link else (None, None)
        if not rid:
            dl = [c for p, c in links(str(cap)) if "dde" in c]
            rid, _ = race_id_from(dl[0]) if dl else (None, None)
        if not rid: continue
        rows = []
        for tr in t.find("tbody").find_all("tr"):
            e = horse_cells(tr)
            c = e.pop("_cells")
            place = txt(c.get("place"))
            fin = int(place) if place.isdigit() else None
            corner = txt(c.get("corner")).split()
            pop = num(txt(c.get("pop")), int)
            e.update(race_id=rid, finish=fin, corner4=int(corner[-1]) if corner and corner[-1].isdigit() else None,
                     time=txt(c.get("time")) or None, last_3f=num(txt(c.get("f_time"))), popularity=pop, odds=None,
                     status=None if fin else (place or "取消"))
            rows.append(e)
        starters = [x for x in rows if x["finish"]]
        if len(starters) < 5: continue
        races.append(dict(race_id=rid, date=h["date"], post_time=h["post_time"], circuit="中央", course=h["course"],
                          race_no=h["race_no"], race_name=h["race_name"], grade=h["grade"], surface=h["surface"],
                          distance=h["distance"], direction=h["direction"], track_condition=h["track_condition"],
                          weather=h["weather"], num_horses=len(starters), odds_cname=odds_link[0] if odds_link else None))
        entries += starters
    return races, entries


def meeting_key(cname):
    """pw01des10052026050320261114/BF → ('2026','05','05','03', '2026-11-14')"""
    m = re.search(r"10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})", cname)
    if not m: return None
    jo, yy, kai, nichi, ymd = m.groups()
    return yy, jo, kai, nichi, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


def parse_entries_all(html, meeting_cname):
    """「全てのレースを表示」の出馬表ページ → (races, entries)（オッズ・馬体重は発表済みなら入る）"""
    mk = meeting_key(meeting_cname)
    soup = BeautifulSoup(html, "lxml")
    races, entries = [], []
    for t in soup.select("table.striped"):
        cap = t.find("caption")
        if not cap: continue
        h = parse_header(cap)
        if h["jump"] or not h["surface"]: continue
        if not mk or not h["race_no"]: continue
        rid = f"{mk[0]}{mk[1]}{mk[2]}{mk[3]}{h['race_no']:02d}"
        rows = []
        for tr in t.find("tbody").find_all("tr"):
            e = horse_cells(tr)
            c = e.pop("_cells")
            txt_all = txt(tr)
            if "取消" in txt_all or "除外" in txt_all: continue
            od = num(txt(c.get("odds")))
            e.update(race_id=rid, finish=None, corner4=None, time=None, last_3f=None, odds=od, popularity=None, status=None)
            rows.append(e)
        if len(rows) < 5: continue
        ods = [x["odds"] for x in rows if x["odds"]]
        if len(ods) == len(rows):
            order = sorted(rows, key=lambda x: x["odds"])
            for i, x in enumerate(order): x["popularity"] = i + 1
        races.append(dict(race_id=rid, date=h["date"], post_time=h["post_time"], circuit="中央", course=h["course"],
                          race_no=h["race_no"], race_name=h["race_name"], grade=h["grade"], surface=h["surface"],
                          distance=h["distance"], direction=h["direction"], track_condition=h["track_condition"],
                          weather=h["weather"], num_horses=len(rows), odds_cname=None))
        entries += rows
    return races, entries


def parse_tanfuku(html):
    """単勝・複勝オッズのページ → {馬番: 単勝オッズ}"""
    soup = BeautifulSoup(html, "lxml")
    t = soup.select_one("table.tanpuku")
    out = {}
    if not t: return out
    for tr in t.find("tbody").find_all("tr"):
        n = num(txt(tr.find("td", class_="num")), int)
        o = num(txt(tr.find("td", class_="odds_tan")))
        if n and o: out[n] = o
    return out


# ---------- ページのたどり方 ----------
def recent_meetings():
    """最近の開催（結果）と今週の開催（出馬表）の一覧"""
    res = [(c, c[7:]) for p, c in links(page("/JRADB/accessS.html", "pw01sli00/AF")) if c.startswith("pw01srl")]
    ent = [(c, c[7:]) for p, c in links(page("/JRADB/accessD.html", "pw01dli00/F3")) if c.startswith("pw01drl")]
    return list(dict.fromkeys(res)), list(dict.fromkeys(ent))


def meeting_date(cname):
    m = re.search(r"(\d{8})/", cname)
    return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None


def results_of_meeting(srl):
    html = page("/JRADB/accessS.html", srl)
    ses = [c for p, c in links(html) if c.startswith("pw01ses")]
    return page("/JRADB/accessS.html", ses[0]) if ses else None


def entries_of_meeting(drl):
    html = page("/JRADB/accessD.html", drl)
    des = [c for p, c in links(html) if c.startswith("pw01des")]
    return page("/JRADB/accessD.html", des[0]) if des else None


def archive_months(start="pw01skl00999999/B3"):
    """過去の開催：月ごとのページを新しい順にたどり、(年月, 開催のcname一覧) を返す"""
    seen, cur = set(), start
    while cur and cur not in seen:
        seen.add(cur)
        html = page("/JRADB/accessS.html", cur)
        mets = list(dict.fromkeys(c for p, c in links(html) if c.startswith("pw01srl")))
        months = sorted({c for p, c in links(html) if re.match(r"pw01skl10\d{6}/", c)})
        here_ym = re.search(r"pw01skl10(\d{4})(\d{2})", cur)
        if here_ym: ym = f"{here_ym.group(1)}-{here_ym.group(2)}"
        else:   # 最初のページ：載っている開催の日付から年月を決める
            ds = sorted(meeting_date(c) for c in mets if meeting_date(c))
            ym = ds[-1][:7] if ds else "9999-12"
        yield ym, mets
        here = re.search(r"pw01skl10(\d{6})", cur)
        here = here.group(1) if here else None
        older = [c for c in months if here is None or c[9:15] < here]
        cur = max(older) if older else None
