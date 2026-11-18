"""過去のレース結果をまとめて取得する（新しい開催から順に、指定した年月までさかのぼる）
使い方: python scraper/backfill.py 2024-11 330   ← 2024年11月まで、最大330分で中断
"""
import os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import jra, store

since = sys.argv[1] if len(sys.argv) > 1 else "2024-11"
budget = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 330 * 60
t0 = time.time()
done = store.done_meetings()
n_meet = 0


def odds_for(html_results, races):
    """そのレース日の出馬表（全レース）から確定単勝オッズを取る。なければレースごとのオッズページ。"""
    out = {}
    dde = [c for p, c in jra.links(html_results) if c.startswith("pw01dde")]
    if dde:
        try:
            dpage = jra.page("/JRADB/accessD.html", dde[0])
            des = [c for p, c in jra.links(dpage) if c.startswith("pw01des")]
            if des:
                _, ents = jra.parse_entries_all(jra.page("/JRADB/accessD.html", des[0]), des[0])
                for x in ents:
                    if x["odds"]: out[(x["race_id"], x["umaban"])] = x["odds"]
        except Exception as e:
            print("  出馬表からのオッズ取得に失敗:", e)
    for r in races:
        if any(k[0] == r["race_id"] for k in out) or not r.get("odds_cname"): continue
        try:
            for u, o in jra.parse_tanfuku(jra.page("/JRADB/accessO.html", r["odds_cname"])).items():
                out[(r["race_id"], u)] = o
        except Exception as e:
            print("  オッズページの取得に失敗:", r["race_id"], e)
    return out


for ym, meetings in jra.archive_months():
    if ym < since: break
    print("==", ym, len(meetings), "開催")
    for srl in meetings:
        if time.time() - t0 > budget:
            print("時間切れのため中断（次回はここから再開）"); sys.exit(3)
        mk = jra.meeting_key(srl)
        if not mk: continue
        key = "".join(mk[:4])
        if key in done: continue
        try:
            html = jra.results_of_meeting(srl)
            if not html: continue
            races, ents = jra.parse_results_all(html)
            od = odds_for(html, races)
            for x in ents: x["odds"] = od.get((x["race_id"], x["umaban"]))
            n = store.upsert(races, ents)
            done.add(key); n_meet += 1
            print(f"  {mk[4]} {jra.COURSE.get(mk[1], mk[1])} {n}レース（オッズ {sum(1 for x in ents if x['odds'])}/{len(ents)}頭）")
        except Exception as e:
            print("  失敗:", srl, e)
print("完了:", n_meet, "開催を追加")
