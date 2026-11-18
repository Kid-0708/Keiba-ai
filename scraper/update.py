"""最新情報の取得：今週の出馬表・オッズ・馬体重と、最近終わったレースの結果"""
import datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import jra, store

today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()
res, ent = jra.recent_meetings()
done = store.done_meetings()

# 1) 結果：直近の開催のうち、未保存のものと今日の開催
for srl, _ in res:
    mk = jra.meeting_key(srl)
    if not mk: continue
    if "".join(mk[:4]) in done and "".join(mk[:4]) in store.payout_meetings() and mk[4] != today: continue
    html = jra.results_of_meeting(srl)
    if not html: continue
    races, ents, pays = jra.parse_results_all(html, with_payouts=True)
    if not races: continue
    store.upsert_payouts(pays)
    # 確定オッズ：同じ日の出馬表（全レース）から
    od = {}
    try:
        dde = [c for p, c in jra.links(html) if c.startswith("pw01dde")]
        if dde:
            des = [c for p, c in jra.links(jra.page("/JRADB/accessD.html", dde[0])) if c.startswith("pw01des")]
            if des:
                _, e2 = jra.parse_entries_all(jra.page("/JRADB/accessD.html", des[0]), des[0])
                od = {(x["race_id"], x["umaban"]): x["odds"] for x in e2 if x["odds"]}
    except Exception as e:
        print("オッズ取得失敗:", e)
    for x in ents: x["odds"] = od.get((x["race_id"], x["umaban"]))
    print("結果", mk[4], jra.COURSE.get(mk[1]), store.upsert(races, ents), "レース")

# 2) 出馬表：今日以降の開催（オッズ・馬体重は発表されていれば入る）
r0, _ = store.load()
for drl, _ in ent:
    mk = jra.meeting_key(drl)
    if not mk or mk[4] < today: continue
    html = jra.entries_of_meeting(drl)
    if not html: continue
    races, ents = jra.parse_entries_all(html, drl)
    # 馬場・天気：同じ日・同じ競馬場で終わったレースの最新の発表を使う（なければ空欄）
    same = r0[(r0.date == mk[4]) & (r0.course == jra.COURSE.get(mk[1]))]
    for r in races:
        s = same[same.surface == r["surface"]].sort_values("race_no")
        if len(s):
            r["track_condition"] = s.iloc[-1].track_condition
            r["weather"] = s.iloc[-1].weather
    print("出馬表", mk[4], jra.COURSE.get(mk[1]), store.upsert(races, ents), "レース")
print("更新完了", today)
