"""取得したデータをCSVに保存・更新する"""
import os
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "csv")
RACE_COLS = ["race_id", "date", "post_time", "circuit", "course", "race_no", "race_name", "grade", "surface", "distance",
             "direction", "track_condition", "weather", "num_horses"]
ENTRY_COLS = ["race_id", "horse_id", "horse_name", "waku", "umaban", "sex", "age", "jockey_id", "jockey_name",
              "weight_carried", "horse_weight", "weight_change", "training_time", "odds", "popularity", "finish",
              "corner4", "time", "last_3f", "trainer"]


def load():
    rp, ep = os.path.join(D, "races.csv"), os.path.join(D, "entries.csv")
    r = pd.read_csv(rp, dtype={"race_id": str}) if os.path.exists(rp) else pd.DataFrame(columns=RACE_COLS)
    e = pd.read_csv(ep, dtype={"race_id": str, "horse_id": str}) if os.path.exists(ep) else pd.DataFrame(columns=ENTRY_COLS)
    return r, e


def upsert(races, entries, keep_results=True):
    """同じレースのデータは新しいもので置き換える（結果が入っているレースを、結果なしの出馬表で上書きしない）"""
    if not races: return 0
    r0, e0 = load()
    nr = pd.DataFrame(races).reindex(columns=RACE_COLS)
    ne = pd.DataFrame(entries).reindex(columns=ENTRY_COLS)
    ne["race_id"] = ne.race_id.astype(str); nr["race_id"] = nr.race_id.astype(str)
    if keep_results and len(e0):
        done = set(e0[e0.finish.notna()].race_id)
        incoming_no_result = set(ne[ne.finish.isna()].race_id)
        skip = done & incoming_no_result
        nr = nr[~nr.race_id.isin(skip)]; ne = ne[~ne.race_id.isin(skip)]
        # 出馬表で分かっている馬場・天気が空なら、既存の値を残す
    ids = set(nr.race_id)
    r = pd.concat([r0[~r0.race_id.isin(ids)], nr], ignore_index=True).sort_values(["date", "race_id"])
    e = pd.concat([e0[~e0.race_id.isin(ids)], ne], ignore_index=True).sort_values(["race_id", "umaban"])
    os.makedirs(D, exist_ok=True)
    r.to_csv(os.path.join(D, "races.csv"), index=False)
    e.to_csv(os.path.join(D, "entries.csv"), index=False)
    # 馬のマスタ。取得済みの血統（父・母・母の父）とページの場所は消さずに残す
    h = e.drop_duplicates("horse_id", keep="last")[["horse_id", "horse_name", "trainer"]].copy()
    old = load_horses()
    keep = [c for c in HORSE_EXTRA if c in old]
    h = h.merge(old[["horse_id"] + keep], on="horse_id", how="left") if keep else h
    for c in HORSE_EXTRA:
        if c not in h: h[c] = None
    h["sire"] = h.sire.fillna("不明")
    h.to_csv(os.path.join(D, "horses.csv"), index=False)
    return len(ids)


HORSE_EXTRA = ["sire", "dam", "dam_sire", "cname"]


def load_horses():
    p = os.path.join(D, "horses.csv")
    return pd.read_csv(p, dtype={"horse_id": str}) if os.path.exists(p) else pd.DataFrame(columns=["horse_id", "horse_name", "trainer"] + HORSE_EXTRA)


def save_horses(h):
    h.to_csv(os.path.join(D, "horses.csv"), index=False)


def done_meetings():
    """保存済みの開催（年・場・回・日の10桁）"""
    r, e = load()
    fin = e[e.finish.notna()].race_id.astype(str)
    return set(fin.str[:10])


def upsert_payouts(pays):
    """払戻金を保存（同じレースは置き換え）"""
    if not pays: return 0
    p = os.path.join(D, "payouts.csv")
    old = pd.read_csv(p, dtype={"race_id": str, "combo": str}) if os.path.exists(p) else pd.DataFrame(columns=["race_id", "bet", "combo", "yen"])
    new = pd.DataFrame(pays); new["race_id"] = new.race_id.astype(str)
    out = pd.concat([old[~old.race_id.isin(set(new.race_id))], new], ignore_index=True).sort_values(["race_id", "bet"])
    out.to_csv(p, index=False)
    return new.race_id.nunique()


def payout_meetings():
    p = os.path.join(D, "payouts.csv")
    if not os.path.exists(p): return set()
    return set(pd.read_csv(p, dtype={"race_id": str}).race_id.str[:10])
