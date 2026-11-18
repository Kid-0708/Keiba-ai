"""競馬予想モデル: 特徴量作成 → LightGBM学習 → バックテスト → SHAP根拠 → アプリ用JSON出力"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb, shap

D = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/keiba_sample/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/gen/app_data.json"
races = pd.read_csv(D + "races.csv", dtype={"race_id": str})
ent = pd.read_csv(D + "entries.csv", dtype={"race_id": str})
ent = ent.merge(pd.read_csv(D + "horses.csv")[["horse_id", "sire"]], on="horse_id")
df = ent.merge(races, on="race_id").sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)
CLS = {c: i for i, c in enumerate(["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"])}
CLS.update({c: i for i, c in enumerate(["C3", "C2", "C1", "B3", "B2", "B1", "A2", "A1", "重賞"])})   # 地方のクラス
COND = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
df["fr"] = df.finish / df.num_horses                         # 着順率(0に近いほど好走)
df["l3_rank"] = df.groupby("race_id").last_3f.rank(method="min") / df.num_horses
df["pos"] = (df.corner4 - 1) / (df.num_horses - 1)          # 4コーナー位置（0＝先頭、1＝最後方）

def sm(vals, k=2, prior=0.5):
    return (np.sum(vals) + prior * k) / (len(vals) + k)

# ---------- 時系列で「そのレースより前」の情報だけを使って特徴量を作る ----------
hist = {}   # horse_id -> list of dict (過去走)
jh = {}     # jockey_id -> [runs, wins, top3]
sh = {}     # sire -> 父馬ごとの成績の累計（血統）
def sire_fr(st, key, prior=0.5, k=15):
    n, sm_ = st.get(key, (0, 0.0)); return (sm_ + prior * k) / (n + k)
rows = []
for date, day in df.groupby("date", sort=True):
    for rid, r in day.groupby("race_id"):
        for _, e in r.iterrows():
            H = hist.get(e.horse_id, [])
            last5 = H[-5:]
            f = {}
            f["h_runs"] = len(H)
            f["h_wins"] = sum(x["finish"] == 1 for x in H)
            f["h_avg_fr5"] = sm([x["fr"] for x in last5])
            f["h_last_fr"] = H[-1]["fr"] if H else np.nan
            f["h_best_fr5"] = min([x["fr"] for x in last5]) if H else np.nan
            f["h_l3f_rank5"] = np.mean([x["l3"] for x in last5]) if H else np.nan
            f["h_pos5"] = np.mean([x["pos"] for x in last5]) if H else np.nan          # 脚質（前に行くか後ろか）
            f["h_lead_rate"] = np.mean([x["pos"] <= 0.12 for x in last5]) if H else np.nan
            rr = [x["fr"] for x in H if x["dir"] == e.direction]; ro = [x["fr"] for x in H if x["dir"] != e.direction]
            f["_dir_diff"] = (sm(ro) - sm(rr)) if len(rr) >= 2 and len(ro) >= 2 else 0.0   # ＋なら今回の回りが得意
            f["_good_dists"] = [x["dist"] for x in H if x["fr"] <= 0.3]
            f["_surf_diff"] = (sm([x["fr"] for x in H if x["surface"] != e.surface]) - sm([x["fr"] for x in H if x["surface"] == e.surface])) if H else 0.0
            f["_trend"] = (np.mean([x["fr"] for x in H[-5:-2]]) - np.mean([x["fr"] for x in H[-2:]])) if len(H) >= 4 else 0.0
            f["h_cls_prev"] = max([x["cls"] for x in H]) if H else np.nan
            f["h_cls_avg5"] = np.mean([x["cls"] for x in last5]) if H else np.nan
            f["h_perf5"] = np.mean([x["cls"] * 0.5 - x["fr"] * 3 for x in last5]) if H else np.nan   # 格×着順の総合指標
            f["h_perf_best"] = max([x["cls"] * 0.5 - x["fr"] * 3 for x in last5]) if H else np.nan
            s = [x["fr"] for x in H if x["surface"] == e.surface]
            f["h_surf_fr"], f["h_surf_runs"] = sm(s), len(s)
            dd = [x["fr"] for x in H if abs(x["dist"] - e.distance) <= 200]
            f["h_dist_fr"], f["h_dist_runs"] = sm(dd), len(dd)
            hv = [x["fr"] for x in H if x["cond"] >= 1]
            f["h_heavy_fr"], f["h_heavy_runs"] = sm(hv), len(hv)
            cc = [x["fr"] for x in H if x["course"] == e.course]
            f["h_course_fr"], f["h_course_runs"] = sm(cc), len(cc)
            jr = jh.get(e.jockey_id, [0, 0, 0])
            f["j_win_rate"] = (jr[1] + 0.08 * 20) / (jr[0] + 20)
            f["j_top3_rate"] = (jr[2] + 0.22 * 20) / (jr[0] + 20)
            st = sh.get(e.sire, {})
            f["sire_all_fr"] = sire_fr(st, "all")
            f["sire_heavy_fr"] = sire_fr(st, "heavy")
            f["sire_surf_fr"] = sire_fr(st, e.surface)
            h_all = sm([x["fr"] for x in H])
            # 道悪適性（マイナスほど道悪が得意）: 本馬の道悪成績と血統の道悪成績を合成
            f["heavy_apt"] = 0.5 * (f["h_heavy_fr"] - h_all) * min(1, f["h_heavy_runs"] / 3) + 0.5 * (f["sire_heavy_fr"] - f["sire_all_fr"]) * 4
            f["cond_x_apt"] = COND[e.track_condition] * f["heavy_apt"]
            f["kin"] = e.weight_carried
            f["kin_diff"] = e.weight_carried - H[-1]["kin"] if H else np.nan
            f["horse_weight"] = e.horse_weight
            f["weight_change"] = e.weight_change
            f["days"] = e.days_since_last
            f["train"] = e.training_time - (39.5 if e.circuit == "地方" else 53.5)
            f["age"] = e.age
            f["sex_c"] = {"牡": 0, "牝": 1, "セ": 2}[e.sex]
            f["waku"] = e.waku
            f["num_horses"] = e.num_horses
            f["distance"] = e.distance
            f["surf_c"] = int(e.surface == "芝")
            f["cond"] = COND[e.track_condition]
            f["cls"] = CLS[e.grade]
            f["nar"] = int(e.circuit == "地方")
            # 表示用の生データ
            f["_avg_fin5"] = np.mean([x["finish"] for x in last5]) if H else np.nan
            f["_surf_fin"] = np.mean([x["finish"] for x in H if x["surface"] == e.surface]) if s else np.nan
            f["_dist_fin"] = np.mean([x["finish"] for x in H if abs(x["dist"] - e.distance) <= 200]) if dd else np.nan
            f["_heavy_fin"] = np.mean([x["finish"] for x in H if x["cond"] >= 1]) if hv else np.nan
            f["_course_fin"] = np.mean([x["finish"] for x in H if x["course"] == e.course]) if cc else np.nan
            f["_jruns"] = jr[0]
            f["_top3_5"] = sum(x["finish"] <= 3 for x in last5)
            f["_surf_top3"] = sum(x["finish"] <= 3 for x in H if x["surface"] == e.surface)
            f["_dist_top3"] = sum(x["finish"] <= 3 for x in H if abs(x["dist"] - e.distance) <= 200)
            f["_heavy_top3"] = sum(x["finish"] <= 3 for x in H if x["cond"] >= 1)
            f["_past"] = [dict(date=x["date"], course=x["course"], surf=x["surface"], dist=x["dist"], cond=x["condname"],
                               grade=x["grade"], fin=x["finish"], n=x["n"], l3=x["l3f"], pop=x["pop"]) for x in H[-5:]][::-1]
            f["idx"] = e.name
            rows.append(f)
        # レース後に履歴を更新（結果があるレースのみ）
        for _, e in r.iterrows():
            if pd.isna(e.finish):
                continue
            hist.setdefault(e.horse_id, []).append(dict(
                fr=e.fr, finish=int(e.finish), l3=e.l3_rank, cls=CLS[e.grade], surface=e.surface, dist=e.distance,
                cond=COND[e.track_condition], condname=e.track_condition, course=e.course, kin=e.weight_carried,
                date=e.date, grade=e.grade, n=int(e.num_horses), l3f=e.last_3f, pop=int(e.popularity), pos=e.pos, dir=e.direction))
            st = sh.setdefault(e.sire, {})
            for key in ["all", e.surface] + (["heavy"] if COND[e.track_condition] >= 1 else []):
                n, sm_ = st.get(key, (0, 0.0)); st[key] = (n + 1, sm_ + e.fr)
            j = jh.setdefault(e.jockey_id, [0, 0, 0])
            j[0] += 1; j[1] += int(e.finish == 1); j[2] += int(e.finish <= 3)

F = pd.DataFrame(rows).set_index("idx").sort_index()
F = F.drop(columns=[c for c in F.columns if c in df.columns])
df = df.join(F)
# レース内での相対値（ほかの出走馬と比べてどうか）
df["rel_perf"] = df.h_perf5 - df.groupby("race_id").h_perf5.transform("mean")
df["cls_up"] = df.cls - df.h_cls_avg5
df["rel_fr5"] = df.h_avg_fr5 - df.groupby("race_id").h_avg_fr5.transform("mean")
df["rel_surf"] = df.h_surf_fr - df.groupby("race_id").h_surf_fr.transform("mean")
df["rel_jwin"] = df.j_win_rate - df.groupby("race_id").j_win_rate.transform("mean")
df["train_rank"] = df.groupby("race_id").train.rank(method="min") / df.num_horses
df["rel_kin"] = df.kin - df.groupby("race_id").kin.transform("mean")
df["j_rank"] = df.groupby("race_id").j_win_rate.rank(ascending=False, method="min")
# 展開：逃げ・先行馬が多いほどハイペース（前の馬に厳しい）
pf = df.h_pos5.fillna(0.5)
df["pace_est"] = (pf <= 0.2).groupby(df.race_id).transform("mean") + 0.5 * ((pf > 0.2) & (pf <= 0.4)).groupby(df.race_id).transform("mean")
PACE_LO, PACE_HI = df.drop_duplicates("race_id").pace_est.quantile([1/3, 2/3]).values   # スロー／ハイの境目（全レースの3分位）
PACE_MID = (PACE_LO + PACE_HI) / 2
df["pace_x_pos"] = (df.pace_est - PACE_MID) * (0.5 - pf)
df["n_leaders"] = (df.h_lead_rate.fillna(0) >= 0.4).groupby(df.race_id).transform("sum")
df["lone_lead"] = ((df.h_lead_rate.fillna(0) >= 0.4) & (df.n_leaders == 1)).astype(int)

GROUPS = {
    "過去の戦績": ["h_runs", "h_wins", "h_avg_fr5", "h_last_fr", "h_best_fr5", "h_l3f_rank5", "h_cls_prev", "h_cls_avg5", "h_perf5", "h_perf_best", "rel_perf", "cls_up", "rel_fr5", "cls"],
    "馬場状態": ["cond", "h_heavy_fr", "h_heavy_runs", "sire_heavy_fr", "heavy_apt", "cond_x_apt"],
    "コース・距離": ["h_surf_fr", "h_surf_runs", "rel_surf", "h_dist_fr", "h_dist_runs", "h_course_fr", "h_course_runs", "sire_surf_fr", "sire_all_fr",
                  "surf_c", "distance", "waku", "num_horses", "nar"],
    "騎手": ["j_win_rate", "j_top3_rate", "rel_jwin"],
    "馬の調子": ["days", "weight_change", "horse_weight", "train", "train_rank", "age", "sex_c"],
    "斤量": ["kin", "kin_diff", "rel_kin"],
    "脚質・展開": ["h_pos5", "h_lead_rate", "pace_est", "pace_x_pos", "lone_lead"],
}
FEATS = [c for g in GROUPS.values() for c in g]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)

def fit(train, target, rounds=400):
    return lgb.train(PARAMS, lgb.Dataset(train[FEATS], target), num_boost_round=rounds)

from scipy.optimize import minimize
def market_p(X):
    inv = 1 / X.odds
    return inv / inv.groupby(X.race_id).transform("sum")

def softmax_race(z, rid):
    z = z - z.groupby(rid).transform("max")
    e = np.exp(z)
    return e / e.groupby(rid).transform("sum")

def fit_combo(X, p_model):
    """Benter方式: 勝率 ∝ exp(a·log(モデル勝率) + b·log(市場勝率)) の a,b を最尤推定"""
    lm, lk, y, rid = np.log(p_model), np.log(market_p(X)), (X.finish == 1).values, X.race_id
    def nll(w):
        p = softmax_race(w[0] * lm + w[1] * lk, rid)
        return -np.log(p[y] + 1e-12).sum()
    return minimize(nll, [0.5, 0.5], method="Nelder-Mead").x

COMBO = None
def predict(m_win, m_top3, X):
    X = X.copy()
    X["p_win_raw"] = m_win.predict(X[FEATS])
    X["p_model"] = X.p_win_raw / X.groupby("race_id").p_win_raw.transform("sum")    # 能力モデル単独の勝率
    a, b = COMBO
    X["p_win"] = softmax_race(a * np.log(X.p_model) + b * np.log(market_p(X)), X.race_id)   # オッズと統合した最終勝率
    X["p_top3"] = m_top3.predict(X[FEATS])
    X["p_top3"] = (X.p_top3 / X.groupby("race_id").p_top3.transform("sum") * np.minimum(3, X.num_horses)).clip(upper=0.97)
    X["ev"] = X.p_win * X.odds
    X["rank"] = X.groupby("race_id").p_win.rank(ascending=False, method="first").astype(int)
    return X

past = df[df.finish.notna()].copy()
future = df[df.finish.isna()].copy()
past["win"] = (past.finish == 1).astype(int)
past["top3"] = (past.finish <= 3).astype(int)

# ---------- バックテスト（2026-05-01以降を「未来」とみなして検証） ----------
SPLIT = "2026-05-01"
tr, te = past[past.date < SPLIT], past[past.date >= SPLIT]
# 統合の重みは、学習期間の後半を「検証用」にして決める（未来の情報を使わない）
VSPLIT = "2025-07-01"
tr1, va = tr[tr.date < VSPLIT], tr[tr.date >= VSPLIT].copy()
m1 = fit(tr1, tr1.win)
va_raw = pd.Series(m1.predict(va[FEATS]), index=va.index)
COMBO = fit_combo(va, va_raw / va_raw.groupby(va.race_id).transform("sum"))
print("combo weights (model, market):", COMBO.round(3))
mw, mt = fit(tr, tr.win), fit(tr, tr.top3)
te = predict(mw, mt, te)
EVTH = 1.2
def stats(sel, label):
    n = len(sel); hit = int((sel.finish == 1).sum())
    return dict(label=label, bets=n, win_hits=hit, win_rate=round(hit / n, 3) if n else 0,
                top3_rate=round(float((sel.finish <= 3).mean()), 3) if n else 0,
                top2_rate=round(float((sel.finish <= 2).mean()), 3) if n else 0,
                roi=round(float(sel[sel.finish == 1].odds.sum() * 100 / n), 1) if n else 0)
bt = dict(
    period=f"{te.date.min()} 〜 {te.date.max()}", races=int(te.race_id.nunique()), train_races=int(tr.race_id.nunique()),
    honmei=stats(te[te["rank"] == 1], "本命（◎）の単勝"),
    fav=stats(te[te.popularity == 1], "1番人気の単勝"),
    ev=stats(te[(te.ev >= EVTH) & (te.p_win >= 0.05)], f"期待値{EVTH}以上の単勝"),
    allbet=stats(te, "全馬の単勝"),
)
bt["by_circuit"] = {c: dict(races=int(g.race_id.nunique()), honmei=stats(g[g["rank"] == 1], "本命（◎）の単勝"),
                            fav=stats(g[g.popularity == 1], "1番人気の単勝"),
                            ev=stats(g[(g.ev >= EVTH) & (g.p_win >= 0.05)], f"期待値{EVTH}以上の単勝"))
                    for c, g in te.groupby("circuit")}
# 確率の当てはまり（予想勝率の帯ごとの実際の勝率）
te["bin"] = pd.cut(te.p_win, [0, .05, .1, .2, .3, 1], labels=["〜5%", "5〜10%", "10〜20%", "20〜30%", "30%〜"])
bt["calib"] = [dict(bin=str(b), n=int(len(g)), pred=round(float(g.p_win.mean()), 3), actual=round(float(g.win.mean()), 3))
               for b, g in te.groupby("bin", observed=True)]
from sklearn.metrics import roc_auc_score, log_loss
bt["auc_model"] = round(roc_auc_score(te.win, te.p_win), 3)
bt["auc_fundamental"] = round(roc_auc_score(te.win, te.p_model), 3)
bt["logloss"] = dict(final=round(log_loss(te.win, te.p_win.clip(1e-6, 1-1e-6)), 4), market=round(log_loss(te.win, market_p(te).clip(1e-6, 1-1e-6)), 4))
mkt = (1 / te.odds) / te.groupby("race_id").odds.transform(lambda o: (1 / o).sum())
bt["auc_market"] = round(roc_auc_score(te.win, mkt), 3)
# 月ごとの累積収支（◎単勝を100円ずつ）
te["ret_h"] = np.where((te["rank"] == 1) & (te.finish == 1), te.odds * 100, 0) - np.where(te["rank"] == 1, 100, 0)
evsel = (te.ev >= EVTH) & (te.p_win >= 0.05)
te["ret_e"] = np.where(evsel & (te.finish == 1), te.odds * 100, 0) - np.where(evsel, 100, 0)
te["ret_f"] = np.where((te.popularity == 1) & (te.finish == 1), te.odds * 100, 0) - np.where(te.popularity == 1, 100, 0)
cum = te.groupby("date")[["ret_h", "ret_e", "ret_f"]].sum().cumsum().round(0)
bt["cum"] = [dict(date=d, honmei=int(r.ret_h), ev=int(r.ret_e), fav=int(r.ret_f)) for d, r in cum.iterrows()]
print(json.dumps({k: v for k, v in bt.items() if k not in ("cum",)}, ensure_ascii=False, indent=1))

# ---------- 本番モデル（過去全レースで学習）で次回開催を予想 ----------
COMBO_BT = COMBO.copy()
va_p = va_raw / va_raw.groupby(va.race_id).transform("sum")
both = pd.concat([va, te]); both_p = pd.concat([va_p, te.p_model])
COMBO = fit_combo(both, both_p)          # 本番用の重みは、学習に使っていない予測（検証期間＋バックテスト期間）で推定
print("final combo:", COMBO.round(3))
TODAY = "2026-11-18"
pre_today = past[past.date < TODAY]                      # 今日のレースは「今日より前」のデータだけで学習したモデルで予想する
mw_all, mt_all = fit(pre_today, pre_today.win), fit(pre_today, pre_today.top3)
future = predict(mw_all, mt_all, future)
expl = shap.TreeExplainer(mw_all)
rv_dates = [d for c in ['中央', '地方'] for d in sorted(te[te.circuit == c].date.unique())[-2:]]
review = te[te.date.isin(rv_dates) & (te.date < TODAY)].copy()   # レース結果用: 中央・地方それぞれ直近2開催日（バックテストモデル）
today_done = past[past.date == TODAY].copy()                       # 今日すでに終わったレース

# ---------- 当日の馬場傾向（内枠/外枠・前/後ろの有利不利）----------
def in_score(X): return 1 - 2 * (X.waku - 1) / 7                     # 内枠ほど＋
def fr_score(X): return (1 - 2 * X.h_pos5.fillna(0.5))                # 前に行く馬ほど＋
def fit_bias(R):
    # 終わったレースの着順（上位3着）に、オッズでは説明できない「内枠」「前」の有利さがどれだけあったかを推定（Plackett-Luce＋リッジ）
    groups = []
    for _, g in R.groupby("race_id"):
        g = g.sort_values("finish")
        groups.append((np.log(market_p(g).values), in_score(g).values, fr_score(g).values))
    def nll(b):
        tot = 0.0
        for off, xi, xf in groups:
            u = off + b[0] * xi + b[1] * xf
            for k in range(3):
                tot -= u[k] - np.log(np.exp(u[k:]).sum())
        return tot + 2.0 * (b ** 2).sum()
    return minimize(nll, [0.0, 0.0], method="Nelder-Mead").x

def bias_info(rid):
    info = races[races.race_id == rid].iloc[0]
    done = past[(past.course == info.course)]
    same = done[(done.date == info.date) & (done.race_no < info.race_no)]
    prev_day = str((pd.Timestamp(info.date) - pd.Timedelta(days=1)).date())
    prev = done[done.date == prev_day]
    if same.race_id.nunique() >= 3: R, src, shrink = same, f"本日の{int(same.race_no.min())}〜{int(same.race_no.max())}R", 0.8
    elif prev.race_id.nunique() >= 3: R, src, shrink = prev, f"前日（同じ開催）の{prev.race_id.nunique()}レース", 0.6
    else:
        last = done.date.max() if len(done) else None
        if last is None: return None
        R, src, shrink = done[done.date == last], f"前回開催（{int(last[5:7])}/{int(last[8:])}）の参考", 0.0
    b = fit_bias(R)
    top3 = R[R.finish <= 3]
    lab = lambda v, p, n, z: p if v > 0.15 else (n if v < -0.15 else z)
    return dict(source=src, applied=shrink > 0, b_in=round(float(b[0]), 3), b_fr=round(float(b[1]), 3), shrink=shrink,
                inside=lab(b[0], "内枠が有利", "外枠が有利", "内外の差なし"), front=lab(b[1], "前に行く馬が有利", "後ろから行く馬が有利", "脚質の差なし"),
                n_races=int(R.race_id.nunique()), win_waku=[int(w) for w in R[R.finish == 1].sort_values("race_no").waku],
                top3_inner=int((top3.waku <= 4).sum()), top3_n=int(len(top3)),
                top3_front=int((top3.h_pos5.fillna(0.5) <= 0.38).sum()))
expl_bt = shap.TreeExplainer(mw)

CONDS = ["良", "稍重", "重", "不良"]

def reasons(e, cond):
    """根拠の文章（数字は最小限、ふだんの言葉で）"""
    out = {}
    n5 = min(5, int(e.h_runs))
    out["過去の戦績"] = ("初出走のため実績なし" if e.h_runs == 0 else
                    f"最近{n5}戦で3着以内が{int(e._top3_5)}回（通算{int(e.h_runs)}戦{int(e.h_wins)}勝）")
    sd = e.sire_heavy_fr - e.sire_all_fr
    blood = f"父{e.sire}は道悪に強い血統" if sd < -0.015 else (f"父{e.sire}は道悪が苦手な血統" if sd > 0.015 else f"父{e.sire}は道悪で平均的")
    own = (f"本馬は道悪で{int(e.h_heavy_runs)}戦{int(e._heavy_top3)}回3着以内" if e.h_heavy_runs else "本馬は道悪の経験なし")
    out["馬場状態"] = ("良馬場なので影響は小さめ。" if cond == 0 else f"{CONDS[cond]}馬場。") + f"{own}。{blood}"
    s = f"{e.surface}で{int(e.h_surf_runs)}戦{int(e._surf_top3)}回3着以内" if e.h_surf_runs else f"{e.surface}は初めて"
    d = f"{e.distance}m前後で{int(e.h_dist_runs)}戦{int(e._dist_top3)}回3着以内" if e.h_dist_runs else f"{e.distance}m前後は初めて"
    out["コース・距離"] = f"{s}。{d}"
    out["騎手"] = f"{e.jockey_name}。このレースの騎手の中で勝率{int(e.j_rank)}番目"
    if pd.isna(e.days): days = "初出走"
    elif e.days >= 90: days = f"約{int(round(e.days / 30))}か月ぶりの休み明け"
    else: days = f"前走から{int(round(e.days / 7))}週間"
    wc = int(e.weight_change)
    wtxt = "馬体重は前走と同じ" if wc == 0 else f"馬体重{'+' if wc > 0 else '−'}{abs(wc)}kg" + ("（大きく変動）" if abs(wc) >= 10 else "")
    out["馬の調子"] = f"{days}。{wtxt}。調教タイムはこのレースで{int(round(e.train_rank * e.num_horses))}番目"
    if pd.isna(e.kin_diff) or e.kin_diff == 0: kd = "（前走と同じ）" if not pd.isna(e.kin_diff) else ""
    else: kd = f"（前走より{abs(e.kin_diff):.0f}kg{'重い' if e.kin_diff > 0 else '軽い'}）"
    out["斤量"] = f"{e.kin:.0f}kg{kd}" if e.kin == int(e.kin) else f"{e.kin:.1f}kg{kd}"
    sty = style_name(e)
    pace = pace_name(e.pace_est)
    if sty == "不明": out["脚質・展開"] = f"脚質はまだ不明。{pace}の予想"
    else:
        fav = "前に行く馬に有利" if e.pace_est < PACE_LO else ("後ろから行く馬に有利" if e.pace_est > PACE_HI else "どの脚質にも公平")
        out["脚質・展開"] = f"{sty}タイプ。{pace}の予想で{fav}" + ("。単騎で逃げられそう" if e.lone_lead else "")
    return out

def style_name(e):
    if e.h_runs < 2 or pd.isna(e.h_pos5): return "不明"
    return "逃げ" if e.h_pos5 <= 0.15 else "先行" if e.h_pos5 <= 0.38 else "差し" if e.h_pos5 <= 0.68 else "追込"

def pace_name(p):
    return "ハイペース" if p > PACE_HI else ("スローペース" if p < PACE_LO else "平均ペース")

def traits(e):
    """馬の特徴（短いラベル）と説明文"""
    t = []
    sty = style_name(e)
    if sty != "不明": t.append(sty)
    gd = e._good_dists
    if len(gd) >= 2:
        lo, hi = int(np.percentile(gd, 30) // 100 * 100), int(np.percentile(gd, 70) // 100 * 100)
        t.append(f"{lo}m前後が得意" if lo == hi else f"{lo}〜{hi}mが得意")
    if e.circuit == "中央" and e.h_runs >= 4 and abs(e._surf_diff) > 0.06:
        t.append(f"{e.surface}向き" if e._surf_diff > 0 else ("ダート向き" if e.surface == "芝" else "芝向き"))
    if e.heavy_apt < -0.045: t.append("道悪が得意")
    elif e.heavy_apt > 0.045: t.append("道悪が苦手")
    if e._dir_diff > 0.07: t.append(f"{e.direction}回りが得意")
    elif e._dir_diff < -0.07: t.append(f"{e.direction}回りが苦手")
    if e.h_runs >= 3 and e.h_l3f_rank5 <= 0.3: t.append("末脚が鋭い")
    if e._trend > 0.15: t.append("調子上向き")
    elif e._trend < -0.15: t.append("調子下降気味")
    if not pd.isna(e.days) and e.days >= 90: t.append("休み明け")
    if e.h_runs == 0: t.append("初出走")
    # 説明文
    parts = []
    if sty in ("逃げ", "先行"): parts.append("スタートから前につけるタイプ")
    elif sty in ("差し", "追込"): parts.append("後方からレース終盤に追い上げるタイプ")
    if "末脚が鋭い" in t: parts.append("最後の伸び（上がり）が速い")
    if "道悪が得意" in t: parts.append("雨で馬場が悪くなるとプラス")
    if "道悪が苦手" in t: parts.append("雨で馬場が悪くなるとマイナス")
    if "調子上向き" in t: parts.append("最近の成績が良くなってきている")
    if "調子下降気味" in t: parts.append("最近は成績を落としている")
    if "休み明け" in t: parts.append("久しぶりのレースで仕上がりに注意")
    desc = "。".join(parts) + "。" if parts else "目立った特徴はまだ少ない馬です。"
    return t, desc

def scenario(X, model, cond):
    """馬場状態を cond にした場合の予想（オッズは発表時点のまま）"""
    Xc = X.copy()
    if cond is not None:
        Xc["cond"] = cond
        Xc["cond_x_apt"] = cond * Xc.heavy_apt
    raw = pd.Series(model.predict(Xc[FEATS]), index=Xc.index)
    pm = raw / raw.groupby(Xc.race_id).transform("sum")
    a, b = COMBO
    pw = softmax_race(a * np.log(pm) + b * np.log(market_p(Xc)), Xc.race_id)
    sv = shap.TreeExplainer(model).shap_values(Xc[FEATS])
    sv = sv[1] if isinstance(sv, list) and len(sv) == 2 else np.asarray(sv)
    S = pd.DataFrame(sv, columns=FEATS, index=Xc.index)
    G = pd.DataFrame({g: S[c].sum(axis=1) for g, c in GROUPS.items()})
    G = G - G.groupby(Xc.race_id).transform("mean")          # 同じレースの他馬と比べたプラス/マイナス
    return pm, pw, G

def build(X, model, with_result, all_conds, use_bias=False):
    conds = list(range(4)) if all_conds else [None]
    sc = {c: scenario(X, model, c) for c in conds}
    BI = {}
    if all_conds:
        # 馬場が変われば人気（オッズ）も動く。発表時点の最終勝率に、馬場変更による評価の変化をそのまま上乗せする
        pm0, pw0, _ = scenario(X, model, None)
        for c in conds:
            pm, _, G = sc[c]
            sc[c] = (pm, softmax_race(np.log(pw0) + np.log(pm) - np.log(pm0), X.race_id), G)
    if use_bias:   # 当日の馬場傾向を反映（馬場状態の切り替え後の確率にも同じだけ上乗せ）
        adj = pd.Series(0.0, index=X.index)
        for rid in X.race_id.unique():
            bi = bias_info(rid); BI[rid] = bi
            if bi and bi["applied"]:
                m = X.race_id == rid
                adj[m] = bi["shrink"] * (bi["b_in"] * in_score(X[m]) + bi["b_fr"] * fr_score(X[m]))
        for c in conds:
            pm, pw, G = sc[c]
            sc[c] = (pm, softmax_race(np.log(pw) + adj, X.race_id), G)
    out = []
    for rid, r in X.groupby("race_id"):
        r = r.sort_values("umaban")
        info = races[races.race_id == rid].iloc[0]
        actual = COND[info.track_condition]
        mk = market_p(r)
        horses = []
        for i, e in r.iterrows():
            scen = {}
            for c in conds:
                pm, pw, G = sc[c]
                cc = actual if c is None else c
                txt = reasons(e, cc)
                scen[CONDS[cc]] = dict(p=round(float(pw[i]), 4), pm=round(float(pm[i]), 4),
                                       f=[round(float(G.loc[i, k]), 3) for k in GROUPS], baba=txt["馬場状態"])
            txt = reasons(e, actual)
            horses.append(dict(
                umaban=int(e.umaban), waku=int(e.waku), name=e.horse_name, sex=e.sex, age=int(e.age), sire=e.sire,
                jockey=e.jockey_name, kin=float(e.kin), hw=int(e.horse_weight), wc=int(e.weight_change),
                odds=float(e.odds), pop=int(e.popularity), p_mkt=round(float(mk[i]), 4),
                texts={k: txt[k] for k in GROUPS}, sc=scen, traits=traits(e)[0], desc=traits(e)[1], style=style_name(e),
                past=e._past, finish=int(e.finish) if with_result else None))
        e0 = r.iloc[0]
        leaders = [int(x.umaban) for _, x in r.iterrows() if style_name(x) == "逃げ"]
        fronts = [int(x.umaban) for _, x in r.iterrows() if style_name(x) == "先行"]
        pace = dict(label=pace_name(e0.pace_est), leaders=leaders, fronts=fronts,
                    favor="前に行く馬（逃げ・先行）" if e0.pace_est < PACE_LO else ("後ろから行く馬（差し・追込）" if e0.pace_est > PACE_HI else "特になし（実力どおり）"))
        out.append(dict(race_id=str(rid), date=info.date, post=info.post_time, bias=BI.get(rid), circuit=info.circuit, pace=pace, course=info.course, no=int(info.race_no), name=info.race_name,
                        grade=info.grade, surface=info.surface, distance=int(info.distance), direction=info.direction,
                        cond=info.track_condition, weather=info.weather, n=int(info.num_horses), horses=horses))
    return out

data = dict(
    generated="2026-11-18", ev_threshold=EVTH, combo=[round(float(x), 3) for x in COMBO], groups=list(GROUPS),
    upcoming=build(future, mw_all, False, True, use_bias=True),
    review=build(review, mw, True, False) + build(today_done, mw_all, True, False),
    backtest=bt,
    importance=sorted([dict(name=g, value=round(float(np.abs(expl.shap_values(past[FEATS].sample(2000, random_state=0))
                                                            )[:, [FEATS.index(c) for c in cs]].sum(axis=1).mean()), 3))
                       for g, cs in GROUPS.items()], key=lambda x: -x["value"]),
)
json.dump(data, open(OUT, "w"), ensure_ascii=False, default=lambda o: None if pd.isna(o) else o)
print("upcoming", len(data["upcoming"]), "review", len(data["review"]), data["importance"])
