"""毎週の自動改善：設定を変えたモデルを何通りか試し、検証期間（直近半年）の成績が一番良いものを採用する
判定は「対数損失（確率の正確さ。小さいほど良い）」で行う。回収率は運の影響が大きいため参考にとどめる。"""
import datetime as dt, json, os, re, subprocess, sys
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG_P = os.path.join(ROOT, "config", "model_config.json")
HIST_P = os.path.join(ROOT, "data", "improve_history.csv")
cur = json.load(open(CFG_P)) if os.path.exists(CFG_P) else {"params": {}, "rounds": 400, "drop": []}

def merged(**kw):
    c = json.loads(json.dumps(cur)); c.setdefault("params", {}); c.setdefault("drop", [])
    for k, v in kw.items():
        if k == "params": c["params"].update(v)
        else: c[k] = v
    return c

variants = [
    ("現在の設定", cur),
    ("木を複雑に（num_leaves 31）", merged(params={"num_leaves": 31})),
    ("木を単純に（num_leaves 7）", merged(params={"num_leaves": 7})),
    ("学習をゆっくり長く", merged(params={"learning_rate": 0.02}, rounds=700)),
    ("学習を速く短く", merged(params={"learning_rate": 0.05}, rounds=250)),
    ("過学習をおさえる（min_child 100）", merged(params={"min_child_samples": 100})),
]
for g in ["馬場状態", "斤量", "脚質・展開", "馬の調子", "騎手"]:
    if g not in cur.get("drop", []):
        variants.append((f"「{g}」を使わない", merged(drop=sorted(set(cur.get("drop", [])) | {g}))))
for g in cur.get("drop", []):
    variants.append((f"「{g}」を使う", merged(drop=[x for x in cur["drop"] if x != g])))

results = []
for name, cfg in variants:
    env = dict(os.environ, EVAL_CONFIG=json.dumps(cfg, ensure_ascii=False))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "model.py"), os.path.join(ROOT, "data", "csv") + "/", "/tmp/eval.json"],
                       env=env, capture_output=True, text=True)
    m = re.search(r"EVAL_RESULT (\{.*\})", p.stdout)
    if not m:
        print("失敗:", name, p.stderr[-500:]); continue
    r = json.loads(m.group(1)); r["name"] = name; r["cfg"] = cfg
    results.append(r); print(f"{name}: 対数損失 {r['logloss']}  AUC {r['auc']}  ◎勝率 {r['win']}  ◎回収率 {r['roi']}")

base = next(r for r in results if r["name"] == "現在の設定")
best = min(results, key=lambda r: r["logloss"])
adopt = best is not base and best["logloss"] < base["logloss"] - 0.0003      # 誤差レベルの差では変えない
if adopt:
    os.makedirs(os.path.dirname(CFG_P), exist_ok=True)
    json.dump(best["cfg"], open(CFG_P, "w"), ensure_ascii=False, indent=1)
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")
row = dict(date=now, adopted=best["name"] if adopt else "変更なし（現在の設定が最良）",
           logloss_before=base["logloss"], logloss_after=best["logloss"] if adopt else base["logloss"],
           win_before=base["win"], win_after=best["win"] if adopt else base["win"],
           roi_before=base["roi"], roi_after=best["roi"] if adopt else base["roi"], tried=len(results))
old = pd.read_csv(HIST_P) if os.path.exists(HIST_P) else pd.DataFrame()
pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HIST_P, index=False)
print("結果:", row["adopted"])
