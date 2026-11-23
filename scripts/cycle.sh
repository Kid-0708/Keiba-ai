#!/usr/bin/env bash
# 1回分の更新：最新情報の取得 → 予想の計算 → 公開
set -u
git pull -q --rebase --autostash || true   # 最新の状態にそろえてから取得する
rm -rf data/raw          # 前回取得したページの保存分を消す（残っているとオッズや結果が古いままになる）
python scraper/update.py || echo "取得でエラー（続行）"
n=$(python -c "import pandas as pd; e=pd.read_csv('data/csv/entries.csv'); print(e[e.finish.notna()].race_id.nunique())" 2>/dev/null || echo 0)
if [ "$n" -ge 1500 ]; then
  python pipeline/model.py data/csv/ data/app_data.json && python pipeline/log_predictions.py && python pipeline/analyze.py && python pipeline/export.py || echo "計算でエラー"
fi
git add -A data docs
git commit -qm "update: $(TZ=Asia/Tokyo date '+%m/%d %H:%M')" || echo "no changes"
# 競合したら今取った新しいデータを優先（-X theirs＝自分の変更）。失敗したら取り直して再挑戦
for i in 1 2 3; do git pull -q --rebase -X theirs && git push -q && break; git rebase --abort 2>/dev/null; sleep 5; done
