#!/usr/bin/env bash
# 1回分の更新：最新情報の取得 → 予想の計算 → 公開
set -u
python scraper/update.py || echo "取得でエラー（続行）"
n=$(python -c "import pandas as pd; e=pd.read_csv('data/csv/entries.csv'); print(e[e.finish.notna()].race_id.nunique())" 2>/dev/null || echo 0)
if [ "$n" -ge 1500 ]; then
  python pipeline/model.py data/csv/ data/app_data.json && python pipeline/log_predictions.py && python pipeline/analyze.py && python pipeline/export.py || echo "計算でエラー"
fi
git add -A data docs
git commit -qm "update: $(TZ=Asia/Tokyo date '+%m/%d %H:%M')" || echo "no changes"
git pull -q --rebase -X ours || true
git push -q || echo "push失敗"
