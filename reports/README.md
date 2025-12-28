# Power BI hook-up

Use the exported score files to build visuals quickly:

- Source data: `datasets/scored_players.csv` (player-level predictions) and `datasets/model_metrics.csv` (RMSE/MAE/R2 for KPI cards).
- Refresh the scores: `venv/bin/python models/export_xgboost_scores.py`.
- In Power BI Desktop: Get Data → CSV → point to `datasets/scored_players.csv`; optionally also `datasets/model_metrics.csv`.
- Recommended visuals:
  - Scatter: Actual `Overall` (Y) vs `Predicted_Overall` (X). Add diagonal reference line (identity) if desired.
  - Histogram: `Residual`.
  - Top N table: sort by `Residual` descending for undervalued; ascending for overvalued. Include `Name`, `Club`, `Position`, `Age_Bucket`.
  - Slicers: `Position_Group`, `Club_Group`, `Nationality`, `Age_Bucket`.
  - KPI cards: `rmse`, `mae`, `r2` from `model_metrics.csv`.
- Suggested filters/fields are precomputed in the CSV: `Age_Bucket`, `Position_Group`, `Club_Group`, `Residual = Overall - Predicted_Overall`.
