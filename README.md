# FIFA Player Overall Prediction

End-to-end workflow for cleaning FIFA player data, exploring relationships, training regression models, and exporting scored players for downstream visualization or valuation analysis.

## Repository structure
- `datasets/` – Source and derived CSVs: `cleaned.csv` (model input), `scored_players.csv` (model output with residuals), `model_metrics.csv` (evaluation summary), `fifa_eda_stats.csv` (aggregated EDA stats).
- `models/` – Modeling notebooks (ridge, random forest, k-NN, neural net, XGBoost) and `export_xgboost_scores.py` for training the tuned XGBoost model and writing predictions.
- `cleaning.ipynb`, `merged.ipynb` – Data wrangling and merge steps prior to modeling.
- `exploratory.ipynb` – EDA and feature investigation.

## Train & score
Run the XGBoost pipeline end-to-end; it reads `datasets/cleaned.csv`, performs a parameter search, evaluates, and writes scored players plus metrics.

Outputs:
- `datasets/scored_players.csv` with `Predicted_Overall` and `Residual = Overall - Predicted_Overall` for each player.
- Console summary of RMSE/MAE/R2; the same metrics are stored in `datasets/model_metrics.csv` (includes CV best params and row counts).

## Power BI Model Evaluation Dashboard

The final model results are shown in a Power BI dashboard, which is exported as a PDF for easy viewing.

**What the report shows:**
- A scatter plot comparing **actual vs predicted player ratings**
- A plot of **prediction errors (residuals) vs predicted ratings** to check for bias
- **Average prediction error by position** to see where the model performs better or worse
- A table of **players with the largest over- and under-predictions**

The dashboard uses `datasets/scored_players.csv` as its data source and focuses on clearly explaining model performance and errors.

## Notebooks
- Use the notebooks in `models/` to inspect alternative algorithms and tuning experiments.
- `exploratory.ipynb` highlights feature importance and correlations; `cleaning.ipynb` documents the preprocessing pipeline prior to model training.
