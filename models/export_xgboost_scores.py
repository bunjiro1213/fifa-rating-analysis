from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from joblib import parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "datasets" / "cleaned.csv"
SCORED_PATH = BASE_DIR / "datasets" / "scored_players.csv"
METRICS_PATH = BASE_DIR / "datasets" / "model_metrics.csv"


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_Bucket"] = pd.cut(
        df["Age"],
        bins=[0, 20, 23, 27, 31, 100],
        labels=["<=20", "21-23", "24-27", "28-31", "32+"],
        right=True,
    )

    def to_position_group(pos: str) -> str:
        if not isinstance(pos, str):
            return "Other"
        if "GK" in pos:
            return "Goalkeeper"
        if any(token in pos for token in ["CB", "RB", "LB", "RWB", "LWB"]):
            return "Defense"
        if any(token in pos for token in ["DM", "CM", "AM", "RM", "LM"]):
            return "Midfield"
        if any(token in pos for token in ["ST", "CF", "LW", "RW", "F", "S"]):
            return "Attack"
        return "Other"

    df["Position_Group"] = df["Position"].apply(to_position_group)

    top_clubs = set(df["Club"].value_counts().head(15).index)
    df["Club_Group"] = df["Club"].where(df["Club"].isin(top_clubs), "Other")
    return df


def enforce_numeric_types(scored: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric fields are written as numbers for downstream tools (e.g., Power BI)."""
    scored = scored.copy()
    numeric_schema = {
        "ID": "int64",
        "Age": "int64",
        "Overall": "int64",
        "Predicted_Overall": "float64",
        "Residual": "float64",
    }
    for col, dtype in numeric_schema.items():
        cleaned = (
            scored[col]
            .astype(str)
            .str.replace(r"[,$]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
        if cleaned.isnull().any():
            raise ValueError(f"Found non-numeric values in column '{col}' after cleaning.")
        if dtype.startswith("int"):
            cleaned = cleaned.round().astype(dtype)
        else:
            cleaned = cleaned.astype(dtype)
        scored[col] = cleaned
    return scored


def build_pipeline(feature_df: pd.DataFrame) -> Pipeline:
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns
    numeric_cols = feature_df.columns.difference(categorical_cols)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("xgb", model)])


def evaluate_and_score() -> None:
    df = pd.read_csv(DATA_PATH)
    target = df["Overall"]
    features = df.drop(columns=["Overall", "ID", "Name"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(features)

    param_grid = {
        "xgb__n_estimators": [400, 800],
        "xgb__learning_rate": [0.05, 0.1],
        "xgb__max_depth": [4, 6],
        "xgb__subsample": [0.8, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0],
        "xgb__min_child_weight": [1, 5],
        "xgb__reg_lambda": [1.0, 5.0],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=4,
        refit=True,
    )
    with parallel_backend("threading"):
        grid.fit(X_train, y_train)

    best_model: Pipeline = grid.best_estimator_

    test_preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)

    full_preds = best_model.predict(features)
    scored = df[["ID", "Name", "Nationality", "Club", "Position", "Age", "Overall"]].copy()
    scored["Predicted_Overall"] = pd.Series(full_preds).round(2).astype(float)
    scored["Residual"] = (scored["Overall"].astype(float) - scored["Predicted_Overall"]).round(2)
    scored = add_segments(scored)
    scored = enforce_numeric_types(scored)

    scored.to_csv(SCORED_PATH, index=False)

    metrics = pd.DataFrame(
        [
            {
                "model": "xgboost_regressor_grid",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "best_params": grid.best_params_,
                "best_cv_rmse": float((-grid.best_score_) ** 0.5),
            }
        ]
    )
    metrics.to_csv(METRICS_PATH, index=False)

    print(
        f"Saved scores -> {SCORED_PATH}\n"
        f"Saved metrics -> {METRICS_PATH}\n"
        f"RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}"
    )


if __name__ == "__main__":
    evaluate_and_score()
