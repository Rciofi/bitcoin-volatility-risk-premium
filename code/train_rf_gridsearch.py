import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # Caminho base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "ml_dataset.csv")

    print(f"Lendo dataset de ML: {data_path}")
    df = pd.read_csv(data_path)

    # Ordena temporalmente
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    target_col = "ret_fut_5d"

    feature_cols = [
        "rv_30d", "iv_30d", "vrp_30d", "vrp_regime_num",
        "ret", "ret_lag_1", "ret_lag_5", "ret_lag_20",
        "d_iv_1d", "d_vrp_1d",
        "month", "weekday", "is_month_start", "is_month_end"
    ]

    df = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df[feature_cols].values
    y = df[target_col].values

    # --------------------------
    # Train-test split (80/20)
    # --------------------------
    n = len(df)
    split_idx = int(n * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTreino: {len(X_train)} observações")
    print(f"Teste : {len(X_test)} observações")

    # --------------------------
    # GRIDSEARCH COM TIME SERIES SPLIT
    # --------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [2, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    print("\nRodando GridSearchCV (TimeSeriesSplit)...")
    grid.fit(X_train, y_train)

    print("\n=== MELHORES PARÂMETROS DO RANDOM FOREST ===")
    print(grid.best_params_)

    # --------------------------
    # Avaliar no conjunto de teste
    # --------------------------
    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== RESULTADOS OTIMIZADOS - RANDOM FOREST ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² : {r2:.4f}")

    # --------------------------
    # Importância das variáveis
    # --------------------------
    importances = best_rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    out_imp = os.path.join(
        base_dir, "data", "rf_gridsearch_feature_importance.csv")
    imp_df.to_csv(out_imp, index=False)

    print("\nImportância das features salva em:")
    print(out_imp)


if __name__ == "__main__":
    main()
