import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    # Diretório base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "ml_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo dataset de ML em: {data_path}")
    df = pd.read_csv(data_path)

    # -----------------------------
    # 1) Preparar dados
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target: retorno futuro de 5 dias
    target_col = "ret_fut_5d"

    # Features numéricas escolhidas
    feature_cols = [
        "rv_30d",
        "iv_30d",
        "vrp_30d",
        "vrp_regime_num",
        "ret",
        "ret_lag_1",
        "ret_lag_5",
        "ret_lag_20",
        "d_iv_1d",
        "d_vrp_1d",
        "month",
        "weekday",
        "is_month_start",
        "is_month_end",
    ]

    # Remove linhas com NaN em features ou target
    data = df.dropna(subset=feature_cols + [target_col]).copy()

    X = data[feature_cols].values
    y = data[target_col].values
    dates = data["date"].values

    # Split temporal 80% / 20%
    n = len(data)
    split_idx = int(n * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    print(f"\nTotal de observações: {n}")
    print(f"Treino: {len(X_train)} ({dates_train[0]} -> {dates_train[-1]})")
    print(f"Teste : {len(X_test)} ({dates_test[0]} -> {dates_test[-1]})")

    # -----------------------------
    # 2) Modelo 1: Regressão Linear
    # -----------------------------
    lin_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    lin_pipeline.fit(X_train, y_train)
    y_pred_lin = lin_pipeline.predict(X_test)

    mse_lin = mean_squared_error(y_test, y_pred_lin)
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    r2_lin = r2_score(y_test, y_pred_lin)

    print("\n=== RESULTADOS - REGRESSÃO LINEAR (ret_fut_5d) ===")
    print(f"MSE: {mse_lin:.6f}")
    print(f"MAE: {mae_lin:.6f}")
    print(f"R² : {r2_lin:.4f}")

    # -----------------------------
    # 3) Modelo 2: Random Forest
    # -----------------------------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\n=== RESULTADOS - RANDOM FOREST (ret_fut_5d) ===")
    print(f"MSE: {mse_rf:.6f}")
    print(f"MAE: {mae_rf:.6f}")
    print(f"R² : {r2_rf:.4f}")

    # -----------------------------
    # 4) Gráfico: Real vs Previsto (Random Forest)
    # -----------------------------
    figs_dir = os.path.join(base_dir, "figuras")
    os.makedirs(figs_dir, exist_ok=True)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_test, y_pred_rf, alpha=0.5, s=20)
    ax.axline((0, 0), slope=1, linestyle="--")
    ax.set_xlabel("Retorno futuro 5D real")
    ax.set_ylabel("Retorno futuro 5D previsto (RF)")
    ax.set_title(
        "Random Forest - VRP e variáveis de estado\nPredição de retorno futuro 5D")

    plt.tight_layout()
    out_fig = os.path.join(figs_dir, "rf_ret_fut_5d_real_vs_pred.png")
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"\nGráfico Real vs Previsto salvo em: {out_fig}")

    # -----------------------------
    # 5) Importância das features (Random Forest)
    # -----------------------------
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    out_imp = os.path.join(base_dir, "data", "rf_feature_importance.csv")
    imp_df.to_csv(out_imp, index=False)

    print("\nImportância das variáveis (Random Forest):")
    print(imp_df)
    print(f"\nTabela de importância salva em: {out_imp}")


if __name__ == "__main__":
    main()
