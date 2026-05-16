import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def main():
    # Diretório base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "ml_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo dataset de ML em: {data_path}")
    df = pd.read_csv(data_path)

    # Ordena temporalmente
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target: retorno futuro 5D
    target_col = "ret_fut_5d"

    # Mesmas features do baseline
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

    # Limpa NaNs em features/target
    data = df.dropna(subset=feature_cols + [target_col]).copy()

    X = data[feature_cols].values
    y = data[target_col].values
    dates = data["date"].values

    # Split temporal 80 / 20
    n = len(data)
    split_idx = int(n * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    print(f"\nTotal de observações: {n}")
    print(f"Treino: {len(X_train)} ({dates_train[0]} -> {dates_train[-1]})")
    print(f"Teste : {len(X_test)} ({dates_test[0]} -> {dates_test[-1]})")

    # ==========================
    # Modelo XGBoost Regressor
    # ==========================

    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
    )

    print("\nTreinando XGBoost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== RESULTADOS - XGBoost (ret_fut_5d) ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² : {r2:.4f}")

    # ==========================
    # Gráfico Real vs Previsto
    # ==========================

    figs_dir = os.path.join(base_dir, "figuras")
    os.makedirs(figs_dir, exist_ok=True)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.axline((0, 0), slope=1, linestyle="--")
    ax.set_xlabel("Retorno futuro 5D real")
    ax.set_ylabel("Retorno futuro 5D previsto (XGBoost)")
    ax.set_title("XGBoost - Predição de retorno futuro 5D")

    plt.tight_layout()
    out_fig = os.path.join(figs_dir, "xgb_ret_fut_5d_real_vs_pred.png")
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"\nGráfico Real vs Previsto salvo em: {out_fig}")

    # ==========================
    # Importância das variáveis
    # ==========================

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    out_imp = os.path.join(base_dir, "data", "xgb_feature_importance.csv")
    imp_df.to_csv(out_imp, index=False)

    print("\nImportância das variáveis (XGBoost):")
    print(imp_df)
    print(f"\nTabela de importância salva em: {out_imp}")


if __name__ == "__main__":
    main()
