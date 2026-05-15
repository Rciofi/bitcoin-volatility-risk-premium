"""
train_linear_cap6.py
--------------------
Treina modelos lineares (MQO, Ridge, LASSO) para o Capítulo 6 da dissertação
sobre o Prêmio de Risco de Volatilidade do Bitcoin (BVRP).

Correções em relação ao script original (não versionado):
  1. Remove variável `close` (série I(1) — antecipa Tarefa E3).
  2. Ridge e LASSO com seleção de alpha via TimeSeriesSplit(n_splits=5, gap=30)
     dentro do conjunto de treino (IC4).
     gap=30: embargo de 30 dias para mitigar vazamento por autocorrelação
     serial (BVRP usa janela móvel de 30 dias — materiais Prof. Murilo, seção 4.5).
  3. Divisão temporal 70/30 (treino=1.228 obs / teste=527 obs),
     consistente com regenerate_cap8.py.

Outputs gerados:
  - chapters/cap6_ml/tables/linear_models_cap6.csv          [SOBRESCREVE]
  - chapters/cap6_ml/tables/linear_model_coefficients_cap6.csv [SOBRESCREVE]
  - chapters/cap6_ml/tables/linear_cv_results_cap6.csv      [NOVO]
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "ml_dataset.csv")
OUT_DIR   = os.path.join(ROOT, "chapters", "cap6_ml", "tables")

TARGET = "ret_fut_1d"

# Variável `close` excluída: série I(1), introduz viés espúrio (Tarefa E3)
FEATURES = [
    "ret",
    "rv_30d",
    "iv_30d",
    "vrp_30d",
    "ret_lag_1",
    "ret_lag_5",
    "ret_lag_20",
    "d_iv_1d",
    "d_vrp_1d",
    "month",
    "weekday",
    "is_month_start",
    "is_month_end",
    "vrp_regime_num",
]

RIDGE_ALPHAS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
LASSO_ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1]
N_SPLITS_CV  = 5
GAP_EMBARGO  = 30    # dias de embargo entre treino e val no walk-forward CV
TRAIN_RATIO  = 0.70  # 1.228 treino / 527 teste


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features_available = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(features_available)
    if missing:
        print(f"[AVISO] Features ausentes no dataset: {missing}")

    df = df.dropna(subset=features_available + [TARGET]).reset_index(drop=True)
    return df, features_available


def split_data(df, features):
    n       = len(df)
    n_train = int(n * TRAIN_RATIO)

    train = df.iloc[:n_train].copy()
    test  = df.iloc[n_train:].copy()

    X_train = train[features].values
    y_train = train[TARGET].values
    X_test  = test[features].values
    y_test  = test[TARGET].values

    return X_train, y_train, X_test, y_test, train, test


def scale(X_train, X_test):
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def calc_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def select_alpha_tscv(model_class, alphas, X_train_s, y_train,
                      n_splits, gap, model_kwargs=None):
    """
    Seleciona alpha ótimo via TimeSeriesSplit dentro do treino.
    Critério: menor RMSE médio entre os n_splits folds.

    gap=30: embargo de 30 observações entre fim do treino e início da
    validação em cada fold — mitiga vazamento por autocorrelação serial
    (janela móvel de 30 dias do BVRP). Vide materiais Prof. Murilo, sec. 4.5.

    Nunca toca o conjunto de teste — isolamento total do holdout externo.
    """
    if model_kwargs is None:
        model_kwargs = {}

    tscv       = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    cv_records = []

    for alpha in alphas:
        rmses = []
        for tr_idx, val_idx in tscv.split(X_train_s):
            X_tr, y_tr = X_train_s[tr_idx], y_train[tr_idx]
            X_vl, y_vl = X_train_s[val_idx], y_train[val_idx]

            model = model_class(alpha=alpha, **model_kwargs)
            model.fit(X_tr, y_tr)
            preds     = model.predict(X_vl)
            rmse_fold = np.sqrt(mean_squared_error(y_vl, preds))
            rmses.append(rmse_fold)

        cv_records.append({
            "alpha":        alpha,
            "rmse_cv_mean": np.mean(rmses),
            "rmse_cv_std":  np.std(rmses),
        })

    cv_df      = pd.DataFrame(cv_records)
    best_alpha = cv_df.loc[cv_df["rmse_cv_mean"].idxmin(), "alpha"]
    return best_alpha, cv_df


# =====================================================================
# MAIN
# =====================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Carrega e divide dados
    # ------------------------------------------------------------------
    df, features = load_data()
    n       = len(df)
    n_train = int(n * TRAIN_RATIO)

    print("=" * 62)
    print("TREINAMENTO — MODELOS LINEARES CAP. 6")
    print("=" * 62)
    print(f"Dataset : {DATA_PATH}")
    print(f"N total : {n} obs  |  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Treino  : {n_train} obs  |  Teste: {n - n_train} obs  (70/30)")
    print(f"Features: {len(features)} variaveis  (sem `close`)")
    print(f"Target  : {TARGET}")
    print(f"CV      : TimeSeriesSplit(n_splits={N_SPLITS_CV}, gap={GAP_EMBARGO})")

    X_train, y_train, X_test, y_test, train_df, test_df = split_data(df, features)
    X_train_s, X_test_s, scaler = scale(X_train, X_test)

    print(f"\nTreino : {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"Teste  : {test_df['date'].min().date()}  -> {test_df['date'].max().date()}")

    # ------------------------------------------------------------------
    # 2. MQO (sem regularizacao)
    # ------------------------------------------------------------------
    print("\n--- MQO (sem regularizacao) ---")
    ols = LinearRegression()
    ols.fit(X_train_s, y_train)

    ols_tr = calc_metrics(y_train, ols.predict(X_train_s))
    ols_te = calc_metrics(y_test,  ols.predict(X_test_s))

    print(f"  Treino -> MAE={ols_tr[0]:.6f}  RMSE={ols_tr[1]:.6f}  R2={ols_tr[2]:.6f}")
    print(f"  Teste  -> MAE={ols_te[0]:.6f}  RMSE={ols_te[1]:.6f}  R2={ols_te[2]:.6f}")

    # ------------------------------------------------------------------
    # 3. Ridge — selecao de alpha via TimeSeriesSplit com embargo
    # ------------------------------------------------------------------
    print(f"\n--- Ridge  (TimeSeriesSplit n_splits={N_SPLITS_CV}, gap={GAP_EMBARGO}) ---")
    print(f"  Grid testado: {RIDGE_ALPHAS}")

    best_ridge_alpha, ridge_cv_df = select_alpha_tscv(
        Ridge, RIDGE_ALPHAS, X_train_s, y_train, N_SPLITS_CV, GAP_EMBARGO
    )
    ridge_cv_df["modelo"] = "Ridge"

    best_ridge_rmse = ridge_cv_df.loc[
        ridge_cv_df["alpha"] == best_ridge_alpha, "rmse_cv_mean"
    ].values[0]
    print(f"  Alpha otimo : {best_ridge_alpha:.4g}  "
          f"(RMSE_CV medio: {best_ridge_rmse:.6f})")

    ridge = Ridge(alpha=best_ridge_alpha)
    ridge.fit(X_train_s, y_train)

    ridge_tr = calc_metrics(y_train, ridge.predict(X_train_s))
    ridge_te = calc_metrics(y_test,  ridge.predict(X_test_s))

    print(f"  Treino -> MAE={ridge_tr[0]:.6f}  RMSE={ridge_tr[1]:.6f}  R2={ridge_tr[2]:.6f}")
    print(f"  Teste  -> MAE={ridge_te[0]:.6f}  RMSE={ridge_te[1]:.6f}  R2={ridge_te[2]:.6f}")

    # ------------------------------------------------------------------
    # 4. LASSO — selecao de alpha via TimeSeriesSplit com embargo
    # ------------------------------------------------------------------
    print(f"\n--- LASSO  (TimeSeriesSplit n_splits={N_SPLITS_CV}, gap={GAP_EMBARGO}) ---")
    print(f"  Grid testado: {LASSO_ALPHAS}")

    best_lasso_alpha, lasso_cv_df = select_alpha_tscv(
        Lasso, LASSO_ALPHAS, X_train_s, y_train, N_SPLITS_CV, GAP_EMBARGO,
        model_kwargs={"max_iter": 10000}
    )
    lasso_cv_df["modelo"] = "Lasso"

    best_lasso_rmse = lasso_cv_df.loc[
        lasso_cv_df["alpha"] == best_lasso_alpha, "rmse_cv_mean"
    ].values[0]
    print(f"  Alpha otimo : {best_lasso_alpha:.4g}  "
          f"(RMSE_CV medio: {best_lasso_rmse:.6f})")

    lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
    lasso.fit(X_train_s, y_train)

    lasso_tr = calc_metrics(y_train, lasso.predict(X_train_s))
    lasso_te = calc_metrics(y_test,  lasso.predict(X_test_s))

    print(f"  Treino -> MAE={lasso_tr[0]:.6f}  RMSE={lasso_tr[1]:.6f}  R2={lasso_tr[2]:.6f}")
    print(f"  Teste  -> MAE={lasso_te[0]:.6f}  RMSE={lasso_te[1]:.6f}  R2={lasso_te[2]:.6f}")

    # ------------------------------------------------------------------
    # 5. Monta tabelas de saida
    # ------------------------------------------------------------------

    # 5a. Metricas (linear_models_cap6.csv)
    rows = [
        {"model": "MQO",
         "set": "train", "MAE": ols_tr[0], "RMSE": ols_tr[1], "R2": ols_tr[2]},
        {"model": "MQO",
         "set": "test",  "MAE": ols_te[0], "RMSE": ols_te[1], "R2": ols_te[2]},
        {"model": f"Ridge(alpha={best_ridge_alpha:.2e})",
         "set": "train", "MAE": ridge_tr[0], "RMSE": ridge_tr[1], "R2": ridge_tr[2]},
        {"model": f"Ridge(alpha={best_ridge_alpha:.2e})",
         "set": "test",  "MAE": ridge_te[0], "RMSE": ridge_te[1], "R2": ridge_te[2]},
        {"model": f"Lasso(alpha={best_lasso_alpha:.2e})",
         "set": "train", "MAE": lasso_tr[0], "RMSE": lasso_tr[1], "R2": lasso_tr[2]},
        {"model": f"Lasso(alpha={best_lasso_alpha:.2e})",
         "set": "test",  "MAE": lasso_te[0], "RMSE": lasso_te[1], "R2": lasso_te[2]},
    ]
    df_metrics = pd.DataFrame(rows)

    # 5b. Coeficientes (linear_model_coefficients_cap6.csv)
    df_coefs = pd.DataFrame({
        "feature": ["intercept"] + features,
        "MQO":     [ols.intercept_]   + list(ols.coef_),
        "Ridge":   [ridge.intercept_] + list(ridge.coef_),
        "Lasso":   [lasso.intercept_] + list(lasso.coef_),
    })

    # 5c. Resultados CV por alpha (linear_cv_results_cap6.csv) — NOVO
    df_cv = pd.concat([ridge_cv_df, lasso_cv_df], ignore_index=True)[
        ["modelo", "alpha", "rmse_cv_mean", "rmse_cv_std"]
    ]

    # ------------------------------------------------------------------
    # 6. Salva CSVs
    # ------------------------------------------------------------------
    path_metrics = os.path.join(OUT_DIR, "linear_models_cap6.csv")
    path_coefs   = os.path.join(OUT_DIR, "linear_model_coefficients_cap6.csv")
    path_cv      = os.path.join(OUT_DIR, "linear_cv_results_cap6.csv")

    df_metrics.to_csv(path_metrics, index=False)
    df_coefs.to_csv(path_coefs,     index=False)
    df_cv.to_csv(path_cv,           index=False)

    # ------------------------------------------------------------------
    # 7. Sumario final
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("RESULTADOS FINAIS — CONJUNTO DE TESTE (holdout externo)")
    print("=" * 62)
    print(df_metrics[df_metrics["set"] == "test"].to_string(index=False))

    print(f"\nAlpha otimo Ridge : {best_ridge_alpha:.4g}")
    print(f"Alpha otimo Lasso : {best_lasso_alpha:.4g}")

    n_zero_lasso = int((np.abs(lasso.coef_) < 1e-10).sum())
    print(f"Coeficientes zerados pelo LASSO: {n_zero_lasso}/{len(features)}")

    print(f"\nArquivos salvos em: {OUT_DIR}")
    print(f"  -> {os.path.basename(path_metrics)}")
    print(f"  -> {os.path.basename(path_coefs)}")
    print(f"  -> {os.path.basename(path_cv)}  [NOVO]")


if __name__ == "__main__":
    main()
