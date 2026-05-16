"""
Regenera os scatter plots reais vs. previstos do cap6 com o alvo correto (BVRP_{t+1}).
Substitui fig_6_11_rf_real_vs_pred.png e fig_6_12_xgb_real_vs_pred.png.
"""
import pathlib, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bvrp_ml_target_fut_1d.csv"
FIGS = ROOT / "figs" / "cap6"
FIGS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1) Carregar dados
# ------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"Dataset: {df.shape[0]} obs | {df['date'].min().date()} a {df['date'].max().date()}")

TARGET    = "bvrp_fut_1d"
FEATURES  = ["close", "ret", "rv_30d", "iv_30d"]

df = df.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)
n = len(df)
print(f"Apos dropna: {n} obs")

# ------------------------------------------------------------------
# 2) Split temporal 70/15/15
# ------------------------------------------------------------------
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train = df.iloc[:n_train]
val   = df.iloc[n_train : n_train + n_val]
test  = df.iloc[n_train + n_val :]

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Train: {train['date'].min().date()} a {train['date'].max().date()}")
print(f"Val  : {val['date'].min().date()}   a {val['date'].max().date()}")
print(f"Test : {test['date'].min().date()}  a {test['date'].max().date()}")

# ------------------------------------------------------------------
# 3) Escalonamento (apenas estatisticas do treino)
# ------------------------------------------------------------------
scaler = StandardScaler()
Xtr = scaler.fit_transform(train[FEATURES])
Xv  = scaler.transform(val[FEATURES])
Xt  = scaler.transform(test[FEATURES])
ytr, yv, yt = train[TARGET].values, val[TARGET].values, test[TARGET].values

def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ------------------------------------------------------------------
# 4) Random Forest (hiperparametros do notebook original)
# ------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(Xtr, ytr)

rf_mae_v, rf_rmse_v, rf_r2_v = metrics(yv, rf.predict(Xv))
rf_mae_t, rf_rmse_t, rf_r2_t = metrics(yt, rf.predict(Xt))
print(f"\nRandom Forest:")
print(f"  Val : MAE={rf_mae_v:.3f} RMSE={rf_rmse_v:.3f} R2={rf_r2_v:.3f}")
print(f"  Test: MAE={rf_mae_t:.3f} RMSE={rf_rmse_t:.3f} R2={rf_r2_t:.3f}")

# ------------------------------------------------------------------
# 5) XGBoost — busca rapida para reproduzir metricas originais
# ------------------------------------------------------------------
# Metricas alvo: Test MAE=2.457, RMSE=3.348, R2=0.854
param_grid = {
    "n_estimators":    [300, 600],
    "max_depth":       [2, 3, 4],
    "learning_rate":   [0.03, 0.1],
    "subsample":       [0.7, 1.0],
    "colsample_bytree":[0.7, 1.0],
}

best_rmse_xgb = np.inf
best_xgb = None
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    m.fit(Xtr, ytr)
    rmse_v = np.sqrt(mean_squared_error(yv, m.predict(Xv)))
    if rmse_v < best_rmse_xgb:
        best_rmse_xgb = rmse_v
        best_xgb = m

xgb_mae_v, xgb_rmse_v, xgb_r2_v = metrics(yv, best_xgb.predict(Xv))
xgb_mae_t, xgb_rmse_t, xgb_r2_t = metrics(yt, best_xgb.predict(Xt))
print(f"\nXGBoost:")
print(f"  Val : MAE={xgb_mae_v:.3f} RMSE={xgb_rmse_v:.3f} R2={xgb_r2_v:.3f}")
print(f"  Test: MAE={xgb_mae_t:.3f} RMSE={xgb_rmse_t:.3f} R2={xgb_r2_t:.3f}")

# ------------------------------------------------------------------
# 6) Scatter plots reais vs. previstos (conjunto de teste)
# ------------------------------------------------------------------
def scatter_real_vs_pred(y_true, y_pred, title, xlabel, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.55, s=20, color="#1f77b4")
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="Linha perfeita")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {outpath}")

rf_pred_t  = rf.predict(Xt)
xgb_pred_t = best_xgb.predict(Xt)

scatter_real_vs_pred(
    yt, rf_pred_t,
    title="Random Forest — valores reais vs. previstos (teste)",
    xlabel=r"BVRP$_{t+1}$ real (p.p.)",
    ylabel=r"BVRP$_{t+1}$ previsto — RF (p.p.)",
    outpath=FIGS / "fig_6_11_rf_real_vs_pred.png",
)

scatter_real_vs_pred(
    yt, xgb_pred_t,
    title="XGBoost — valores reais vs. previstos (teste)",
    xlabel=r"BVRP$_{t+1}$ real (p.p.)",
    ylabel=r"BVRP$_{t+1}$ previsto — XGBoost (p.p.)",
    outpath=FIGS / "fig_6_12_xgb_real_vs_pred.png",
)

print("\nConcluido.")
