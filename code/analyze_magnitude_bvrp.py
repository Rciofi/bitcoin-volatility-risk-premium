"""
Teste de magnitude (|ret_fut_30d|) contra BVRP, na fonte canonica do Cap. 5.

Motivacao: o teste anterior (test_log_transform_cap5.py) rodou em
code/data/dataset_bvrp_with_skew.csv -- dataset arquivado (identico ao que
esta em data/archive/), N=1288, com vrp_30d calculado em formula/escala
DIFERENTE da canonica (corr=-0.51 com o vrp_30d de vrp_with_targets.csv
nas mesmas datas). O p=0.003 relatado la nao e comparavel ao resto do
Cap. 5 e pode nem existir na variavel certa.

Este script refaz o teste em data/vrp_with_targets.csv (N=1.775, mesma
fonte de regenerate_cap5.py), e testa robustez via:
  1. OLS HAC + Breusch-Pagan (paralelo ao teste original)
  2. Moving Block Bootstrap (mesma funcao de regenerate_cap5.py)
  3. Decomposicao RV/IV (mesmo molde do modelo E1)

Uso: py code/analyze_magnitude_bvrp.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "vrp_with_targets.csv"   # fonte canonica (1.775 obs)

BVRP_COL = "vrp_30d"
RV_COL   = "rv_30d"
IV_COL   = "iv_30d"
RET_COL  = "ret_fut_30d"
HAC_LAGS = 30
BLOCK_SIZE  = 30
N_BOOTSTRAP = 1000
SEED = 42

df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df = df.dropna(subset=[BVRP_COL, RET_COL, RV_COL, IV_COL]).reset_index(drop=True)
print("=" * 70)
print("Dataset: %s | N=%d | %s a %s" % (
    DATA.name, len(df), df["date"].min().date(), df["date"].max().date()))
print("BVRP (vrp_30d): mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
    df[BVRP_COL].mean(), df[BVRP_COL].std(), df[BVRP_COL].min(), df[BVRP_COL].max()))

y_abs = df[RET_COL].abs()

# ---------------------------------------------------------------------------
# 1. OLS HAC + Breusch-Pagan: |ret_fut_30d| ~ BVRP
# ---------------------------------------------------------------------------
def run_ols_hac(y, X, lags):
    Xc = sm.add_constant(X)
    mask = y.notna() & Xc.notna().all(axis=1)
    return sm.OLS(y[mask], Xc[mask]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

res = run_ols_hac(y_abs, df[[BVRP_COL]], lags=HAC_LAGS)
beta, t, p, r2, n = (res.params[BVRP_COL], res.tvalues[BVRP_COL],
                      res.pvalues[BVRP_COL], res.rsquared, int(res.nobs))

res_ols = sm.OLS(y_abs, sm.add_constant(df[[BVRP_COL]])).fit()
bp_stat, bp_p, _, _ = het_breuschpagan(res_ols.resid, res_ols.model.exog)

print("\n--- 1. OLS HAC: |ret_fut_30d| ~ BVRP (canonico) ---")
print("  N=%d  beta=%+.6f  t=%+.3f  p=%.4f  R2=%.4f" % (n, beta, t, p, r2))
print("  Breusch-Pagan: stat=%.3f  p=%.4f  (%s)" % (
    bp_stat, bp_p, "HETEROCED." if bp_p < 0.05 else "homocedastico"))

# ---------------------------------------------------------------------------
# 2. Moving Block Bootstrap (identico a regenerate_cap5.py)
# ---------------------------------------------------------------------------
def moving_block_bootstrap_beta(y, X_df, bvrp_col,
                                 block_size=30, n_bootstrap=1000, seed=42):
    Xc        = sm.add_constant(X_df)
    mask      = y.notna() & Xc.notna().all(axis=1)
    y_arr     = y[mask].values
    X_arr     = Xc[mask].values
    n         = len(y_arr)
    col_names = list(Xc[mask].columns)
    bvrp_idx  = col_names.index(bvrp_col)

    rng       = np.random.default_rng(seed)
    betas     = np.empty(n_bootstrap)
    n_blocks  = int(np.ceil(n / block_size))
    max_start = n - block_size

    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        coefs  = np.linalg.lstsq(X_arr[idx], y_arr[idx], rcond=None)[0]
        betas[b] = coefs[bvrp_idx]
    return betas

betas_b  = moving_block_bootstrap_beta(
    y_abs, df[[BVRP_COL]], bvrp_col=BVRP_COL,
    block_size=BLOCK_SIZE, n_bootstrap=N_BOOTSTRAP, seed=SEED)
ic_lower = float(np.percentile(betas_b, 2.5))
ic_upper = float(np.percentile(betas_b, 97.5))
p_boot   = float(min((betas_b <= 0).mean() if beta >= 0 else (betas_b >= 0).mean(), 0.5) * 2)

print("\n--- 2. Moving Block Bootstrap (block=%d, B=%d, seed=%d) ---" % (
    BLOCK_SIZE, N_BOOTSTRAP, SEED))
print("  beta_hat=%+.6f  IC95=[%+.6f, %+.6f]  p_hac=%.4f  p_boot=%.4f" % (
    beta, ic_lower, ic_upper, p, p_boot))
print("  Zero dentro do IC bootstrap? %s" % ("SIM" if ic_lower < 0 < ic_upper else "NAO"))

# ---------------------------------------------------------------------------
# 3. Decomposicao RV/IV (molde E1): |ret_fut_30d| ~ RV + IV, sem BVRP
# ---------------------------------------------------------------------------
res_dec = run_ols_hac(y_abs, df[[RV_COL, IV_COL]], lags=HAC_LAGS)
b1, b2 = res_dec.params[RV_COL], res_dec.params[IV_COL]
t1, t2 = res_dec.tvalues[RV_COL], res_dec.tvalues[IV_COL]
p1, p2 = res_dec.pvalues[RV_COL], res_dec.pvalues[IV_COL]
simetria = abs(b1 + b2) / (abs(b1) + abs(b2) + 1e-15) * 100

print("\n--- 3. Decomposicao RV/IV: |ret_fut_30d| ~ RV + IV (sem BVRP) ---")
print("  beta_RV=%+.6f  t=%+.3f  p=%.4f" % (b1, t1, p1))
print("  beta_IV=%+.6f  t=%+.3f  p=%.4f" % (b2, t2, p2))
print("  beta1+beta2=%+.6f  simetria=%.1f%%  R2=%.4f  N=%d" % (
    b1 + b2, simetria, res_dec.rsquared, int(res_dec.nobs)))
print("  Interpretacao: simetria baixa (%s) -> %s" % (
    "beta_RV ~= -beta_IV" if simetria < 20 else "beta_RV e beta_IV NAO sao opostos",
    "BVRP capta a combinacao relevante" if simetria < 20
    else "sinal pode vir de RV ou IV isoladamente, nao do BVRP como diferenca"))

print("\n" + "=" * 70)
print("RESUMO")
print("  |y| ~ BVRP (canonico, N=%d): p_HAC=%.4f  p_MBB=%.4f" % (n, p, p_boot))
print("  Dominancia: |beta_RV|=%.6f vs |beta_IV|=%.6f -> %s domina" % (
    abs(b1), abs(b2), "RV" if abs(b1) > abs(b2) else "IV"))
print("=" * 70)
