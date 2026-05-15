"""
Teste rápido: transformação log/arcsinh no eixo y para o scatter BVRP x ret_fut_30d.
Sugestão do orientador (Cap. 5, Figura 5.2).

Uso: py code/test_log_transform_cap5.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "dataset_bvrp_with_skew.csv"
OUT  = ROOT / "figs" / "cap5"
OUT.mkdir(parents=True, exist_ok=True)

BVRP_COL = "vrp_30d"
RET_COL  = "ret_fut_30d"
HAC_LAGS = 30

df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
tmp = df[[BVRP_COL, RET_COL]].dropna().reset_index(drop=True)

x = tmp[BVRP_COL].values
y = tmp[RET_COL].values

print("=" * 60)
print(f"N = {len(tmp)}")
print(f"BVRP: min={x.min():.4f}  max={x.max():.4f}  mean={x.mean():.4f}")
print(f"ret_fut_30d: min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")

# ---------------------------------------------------------------------------
# Função auxiliar: OLS com HAC + teste BP de heteroscedasticidade
# ---------------------------------------------------------------------------
def fit_and_report(y_use, x_use, label):
    Xc = sm.add_constant(x_use)
    res = sm.OLS(y_use, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    beta = res.params[BVRP_COL]
    t    = res.tvalues[BVRP_COL]
    p    = res.pvalues[BVRP_COL]
    r2   = res.rsquared

    # Breusch-Pagan (OLS sem HAC para o teste diagnóstico)
    res_ols = sm.OLS(y_use, Xc).fit()
    bp_stat, bp_p, _, _ = het_breuschpagan(res_ols.resid, res_ols.model.exog)

    print(f"\n--- {label} ---")
    print(f"  beta={beta:+.6f}  t={t:+.3f}  p={p:.4f}  R²={r2:.4f}")
    print(f"  Breusch-Pagan: stat={bp_stat:.3f}  p={bp_p:.4f}  "
          f"({'HETEROCED.' if bp_p < 0.05 else 'homocedástico'})")
    return res, r2, bp_p

# ---------------------------------------------------------------------------
# 1. Baseline: y = ret_fut_30d (original)
# ---------------------------------------------------------------------------
res_base, r2_base, bp_base = fit_and_report(
    pd.Series(y, name=BVRP_COL),
    pd.DataFrame({BVRP_COL: x}),
    "Original (y = ret_fut_30d)"
)

# ---------------------------------------------------------------------------
# 2. arcsinh(y) — preserva sinal, comprime caudas, definida para todo y
# ---------------------------------------------------------------------------
y_arcsinh = np.arcsinh(y)
res_arch, r2_arch, bp_arch = fit_and_report(
    pd.Series(y_arcsinh, name=BVRP_COL),
    pd.DataFrame({BVRP_COL: x}),
    "arcsinh(y)"
)

# ---------------------------------------------------------------------------
# 3. sign(y) * log(1 + |y|) — transformação log simétrica
# ---------------------------------------------------------------------------
y_slog = np.sign(y) * np.log1p(np.abs(y))
res_slog, r2_slog, bp_slog = fit_and_report(
    pd.Series(y_slog, name=BVRP_COL),
    pd.DataFrame({BVRP_COL: x}),
    "sign(y)*log(1+|y|)"
)

# ---------------------------------------------------------------------------
# 4. |y| — BVRP como preditor da magnitude do retorno (teste adicional)
# ---------------------------------------------------------------------------
y_abs = np.abs(y)
res_abs, r2_abs, bp_abs = fit_and_report(
    pd.Series(y_abs, name=BVRP_COL),
    pd.DataFrame({BVRP_COL: x}),
    "|y| — magnitude do retorno"
)

# ---------------------------------------------------------------------------
# Figura comparativa: 4 painéis (scatter + reta OLS)
# ---------------------------------------------------------------------------
def add_panel(ax, xv, yv, res, xlabel, ylabel, title):
    x_line = np.linspace(xv.min(), xv.max(), 300)
    Xc_line = sm.add_constant(pd.DataFrame({BVRP_COL: x_line}))
    y_line = res.predict(Xc_line)
    beta = res.params[BVRP_COL]
    r2   = res.rsquared
    ax.scatter(xv, yv, alpha=0.25, s=10, color="steelblue")
    ax.plot(x_line, y_line, color="navy", linewidth=1.8,
            label=f"$\\hat{{\\beta}}={beta:+.4f}$, $R^2={r2:.3f}$")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Transformações de y: BVRP → retorno futuro 30d", fontsize=12, y=1.01)

add_panel(axes[0, 0], x, y, res_base,
          "BVRP$_t$", "ret\_fut\_30d (original)", "Original")
add_panel(axes[0, 1], x, y_arcsinh, res_arch,
          "BVRP$_t$", "arcsinh(ret\_fut\_30d)", "arcsinh(y)")
add_panel(axes[1, 0], x, y_slog, res_slog,
          "BVRP$_t$", "sign(y)·log(1+|y|)", "log simétrico")
add_panel(axes[1, 1], x, y_abs, res_abs,
          "BVRP$_t$", "|ret\_fut\_30d|", "|y| — magnitude")

plt.tight_layout()
out_fig = OUT / "scatter_bvrp_log_transform_test.png"
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nFigura comparativa salva: {out_fig}")

# ---------------------------------------------------------------------------
# Figura extra: resíduos do baseline vs arcsinh (diagnóstico)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Diagnóstico de resíduos: original vs arcsinh(y)", fontsize=11)

Xc = sm.add_constant(pd.DataFrame({BVRP_COL: x}))
fitted_base  = sm.OLS(y,          Xc).fit()
fitted_arch2 = sm.OLS(y_arcsinh,  Xc).fit()

for ax, res_d, title in [
    (axes[0], fitted_base,  "Original"),
    (axes[1], fitted_arch2, "arcsinh(y)"),
]:
    ax.scatter(res_d.fittedvalues, res_d.resid, alpha=0.25, s=10, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valores ajustados", fontsize=9)
    ax.set_ylabel("Resíduos", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_resid = OUT / "residuos_base_vs_arcsinh.png"
plt.savefig(out_resid, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figura resíduos salva: {out_resid}")

# ---------------------------------------------------------------------------
# Sumário final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMÁRIO COMPARATIVO")
print(f"{'Especificação':<28} {'R²':>6}  {'BP p-val':>9}  Heteroced?")
print("-" * 60)
for label, r2, bp in [
    ("Original",             r2_base, bp_base),
    ("arcsinh(y)",           r2_arch, bp_arch),
    ("sign(y)*log(1+|y|)",  r2_slog, bp_slog),
    ("|y| (magnitude)",      r2_abs,  bp_abs),
]:
    flag = "SIM" if bp < 0.05 else "não"
    print(f"  {label:<26} {r2:>6.4f}  {bp:>9.4f}  {flag}")
print("=" * 60)
