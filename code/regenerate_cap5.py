"""
Regenera todas as figuras e tabelas do Cap. 5 usando os dados originais da dissertacao.
Reproducao fiel do notebook cap5_ols_vrp.ipynb.
Saida: figs/cap5/ e tables/cap5/

Uso: py code/regenerate_cap5.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA  = ROOT / "data" / "dataset_bvrp_with_skew.csv"
FIGS  = ROOT / "figs"  / "cap5"
TABS  = ROOT / "tables" / "cap5"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

BVRP_COL = "vrp_30d"
RV_COL   = "rv_30d"
IV_COL   = "iv_30d_dec"
HAC_LAGS = 5

HORIZONS = [1, 5, 10, 20, 30, 60]

# ---------------------------------------------------------------------------
# 1. Carregar dados (retornos futuros ja estao no arquivo)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df = df.dropna(subset=[BVRP_COL]).reset_index(drop=True)
print("Dataset: %d obs | %s a %s" % (len(df), df["date"].min().date(), df["date"].max().date()))
print("BVRP: mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (
    df[BVRP_COL].mean(), df[BVRP_COL].std(), df[BVRP_COL].min(), df[BVRP_COL].max()))

# ---------------------------------------------------------------------------
# 2. OLS com HAC (maxlags=5 fixo, conforme notebook original)
# ---------------------------------------------------------------------------
def run_ols_hac(y, X, lags=HAC_LAGS):
    Xc = sm.add_constant(X)
    mask = y.notna() & Xc.notna().all(axis=1)
    return sm.OLS(y[mask], Xc[mask]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

# ---------------------------------------------------------------------------
# 3. Modelo basico: apenas BVRP
# ---------------------------------------------------------------------------
rows_basic = []
for h in HORIZONS:
    col = "ret_fut_%dd" % h
    if col not in df.columns:
        print("AVISO: coluna %s nao encontrada" % col)
        continue
    res = run_ols_hac(df[col], df[[BVRP_COL]])
    rows_basic.append({
        "horizon": h,
        "beta":    res.params[BVRP_COL],
        "t_stat":  res.tvalues[BVRP_COL],
        "p_value": res.pvalues[BVRP_COL],
        "R2":      res.rsquared,
        "N":       int(res.nobs),
    })

basic = pd.DataFrame(rows_basic)
print("\nModelo basico (HAC maxlags=%d):" % HAC_LAGS)
print(basic.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Modelo com controles: BVRP + RV + IV
# ---------------------------------------------------------------------------
rows_ctrl = []
for h in HORIZONS:
    col = "ret_fut_%dd" % h
    if col not in df.columns:
        continue
    res = run_ols_hac(df[col], df[[BVRP_COL, RV_COL, IV_COL]])
    rows_ctrl.append({
        "horizon":   h,
        "beta_BVRP": res.params[BVRP_COL],
        "t_BVRP":    res.tvalues[BVRP_COL],
        "p_BVRP":    res.pvalues[BVRP_COL],
        "beta_RV":   res.params[RV_COL],
        "beta_IV":   res.params[IV_COL],
        "R2":        res.rsquared,
        "N":         int(res.nobs),
    })

ctrl = pd.DataFrame(rows_ctrl)
print("\nModelo com controles:")
print(ctrl.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Tabelas LaTeX
# ---------------------------------------------------------------------------
tab_basic = TABS / "tab_ols_basico_multihoriz.tex"
tab_basic.write_text(r"""\begin{table}[H]
\centering
\caption{Resultados OLS (HAC) do BVRP para múltiplos horizontes de previsão.}
\label{tab:ols_basico_multihoriz}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{cccccr}
\toprule
Horizonte ($h$) & $\hat{\beta}_h$ & Estat.\ $t$ & $p$-valor & $R^2$ & $N$ \\
\midrule
""" + "\n".join(
    " %2d & %+.6f & %+.3f & %.3f & %.4f & %d \\\\" % (
        int(r.horizon), r.beta, r.t_stat, r.p_value, r.R2, int(r.N))
    for r in basic.itertuples()
) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\smallskip
\begin{minipage}{0.92\linewidth}
\footnotesize
\textit{Nota}: Erros padrão HAC (Newey--West) com 5 defasagens. $\hat{\beta}_h$ é o coeficiente
do BVRP na regressão $R_{t+h} = \alpha_h + \beta_h\,\text{BVRP}_t + \varepsilon_{t+h}$.
Horizontes em dias.
\end{minipage}
\end{table}
""", encoding="utf-8")
print("\nTabela basica salva: %s" % tab_basic)

tab_ctrl = TABS / "tab_ols_controles_multihoriz.tex"
tab_ctrl.write_text(r"""\begin{table}[H]
\centering
\caption{Resultados OLS (HAC) com controles de volatilidade (RV e IV) para múltiplos horizontes de previsão.}
\label{tab:ols_controles_multihoriz}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{cccccccr}
\toprule
Horizonte ($h$) & $\hat{\beta}_{\text{BVRP}}$ & Estat.\ $t$ & $p$-valor & $\hat{\beta}_{\text{RV}}$ & $\hat{\beta}_{\text{IV}}$ & $R^2$ & $N$ \\
\midrule
""" + "\n".join(
    " %2d & %+.6f & %+.3f & %.3f & %+.6f & %+.6f & %.4f & %d \\\\" % (
        int(r.horizon), r.beta_BVRP, r.t_BVRP, r.p_BVRP,
        r.beta_RV, r.beta_IV, r.R2, int(r.N))
    for r in ctrl.itertuples()
) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\smallskip
\begin{minipage}{0.92\linewidth}
\footnotesize
\textit{Nota}: Erros padrão HAC (Newey--West) com 5 defasagens. Modelo: $R_{t+h} = \alpha_h +
\beta_{1,h}\,\text{BVRP}_t + \beta_{2,h}\,\text{RV}_t + \beta_{3,h}\,\text{IV}_t + \varepsilon_{t+h}$.
Como $\text{BVRP} = \text{IV} - \text{RV}$ por construção, há multicolinearidade entre os regressores;
coeficientes individuais devem ser interpretados com cautela. Horizontes em dias.
\end{minipage}
\end{table}
""", encoding="utf-8")
print("Tabela controles salva: %s" % tab_ctrl)

# ---------------------------------------------------------------------------
# 6. Figura 1 -- Scatter BVRP vs retorno futuro 30d
# ---------------------------------------------------------------------------
h = 30
col = "ret_fut_%dd" % h
tmp = df[[BVRP_COL, col]].dropna()
x = tmp[BVRP_COL].values
y = tmp[col].values

res30 = run_ols_hac(tmp[col], tmp[[BVRP_COL]])
alpha_val = res30.params["const"]
beta_val  = res30.params[BVRP_COL]
x_line = np.linspace(x.min(), x.max(), 300)
y_line = alpha_val + beta_val * x_line

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.30, s=14, color="steelblue")
ax.plot(x_line, y_line, color="navy", linewidth=1.8,
        label="OLS  ($\\hat{\\beta}=%.4f$,  $R^2=%.3f$)" % (beta_val, res30.rsquared))
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax.set_xlabel("BVRP$_t$")
ax.set_ylabel("Retorno futuro acumulado (30 dias)")
ax.set_title("BVRP e retorno futuro do Bitcoin (h = 30 dias)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = FIGS / "scatter_bvrp_ret_fut_30d.png"
plt.savefig(out, dpi=300)
plt.close()
print("Figura 1 salva: %s" % out)

# ---------------------------------------------------------------------------
# 7. Figura 2 -- Beta por horizonte (modelo basico)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(basic["horizon"], basic["beta"], marker="o", linewidth=2, color="steelblue")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
sig = basic[basic["p_value"] < 0.05]
if len(sig) > 0:
    ax.scatter(sig["horizon"], sig["beta"], s=80, zorder=5, color="navy",
               label="$p < 0{,}05$")
    ax.legend(fontsize=9)
ax.set_xticks(HORIZONS)
ax.set_xlabel("Horizonte $h$ (dias)")
ax.set_ylabel("$\\hat{\\beta}_h$ (BVRP)")
ax.set_title("Coeficiente do BVRP por horizonte --- OLS basico (HAC)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = FIGS / "beta_por_horizonte_basico.png"
plt.savefig(out, dpi=300)
plt.close()
print("Figura 2 salva: %s" % out)

# ---------------------------------------------------------------------------
# 8. Figura 3 -- Comparacao basico vs controles
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(basic["horizon"], basic["beta"], marker="o", linewidth=2,
        label="OLS basico (BVRP)", color="steelblue")
ax.plot(ctrl["horizon"], ctrl["beta_BVRP"], marker="s", linewidth=2,
        label="OLS com controles (BVRP | RV, IV)", color="darkorange")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(HORIZONS)
ax.set_xlabel("Horizonte $h$ (dias)")
ax.set_ylabel("$\\hat{\\beta}$ (BVRP)")
ax.set_title("BVRP: coeficiente por horizonte --- basico vs controles (HAC)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = FIGS / "beta_comparacao_basico_vs_controles.png"
plt.savefig(out, dpi=300)
plt.close()
print("Figura 3 salva: %s" % out)

print("\nCap. 5 regenerado com sucesso.")
print("  Figuras em: %s" % FIGS)
print("  Tabelas em: %s" % TABS)
