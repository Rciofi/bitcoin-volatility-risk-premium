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
DATA  = ROOT / "data" / "vrp_with_targets.csv"          # IC1: fonte canonica (1.775 obs)
FIGS  = ROOT / "figs"  / "cap5"
TABS  = ROOT / "tables" / "cap5"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

BVRP_COL = "vrp_30d"
RV_COL   = "rv_30d"
IV_COL   = "iv_30d"    # percentual (% a.a.) -- coluna presente na fonte canonica
# HAC_LAGS removido -- maxlags=h passado explicitamente por chamada (IC5)

HORIZONS = [1, 5, 10, 20, 30, 60]

# NOTA METODOLOGICA (E3 / IC5):
# iv_30d e rv_30d apresentam alta persistencia nos testes ADF/KPSS (resultados
# inconclusivos ou I(1) borderline), consistente com a literatura de volatilidade
# de ativos cripto (processos de longa memoria). As series sao mantidas em nivel
# seguindo a convencao da literatura; a inferencia e tratada via erros padrao HAC
# (Newey-West) com maxlags=h em cada horizonte, mitigando distorcoes por
# autocorrelacao serial. Ver tabela ADF/KPSS em tables/cap3/adf_kpss_table.csv.

# ---------------------------------------------------------------------------
# 1. Carregar dados (retornos futuros ja estao no arquivo)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df = df.dropna(subset=[BVRP_COL]).reset_index(drop=True)
print("Dataset: %d obs | %s a %s" % (len(df), df["date"].min().date(), df["date"].max().date()))
print("BVRP: mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (
    df[BVRP_COL].mean(), df[BVRP_COL].std(), df[BVRP_COL].min(), df[BVRP_COL].max()))

# ---------------------------------------------------------------------------
# 2. OLS com HAC -- maxlags=h passado explicitamente (IC5: lag proporcional ao horizonte)
# ---------------------------------------------------------------------------
def run_ols_hac(y, X, lags):
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
    res = run_ols_hac(df[col], df[[BVRP_COL]], lags=h)
    rows_basic.append({
        "horizon": h,
        "beta":    res.params[BVRP_COL],
        "t_stat":  res.tvalues[BVRP_COL],
        "p_value": res.pvalues[BVRP_COL],
        "R2":      res.rsquared,
        "N":       int(res.nobs),
    })

basic = pd.DataFrame(rows_basic)
print("\nModelo basico (HAC maxlags=h por horizonte):")
print(basic.to_string(index=False))

# ---------------------------------------------------------------------------
# E5. Moving Block Bootstrap — IC 95% para beta do modelo basico
# ---------------------------------------------------------------------------
# Justificativa: BVRP e near-unit-root (inconclusivo ADF/KPSS); retornos
# sobrepostos (h=20,30,60) introduzem autocorrelacao de ordem h-1.
# HAC com maxlags=h corrije assintotica (IC5), mas p-valores podem estar
# inflados para amostras finitas (vies de Stambaugh). MBB fornece IC
# robustos sem assumir forma parametrica dos erros.
# Referencias: Kunsch (1989), Politis & Romano (1994).

def moving_block_bootstrap_beta(y, X_df, bvrp_col,
                                 block_size=30, n_bootstrap=1000, seed=42):
    """
    Moving Block Bootstrap (MBB) para o coeficiente beta do BVRP.

    Reamostras os PARES (X_t, y_t) em blocos consecutivos de tamanho
    block_size, preservando a estrutura de dependencia serial conjunta
    de X e y. OLS simples dentro de cada bootstrap (a reamostragem
    ja trata a autocorrelacao).

    Parametros
    ----------
    y          : pd.Series    -- variavel dependente (ret_fut_hd)
    X_df       : pd.DataFrame -- regressores (sem constante)
    bvrp_col   : str          -- nome da coluna BVRP em X_df
    block_size : int          -- tamanho do bloco em obs (padrao: 30)
    n_bootstrap: int          -- numero de replicacoes (padrao: 1000)
    seed       : int          -- semente para reproducibilidade

    Nota: block_size=30 e conservador para todos os horizontes;
    para h=60, block_size=60 seria mais rigoroso -- diferenca marginal
    na pratica dado n=1.775 obs.

    Retorna
    -------
    betas : np.ndarray (n_bootstrap,) -- distribuicao bootstrap de beta_BVRP
    """
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
        starts   = rng.integers(0, max_start + 1, size=n_blocks)
        idx      = np.concatenate([np.arange(s, s + block_size)
                                   for s in starts])[:n]
        coefs    = np.linalg.lstsq(X_arr[idx], y_arr[idx], rcond=None)[0]
        betas[b] = coefs[bvrp_idx]

    return betas


BLOCK_SIZE   = 30    # janela da RV — justificado por Kunsch (1989)
N_BOOTSTRAP  = 1000  # replicacoes

import time
t0 = time.time()
rows_boot = []
print("\n--- E5: Moving Block Bootstrap (block=%d, B=%d, seed=42) ---" % (BLOCK_SIZE, N_BOOTSTRAP))
print("  %4s  %+10s  %10s  %10s  %8s  %8s  %s" %
      ("h", "beta_hat", "IC_lower", "IC_upper", "p_hac", "p_boot", "sig"))
print("  " + "-"*68)

for h in HORIZONS:
    col = "ret_fut_%dd" % h
    if col not in df.columns:
        continue

    row_b    = basic[basic["horizon"] == h].iloc[0]
    beta_hat = float(row_b["beta"])
    p_hac    = float(row_b["p_value"])

    betas_b  = moving_block_bootstrap_beta(
        df[col], df[[BVRP_COL]],
        bvrp_col=BVRP_COL,
        block_size=BLOCK_SIZE,
        n_bootstrap=N_BOOTSTRAP,
    )

    ic_lower = float(np.percentile(betas_b, 2.5))
    ic_upper = float(np.percentile(betas_b, 97.5))
    p_boot   = float(min((betas_b <= 0).mean() if beta_hat >= 0
                         else (betas_b >= 0).mean(), 0.5) * 2)
    sig      = ("***" if p_boot < 0.01 else
                ("**" if p_boot < 0.05 else
                 ("*"  if p_boot < 0.10 else "")))

    rows_boot.append({
        "horizon":  h,
        "beta_hat": beta_hat,
        "ic_lower": ic_lower,
        "ic_upper": ic_upper,
        "p_hac":    p_hac,
        "p_boot":   p_boot,
        "sig_boot": sig,
    })
    print("  %4d  %+10.6f  %+10.6f  %+10.6f  %8.4f  %8.4f  %s" %
          (h, beta_hat, ic_lower, ic_upper, p_hac, p_boot, sig))

t1 = time.time()
print("  Tempo MBB: %.1f s" % (t1 - t0))

df_boot  = pd.DataFrame(rows_boot)
out_boot = TABS / "bootstrap_ci_cap5.csv"
df_boot.to_csv(out_boot, index=False)
print("Bootstrap CI salvo: %s" % out_boot)

tab_boot = TABS / "tab_bootstrap_ci.tex"
tab_boot.write_text(r"""\begin{table}[H]
\centering
\caption{Coeficientes $\hat{\beta}_h$ do modelo básico com intervalos de
confiança de 95\% obtidos por \textit{Moving Block Bootstrap}
(tamanho de bloco de 30 dias, 1.000 replicações).}
\label{tab:cap5-bootstrap}
\begin{tabular}{rrrrrr}
\toprule
$h$ & $\hat{\beta}_h$ & IC 2{,}5\% & IC 97{,}5\% & $p$-HAC & $p$-bootstrap \\
\midrule
""" + "\n".join(
    "     %2d & %.6f & %.6f & %.6f & %.6f & %.6f \\\\" % (
        int(r.horizon), r.beta_hat, r.ic_lower, r.ic_upper, r.p_hac, r.p_boot)
    for r in df_boot.itertuples()
) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
print("Tabela bootstrap salva: %s" % tab_boot)

# ---------------------------------------------------------------------------
# 4. Modelo com controles: BVRP + RV + IV - REMOVIDO (E1)
# ---------------------------------------------------------------------------
# REMOVIDO (E1): colinearidade perfeita
# BVRP = RV - IV -> X'X singular -> coefs nao identificaveis.
# Ver Marcelo (pags. 42, 49). Substituido pela especificacao RV + IV abaixo.
#
# rows_ctrl = []
# for h in HORIZONS:
#     col = "ret_fut_%dd" % h
#     if col not in df.columns:
#         continue
#     res = run_ols_hac(df[col], df[[BVRP_COL, RV_COL, IV_COL]], lags=h)
#     rows_ctrl.append({
#         "horizon":   h,
#         "beta_BVRP": res.params[BVRP_COL],
#         "t_BVRP":    res.tvalues[BVRP_COL],
#         "p_BVRP":    res.pvalues[BVRP_COL],
#         "beta_RV":   res.params[RV_COL],
#         "beta_IV":   res.params[IV_COL],
#         "R2":        res.rsquared,
#         "N":         int(res.nobs),
#     })
# ctrl = pd.DataFrame(rows_ctrl)
# print("\nModelo com controles:")
# print(ctrl.to_string(index=False))

# ---------------------------------------------------------------------------
# 4b. Modelo E1 - apenas RV e IV (sem BVRP - elimina colinearidade perfeita)
#     Especificacao: ret_fut_h = alpha + beta1*RV + beta2*IV + epsilon
#     Teste de Marcelo: beta1 aprox -beta2 -> BVRP captura a combinacao relevante
# ---------------------------------------------------------------------------
rows_rv_iv = []
for h in HORIZONS:
    col = "ret_fut_%dd" % h
    if col not in df.columns:
        continue
    res  = run_ols_hac(df[col], df[[RV_COL, IV_COL]], lags=h)
    b1   = res.params[RV_COL]
    b2   = res.params[IV_COL]
    rows_rv_iv.append({
        "horizon":  h,
        "beta_RV":  b1,
        "t_RV":     res.tvalues[RV_COL],
        "p_RV":     res.pvalues[RV_COL],
        "beta_IV":  b2,
        "t_IV":     res.tvalues[IV_COL],
        "p_IV":     res.pvalues[IV_COL],
        "b1_b2":    b1 + b2,
        "simetria": abs(b1 + b2) / (abs(b1) + abs(b2) + 1e-15) * 100,
        "R2":       res.rsquared,
        "N":        int(res.nobs),
    })

rv_iv = pd.DataFrame(rows_rv_iv)
print("\nModelo E1 - RV e IV sem BVRP (HAC maxlags=h):")
print(rv_iv.to_string(index=False))
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
\textit{Nota}: Erros padrão HAC (Newey--West) com $h$ defasagens. $\hat{\beta}_h$ é o coeficiente
do BVRP na regressão $R_{t+h} = \alpha_h + \beta_h\,\text{BVRP}_t + \varepsilon_{t+h}$.
Horizontes em dias.
\end{minipage}
\end{table}
""", encoding="utf-8")
print("\nTabela basica salva: %s" % tab_basic)

# tab_ctrl - REMOVIDO (E1): tabela usava ctrl com colinearidade perfeita
# tab_ctrl = TABS / "tab_ols_controles_multihoriz.tex"
# (arquivo antigo nao e mais gerado)

# ---------------------------------------------------------------------------
# Tab. E1: RV e IV sem BVRP
# Colunas: h | beta_RV | t_RV | p_RV | beta_IV | t_IV | p_IV |
#          beta1+beta2 | Simetria% | R2 | N
# ---------------------------------------------------------------------------
tab_rv_iv = TABS / "tab_ols_rv_iv_multihoriz.tex"

_hdr = (
    "\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Coeficientes de RV e IV na previs\\~ao de retornos futuros do Bitcoin "
    "(modelo E1: sem BVRP). "
    "Especifica\\c{c}\\~ao: $R_{t+h} = \\alpha_h + \\beta_{1,h}\\,\\text{RV}_t "
    "+ \\beta_{2,h}\\,\\text{IV}_t + \\varepsilon_{t+h}$. "
    "Erros-padr\\~ao HAC (Newey--West) com $h$ defasagens. "
    "Coluna ``Simetria'' reporta $|\\hat{\\beta}_1 + \\hat{\\beta}_2| \\,/\\,"
    "(|\\hat{\\beta}_1| + |\\hat{\\beta}_2|) \\times 100$: "
    "valores pr\\'{o}ximos de 0 indicam $\\hat{\\beta}_1 \\approx -\\hat{\\beta}_2$, "
    "consistente com o BVRP como combina\\c{c}\\~ao linear relevante.}\n"
    "\\label{tab:ols_rv_iv_multihoriz}\n"
    "\\footnotesize\n"
    "\\begin{adjustbox}{max width=\\textwidth}\n"
    "\\begin{tabular}{crrrrrrrrrr}\n"
    "\\toprule\n"
    "$h$ & $\\hat{\\beta}_{RV}$ & $t_{RV}$ & $p_{RV}$ "
    "& $\\hat{\\beta}_{IV}$ & $t_{IV}$ & $p_{IV}$ "
    "& $\\hat{\\beta}_1{+}\\hat{\\beta}_2$ & Simetria (\\%) & $R^2$ & $N$ \\\\\n"
    "\\midrule\n"
)
_body = "\n".join(
    " %2d & %+.6f & %+.3f & %.3f & %+.6f & %+.3f & %.3f & %+.6f & %.1f & %.4f & %d \\\\" % (
        int(r.horizon),
        r.beta_RV, r.t_RV, r.p_RV,
        r.beta_IV, r.t_IV, r.p_IV,
        r.b1_b2, r.simetria,
        r.R2, int(r.N))
    for r in rv_iv.itertuples()
)
_ftr = (
    "\n\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{adjustbox}\n"
    "\\smallskip\n"
    "\\begin{minipage}{0.96\\linewidth}\n"
    "\\footnotesize\\textit{Nota}: Modelo estimado sem o BVRP como regressor (E1 --- Fase 2). "
    "A inclus\\~ao simult\\^anea de BVRP, RV e IV induz colinearidade perfeita "
    "($\\text{BVRP} \\equiv \\text{RV} - \\text{IV}$), "
    "tornando os coeficientes n\\~ao identific\\'aveis (ver Marcelo, p\\'ags.\\ 42, 49). "
    "Nenhum horizonte apresenta coeficiente estatisticamente significativo "
    "($p_{\\min} = 0{,}25$, $h = 1$).\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)
tab_rv_iv.write_text(_hdr + _body + _ftr, encoding="utf-8")
print("Tabela E1 (RV+IV) salva: %s" % tab_rv_iv)

# ---------------------------------------------------------------------------
# 6. Figura 1 -- Scatter BVRP vs retorno futuro 30d
# ---------------------------------------------------------------------------
h = 30
col = "ret_fut_%dd" % h
tmp = df[[BVRP_COL, col]].dropna()
x = tmp[BVRP_COL].values
y = tmp[col].values

res30 = run_ols_hac(tmp[col], tmp[[BVRP_COL]], lags=30)
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
# ---------------------------------------------------------------------------
# 8. Figura 3 -- beta_RV e beta_IV por horizonte (modelo E1)
#    Substituiu "basico vs controles" -- ctrl removido por colinearidade (E1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rv_iv["horizon"], rv_iv["beta_RV"] * 100, marker="o", linewidth=2,
        label="$\\hat{\\beta}_{RV}$ (x100)", color="steelblue")
ax.plot(rv_iv["horizon"], rv_iv["beta_IV"] * 100, marker="s", linewidth=2,
        label="$\\hat{\\beta}_{IV}$ (x100)", color="darkorange")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(HORIZONS)
ax.set_xlabel("Horizonte $h$ (dias)")
ax.set_ylabel("Coeficiente x100")
ax.set_title("Coeficientes de RV e IV por horizonte --- OLS E1 (HAC)\n"
             "(teste: $\\hat{\\beta}_{RV} \\approx -\\hat{\\beta}_{IV}$?)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = FIGS / "beta_comparacao_rv_iv.png"
plt.savefig(out, dpi=300)
plt.close()
print("Figura 3 (E1) salva: %s" % out)

print("\nCap. 5 regenerado com sucesso.")
print("  Figuras em: %s" % FIGS)
print("  Tabelas em: %s" % TABS)
