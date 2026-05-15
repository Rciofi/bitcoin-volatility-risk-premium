# -*- coding: utf-8 -*-
"""
Gera todas as figuras e tabelas do Capítulo 9 — Regimes de Volatilidade e BVRP.

Outputs:
  figs/cap9/fig_cap9_01_rv30d_regimes.png        — série temporal RV com limiares
  figs/cap9/fig_cap9_02_hist_rv30d.png            — histograma RV 30d com limiares
  figs/cap9/fig_cap9_03_boxplot_bvrp_regimes.png  — BVRP por regime
  figs/cap9/fig_cap9_04_boxplot_ret_regimes.png   — retornos futuros por regime (3 horizontes)
  figs/cap9/fig_cap9_05_scatter_bvrp_ret.png      — BVRP vs retornos futuros (3 horizontes)
  tables/cap9/tab_cap9_01_frequencia_regimes.tex
  tables/cap9/tab_cap9_02_estatisticas_regimes.tex
  tables/cap9/tab_cap9_03_retornos_regimes.tex
  tables/cap9/tab_cap9_04_ttest.tex
  tables/cap9/tab_cap9_05_regressoes.tex
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vrp_with_regimes.csv"   # IC1: fonte canonica com ret_fut_* (1.775 obs)
OUT_FIGS = ROOT / "figs" / "cap9"
OUT_TABS = ROOT / "tables" / "cap9"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABS.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 20, 30]
REGIMES = ["Baixa volatilidade", "Média volatilidade", "Alta volatilidade"]
COLORS = {"Baixa volatilidade": "#2e75b6", "Média volatilidade": "#70ad47", "Alta volatilidade": "#c00000"}

# ── 1. Carregar e preparar dados ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# Retornos futuros já estão no dataset (1d, 5d, 10d, 20d, 30d)
ret_cols = [f"ret_fut_{h}d" for h in HORIZONS]
df = df.dropna(subset=["vrp_30d", "rv_30d"] + ret_cols)
df = df.reset_index(drop=True)

print(f"Período: {df['date'].min().date()} a {df['date'].max().date()}, N={len(df)}")

# ── 2. Definir regimes por quantis (q25 / q75 da RV 30d) ─────────────────────
# NOTA: quantis calculados sobre amostra completa (sem janela expansiva).
# Uso e analise descritiva condicional — lookahead menos critico que em modelos preditivos.
# Para analise preditiva com regime_vol, considerar janela expansiva futuramente.
q25 = df["rv_30d"].quantile(0.25)
q75 = df["rv_30d"].quantile(0.75)

df["regime_vol"] = "Média volatilidade"
df.loc[df["rv_30d"] <= q25, "regime_vol"] = "Baixa volatilidade"
df.loc[df["rv_30d"] >= q75, "regime_vol"] = "Alta volatilidade"
df["regime_vol"] = pd.Categorical(df["regime_vol"], categories=REGIMES, ordered=True)

df["dummy_alta_vol"] = (df["regime_vol"] == "Alta volatilidade").astype(int)

print(f"Limiar q25 (baixa vol): {q25:.2f}  |  Limiar q75 (alta vol): {q75:.2f}")
print(df["regime_vol"].value_counts().sort_index())


# ── Helpers ──────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_br(x, decimals=4):
    """Formata número com vírgula decimal (padrão ABNT)."""
    return f"{x:.{decimals}f}".replace(".", ",")


# ── 3. Tab 9.1 — Frequência dos regimes ──────────────────────────────────────
freq = df["regime_vol"].value_counts().sort_index()
freq_pct = freq / len(df)

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Distribuição dos regimes de volatilidade com base nos quantis da RV 30 dias.}",
    r"\label{tab:cap9_frequencia_regimes}",
    r"\begin{tabular}{lrr}",
    r"\toprule",
    r"Regime & Observações & Participação (\%) \\",
    r"\midrule",
]
for regime in REGIMES:
    n = freq[regime]
    pct = freq_pct[regime] * 100
    lines.append(f"{regime} & {n} & {fmt_br(pct, 1)}\\% \\\\")
lines += [
    r"\midrule",
    f"Total & {len(df)} & {fmt_br(100.0, 1)}\\% \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize\textit{Nota}: Regime de baixa volatilidade: RV 30d $\leq$ "
    + f"{fmt_br(q25, 1)}" + r"\%; alta volatilidade: RV 30d $\geq$ "
    + f"{fmt_br(q75, 1)}" + r"\%. Período: "
    + f"{df['date'].min().strftime('%d/%m/%Y')}" + " a "
    + f"{df['date'].max().strftime('%d/%m/%Y')}.",
    r"\end{table}",
]
(OUT_TABS / "tab_cap9_01_frequencia_regimes.tex").write_text("\n".join(lines), encoding="utf-8")
print("OK tab_cap9_01_frequencia_regimes.tex")

# ── 4. Tab 9.2 — Estatísticas de RV, IV e BVRP por regime ───────────────────
vars_tab = {"rv_30d": "RV 30d", "iv_30d": "IV 30d", "vrp_30d": "BVRP"}

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Estatísticas descritivas de RV, IV e BVRP por regime de volatilidade.}",
    r"\label{tab:cap9_estatisticas_regimes}",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Variável & Regime & Média & Mediana & Desvio-padrão & N \\",
    r"\midrule",
]
for col, label in vars_tab.items():
    for i, regime in enumerate(REGIMES):
        sub = df.loc[df["regime_vol"] == regime, col]
        prefix = label if i == 0 else ""
        lines.append(
            f"{prefix} & {regime} & {fmt_br(sub.mean())} & {fmt_br(sub.median())} "
            f"& {fmt_br(sub.std())} & {len(sub)} \\\\"
        )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(OUT_TABS / "tab_cap9_02_estatisticas_regimes.tex").write_text("\n".join(lines), encoding="utf-8")
print("OK tab_cap9_02_estatisticas_regimes.tex")

# ── 5. Tab 9.3 — Retornos futuros médios por regime ──────────────────────────
lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Retorno futuro médio (\%) por regime de volatilidade e horizonte.}",
    r"\label{tab:cap9_retornos_regimes}",
    r"\begin{tabular}{l" + "r" * len(HORIZONS) + "}",
    r"\toprule",
    r"Regime & " + " & ".join([f"{h}d" for h in HORIZONS]) + r" \\",
    r"\midrule",
]
for regime in REGIMES:
    sub = df.loc[df["regime_vol"] == regime]
    vals = " & ".join([fmt_br(sub[f"ret_fut_{h}d"].mean() * 100) for h in HORIZONS])
    lines.append(f"{regime} & {vals} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT_TABS / "tab_cap9_03_retornos_regimes.tex").write_text("\n".join(lines), encoding="utf-8")
print("OK tab_cap9_03_retornos_regimes.tex")

# ── 6. Tab 9.4 — Teste t: Alta vol vs Baixa vol ──────────────────────────────
test_rows = []
for h in HORIZONS:
    col = f"ret_fut_{h}d"
    alta = df.loc[df["regime_vol"] == "Alta volatilidade", col].dropna()
    baixa = df.loc[df["regime_vol"] == "Baixa volatilidade", col].dropna()
    t_stat, p_val = stats.ttest_ind(alta, baixa, equal_var=False)
    test_rows.append({
        "h": h, "media_alta": alta.mean(), "media_baixa": baixa.mean(),
        "dif": alta.mean() - baixa.mean(), "t": t_stat, "p": p_val,
    })

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Teste $t$ de Welch: diferença de retornos futuros entre regimes de alta e baixa volatilidade.}",
    r"\label{tab:cap9_ttest}",
    r"\begin{tabular}{lrrrrc}",
    r"\toprule",
    r"Horizonte & Média Alta Vol & Média Baixa Vol & Diferença & $t$-stat & $p$-valor \\",
    r"\midrule",
]
for r in test_rows:
    sig = stars(r["p"])
    lines.append(
        f"{r['h']}d & {fmt_br(r['media_alta']*100)} & {fmt_br(r['media_baixa']*100)} "
        f"& {fmt_br(r['dif']*100)} & {fmt_br(r['t'])} & {fmt_br(r['p'])}{sig} \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize\textit{Nota}: Retornos expressos em \%. *** $p<0{,}01$; ** $p<0{,}05$; * $p<0{,}10$.",
    r"\end{table}",
]
(OUT_TABS / "tab_cap9_04_ttest.tex").write_text("\n".join(lines), encoding="utf-8")
print("OK tab_cap9_04_ttest.tex")

# ── 7. Tab 9.5 — Regressões condicionais ─────────────────────────────────────
# ret_h = α + β₁·BVRP + β₂·Alta_Vol + β₃·(BVRP × Alta_Vol) + ε  (HAC NW)
reg_rows = {v: {h: None for h in HORIZONS} for v in ["const", "vrp_30d", "dummy_alta_vol", "inter", "r2", "n"]}

for h in HORIZONS:
    y_col = f"ret_fut_{h}d"
    sub = df[[y_col, "vrp_30d", "dummy_alta_vol"]].dropna().copy()
    sub["inter"] = sub["vrp_30d"] * sub["dummy_alta_vol"]
    y = sub[y_col]
    X = sm.add_constant(sub[["vrp_30d", "dummy_alta_vol", "inter"]])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    print(f"\n=== Horizonte {h}d  (N={int(res.nobs)}, R²={res.rsquared:.4f}) ===")
    print(res.summary())
    for var in ["const", "vrp_30d", "dummy_alta_vol", "inter"]:
        reg_rows[var][h] = (res.params[var], res.tvalues[var], res.pvalues[var])
    reg_rows["r2"][h] = res.rsquared
    reg_rows["n"][h] = int(res.nobs)

header_h = " & ".join([f"{h}d" for h in HORIZONS])
var_labels = {
    "const": "Constante",
    "vrp_30d": "BVRP",
    "dummy_alta_vol": "Alta Volatilidade",
    "inter": "BVRP $\\times$ Alta Vol",
}

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Regressões condicionais do retorno futuro em função do BVRP e do regime de volatilidade.",
    r"Erros-padrão de Newey--West com $h$ defasagens. Coeficientes com $t$-estatísticas entre parênteses.}",
    r"\label{tab:cap9_regressoes}",
    r"\begin{tabular}{l" + "r" * len(HORIZONS) + "}",
    r"\toprule",
    r"Variável & " + header_h + r" \\",
    r"\midrule",
]

for var, label in var_labels.items():
    coef_line = label
    tstat_line = ""
    for h in HORIZONS:
        coef, tval, pval = reg_rows[var][h]
        sig = stars(pval)
        coef_line += f" & {fmt_br(coef)}{sig}"
        tstat_line += f" & ({fmt_br(tval)})"
    lines.append(coef_line + r" \\")
    lines.append(tstat_line + r" \\")
    lines.append(r"\addlinespace[2pt]")

lines.append(r"\midrule")
r2_line = "$R^2$"
n_line = "$N$"
for h in HORIZONS:
    r2_line += f" & {fmt_br(reg_rows['r2'][h])}"
    n_line += f" & {reg_rows['n'][h]}"
lines.append(r2_line + r" \\")
lines.append(n_line + r" \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\footnotesize\textit{Nota}: *** $p<0{,}01$; ** $p<0{,}05$; * $p<0{,}10$.",
    r"\end{table}",
]
(OUT_TABS / "tab_cap9_05_regressoes.tex").write_text("\n".join(lines), encoding="utf-8")
print("OK tab_cap9_05_regressoes.tex")

# ── 8. Fig 9.1 — Série temporal RV com limiares de regime ────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["rv_30d"], linewidth=1.2, color="#1f4e79", label="RV 30d")
ax.axhline(q25, linestyle="--", linewidth=1.2, color=COLORS["Baixa volatilidade"], label=f"q25 = {q25:.1f}%")
ax.axhline(q75, linestyle="--", linewidth=1.2, color=COLORS["Alta volatilidade"], label=f"q75 = {q75:.1f}%")
ax.fill_between(df["date"], 0, df["rv_30d"],
                where=df["rv_30d"] <= q25, alpha=0.15, color=COLORS["Baixa volatilidade"])
ax.fill_between(df["date"], 0, df["rv_30d"],
                where=df["rv_30d"] >= q75, alpha=0.15, color=COLORS["Alta volatilidade"])
ax.set_ylabel("Volatilidade Realizada 30d (%)", fontsize=10)
ax.set_xlabel("Data", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("Regimes de Volatilidade — RV 30 dias", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap9_01_rv30d_regimes.png", dpi=300)
plt.close(fig)
print("OK fig_cap9_01_rv30d_regimes.png")

# ── 9. Fig 9.2 — Histograma RV 30d ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["rv_30d"].dropna(), bins=40, density=True, color="#2e75b6", alpha=0.75, edgecolor="white")
ax.axvline(q25, linestyle="--", linewidth=1.4, color=COLORS["Baixa volatilidade"], label=f"q25 = {q25:.1f}%")
ax.axvline(q75, linestyle="--", linewidth=1.4, color=COLORS["Alta volatilidade"], label=f"q75 = {q75:.1f}%")
ax.set_xlabel("RV 30 dias (%)", fontsize=10)
ax.set_ylabel("Densidade", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("Distribuição da Volatilidade Realizada 30 dias", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap9_02_hist_rv30d.png", dpi=300)
plt.close(fig)
print("OK fig_cap9_02_hist_rv30d.png")

# ── 10. Fig 9.3 — Boxplot BVRP por regime ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
data_by_regime = [df.loc[df["regime_vol"] == r, "vrp_30d"].dropna().values for r in REGIMES]
bp = ax.boxplot(data_by_regime, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=1.5))
for patch, regime in zip(bp["boxes"], REGIMES):
    patch.set_facecolor(COLORS[regime])
    patch.set_alpha(0.6)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(REGIMES, fontsize=9)
ax.axhline(0, linestyle="--", linewidth=0.8, color="grey")
ax.set_ylabel("BVRP (%)", fontsize=10)
ax.set_title("Distribuição do BVRP por Regime de Volatilidade", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap9_03_boxplot_bvrp_regimes.png", dpi=300)
plt.close(fig)
print("OK fig_cap9_03_boxplot_bvrp_regimes.png")

# ── 11. Fig 9.4 — Boxplot retornos futuros por regime (3 horizontes) ─────────
horizons_fig = [5, 10, 20]
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
for ax, h in zip(axes, horizons_fig):
    col = f"ret_fut_{h}d"
    data_by_regime = [df.loc[df["regime_vol"] == r, col].dropna().values * 100 for r in REGIMES]
    bp = ax.boxplot(data_by_regime, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, regime in zip(bp["boxes"], REGIMES):
        patch.set_facecolor(COLORS[regime])
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Baixa", "Média", "Alta"], fontsize=8)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="grey")
    ax.set_ylabel("Retorno futuro (%)" if h == 5 else "", fontsize=9)
    ax.set_title(f"Horizonte {h}d", fontsize=10)
fig.suptitle("Retorno Futuro por Regime de Volatilidade", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap9_04_boxplot_ret_regimes.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("OK fig_cap9_04_boxplot_ret_regimes.png")

# ── 12. Fig 9.5 — Scatter BVRP vs retorno futuro (3 horizontes) ──────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, h in zip(axes, horizons_fig):
    col = f"ret_fut_{h}d"
    for regime in REGIMES:
        sub = df.loc[df["regime_vol"] == regime]
        ax.scatter(sub["vrp_30d"], sub[col] * 100,
                   alpha=0.35, s=12, color=COLORS[regime], label=regime)
    # linha de tendência global
    slope, intercept, *_ = stats.linregress(df["vrp_30d"].dropna(), df[col].dropna() * 100)
    x_line = np.linspace(df["vrp_30d"].min(), df["vrp_30d"].max(), 200)
    ax.plot(x_line, intercept + slope * x_line, linewidth=1.4, color="black", linestyle="--")
    ax.set_xlabel("BVRP (%)", fontsize=9)
    ax.set_ylabel("Retorno futuro (%)" if h == 5 else "", fontsize=9)
    ax.set_title(f"Horizonte {h}d", fontsize=10)
    if h == 20:
        ax.legend(fontsize=7, markerscale=1.5)
fig.suptitle("BVRP versus Retorno Futuro por Regime", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap9_05_scatter_bvrp_ret.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("OK fig_cap9_05_scatter_bvrp_ret.png")

print("\nTodos os outputs do Cap. 9 gerados com sucesso.")
print(f"  Figuras : {OUT_FIGS}")
print(f"  Tabelas : {OUT_TABS}")
