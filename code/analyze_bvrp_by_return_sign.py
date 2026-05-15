"""
analyze_bvrp_by_return_sign.py
===============================
Análise condicional do BVRP por sinal do retorno corrente do Bitcoin.

Investiga se o BVRP exibe comportamento assimétrico em função do sinal
do retorno corrente do BTC (analogia ao leverage effect documentado em
equity para a volatilidade implícita).

Saídas:
  - tables/cap9/bvrp_by_return_sign.csv      (estatísticas brutas)
  - tables/cap9/tab_bvrp_sinal_retorno.tex   (tabela LaTeX)
  - figs/cap9/fig_cap9_bvrp_sinal.png        (boxplot por grupo)

Autor: Rodrigo Ciofi
Dissertação: Prêmio de Risco de Volatilidade do Bitcoin (BVRP) — FGV EESP
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Caminhos ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "vrp_with_targets.csv")
TABLE_DIR   = os.path.join(PROJECT_DIR, "tables", "cap9")
FIG_DIR     = os.path.join(PROJECT_DIR, "figs",   "cap9")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

# ─── 1. Carregar dados ────────────────────────────────────────────────────────
print("=" * 60)
print("ANÁLISE CONDICIONAL DO BVRP POR SINAL DO RETORNO BTC")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
n_total = len(df)
print(f"\nTotal de observações carregadas: {n_total}")

# ─── 2. Criar variável sinal_ret ─────────────────────────────────────────────
n_zero = (df["ret"] == 0).sum()
df_work = df[df["ret"] != 0].copy()
df_pos  = df_work[df_work["ret"] >  0]
df_neg  = df_work[df_work["ret"] <  0]

n_pos  = len(df_pos)
n_neg  = len(df_neg)
n_used = n_pos + n_neg

print(f"\n--- Partição por sinal do retorno corrente ---")
print(f"  ret > 0  (positivo)  : {n_pos:>5} obs")
print(f"  ret < 0  (negativo)  : {n_neg:>5} obs")
print(f"  ret == 0 (descartado): {n_zero:>5} obs")
print(f"  Total utilizado      : {n_used:>5} obs")

# ─── 3. Estatísticas descritivas do BVRP por grupo ───────────────────────────
def desc_stats(series, label):
    return {
        "grupo"  : label,
        "N"      : len(series),
        "media"  : series.mean(),
        "mediana": series.median(),
        "dp"     : series.std(ddof=1),
        "minimo" : series.min(),
        "maximo" : series.max(),
    }

stats_pos = desc_stats(df_pos["vrp_30d"], "ret > 0")
stats_neg = desc_stats(df_neg["vrp_30d"], "ret < 0")

print("\n--- Estatísticas descritivas do BVRP (vrp_30d) ---")
for s in [stats_pos, stats_neg]:
    print(f"\n  Grupo : {s['grupo']}")
    print(f"    N       : {s['N']}")
    print(f"    Média   : {s['media']:+.4f} p.p.")
    print(f"    Mediana : {s['mediana']:+.4f} p.p.")
    print(f"    D.P.    : {s['dp']:.4f} p.p.")
    print(f"    Mín     : {s['minimo']:+.4f} p.p.")
    print(f"    Máx     : {s['maximo']:+.4f} p.p.")

# ─── 4. Teste t de Welch — vrp_30d ────────────────────────────────────────────
t_stat, p_val = stats.ttest_ind(
    df_pos["vrp_30d"], df_neg["vrp_30d"], equal_var=False
)
diff_means = stats_pos["media"] - stats_neg["media"]

print("\n--- Teste t de Welch: BVRP médio (ret>0) − BVRP médio (ret<0) ---")
print(f"  Diferença de médias : {diff_means:+.4f} p.p.")
print(f"  Estatística t       : {t_stat:+.4f}")
print(f"  p-valor             : {p_val:.4f}")

# ─── 5. Retornos futuros condicionais ─────────────────────────────────────────
future_cols = ["ret_fut_5d", "ret_fut_20d", "ret_fut_30d"]
future_labels = {
    "ret_fut_5d" : "Retorno futuro  5 dias",
    "ret_fut_20d": "Retorno futuro 20 dias",
    "ret_fut_30d": "Retorno futuro 30 dias",
}

print("\n--- Retornos futuros médios por grupo (teste de Welch) ---")
future_results = []
for col in future_cols:
    s_pos = df_pos[col].dropna()
    s_neg = df_neg[col].dropna()
    mean_pos = s_pos.mean()
    mean_neg = s_neg.mean()
    diff_f   = mean_pos - mean_neg
    t_f, p_f = stats.ttest_ind(s_pos, s_neg, equal_var=False)
    print(f"\n  {future_labels[col]}")
    print(f"    Média (ret>0): {mean_pos:+.4f}")
    print(f"    Média (ret<0): {mean_neg:+.4f}")
    print(f"    Diferença    : {diff_f:+.4f}")
    print(f"    p-valor Welch: {p_f:.4f}")
    future_results.append({
        "variavel": col,
        "media_pos": mean_pos,
        "media_neg": mean_neg,
        "diferenca": diff_f,
        "t_stat"   : t_f,
        "p_valor"  : p_f,
    })

# ─── 6. Salvar CSV com estatísticas brutas ───────────────────────────────────
rows_csv = []

# BVRP
rows_csv.append({
    "variavel"  : "vrp_30d",
    "label"     : "BVRP (vrp_30d)",
    "N_pos"     : stats_pos["N"],
    "N_neg"     : stats_neg["N"],
    "media_pos" : stats_pos["media"],
    "media_neg" : stats_neg["media"],
    "mediana_pos": stats_pos["mediana"],
    "mediana_neg": stats_neg["mediana"],
    "dp_pos"    : stats_pos["dp"],
    "dp_neg"    : stats_neg["dp"],
    "diferenca" : diff_means,
    "t_stat"    : t_stat,
    "p_valor"   : p_val,
})
# Retornos futuros
for r in future_results:
    rows_csv.append({
        "variavel"  : r["variavel"],
        "label"     : future_labels[r["variavel"]],
        "N_pos"     : len(df_pos[r["variavel"]].dropna()),
        "N_neg"     : len(df_neg[r["variavel"]].dropna()),
        "media_pos" : r["media_pos"],
        "media_neg" : r["media_neg"],
        "mediana_pos": np.nan,
        "mediana_neg": np.nan,
        "dp_pos"    : np.nan,
        "dp_neg"    : np.nan,
        "diferenca" : r["diferenca"],
        "t_stat"    : r["t_stat"],
        "p_valor"   : r["p_valor"],
    })

df_out = pd.DataFrame(rows_csv)
csv_path = os.path.join(TABLE_DIR, "bvrp_by_return_sign.csv")
df_out.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\n✓ CSV salvo: {csv_path}")

# ─── 7. Tabela LaTeX ─────────────────────────────────────────────────────────
def fmt_num(val, decimals=3):
    """Formata número com vírgula decimal (padrão pt-BR)."""
    if pd.isna(val):
        return "---"
    return f"{val:.{decimals}f}".replace(".", ",")

def fmt_pval(p):
    """Formata p-valor com asteriscos de significância."""
    if pd.isna(p):
        return "---"
    stars = ""
    if p < 0.01:
        stars = "^{***}"
    elif p < 0.05:
        stars = "^{**}"
    elif p < 0.10:
        stars = "^{*}"
    return f"${fmt_num(p, 3)}{stars}$"

def fmt_diff(val, decimals=3):
    sign = "+" if val > 0 else ""
    return f"{sign}{fmt_num(val, decimals)}"

tex_path = os.path.join(TABLE_DIR, "tab_bvrp_sinal_retorno.tex")

tex_lines = []
tex_lines.append(r"% Tabela gerada automaticamente por analyze_bvrp_by_return_sign.py")
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"  \centering")
tex_lines.append(r"  \caption{BVRP e retornos futuros do Bitcoin condicionais ao sinal do retorno")
tex_lines.append(r"    corrente. Estatísticas reportadas em pontos percentuais.")
tex_lines.append(r"    Diferença = média do grupo \textit{ret}>0 menos média do grupo \textit{ret}<0.")
tex_lines.append(r"    Teste de Welch (variâncias desiguais): $^{*}p<0{,}10$; $^{**}p<0{,}05$; $^{***}p<0{,}01$.}")
tex_lines.append(r"  \label{tab:cap9_sinal_retorno}")
tex_lines.append(r"  \begin{tabular}{lrrrr}")
tex_lines.append(r"    \toprule")
tex_lines.append(r"    \textbf{Variável} & \textbf{BVRP $|$ \textit{ret}>0} & \textbf{BVRP $|$ \textit{ret}<0} & \textbf{Diferença} & \textbf{\textit{p}-valor Welch} \\")
tex_lines.append(r"    \midrule")

# Linha BVRP
tex_lines.append(
    f"    BVRP médio (\\texttt{{vrp\\_30d}}) & "
    f"${fmt_num(stats_pos['media'])}$ & "
    f"${fmt_num(stats_neg['media'])}$ & "
    f"${fmt_diff(diff_means)}$ & "
    f"{fmt_pval(p_val)} \\\\"
)

tex_lines.append(r"    \midrule")

# Linhas de retornos futuros
for r in future_results:
    lbl_map = {
        "ret_fut_5d" : "Retorno futuro médio ($h=5$)",
        "ret_fut_20d": "Retorno futuro médio ($h=20$)",
        "ret_fut_30d": "Retorno futuro médio ($h=30$)",
    }
    tex_lines.append(
        f"    {lbl_map[r['variavel']]} & "
        f"${fmt_num(r['media_pos'])}$ & "
        f"${fmt_num(r['media_neg'])}$ & "
        f"${fmt_diff(r['diferenca'])}$ & "
        f"{fmt_pval(r['p_valor'])} \\\\"
    )

tex_lines.append(r"    \midrule")

# Linha N
tex_lines.append(
    f"    $N$ & ${n_pos}$ & ${n_neg}$ & & \\\\"
)

tex_lines.append(r"    \bottomrule")
tex_lines.append(r"  \end{tabular}")
tex_lines.append(r"\end{table}")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines) + "\n")

print(f"✓ Tabela LaTeX salva: {tex_path}")

# ─── 8. Figura: Boxplot BVRP por grupo ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

data_plot = [df_pos["vrp_30d"].values, df_neg["vrp_30d"].values]
labels_plot = [f"ret > 0\n(N={n_pos})", f"ret < 0\n(N={n_neg})"]

bp = ax.boxplot(
    data_plot,
    labels=labels_plot,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=2.5, alpha=0.4, linestyle="none"),
)

colors = ["#4C9BE8", "#E8694C"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Marcar médias
for i, (data, color) in enumerate(zip(data_plot, colors), start=1):
    mean_val = np.mean(data)
    ax.plot(i, mean_val, marker="D", color="white",
            markeredgecolor="black", markersize=6, zorder=5,
            label="_nolegend_")
    ax.annotate(
        f"Média: {mean_val:+.2f}",
        xy=(i, mean_val),
        xytext=(i + 0.22, mean_val),
        fontsize=8.5,
        va="center",
    )

# Linha horizontal em zero
ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)

ax.set_ylabel("BVRP — $RV_{30d} - IV_{30d}$ (p.p.)", fontsize=10)
ax.set_title("Distribuição do BVRP condicionada ao sinal do retorno corrente do BTC",
             fontsize=10, pad=10)
ax.set_xlabel("Grupo (sinal do retorno corrente)", fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Legenda de significância
p_str = f"Teste de Welch: $t = {t_stat:+.3f}$,  $p = {p_val:.3f}$"
ax.text(0.98, 0.02, p_str, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

patch_pos = mpatches.Patch(color="#4C9BE8", alpha=0.75, label="ret > 0 (positivo)")
patch_neg = mpatches.Patch(color="#E8694C", alpha=0.75, label="ret < 0 (negativo)")
ax.legend(handles=[patch_pos, patch_neg], fontsize=8.5, loc="upper right")

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig_cap9_bvrp_sinal.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Figura salva: {fig_path}")

print("\n" + "=" * 60)
print("EXECUÇÃO CONCLUÍDA")
print("=" * 60)
