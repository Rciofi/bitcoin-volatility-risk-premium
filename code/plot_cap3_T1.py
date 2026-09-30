"""
plot_cap3_T1.py — figuras do Cap. 3 com as duas definições do BVRP (T1)

Versões T1 das três figuras de code/plot_vrp_timeseries.py, com a definição
prospectiva (bvrp_30d_fut, RV de t+1 a t+30 - IV_t) e a retrospectiva
(vrp_30d, proxy; RV de t-29 a t - IV_t) lado a lado. NÃO toca em figs/:
grava só em --out-dir.

Amostra: data/vrp_with_targets.csv (amostra de referência, sem data fixa).
Cores (validadas com o validador da skill dataviz, par categórico):
prospectiva = laranja, retrospectiva = azul; IV = cinza tracejado (referência).

Uso:  python code/plot_cap3_T1.py --out-dir outputs/T1/figs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "vrp_with_targets.csv")

COR_PROSP = "#d95f02"
COR_RETRO = "#2b6cb0"
COR_IV = "#6b6b6b"
ROTULO = {
    "bvrp_30d_fut": "BVRP prospectivo: RV(t+1 a t+30) − IV(t)",
    "vrp_30d":      "BVRP retrospectivo (proxy): RV(t−29 a t) − IV(t)",
}
COR = {"bvrp_30d_fut": COR_PROSP, "vrp_30d": COR_RETRO}

EVENTOS = [
    ("2022-05-09", "LUNA"),
    ("2022-06-13", "3AC/Celsius"),
    ("2022-11-08", "FTX"),
    ("2023-03-10", "Crise banc. EUA"),
    ("2024-01-10", "ETF spot SEC"),
]


def fig_serie(df, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(df["date"], df["rv_30d"], color=COR_RETRO, linewidth=0.8,
            label="RV 30d retrospectiva (t−29 a t)")
    ax.plot(df["date"], df["rv_30d_fut"], color=COR_PROSP, linewidth=0.8,
            label="RV 30d prospectiva (t+1 a t+30)")
    ax.plot(df["date"], df["iv_30d"], color=COR_IV, linewidth=0.8, linestyle="--",
            label="IV 30d (DVOL Deribit)")
    ax.set_ylabel("Volatilidade (% a.a.)")
    ax.set_title("(a) Volatilidade realizada (retrospectiva e prospectiva) e implícita",
                 loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col in ["bvrp_30d_fut", "vrp_30d"]:
        ax.plot(df["date"], df[col], color=COR[col], linewidth=0.8, label=ROTULO[col])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("BVRP 30D (p.p. de vol.)")
    ax.set_title("(b) Prêmio de risco de volatilidade: definições prospectiva e retrospectiva",
                 loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Data")

    for i, (data_ev, rotulo) in enumerate(EVENTOS):
        x = pd.to_datetime(data_ev)
        for a in axes:
            a.axvline(x, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)
        ymin, ymax = axes[0].get_ylim()
        y_offset = -4 if i % 2 == 0 else -20
        axes[0].annotate(rotulo, xy=(x, ymax), xytext=(0, y_offset),
                         textcoords="offset points", ha="center", va="top", fontsize=7,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.4, alpha=0.85))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_histograma(df, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    todos = pd.concat([df["bvrp_30d_fut"], df["vrp_30d"]])
    bins = np.linspace(todos.min(), todos.max(), 31)
    grade = np.linspace(todos.min(), todos.max(), 400)
    for col in ["bvrp_30d_fut", "vrp_30d"]:
        s = df[col].dropna()
        # contorno (step) em vez de barras translúcidas: a sobreposição de
        # duas cores preenchidas gera uma terceira cor que não é nenhuma série
        ax.hist(s, bins=bins, density=True, histtype="step", color=COR[col],
                linewidth=1, alpha=0.8)
        ax.plot(grade, stats.gaussian_kde(s)(grade), color=COR[col], linewidth=2,
                label=f"{ROTULO[col]} — média {s.mean():.2f}".replace(".", ","))
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("BVRP 30D (p.p. de vol.)")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_boxplot_ano(df, out):
    anos = sorted(df["date"].dt.year.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    largura = 0.36
    for k, col in enumerate(["bvrp_30d_fut", "vrp_30d"]):
        dados = [df.loc[df["date"].dt.year == a, col].dropna().values for a in anos]
        pos = np.arange(len(anos)) + (k - 0.5) * (largura + 0.04)
        bp = ax.boxplot(dados, positions=pos, widths=largura, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=2, alpha=0.5,
                                        markerfacecolor=COR[col], markeredgecolor=COR[col]))
        for patch in bp["boxes"]:
            patch.set_facecolor(COR[col])
            patch.set_alpha(0.55)
        ax.plot([], [], color=COR[col], linewidth=6, alpha=0.55, label=ROTULO[col])
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xticks(np.arange(len(anos)))
    # último ano da amostra é incompleto (C3.2): explicita os meses cobertos
    ult = df["date"].max()
    meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    rotulos = [str(a) for a in anos]
    if ult.month < 12:
        rotulos[-1] = f"{anos[-1]}\n(jan–{meses[ult.month - 1]})"
    ax.set_xticklabels(rotulos)
    ax.set_xlabel("Ano")
    ax.set_ylabel("BVRP 30D (p.p. de vol.)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T1", "figs"))
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    faltando = {"rv_30d_fut", "bvrp_30d_fut"} - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas do T1 ausentes em {DATA_PATH}: {sorted(faltando)}. Rode o pipeline.")
    print(f"Amostra: N={len(df)}  {df['date'].min().date()} a {df['date'].max().date()}")

    saidas = {
        "vrp_timeseries_T1.png": fig_serie,
        "vrp_histogram_kde_T1.png": fig_histograma,
        "vrp_boxplot_T1.png": fig_boxplot_ano,
    }
    for nome, f in saidas.items():
        caminho = os.path.join(out_dir, nome)
        f(df, caminho)
        print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    main()
