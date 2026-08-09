"""Regenera a Figura 5.1 -- coeficiente do BVRP por horizonte com IC bootstrap.

O PNG original (figs/cap5/beta_por_horizonte_basico_bootstrap.png) nao tinha
gerador versionado: era um arquivo solto, com o titulo dizendo "OLS" e com os
rotulos de p-valor posicionados fora da area do grafico. Este script recria a
figura a partir de tables/cap5/bootstrap_ci_cap5.csv, que ja contem beta_hat,
os limites do IC 95% bootstrap e o p-HAC de cada horizonte.

Uso:
    python scripts/regenerate_cap5_fig_bootstrap.py [--out CAMINHO.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "tables" / "cap5" / "bootstrap_ci_cap5.csv"
OUT = ROOT / "figs" / "cap5" / "beta_por_horizonte_basico_bootstrap.png"

# Parametros do bootstrap, para a legenda (ver regenerate_cap5.py).
N_BOOTSTRAP = 1000
BLOCK_SIZE = 30

COR_LINHA = "#1f4e79"
COR_BANDA = "#c5d9ec"
COR_ZERO = "#d62728"


def virgula(x, casas=2):
    """Formata numero com virgula decimal para uso dentro de mathtext."""
    return ("%.*f" % (casas, x)).replace(".", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = pd.read_csv(CSV).sort_values("horizon").reset_index(drop=True)

    h = df["horizon"].to_numpy()
    # Coeficientes em pontos percentuais x100, como no eixo original.
    beta = df["beta_hat"].to_numpy() * 100
    lo = df["ic_lower"].to_numpy() * 100
    hi = df["ic_upper"].to_numpy() * 100
    p_hac = df["p_hac"].to_numpy()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(10, 6.2))

    ax.fill_between(
        h, lo, hi, color=COR_BANDA, alpha=0.85, linewidth=0,
        label="IC 95%% bootstrap (B=%s, bloco=%dd)"
              % ("{:,}".format(N_BOOTSTRAP).replace(",", "."), BLOCK_SIZE),
    )
    ax.axhline(0, color=COR_ZERO, linestyle="--", linewidth=1.6,
               label=r"$\beta = 0$ (referência)")
    ax.plot(h, beta, "o-", color=COR_LINHA, linewidth=2.2, markersize=8,
            label=r"$\hat{\beta}_h$ estimado (HAC)")

    # p-HAC junto de cada ponto (no original escaparam para fora dos eixos).
    # Os horizontes 1, 5 e 10 ficam proximos no eixo linear: alterna a altura
    # do rotulo nesse trecho para os textos nao se sobreporem.
    for i, (hi_, b_, p_) in enumerate(zip(h, beta, p_hac)):
        dy = -20 if (i >= 3 or i % 2 == 0) else -38
        ax.annotate(
            r"$p = %s$" % virgula(p_),
            xy=(hi_, b_), xytext=(0, dy), textcoords="offset points",
            ha="center", va="top", fontsize=10, color="#333333",
        )

    ax.set_title(
        "Coeficiente do BVRP por horizonte — MQO básico com IC 95% bootstrap\n"
        "(retorno futuro acumulado de $h$ dias; BVRP e retornos em % ×100)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel(r"Horizonte $h$ (dias)")
    ax.set_ylabel(r"Coeficiente $\hat{\beta}$ (BVRP) $\times 100$")

    ax.set_xticks(h)
    ax.set_xticklabels([str(int(v)) for v in h])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#999999")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figura salva: %s" % args.out)


if __name__ == "__main__":
    main()
