"""Regenera a figura de coeficientes de RV e IV do Cap. 5 (modelo E1).

Reproduz exatamente a figura produzida pelo bloco 8 de code/regenerate_cap5.py
-- mesmo estilo, mesmas cores -- mas lendo os coeficientes ja publicados em
archive/tables/retornos_linear/tab_ols_rv_iv_multihoriz.tex, sem reestimar nada. Serve para
corrigir o rotulo do titulo ("OLS E1" -> "MQO E1", alem do travessao, que no
matplotlib aparecia como tres hifens literais) sem rodar o pipeline inteiro,
que reescreveria tabelas e o CSV do bootstrap.

Uso:
    python scripts/regenerate_cap5_fig_rv_iv.py [--out CAMINHO.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "archive" / "tables" / "retornos_linear" / "tab_ols_rv_iv_multihoriz.tex"
OUT = ROOT / "archive" / "figs" / "retornos_linear" / "beta_comparacao_rv_iv.png"

# Linha da tabela: h & beta_RV & t & p & beta_IV & ...
LINHA = re.compile(
    r"^\s*(\d+)\s*&\s*([+-]?[\d.]+)\s*&[^&]*&[^&]*&\s*([+-]?[\d.]+)\s*&"
)


def ler_coeficientes():
    """Extrai (horizonte, beta_RV, beta_IV) das linhas de dados da tabela."""
    linhas = []
    for linha in TAB.read_text(encoding="utf-8").splitlines():
        m = LINHA.match(linha)
        if m:
            linhas.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not linhas:
        raise SystemExit("Nenhuma linha de dados encontrada em %s" % TAB)
    linhas.sort()
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    dados = ler_coeficientes()
    horizons = [d[0] for d in dados]
    beta_rv = [d[1] * 100 for d in dados]
    beta_iv = [d[2] * 100 for d in dados]
    print("Horizontes lidos de %s: %s" % (TAB.name, horizons))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, beta_rv, marker="o", linewidth=2,
            label="$\\hat{\\beta}_{RV}$ (x100)", color="steelblue")
    ax.plot(horizons, beta_iv, marker="s", linewidth=2,
            label="$\\hat{\\beta}_{IV}$ (x100)", color="darkorange")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizonte $h$ (dias)")
    ax.set_ylabel("Coeficiente x100")
    ax.set_title("Coeficientes de RV e IV por horizonte — MQO E1 (HAC)\n"
                 "(teste: $\\hat{\\beta}_{RV} \\approx -\\hat{\\beta}_{IV}$?)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()
    print("Figura salva: %s" % args.out)


if __name__ == "__main__":
    main()
