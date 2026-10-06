"""
plot_vrp_vs_return_T12.py — dispersão BVRP x retorno futuro de 30 dias (T12)

Substitui, para o Cap. 3, a figura de code/plot_vrp_vs_return.py com h = 20
(figs/cap3/vrp_vs_return_20d.png) por h = 30, alinhando o horizonte ao do BVRP
(plano, T12). Figura descritiva: usa a proxy vrp_30d (conhecida em t) e a
amostra descritiva (data/vrp_with_targets.csv, N = 1.806). A reta é de MQO, e
a legenda traz b com erro-padrão de Newey–West com h+1 = 31 defasagens
(hac_utils), no lugar da correlação simples da versão antiga.

Grava em outputs/T12/; com --publicar-cap3, copia a figura para
figs/cap3/vrp_vs_return_30d.png, o nome usado no Cap. 3. Rótulo do eixo na
notação da seção 14 de docs/pendencias_T1.md (BVRP^proxy).

Uso:  python code/plot_vrp_vs_return_T12.py [--publicar-cap3]
"""

import argparse
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")   # β, R² no console mesmo com saída redirecionada (Windows)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hac_utils import mqo_newey_west  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "T12")
H = 30
AZUL, LARANJA = "#2b6cb0", "#d95f02"


def br(v, d):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publicar-cap3", action="store_true",
                    help="copia a figura para figs/cap3/vrp_vs_return_30d.png")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"]).sort_values("date")
    x, y = df["vrp_30d"], 100 * df[f"ret_fut_{H}d"]
    res = mqo_newey_west(y, x, h=H)
    a, b = res.params["const"], res.params["vrp_30d"]
    pd.DataFrame([{"h": H, "n": int(res.nobs), "inicio": df.date.min().date(), "fim": df.date.max().date(),
                   "beta_pp_por_pp": b, "ep_hac": res.bse["vrp_30d"], "t_hac": res.tvalues["vrp_30d"],
                   "p_hac": res.pvalues["vrp_30d"], "r2": res.rsquared}]).to_csv(
        os.path.join(OUT, "regressao_T12.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, s=12, alpha=0.4, color=AZUL, edgecolor="none", label=f"datas (N = {len(df):,})".replace(",", "."))
    g = np.linspace(x.min(), x.max(), 100)
    ax.plot(g, a + b * g, color=LARANJA, linewidth=2,
            label=f"MQO: β = {br(b, 3)} (EP HAC {br(res.bse['vrp_30d'], 3)}; p = {br(res.pvalues['vrp_30d'], 2)}); "
                  f"R² = {br(res.rsquared, 3)}")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"BVRP$^{\mathrm{proxy}}$ em $t$ (p.p.)")
    ax.set_ylabel(f"Retorno futuro de {H} dias (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_T12_vrp_vs_ret30d.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.publicar_cap3:
        shutil.copyfile(os.path.join(OUT, "fig_T12_vrp_vs_ret30d.png"),
                        os.path.join(ROOT, "figs", "cap3", "vrp_vs_return_30d.png"))
        print("Figura copiada para figs/cap3/vrp_vs_return_30d.png")
    print(f"T12: N = {int(res.nobs)}, β = {b:.4f}, EP HAC (31) = {res.bse['vrp_30d']:.4f}, "
          f"p = {res.pvalues['vrp_30d']:.3f}, R² = {res.rsquared:.4f}. Saídas em {OUT}")


if __name__ == "__main__":
    main()
