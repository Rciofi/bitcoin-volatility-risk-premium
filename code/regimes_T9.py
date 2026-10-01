"""
regimes_T9.py — regimes de volatilidade sem look-ahead (T9 do plano de revisão)

Variável de regime: volatilidade histórica de 30 dias (vh_30d), conhecida em t
(retornos de t-29 a t). Dois regimes (D4): alta volatilidade se vh_30d > corte.

Corte (decisão do autor, docs/pendencias_T1.md, seção 10):
  - PRINCIPAL: mistura de duas normais em log(vh_30d), estimada SÓ com a
    primeira janela de estimação do split_utils (23/04/2021 a 22/04/2023),
    convergida (tol = 1e-8, 10 inicializações, assert converged_); o corte é
    o ponto em que a probabilidade a posteriori do componente de alta passa
    de 1/2 (corte_mistura_log). Fica fixo para toda a amostra: 37,3% a.a.,
    que coincide com a antimoda da densidade de núcleo da 1ª janela (37,5).
  - ROBUSTEZ: o mesmo método reestimado em cada origem do split_utils
    (janela expansiva), só com os dados até a origem (vh_30d de s <= t é
    conhecida em t).

Por que não o BVRP com corte ~20 (o "~20" do plano): a moda local do BVRP em
+27 e a antimoda em +20 do histograma publicado vinham em boa parte do buraco
de março/2023 corrigido no T0 (30 das 73 datas com BVRP > 20 eram de
abril/2023); depois do T0, o BVRP é unimodal. A bimodalidade da RV (Fig. 8.2)
sobrevive.

Saídas: data/regimes_T9.csv (sem alterar ml_dataset_T4.csv) e, em
outputs/T9/, cortes por origem e figuras.

Uso:  python code/regimes_T9.py
"""

import io
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde, norm  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_utils import gerar_divisoes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "T9")
VAR = "vh_30d"
H = 30
COMMIT_PUBLICADO = "2b5adff"  # dados antes do T0 (histograma publicado da Fig. 3.2)
AZUL, LARANJA, CINZA = "#2b6cb0", "#d95f02", "#6b6b6b"  # par validado (skill dataviz) + cinza neutro


TOL_EM, N_INIT, MAX_ITER = 1e-8, 10, 10_000


def br(v, d):
    """Número com vírgula decimal (só o número; não mexe no resto do rótulo, como "a.a.")."""
    return f"{v:.{d}f}".replace(".", ",")


def ajustar_mistura(vh, random_state=0):
    """
    Mistura de duas normais em log(vh): (pesos, médias, desvios), ordenados
    pela média. Tolerância apertada, várias inicializações e convergência
    obrigatória: com o padrão do scikit-learn (tol = 1e-3), a estimação da
    primeira janela parava em 7 iterações, numa solução de log-verossimilhança
    -154,9 (corte 47,4), em vez da convergida, -134,2 (corte 37,3).
    """
    x = np.log(np.asarray(vh, dtype=float)).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=N_INIT, random_state=random_state,
                         tol=TOL_EM, max_iter=MAX_ITER).fit(x)
    assert gm.converged_, "mistura não convergiu"
    o = np.argsort(gm.means_.ravel())
    return gm.weights_[o], gm.means_.ravel()[o], np.sqrt(gm.covariances_.ravel()[o])


def _raizes_igual_posteriori(w, m, s):
    """Raízes reais de log(w2 N2) - log(w1 N1) = 0 em x = log(vh): uma quadrática."""
    a = 1 / (2 * s[0] ** 2) - 1 / (2 * s[1] ** 2)
    b = m[1] / s[1] ** 2 - m[0] / s[0] ** 2
    c = (m[0] ** 2 / (2 * s[0] ** 2) - m[1] ** 2 / (2 * s[1] ** 2)
         + np.log(w[1] / s[1]) - np.log(w[0] / s[0]))
    r = np.roots([a, b, c]) if abs(a) > 1e-15 else np.array([-c / b])
    return np.sort(r[np.isreal(r)].real), (a, b)


def corte_mistura_log(vh, random_state=0):
    """
    Corte em nível (% a.a.): o ponto x = log(vh) em que a probabilidade a
    posteriori do componente de ALTA volatilidade passa de < 1/2 para > 1/2.
    Com variâncias diferentes, f(x) = log(w2 N2) - log(w1 N1) é uma parábola e
    pode haver duas raízes; o corte é a MAIOR raiz em que f é crescente e que
    fica abaixo da média alta. Na 1ª janela: raízes 17,1 e 37,3 -> 37,3. Se o
    componente de alta domina todo o intervalo entre as médias (algumas origens
    de 2025), essa raiz fica logo abaixo da média baixa, e é ela que se usa.
    """
    w, m, s = ajustar_mistura(vh, random_state)
    r, (a, b) = _raizes_igual_posteriori(w, m, s)
    crescente = [x for x in r if 2 * a * x + b > 0 and x < m[1]]
    if not crescente:
        raise ValueError(f"sem ponto de igual posteriori abaixo da média alta: raízes {np.exp(r)}")
    return float(np.exp(max(crescente)))


def antimoda_kde(vh):
    """Antimoda da densidade de núcleo (em nível) entre as médias da mistura -- verificação independente."""
    vh = np.asarray(vh, dtype=float)
    w, m, s = ajustar_mistura(vh)
    g = np.linspace(np.exp(m[0]), np.exp(m[1]), 20_001)
    return float(g[np.argmin(gaussian_kde(vh)(g))])


def carregar():
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def cortes_por_origem(df):
    """Robustez: corte reestimado em cada origem (janela expansiva), com dados até a origem."""
    div = gerar_divisoes(len(df), h=H, janela="expansiva")
    linhas = []
    for o in div.origens:
        vh = df[VAR].to_numpy()[:o.t + 1]
        w, m, s = ajustar_mistura(vh)
        linhas.append({"origem": df.date[o.t], "n_obs_corte": o.t + 1, "corte": corte_mistura_log(vh),
                       "media_baixa": float(np.exp(m[0])), "media_alta": float(np.exp(m[1])),
                       "peso_baixa": float(w[0]), "desvio_log_baixa": float(s[0]), "desvio_log_alta": float(s[1])})
    return pd.DataFrame(linhas)


def regime_expansivo_por_data(df, cortes, corte_fixo):
    """Corte vigente em cada data: o da origem mais recente <= data; antes da 1ª origem, o corte fixo."""
    c = pd.Series(cortes.corte.to_numpy(), index=pd.to_datetime(cortes.origem))
    vigente = c.reindex(df.date, method="ffill").to_numpy()
    return np.where(np.isnan(vigente), corte_fixo, vigente)


def figuras(df, pj, corte, cortes):
    # (1) histograma da 1ª janela em log(RV), com a mistura e o corte
    w, m, s = ajustar_mistura(pj[VAR])
    x = np.log(pj[VAR].to_numpy())
    g = np.linspace(x.min() - 0.1, x.max() + 0.1, 600)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(x, bins=35, density=True, color=AZUL, alpha=0.45, label="1ª janela (23/04/2021 a 22/04/2023)")
    for k, cor in enumerate([AZUL, LARANJA]):
        ax.plot(g, w[k] * norm.pdf(g, m[k], s[k]), color=cor, linewidth=2,
                label=f"componente {k + 1}: média {br(np.exp(m[k]), 1)}% a.a., peso {br(w[k], 2)}")
    ax.axvline(np.log(corte), color="black", linestyle="--", linewidth=1.2,
               label=f"corte = {br(corte, 1)}% a.a.")
    tic = [20, 30, 40, 50, 60, 80, 100, 120]
    ax.set_xticks(np.log(tic))
    ax.set_xticklabels([str(t) for t in tic])
    ax.set_xlabel("Volatilidade histórica de 30 dias (% a.a., escala log)")
    ax.set_ylabel("densidade (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_T9_histograma_rv.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (2) bimodalidade do BVRP: publicada (antes do T0) x atual
    try:
        txt = subprocess.run(["git", "-C", ROOT, "show", f"{COMMIT_PUBLICADO}:data/vrp_with_targets.csv"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  [aviso] commit {COMMIT_PUBLICADO} indisponível: figura da bimodalidade do BVRP não gerada")
        txt = None
    if txt is not None:
        _figura_bvrp(df, pd.read_csv(io.StringIO(txt), parse_dates=["date"]))

    # (3) evolução do corte nas origens (robustez)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(cortes.origem), cortes.corte, color=LARANJA, marker="o", markersize=3, linewidth=1.4,
            label="corte reestimado em cada origem (janela expansiva)")
    ax.axhline(corte, color=AZUL, linestyle="--", linewidth=1.4,
               label=f"corte fixo da 1ª janela = {br(corte, 1)}% a.a.")
    ax.set_ylabel("corte da RV de 30 dias (% a.a.)")
    ax.set_xlabel("origem de previsão")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_T9_cortes_por_origem.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _figura_bvrp(df, old):
    """Bimodalidade do BVRP: histograma publicado (antes do T0) x atual."""
    abr = (old.date >= "2023-04-01") & (old.date <= "2023-04-30")
    bins = np.linspace(-45, 35, 41)
    gg = np.linspace(-45, 35, 600)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    ax = axes[0]
    ax.hist([old.vrp_30d[~abr], old.vrp_30d[abr]], bins=bins, stacked=True, density=True, alpha=0.55,
            color=[AZUL, LARANJA], label=["demais datas", "abril/2023 (artefato do buraco de março)"])
    ax.plot(gg, gaussian_kde(old.vrp_30d)(gg), color=CINZA, linewidth=2, label="densidade de núcleo")
    ax.axvline(20.2, color="black", linestyle="--", linewidth=1, label="antimoda em +20,2 (o \"~20\")")
    ax.set_title(f"(a) Publicado, antes do T0 (N = {len(old)}): 30 das {int((old.vrp_30d > 20).sum())} "
                 "datas com BVRP > 20 eram de abril/2023", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.hist(df.vrp_30d, bins=bins, density=True, color=AZUL, alpha=0.55, label="amostra de modelagem")
    ax.plot(gg, gaussian_kde(df.vrp_30d)(gg), color=CINZA, linewidth=2, label="densidade de núcleo")
    ax.set_title(f"(b) Depois do T0 (N = {len(df)}): unimodal", loc="left", fontsize=10)
    ax.set_xlabel("BVRP retrospectivo (proxy, p.p.)")
    ax.legend(fontsize=8, loc="upper left")
    for a in axes:
        a.set_ylabel("densidade")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_T9_bimodalidade_bvrp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = carregar()
    div = gerar_divisoes(len(df), h=H)
    pj = df.iloc[div.primeira_janela]
    corte = corte_mistura_log(pj[VAR])
    w, m, s = ajustar_mistura(pj[VAR])
    print(f"Primeira janela: {pj.date.iloc[0].date()} a {pj.date.iloc[-1].date()} ({len(pj)} obs.)")
    print(f"Mistura em log({VAR}): médias {np.round(np.exp(m), 1)}, pesos {np.round(w, 3)}; corte = {corte:.4f}")
    print(f"Antimoda da densidade de núcleo (verificação independente): {antimoda_kde(pj[VAR]):.2f}")

    cortes = cortes_por_origem(df)
    cortes.to_csv(os.path.join(OUT, "cortes_por_origem_T9.csv"), index=False)
    corte_exp = regime_expansivo_por_data(df, cortes, corte)

    reg = pd.DataFrame({"date": df.date, VAR: df[VAR], "corte_fixo": corte,
                        "regime_alta_fixo": (df[VAR] > corte).astype(int),
                        "corte_expansivo_vigente": corte_exp,
                        "regime_alta_expansivo": (df[VAR].to_numpy() > corte_exp).astype(int)})
    reg.to_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), index=False)

    oos = df.date >= cortes.origem.iloc[0]
    print(f"Regime de alta (corte fixo): {100 * reg.regime_alta_fixo[div.primeira_janela].mean():.1f}% da 1ª janela, "
          f"{100 * reg.regime_alta_fixo[oos].mean():.1f}% fora da amostra")
    print(f"Corte por origem: de {cortes.corte.iloc[0]:.1f} ({str(cortes.origem.iloc[0])[:10]}) a "
          f"{cortes.corte.iloc[-1]:.1f} ({str(cortes.origem.iloc[-1])[:10]}); mín. {cortes.corte.min():.1f}, "
          f"máx. {cortes.corte.max():.1f}")
    figuras(df, pj, corte, cortes)
    print(f"Saídas: data/regimes_T9.csv e {OUT}")


if __name__ == "__main__":
    main()
