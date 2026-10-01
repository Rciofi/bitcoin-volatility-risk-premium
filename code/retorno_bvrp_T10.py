"""
retorno_bvrp_T10.py — retorno futuro sobre o BVRP previsto, versão linear (T10)

Amostra: as 987 datas fora da amostra do T5 (21/06/2023 a 03/03/2026, janela
expansiva), em que o BVRP previsto existe. Alvos: ret_fut_h de
data/vrp_with_targets.csv, h = 1, 5, 10, 20, 30, 60. Erro-padrão de
Newey–West com h+1 defasagens (hac_utils.mqo_newey_west).

Especificações
  1. Principal: ret_fut_h = a + b · BVRP_previsto(t) + e. Regressor principal:
     Ridge do T5; robustez: floresta aleatória (seção 9.2 das pendências,
     fixado antes de ver os resultados).
  2. Componentes (C6.5): ret_fut_h ~ vh_30d + iv_30d (proxy), mesma amostra;
     teste de b1 + b2 = 0.
  3. Regime (T9): ret_fut_h ~ BVRP_previsto + D + D·BVRP_previsto, D =
     regime_alta_fixo (corte de 37,3); Wald conjunto de D e da interação.
     Robustez: D = regime_alta_expansivo (corte reestimado, vigente em t).
  4. Comparação com o publicado: ret_fut_h ~ vrp_30d na amostra descritiva
     completa (N = 1.806), com maxlags = h (como nas tabelas publicadas do
     Cap. 5) e com h+1 (T3).

Regressor gerado (Pagan, 1984) — bootstrap em blocos móveis em DOIS NÍVEIS
  1º nível (incerteza do regressor gerado): em cada uma das 33 origens, o
  primeiro estágio é reestimado num treino reamostrado em blocos DENTRO da
  janela de treino da própria origem (posições 0 .. t-30: embargo exato) e
  prevê as datas reais de teste da origem.
  2º nível (incerteza amostral): os 987 pares (previsão reestimada, retorno
  real) são reamostrados em blocos dentro do período fora da amostra, e b_h
  é reestimado.
  Desenho anterior, abandonado: reamostrar a série inteira e aplicar as
  origens por posição misturava épocas nas "datas fora da amostra"; o
  bootstrap estimava outro parâmetro (média dos b* perto de zero; IC sem o b^
  em 0 de 24 combinações no teste de fumaça).
    - PRINCIPAL: Ridge com o α de cada origem do T5 FIXO, bloco de 60, B = 999;
    - sensibilidade: α fixo com blocos de 30 e 90 (B = 999); α REESCOLHIDO
      por validação cruzada embargada em cada treino reamostrado (B = 999) --
      distorcido: blocos repetidos caem no treino e na validação da mesma
      dobra, e a validação cruzada passa a favorecer α menores (distorção do
      procedimento, não incerteza real da seleção); floresta aleatória com os
      hiperparâmetros de cada origem do T5 fixos (bloco de 60, B = 199);
    - contraprova: só o 2º nível (previsões originais fixas), B = 999.
  Atenuação: a reestimação do 1º estágio acrescenta ruído à previsão e atenua
  o b do 2º estágio; λ = média dos b* / b^ mede essa atenuação (no mundo do
  bootstrap, estima a atenuação de b^); viés = média dos b* - b^.
  Inferência, para cada h, na ESCALA COERENTE: os b* e seu EP estão na escala
  atenuada, então o EP de b^ é EP_boot/λ; teste de H0: b = 0 por
  b^/(EP_boot/λ) = média dos b*/EP_boot (normal); IC principal
  b^ ± 1,96·EP_boot/λ. (O teste b^/EP_boot, usado antes, misturava escalas --
  b^ não atenuado sobre EP atenuado -- e dava p pequenos demais: o EP em dois
  níveis saía MENOR que o EP só do 2º nível.) Ao lado: EP HAC (ignora o 1º
  estágio), EP só do 2º nível e as razões entre os EPs; IC básico
  (2b^ - q97,5; 2b^ - q2,5) como complemento -- refere-se à relação corrigida
  da atenuação, não a b^.
  Ressalvas: nas junções entre blocos, a ordem temporal deixa de valer
  (limitação padrão do bootstrap em blocos); no 2º nível há só ~16 blocos de
  60 dias, então os intervalos por percentis são grosseiros e o EP HAC é
  reportado ao lado.

Comparações múltiplas: 6 horizontes; limite de Bonferroni 0,05/6 = 0,0083.

Uso:  python code/retorno_bvrp_T10.py            (rodada completa)
      python code/retorno_bvrp_T10.py --rapido   (teste de fumaça: B pequeno)
"""

import argparse
import ast
import os
import re
import sys
import time
import warnings

sys.stdout.reconfigure(encoding="utf-8")   # β, − no console mesmo com saída redirecionada (Windows)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import previsao_bvrp_T5 as P  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "T10")
HS = [1, 5, 10, 20, 30, 60]
H_ALVO = 30
BONFERRONI = 0.05 / len(HS)
VARIANTE_PRINCIPAL = "Ridge_alpha_fixo"   # dois níveis, α de cada origem do T5, bloco de 60
SEMENTE = 20261001
AZUL, LARANJA = "#2b6cb0", "#d95f02"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"])
    vt = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    reg = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])
    df = (df.merge(vt[["date"] + [f"ret_fut_{h}d" for h in HS]], on="date", how="left")
            .merge(reg[["date", "regime_alta_fixo", "regime_alta_expansivo"]], on="date", how="left")
            .sort_values("date").reset_index(drop=True))
    assert df[[f"ret_fut_{h}d" for h in HS] + ["regime_alta_fixo", "regime_alta_expansivo"]].notna().all().all()
    prev = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
    prev = prev[prev.janela == "expansiva"]
    hp = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "hiperparametros_T5.csv"), parse_dates=["origem"])
    hp = hp[hp.janela == "expansiva"]
    return df, vt.sort_values("date").reset_index(drop=True), prev, hp


def _params(s):
    """Lê o repr gravado no T5 (com np.float64(...)) como dict."""
    return ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", s))


# ---------------------------------------------------------------------------
# Regressões
# ---------------------------------------------------------------------------
def principal(df, prev):
    linhas = []
    for modelo in ["Ridge", "floresta_aleatoria"]:
        p = prev[prev.modelo == modelo].set_index("date")["previsao"].rename("bvrp_previsto")
        d = df.set_index("date").loc[p.index]
        for h in HS:
            res = mqo_newey_west(d[f"ret_fut_{h}d"], p, h=h)
            linhas.append({"modelo": modelo, "h": h, "n": int(res.nobs), "beta": res.params["bvrp_previsto"],
                           "ep_hac": res.bse["bvrp_previsto"], "t_hac": res.tvalues["bvrp_previsto"],
                           "p_hac": res.pvalues["bvrp_previsto"], "r2": res.rsquared,
                           "sobrevive_bonferroni_hac": bool(res.pvalues["bvrp_previsto"] < BONFERRONI)})
    return pd.DataFrame(linhas)


def componentes(df, prev):
    datas = prev[prev.modelo == "Ridge"].date
    d = df.set_index("date").loc[datas]
    linhas = []
    for h in HS:
        res = mqo_newey_west(d[f"ret_fut_{h}d"], d[["vh_30d", "iv_30d"]], h=h)
        soma = res.t_test("vh_30d + iv_30d = 0")
        linhas.append({"h": h, "n": int(res.nobs),
                       "b_vh": res.params["vh_30d"], "p_vh": res.pvalues["vh_30d"],
                       "b_iv": res.params["iv_30d"], "p_iv": res.pvalues["iv_30d"],
                       "b_soma": float(soma.effect.ravel()[0]), "ep_soma": float(soma.sd.ravel()[0]),
                       "p_soma": float(soma.pvalue), "r2": res.rsquared})
    return pd.DataFrame(linhas)


def regime(df, prev):
    p = prev[prev.modelo == "Ridge"].set_index("date")["previsao"]
    d = df.set_index("date").loc[p.index]
    linhas = []
    for corte in ["fixo", "expansivo"]:   # fixo: principal; expansivo: robustez (T9)
        X = pd.DataFrame({"bvrp_previsto": p, "regime_alta": d[f"regime_alta_{corte}"]})
        X["interacao"] = X.bvrp_previsto * X.regime_alta
        for h in HS:
            res = mqo_newey_west(d[f"ret_fut_{h}d"], X, h=h)
            w = res.wald_test("regime_alta = 0, interacao = 0", use_f=False, scalar=True)
            linhas.append({"corte": corte, "h": h, "n": int(res.nobs), "pct_alta": 100 * float(X.regime_alta.mean()),
                           **{f"b_{c}": res.params[c] for c in X.columns},
                           **{f"p_{c}": res.pvalues[c] for c in X.columns},
                           "qui2_conjunto": float(w.statistic), "p_conjunto": float(w.pvalue), "r2": res.rsquared})
    return pd.DataFrame(linhas)


def publicado(vt):
    """ret_fut_h ~ vrp_30d na amostra descritiva (N = 1.806): maxlags = h (publicado) vs. h+1 (T3)."""
    linhas = []
    for h in HS:
        y, x = vt[f"ret_fut_{h}d"], vt[["vrp_30d"]]
        a = sensibilidade_defasagens(y, x, lags=[h])[h]
        b = mqo_newey_west(y, x, h=h)
        linhas.append({"h": h, "n": int(b.nobs), "beta": b.params["vrp_30d"],
                       "ep_maxlags_h": a.bse["vrp_30d"], "p_maxlags_h": a.pvalues["vrp_30d"],
                       "ep_maxlags_h1": b.bse["vrp_30d"], "p_maxlags_h1": b.pvalues["vrp_30d"],
                       "razao_ep": b.bse["vrp_30d"] / a.bse["vrp_30d"]})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Bootstrap em blocos móveis com reestimação do primeiro estágio
# ---------------------------------------------------------------------------
def indices_blocos(n, b, rng):
    """Blocos móveis (não circulares) de tamanho b concatenados até n posições (0 .. n-1)."""
    inicios = rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))
    return np.concatenate([np.arange(s, s + b) for s in inicios])[:n]


def sortear_indices(r, b, origens, n_oos):
    """
    Índices de uma reamostragem do bootstrap em DOIS NÍVEIS (determinístico na semente):
      1º nível: para cada origem, posições do treino da própria origem (0 .. t-30),
                reamostradas em blocos -> embargo exato;
      2º nível: posições 0 .. n_oos-1 das datas fora da amostra, reamostradas em blocos.
    """
    rng = np.random.default_rng([SEMENTE, b, r])
    treinos = [o.treino[indices_blocos(len(o.treino), b, rng)] for o in origens]
    return treinos, indices_blocos(n_oos, b, rng)


def primeiro_estagio(modelo, X, y, origens, params_fixos=None, treinos=None):
    """
    Previsões nas datas fora da amostra (variáveis reais). treinos=None usa o
    treino original de cada origem; senão, as posições reamostradas (1º nível).
    α/hiperparâmetros reescolhidos (params_fixos=None, validação cruzada
    embargada sobre o treino usado) ou fixos no valor de cada origem do T5.
    """
    prev = []
    for k, o in enumerate(origens):
        tr = o.treino if treinos is None else treinos[k]
        Xtr, ytr = X[tr], y[tr]
        if params_fixos is None:
            p, _ = P.escolher(modelo, P.GRADE_RIDGE, Xtr, ytr, np.arange(len(tr)))
        else:
            p = params_fixos[k]
        prever, _ = P.ajustar(modelo, p, Xtr, ytr, n_jobs=1)
        prev.append(prever(X[o.teste]))
    return np.concatenate(prev)


def betas(prev, R):
    """b_h por MQO (sem HAC: só o coeficiente) para cada coluna de R."""
    Z = np.column_stack([np.ones(len(prev)), prev])
    return np.linalg.lstsq(Z, R, rcond=None)[0][1]


def uma_reamostragem(r, modelo, X, y, R, origens, oos, b, params_fixos):
    """1º nível: reestima o 1º estágio em treinos reamostrados; 2º nível: reamostra os pares fora da amostra."""
    treinos, j = sortear_indices(r, b, origens, len(oos))
    prev = primeiro_estagio(modelo, X, y, origens, params_fixos, treinos)
    return betas(prev[j], R[oos][j])


def bootstrap_so_segundo_nivel(prev_fixa, R_oos, B, b):
    """Contraprova: previsões originais fixas, só os pares fora da amostra reamostrados em blocos."""
    res = [betas(prev_fixa[j], R_oos[j])
           for j in (indices_blocos(len(prev_fixa), b, np.random.default_rng([SEMENTE, b, r, 2])) for r in range(B))]
    return pd.DataFrame(np.array(res), columns=HS).assign(variante="Ridge_so_segundo_nivel", bloco=b)


def bootstrap(rotulo, modelo, X, y, R, B, b, params_fixos=None):
    origens = gerar_divisoes(len(y), h=H_ALVO, janela="expansiva").origens
    oos = np.concatenate([o.teste for o in origens])
    t0 = time.perf_counter()
    res = Parallel(n_jobs=-1)(delayed(uma_reamostragem)(r, modelo, X, y, R, origens, oos, b, params_fixos)
                              for r in range(B))
    print(f"  bootstrap {rotulo}: B = {B}, bloco = {b}, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(np.array(res), columns=HS).assign(variante=rotulo, bloco=b)


def resumo_bootstrap(boot, pr):
    """
    Inferência na escala coerente: os b* do bootstrap em dois níveis estão
    atenuados por λ = média(b*)/b^, e o EP bruto também; o EP de b^ é
    EP_boot/λ, e o teste b^/(EP_boot/λ) equivale a média(b*)/EP_boot.
    """
    ep_so2 = boot[boot.variante == "Ridge_so_segundo_nivel"][HS].std(ddof=1)
    linhas = []
    for (var, b), g in boot.groupby(["variante", "bloco"]):
        modelo = "floresta_aleatoria" if var.startswith("floresta") else "Ridge"
        for h in HS:
            est = pr[(pr.modelo == modelo) & (pr.h == h)].iloc[0]
            bh = g[h].to_numpy()
            ep = float(bh.std(ddof=1))
            media = float(bh.mean())
            lam = media / est.beta
            if var == "Ridge_so_segundo_nivel":        # sem 1º estágio: nada a corrigir
                ep_c = ep
            else:
                ep_c = ep / lam if lam > 0 else np.nan     # λ <= 0: correção indefinida
            z = est.beta / ep_c
            p = float(2 * stats.norm.sf(abs(z)))
            q025, q975 = np.percentile(bh, [2.5, 97.5])
            so2 = float(ep_so2[h]) if modelo == "Ridge" else np.nan
            linhas.append({"variante": var, "bloco": b, "B": len(bh), "h": h, "beta": est.beta,
                           "ep_hac": est.ep_hac, "p_hac": est.p_hac,
                           "media_boot": media, "vies": media - est.beta, "razao_media_beta": lam,
                           "ep_boot": ep, "ep_corrigido": ep_c, "ep_so_segundo_nivel": so2,
                           "razao_ep_boot_so2": ep / so2, "razao_ep_corrigido_so2": ep_c / so2,
                           "razao_ep_corrigido_hac": ep_c / est.ep_hac,
                           "z_boot": z, "p_boot": p,
                           "ic_inf": est.beta - 1.96 * ep_c, "ic_sup": est.beta + 1.96 * ep_c,
                           "ic_basico_inf": 2 * est.beta - q975, "ic_basico_sup": 2 * est.beta - q025,
                           "sobrevive_bonferroni_boot": bool(p < BONFERRONI)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
def figura(pr, rb, caminho):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pr = pr[pr.modelo == "Ridge"].set_index("h")
    rb = rb[(rb.variante == VARIANTE_PRINCIPAL) & (rb.bloco == 60)].set_index("h")
    x = np.arange(len(HS))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.errorbar(x - 0.12, pr.beta, yerr=1.96 * pr.ep_hac, fmt="o", color=AZUL, capsize=4,
                label=r"$\hat\beta$ ± 1,96 EP HAC (h+1) — ignora o 1º estágio")
    ax.errorbar(x + 0.12, rb.beta, yerr=1.96 * rb.ep_corrigido, fmt="s", color=LARANJA, capsize=4,
                label=r"$\hat\beta$ ± 1,96 EP bootstrap em dois níveis corrigido da atenuação (EP / $\lambda$)")
    ax.vlines(x + 0.3, rb.ic_basico_inf, rb.ic_basico_sup, color=LARANJA, linewidth=1.2, linestyle="--",
              label=r"IC básico (relação corrigida da atenuação, não $\hat\beta$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}" for h in HS])
    ax.set_xlabel("Horizonte h (dias)")
    ax.set_ylabel("β: retorno futuro sobre o BVRP previsto")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="teste de fumaça: B = 8 (floresta: 4)")
    ap.add_argument("--B", type=int, default=999, help="reamostragens do Ridge (cada variante)")
    ap.add_argument("--B-floresta", type=int, default=199, help="reamostragens da floresta")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    df, vt, prev, hp = carregar()

    pr, co, rg, pu = principal(df, prev), componentes(df, prev), regime(df, prev), publicado(vt)
    for nome, t in [("principal", pr), ("componentes", co), ("regime", rg), ("publicado_h_vs_h1", pu)]:
        t.to_csv(os.path.join(OUT, f"{nome}_T10.csv"), index=False)

    # bootstrap
    cols_t4 = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), nrows=1).columns
    feats_arv = [c for c in cols_t4 if c not in ("date", P.ALVO)]       # 17, como no T5
    feats_lin = [c for c in feats_arv if c != "vh_30d"]                # 16, como no T5
    y = df[P.ALVO].to_numpy(float)
    R = df[[f"ret_fut_{h}d" for h in HS]].to_numpy(float)
    alfa_fixo = [_params(s) for s in hp[hp.modelo == "Ridge"].sort_values("origem").params]
    rf_fixo = [_params(s) for s in hp[hp.modelo == "floresta_aleatoria"].sort_values("origem").params]
    B, B_rf = (8, 4) if args.rapido else (args.B, args.B_floresta)
    Xl = df[feats_lin].to_numpy(float)
    boot = [bootstrap(VARIANTE_PRINCIPAL, "Ridge", Xl, y, R, B, b, alfa_fixo) for b in (60, 30, 90)]
    boot.append(bootstrap("Ridge_alpha_reescolhido", "Ridge", Xl, y, R, B, 60))
    boot.append(bootstrap("floresta_hiperparametros_fixos", "floresta_aleatoria", df[feats_arv].to_numpy(float),
                          y, R, B_rf, 60, rf_fixo))
    p_r = prev[prev.modelo == "Ridge"].sort_values("date")
    R_oos = df.set_index("date").loc[p_r.date, [f"ret_fut_{h}d" for h in HS]].to_numpy(float)
    boot.append(bootstrap_so_segundo_nivel(p_r.previsao.to_numpy(), R_oos, B, 60))
    boot = pd.concat(boot)
    boot.to_csv(os.path.join(OUT, "bootstrap_betas_T10.csv"), index=False)
    rb = resumo_bootstrap(boot, pr)
    rb.to_csv(os.path.join(OUT, "bootstrap_resumo_T10.csv"), index=False)
    figura(pr, rb, os.path.join(OUT, "fig_T10_betas.png"))

    pd.set_option("display.width", 230)
    f = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== 1. Principal (987 datas; HAC h+1) ===\n" + pr.to_string(index=False, float_format=f))
    print("\n=== Bootstrap (resumo) ===\n" + rb.to_string(index=False, float_format=f))
    print("\n=== 2. Componentes (vh_30d + iv_30d) ===\n" + co.to_string(index=False, float_format=f))
    print("\n=== 3. Regime ===\n" + rg.to_string(index=False, float_format=f))
    print("\n=== 4. Publicado: maxlags h vs. h+1 (N = 1.806) ===\n" + pu.to_string(index=False, float_format=f))
    print(f"\nLimite de Bonferroni (6 horizontes): {BONFERRONI:.4f}. Saídas em {OUT}")


if __name__ == "__main__":
    main()
