"""
retorno_bvrp_T11.py — BVRP previsto e retornos futuros, versão não linear (T11)

Leitura A (PRINCIPAL; fórmula da anotação da p. 75 e estrutura D1 fechada na
reunião, 1:19–1:21): modelo único em árvore, em duas etapas,
    ret_fut_h(t) = a + β_h · BVRP_previsto(t) + e,
    BVRP_previsto(t) = f(variáveis em t) − iv_30d(t),
com f = floresta aleatória para rv_30d_fut = RV(t+1 a t+30) = alvo + iv_30d
(forma "rv_menos_iv", principal, fixada antes dos resultados: seção 12 das
pendências). Robustez: forma direta do T5 (a floresta prevê o BVRP) e o Ridge
nas duas formas.
Leitura B (COMPLEMENTO; reunião, 1:17:29 e 1:18:42): floresta direto no
retorno, ret_fut_h(t) = g_h(variáveis em t), por horizonte, sem bootstrap.

Etapas (--etapas; cada uma lê o que a anterior gravou):
  previsao        f para rv_30d_fut: floresta (17 variáveis) e Ridge (16), as 33
                  origens do T5, embargo s <= t − 30, validação cruzada embargada
                  em cada reestimação com as grades estendidas; janela expansiva
                  (principal) e móvel de 730. Métricas do T5 (P.avaliar) com as
                  duas formas lado a lado e a referência "média da RV − IV" (só a
                  IV, sem modelo). Placebo de vazamento no ESPAÇO DA RV: na forma
                  rv_menos_iv, até uma previsão sem informação (constante − IV)
                  carrega a IV, então o R² do BVRP não serve de critério.
                  Robustez (T9): floresta com regime_alta_fixo como variável a mais.
  retorno         987 datas; h = 1, 5, 10, 20, 30, 60; quatro regressores; EP HAC
                  h+1; bootstrap em dois níveis do T10 (hiperparâmetros fixos por
                  origem, bloco de 60, escala coerente, contraprova só do 2º
                  nível), com as MESMAS reamostragens em todos os h e regressores;
                  F1: Wald de β, D e D·β (3 coef.) e de D e D·β (2 coef.), por h.
  conjunto        F2: H0: β_h = 0 para todo h, com a covariância dos β*, na escala
                  coerente; max-|t| do bootstrap; versão HAC empilhada. Também
                  nas réplicas do T10 (outputs/T10/bootstrap_betas_T10.csv).
  quebras         importância (impureza e permutação fora da amostra), dependência
                  parcial em vh_30d e iv_30d (padrão e coerente, com vrp_30d =
                  vh_30d − iv_30d recalculado), limiares das árvores × corte do T9.
  arvore_retorno  leitura B.

Teste F conjunto (F2), na escala coerente do T10: com B* (réplicas × 6
horizontes), média β̄*, covariância Σ* e λ_h = β̄*_h / β̂_h,
    W = β̂' (Λ⁻¹ Σ* Λ⁻¹)⁻¹ β̂ = β̄*' Σ*⁻¹ β̄*  ~  χ²(6),   F = W / 6,
o análogo multivariado do z = β̄*/EP_boot do T10 (definido mesmo com λ_h <= 0).
Na contraprova só do 2º nível, sem 1º estágio, W = β̂' Σ*⁻¹ β̂. max-|t|:
t*_h = (β*_h − β̄*_h)/EP*_h, M = max_h |t*_h|; p ajustado de cada h =
(1 + #{M >= |z_h|}) / (B + 1) -- correção para os 6 horizontes que leva em
conta a correlação entre eles (menos conservadora que Bonferroni).

Uso:  python code/retorno_bvrp_T11.py                     (rodada completa)
      python code/retorno_bvrp_T11.py --rapido --out-dir <dir> --saida-dados <arq>
"""

import argparse
import os
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
from sklearn.metrics import r2_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import previsao_bvrp_T5 as P  # noqa: E402
import retorno_bvrp_T10 as T10  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from hac_utils import mqo_newey_west  # noqa: E402
from split_utils import divisoes_validacao_cruzada, gerar_divisoes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = T10.HS
H_ALVO = 30
BONFERRONI = T10.BONFERRONI
SEMENTE = T10.SEMENTE            # mesma semente do T10: mesmas reamostragens
ALVO_RV = "rv_30d_fut"
CORTE_T9 = 37.3
ETAPAS = ["previsao", "retorno", "conjunto", "quebras", "arvore_retorno"]

# regressor: (modelo, forma). PRINCIPAL fixado antes dos resultados (pendências, seção 12).
REGRESSORES = {"floresta_rv_menos_iv": ("floresta_aleatoria", "rv_menos_iv"),
               "floresta_direta": ("floresta_aleatoria", "direta"),
               "Ridge_rv_menos_iv": ("Ridge", "rv_menos_iv"),
               "Ridge_direto": ("Ridge", "direta")}
PRINCIPAL = "floresta_rv_menos_iv"
# B = 499 (e não 999) no principal se a validação cruzada escolher max_features = 1
# em pelo menos 1/3 das origens (cada ajuste fica ~4x mais caro).
LIMIAR_MAX_FEATURES_1 = 11
# Critério do placebo (só verificação de vazamento): mediana do R² placebo <= 0,01.
# Com o alvo embaralhado, a floresta colapsa na média e "<= 0" falha por acaso
# (pendências, seção 12).
TOL_PLACEBO = 0.01
VOLS = ["vh_30d", "vh_60d", "vh_90d", "vrp_30d"]   # painéis dos limiares (1:13:02: qual volatilidade importa)

AZUL, LARANJA, CINZA = "#2b6cb0", "#d95f02", "#9aa5b1"   # par validado (skill dataviz, T1/T10)
NOMES = {"floresta_rv_menos_iv": "Floresta, f(variáveis) − IV", "floresta_direta": "Floresta, direta",
         "Ridge_rv_menos_iv": "Ridge, f(variáveis) − IV", "Ridge_direto": "Ridge, direto",
         "rv_menos_iv": "f(variáveis) − IV", "direta": "direta"}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    """df do T10 (T4 + ret_fut_h + regimes do T9) com rv_30d_fut; variáveis do T5."""
    df, vt, prev_t5, hp_t5 = T10.carregar()
    df[ALVO_RV] = df[P.ALVO] + df["iv_30d"]     # = vh_30d(t+30) (test_T11, teste 1)
    cols_t4 = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), nrows=1).columns
    feats_arv = [c for c in cols_t4 if c not in ("date", P.ALVO)]   # 17
    feats_lin = [c for c in feats_arv if c != "vh_30d"]            # 16
    return df, vt, prev_t5, hp_t5, feats_arv, feats_lin


def grades_de(rapido):
    g = {"Ridge": P.GRADE_RIDGE, "floresta_aleatoria": P.GRADE_RF}
    return {k: v[:2] for k, v in g.items()} if rapido else g


def chave(d):
    return str(d)[:10]


# ---------------------------------------------------------------------------
# Etapa 1: previsão de rv_30d_fut (forma f(variáveis) − IV)
# ---------------------------------------------------------------------------
def prever(df, feats, nome, janela, grade, *, alvo=ALVO_RV, hiper_fixos=None, placebo=0, n_jobs=-1,
           verboso=True):
    """
    Previsões de `alvo` nas origens do split_utils (h = 30). Na forma
    rv_menos_iv (alvo = rv_30d_fut), previsao = previsao_alvo − iv_30d(t).
    hiper_fixos: {origem AAAA-MM-DD: params} ou None (validação cruzada
    embargada em cada reestimação, P.escolher). placebo = nº da permutação
    (alvo embaralhado dentro do treino de cada origem).
    """
    datas = df["date"].to_numpy()
    X = df[feats].to_numpy(float)
    y_real = df[alvo].to_numpy(float)
    menos_iv = df["iv_30d"].to_numpy(float) if alvo == ALVO_RV else np.zeros(len(df))
    bvrp = df[P.ALVO].to_numpy(float)
    linhas, hiper = [], []
    origens = gerar_divisoes(len(df), h=H_ALVO, janela=janela).origens
    for k, o in enumerate(origens):
        t0 = time.perf_counter()
        y = y_real.copy()
        if placebo:
            y[o.treino] = np.random.default_rng([P.SEMENTE, int(placebo), o.t]).permutation(y_real[o.treino])
        if hiper_fixos is not None:
            p, mse_cv = hiper_fixos[chave(datas[o.t])], np.nan
        else:
            p, mse_cv = P.escolher(nome, grade, X, y, o.treino)
        f, _ = P.ajustar(nome, p, X[o.treino], y[o.treino], n_jobs=n_jobs)
        pa = f(X[o.teste])
        ptr = f(X[o.treino])
        hiper.append({"origem": datas[o.t], "janela": janela, "modelo": nome, "n_treino": len(o.treino),
                      "treino_fim": datas[o.treino[-1]], "params": repr(p), "params_obj": p, "mse_cv": mse_cv,
                      "r2_dentro_alvo": r2_score(y[o.treino], ptr),
                      "r2_dentro_amostra": r2_score(bvrp[o.treino], ptr - menos_iv[o.treino])})
        for u, v in zip(o.teste, pa):
            linhas.append({"date": datas[u], "origem": datas[o.t], "janela": janela, "modelo": nome,
                           "previsao_alvo": float(v), "previsao": float(v - menos_iv[u]),
                           "media_alvo_treino": float(y[o.treino].mean())})
        if verboso:
            print(f"  [{nome}, {janela}{f', placebo {placebo}' if placebo else ''}] origem {k + 1}/{len(origens)} "
                  f"{chave(datas[o.t])}: {p}, {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(hiper)


def placebo_rv(b, nome, df, feats, grade, hiper_fixos, n_jobs):
    """Uma permutação: R² fora da amostra NO ESPAÇO DA RV contra a média histórica da RV."""
    pp, _ = prever(df, feats, nome, "expansiva", grade, hiper_fixos=hiper_fixos, placebo=b, n_jobs=n_jobs,
                   verboso=False)
    rv = df.set_index("date")[ALVO_RV].loc[pp.date]
    return {"permutacao": b, "modelo": nome,
            "r2_oos_rv_vs_media_rv": r2_fora_da_amostra(rv, pp.previsao_alvo, pp.media_alvo_treino)}


def etapa_previsao(df, prev_t5_todas, hp_t5_todas, feats_arv, feats_lin, args, out, saida):
    grades = grades_de(args.rapido)
    prevs, hipers = [], []
    for janela in ("expansiva", "movel"):
        for nome, feats in (("floresta_aleatoria", feats_arv), ("Ridge", feats_lin)):
            p, h = prever(df, feats, nome, janela, grades[nome])
            prevs.append(p.assign(rotulo=nome))
            hipers.append(h.assign(rotulo=nome))
    # robustez da seção 10: a dummy do T9 como variável a mais (só floresta, expansiva)
    p, h = prever(df, feats_arv + ["regime_alta_fixo"], "floresta_aleatoria", "expansiva", grades["floresta_aleatoria"])
    prevs.append(p.assign(rotulo="floresta_aleatoria_com_regime"))
    hipers.append(h.assign(rotulo="floresta_aleatoria_com_regime"))
    prev, hiper = pd.concat(prevs), pd.concat(hipers)

    os.makedirs(os.path.dirname(saida), exist_ok=True)
    prev[["date", "origem", "janela", "rotulo", "modelo", "previsao_alvo", "previsao"]].to_csv(saida, index=False)
    hiper.drop(columns="params_obj").to_csv(os.path.join(out, "hiperparametros_T11.csv"), index=False)
    lim = pd.concat([P.limites_da_grade(g).assign(rotulo=r) for r, g in hiper.groupby("rotulo")])
    lim.to_csv(os.path.join(out, "limites_grade_T11.csv"), index=False)

    # Métricas do T5, com as duas formas lado a lado
    t5 = prev_t5_todas[prev_t5_todas.modelo.isin(P.REFERENCIAS + ["floresta_aleatoria", "Ridge"])]
    t5 = t5.assign(modelo=t5.modelo.replace({"floresta_aleatoria": "floresta_direta", "Ridge": "Ridge_direto"}))
    novos = prev.assign(modelo=prev.rotulo.map({"floresta_aleatoria": "floresta_rv_menos_iv",
                                                "Ridge": "Ridge_rv_menos_iv",
                                                "floresta_aleatoria_com_regime": "floresta_rv_menos_iv_com_regime"}))
    # referência adicional: média histórica da RV − IV (só a IV, sem modelo)
    so_iv = prev[prev.rotulo == "Ridge"].assign(
        modelo="media_rv_menos_iv",
        previsao=lambda d: d.media_alvo_treino - df.set_index("date").loc[d.date, "iv_30d"].to_numpy())
    aval = pd.concat([t5[["date", "janela", "modelo", "previsao"]],
                      novos[["date", "janela", "modelo", "previsao"]], so_iv[["date", "janela", "modelo", "previsao"]]])
    hp_t5 = hp_t5_todas[hp_t5_todas.modelo.isin(["floresta_aleatoria", "Ridge"])]
    hp_t5 = hp_t5.assign(modelo=hp_t5.modelo.replace({"floresta_aleatoria": "floresta_direta", "Ridge": "Ridge_direto"}))
    hp_novos = hiper.assign(modelo=hiper.rotulo.map({"floresta_aleatoria": "floresta_rv_menos_iv",
                                                     "Ridge": "Ridge_rv_menos_iv",
                                                     "floresta_aleatoria_com_regime": "floresta_rv_menos_iv_com_regime"}))
    met, dm, _ = P.avaliar(aval, pd.concat([hp_t5, hp_novos])[["janela", "modelo", "r2_dentro_amostra"]], df,
                           "media_historica")
    met.to_csv(os.path.join(out, "metricas_previsao_T11.csv"), index=False)
    dm.to_csv(os.path.join(out, "diebold_mariano_T11.csv"), index=False)

    # Placebo de vazamento (expansiva), no espaço da RV; critério: mediana <= TOL_PLACEBO
    fixos = {chave(r.origem): r.params_obj for r in hiper[(hiper.janela == "expansiva")
                                                          & (hiper.rotulo == "floresta_aleatoria")].itertuples()}
    t0 = time.perf_counter()
    linhas = [placebo_rv(b, "floresta_aleatoria", df, feats_arv, None, fixos, -1)
              for b in range(1, args.permutacoes_arvores + 1)]
    linhas += Parallel(n_jobs=-1)(delayed(placebo_rv)(b, "Ridge", df, feats_lin, grades["Ridge"], None, 1)
                                  for b in range(1, args.permutacoes_lineares + 1))
    print(f"  placebo: {time.perf_counter() - t0:.0f}s", flush=True)
    pl = pd.DataFrame(linhas)
    pl.to_csv(os.path.join(out, "placebo_permutacoes_T11.csv"), index=False)
    res = pl.groupby("modelo")["r2_oos_rv_vs_media_rv"].agg(["count", "median", "mean", "min", "max"])
    res.columns = ["n_permutacoes"] + [f"r2_placebo_{c}" for c in ["mediana", "media", "min", "max"]]
    res = res.reset_index().assign(passa=lambda d: d.r2_placebo_mediana <= TOL_PLACEBO)
    res.to_csv(os.path.join(out, "placebo_T11.csv"), index=False)

    figura_previsoes(aval[aval.janela == "expansiva"], df, os.path.join(out, "fig_T11_previsoes.png"))
    cols = ["janela", "modelo", "r2_oos_vs_media_historica", "r2_oos_vs_persistencia_viavel", "rmse", "mae",
            "cw_p_unilateral_vs_media_historica", "r2_dentro_amostra_media", "pct_negativas"]
    print("\n=== Previsão do BVRP: forma direta × f(variáveis) − IV ===\n"
          + met[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Placebo (espaço da RV) ===\n" + res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def figura_previsoes(aval, df, caminho):
    w = aval.pivot(index="date", columns="modelo", values="previsao").sort_index()
    y = df.set_index("date")[P.ALVO].loc[w.index]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, m in zip(axes, ["floresta_rv_menos_iv", "floresta_direta"]):
        r2 = r2_fora_da_amostra(y, w[m], w["media_historica"])
        ax.plot(w.index, y, color=AZUL, linewidth=1.2, label="BVRP prospectivo realizado")
        ax.plot(w.index, w[m], color=LARANJA, linewidth=1.0, linestyle="--",
                label=f"{NOMES[m]} (R² fora da amostra vs. média = {f'{r2:.3f}'.replace('.', ',')})")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylabel("p.p.")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Etapa 2: retorno sobre o BVRP previsto
# ---------------------------------------------------------------------------
def regressores(prev_t11, prev_t5):
    """BVRP previsto (janela expansiva, 987 datas) de cada regressor."""
    e = prev_t11[prev_t11.janela == "expansiva"]
    fontes = {"floresta_rv_menos_iv": e[e.rotulo == "floresta_aleatoria"],
              "Ridge_rv_menos_iv": e[e.rotulo == "Ridge"],
              "floresta_direta": prev_t5[prev_t5.modelo == "floresta_aleatoria"],
              "Ridge_direto": prev_t5[prev_t5.modelo == "Ridge"]}
    return {k: v.set_index("date")["previsao"].sort_index().rename("bvrp_previsto") for k, v in fontes.items()}


def principal(df, regs):
    linhas = []
    for reg, p in regs.items():
        d = df.set_index("date").loc[p.index]
        for h in HS:
            res = mqo_newey_west(d[f"ret_fut_{h}d"], p, h=h)
            linhas.append({"regressor": reg, "h": h, "n": int(res.nobs), "beta": res.params["bvrp_previsto"],
                           "ep_hac": res.bse["bvrp_previsto"], "t_hac": res.tvalues["bvrp_previsto"],
                           "p_hac": res.pvalues["bvrp_previsto"], "r2": res.rsquared,
                           "sobrevive_bonferroni_hac": bool(res.pvalues["bvrp_previsto"] < BONFERRONI)})
    return pd.DataFrame(linhas)


def regime_f1(df, regs):
    """F1 (anotação da p. 82): ret ~ BVRP previsto + D + D·BVRP previsto; Wald de 3 e de 2 coeficientes (HAC h+1)."""
    linhas = []
    for reg, p in regs.items():
        d = df.set_index("date").loc[p.index]
        for corte in ["fixo", "expansivo"]:
            X = pd.DataFrame({"bvrp_previsto": p, "regime_alta": d[f"regime_alta_{corte}"]})
            X["interacao"] = X.bvrp_previsto * X.regime_alta
            for h in HS:
                res = mqo_newey_west(d[f"ret_fut_{h}d"], X, h=h)
                w3 = res.wald_test("bvrp_previsto = 0, regime_alta = 0, interacao = 0", use_f=False, scalar=True)
                w2 = res.wald_test("regime_alta = 0, interacao = 0", use_f=False, scalar=True)
                linhas.append({"regressor": reg, "corte": corte, "h": h, "n": int(res.nobs),
                               **{f"b_{c}": res.params[c] for c in X.columns},
                               **{f"p_{c}": res.pvalues[c] for c in X.columns},
                               "qui2_3coef": float(w3.statistic), "p_3coef": float(w3.pvalue),
                               "qui2_2coef": float(w2.statistic), "p_2coef": float(w2.pvalue),
                               "r2": res.rsquared})
    return pd.DataFrame(linhas)


def uma_reamostragem(r, modelo, X, y, R, origens, oos, b, params_fixos, iv_oos):
    """Como T10.uma_reamostragem; na forma rv_menos_iv, subtrai a IV REAL das datas de teste."""
    treinos, j = T10.sortear_indices(r, b, origens, len(oos))
    prev = T10.primeiro_estagio(modelo, X, y, origens, params_fixos, treinos)
    if iv_oos is not None:
        prev = prev - iv_oos
    return T10.betas(prev[j], R[oos][j])


def bootstrap(reg, modelo, X, y, R, B, b, params_fixos, iv=None):
    origens = gerar_divisoes(len(y), h=H_ALVO, janela="expansiva").origens
    oos = np.concatenate([o.teste for o in origens])
    iv_oos = None if iv is None else iv[oos]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=-1)(delayed(uma_reamostragem)(r, modelo, X, y, R, origens, oos, b, params_fixos, iv_oos)
                              for r in range(B))
    print(f"  bootstrap {reg}: B = {B}, bloco = {b}, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(np.array(res), columns=HS).assign(regressor=reg, variante="dois_niveis", bloco=b, r=np.arange(B))


def bootstrap_so_segundo_nivel(reg, prev_fixa, R_oos, B, b):
    """Contraprova (mesmas sementes de T10.bootstrap_so_segundo_nivel)."""
    res = [T10.betas(prev_fixa[j], R_oos[j])
           for j in (T10.indices_blocos(len(prev_fixa), b, np.random.default_rng([SEMENTE, b, r, 2])) for r in range(B))]
    return pd.DataFrame(np.array(res), columns=HS).assign(regressor=reg, variante="so_segundo_nivel", bloco=b,
                                                          r=np.arange(B))


def resumo_bootstrap(boot, pr):
    """Como T10.resumo_bootstrap, por regressor (escala coerente: EP de β̂ = EP_boot/λ)."""
    so2 = {reg: g[HS].std(ddof=1) for reg, g in
           boot[(boot.variante == "so_segundo_nivel") & (boot.bloco == 60)].groupby("regressor")}
    linhas = []
    for (reg, var, b), g in boot.groupby(["regressor", "variante", "bloco"]):
        for h in HS:
            est = pr[(pr.regressor == reg) & (pr.h == h)].iloc[0]
            bh = g[h].to_numpy()
            ep, media = float(bh.std(ddof=1)), float(bh.mean())
            lam = media / est.beta
            ep_c = ep if var == "so_segundo_nivel" else (ep / lam if lam > 0 else np.nan)
            z = est.beta / ep_c
            p = float(2 * stats.norm.sf(abs(z)))
            q025, q975 = np.percentile(bh, [2.5, 97.5])
            s2 = float(so2[reg][h]) if reg in so2 else np.nan
            linhas.append({"regressor": reg, "variante": var, "bloco": b, "B": len(bh), "h": h, "beta": est.beta,
                           "ep_hac": est.ep_hac, "p_hac": est.p_hac,
                           "media_boot": media, "vies": media - est.beta, "razao_media_beta": lam,
                           "ep_boot": ep, "ep_corrigido": ep_c, "ep_so_segundo_nivel": s2,
                           "razao_ep_boot_so2": ep / s2, "razao_ep_corrigido_so2": ep_c / s2,
                           "razao_ep_corrigido_hac": ep_c / est.ep_hac, "z_boot": z, "p_boot": p,
                           "ic_inf": est.beta - 1.96 * ep_c, "ic_sup": est.beta + 1.96 * ep_c,
                           "ic_basico_inf": 2 * est.beta - q975, "ic_basico_sup": 2 * est.beta - q025,
                           "sobrevive_bonferroni_boot": bool(p < BONFERRONI)})
    return pd.DataFrame(linhas)


def params_fixos(hiper, rotulo, hp_t5, modelo, forma):
    if forma == "rv_menos_iv":
        g = hiper[(hiper.janela == "expansiva") & (hiper.rotulo == rotulo)].sort_values("origem")
    else:
        g = hp_t5[hp_t5.modelo == modelo].sort_values("origem")
    return [T10._params(s) for s in g.params]


def etapa_retorno(df, prev_t5, hp_t5, feats_arv, feats_lin, args, out, saida):
    prev_t11 = pd.read_csv(saida, parse_dates=["date", "origem"])
    hiper = pd.read_csv(os.path.join(out, "hiperparametros_T11.csv"), parse_dates=["origem"])
    regs = regressores(prev_t11, prev_t5)
    pr, f1 = principal(df, regs), regime_f1(df, regs)
    pr.to_csv(os.path.join(out, "principal_T11.csv"), index=False)
    f1.to_csv(os.path.join(out, "f1_regime_T11.csv"), index=False)

    R = df[[f"ret_fut_{h}d" for h in HS]].to_numpy(float)
    iv = df["iv_30d"].to_numpy(float)
    B, B_sens = (8, 8) if args.rapido else (args.B, args.B_sensibilidade)
    boot = []
    for reg, (modelo, forma) in REGRESSORES.items():
        fixos = params_fixos(hiper, modelo, hp_t5, modelo, forma)
        X = df[feats_arv if modelo == "floresta_aleatoria" else feats_lin].to_numpy(float)
        y = df[ALVO_RV if forma == "rv_menos_iv" else P.ALVO].to_numpy(float)
        iv_ = iv if forma == "rv_menos_iv" else None
        b_reg = B
        if reg == PRINCIPAL:
            n_mf1 = sum(np.isclose(p["max_features"], 1.0) for p in fixos)
            if n_mf1 >= LIMIAR_MAX_FEATURES_1 and not args.rapido:
                b_reg = 499
            print(f"  {reg}: max_features = 1 em {n_mf1}/{len(fixos)} origens -> B = {b_reg}", flush=True)
        boot.append(bootstrap(reg, modelo, X, y, R, b_reg, 60, fixos, iv_))
        if reg == PRINCIPAL:
            boot += [bootstrap(reg, modelo, X, y, R, B_sens, b, fixos, iv_) for b in (30, 90)]
        p = regs[reg]
        R_oos = df.set_index("date").loc[p.index, [f"ret_fut_{h}d" for h in HS]].to_numpy(float)
        boot.append(bootstrap_so_segundo_nivel(reg, p.to_numpy(), R_oos, B, 60))
    boot = pd.concat(boot)
    boot.to_csv(os.path.join(out, "bootstrap_betas_T11.csv"), index=False)
    rb = resumo_bootstrap(boot, pr)
    rb.to_csv(os.path.join(out, "bootstrap_resumo_T11.csv"), index=False)
    figura_betas(pr, rb, os.path.join(out, "fig_T11_betas.png"))

    pd.set_option("display.width", 230)
    f = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== Principal (987 datas; HAC h+1) ===\n" + pr.to_string(index=False, float_format=f))
    cols = ["regressor", "variante", "bloco", "B", "h", "beta", "razao_media_beta", "ep_corrigido", "p_boot",
            "ic_inf", "ic_sup", "sobrevive_bonferroni_boot"]
    print("\n=== Bootstrap (resumo) ===\n" + rb[cols].to_string(index=False, float_format=f))
    print("\n=== F1: Wald de 3 e de 2 coeficientes ===\n"
          + f1[["regressor", "corte", "h", "qui2_3coef", "p_3coef", "qui2_2coef", "p_2coef"]].to_string(
              index=False, float_format=f))


def figura_betas(pr, rb, caminho):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(HS))
    for ax, reg in zip(axes, [PRINCIPAL, "floresta_direta"]):
        a = pr[pr.regressor == reg].set_index("h")
        b = rb[(rb.regressor == reg) & (rb.variante == "dois_niveis") & (rb.bloco == 60)].set_index("h")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.errorbar(x - 0.12, a.beta, yerr=1.96 * a.ep_hac, fmt="o", color=AZUL, capsize=4,
                    label=r"$\hat\beta$ ± 1,96 EP HAC (h+1) — ignora o 1º estágio")
        ax.errorbar(x + 0.12, b.beta, yerr=1.96 * b.ep_corrigido, fmt="s", color=LARANJA, capsize=4,
                    label=r"$\hat\beta$ ± 1,96 EP bootstrap em dois níveis (EP / $\lambda$)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}" for h in HS])
        ax.set_xlabel("Horizonte h (dias)")
        ax.set_title(NOMES[reg], fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("β: retorno futuro sobre o BVRP previsto")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Etapa 3: teste conjunto nos 6 horizontes (F2)
# ---------------------------------------------------------------------------
def wald_conjunto(Bstar, beta_hat, *, coerente=True):
    """
    F2 com a covariância dos β* (mesmas reamostragens em todos os h).
    coerente=True (dois níveis): W = β̄*' Σ*⁻¹ β̄* (= β̂' (Λ⁻¹Σ*Λ⁻¹)⁻¹ β̂);
    coerente=False (só 2º nível): W = β̂' Σ*⁻¹ β̂. Também o max-|t| do bootstrap.
    """
    Bstar = np.asarray(Bstar, float)
    beta_hat = np.asarray(beta_hat, float)
    if Bstar.shape[0] <= Bstar.shape[1]:
        raise ValueError(f"teste conjunto: B = {Bstar.shape[0]} réplicas não identifica a covariância de "
                         f"{Bstar.shape[1]} horizontes (precisa de B > {Bstar.shape[1]})")
    m = Bstar.mean(axis=0)
    S = np.atleast_2d(np.cov(Bstar, rowvar=False, ddof=1))
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    centro = m if coerente else beta_hat
    k = len(m)
    W = float(centro @ np.linalg.solve(S, centro))
    z = centro / sd
    M = np.abs((Bstar - m) / sd).max(axis=1)
    p_aj = np.array([(1 + np.sum(M >= abs(zh))) / (len(M) + 1) for zh in z])
    glob = {"B": len(Bstar), "W": W, "gl": k, "p_wald": float(stats.chi2.sf(W, k)), "F": W / k,
            "num_condicao_corr": float(np.linalg.cond(corr)), "autovalor_min_corr": float(np.linalg.eigvalsh(corr)[0]),
            "max_abs_z": float(np.abs(z).max()), "crit_max_t_95": float(np.quantile(M, 0.95)),
            "p_max_t": float((1 + np.sum(M >= np.abs(z).max())) / (len(M) + 1))}
    por_h = pd.DataFrame({"h": HS[:k], "beta": beta_hat, "media_boot": m, "lambda": m / beta_hat, "ep_boot": sd,
                          "z": z, "p_normal": 2 * stats.norm.sf(np.abs(z)),
                          "p_bonferroni": np.minimum(1, k * 2 * stats.norm.sf(np.abs(z))), "p_max_t": p_aj})
    return glob, por_h, corr


def wald_hac_empilhado(x, R, maxlags):
    """
    F2 sem o 1º estágio: k regressões ret_h = a_h + b_h x empilhadas, covariância
    de Newey–West (Bartlett, sem correção de amostra pequena, como hac_utils)
    dos escores conjuntos, com maxlags comum; W = b' V_bb⁻¹ b ~ χ²(k).
    """
    x = np.asarray(x, float)
    R = np.asarray(R, float).reshape(len(x), -1)
    n, k = R.shape
    Z = np.column_stack([np.ones(n), x])
    ZZi = np.linalg.inv(Z.T @ Z)
    theta = ZZi @ Z.T @ R                                   # 2 x k
    U = R - Z @ theta
    s = np.hstack([Z * U[:, [i]] for i in range(k)])        # n x 2k
    S = s.T @ s
    for lag in range(1, maxlags + 1):
        G = s[lag:].T @ s[:-lag]
        S += (1 - lag / (maxlags + 1)) * (G + G.T)
    pao = np.kron(np.eye(k), ZZi)
    V = pao @ S @ pao
    idx = np.arange(1, 2 * k, 2)
    b, Vb = theta[1], V[np.ix_(idx, idx)]
    W = float(b @ np.linalg.solve(Vb, b))
    return {"W": W, "gl": k, "p_wald": float(stats.chi2.sf(W, k)), "F": W / k, "maxlags": maxlags,
            "ep": np.sqrt(np.diag(Vb))}


def _correlacao_longa(corr, **chaves):
    return pd.DataFrame([{**chaves, "h_1": a, "h_2": b, "corr": corr[i, j]}
                         for i, a in enumerate(HS) for j, b in enumerate(HS)])


def conjunto_de(boot, betas_est, rotulo_col):
    """Aplica wald_conjunto a cada (regressor/variante, bloco)."""
    glob, porh, corrs = [], [], []
    for chaves, g in boot.groupby(rotulo_col):
        chaves = dict(zip(rotulo_col, chaves))
        coerente = "so_segundo_nivel" not in str(chaves.get("variante", ""))
        bh = betas_est(chaves)
        G, ph, c = wald_conjunto(g[HS].to_numpy(), bh, coerente=coerente)
        glob.append({**chaves, "escala": "coerente" if coerente else "sem 1º estágio", **G})
        porh.append(ph.assign(**chaves))
        corrs.append(_correlacao_longa(c, **chaves))
    return pd.DataFrame(glob), pd.concat(porh), pd.concat(corrs)


def etapa_conjunto(df, prev_t5, args, out, saida):
    # T11
    boot = pd.read_csv(os.path.join(out, "bootstrap_betas_T11.csv")).rename(columns={str(h): h for h in HS})
    pr = pd.read_csv(os.path.join(out, "principal_T11.csv"))
    est = lambda c: pr[pr.regressor == c["regressor"]].sort_values("h").beta.to_numpy()  # noqa: E731
    g, ph, c = conjunto_de(boot, est, ["regressor", "variante", "bloco"])
    prev_t11 = pd.read_csv(saida, parse_dates=["date", "origem"])
    hac = []
    for reg, p in regressores(prev_t11, prev_t5).items():
        R = df.set_index("date").loc[p.index, [f"ret_fut_{h}d" for h in HS]].to_numpy(float)
        w = wald_hac_empilhado(p.to_numpy(), R, max(HS) + 1)
        hac.append({"regressor": reg, **{k: v for k, v in w.items() if k != "ep"}})
    g.to_csv(os.path.join(out, "teste_conjunto_T11.csv"), index=False)
    ph.to_csv(os.path.join(out, "max_t_por_horizonte_T11.csv"), index=False)
    c.to_csv(os.path.join(out, "correlacao_betas_T11.csv"), index=False)
    pd.DataFrame(hac).to_csv(os.path.join(out, "teste_conjunto_hac_T11.csv"), index=False)

    # T10 (réplicas gravadas, sem recalcular)
    b10 = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "bootstrap_betas_T10.csv")).rename(
        columns={str(h): h for h in HS})
    p10 = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "principal_T10.csv"))
    est10 = lambda c: p10[p10.modelo == ("floresta_aleatoria" if c["variante"].startswith("floresta")  # noqa: E731
                                         else "Ridge")].sort_values("h").beta.to_numpy()
    g10, ph10, c10 = conjunto_de(b10, est10, ["variante", "bloco"])
    g10.to_csv(os.path.join(out, "teste_conjunto_T10.csv"), index=False)
    ph10.to_csv(os.path.join(out, "max_t_por_horizonte_T10.csv"), index=False)
    c10.to_csv(os.path.join(out, "correlacao_betas_T10.csv"), index=False)

    f = lambda v: f"{v:.4g}"  # noqa: E731
    cols = ["escala", "B", "W", "p_wald", "F", "num_condicao_corr", "p_max_t", "crit_max_t_95"]
    print("\n=== F2 (T11) ===\n" + g[["regressor", "variante", "bloco"] + cols].to_string(index=False, float_format=f))
    print("\n=== F2, versão HAC empilhada (T11; maxlags = 61) ===\n" + pd.DataFrame(hac).to_string(index=False, float_format=f))
    print("\n=== F2 (réplicas do T10) ===\n" + g10[["variante", "bloco"] + cols].to_string(index=False, float_format=f))


# ---------------------------------------------------------------------------
# Etapa 4: como a árvore escolhe as quebras
# ---------------------------------------------------------------------------
def ganhos_por_no(arvore):
    """(variável, limiar, ganho) dos nós internos; ganho = redução ponderada da impureza / n da raiz."""
    t = arvore.tree_
    interno = t.children_left != -1
    wn, imp = t.weighted_n_node_samples, t.impurity
    esq, dir_ = t.children_left[interno], t.children_right[interno]
    ganho = (wn[interno] * imp[interno] - wn[esq] * imp[esq] - wn[dir_] * imp[dir_]) / wn[0]
    return t.feature[interno], t.threshold[interno], ganho


def prever_bvrp(m, X, forma, j_iv):
    return m.predict(X) - X[:, j_iv] if forma == "rv_menos_iv" else m.predict(X)


def etapa_quebras(df, prev_t5, hp_t5, feats_arv, args, out, saida):
    hiper = pd.read_csv(os.path.join(out, "hiperparametros_T11.csv"), parse_dates=["origem"])
    prev_t11 = pd.read_csv(saida, parse_dates=["date", "origem"])
    regs = regressores(prev_t11, prev_t5)
    origens = gerar_divisoes(len(df), h=H_ALVO, janela="expansiva").origens
    X = df[feats_arv].to_numpy(float)
    bvrp = df[P.ALVO].to_numpy(float)
    j_vh, j_iv, j_vrp = (feats_arv.index(c) for c in ("vh_30d", "iv_30d", "vrp_30d"))
    n_rep, n_grade, n_fundo = (2, 10, 50) if args.rapido else (10, 40, 200)
    grades = {c: np.quantile(df[c], np.linspace(0.02, 0.98, n_grade)) for c in ("vh_30d", "iv_30d")}
    rng = np.random.default_rng(SEMENTE)
    imp, pdp, cortes, pd2 = [], [], [], []
    for forma, reg in (("rv_menos_iv", "floresta_rv_menos_iv"), ("direta", "floresta_direta")):
        y = df[ALVO_RV if forma == "rv_menos_iv" else P.ALVO].to_numpy(float)
        fixos = params_fixos(hiper, "floresta_aleatoria", hp_t5, "floresta_aleatoria", forma)
        modelos, mdi = [], []
        for k, o in enumerate(origens):
            _, m = P.ajustar("floresta_aleatoria", fixos[k], X[o.treino], y[o.treino])
            modelos.append(m)
            mdi.append(m.feature_importances_)
            fs, ts, gs = zip(*(ganhos_por_no(arv) for arv in m.estimators_))
            cortes.append(pd.DataFrame({"forma": forma, "origem": df.date[o.t],
                                        "variavel": np.array(feats_arv)[np.concatenate(fs)],
                                        "limiar": np.concatenate(ts),
                                        "ganho": np.concatenate(gs) / len(m.estimators_)}))
            # dependência parcial: fundo = amostra do treino da origem
            fundo = X[rng.choice(o.treino, size=min(n_fundo, len(o.treino)), replace=False)]
            for var, j in (("vh_30d", j_vh), ("iv_30d", j_iv)):
                for versao in ("padrao", "coerente"):
                    Xg = np.repeat(fundo[None], n_grade, axis=0)          # grade x fundo x variáveis
                    Xg[:, :, j] = grades[var][:, None]
                    if versao == "coerente":
                        Xg[:, :, j_vrp] = Xg[:, :, j_vh] - Xg[:, :, j_iv]
                    v = prever_bvrp(m, Xg.reshape(-1, X.shape[1]), forma, j_iv).reshape(n_grade, -1).mean(axis=1)
                    pdp.append(pd.DataFrame({"forma": forma, "versao": versao, "variavel": var,
                                             "origem": df.date[o.t], "valor": grades[var], "bvrp_previsto": v}))
        # coerência com a etapa 1 / T5
        oos = np.concatenate([o.teste for o in origens])
        prev_oos = np.concatenate([prever_bvrp(m, X[o.teste], forma, j_iv) for m, o in zip(modelos, origens)])
        assert np.allclose(prev_oos, regs[reg].to_numpy(), atol=1e-8), f"{forma}: refeito difere das previsões gravadas"
        # importância por permutação fora da amostra (987 datas; a IV subtraída é sempre a real)
        Xo = X[oos]
        limites = np.cumsum([0] + [len(o.teste) for o in origens])
        iv_real = X[oos, j_iv]

        def prever_oos(Xp):
            pa = np.concatenate([m.predict(Xp[a:b]) for m, a, b in zip(modelos, limites[:-1], limites[1:])])
            return pa - iv_real if forma == "rv_menos_iv" else pa
        base = np.mean((bvrp[oos] - prever_oos(Xo)) ** 2)
        for j, var in enumerate(feats_arv):
            d = []
            for _ in range(n_rep):
                Xp = Xo.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                d.append(np.mean((bvrp[oos] - prever_oos(Xp)) ** 2) - base)
            imp.append({"forma": forma, "variavel": var, "mdi_media": float(np.mean([x[j] for x in mdi])),
                        "mdi_dp": float(np.std([x[j] for x in mdi], ddof=1)),
                        "perm_delta_mse_media": float(np.mean(d)), "perm_delta_mse_dp": float(np.std(d, ddof=1))})
        # dependência parcial 2D na última origem (versão coerente)
        m, o = modelos[-1], origens[-1]
        fundo = X[rng.choice(o.treino, size=min(n_fundo, len(o.treino)), replace=False)]
        g_vh, g_iv = np.meshgrid(grades["vh_30d"], grades["iv_30d"], indexing="ij")
        Xg = np.repeat(fundo[None], g_vh.size, axis=0)
        Xg[:, :, j_vh] = g_vh.ravel()[:, None]
        Xg[:, :, j_iv] = g_iv.ravel()[:, None]
        Xg[:, :, j_vrp] = Xg[:, :, j_vh] - Xg[:, :, j_iv]
        v = prever_bvrp(m, Xg.reshape(-1, X.shape[1]), forma, j_iv).reshape(g_vh.size, -1).mean(axis=1)
        pd2.append(pd.DataFrame({"forma": forma, "origem": df.date[o.t], "vh_30d": g_vh.ravel(),
                                 "iv_30d": g_iv.ravel(), "bvrp_previsto": v}))
        print(f"  quebras ({forma}): ok", flush=True)

    imp, pdp, pd2 = pd.DataFrame(imp), pd.concat(pdp), pd.concat(pd2)
    cortes = pd.concat(cortes)
    # agregados dos limiares: ganho por faixa de 1 p.p. (todas as origens) e resumo de vh_30d por origem
    ag = (cortes.assign(faixa=np.floor(cortes.limiar)).groupby(["forma", "variavel", "faixa"])
          .agg(ganho=("ganho", "sum"), n_cortes=("ganho", "size")).reset_index())
    tot = cortes.groupby(["forma", "origem"]).ganho.sum()
    res_vh = []
    for (forma, origem), g in cortes[cortes.variavel == "vh_30d"].groupby(["forma", "origem"]):
        g = g.sort_values("limiar")
        acum = g.ganho.cumsum() / g.ganho.sum()
        res_vh.append({"forma": forma, "origem": origem, "fracao_ganho_vh_30d": g.ganho.sum() / tot[(forma, origem)],
                       "limiar_mediana_ponderada": float(g.limiar.to_numpy()[np.searchsorted(acum.to_numpy(), 0.5)]),
                       "fracao_ganho_vh_30d_entre_35_e_40": float(g.ganho[g.limiar.between(35, 40)].sum() / g.ganho.sum())})
    res_vh = pd.DataFrame(res_vh)
    imp.to_csv(os.path.join(out, "importancia_T11.csv"), index=False)
    pdp.to_csv(os.path.join(out, "dependencia_parcial_T11.csv"), index=False)
    pd2.to_csv(os.path.join(out, "dependencia_parcial_2d_T11.csv"), index=False)
    ag.to_csv(os.path.join(out, "cortes_T11.csv"), index=False)
    res_vh.to_csv(os.path.join(out, "cortes_vh30d_por_origem_T11.csv"), index=False)
    reg9 = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])
    figuras_quebras(imp, pdp, pd2, ag, reg9, out)
    print("\n=== Importância (média nas origens / permutação fora da amostra) ===\n"
          + imp.sort_values(["forma", "perm_delta_mse_media"], ascending=[True, False]).to_string(
              index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Limiares em vh_30d (por origem; resumo) ===\n"
          + res_vh.groupby("forma").describe().T.to_string(float_format=lambda v: f"{v:.3g}"))


def figuras_quebras(imp, pdp, pd2, ag, reg9, out):
    fm = lambda v: f"{v:g}".replace(".", ",")  # noqa: E731
    formas = ["rv_menos_iv", "direta"]
    # importância
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, forma in enumerate(formas):
        g = imp[imp.forma == forma].sort_values("perm_delta_mse_media")
        for ax, col, tit in ((axes[i, 0], "mdi_media", "impureza (média das 33 origens)"),
                             (axes[i, 1], "perm_delta_mse_media", "permutação fora da amostra (Δ EQM)")):
            ax.barh(g.variavel, g[col], color=AZUL, height=0.6)
            ax.set_title(f"Floresta, {NOMES[forma]}: {tit}", fontsize=9)
            ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_T11_importancia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # dependência parcial
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    ultima = pdp.origem.max()
    for i, forma in enumerate(formas):
        for jj, var in enumerate(["vh_30d", "iv_30d"]):
            ax = axes[i, jj]
            g = pdp[(pdp.forma == forma) & (pdp.variavel == var)]
            for o, gg in g[g.versao == "coerente"].groupby("origem"):
                ax.plot(gg.valor, gg.bvrp_previsto, color=CINZA, linewidth=0.6, alpha=0.6)
            u = g[(g.origem == ultima) & (g.versao == "coerente")]
            ax.plot(u.valor, u.bvrp_previsto, color=AZUL, linewidth=2, label="última origem (coerente)")
            u = g[(g.origem == ultima) & (g.versao == "padrao")]
            ax.plot(u.valor, u.bvrp_previsto, color=LARANJA, linewidth=1.5, linestyle="--",
                    label="última origem (padrão)")
            if var == "vh_30d":
                ax.axvline(CORTE_T9, color="black", linewidth=1, linestyle=":", label=f"corte do T9 ({fm(CORTE_T9)})")
            ax.set_xlabel(f"{var} (% a.a.)")
            ax.set_ylabel("BVRP previsto (p.p.)")
            ax.set_title(f"Floresta, {NOMES[forma]}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_T11_dependencia_parcial.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # dependência parcial 2D
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, forma in zip(axes, formas):
        g = pd2[pd2.forma == forma].pivot(index="iv_30d", columns="vh_30d", values="bvrp_previsto")
        im = ax.pcolormesh(g.columns, g.index, g.to_numpy(), cmap="Blues_r", shading="nearest")
        ax.axvline(CORTE_T9, color="black", linewidth=1, linestyle=":")
        ax.set_xlabel("vh_30d (% a.a.)")
        ax.set_ylabel("iv_30d (% a.a.)")
        ax.set_title(f"Floresta, {NOMES[forma]} (última origem, coerente)", fontsize=9)
        fig.colorbar(im, ax=ax, label="BVRP previsto (p.p.)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_T11_dependencia_parcial_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # limiares escolhidos pelas árvores
    fig, axes = plt.subplots(2, len(VOLS), figsize=(14, 6.5))
    oos = reg9[reg9.date >= pd.Timestamp("2023-06-21")]
    for i, forma in enumerate(formas):
        for jj, var in enumerate(VOLS):
            ax = axes[i, jj]
            g = ag[(ag.forma == forma) & (ag.variavel == var)]
            ax.bar(g.faixa + 0.5, g.ganho, width=0.9, color=AZUL)
            if var == "vh_30d":
                ax.axvline(CORTE_T9, color="black", linewidth=1, linestyle=":", label=f"corte fixo do T9 ({fm(CORTE_T9)})")
                ax.axvspan(oos.corte_expansivo_vigente.min(), oos.corte_expansivo_vigente.max(), color=LARANJA,
                           alpha=0.15, label="faixa do corte expansivo (T9)")
                ax.legend(fontsize=7)
            ax.set_title(f"{NOMES[forma]}: {var}", fontsize=9)
            ax.set_xlabel("limiar (p.p.)")
            ax.grid(True, alpha=0.3)
        axes[i, 0].set_ylabel("redução de impureza (soma nas origens)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_T11_cortes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Etapa 5: leitura B — floresta direto no retorno
# ---------------------------------------------------------------------------
def escolher_h(grade, X, y, idx_treino, h):
    """P.escolher com embargo h nas dobras (P.escolher fixa h = 30)."""
    dobras = divisoes_validacao_cruzada(idx_treino, h=h, n_dobras=P.N_DOBRAS)
    erros = Parallel(n_jobs=-1)(delayed(P._mse_dobra)("floresta_aleatoria", p, X, y, tr, va, 1)
                                for p in grade for tr, va in dobras)
    mses = np.asarray(erros).reshape(len(grade), len(dobras)).mean(axis=1)
    i = int(np.argmin(mses))
    return grade[i], float(mses[i])


def arvore_retorno(df, feats, h, grade, *, hiper_fixos=None, placebo=0, n_jobs=-1):
    """ret_fut_h(t) = g_h(variáveis em t); origens do split_utils com h (treino s <= t − h)."""
    datas = df["date"].to_numpy()
    X = df[feats].to_numpy(float)
    y_real = df[f"ret_fut_{h}d"].to_numpy(float)
    linhas, hiper = [], []
    for o in gerar_divisoes(len(df), h=h, janela="expansiva").origens:
        y = y_real.copy()
        if placebo:
            y[o.treino] = np.random.default_rng([P.SEMENTE, int(placebo), o.t, h]).permutation(y_real[o.treino])
        if hiper_fixos is not None:
            p, mse_cv = hiper_fixos[(h, chave(datas[o.t]))], np.nan
        else:
            p, mse_cv = escolher_h(grade, X, y, o.treino, h)
        f, _ = P.ajustar("floresta_aleatoria", p, X[o.treino], y[o.treino], n_jobs=n_jobs)
        hiper.append({"h": h, "origem": datas[o.t], "modelo": "floresta_aleatoria", "janela": "expansiva",
                      "n_treino": len(o.treino), "treino_fim": datas[o.treino[-1]], "params": repr(p),
                      "params_obj": p, "mse_cv": mse_cv, "r2_dentro_amostra": r2_score(y[o.treino], f(X[o.treino]))})
        media = y[o.treino].mean()
        for u, v in zip(o.teste, f(X[o.teste])):
            linhas.append({"h": h, "date": datas[u], "origem": datas[o.t], "previsao": float(v),
                           "media_historica": float(media), "realizado": y_real[u]})
    return pd.DataFrame(linhas), pd.DataFrame(hiper)


def placebo_retorno(b, h, df, feats, fixos):
    pp, _ = arvore_retorno(df, feats, h, None, hiper_fixos=fixos, placebo=b, n_jobs=1)
    return {"h": h, "permutacao": b,
            "r2_oos_vs_media_historica": r2_fora_da_amostra(pp.realizado, pp.previsao, pp.media_historica)}


def etapa_arvore_retorno(df, feats_arv, args, out):
    grade = grades_de(args.rapido)["floresta_aleatoria"]
    prevs, hipers = [], []
    for h in HS:
        t0 = time.perf_counter()
        p, hp = arvore_retorno(df, feats_arv, h, grade)
        prevs.append(p)
        hipers.append(hp)
        print(f"  leitura B, h = {h}: {time.perf_counter() - t0:.0f}s", flush=True)
    prev, hiper = pd.concat(prevs), pd.concat(hipers)
    prev.to_csv(os.path.join(out, "previsoes_arvore_retorno_T11.csv"), index=False)
    hiper.drop(columns="params_obj").to_csv(os.path.join(out, "hiperparametros_arvore_retorno_T11.csv"), index=False)
    pd.concat([P.limites_da_grade(g).assign(h=h) for h, g in hiper.groupby("h")]).to_csv(
        os.path.join(out, "limites_grade_arvore_retorno_T11.csv"), index=False)
    met = []
    for h, g in prev.groupby("h"):
        cw = clark_west(g.realizado, g.previsao, g.media_historica, h=h)
        dm = diebold_mariano(g.realizado, g.previsao, g.media_historica, h=h)
        met.append({"h": h, "n": len(g), "r2_oos_vs_media_historica":
                    r2_fora_da_amostra(g.realizado, g.previsao, g.media_historica), **cw, **dm,
                    "sobrevive_bonferroni_cw": bool(cw["cw_p_unilateral"] < BONFERRONI),
                    "r2_dentro_amostra_media": float(hiper[hiper.h == h].r2_dentro_amostra.mean())})
    met = pd.DataFrame(met)
    met.to_csv(os.path.join(out, "arvore_retorno_T11.csv"), index=False)
    fixos = {(r.h, chave(r.origem)): r.params_obj for r in hiper.itertuples()}
    n_perm = 2 if args.rapido else args.permutacoes_arvores
    t0 = time.perf_counter()
    pl = pd.DataFrame(Parallel(n_jobs=-1)(delayed(placebo_retorno)(b, h, df, feats_arv, fixos)
                                          for h in HS for b in range(1, n_perm + 1)))
    print(f"  placebo da leitura B: {time.perf_counter() - t0:.0f}s", flush=True)
    pl.to_csv(os.path.join(out, "placebo_permutacoes_arvore_retorno_T11.csv"), index=False)
    res = pl.groupby("h").r2_oos_vs_media_historica.agg(["count", "median", "min", "max"]).reset_index()
    res = res.assign(passa=lambda d: d["median"] <= TOL_PLACEBO)
    res.to_csv(os.path.join(out, "placebo_arvore_retorno_T11.csv"), index=False)
    print("\n=== Leitura B: floresta direto no retorno ===\n" + met.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Placebo da leitura B ===\n" + res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", nargs="+", default=ETAPAS, choices=ETAPAS)
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T11"))
    ap.add_argument("--saida-dados", default=os.path.join("data", "previsoes_bvrp_T11.csv"))
    ap.add_argument("--B", type=int, default=999, help="reamostragens por regressor (bloco de 60)")
    ap.add_argument("--B-sensibilidade", type=int, default=199, help="blocos de 30 e 90 (só o principal)")
    ap.add_argument("--permutacoes-arvores", type=int, default=10)
    ap.add_argument("--permutacoes-lineares", type=int, default=200)
    ap.add_argument("--rapido", action="store_true",
                    help="teste de fumaça: 2 combinações por grade, B = 8, poucas permutações")
    args = ap.parse_args()
    if args.rapido:
        args.permutacoes_arvores, args.permutacoes_lineares = 2, 4
    warnings.filterwarnings("ignore")
    absoluto = (lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p))
    out, saida = absoluto(args.out_dir), absoluto(args.saida_dados)
    os.makedirs(out, exist_ok=True)

    df, vt, prev_t5, hp_t5, feats_arv, feats_lin = carregar()
    t0 = time.perf_counter()
    if "previsao" in args.etapas:
        prev_todas = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
        hp_todas = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "hiperparametros_T5.csv"), parse_dates=["origem"])
        etapa_previsao(df, prev_todas, hp_todas, feats_arv, feats_lin, args, out, saida)
    if "retorno" in args.etapas:
        etapa_retorno(df, prev_t5, hp_t5, feats_arv, feats_lin, args, out, saida)
    if "conjunto" in args.etapas:
        etapa_conjunto(df, prev_t5, args, out, saida)
    if "quebras" in args.etapas:
        etapa_quebras(df, prev_t5, hp_t5, feats_arv, args, out, saida)
    if "arvore_retorno" in args.etapas:
        etapa_arvore_retorno(df, feats_arv, args, out)
    print(f"\nLimite de Bonferroni (6 horizontes): {BONFERRONI:.4f}. Saídas em {out} e {saida} "
          f"({(time.perf_counter() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
