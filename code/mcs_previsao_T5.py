"""
mcs_previsao_T5.py — model confidence set das previsões do BVRP (M2, seção 25 das pendências)

Conjunto inicial: as 8 previsões de data/previsoes_bvrp_T5.csv (MQO, Ridge, LASSO, floresta
aleatória, XGBoost, média histórica, persistência viável e proxy retrospectiva), em cada
janela (expansiva e móvel). Perda quadrática, com o alvo de data/ml_dataset_T4.csv.
Bootstrap em blocos móveis (mcs_utils, a mesma função do T10): bloco de 60 (principal), 30
e 90 (sensibilidade); B = 9.999; semente default_rng([20261001, bloco, r]). Estatísticas
T_max (principal) e T_R (sensibilidade); conjuntos a 10% (principal) e 25%. As duas
estatísticas usam as mesmas reamostragens.

Conferência (sai com erro se não bater): 987 datas por janela, sem NaN, e RMSE de cada
uma das 8 previsões igual ao de outputs/T5/metricas_T5.csv (tolerância 1e-9); um modelo
sem linha em metricas_T5.csv também é erro.

Saídas em outputs/MCS/:
  mcs_T5.csv                  uma linha por janela x bloco x estatística x modelo
  coincidencias_lasso_T5.csv  datas e origens em que o LASSO prevê a média histórica
  comparacao_cap5_MCS.csv     por janela e modelo: R², Clark–West e DM contra a média
                              (como no Cap. 5 atual, com o limite de Bonferroni de 0,01)
                              ao lado do valor-p do MCS (T_max e T_R, bloco de 60)
  execucao_MCS.txt            B, blocos, semente, tempo de cada rodada e versões

Uso:  python code/mcs_previsao_T5.py [--B 9999] [--blocos 60 30 90]
"""
import argparse
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcs_utils as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "MCS")
ALVO = "alvo_bvrp_30d_fut"
MODELOS = ["MQO", "Ridge", "LASSO", "floresta_aleatoria", "XGBoost",
           "media_historica", "persistencia_viavel", "proxy_retrospectiva"]
JANELAS = ["expansiva", "movel"]
ALFAS = [0.10, 0.25]
BLOCO_PRINCIPAL = 60
BONFERRONI = 0.05 / 5
TOL = 1e-9
TOL_IGUAL = 1e-9   # previsões "coincidentes" com tolerância (além da igualdade exata)


def conferir(cond, msg):
    if not cond:
        raise SystemExit(f"CONFERÊNCIA FALHOU: {msg}")


def carregar():
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"]).set_index("date")
    prev = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date"])
    met = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "metricas_T5.csv")).set_index(["janela", "modelo"])
    largo, perdas, origens = {}, {}, {}
    for j in JANELAS:
        pj = prev[prev.janela == j]
        w = pj.pivot(index="date", columns="modelo", values="previsao").sort_index()[MODELOS]
        y = df.loc[w.index, ALVO]
        conferir(len(w) == 987 and not w.isna().any().any() and not y.isna().any(), f"{j}: datas ou NaN")
        L = w.sub(y, axis=0) ** 2
        conferidos = []
        for m in MODELOS:
            conferir((j, m) in met.index, f"{j} {m}: sem linha em metricas_T5.csv")
            conferir(abs(np.sqrt(L[m].mean()) - met.loc[(j, m), "rmse"]) < TOL, f"RMSE {j} {m}")
            conferidos.append(m)
        print(f"RMSE conferido ({j}, {len(conferidos)} de {len(MODELOS)}): {', '.join(conferidos)}")
        largo[j], perdas[j] = w, L
        origens[j] = pj.drop_duplicates("date").set_index("date").origem.reindex(w.index)
    return largo, perdas, origens, met


def coincidencias(largo, origens):
    linhas = []
    for j, w in largo.items():
        dif = (w["LASSO"] - w["media_historica"]).abs()
        exata, tol = dif == 0, dif <= TOL_IGUAL
        por_origem = tol.groupby(origens[j]).all()
        linhas.append({"janela": j, "datas": len(w), "datas_iguais_exato": int(exata.sum()),
                       "datas_iguais_tol": int(tol.sum()), "origens": int(por_origem.size),
                       "origens_iguais_tol": int(por_origem.sum()), "dif_max": float(dif.max())})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--blocos", type=int, nargs="+", default=[60, 30, 90])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    largo, perdas, origens, met = carregar()
    print("Conferência: 987 datas por janela, sem NaN; RMSE das 8 previsões igual ao de outputs/T5/metricas_T5.csv.")

    coin = coincidencias(largo, origens)
    coin.to_csv(os.path.join(OUT, "coincidencias_lasso_T5.csv"), index=False)
    print("\nLASSO = média histórica:")
    print(coin.to_string(index=False))

    resultados, log = [], []
    for j in JANELAS:
        L = perdas[j]
        for b in args.blocos:
            t0 = time.perf_counter()
            mb = M.medias_bootstrap(L, M.sortear_indices(len(L), bloco=b, B=args.B))
            for est in ("max", "R"):
                r = M.mcs(L, medias_boot=mb, estatistica=est)
                for a in ALFAS:
                    r[f"no_mcs_{int(a * 100)}"] = r.p_mcs >= a
                resultados.append(r.assign(janela=j, bloco=b, estatistica=f"T_{est}", B=args.B))
            seg = time.perf_counter() - t0
            log.append(f"{j}, bloco {b}: {seg:.1f} s")
            print(f"  {j}, bloco {b}: {seg:.1f} s")
    res = pd.concat(resultados, ignore_index=True)
    col = ["janela", "bloco", "estatistica", "B", "modelo", "perda_media", "ordem_eliminacao",
           "estatistica_passo", "p_passo", "p_mcs", "no_mcs_10", "no_mcs_25", "grupo_identico"]
    res[col].to_csv(os.path.join(OUT, "mcs_T5.csv"), index=False)

    # Comparação com o Cap. 5 atual (R², CW e DM contra a média; Bonferroni 0,01)
    dm = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "diebold_mariano_T5.csv"))
    comp = []
    for j in JANELAS:
        for m in MODELOS:
            lin = {"janela": j, "modelo": m}
            if m != "media_historica":
                lin["r2_vs_media"] = met.loc[(j, m), "r2_oos_vs_media_historica"]
                lin["cw_p"] = met.loc[(j, m), "cw_p_unilateral_vs_media_historica"]
                d = dm[(dm.janela == j) & (((dm.modelo_1 == m) & (dm.modelo_2 == "media_historica"))
                                           | ((dm.modelo_2 == m) & (dm.modelo_1 == "media_historica")))]
                conferir(len(d) == 1, f"DM {j} {m}")
                lin["dm_p"] = float(d.dm_p_bilateral.iloc[0])
                lin["dm_passa_bonferroni"] = lin["dm_p"] < BONFERRONI
            for est in ("T_max", "T_R"):
                s = res[(res.janela == j) & (res.bloco == BLOCO_PRINCIPAL) & (res.estatistica == est)
                        & (res.modelo == m)].iloc[0]
                lin[f"p_mcs_{est}"], lin[f"no_mcs_10_{est}"] = s.p_mcs, bool(s.no_mcs_10)
            comp.append(lin)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "comparacao_cap5_MCS.csv"), index=False)

    # Resumo no console
    for j in JANELAS:
        print(f"\n=== Janela {j} ===")
        for b in args.blocos:
            for est in ("T_max", "T_R"):
                s = res[(res.janela == j) & (res.bloco == b) & (res.estatistica == est)]
                print(f"  bloco {b:>2}, {est:5}: MCS 10% = {s[s.no_mcs_10].modelo.tolist()}; "
                      f"MCS 25% = {s[s.no_mcs_25].modelo.tolist()}")
        s = res[(res.janela == j) & (res.bloco == BLOCO_PRINCIPAL) & (res.estatistica == "T_max")]
        print(s[["modelo", "perda_media", "ordem_eliminacao", "estatistica_passo", "p_passo", "p_mcs"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for b in args.blocos:
            for est in ("T_max", "T_R"):
                s = res[(res.janela == j) & (res.bloco == b) & (res.estatistica == est)].set_index("modelo")
                for a in ("10", "25"):
                    l_, m_ = s.loc["LASSO", f"no_mcs_{a}"], s.loc["media_historica", f"no_mcs_{a}"]
                    estado = "ficam juntos" if l_ and m_ else "saem juntos" if not (l_ or m_) else \
                        f"separados (LASSO {'dentro' if l_ else 'fora'}, média {'dentro' if m_ else 'fora'})"
                    print(f"  LASSO × média, bloco {b}, {est}, {a}%: {estado}")
    print("\nComparação com o Cap. 5 atual (bloco de 60):")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    with open(os.path.join(OUT, "execucao_MCS.txt"), "w", encoding="utf-8") as f:
        f.write(f"B = {args.B}; blocos = {args.blocos}; semente = default_rng([{M.SEMENTE}, bloco, r])\n")
        f.write(f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}\n")
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
