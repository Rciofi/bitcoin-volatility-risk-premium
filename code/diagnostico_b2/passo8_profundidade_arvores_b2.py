"""
passo8_profundidade_arvores_b2.py — Rodada 3, B2, passo 8 (#135): sensibilidade à profundidade das árvores

Floresta aleatória e XGBoost com a grade do T5 restrita a profundidade >= 3, nas 33 origens das duas janelas,
mesma semente e mesmo protocolo (previsao_bvrp_T5.rodar). Floresta: max_depth em {3, 6, None} (45 das 75
combinações); XGBoost: max_depth = 4 (24 das 72; a grade do T5 só tem a profundidade 4 acima de 3).
Sanidade: com a grade completa, a 1ª origem da janela expansiva reproduz as previsões e os hiperparâmetros
gravados do T5. Avaliação: R² fora, CW e DM contra a média histórica, DM contra o modelo original e MCS
(T_max, blocos de 60, B = 9.999) com os oito modelos, trocando os dois originais pelos restritos.
Nada vai para tables/ ou figs/; checkpoint por janela (previsões e hiperparâmetros gravados são pulados).

Uso:  python code/diagnostico_b2/passo8_profundidade_arvores_b2.py [--out PASTA]
      (padrão: outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo8/)
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.stdout.reconfigure(encoding="utf-8")
import mcs_utils  # noqa: E402
import previsao_bvrp_T5 as P  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--out")
_args = _ap.parse_args()
OUT = _args.out or os.path.join(REPO, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo8")
os.makedirs(OUT, exist_ok=True)
print(f"Saídas em {OUT}", flush=True)
warnings.filterwarnings("ignore")
ARV = ["floresta_aleatoria", "XGBoost"]
GRADES = {"floresta_aleatoria": [g for g in P.GRADE_RF if g["max_depth"] is None or g["max_depth"] >= 3],
          "XGBoost": [g for g in P.GRADE_XGB if g["max_depth"] >= 3]}
MCS_BLOCO, MCS_B, TOL = 60, 9999, 1e-9

df = pd.read_csv(os.path.join(REPO, "data", "ml_dataset_T4.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
feats_arv = [c for c in df.columns if c not in ("date", P.ALVO)]
feats_lin = [c for c in feats_arv if c != "vh_30d"]
prev_t5 = pd.read_csv(os.path.join(REPO, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
hip_t5 = pd.read_csv(os.path.join(REPO, "outputs", "T5", "hiperparametros_T5.csv"), parse_dates=["origem"])
print(f"Grades restritas: floresta {len(GRADES['floresta_aleatoria'])} de {len(P.GRADE_RF)}; "
      f"XGBoost {len(GRADES['XGBoost'])} de {len(P.GRADE_XGB)}", flush=True)

# 1. Sanidade
arq_s = os.path.join(OUT, "sanidade.csv")
if not os.path.exists(arq_s):
    t0 = time.perf_counter()
    p, h = P.rodar(df, feats_lin, feats_arv, "expansiva", {"floresta_aleatoria": P.GRADE_RF, "XGBoost": P.GRADE_XGB},
                   max_origens=1, modelos=ARV, verboso=False)
    linhas = []
    for m in ARV:
        a = p[p.modelo == m].set_index("date").previsao
        b = prev_t5[(prev_t5.janela == "expansiva") & (prev_t5.modelo == m)].set_index("date").previsao.loc[a.index]
        ha = h[h.modelo == m].params.iloc[0]
        hb = hip_t5[(hip_t5.janela == "expansiva") & (hip_t5.modelo == m)].sort_values("origem").params.iloc[0]
        linhas.append({"item": f"{m}: previsões da 1ª origem = T5 (máx. |dif|)", "dif": float(np.max(np.abs(a - b)))})
        linhas.append({"item": f"{m}: hiperparâmetros da 1ª origem = T5", "dif": float(ha != hb)})
    pd.DataFrame(linhas).assign(segundos=time.perf_counter() - t0).to_csv(arq_s, index=False)
san = pd.read_csv(arq_s).assign(bate=lambda d: d.dif <= TOL)
print("=== Sanidade ===\n" + san.to_string(index=False), flush=True)
if not san.bate.all():
    sys.exit(2)

# 2. Grade restrita, por janela (checkpoint)
prevs, hips = [], []
for janela in ("expansiva", "movel"):
    arq_p, arq_h = os.path.join(OUT, f"previsoes_{janela}.csv"), os.path.join(OUT, f"hiperparametros_{janela}.csv")
    if not os.path.exists(arq_p):
        t0 = time.perf_counter()
        p, h = P.rodar(df, feats_lin, feats_arv, janela, GRADES, modelos=ARV, verboso=False)
        h.drop(columns="params_obj").to_csv(arq_h, index=False)
        p[p.modelo.isin(ARV)].to_csv(arq_p, index=False)
        print(f"  {janela}: {time.perf_counter() - t0:.0f}s", flush=True)
    prevs.append(pd.read_csv(arq_p, parse_dates=["date"]))
    hips.append(pd.read_csv(arq_h))
rest, hip = pd.concat(prevs), pd.concat(hips)

# 3. Avaliação
alvo = df.set_index("date")[P.ALVO]
mcs_t5 = pd.read_csv(os.path.join(REPO, "outputs", "MCS", "mcs_T5.csv"))
mcs_t5 = mcs_t5[(mcs_t5.bloco == MCS_BLOCO) & (mcs_t5.estatistica == "T_max")].set_index(["janela", "modelo"])
met_t5 = pd.read_csv(os.path.join(REPO, "outputs", "T5", "metricas_T5.csv")).set_index(["janela", "modelo"])
linhas = []
for janela in ("expansiva", "movel"):
    w = prev_t5[prev_t5.janela == janela].pivot(index="date", columns="modelo", values="previsao").sort_index()
    r = rest[rest.janela == janela].pivot(index="date", columns="modelo", values="previsao").sort_index()
    assert (w.index == r.index).all() and len(w) == 987
    y, mh = alvo.loc[w.index].to_numpy(), w["media_historica"].to_numpy()
    w2 = w.copy()
    for m in ARV:
        w2[m] = r[m]
    perdas = w2[P.MODELOS + ["media_historica", "persistencia_viavel", "proxy_retrospectiva"]].sub(y, axis=0) ** 2
    idx = mcs_utils.sortear_indices(len(perdas), bloco=MCS_BLOCO, B=MCS_B)
    mc = mcs_utils.mcs(perdas, medias_boot=mcs_utils.medias_bootstrap(perdas, idx), estatistica="max").set_index("modelo")
    for m in P.MODELOS + ["media_historica"]:
        for versao, f in (("original", w[m]), ("profundidade >= 3", r[m] if m in ARV else None)):
            if f is None:
                continue
            linha = {"janela": janela, "modelo": m, "versao": versao,
                     "mcs_p": float(mcs_t5.loc[(janela, m), "p_mcs"] if versao == "original" else mc.loc[m, "p_mcs"])}
            if m != "media_historica":
                linha.update({"r2_fora": r2_fora_da_amostra(y, f, mh), "p_cw": clark_west(y, f, mh, h=P.H)["cw_p_unilateral"],
                              "p_dm_vs_media": diebold_mariano(y, f, mh, h=P.H)["dm_p_bilateral"]})
                if versao != "original":
                    linha["p_dm_vs_original"] = diebold_mariano(y, f, w[m], h=P.H)["dm_p_bilateral"]
                    linha["r2_dentro"] = float(hip[(hip.janela == janela) & (hip.modelo == m)].r2_dentro_amostra.mean())
                else:
                    linha["r2_dentro"] = float(met_t5.loc[(janela, m), "r2_dentro_amostra_media"])
            linhas.append(linha)
    # MCS dos demais modelos com o conjunto trocado
    for m in ["MQO", "Ridge", "LASSO", "media_historica"]:
        linhas.append({"janela": janela, "modelo": m, "versao": "MCS com os restritos", "mcs_p": float(mc.loc[m, "p_mcs"])})
res = pd.DataFrame(linhas)
res.to_csv(os.path.join(OUT, "resultados.csv"), index=False)
prof = hip.assign(d=hip.params.str.extract(r"'max_depth': (\w+)")[0]).groupby(["janela", "modelo", "d"]).size()
prof.rename("n_origens").reset_index().to_csv(os.path.join(OUT, "profundidades.csv"), index=False)
pd.set_option("display.width", 220)
print("\n=== Resultados ===\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n=== Profundidade escolhida (nº de origens) ===\n" + prof.to_string())
