"""
passo4_floresta_b2.py — Rodada 3, B2, passo 4, Etapa B: floresta direto no retorno com R_log

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é
alterado. Grava numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo4B/, ou,
com --out, retoma uma pasta existente (o que já foi gravado é pulado).

Modelo: retorno_bvrp_T11.arvore_retorno sem mudanças (floresta de 500 árvores, grade
P.GRADE_RF de 75 combinações, validação cruzada de 5 dobras com embargo h, semente 42,
33 origens da janela expansiva, treino s <= t − h), com a coluna ret_fut_{h}d trocada pelo
log-retorno log1p(ret_fut_{h}d) (fração, a escala do T11).

1. Sanidade (antes de tudo; grava sanidade_h1.csv): h = 1 com R_simples, sem trocar a
   coluna, contra outputs/T11/previsoes_arvore_retorno_T11.csv (previsão, média histórica e
   realizado, 1e-9) e hiperparametros_arvore_retorno_T11.csv (mesmos parâmetros e mse_cv
   por origem). Se não bater, grava a maior diferença absoluta e para com código 2, sem
   rodar os horizontes. Uma sanidade já gravada não é refeita.
2. R_log, h ∈ {1, 5, 10, 20, 30, 60}: um checkpoint por h em <pasta>/horizontes/
   (hiperparâmetros e, por último, previsões; um h com previsões gravadas é pulado).
3. Agregado (quando os 6 h existem), com y e previsões em p.p. (× 100):
   R² fora contra a média histórica e contra zero; Clark–West e Diebold–Mariano contra a
   média (HAC h+1); R² dentro da amostra médio; Diebold–Mariano da floresta contra o Ridge
   e o MQO da Etapa A (--etapa-a, oos_previsoes.csv). Conferência: a média histórica e o
   realizado são os da Etapa A (1e-9). Placebo: 10 permutações por h (como o T11),
   critério do T11 (mediana do R² placebo <= 0,01).

Uso:  python code/diagnostico_b2/passo4_floresta_b2.py --etapa-a <pasta da Etapa A>
      python code/diagnostico_b2/passo4_floresta_b2.py --etapa-a <pasta> --out <pasta da Etapa B>
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed, cpu_count  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
import previsao_bvrp_T5 as P  # noqa: E402
import retorno_bvrp_T11 as T11  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402

HS = T11.HS
GRADE = P.GRADE_RF
N_OOS = 987
N_PERM = 10
TOL = 1e-9
REF_T11 = os.path.join(ROOT, "outputs", "T11")


def carregar():
    df, _, _, _, feats_arv, _ = T11.carregar()                                    # 17 variáveis em nível
    d_log = df.copy()
    for h in HS:
        d_log[f"ret_fut_{h}d"] = np.log1p(df[f"ret_fut_{h}d"])
    return df, d_log, feats_arv


def tempo(out, etapa, seg):
    arq = os.path.join(out, "tempo.csv")
    pd.DataFrame([{"etapa": etapa, "segundos": seg, "nucleos": cpu_count()}]).to_csv(
        arq, mode="a", index=False, header=not os.path.exists(arq))


# ---------------------------------------------------------------------------
# 1. Sanidade
# ---------------------------------------------------------------------------
def sanidade(df, feats, out):
    arq = os.path.join(out, "sanidade_h1.csv")
    if not os.path.exists(arq):
        t0 = time.perf_counter()
        p, hp = T11.arvore_retorno(df, feats, 1, GRADE)
        ref = pd.read_csv(os.path.join(REF_T11, "previsoes_arvore_retorno_T11.csv"), parse_dates=["date"]).query("h == 1")
        p = p.assign(date=pd.to_datetime(p.date))
        m = p.merge(ref, on="date", suffixes=("", "_t11"))
        assert len(m) == len(p) == len(ref) == N_OOS
        linhas = [{"item": f"h = 1, R_simples: {c} = T11 (máx. |dif|)",
                   "dif": float(np.max(np.abs(m[c] - m[f"{c}_t11"]))), "tol": TOL}
                  for c in ("previsao", "media_historica", "realizado")]
        hr = pd.read_csv(os.path.join(REF_T11, "hiperparametros_arvore_retorno_T11.csv"), parse_dates=["origem"]).query("h == 1")
        hm = hp.assign(origem=pd.to_datetime(hp.origem)).merge(hr, on="origem", suffixes=("", "_t11"))
        assert len(hm) == len(hp) == len(hr)
        linhas.append({"item": "h = 1: origens com parâmetros diferentes do T11",
                       "dif": float((hm.params != hm.params_t11).sum()), "tol": 0.0})
        linhas.append({"item": "h = 1: mse_cv = T11 (máx. |dif|)",
                       "dif": float(np.max(np.abs(hm.mse_cv - hm.mse_cv_t11))), "tol": TOL})
        s = pd.DataFrame(linhas).assign(bate=lambda d: d.dif <= d.tol)
        s.to_csv(arq, index=False)
        tempo(out, "1_sanidade_h1", time.perf_counter() - t0)
    s = pd.read_csv(arq)
    print("=== Sanidade (h = 1, R_simples × T11) ===\n" + s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if not s.bate.all():
        print(f"\nSANIDADE FALHOU: maior diferença absoluta = {s.dif.max():.6g}. Os horizontes não foram rodados.")
        sys.exit(2)


# ---------------------------------------------------------------------------
# 2. Horizontes (checkpoint por h)
# ---------------------------------------------------------------------------
def horizontes(d_log, feats, out):
    dir_h = os.path.join(out, "horizontes")
    os.makedirs(dir_h, exist_ok=True)
    for h in HS:
        arq = os.path.join(dir_h, f"previsoes_h{h}.csv")
        if os.path.exists(arq):
            print(f"  h = {h}: já gravado, pulado", flush=True)
            continue
        t0 = time.perf_counter()
        p, hp = T11.arvore_retorno(d_log, feats, h, GRADE)
        hp.drop(columns="params_obj").to_csv(os.path.join(dir_h, f"hiperparametros_h{h}.csv"), index=False)
        p.to_csv(arq, index=False)                                                 # por último: marca o h como feito
        seg = time.perf_counter() - t0
        tempo(out, f"2_floresta_h{h}", seg)
        print(f"  h = {h}: {seg:.0f}s", flush=True)
    return (pd.concat([pd.read_csv(os.path.join(dir_h, f"previsoes_h{h}.csv"), parse_dates=["date", "origem"]) for h in HS]),
            pd.concat([pd.read_csv(os.path.join(dir_h, f"hiperparametros_h{h}.csv"), parse_dates=["origem"]) for h in HS]))


# ---------------------------------------------------------------------------
# 3. Agregado
# ---------------------------------------------------------------------------
def agregado(d_log, feats, prev, hiper, etapa_a, out):
    lin = pd.read_csv(os.path.join(etapa_a, "oos_previsoes.csv"), parse_dates=["date"])
    df_a = d_log.set_index("date")
    met, conf = [], []
    for h, g in prev.groupby("h"):
        g = g.sort_values("date")
        assert len(g) == N_OOS
        y, f, mh = 100 * g.realizado.to_numpy(), 100 * g.previsao.to_numpy(), 100 * g.media_historica.to_numpy()
        la = lin[lin.h == h].set_index("date").loc[g.date]
        conf.append({"item": f"h = {h}: média histórica = Etapa A (máx. |dif|)",
                     "dif": float(np.max(np.abs(mh - la.media_historica.to_numpy()))), "tol": TOL})
        conf.append({"item": f"h = {h}: realizado = 100·log1p(ret_fut) (máx. |dif|)",
                     "dif": float(np.max(np.abs(y - 100 * df_a.loc[g.date, f"ret_fut_{h}d"].to_numpy()))), "tol": TOL})
        cw = clark_west(y, f, mh, h=h)
        dm = diebold_mariano(y, f, mh, h=h)                                         # dm_media > 0: a média erra menos
        linha = {"h": h, "n": len(g), "r2_oos_vs_media": r2_fora_da_amostra(y, f, mh),
                 "r2_oos_vs_zero": r2_fora_da_amostra(y, f, np.zeros(len(y))), **cw, **dm,
                 "r2_dentro_amostra_media": float(hiper[hiper.h == h].r2_dentro_amostra.mean())}
        for nome in ("Ridge", "MQO"):
            d = diebold_mariano(y, f, la[nome].to_numpy(), h=h)                    # > 0: o linear erra menos
            linha.update({f"dm_vs_{nome}_media": d["dm_media"], f"dm_vs_{nome}_t": d["dm_t"],
                          f"dm_vs_{nome}_p_bilateral": d["dm_p_bilateral"]})
        met.append(linha)
    conf = pd.DataFrame(conf).assign(bate=lambda d: d.dif <= d.tol)
    conf.to_csv(os.path.join(out, "conferencia_etapa_a.csv"), index=False)
    met = pd.DataFrame(met)
    met.to_csv(os.path.join(out, "floresta_retorno_metricas.csv"), index=False)
    hp_obj = hiper.assign(params_obj=hiper.params.map(T11.T10._params))
    pd.concat([P.limites_da_grade(g).assign(h=h) for h, g in hp_obj.groupby("h")]).to_csv(   # como o T11
        os.path.join(out, "floresta_limites_grade.csv"), index=False)

    arq = os.path.join(out, "placebo_permutacoes.csv")
    if not os.path.exists(arq):
        t0 = time.perf_counter()
        fixos = {(r.h, T11.chave(r.origem)): T11.T10._params(r.params) for r in hiper.itertuples()}
        pl = pd.DataFrame(Parallel(n_jobs=-1)(delayed(T11.placebo_retorno)(b, h, d_log, feats, fixos)
                                              for h in HS for b in range(1, N_PERM + 1)))
        pl.to_csv(arq, index=False)
        tempo(out, "3_placebo", time.perf_counter() - t0)
    pl = pd.read_csv(arq)
    res = pl.groupby("h").r2_oos_vs_media_historica.agg(["count", "median", "min", "max"]).reset_index()
    res = res.assign(passa=lambda d: d["median"] <= T11.TOL_PLACEBO)
    res.to_csv(os.path.join(out, "placebo.csv"), index=False)
    return conf, met, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa-a", required=True, help="pasta da Etapa A (oos_previsoes.csv)")
    ap.add_argument("--out", help="pasta da Etapa B a retomar")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assert os.path.isfile(os.path.join(args.etapa_a, "oos_previsoes.csv")), "Etapa A sem oos_previsoes.csv"
    out = args.out or os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo4B")
    os.makedirs(out, exist_ok=bool(args.out))
    print(f"Saídas em {out}; núcleos: {cpu_count()}", flush=True)
    t_ini = time.perf_counter()
    df, d_log, feats = carregar()
    sanidade(df, feats, out)
    prev, hiper = horizontes(d_log, feats, out)
    conf, met, res = agregado(d_log, feats, prev, hiper, args.etapa_a, out)
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== Conferência com a Etapa A ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\n=== Floresta direto no retorno (R_log) ===\n" + met.to_string(index=False, float_format=fmt))
    print("\n=== Placebo ===\n" + res.to_string(index=False, float_format=fmt))
    print(f"\nTempo desta execução: {(time.perf_counter() - t_ini) / 60:.1f} min")
    if not conf.bate.all():
        sys.exit(3)


if __name__ == "__main__":
    main()
