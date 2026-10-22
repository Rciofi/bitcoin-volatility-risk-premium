"""
passo6_ponte_retorno_b2.py — Rodada 3, B2, passo 6: ponte entre o retorno simples e o log-retorno
na floresta direto no retorno

Só lê arquivos existentes e não reestima nenhuma floresta; nada vai para tables/ ou figs/.
Grava numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo6/.

Entradas (mesmo protocolo, só o alvo muda: retorno_bvrp_T11.arvore_retorno, 17 variáveis,
P.GRADE_RF, 500 árvores, 5 dobras com embargo h, 33 origens, random_state 42):
  retorno simples: outputs/T11/previsoes_arvore_retorno_T11.csv e hiperparametros_arvore_retorno_T11.csv
  log-retorno:     outputs/diagnostico_b2/20261021_105614_passo4B/horizontes/{previsoes,hiperparametros}_h{h}.csv
Avaliação com avaliacao_utils (r2_fora_da_amostra, clark_west, diebold_mariano; HAC com h+1
defasagens), em pontos percentuais.

Conferência (conferencia.csv; sai com código 3 se algo falhar; tolerância 1e-9):
  - 987 datas por h nos dois arquivos, todas casadas, com a mesma origem;
    realizado_log = log1p(realizado_simples);
  - R², p CW e p DM recalculados = outputs/T11/arvore_retorno_T11.csv (simples) e
    floresta_retorno_metricas.csv do passo 4B (log);
  - hiperparâmetros: mesmas 33 origens, n_treino e treino_fim por h nos dois retornos.

Saídas:
  dois_retornos.csv            (a) R² fora, CW e DM contra a média, 6 h × {simples, log}
  conversao.csv                (d) retorno simples, 6 h: realizado winsorizado a 1% e 99% (quantis das
                               987 realizações do h; previsões inalteradas); expm1(previsão log) contra a
                               média histórica do simples e contra expm1(média histórica do log)
  concentracao_h60.csv         (b) h = 60: datas para 50% e 80% de Σe_f², Σe_m² e Σ|d_t|, d_t = e_f² − e_m²;
                               Σd_t; fração de Σd_t nas 20 datas de maior e_f² (simples)
  perdas_por_data_h60.csv      (b) as 987 datas de h = 60, com e², d_t nos dois retornos
  maiores_erros_h60.csv        (b) as 15 datas de maior e_f² no retorno simples
  maiores_retornos_h60.csv     (c) 10 maiores retornos simples de 60 dias nas 987 datas e 5 maiores
                               na amostra de modelagem (1.776 datas), com o log-retorno
  hiperparametros.csv          por h e retorno: origens com os mesmos parâmetros nos dois retornos,
                               contagem de max_depth escolhido e R² dentro da amostra médio
  conferencia.csv, tempo.csv

Uso:  python code/diagnostico_b2/passo6_ponte_retorno_b2.py
"""
import argparse
import ast
import os
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402

HS = (1, 5, 10, 20, 30, 60)
N_OOS = 987
TOL = 1e-9
PASTA_4B = "20261021_105614_passo4B"
PROFUNDIDADES = (1, 2, 3, 6, None)


def caminhos():
    return (os.path.join(ROOT, "outputs", "T11"), os.path.join(ROOT, "outputs", "diagnostico_b2", PASTA_4B))


def carregar():
    t11, p4b = caminhos()
    s = pd.read_csv(os.path.join(t11, "previsoes_arvore_retorno_T11.csv"), parse_dates=["date", "origem"])
    lg = pd.concat([pd.read_csv(os.path.join(p4b, "horizontes", f"previsoes_h{h}.csv"), parse_dates=["date", "origem"])
                    for h in HS])
    m = s.merge(lg, on=["h", "date"], suffixes=("_s", "_l"), validate="1:1")
    conf = [("linhas: 987 por h nos dois arquivos, todas casadas",
             float(abs(len(m) - 6 * N_OOS) + abs(len(s) - 6 * N_OOS) + abs(len(lg) - 6 * N_OOS)), 0.0),
            ("mesma origem por data", float((m.origem_s != m.origem_l).sum()), 0.0),
            ("realizado_log = log1p(realizado_simples) (máx. |dif|)",
             float(np.max(np.abs(m.realizado_l - np.log1p(m.realizado_s)))), TOL)]
    return m.sort_values(["h", "date"]).reset_index(drop=True), conf


def metricas(y, f, ref, h):
    cw, dm = clark_west(y, f, ref, h=h), diebold_mariano(y, f, ref, h=h)
    return {"r2_fora": r2_fora_da_amostra(y, f, ref), "p_cw": cw["cw_p_unilateral"],
            "dm_t": dm["dm_t"], "p_dm": dm["dm_p_bilateral"]}


def n_para(x, frac):
    """Menor número de datas (das maiores para as menores) cuja soma atinge frac do total."""
    v = np.sort(np.asarray(x))[::-1]
    return int(np.searchsorted(np.cumsum(v) / v.sum(), frac) + 1)


def serie(g, sufixo):
    return tuple(100 * g[c + sufixo].to_numpy() for c in ("realizado", "previsao", "media_historica"))


def dois_retornos(m, conf):
    t11, p4b = caminhos()
    linhas = []
    for h in HS:
        g = m[m.h == h]
        for ret, sx in (("simples", "_s"), ("log", "_l")):
            linhas.append({"h": h, "retorno": ret, **metricas(*serie(g, sx), h)})
    a = pd.DataFrame(linhas)
    ref_s = pd.read_csv(os.path.join(t11, "arvore_retorno_T11.csv")).set_index("h")
    ref_l = pd.read_csv(os.path.join(p4b, "floresta_retorno_metricas.csv")).set_index("h")
    for h in HS:
        x_s = a[(a.h == h) & (a.retorno == "simples")].iloc[0]
        x_l = a[(a.h == h) & (a.retorno == "log")].iloc[0]
        conf.append((f"h = {h}: simples = T11 (R², p CW, p DM; máx. |dif|)", float(max(
            abs(x_s.r2_fora - ref_s.loc[h, "r2_oos_vs_media_historica"]),
            abs(x_s.p_cw - ref_s.loc[h, "cw_p_unilateral"]), abs(x_s.p_dm - ref_s.loc[h, "dm_p_bilateral"]))), TOL))
        conf.append((f"h = {h}: log = passo 4B (R², p CW, p DM; máx. |dif|)", float(max(
            abs(x_l.r2_fora - ref_l.loc[h, "r2_oos_vs_media"]),
            abs(x_l.p_cw - ref_l.loc[h, "cw_p_unilateral"]), abs(x_l.p_dm - ref_l.loc[h, "dm_p_bilateral"]))), TOL))
    return a


def concentracao_h60(m):
    g = m[m.h == 60].copy()
    linhas = []
    for ret, sx in (("simples", "_s"), ("log", "_l")):
        y, f, mh = serie(g, sx)
        ef2, em2 = (y - f) ** 2, (y - mh) ** 2
        d = ef2 - em2
        for perda, x in (("e2_floresta", ef2), ("e2_media", em2), ("abs_d", np.abs(d))):
            linhas.append({"retorno": ret, "perda": perda, "soma": x.sum(), "n_50pct": n_para(x, 0.5),
                           "n_80pct": n_para(x, 0.8), "n_datas": len(x)})
        linhas.append({"retorno": ret, "perda": "soma_d", "soma": d.sum(), "n_datas": len(d)})
        if ret == "simples":
            top20 = np.argsort(-ef2)[:20]
            linhas.append({"retorno": ret, "perda": "fracao_soma_d_top20_e2_floresta", "soma": d[top20].sum() / d.sum(),
                           "n_datas": 20})
        g[f"y{sx}"], g[f"f{sx}"], g[f"mh{sx}"], g[f"e2_floresta{sx}"], g[f"e2_media{sx}"], g[f"d{sx}"] = y, f, mh, ef2, em2, d
    por_data = g[["date", "origem_s"] + [f"{c}{sx}" for sx in ("_s", "_l")
                                         for c in ("y", "f", "mh", "e2_floresta", "e2_media", "d")]]
    por_data = por_data.rename(columns={"origem_s": "origem"})
    top = por_data.nlargest(15, "e2_floresta_s")
    return pd.DataFrame(linhas), por_data, top


def maiores_retornos_h60(m):
    g = m[m.h == 60]
    a = g.nlargest(10, "realizado_s")[["date", "realizado_s", "realizado_l"]].assign(amostra="fora_da_amostra_987")
    ml = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"])[["date"]]
    vt = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])[["date", "ret_fut_60d"]]
    mod = ml.merge(vt, on="date", validate="1:1")
    assert len(mod) == 1776
    b = (mod.nlargest(5, "ret_fut_60d").rename(columns={"ret_fut_60d": "realizado_s"})
         .assign(realizado_l=lambda d: np.log1p(d.realizado_s), amostra="modelagem_1776"))
    r = pd.concat([a, b])
    r["retorno_simples_pct"], r["log_retorno_pct"] = 100 * r.realizado_s, 100 * r.realizado_l
    y = 100 * g.realizado_s
    r = r.drop(columns=["realizado_s", "realizado_l"]).assign(
        q01_simples_987=float(np.quantile(y, 0.01)), q99_simples_987=float(np.quantile(y, 0.99)),
        assimetria_simples_987=float(g.realizado_s.skew()), assimetria_log_987=float(g.realizado_l.skew()))
    return r


def conversao(m):
    linhas = []
    for h in HS:
        g = m[m.h == h]
        y, f, mh = serie(g, "_s")
        lo, hi = np.quantile(y, [0.01, 0.99])
        linhas.append({"h": h, "variante": "winsor_1_99", **metricas(np.clip(y, lo, hi), f, mh, h),
                       "q01": lo, "q99": hi, "n_cortados": int(((y < lo) | (y > hi)).sum())})
        f_exp = 100 * np.expm1(g.previsao_l.to_numpy())
        linhas.append({"h": h, "variante": "expm1_log_vs_media_simples", **metricas(y, f_exp, mh, h)})
        mh_exp = 100 * np.expm1(g.media_historica_l.to_numpy())
        linhas.append({"h": h, "variante": "expm1_log_vs_expm1_media_log", **metricas(y, f_exp, mh_exp, h)})
    return pd.DataFrame(linhas)


def hiperparametros(conf):
    t11, p4b = caminhos()
    s = pd.read_csv(os.path.join(t11, "hiperparametros_arvore_retorno_T11.csv"))
    lg = pd.concat([pd.read_csv(os.path.join(p4b, "horizontes", f"hiperparametros_h{h}.csv")) for h in HS])
    x = s.merge(lg, on=["h", "origem"], suffixes=("_s", "_l"), validate="1:1")
    conf.append(("hiperparâmetros: 33 origens por h, mesmos n_treino e treino_fim",
                 float(abs(len(x) - 6 * 33) + (x.n_treino_s != x.n_treino_l).sum() + (x.treino_fim_s != x.treino_fim_l).sum()), 0.0))
    linhas = []
    for h, g in x.groupby("h"):
        iguais = int((g.params_s == g.params_l).sum())
        for ret, sx in (("simples", "_s"), ("log", "_l")):
            p = g[f"params{sx}"].map(ast.literal_eval)
            prof = p.map(lambda d: d["max_depth"])
            linha = {"h": h, "retorno": ret, "n_origens": len(g), "origens_mesmos_params": iguais,
                     "r2_dentro_media": float(g[f"r2_dentro_amostra{sx}"].mean())}
            for d in PROFUNDIDADES:
                linha[f"max_depth_{d if d is not None else 'None'}"] = int((prof.isna() if d is None else prof == d).sum())
            linhas.append(linha)
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="pasta de saída (padrão: outputs/diagnostico_b2/<carimbo>_passo6)")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo6")
    os.makedirs(out, exist_ok=bool(args.out))
    t0 = time.perf_counter()
    m, conf = carregar()
    a = dois_retornos(m, conf)
    c, por_data, top = concentracao_h60(m)
    r = maiores_retornos_h60(m)
    d = conversao(m)
    hp = hiperparametros(conf)
    conf = pd.DataFrame(conf, columns=["item", "dif", "tol"]).assign(bate=lambda x: x.dif <= x.tol)
    for nome, df in (("dois_retornos", a), ("conversao", d), ("concentracao_h60", c), ("perdas_por_data_h60", por_data),
                     ("maiores_erros_h60", top), ("maiores_retornos_h60", r), ("hiperparametros", hp),
                     ("conferencia", conf)):
        df.to_csv(os.path.join(out, f"{nome}.csv"), index=False, lineterminator="\n")
    pd.DataFrame([{"etapa": "total", "segundos": time.perf_counter() - t0}]).to_csv(
        os.path.join(out, "tempo.csv"), index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"Saídas em {out}")
    print("=== Conferência ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\n=== (a) dois retornos ===\n" + a.pivot(index="h", columns="retorno", values=["r2_fora", "p_cw", "p_dm"]).to_string(float_format=f4))
    print("\n=== (b) concentração, h = 60 ===\n" + c.to_string(index=False, float_format=f4))
    print("\n=== (c) maiores retornos de 60 dias ===\n" + r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== (d) conversão ===\n" + d.to_string(index=False, float_format=f4))
    print("\n=== hiperparâmetros ===\n" + hp.to_string(index=False, float_format=f4))
    if not conf.bate.all():
        sys.exit(3)


if __name__ == "__main__":
    main()
