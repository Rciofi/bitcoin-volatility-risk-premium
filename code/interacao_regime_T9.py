"""
interacao_regime_T9.py — interação regime × variáveis no modelo linear (T9)

Modelo: alvo_bvrp_30d_fut ~ X + D + D·X, com X = as 16 variáveis lineares do
T5 (todas menos vh_30d) e D = 1 no regime de alta volatilidade
(data/regimes_T9.csv).

Duas variantes do regime:
  fixo       corte da primeira janela (principal);
  expansivo  corte reestimado em cada origem do split_utils com dados até a
             origem (robustez). Na avaliação fora da amostra, o corte de cada
             origem vale para o treino e o teste daquela origem; no teste F,
             vale o corte da origem mais recente <= data (antes da 1ª origem,
             o corte fixo).

1. Teste F (dentro da amostra, N = 1.776): MQO com erro-padrão de
   Newey–West via hac_utils.mqo_newey_west(h = 30), isto é, maxlags = 31.
   H0: as 16 interações são nulas (e, à parte, interações + dummy). Teste de
   Wald com a matriz HAC: estatística qui-quadrado (inferência pela normal,
   coerente com hac_utils) e a versão F = qui²/q.
2. Fora da amostra: Ridge (α por validação cruzada embargada, como no T5) e
   MQO com D e D·X, nas mesmas 33 origens do split_utils. Comparação com os
   mesmos modelos sem regime (data/previsoes_bvrp_T5.csv): R² contra a média
   histórica, Clark–West contra a média e Diebold–Mariano contra o modelo sem
   regime (HAC h+1).

Uso:  python code/interacao_regime_T9.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import previsao_bvrp_T5 as P  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from hac_utils import mqo_newey_west  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "T9")
ALVO, H = P.ALVO, P.H


def montar(df, feats, d):
    """Matriz X, D e D·X (nomes int_<variável>)."""
    X = df[feats].copy()
    X["regime_alta"] = d
    for f in feats:
        X[f"int_{f}"] = d * df[f]
    return X


def teste_f(df, feats, d, rotulo):
    X = montar(df, feats, d)
    res = mqo_newey_west(df[ALVO], X, h=H)
    nomes = list(res.params.index)
    linhas = []
    for alvo_teste, cols in [("interações (16)", [c for c in nomes if c.startswith("int_")]),
                             ("interações + dummy (17)", [c for c in nomes if c.startswith("int_") or c == "regime_alta"])]:
        R = np.zeros((len(cols), len(nomes)))
        for i, c in enumerate(cols):
            R[i, nomes.index(c)] = 1.0
        w = res.wald_test(R, use_f=False, scalar=True)
        q = len(cols)
        chi2 = float(w.statistic)
        linhas.append({"regime": rotulo, "hipotese_nula": alvo_teste, "q": q, "n": int(res.nobs),
                       "pct_alta": 100 * float(np.mean(d)), "qui2": chi2, "p_qui2": float(w.pvalue),
                       "F": chi2 / q, "p_F": float(stats.f.sf(chi2 / q, q, res.df_resid)),
                       "r2_dentro_amostra": float(res.rsquared)})
    return linhas


def fora_da_amostra(df, feats, regime_por_origem, rotulo):
    """Ridge e MQO com regime e interações nas origens do split_utils (janela expansiva)."""
    y_all = df[ALVO].to_numpy(dtype=float)
    datas = df.date.to_numpy()
    div = gerar_divisoes(len(df), h=H, janela="expansiva")
    linhas = []
    for o in div.origens:
        d = regime_por_origem(o)
        X = montar(df, feats, d).to_numpy(dtype=float)
        for nome in ["MQO", "Ridge"]:
            p = {} if nome == "MQO" else P.escolher("Ridge", P.GRADE_RIDGE, X, y_all, o.treino)[0]
            prever, _ = P.ajustar(nome, p, X[o.treino], y_all[o.treino])
            for u, v in zip(o.teste, prever(X[o.teste])):
                linhas.append({"date": datas[u], "origem": datas[o.t], "regime": rotulo,
                               "modelo": f"{nome}_regime", "previsao": float(v), "params": repr(p)})
    return pd.DataFrame(linhas)


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"]).sort_values("date")
    reg = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])
    df = df.merge(reg[["date", "regime_alta_fixo", "regime_alta_expansivo"]], on="date", how="left").reset_index(drop=True)
    assert df[["regime_alta_fixo", "regime_alta_expansivo"]].notna().all().all()
    cortes = pd.read_csv(os.path.join(OUT, "cortes_por_origem_T9.csv"), parse_dates=["origem"])
    feats = [c for c in df.columns if c not in ("date", ALVO, "vh_30d", "regime_alta_fixo", "regime_alta_expansivo")]
    assert len(feats) == 16, feats

    # 1) teste F
    tf = pd.DataFrame(teste_f(df, feats, df.regime_alta_fixo.to_numpy(), "fixo") +
                      teste_f(df, feats, df.regime_alta_expansivo.to_numpy(), "expansivo"))
    tf.to_csv(os.path.join(OUT, "teste_F_T9.csv"), index=False)

    # 2) fora da amostra
    corte_da_origem = dict(zip(cortes.origem, cortes.corte))
    vh = df.vh_30d.to_numpy()
    prevs = pd.concat([
        fora_da_amostra(df, feats, lambda o: df.regime_alta_fixo.to_numpy(), "fixo"),
        fora_da_amostra(df, feats, lambda o: (vh > corte_da_origem[pd.Timestamp(df.date[o.t])]).astype(int), "expansivo"),
    ])
    prevs.to_csv(os.path.join(OUT, "previsoes_interacao_T9.csv"), index=False)

    base = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
    base = base[base.janela == "expansiva"].pivot(index="date", columns="modelo", values="previsao")
    alvo = df.set_index("date")[ALVO]
    met = []
    for (rot, mod), g in prevs.groupby(["regime", "modelo"]):
        g = g.set_index("date").sort_index()
        y, p = alvo.loc[g.index], g.previsao
        media, sem = base.loc[g.index, "media_historica"], base.loc[g.index, mod.replace("_regime", "")]
        met.append({"regime": rot, "modelo": mod, "n": len(g),
                    "r2_oos_vs_media": r2_fora_da_amostra(y, p, media),
                    "r2_oos_sem_regime_vs_media": r2_fora_da_amostra(y, sem, media),
                    **{f"{k}_vs_media": v for k, v in clark_west(y, p, media, h=H).items()},
                    **{f"{k}_vs_sem_regime": v for k, v in diebold_mariano(y, p, sem, h=H).items()}})
    met = pd.DataFrame(met)
    met.to_csv(os.path.join(OUT, "metricas_interacao_T9.csv"), index=False)

    pd.set_option("display.width", 220)
    print("=== Teste F (Wald, HAC com 31 defasagens) ===")
    print(tf.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Fora da amostra (987 previsões, janela expansiva) ===")
    print(met.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaídas em {OUT}")


if __name__ == "__main__":
    main()
