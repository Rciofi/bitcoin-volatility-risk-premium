"""
descritivas_C7.py — descritivas do Cap. 7 (Fase 2, C7.2 e C7.3)

Duas análises na amostra de modelagem (data/ml_dataset_T4.csv, N = 1.776),
sem reestimar nenhum modelo:

1. Retorno futuro médio por regime (C7.2), h = 1, 5, 10, 20, 30, 60.
   Diferença entre regimes por MQO de ret_fut_h numa dummy, EP de Newey–West
   com h+1 defasagens (hac_utils).
   (i)  principal: regime de alta pelo corte fixo de 37,3% a.a.
        (regime_alta_fixo de data/regimes_T9.csv);
   (ii) complemento: "volatilidade extrema", vh_30d no quartil superior da
        própria vh_30d em janela expansiva (dados até t, inclusive). O quartil
        só é calculado a partir de N_INICIAL_PADRAO = 730 observações, o mesmo
        burn-in da primeira janela de estimação (split_utils); as datas
        anteriores ficam fora da análise (ii).

2. BVRP e variação da IV pelo sinal do retorno do dia (C7.3).
   Grupos: ret_1d > 0 e ret_1d < 0 (ret_1d é log-retorno; o sinal é o mesmo
   do retorno simples). Diferença por MQO numa dummy (1 = retorno negativo),
   EP HAC: BVRP (alvo de 30 dias sobrepostos) com h = 30 (31 defasagens);
   d_iv_1d com h = 1 (2 defasagens). Complemento: d_iv_1d na parte positiva
   e na parte negativa de r_t (inclinações separadas, HAC com 2 defasagens),
   que distingue a assimetria de uma simples correlação negativa.

Saídas em outputs/C7/: retorno_por_regime_C7.csv, sinal_retorno_C7.csv,
resposta_iv_C7.csv e log_C7.txt.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from hac_utils import mqo_newey_west  # noqa: E402
from split_utils import N_INICIAL_PADRAO  # noqa: E402

HORIZONTES = (1, 5, 10, 20, 30, 60)
QUANTIL_EXTREMO = 0.75


def carregar(root):
    ml = pd.read_csv(os.path.join(root, "data", "ml_dataset_T4.csv"), parse_dates=["date"])
    alvo = pd.read_csv(os.path.join(root, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    reg = pd.read_csv(os.path.join(root, "data", "regimes_T9.csv"), parse_dates=["date"])
    cols_ret = [f"ret_fut_{h}d" for h in HORIZONTES]
    df = (ml[["date", "vh_30d", "iv_30d", "d_iv_1d", "ret_1d", "alvo_bvrp_30d_fut"]]
          .merge(alvo[["date"] + cols_ret], on="date", how="left", validate="1:1")
          .merge(reg[["date", "vh_30d", "regime_alta_fixo"]].rename(columns={"vh_30d": "vh_30d_reg"}),
                 on="date", how="left", validate="1:1"))
    assert len(df) == 1776, len(df)
    assert (df["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "calendário com lacunas"
    assert np.allclose(df["vh_30d"], df["vh_30d_reg"]), "vh_30d diverge entre T4 e T9"
    assert df[cols_ret + ["regime_alta_fixo"]].notna().all().all()
    return df.drop(columns="vh_30d_reg")


def quartil_expansivo(vh):
    """Quartil superior da vh_30d com dados até t (inclusive), a partir de 730 observações."""
    return vh.expanding(min_periods=N_INICIAL_PADRAO).quantile(QUANTIL_EXTREMO)


def diferenca_por_regime(df, dummy, definicao):
    linhas = []
    for h in HORIZONTES:
        y = df[f"ret_fut_{h}d"] * 100  # % de retorno simples
        d = df[dummy]
        res = mqo_newey_west(y, d, h=h)
        g0, g1 = y[d == 0], y[d == 1]
        assert np.isclose(res.params[dummy], g1.mean() - g0.mean())
        linhas.append({"definicao": definicao, "h": h, "n": int(len(y)),
                       "n_baixa": int(len(g0)), "n_alta": int(len(g1)),
                       "media_baixa": g0.mean(), "media_alta": g1.mean(),
                       "diferenca_alta_menos_baixa": res.params[dummy],
                       "ep_hac": res.bse[dummy], "p_hac": res.pvalues[dummy],
                       "maxlags": h + 1})
    return linhas


def sinal_retorno(df):
    sub = df[df["ret_1d"] != 0].copy()
    sub["negativo"] = (sub["ret_1d"] < 0).astype(int)
    linhas = []
    for var, h in (("alvo_bvrp_30d_fut", 30), ("d_iv_1d", 1)):
        res = mqo_newey_west(sub[var], sub["negativo"], h=h)
        g_pos, g_neg = sub.loc[sub["negativo"] == 0, var], sub.loc[sub["negativo"] == 1, var]
        assert np.isclose(res.params["negativo"], g_neg.mean() - g_pos.mean())
        linhas.append({"variavel": var, "n_pos": int(len(g_pos)), "n_neg": int(len(g_neg)),
                       "n_zero_excluidos": int((df["ret_1d"] == 0).sum()),
                       "media_pos": g_pos.mean(), "media_neg": g_neg.mean(),
                       "diferenca_neg_menos_pos": res.params["negativo"],
                       "ep_hac": res.bse["negativo"], "p_hac": res.pvalues["negativo"],
                       "maxlags": h + 1})
    return pd.DataFrame(linhas)


def resposta_iv(df):
    """d_iv_1d = a + b+ · max(r_t, 0) + b− · min(r_t, 0) + e, r_t em %; HAC com 2 defasagens (h = 1)."""
    X = pd.DataFrame({"r_pos": df["ret_1d"].clip(lower=0) * 100, "r_neg": df["ret_1d"].clip(upper=0) * 100})
    res = mqo_newey_west(df["d_iv_1d"], X, h=1)
    w = res.wald_test("r_pos + r_neg = 0", use_f=False, scalar=True)
    return pd.DataFrame([{"coef": c, "estimativa": res.params[c], "ep_hac": res.bse[c], "p_hac": res.pvalues[c]}
                         for c in ("const", "r_pos", "r_neg")]
                        + [{"coef": "simetria_b_pos_mais_b_neg", "estimativa": res.params["r_pos"] + res.params["r_neg"],
                            "ep_hac": np.nan, "p_hac": float(w.pvalue)}]).assign(n=int(res.nobs), maxlags=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=os.path.join(ROOT, "outputs", "C7"))
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    df = carregar(ROOT)
    df["q75_expansivo"] = quartil_expansivo(df["vh_30d"])
    df["extremo_expansivo"] = (df["vh_30d"] >= df["q75_expansivo"]).astype(int)

    linhas = diferenca_por_regime(df, "regime_alta_fixo", "corte_fixo_37_3")
    ext = df[df["q75_expansivo"].notna()]
    linhas += diferenca_por_regime(ext, "extremo_expansivo", "quartil_superior_expansivo")
    reg = pd.DataFrame(linhas)
    sinal = sinal_retorno(df)
    resp = resposta_iv(df)

    reg.to_csv(os.path.join(args.saida, "retorno_por_regime_C7.csv"), index=False)
    sinal.to_csv(os.path.join(args.saida, "sinal_retorno_C7.csv"), index=False)
    resp.to_csv(os.path.join(args.saida, "resposta_iv_C7.csv"), index=False)
    log = [
        f"Amostra: {df['date'].min():%d/%m/%Y} a {df['date'].max():%d/%m/%Y}, N = {len(df)}",
        f"Corte fixo: fração em alta = {df['regime_alta_fixo'].mean():.3f}",
        f"Quartil expansivo: a partir de {ext['date'].min():%d/%m/%Y} (N = {len(ext)}); "
        f"q75 de {ext['q75_expansivo'].min():.1f} a {ext['q75_expansivo'].max():.1f}; "
        f"fração extrema = {ext['extremo_expansivo'].mean():.3f}",
        "", reg.to_string(), "", sinal.to_string(), "", resp.to_string(),
    ]
    with open(os.path.join(args.saida, "log_C7.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
