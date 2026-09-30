"""
build_ml_dataset_T4.py — variáveis explicativas do T4 (plano de revisão, set/2026)

Gera data/ml_dataset_T4.csv: o alvo do Cap. 5 novo (alvo_bvrp_30d_fut) e as
variáveis explicativas disponíveis em t. NÃO altera data/ml_dataset.csv nem
code/build_ml_dataset.py, que alimentam os scripts antigos.

Regra: nenhuma variável explicativa usa informação posterior a t. Todas saem
de construir_features(precos, dvol), uma função pura das séries brutas até t
(testada em code/test_T4_vazamento.py cortando os dados brutos em t).

Convenção de datas: o close do BTC (Binance) e o DVOL (Deribit) da data t são
as velas que abrem às 00:00 UTC de t; ambos são observados no fim do dia t.

Variáveis (decisões do T4 em docs/pendencias_T1.md, seção 6):
  Volatilidade histórica, vh_kd = sqrt(365 · média(r²) em [t-k+1, t]) · 100,
  r = Δlog(close), mesma fórmula da rv_30d (vh_30d é idêntica a ela):
    vh_1d, vh_5d, vh_30d (estrutura HAR de Corsi, 2009), vh_60d, vh_90d
  Volatilidade implícita: iv_30d (nível), d_iv_1d, d_iv_5d (diferenças),
    iv_menos_ma5d, iv_menos_ma30d (IV menos sua média móvel; as médias em
    nível são I(1))
  Preço (close é I(1), entra transformado): ret_1d (Δlog), ret_acum_5d,
    ret_acum_30d, log_close_ma30d = log(close / média móvel de 30 dias)
  Prêmio (sempre a PROXY retrospectiva, nunca o prospectivo):
    vrp_30d = vh_30d - iv_30d, d_vrp_1d
  ATENÇÃO: vrp_30d = vh_30d - iv_30d exatamente; as três não podem entrar
  juntas num MQO.

Variáveis de longa memória (volatilidades e iv_30d) ficam em nível, mesmo
com o ADF na fronteira (vh_90d, iv_30d); só o que tem evidência robusta de
raiz unitária é transformado. Resultados ADF/KPSS na amostra final, com a
classe do teste e a classificação adotada, em
outputs/T4/dicionario_adf_kpss_T4.csv.

Amostra: datas de data/vrp_with_targets.csv (referência, N = 1.806, já com o
corte de h = 60) em que todas as variáveis existem. A perda vem só do início
do DVOL (24/03/2021): iv_menos_ma30d exige 30 dias de IV -> N = 1.777, a
partir de 22/04/2021.

Uso:  python code/build_ml_dataset_T4.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_vrp_dataset import check_daily_continuity  # noqa: E402
from validate_tab_3_adf_kpss import classifica  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_DOC = os.path.join(ROOT, "outputs", "T4")

ALVO = "alvo_bvrp_30d_fut"

# nome: (fórmula, janela, justificativa de disponibilidade em t, transformação)
DICIONARIO = {
    "vh_1d":           ("sqrt(365·r_t²)·100 = |r_t|·sqrt(365)·100", "[t, t]", "só r_t", "nível (longa memória)"),
    "vh_5d":           ("sqrt(365·média(r²))·100", "[t-4, t]", "retornos até t", "nível (longa memória)"),
    "vh_30d":          ("sqrt(365·média(r²))·100 (= rv_30d)", "[t-29, t]", "retornos até t", "nível (longa memória)"),
    "vh_60d":          ("sqrt(365·média(r²))·100", "[t-59, t]", "retornos até t", "nível (longa memória)"),
    "vh_90d":          ("sqrt(365·média(r²))·100", "[t-89, t]", "retornos até t", "nível"),
    "iv_30d":          ("DVOL_t", "t", "DVOL observado no fim de t", "nível"),
    "d_iv_1d":         ("IV_t - IV_{t-1}", "[t-1, t]", "IV até t", "diferença"),
    "d_iv_5d":         ("IV_t - IV_{t-5}", "[t-5, t]", "IV até t", "diferença"),
    "iv_menos_ma5d":   ("IV_t - média(IV)", "[t-4, t]", "IV até t", "desvio da média móvel (média em nível é I(1))"),
    "iv_menos_ma30d":  ("IV_t - média(IV)", "[t-29, t]", "IV até t", "desvio da média móvel (média em nível é I(1))"),
    "ret_1d":          ("log(close_t / close_{t-1})", "[t-1, t]", "preços até t", "log-diferença de close (I(1))"),
    "ret_acum_5d":     ("soma de r", "[t-4, t]", "retornos até t", "soma de log-diferenças"),
    "ret_acum_30d":    ("soma de r", "[t-29, t]", "retornos até t", "soma de log-diferenças"),
    "log_close_ma30d": ("log(close_t / média(close))", "[t-29, t]", "preços até t", "preço relativo à média móvel (close é I(1))"),
    "vrp_30d":         ("vh_30d - iv_30d (proxy retrospectiva)", "[t-29, t]", "retornos e IV até t", "nível (longa memória)"),
    "d_vrp_1d":        ("vrp_30d_t - vrp_30d_{t-1} (proxy)", "[t-30, t]", "proxy até t; nunca o prospectivo", "diferença"),
}
FEATURES = list(DICIONARIO)

# Critério de transformação (docs/pendencias_T1.md, seção 6): transformar o que
# tem evidência robusta de raiz unitária (log do preço, médias móveis da IV em
# nível); manter em nível as volatilidades persistentes mesmo com o ADF na
# fronteira. Nestas duas o ADF é instável (vh_90d muda de classe ao retirar
# 29 obs.; janelas vizinhas compartilham 89 de 90 dias), então a
# classificação adotada difere do resultado mecânico do teste.
LONGA_MEMORIA_FRONTEIRA = {"vh_90d", "iv_30d"}
ROTULO_FRONTEIRA = "longa memória (ADF instável/na fronteira)"


def carregar_brutos():
    """Séries brutas: preços (date, close) e DVOL (date, iv), contínuas."""
    p = pd.read_csv(os.path.join(DATA, "btc_prices.csv"))
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)[["date", "close"]]
    check_daily_continuity(p["date"], "btc_prices.csv")

    v = pd.read_csv(os.path.join(DATA, "dvol_30d_full.csv"))
    v["date"] = pd.to_datetime(v["timestamp"]).dt.tz_localize(None).dt.normalize()
    v = v.sort_values("date").reset_index(drop=True)
    v = v.rename(columns={"close": "iv"})[["date", "iv"]]
    check_daily_continuity(v["date"], "dvol_30d_full.csv")
    return p, v


def construir_features(precos, dvol):
    """
    Variáveis explicativas a partir das séries brutas. Função pura: usa só
    janelas para trás (rolling, diff), então o valor em t depende apenas de
    dados até t. Retorna DataFrame indexado por data com FEATURES (NaN onde
    a janela ainda não está completa).
    """
    p = precos.set_index("date")["close"].astype(float)
    r = np.log(p).diff()
    f = pd.DataFrame(index=p.index)
    for k in (1, 5, 30, 60, 90):
        f[f"vh_{k}d"] = np.sqrt((r ** 2).rolling(k).mean() * 365) * 100
    f["ret_1d"] = r
    f["ret_acum_5d"] = r.rolling(5).sum()
    f["ret_acum_30d"] = r.rolling(30).sum()
    f["log_close_ma30d"] = np.log(p / p.rolling(30).mean())

    iv = dvol.set_index("date")["iv"].astype(float)
    g = pd.DataFrame({"iv_30d": iv})
    g["d_iv_1d"] = iv.diff(1)
    g["d_iv_5d"] = iv.diff(5)
    g["iv_menos_ma5d"] = iv - iv.rolling(5).mean()
    g["iv_menos_ma30d"] = iv - iv.rolling(30).mean()

    x = f.join(g, how="inner")
    x["vrp_30d"] = x["vh_30d"] - x["iv_30d"]
    x["d_vrp_1d"] = x["vrp_30d"].diff(1)
    return x[FEATURES]


def main():
    warnings.filterwarnings("ignore")
    precos, dvol = carregar_brutos()
    feats = construir_features(precos, dvol)

    ref = pd.read_csv(os.path.join(DATA, "vrp_with_targets.csv"), parse_dates=["date"]).set_index("date")
    print(f"Amostra de referência: N={len(ref)}  {ref.index.min().date()} a {ref.index.max().date()}")

    # coerência com o pipeline: vh_30d e vrp_30d idênticas às colunas do T1
    comum = feats.loc[ref.index]
    for a, b in [("vh_30d", "rv_30d"), ("iv_30d", "iv_30d"), ("vrp_30d", "vrp_30d")]:
        dif = (comum[a] - ref[b]).abs().max()
        assert dif < 1e-9, f"{a} difere de {b} do pipeline (máx {dif})"

    df = comum.copy()
    df.insert(0, ALVO, ref["bvrp_30d_fut"])
    perdidas = df[df.isna().any(axis=1)]
    df = df.dropna()
    df.index.name = "date"
    colunas_fut = [c for c in df.columns if "fut" in c and c != ALVO]
    assert not colunas_fut, f"colunas prospectivas além do alvo: {colunas_fut}"

    out = os.path.join(DATA, "ml_dataset_T4.csv")
    df.to_csv(out)
    print(f"Salvo: {out}")
    print(f"N final = {len(df)}  ({df.index.min().date()} a {df.index.max().date()}); "
          f"perdidas {len(perdidas)} no início ({perdidas.index.min().date()} a {perdidas.index.max().date()}):")
    print("  NaN por variável nas datas perdidas:", perdidas.isna().sum()[lambda s: s > 0].to_dict())

    # documentação: dicionário + ADF/KPSS + descritivas, na amostra final
    os.makedirs(OUT_DOC, exist_ok=True)
    linhas = []
    for c in [ALVO] + FEATURES:
        s = df[c]
        a = adfuller(s, autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        formula, janela, disp, transf = (
            ("RV(t+1..t+30) - IV_t", "[t+1, t+30]", "NÃO: conhecido só em t+30 (é o alvo)", "nível")
            if c == ALVO else DICIONARIO[c])
        classe_teste = classifica(a[1], k[1])
        adotada = ROTULO_FRONTEIRA if c in LONGA_MEMORIA_FRONTEIRA else classe_teste
        linhas.append({"variavel": c, "formula": formula, "janela": janela,
                       "disponivel_em_t": disp, "adf_stat": a[0], "adf_p": a[1],
                       "kpss_stat": k[0], "kpss_p": k[1], "classe_teste": classe_teste,
                       "classificacao_adotada": adotada, "transformacao": transf})
    pd.DataFrame(linhas).to_csv(os.path.join(OUT_DOC, "dicionario_adf_kpss_T4.csv"), index=False)
    desc = df.describe(percentiles=[0.05, 0.5, 0.95]).T
    desc.to_csv(os.path.join(OUT_DOC, "desc_stats_T4.csv"))
    print(f"Documentação em: {OUT_DOC}")


if __name__ == "__main__":
    main()
