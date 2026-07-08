# -*- coding: utf-8 -*-
"""
Validador de reprodutibilidade da Tabela 3.2 (testes de estacionariedade ADF/KPSS).
NAO regenera a tabela .tex — os valores publicados sao a fonte oficial. Este script
CONFIRMA que as CONCLUSOES (ordem de integracao) permanecem validas sobre o dataset
canonico. Serve como teste de regressao: falha apenas se alguma conclusao mudar.

Uso:  MPLBACKEND=Agg python3 code/validate_tab_3_adf_kpss.py
"""
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

CSV = "data/vrp_with_targets.csv"
ALPHA = 0.05

# Conclusoes publicadas na Tab. 3.2. "persistente" engloba as tres variaveis de
# volatilidade (RV/IV/BVRP): todas sao altamente persistentes. A IV e um caso de
# fronteira (ADF borderline, p entre 0,05 e 0,10) que a propria Tab. 3.2 documenta
# em rodape -- classificado aqui como persistente, nao I(1) puro.
ESPERADO = {
    "close":   "I(1)",
    "retorno": "I(0)",
    "rv_30d":  "persistente",
    "iv_30d":  "persistente",
    "vrp_30d": "persistente",
}

def classifica(adf_p, kpss_p):
    # Faixas: forte (<0,05), borderline (0,05-0,10), nao-rejeita (>=0,10)
    adf_rej   = adf_p  < ALPHA           # rejeita raiz unitaria
    adf_bord  = ALPHA <= adf_p < 0.10    # zona de fronteira do ADF
    kpss_rej  = kpss_p < ALPHA           # rejeita estacionariedade

    # I(1) puro: ADF claramente NAO rejeita (p>=0,10) e KPSS rejeita
    if (not adf_rej and not adf_bord) and kpss_rej:
        return "I(1)"
    # I(0): ADF rejeita e KPSS NAO rejeita
    if adf_rej and not kpss_rej:
        return "I(0)"
    # Persistente / longa memoria: KPSS rejeita E (ADF rejeita OU esta na fronteira).
    # Cobre RV/BVRP (ambos rejeitam) e IV (ADF borderline + KPSS rejeita).
    if kpss_rej and (adf_rej or adf_bord):
        return "persistente"
    return "inconclusivo"

def main():
    df = pd.read_csv(CSV, parse_dates=["date"])
    print(f"Dataset: {CSV}  |  N={len(df)}  |  {df['date'].min().date()} a {df['date'].max().date()}\n")

    series = {}
    if "close" in df.columns:
        series["close"]   = df["close"]
        series["retorno"] = np.log(df["close"]).diff()
    series["rv_30d"]  = df["rv_30d"]
    series["iv_30d"]  = df["iv_30d"]
    series["vrp_30d"] = df["vrp_30d"]

    print(f"{'Variavel':10s} {'ADF':>9s} {'p(ADF)':>8s} {'KPSS':>9s} {'p(KPSS)':>8s}  {'Concl.':>12s}  {'Esperado':>12s}  OK?")
    print("-" * 88)

    todas_ok = True
    for nome, s in series.items():
        s2 = s.dropna()
        adf_stat, adf_p, *_ = adfuller(s2, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s2, regression="c", nlags="auto")
        concl = classifica(adf_p, kpss_p)
        esp = ESPERADO.get(nome, "?")
        ok = (concl == esp)
        todas_ok = todas_ok and ok
        print(f"{nome:10s} {adf_stat:9.4f} {adf_p:8.4f} {kpss_stat:9.4f} {kpss_p:8.4f}  {concl:>12s}  {esp:>12s}  {'OK' if ok else 'FALHA'}")

    print("-" * 88)
    if todas_ok:
        print("\nRESULTADO: todas as conclusoes da Tab. 3.2 CONFIRMADAS sobre o dataset canonico.")
        print("(Diferencas em 2a casa decimal vs. tabela publicada sao esperadas — versao/nlags do statsmodels.)")
        sys.exit(0)
    else:
        print("\nRESULTADO: ALGUMA CONCLUSAO DIVERGE. Investigar antes de confiar na Tab. 3.2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
