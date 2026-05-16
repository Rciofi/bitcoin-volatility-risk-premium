# -*- coding: utf-8 -*-
"""Estrategia condicional ao BVRP via quantil expansivo (sem lookahead).

[P2] Substituicao do limiar fixo pela janela expansiva:
  - vrp_quantile_position() e run_bvrp_strategy() agora usam
    expanding_quantile_signal() para eliminar lookahead bias.
"""

import numpy as np
import pandas as pd

MIN_PERIODS = 252  # burn-in: minimo de observacoes para calcular o limiar


def expanding_quantile_signal(bvrp_series, quantile=0.80, min_obs=MIN_PERIODS):
    """[P2] Sinal binario com quantil em janela expansiva (vetorizado).

    - Em cada t, o limiar usa APENAS dados de [0, t-1] (ex-ante)
    - O sinal usa BVRP defasado em 1 dia
    - Burn-in: primeiros min_obs dias sem sinal (signal=0)

    Retorna: (signal Series, threshold Series)
    """
    th_series = bvrp_series.expanding(min_periods=min_obs).quantile(quantile)
    signal = (bvrp_series.shift(1) >= th_series.shift(1)).astype(float)
    signal = signal.fillna(0.0)
    signal.iloc[:min_obs] = 0.0
    return signal, th_series


def vrp_quantile_position(
    vrp: pd.Series,
    q: float = 0.8,
    min_obs: int = MIN_PERIODS,
) -> pd.Series:
    """Gera posicao 1 quando VRP >= quantil q (janela expansiva, ex-ante).

    vrp:     serie temporal do VRP (index = datas)
    q:       quantil de corte (padrao: 0.80)
    min_obs: periodo minimo de burn-in (padrao: 252 dias)

    Retorna: Series de posicoes (0 ou 1)
    """
    vrp = vrp.dropna().astype(float)
    pos, _ = expanding_quantile_signal(vrp, quantile=q, min_obs=min_obs)
    pos.name = "position"
    return pos


def run_bvrp_strategy(
    returns: pd.Series,
    bvrp: pd.Series,
    q: float = 0.80,
    lag: int = 1,
    min_obs: int = MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Estrategia condicional ao BVRP (quantil expansivo, sem lookahead).

    - Limiar calculado com janela expansiva (expanding_quantile_signal)
    - expanding_quantile_signal ja embute lag=1 (shift do sinal)
    - Burn-in: primeiros min_obs dias sem posicao

    Parametros
    ----------
    returns : retornos diarios do ativo
    bvrp    : serie do BVRP (vrp_30d)
    q       : quantil de corte (padrao: 0.80)
    lag     : defasagem extra alem do lag=1 ja embutido (raramente necessario)
    min_obs : burn-in minimo em dias

    Retorna
    -------
    (strategy_returns, position)
    """
    idx = returns.index.intersection(bvrp.index)
    r = returns.loc[idx].astype(float)
    s = bvrp.loc[idx].astype(float)

    pos, _ = expanding_quantile_signal(s, quantile=q, min_obs=min_obs)

    # lag adicional alem do ja embutido na expanding_quantile_signal (lag=1)
    if lag > 1:
        pos = pos.shift(lag - 1).fillna(0.0)

    strat_ret = pos * r
    strat_ret.name = f"BVRP q{int(q * 100)}"
    pos.name = f"pos_BVRP_q{int(q * 100)}"

    return strat_ret, pos
