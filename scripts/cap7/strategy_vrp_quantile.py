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

    vrp:     serie temporal do VRP (index 