\
import numpy as np
import pandas as pd


def vrp_quantile_position(
    vrp: pd.Series,
    q: float = 0.8
) -> pd.Series:
    """
    Gera posição 1 quando VRP >= quantil q,
    caso contrário posição 0.

    vrp: série temporal do VRP (index = datas)
    """
    vrp = vrp.dropna().astype(float)
    threshold = vrp.quantile(q)

    pos = (vrp >= threshold).astype(int)
    pos.name = "position"
    return pos


def run_bvrp_strategy(
    returns: pd.Series,
    bvrp: pd.Series,
    q: float = 0.80,
    lag: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """
    Estratégia condicional ao BVRP (quantil superior):
    - posição = 1 se bvrp > quantil(q), senão 0
    - defasa o sinal em 'lag' dias para evitar look-ahead
    Retorna:
      (strategy_returns, position)
    """
    # alinhar no mesmo índice
    idx = returns.index.intersection(bvrp.index)
    r = returns.loc[idx].astype(float)
    s = bvrp.loc[idx].astype(float)

    thr = float(s.quantile(q))
    raw_pos = (s > thr).astype(float)

    pos = raw_pos.shift(lag).fillna(0.0)
    strat_ret = pos * r
    strat_ret.name = f"BVRP q{int(q*100)}"
    pos.name = f"pos_BVRP_q{int(q*100)}"

    return strat_ret, pos



