\
import numpy as np
import pandas as pd


def run_buy_and_hold(returns: pd.Series) -> pd.Series:
    """
    Buy-and-Hold:
    Mantém exposição unitária ao ativo durante todo o período.

    returns: retornos simples (decimal), ex: 0.01 = +1%
    """
    r = returns.dropna().astype(float).copy()
    r.name = r.name or "returns"
    return r


def performance_metrics(returns: pd.Series, freq: int = 365, rf: float = 0.0) -> pd.Series:
    """
    Métricas (diárias -> anualizadas):
    - Annual Return
    - Annual Volatility
    - Sharpe Ratio (com rf opcional, default 0)
    - Max Drawdown
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return pd.Series(
            {"Annual Return": np.nan, "Annual Volatility": np.nan, "Sharpe Ratio": np.nan, "Max Drawdown": np.nan},
            name="Buy-and-Hold"
        )

    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=1) * np.sqrt(freq)

    # Sharpe: (E[R]-rf)/vol
    excess = ann_ret - rf
    sharpe = np.nan if (ann_vol == 0 or np.isnan(ann_vol)) else (excess / ann_vol)

    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    out = pd.Series(
        {
            "Annual Return": ann_ret,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        },
        name="Buy-and-Hold"
    )
    return out
