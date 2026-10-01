"""
avaliacao_utils.py — métricas de previsão fora da amostra (T5 do plano de revisão)

  - r2_fora_da_amostra: 1 - SSE(modelo) / SSE(referência) (Campbell e
    Thompson, 2008, quando a referência é a média histórica).
  - clark_west: teste de Clark e West (2007) de igualdade de erro quadrático
    médio para modelos aninhados, H1 unilateral (o modelo é melhor que a
    referência).
  - diebold_mariano: teste de Diebold e Mariano (1995), bilateral, com perda
    quadrática.

Os dois testes regridem a série de diferenças de perda numa constante com
erro-padrão de Newey–West via hac_utils.mqo_newey_west(h=h), ou seja,
maxlags = h+1 (Bartlett, sem correção de amostra pequena, inferência pela
normal). Com alvos de h = 30 dias sobrepostos, isso corrige a
autocorrelação das perdas; ainda assim, 987 previsões equivalem a ~33
observações independentes, e os testes têm pouco poder.
"""

import numpy as np
from scipy import stats

from hac_utils import mqo_newey_west


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def r2_fora_da_amostra(y, prev, prev_ref):
    """1 - sum((y - prev)^2) / sum((y - prev_ref)^2)."""
    y, prev, prev_ref = _arr(y, prev, prev_ref)
    return float(1 - np.sum((y - prev) ** 2) / np.sum((y - prev_ref) ** 2))


def clark_west(y, prev_modelo, prev_ref, *, h):
    """
    f = (y - ref)^2 - [(y - modelo)^2 - (ref - modelo)^2]; regressão de f na
    constante com Newey–West (maxlags = h+1). Retorna a média de f, a
    estatística t e o valor-p unilateral P(Z > t).
    """
    y, pm, pr = _arr(y, prev_modelo, prev_ref)
    f = (y - pr) ** 2 - ((y - pm) ** 2 - (pr - pm) ** 2)
    res = mqo_newey_west(f, h=h)
    t = float(res.tvalues.iloc[0])
    return {"cw_media": float(f.mean()), "cw_t": t, "cw_p_unilateral": float(stats.norm.sf(t))}


def diebold_mariano(y, prev_1, prev_2, *, h):
    """
    d = (y - prev_1)^2 - (y - prev_2)^2; regressão de d na constante com
    Newey–West (maxlags = h+1). d > 0 em média: o modelo 2 erra menos.
    Retorna a média de d, a estatística t e o valor-p bilateral.
    """
    y, p1, p2 = _arr(y, prev_1, prev_2)
    d = (y - p1) ** 2 - (y - p2) ** 2
    res = mqo_newey_west(d, h=h)
    t = float(res.tvalues.iloc[0])
    return {"dm_media": float(d.mean()), "dm_t": t, "dm_p_bilateral": float(2 * stats.norm.sf(abs(t)))}
