"""
hac_utils.py — MQO com erro-padrão de Newey–West (T3 do plano de revisão)

Funções públicas:
  - mqo_newey_west(y, X, h=...): resultado PRINCIPAL. O usuário informa o
    horizonte h e a função aplica sempre maxlags = h+1. É a única que as
    regressões da dissertação (T10, T11) devem usar.
  - sensibilidade_defasagens(y, X, lags=[...]): só para análise de
    sensibilidade ao número de defasagens (ex.: regra automática do pacote,
    janelas maiores). Não substitui o resultado principal.

As duas chamam a mesma função interna, _mqo_hac_bartlett, que fixa a
convenção (não configurável):
  - Núcleo de Bartlett, pesos w_j = 1 - j/(L+1) para j = 0, ..., L, em que
    L = maxlags é a maior defasagem incluída na janela (a defasagem L+1 tem
    peso zero). É a convenção de Newey e West (1987), a mesma do
    `newey, lag(L)` do Stata e do `cov_type="HAC"` do statsmodels.
  - No resultado principal, L = h+1: h defasagens para a sobreposição das
    janelas de h dias e +1 para a autocorrelação/heterocedasticidade
    residual (plano, T3). Por isso "h+1 defasagens" no texto corresponde a
    maxlags = h+1 no código (h = 30 -> maxlags = 31).
  - Sem correção de amostra pequena (use_correction=False, padrão do
    statsmodels para HAC).
  - Inferência pela distribuição normal (use_t=False): valor-p e intervalo
    de confiança usam a normal, não a t de Student.

Referências:
  Newey, W. K.; West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3), 703-708.
  Newey, W. K.; West, K. D. (1994). Automatic lag selection in covariance
    matrix estimation. Review of Economic Studies, 61(4), 631-653.
    (regra automática floor(4(T/100)^(2/9)) do statsmodels; só entra como
    caso de sensibilidade)

Detalhes do statsmodels 0.14.6 em docs/pendencias_T1.md, seção 4.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import weights_bartlett


def _inteiro_nao_negativo(v, nome):
    if isinstance(v, (bool, np.bool_)) or int(v) != v or v < 0:
        raise ValueError(f"{nome} deve ser um inteiro >= 0, recebido {v!r}")
    return int(v)


def maxlags_para_horizonte(h):
    """Número de defasagens do núcleo (maxlags) para o horizonte h: h+1."""
    return _inteiro_nao_negativo(h, "h") + 1


def _preparar(y, X, constante):
    """Alinha y e X, acrescenta a constante e descarta linhas com NaN."""
    y = y.astype(float) if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    if X is None:
        exog = pd.DataFrame({"const": 1.0}, index=y.index)
    else:
        exog = X.to_frame() if isinstance(X, pd.Series) else pd.DataFrame(X)
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            exog.index = y.index
        exog = exog.astype(float)
        if constante:
            exog = sm.add_constant(exog, has_constant="add")
    mask = y.notna() & exog.notna().all(axis=1)
    return y[mask], exog[mask]


def _mqo_hac_bartlett(y, exog, maxlags):
    """Função interna: MQO com HAC na convenção fixa (ver docstring do módulo)."""
    L = _inteiro_nao_negativo(maxlags, "maxlags")
    res = sm.OLS(y, exog).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": L, "weights_func": weights_bartlett, "use_correction": False},
        use_t=False,
    )
    assert res.use_t is False, "convenção: inferência pela normal"
    assert res.cov_kwds["maxlags"] == L
    return res


def mqo_newey_west(y, X=None, *, h, constante=True):
    """
    Resultado PRINCIPAL: MQO com erro-padrão de Newey–West, maxlags = h+1
    (Bartlett, sem correção de amostra pequena, inferência pela normal).

    Parâmetros
    ----------
    y : array-like ou pd.Series
        Variável dependente, em ordem temporal, sem lacunas no calendário.
    X : array-like, pd.Series, pd.DataFrame ou None
        Regressores (sem constante). None = regressão só na constante, cujo
        coeficiente é a média amostral de y (teste de média, H1).
    h : int (obrigatório, keyword)
        Horizonte em dias; a função usa maxlags = h+1.
    constante : bool
        Acrescenta a constante a X (ignorado quando X é None).

    Linhas com NaN em y ou em X são descartadas antes da estimação.

    Retorna
    -------
    statsmodels RegressionResults, com cov_type="HAC" e use_t=False.
    """
    L = maxlags_para_horizonte(h)
    y_, exog = _preparar(y, X, constante)
    return _mqo_hac_bartlett(y_, exog, L)


def sensibilidade_defasagens(y, X=None, *, lags, constante=True):
    """
    SÓ PARA ANÁLISE DE SENSIBILIDADE ao número de defasagens do núcleo. O
    resultado principal da dissertação usa mqo_newey_west (maxlags = h+1);
    esta função existe para reportar, ao lado dele, como o erro-padrão muda
    com outros valores de L (ex.: 7, a regra automática do pacote; 60 e 90).

    Mesma convenção e mesmo tratamento de y/X de mqo_newey_west.

    Parâmetros
    ----------
    lags : iterável de int (obrigatório, keyword)
        Valores de maxlags (L, maior defasagem incluída) a estimar.

    Retorna
    -------
    dict {L: statsmodels RegressionResults}, na ordem de `lags`.
    """
    y_, exog = _preparar(y, X, constante)
    return {int(L): _mqo_hac_bartlett(y_, exog, L) for L in lags}
