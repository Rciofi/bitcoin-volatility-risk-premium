"""
wald_utils.py — Wald HAC em numpy, rotação, Freedman–Lane, bootstrap de pares e Holm (B2)

Reúne as funções dos diagnósticos do B2 (code/diagnostico_b2/), hoje duplicadas em
bootstrap_wald_b2.py (hac_wald) e mc_rotacao_poder_b2.py (wald), para os scripts oficiais
dos Caps. 6 e 7. Os resultados são os mesmos dos diagnósticos (test_wald_utils.py confere).

Convenções (as de hac_utils): núcleo de Bartlett com pesos 1 − l/(L + 1), l = 1..L, sem
correção de amostra pequena; W = d' V⁻¹ d com V = (Z'Z)⁻¹ S (Z'Z)⁻¹ restrita às
inclinações testadas; W = nan se Z'Z ou V for singular.

  hac_wald(y, Z, Ls, ZZi=None, b0=None)          W de todas as inclinações, para cada L
  hac_wald_sub(y, Z, Ls, subs, ZZi=None, b0=None) W de cada subconjunto de colunas
  somas_h(eps, n, h), retornos_h(r, datas, h)   y_t = Σ eps(t+1..t+h) a partir dos diários
  idx_circ(n, bloco, rng)                        índices de blocos circulares
  rotacoes(y, Z, Ls, ks, ZZi=None)               W de roll(y, k), K × L (em paralelo)
  rotacoes_freedman_lane(y, Z, Ls, idx, ks)      W da hipótese idx com y* = ajustado restrito
                                                 + roll(resíduo restrito, k), K × L
  bootstrap_pares(y, Z, Ls, bloco, B, chave)     W* de pares (Z, y) em blocos circulares,
                                                 centrado em β̂; B × L
  p_mc(W0, Wk)                                   (1 + #{Wk >= W0}) / (K_válidas + 1)
  postos(W0, Wk)                                 p de cada elemento no conjunto {W0} ∪ {Wk}
  min_p(Wobs, Wk)                                min_h p_h com o mesmo k em todos os h
  holm(p)                                        p ajustados de Holm
  hac_wald_lin(y, Z, Ls, Rs, ZZi=None, b0=None)  W de H0: R b = R b0 (ou 0), restrições lineares
  rotacoes_freedman_lane_lin(y, Z, Ls, R, ks)    Freedman–Lane com restrito Z·N, N = base do núcleo de R
  bootstrap_pares_lin(y, Z, Ls, Rs, bloco, B, chave)  W* das restrições R, mesmas sementes de bootstrap_pares
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import null_space

SEMENTE = 20261020
N_PEDACOS = 32                   # nº fixo de pedaços paralelos: o resultado não depende do nº de núcleos


# ---------------------------------------------------------------------------
# Wald HAC
# ---------------------------------------------------------------------------
def inv_ou_none(Z):
    try:
        return np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None


def _hac(y, Z, Ls, ZZi):
    """b e, para cada L, V = (Z'Z)⁻¹ S_L (Z'Z)⁻¹."""
    b = ZZi @ (Z.T @ y)
    s = Z * (y - Z @ b)[:, None]
    S0 = s.T @ s
    G = [s[l:].T @ s[:-l] for l in range(1, max(Ls) + 1)]
    Vs = []
    for L in Ls:
        S = S0.copy()
        for l in range(1, L + 1):
            S += (1 - l / (L + 1)) * (G[l - 1] + G[l - 1].T)
        Vs.append(ZZi @ S @ ZZi)
    return b, Vs


def _forma(d, V):
    try:
        w = float(d @ np.linalg.solve(V, d))
    except np.linalg.LinAlgError:
        return np.nan
    return w if np.isfinite(w) else np.nan


def hac_wald_sub(y, Z, Ls, subs, ZZi=None, b0=None):
    """W de H0: b[idx] = b0[idx] (ou 0), para cada idx de subs (dict) e cada L. Retorna (subs × L, b)."""
    if ZZi is None:
        ZZi = inv_ou_none(Z)
        if ZZi is None:
            return np.full((len(subs), len(Ls)), np.nan), None
    b, Vs = _hac(y, Z, Ls, ZZi)
    d = b - (0.0 if b0 is None else b0)
    W = np.array([[_forma(d[idx], V[np.ix_(idx, idx)]) for V in Vs] for idx in subs.values()])
    return W, b


def hac_wald(y, Z, Ls, ZZi=None, b0=None):
    """W de todas as inclinações (colunas 1..k−1 de Z), para cada L. Retorna (W, b)."""
    W, b = hac_wald_sub(y, Z, Ls, {"todas": list(range(1, Z.shape[1]))}, ZZi, b0)
    return W[0], b


# ---------------------------------------------------------------------------
# Retornos de h dias e blocos circulares
# ---------------------------------------------------------------------------
def retornos_h(r, datas, h):
    """Diários das datas d0 .. d_fim + h (comprimento n + h), para gerar y_t = Σ r(t+1..t+h)."""
    import pandas as pd
    rc = r.loc[datas[0]: datas[-1] + pd.Timedelta(days=h)].to_numpy()
    assert len(rc) == len(datas) + h
    return rc


def somas_h(eps, n, h):
    c = np.concatenate([[0.0], np.cumsum(eps)])
    t = np.arange(n)
    return c[t + h + 1] - c[t + 1]


def idx_circ(n, bloco, rng):
    ini = rng.integers(0, n, size=-(-n // bloco))
    return ((ini[:, None] + np.arange(bloco)).ravel()[:n]) % n


def dividir(total, partes=N_PEDACOS):
    base = [total // partes + (1 if i < total % partes else 0) for i in range(partes)]
    return [x for x in base if x > 0]


def _pedacos(ks):
    return [c for c in np.array_split(np.asarray(ks), N_PEDACOS) if len(c)]


# ---------------------------------------------------------------------------
# Rotação, Freedman–Lane e bootstrap de pares
# ---------------------------------------------------------------------------
def _rot(y, Z, ZZi, Ls, ks):
    return np.array([hac_wald(np.roll(y, k), Z, Ls, ZZi)[0] for k in ks])


def rotacoes(y, Z, Ls, ks, ZZi=None, n_jobs=-1):
    """W de roll(y, k) para cada k de ks: K × L."""
    ZZi = np.linalg.inv(Z.T @ Z) if ZZi is None else ZZi
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_rot)(y, Z, ZZi, Ls, p) for p in _pedacos(ks)))


def _rot_fl(ajustado, resid, Z, ZZi, Ls, idx, ks):
    return np.array([hac_wald_sub(ajustado + np.roll(resid, k), Z, Ls, {"h": idx}, ZZi)[0][0] for k in ks])


def rotacoes_freedman_lane(y, Z, Ls, idx, ks, ZZi=None, n_jobs=-1):
    """
    W da hipótese b[idx] = 0 com y* = ajustado restrito + roll(resíduo restrito, k): K × L. O modelo
    restrito é Z sem as colunas idx; com idx = todas as inclinações, é a rotação de y.
    """
    ZZi = np.linalg.inv(Z.T @ Z) if ZZi is None else ZZi
    Zr = np.delete(Z, idx, axis=1)
    ajustado = Zr @ np.linalg.lstsq(Zr, y, rcond=None)[0]
    resid = y - ajustado
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_rot_fl)(ajustado, resid, Z, ZZi, Ls, idx, p)
                                                  for p in _pedacos(ks)))


def _boot(y, Z, Ls, bloco, b_hat, reps, semente):
    rng = np.random.default_rng(semente)
    return np.array([hac_wald(y[j], Z[j], Ls, b0=b_hat)[0]
                     for j in (idx_circ(len(y), bloco, rng) for _ in range(reps))])


def bootstrap_pares(y, Z, Ls, bloco, B, chave, semente=SEMENTE, n_jobs=-1):
    """W observado e W* (B × L) de pares (Z, y) em blocos circulares, centrado em β̂.
    Sementes default_rng([semente, *chave, pedaço]), 32 pedaços (= bootstrap_wald_b2.wald_bootstrap)."""
    W, b_hat = hac_wald(y, Z, Ls)
    res = Parallel(n_jobs=n_jobs)(delayed(_boot)(y, Z, Ls, bloco, b_hat, reps, [semente, *chave, k])
                                  for k, reps in enumerate(dividir(B)))
    return W, np.vstack(res)


# ---------------------------------------------------------------------------
# Valores-p
# ---------------------------------------------------------------------------
def p_mc(W0, Wk):
    """(1 + #{Wk >= W0}) / (K_válidas + 1) e nº de Wk não finitos."""
    v = Wk[np.isfinite(Wk)]
    return float((1 + np.sum(v >= W0)) / (len(v) + 1)), int(len(Wk) - len(v))


def postos(W0, Wk):
    """p do observado e de cada Wk no conjunto {W0} ∪ {Wk}: #{s >= x} / (K + 1)."""
    S = np.concatenate([[W0], Wk])
    p = (len(S) - np.searchsorted(np.sort(S), S, side="left")) / len(S)
    return p[0], p[1:]


def min_p(Wobs, Wk):
    """
    Wobs: (H,), Wk: (K, H), o mesmo k em todas as colunas, só com valores finitos.
    T = min_h p_h; p = (1 + #{T(k) <= T}) / (K + 1). Retorna (p, T, p_h observados).
    """
    pp = [postos(Wobs[j], Wk[:, j]) for j in range(len(Wobs))]
    pobs = np.array([a for a, _ in pp])
    Tk = np.column_stack([b for _, b in pp]).min(axis=1)
    T = pobs.min()
    return float((1 + np.sum(Tk <= T)) / (len(Tk) + 1)), float(T), pobs


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[o]), 1.0)
    out = np.empty(m)
    out[o] = adj
    return out


# ---------------------------------------------------------------------------
# Restrições lineares (H0: R b = 0)
# ---------------------------------------------------------------------------
def _matriz(R):
    return np.atleast_2d(np.asarray(R, float))


def hac_wald_lin(y, Z, Ls, Rs, ZZi=None, b0=None):
    """
    W de H0: R (b − b0) = 0 (b0 = 0 se None) para cada matriz R de Rs (dict nome -> R, q × k) e cada L:
    W = d' (R V R')⁻¹ d, d = R (b − b0). Com R que seleciona colunas, é igual a hac_wald_sub.
    Retorna (len(Rs) × L, b).
    """
    if ZZi is None:
        ZZi = inv_ou_none(Z)
        if ZZi is None:
            return np.full((len(Rs), len(Ls)), np.nan), None
    b, Vs = _hac(y, Z, Ls, ZZi)
    d = b - (0.0 if b0 is None else b0)
    W = np.array([[_forma(R @ d, R @ V @ R.T) for V in Vs] for R in map(_matriz, Rs.values())])
    return W, b


def _rot_fl_lin(ajustado, resid, Z, ZZi, Ls, R, ks):
    return np.array([hac_wald_lin(ajustado + np.roll(resid, k), Z, Ls, {"h": R}, ZZi)[0][0] for k in ks])


def rotacoes_freedman_lane_lin(y, Z, Ls, R, ks, ZZi=None, n_jobs=-1):
    """
    Freedman–Lane para H0: R b = 0, restrição linear homogênea: o modelo restrito é Z·N, com N uma base
    do núcleo de R (b = N c). y* = ajustado restrito + roll(resíduo restrito, k); W da restrição na
    regressão completa. Com R que seleciona colunas, é igual a rotacoes_freedman_lane. K × L.
    """
    ZZi = np.linalg.inv(Z.T @ Z) if ZZi is None else ZZi
    R = _matriz(R)
    Zr = Z @ null_space(R)
    ajustado = Zr @ np.linalg.lstsq(Zr, y, rcond=None)[0]
    resid = y - ajustado
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_rot_fl_lin)(ajustado, resid, Z, ZZi, Ls, R, p)
                                                  for p in _pedacos(ks)))


def _boot_lin(y, Z, Ls, Rs, bloco, b_hat, reps, semente):
    rng = np.random.default_rng(semente)
    return np.array([hac_wald_lin(y[j], Z[j], Ls, Rs, b0=b_hat)[0]
                     for j in (idx_circ(len(y), bloco, rng) for _ in range(reps))])


def bootstrap_pares_lin(y, Z, Ls, Rs, bloco, B, chave, semente=SEMENTE, n_jobs=-1):
    """W (len(Rs) × L) e W* (B × len(Rs) × L) de pares (Z, y) em blocos circulares, centrado em β̂.
    Mesmas sementes e pedaços de bootstrap_pares: os índices sorteados não dependem de Rs."""
    W, b_hat = hac_wald_lin(y, Z, Ls, Rs)
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_lin)(y, Z, Ls, Rs, bloco, b_hat, reps, [semente, *chave, k])
                                  for k, reps in enumerate(dividir(B)))
    return W, np.concatenate(res)
