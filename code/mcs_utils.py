"""
mcs_utils.py — model confidence set (Hansen, Lunde e Nason, 2011), seção 25 das pendências

  - indices_blocos: blocos móveis não circulares (cópia de retorno_bvrp_T10.py).
  - sortear_indices: B reamostragens em blocos, semente default_rng([SEMENTE, bloco, r]).
  - medias_bootstrap: perdas médias de cada modelo em cada reamostragem.
  - mcs: eliminação sequencial com T_max ou T_R; valores-p do MCS.
  - conjunto: modelos com valor-p do MCS >= alpha.

Procedimento (HLN, 2011, seções 3.1.2 e 3.2). Perdas L (n x m); L-barra_i é a média
de cada modelo e L-barra*_{b,i} a média na reamostragem b; zeta*_{b,i} = L-barra*_{b,i} - L-barra_i.
Os índices de cada reamostragem são os mesmos para todos os modelos (preserva a correlação
entre as perdas). As variâncias vêm das mesmas reamostragens:
  T_max: d_i = L-barra_i - média_{k em M} L-barra_k; var_i = média_b (zeta*_{b,i} - média_{k em M} zeta*_{b,k})^2;
         t_i = d_i / raiz(var_i); T = max_i t_i; elimina argmax_i t_i.
  T_R:   d_ij = L-barra_i - L-barra_j; var_ij = média_b (zeta*_{b,i} - zeta*_{b,j})^2;
         T = max_{i,j} |t_ij|; elimina argmax_i max_j t_ij.
Valor-p do passo: fração das reamostragens com T* >= T (T* com as mesmas variâncias e
zeta* centrados). Valor-p do MCS: máximo acumulado dos valores-p dos passos; o último
modelo recebe 1. O MCS a alpha reúne os modelos com valor-p >= alpha.

Casos degenerados:
  - colunas com perdas idênticas em todas as datas entram como um só modelo e recebem o
    mesmo valor-p (sem isso, o empate seria resolvido pela ordem das colunas);
  - variância <= TOL_VAR x escala: a estatística vale 0 se a diferença média também é
    ~0 e +-inf se não é; no bootstrap, a contribuição é 0. Não há divisão por zero.

Só numpy e pandas; o pacote arch não é usado. As perdas são passadas prontas (no M2,
erro quadrático).
"""

import numpy as np
import pandas as pd

SEMENTE = 20261001   # a mesma do T10 (retorno_bvrp_T10.SEMENTE)
TOL_VAR = 1e-12      # variância relativa abaixo da qual a diferença é tratada como nula


# Copiada de code/retorno_bvrp_T10.py (indices_blocos), sem alteração; a equivalência é
# conferida em test_mcs_utils.py. Não importada para não carregar os efeitos de importação
# daquele script (codificação do console, backend do matplotlib, scikit-learn e xgboost).
def indices_blocos(n, b, rng):
    """Blocos móveis (não circulares) de tamanho b concatenados até n posições (0 .. n-1)."""
    inicios = rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))
    return np.concatenate([np.arange(s, s + b) for s in inicios])[:n]


def sortear_indices(n, *, bloco, B, semente=SEMENTE):
    """Matriz B x n de índices; a reamostragem r usa default_rng([semente, bloco, r])."""
    return np.vstack([indices_blocos(n, bloco, np.random.default_rng([semente, bloco, r]))
                      for r in range(B)])


def medias_bootstrap(perdas, indices):
    """Matriz B x m das perdas médias de cada modelo em cada reamostragem."""
    L = np.asarray(perdas, dtype=float)
    return np.vstack([L[idx].mean(axis=0) for idx in indices])


def _t(d, var, escala):
    """d / raiz(var), com var ~ 0 tratada sem divisão: 0 se d ~ 0, +-inf se não."""
    nula = var <= TOL_VAR * escala
    t = np.zeros_like(d)
    np.divide(d, np.sqrt(var), out=t, where=~nula)
    t[nula & (np.abs(d) > np.sqrt(TOL_VAR * escala))] = np.inf
    t[nula] *= np.sign(d[nula])
    return t, nula


def _passo(Lbar, zeta, ativos, estatistica, escala):
    """Estatística, valor-p e posição (em `ativos`) do modelo a eliminar."""
    Lb, z = Lbar[ativos], zeta[:, ativos]
    if estatistica == "max":
        d = Lb - Lb.mean()
        zc = z - z.mean(axis=1, keepdims=True)
        t, nula = _t(d, (zc ** 2).mean(axis=0), escala)
        t_boot = np.where(nula, 0.0, zc / np.sqrt(np.where(nula, 1.0, (zc ** 2).mean(axis=0))))
        T, T_boot, elimina = t.max(), t_boot.max(axis=1), int(np.argmax(t))
    elif estatistica == "R":
        d = Lb[:, None] - Lb[None, :]
        dz = z[:, :, None] - z[:, None, :]
        var = (dz ** 2).mean(axis=0)
        t, nula = _t(d, var, escala)
        t_boot = np.where(nula, 0.0, np.abs(dz) / np.sqrt(np.where(nula, 1.0, var)))
        T, T_boot = np.abs(t).max(), t_boot.max(axis=(1, 2))
        elimina = int(np.argmax(t.max(axis=1)))
    else:
        raise ValueError("estatistica deve ser 'max' ou 'R'")
    return float(T), float(np.mean(T_boot >= T)), elimina


def mcs(perdas, *, medias_boot, estatistica="max"):
    """
    MCS sobre as colunas de `perdas` (DataFrame n x m, sem NaN), com as médias bootstrap de
    medias_bootstrap(perdas, indices). Retorna um DataFrame por modelo: perda média,
    ordem de eliminação (o último fica com a maior), estatística e valor-p do passo em
    que saiu, valor-p do MCS e o grupo de perdas idênticas.
    """
    if not isinstance(perdas, pd.DataFrame):
        raise TypeError("perdas deve ser um DataFrame com um modelo por coluna")
    if estatistica not in ("max", "R"):
        raise ValueError("estatistica deve ser 'max' ou 'R'")
    if perdas.isna().any().any():
        raise ValueError("perdas com NaN")
    L = perdas.to_numpy(dtype=float)
    Lbar = L.mean(axis=0)
    medias_boot = np.asarray(medias_boot, dtype=float)
    if medias_boot.ndim != 2 or medias_boot.shape[1] != L.shape[1]:
        raise ValueError("medias_boot não tem uma coluna por modelo")
    zeta = medias_boot - Lbar
    escala = float(np.mean(L ** 2)) / len(L)   # ordem de grandeza da variância de uma média

    # grupos de colunas idênticas: um representante por grupo
    grupo, reps = np.empty(L.shape[1], dtype=int), []
    for j in range(L.shape[1]):
        igual = [g for g, r in enumerate(reps) if np.array_equal(L[:, j], L[:, r])]
        if igual:
            grupo[j] = igual[0]
        else:
            grupo[j] = len(reps)
            reps.append(j)

    ativos, saida, p_acum = list(reps), {}, 0.0
    for ordem in range(1, len(reps)):
        T, p, k = _passo(Lbar, zeta, ativos, estatistica, escala)
        p_acum = max(p_acum, p)
        saida[ativos[k]] = (ordem, T, p, p_acum)
        ativos.pop(k)
    saida[ativos[0]] = (len(reps), np.nan, np.nan, 1.0)

    linhas = []
    for j, nome in enumerate(perdas.columns):
        ordem, T, p, p_mcs = saida[reps[grupo[j]]]
        linhas.append({"modelo": nome, "perda_media": Lbar[j], "ordem_eliminacao": ordem,
                       "estatistica_passo": T, "p_passo": p, "p_mcs": p_mcs,
                       "grupo_identico": int(grupo[j])})
    return pd.DataFrame(linhas).sort_values("ordem_eliminacao", kind="stable").reset_index(drop=True)


def conjunto(resultado, alpha):
    """Modelos do MCS a alpha: valor-p do MCS >= alpha."""
    return resultado.loc[resultado["p_mcs"] >= alpha, "modelo"].tolist()
