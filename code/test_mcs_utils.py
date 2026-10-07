"""
test_mcs_utils.py — testes de code/mcs_utils.py (M1, seção 25 das pendências)

Confere:
  1. indices_blocos é idêntica à de retorno_bvrp_T10.py: mesmos índices para a mesma
     semente default_rng([20261001, b, r]) em quatro combinações de n e b
     (987/60, 987/30, 987/90, 100/7), r = 0..4; SEMENTE igual à do T10.
  2. Modelo claramente pior (perda + 5) é o primeiro eliminado, com valor-p ~ 0, e fica
     fora do MCS a 10% e a 25%, com T_max e T_R.
  3. Perdas idênticas: duas cópias da mesma coluna ficam no mesmo grupo e com o mesmo
     valor-p do MCS; perdas iguais em parte das datas (como LASSO e média histórica)
     não geram NaN, aviso nem divisão por zero.
  4. Estabilidade: a mesma semente reproduz o resultado exatamente; outra semente
     (B = 1.999) dá o mesmo MCS a 10% e valores-p com diferença < 0,03.
  5. Caso pequeno à mão (n = 6, m = 3, B = 4 índices fixados, sem empate nas perdas
     médias): ordem de eliminação, estatística e valor-p de cada passo e valor-p do MCS
     refeitos com laços explícitos, direto das fórmulas de HLN; pelo menos um valor-p
     intermediário (nem 0 nem 1).
  6. Com dois modelos, T_max = T_R (d_i = +-d_12/2 e var_i = var_12/4) e os valores-p
     coincidem.
  7. Entradas inválidas: NaN, estatística desconhecida e medias_boot com colunas a menos.

Uso:  python code/test_mcs_utils.py   (sai com código 1 se algo falhar)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcs_utils as M  # noqa: E402

TOL = 1e-12
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


def rodar(perdas, bloco, B, semente=M.SEMENTE, estatistica="max"):
    idx = M.sortear_indices(len(perdas), bloco=bloco, B=B, semente=semente)
    return M.mcs(perdas, medias_boot=M.medias_bootstrap(perdas, idx), estatistica=estatistica)


def perdas_simuladas(n=600, semente=7):
    """Três modelos com erros AR(1) correlacionados; perdas quadráticas."""
    rng = np.random.default_rng(semente)
    e = np.zeros((n, 3))
    choque = rng.standard_normal((n, 3)) @ np.array([[1, .6, .6], [0, .8, .3], [0, 0, .74]])
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + choque[t]
    return pd.DataFrame(e ** 2, columns=["A", "B", "C"])


# 1) Equivalência com o T10 (a importação do T10 só ocorre neste teste)
print("1) indices_blocos idêntica à do T10")
import retorno_bvrp_T10 as T10  # noqa: E402
check(M.SEMENTE == T10.SEMENTE, f"SEMENTE = {M.SEMENTE} (T10: {T10.SEMENTE})")
for n, b in [(987, 60), (987, 30), (987, 90), (100, 7)]:
    iguais = all(np.array_equal(M.indices_blocos(n, b, np.random.default_rng([M.SEMENTE, b, r])),
                                T10.indices_blocos(n, b, np.random.default_rng([M.SEMENTE, b, r])))
                 for r in range(5))
    check(iguais, f"n = {n}, b = {b}: mesmos índices em r = 0..4")

# 2) Modelo claramente pior
print("2) Modelo claramente pior é eliminado")
base = perdas_simuladas()
pior = base.assign(D=base["A"] + 5.0)
for est in ("max", "R"):
    r = rodar(pior, bloco=30, B=999, estatistica=est)
    primeiro = r.iloc[0]
    check(primeiro.modelo == "D" and primeiro.p_mcs < 0.01,
          f"T_{est}: D sai primeiro (p_MCS = {primeiro.p_mcs:.4f})")
    check("D" not in M.conjunto(r, 0.10) and "D" not in M.conjunto(r, 0.25), f"T_{est}: D fora a 10% e 25%")

# 3) Perdas idênticas e variância zero
print("3) Perdas idênticas e iguais em parte das datas")
copia = base.assign(A2=base["A"])
parcial = base["A"].copy()
parcial.iloc[:300] = base["B"].iloc[:300]          # igual a B na primeira metade
for est in ("max", "R"):
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise"):
        warnings.simplefilter("error")
        r = rodar(copia, bloco=30, B=499, estatistica=est).set_index("modelo")
        rp = rodar(base.assign(P=parcial), bloco=30, B=499, estatistica=est)
        tudo_igual = pd.DataFrame({"X": base["A"], "Y": base["A"]})
        ri = rodar(tudo_igual, bloco=30, B=99, estatistica=est)
    check(r.loc["A", "grupo_identico"] == r.loc["A2", "grupo_identico"]
          and r.loc["A", "p_mcs"] == r.loc["A2", "p_mcs"],
          f"T_{est}: A e A2 no mesmo grupo, p_MCS = {r.loc['A', 'p_mcs']:.4f}")
    check(not rp[["p_passo", "p_mcs"]].iloc[:-1].isna().any().any(),
          f"T_{est}: parcialmente iguais sem NaN, aviso ou divisão por zero")
    check((ri["p_mcs"] == 1.0).all(), f"T_{est}: só cópias -> um grupo, p_MCS = 1")

# 4) Estabilidade entre sementes
print("4) Estabilidade entre sementes")
for est in ("max", "R"):
    a = rodar(base, bloco=30, B=1999, estatistica=est)
    a2 = rodar(base, bloco=30, B=1999, estatistica=est)
    b_ = rodar(base, bloco=30, B=1999, semente=12345, estatistica=est)
    check(a.equals(a2), f"T_{est}: mesma semente -> resultado idêntico")
    pa, pb = a.set_index("modelo")["p_mcs"], b_.set_index("modelo")["p_mcs"]
    dif = (pa - pb.reindex(pa.index)).abs().max()
    check(set(M.conjunto(a, 0.10)) == set(M.conjunto(b_, 0.10)),
          f"T_{est}: mesmo MCS a 10% ({sorted(M.conjunto(a, 0.10))})")
    check(dif < 0.03, f"T_{est}: diferença máxima de p_MCS = {dif:.4f}")

# 5) Caso pequeno à mão
print("5) Caso pequeno (n = 6, m = 3, B = 4): laços explícitos")
Lp = pd.DataFrame({"a": [1.0, 2, 3, 2, 1, 2], "b": [2.0, 3, 3, 4, 2, 3], "c": [1.0, 1, 4, 2, 2, 2]})
idx = np.array([[0, 1, 2, 3, 4, 5], [2, 3, 2, 3, 4, 5], [0, 1, 0, 1, 4, 5], [3, 4, 5, 3, 4, 5]])
mb = M.medias_bootstrap(Lp, idx)
Lv = Lp.to_numpy()
mb_mao = np.array([[sum(Lv[i, j] for i in linha) / 6 for j in range(3)] for linha in idx])
check(np.allclose(mb, mb_mao, atol=TOL, rtol=0), "médias bootstrap iguais às do laço")


def mcs_mao(est):
    """Fórmulas de HLN com laços explícitos, sobre as médias bootstrap já conferidas (mb)."""
    Lbar = [sum(Lv[:, j]) / 6 for j in range(3)]
    zeta = [[mb[b][j] - Lbar[j] for j in range(3)] for b in range(4)]
    ativos, ordem, p_acum = [0, 1, 2], [], 0.0
    while len(ativos) > 1:
        k = len(ativos)
        if est == "max":
            mL = sum(Lbar[i] for i in ativos) / k
            var = {i: sum((zeta[b][i] - sum(zeta[b][j] for j in ativos) / k) ** 2 for b in range(4)) / 4
                   for i in ativos}
            t = {i: (Lbar[i] - mL) / var[i] ** 0.5 for i in ativos}
            T = max(t.values())
            Tb = [max((zeta[b][i] - sum(zeta[b][j] for j in ativos) / k) / var[i] ** 0.5 for i in ativos)
                  for b in range(4)]
            e = max(ativos, key=lambda i: t[i])
        else:
            var = {(i, j): sum((zeta[b][i] - zeta[b][j]) ** 2 for b in range(4)) / 4
                   for i in ativos for j in ativos if i != j}
            t = {ij: (Lbar[ij[0]] - Lbar[ij[1]]) / v ** 0.5 for ij, v in var.items()}
            T = max(abs(x) for x in t.values())
            Tb = [max(abs(zeta[b][i] - zeta[b][j]) / var[(i, j)] ** 0.5 for (i, j) in var) for b in range(4)]
            e = max(ativos, key=lambda i: max(t[(i, j)] for j in ativos if j != i))
        p = sum(x >= T for x in Tb) / 4
        p_acum = max(p_acum, p)
        ordem.append((Lp.columns[e], T, p, p_acum))
        ativos.remove(e)
    ordem.append((Lp.columns[ativos[0]], np.nan, np.nan, 1.0))
    return ordem


for est in ("max", "R"):
    esperado = mcs_mao(est)
    r = M.mcs(Lp, medias_boot=mb, estatistica=est)
    check(r.modelo.tolist() == [e[0] for e in esperado], f"T_{est}: ordem {r.modelo.tolist()}")
    check(np.allclose(r.estatistica_passo, [e[1] for e in esperado], atol=TOL, rtol=0, equal_nan=True),
          f"T_{est}: estatística de cada passo {np.round(r.estatistica_passo.to_numpy(), 6).tolist()}")
    check(np.allclose(r.p_passo, [e[2] for e in esperado], atol=TOL, rtol=0, equal_nan=True)
          and np.allclose(r.p_mcs, [e[3] for e in esperado], atol=TOL, rtol=0),
          f"T_{est}: p do passo {r.p_passo.tolist()} e p_MCS {r.p_mcs.tolist()} iguais aos do laço")
    check(((r.p_mcs > 0) & (r.p_mcs < 1)).any(), f"T_{est}: há valor-p intermediário (nem 0 nem 1)")

# 6) Dois modelos: T_max = T_R
print("6) Dois modelos: T_max = T_R")
dois = base[["A", "B"]]
rm, rr = rodar(dois, bloco=30, B=999, estatistica="max"), rodar(dois, bloco=30, B=999, estatistica="R")
check(abs(rm.estatistica_passo.iloc[0] - rr.estatistica_passo.iloc[0]) < 1e-9
      and rm.p_passo.iloc[0] == rr.p_passo.iloc[0] and rm.modelo.iloc[0] == rr.modelo.iloc[0],
      f"T_max = {rm.estatistica_passo.iloc[0]:.6f}, T_R = {rr.estatistica_passo.iloc[0]:.6f}; "
      f"mesmo p ({rm.p_passo.iloc[0]:.4f}) e mesmo eliminado ({rm.modelo.iloc[0]})")

# 7) Entradas inválidas
print("7) Entradas inválidas")
for rotulo, f in {"NaN": lambda: M.mcs(base.mask(base.index == 0), medias_boot=np.zeros((5, 3))),
                  "estatística": lambda: M.mcs(base, medias_boot=np.zeros((5, 3)), estatistica="x"),
                  "colunas": lambda: M.mcs(base, medias_boot=np.zeros((5, 2)))}.items():
    try:
        f()
        check(False, f"{rotulo}: deveria ser rejeitado")
    except (ValueError, TypeError):
        check(True, f"{rotulo}: rejeitado")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
