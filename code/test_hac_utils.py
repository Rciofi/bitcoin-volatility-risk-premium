"""
test_hac_utils.py — testes de code/hac_utils.py (T3)

Confere:
  1. Convenção: maxlags = h+1 registrado no resultado, use_t = False e
     valor-p pela normal; h inválido é rejeitado.
  2. O EP do BVRP prospectivo (média, h = 30) é o do T2: 1,703, idêntico ao
     valor gravado em outputs/T2/h1_hac.csv (maxlags = 31).
  3. EP calculado à mão, com pesos de Bartlett 1 - j/(L+1) e sem correção de
     amostra pequena, bate com a função:
       a) teste de média (regressão na constante), L = 7 e 90
          (sensibilidade_defasagens) e L = 31 (mqo_newey_west, h = 30), nas
          duas definições do BVRP;
       b) regressão com regressor (ret_fut_30d ~ vrp_30d, h = 30), sanduíche
          (X'X)^-1 S (X'X)^-1 completo.
  4. mqo_newey_west(h) e sensibilidade_defasagens(lags=[h+1]) dão resultados
     idênticos (média e regressão com regressor; h = 1, 30 e 60).
  5. Linhas com NaN são descartadas antes da estimação.

Uso:  python code/test_hac_utils.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hac_utils import maxlags_para_horizonte, mqo_newey_west, sensibilidade_defasagens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-10
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


def cov_hac_manual(X, u, L):
    """(X'X)^-1 S (X'X)^-1, S = sum_j w_j (Gamma_j + Gamma_j'), w_j = 1 - j/(L+1)."""
    xu = X * u[:, None]
    S = xu.T @ xu
    for j in range(1, L + 1):
        g = xu[j:].T @ xu[:-j]
        S += (1 - j / (L + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


df = pd.read_csv(ROOT / "data" / "vrp_with_targets.csv", parse_dates=["date"]).sort_values("date")

# 1) Convenção
print("1) Convenção")
res = mqo_newey_west(df["bvrp_30d_fut"], h=30)
check(res.cov_kwds["maxlags"] == 31, f"h = 30 -> maxlags = {res.cov_kwds['maxlags']} (esperado 31)")
check(res.use_t is False, "use_t = False (inferência pela normal)")
check(abs(res.pvalues.iloc[0] - 2 * stats.norm.sf(abs(res.tvalues.iloc[0]))) < 1e-15,
      "valor-p = 2 * (1 - Phi(|t|))")
for ruim in (-1, 2.5, True):
    try:
        maxlags_para_horizonte(ruim)
        check(False, f"h = {ruim!r} deveria ser rejeitado")
    except ValueError:
        check(True, f"h = {ruim!r} rejeitado")
try:
    sensibilidade_defasagens(df["bvrp_30d_fut"], lags=[7, -3])
    check(False, "lags com valor negativo deveria ser rejeitado")
except ValueError:
    check(True, "lags = [7, -3] rejeitado")

# 2) EP do T2
print("2) EP do BVRP prospectivo (média, h = 30) igual ao do T2")
ep = float(res.bse.iloc[0])
t2 = pd.read_csv(ROOT / "outputs" / "T2" / "h1_hac.csv")
ep_t2 = float(t2[(t2.definicao == "bvrp_30d_fut") & (t2.maxlags == 31)]["ep"].iloc[0])
check(round(ep, 3) == 1.703, f"EP = {ep:.6f} (arredondado: {ep:.3f}; esperado 1,703)")
check(abs(ep - ep_t2) < TOL, f"EP idêntico ao de outputs/T2/h1_hac.csv ({ep_t2:.12f})")

# 3a) Média: manual x função
print("3a) Teste de média: EP manual (Bartlett) x função")
for col in ["bvrp_30d_fut", "vrp_30d"]:
    x = df[col].dropna().to_numpy()
    X = np.ones((len(x), 1))
    u = x - x.mean()
    fits = sensibilidade_defasagens(x, lags=[7, 90])
    fits[31] = mqo_newey_west(x, h=30)
    for L in (7, 31, 90):
        se_man = float(np.sqrt(cov_hac_manual(X, u, L)[0, 0]))
        se_fun = float(fits[L].bse.iloc[0])
        check(abs(se_man - se_fun) < TOL, f"{col}, L = {L}: manual {se_man:.10f} | função {se_fun:.10f}")

# 3b) Regressão com regressor: manual x função
print("3b) Regressão ret_fut_30d ~ vrp_30d (h = 30): sanduíche manual x função")
sub = df[["ret_fut_30d", "vrp_30d"]].dropna()
y, X = sub["ret_fut_30d"].to_numpy(), np.column_stack([np.ones(len(sub)), sub["vrp_30d"].to_numpy()])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
se_man = np.sqrt(np.diag(cov_hac_manual(X, y - X @ beta, 31)))
r = mqo_newey_west(sub["ret_fut_30d"], sub["vrp_30d"], h=30)
check(np.allclose(r.params.to_numpy(), beta, atol=TOL, rtol=0), "coeficientes MQO iguais")
check(np.allclose(r.bse.to_numpy(), se_man, atol=TOL, rtol=0),
      f"EP manual {np.round(se_man, 8).tolist()} | função {np.round(r.bse.to_numpy(), 8).tolist()}")

# 4) Equivalência entre as funções públicas
print("4) mqo_newey_west(h) == sensibilidade_defasagens(lags=[h+1])")
for rotulo, (yy, XX) in {"média (bvrp_30d_fut)": (df["bvrp_30d_fut"], None),
                         "ret_fut_30d ~ vrp_30d": (df["ret_fut_30d"], df["vrp_30d"])}.items():
    for h in (1, 30, 60):
        a = mqo_newey_west(yy, XX, h=h)
        b = sensibilidade_defasagens(yy, XX, lags=[h + 1])[h + 1]
        igual = (np.array_equal(a.params.to_numpy(), b.params.to_numpy())
                 and np.array_equal(a.bse.to_numpy(), b.bse.to_numpy())
                 and np.array_equal(a.pvalues.to_numpy(), b.pvalues.to_numpy()))
        check(igual, f"{rotulo}, h = {h}: coeficientes, EP e valor-p idênticos")

# 5) NaN
print("5) Tratamento de NaN")
yv = df["bvrp_30d_fut"].copy()
yv.iloc[[0, 10]] = np.nan
check(int(mqo_newey_west(yv, h=30).nobs) == len(yv) - 2, "2 linhas com NaN descartadas")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
