"""
test_wald_utils.py — testes de code/wald_utils.py (B2)

Confere:
  1. hac_wald × hac_utils (statsmodels): R_log de 30 dias nos 16 previsores (amostra completa,
     L = 31 e 61; W = 20,008 do passo 1) e no BVRP previsto do Ridge (987 datas, L = 2, 31, 121).
  2. hac_wald_sub: todas as inclinações = hac_wald; uma inclinação = t² do statsmodels; o p
     da interação e do teste conjunto do regime (R_simples, h = 30, corte fixo) = regime_T10.csv.
  3. somas_h × soma direta (h = 1, 30, 60); retornos_h dá R_log = 100·log1p(ret_fut_hd);
     idx_circ: n índices, blocos contíguos módulo n.
  4. Reproduz os diagnósticos já commitados (ca7307c): p da rotação do passo 3 (g = 90,
     L = 31: 0,4218 na amostra completa e 0,1953 nas 987 datas) e p do bootstrap do passo 2
     (bloco 60, chave [1, 0, 60]: 0,5895).
  5. Freedman–Lane com idx = todas as inclinações = rotação de y (mesmos W).
  6. p_mc ignora não finitos; postos dá o p_mc do observado e trata empates; min_p num
     exemplo calculado à mão; holm = statsmodels.multipletests(method = "holm").
  7. Restrições lineares: hac_wald_lin com R de seleção = hac_wald_sub; β + γ = 0 = t² do
     statsmodels (t_test); Freedman–Lane linear com R de seleção = rotacoes_freedman_lane e, em
     β + γ = 0, = o modelo restrito reparametrizado à mão; bootstrap_pares_lin com R de todas as
     inclinações = bootstrap_pares (mesmos W*).

Uso:  python code/test_wald_utils.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wald_utils as wu  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASSO2 = ROOT / "outputs" / "diagnostico_b2" / "20261020_202703_passo2"
PASSO3 = ROOT / "outputs" / "diagnostico_b2" / "20261020_203512_passo3"
TOL = 1e-10
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def w_sm(res, q):
    return float(np.squeeze(res.wald_test(np.eye(q + 1)[1:], use_f=False, scalar=True).statistic))


# Dados (como bootstrap_wald_b2.carregar)
df = pd.read_csv(ROOT / "data" / "ml_dataset_T4.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
feats = [c for c in df.columns if c not in ("date", "alvo_bvrp_30d_fut", "vh_30d")]
vt = pd.read_csv(ROOT / "data" / "vrp_with_targets.csv", parse_dates=["date"])
reg = pd.read_csv(ROOT / "data" / "regimes_T9.csv", parse_dates=["date"])
df = (df.merge(vt[["date", "ret_fut_30d", "ret_fut_60d", "ret_fut_1d"]], on="date")
        .merge(reg[["date", "regime_alta_fixo"]], on="date"))
df["R_log"] = 100 * np.log1p(df.ret_fut_30d)
px = pd.read_csv(ROOT / "data" / "btc_prices.csv", parse_dates=["date"]).set_index("date").close.sort_index()
r = 100 * np.log(px / px.shift(1))
t5 = pd.read_csv(ROOT / "data" / "previsoes_bvrp_T5.csv", parse_dates=["date"])
ridge = t5[(t5.janela == "expansiva") & (t5.modelo == "Ridge")].set_index("date").previsao.sort_index()
d987 = df.set_index("date").loc[ridge.index]
Zf = np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)])
Z16 = np.column_stack([np.ones(len(d987)), d987[feats].to_numpy(float)])
Z1 = np.column_stack([np.ones(len(d987)), ridge.to_numpy(float)])
y, y987 = df.R_log.to_numpy(float), d987.R_log.to_numpy(float)

print("1) hac_wald × hac_utils")
W, _ = wu.hac_wald(y, Zf, (31, 61))
check(rel(W[0], w_sm(mqo_newey_west(df.R_log, df[feats], h=30), 16)) < 1e-8, f"q16 completa, L = 31: W = {W[0]:.6f}")
check(rel(W[1], w_sm(sensibilidade_defasagens(df.R_log, df[feats], lags=[61])[61], 16)) < 1e-8, "q16 completa, L = 61")
check(round(W[0], 3) == 20.008, "W = 20,008 (passo 1)")
for L in (2, 31, 121):
    W1, _ = wu.hac_wald(y987, Z1, (L,))
    ref = w_sm(sensibilidade_defasagens(d987.R_log, ridge.rename("x"), lags=[L])[L], 1)
    check(rel(W1[0], ref) < 1e-8, f"um fator Ridge, L = {L}")

print("2) hac_wald_sub")
Ws, _ = wu.hac_wald_sub(y, Zf, (31, 61), {"todas": list(range(1, 17))})
check(np.allclose(Ws[0], wu.hac_wald(y, Zf, (31, 61))[0], rtol=1e-12, atol=0), "todas as inclinações = hac_wald")
res = mqo_newey_west(df.R_log, df[feats], h=30)
W3, _ = wu.hac_wald_sub(y, Zf, (31,), {"x3": [3]})
check(rel(W3[0, 0], float(res.tvalues.iloc[3]) ** 2) < 1e-8, "uma inclinação = t² do statsmodels")
rs = 100 * d987.ret_fut_30d.to_numpy(float)
D = d987.regime_alta_fixo.to_numpy(float)
Zr = np.column_stack([np.ones(len(D)), ridge.to_numpy(float), D, D * ridge.to_numpy(float)])
Wr, _ = wu.hac_wald_sub(rs, Zr, (31,), {"int": [3], "conj": [2, 3]})
t10 = pd.read_csv(ROOT / "outputs" / "T10" / "regime_T10.csv").query("corte == 'fixo' and h == 30").iloc[0]
check(abs(stats.chi2.sf(Wr[0, 0], 1) - t10.p_interacao) < 1e-8, "regime, p da interação = T10")
check(abs(stats.chi2.sf(Wr[1, 0], 2) - t10.p_conjunto) < 1e-8, "regime, p do conjunto = T10")

print("3) somas_h, retornos_h e idx_circ")
rng = np.random.default_rng(0)
eps = rng.normal(size=200)
for h in (1, 30, 60):
    direta = np.array([eps[t + 1: t + h + 1].sum() for t in range(200 - h)])
    check(np.allclose(wu.somas_h(eps, 200 - h, h), direta, rtol=0, atol=1e-12), f"somas_h, h = {h}")
for h, col in ((1, "ret_fut_1d"), (30, "ret_fut_30d"), (60, "ret_fut_60d")):
    rc = wu.retornos_h(r, pd.DatetimeIndex(df.date), h)
    check(np.max(np.abs(wu.somas_h(rc, len(df), h) - 100 * np.log1p(df[col]))) < 1e-8, f"retornos_h, h = {h}")
j = wu.idx_circ(987, 60, np.random.default_rng(1))
check(len(j) == 987 and j.min() >= 0 and j.max() < 987, "idx_circ: 987 índices em [0, 987)")
saltos = np.flatnonzero(np.diff(j) % 987 != 1)
check(np.all(np.diff(np.concatenate([[-1], saltos])) % 60 == 0), "idx_circ: blocos de 60 contíguos módulo n")

print("4) Reprodução dos passos 2 e 3")
for nome, yy, Z, ref in (("q16 completa", y, Zf, 0.42177722152690866), ("q16 987", y987, Z16, 0.19530284301606923)):
    W0, _ = wu.hac_wald(yy, Z, (31,))
    ks = np.arange(90, len(yy) - 90 + 1)
    p, _ = wu.p_mc(W0[0], wu.rotacoes(yy, Z, (31,), ks)[:, 0])
    check(abs(p - ref) < 1e-12, f"rotação, {nome}: p = {p:.6f} (passo 3: {ref:.6f})")
W0, Wb = wu.bootstrap_pares(y, Zf, (31, 61), 60, 1999, [1, 0, 60])
ref = pd.read_csv(PASSO2 / "wald_bootstrap.csv").query("caso == 'q16_completa' and bloco == 60 and L == 31").p_boot.iloc[0]
check(abs(wu.p_mc(W0[0], Wb[:, 0])[0] - ref) < 1e-12, f"bootstrap, q16 completa, bloco 60: p = {ref} (passo 2)")

print("5) Freedman–Lane")
ks = np.arange(90, 300)
check(np.allclose(wu.rotacoes_freedman_lane(y987, Z1, (31,), [1], ks), wu.rotacoes(y987, Z1, (31,), ks),
                  rtol=1e-9, atol=0), "idx = todas as inclinações: FL = rotação de y")

print("6) Valores-p")
check(wu.p_mc(2.0, np.array([1.0, 2.0, 3.0, np.nan])) == (0.75, 1), "p_mc = (1 + 2)/(3 + 1), 1 não finito")
p0, pk = wu.postos(2.0, np.array([1.0, 2.0, 3.0]))
check(p0 == 0.75 and np.allclose(pk, [1.0, 0.75, 0.25]), "postos com empate")
Wobs = np.array([5.0, 1.0])
Wk = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
# p_h obs: h1 = 1/4, h2 = 4/4; T = 1/4. p_h por k: k1 (4/4, 3/4), k2 (3/4, 2/4), k3 (2/4, 1/4) -> T(k) = 3/4, 2/4, 1/4
p, T, pobs = wu.min_p(Wobs, Wk)
check(T == 0.25 and np.allclose(pobs, [0.25, 1.0]) and p == 0.5, "min_p: p = (1 + 1)/(3 + 1)")
for pv in ([0.01, 0.04, 0.03, 0.2, 0.5, 0.04], [0.3, 0.3, 0.3], [0.001, 0.9, 0.02, 0.011, 0.049, 0.7]):
    check(np.allclose(wu.holm(pv), multipletests(pv, method="holm")[1], rtol=0, atol=1e-15), f"holm {pv}")

print("7) Restrições lineares")
R_gama = np.array([[0, 0, 0, 1.0]])
R_soma = np.array([[0, 1.0, 0, 1.0]])
R_todas = np.eye(4)[1:]
Wl, bl = wu.hac_wald_lin(rs, Zr, (31, 61), {"g": R_gama, "c": np.eye(4)[[2, 3]], "t": R_todas})
Ws, _ = wu.hac_wald_sub(rs, Zr, (31, 61), {"g": [3], "c": [2, 3], "t": [1, 2, 3]})
check(np.allclose(Wl, Ws, rtol=1e-12, atol=0), "R de seleção = hac_wald_sub (γ; δ e γ; todas)")
Xr = pd.DataFrame({"b": ridge.to_numpy(float), "D": D, "DB": D * ridge.to_numpy(float)}, index=d987.index)
res_r = sensibilidade_defasagens(pd.Series(rs, index=d987.index), Xr, lags=[31])[31]
tt = res_r.t_test("b + DB = 0")
Wsoma, _ = wu.hac_wald_lin(rs, Zr, (31,), {"s": R_soma})
check(rel(Wsoma[0, 0], float(np.squeeze(tt.tvalue)) ** 2) < 1e-8, f"β + γ = 0: W = t² do statsmodels ({Wsoma[0, 0]:.6f})")
check(abs(bl[1] + bl[3] - float(np.squeeze(tt.effect))) < 1e-10, "β̂ + γ̂ = efeito do t_test")
ks = np.arange(90, 300)
check(np.allclose(wu.rotacoes_freedman_lane_lin(y987, Z1, (31,), [[0, 1.0]], ks),
                  wu.rotacoes_freedman_lane(y987, Z1, (31,), [1], ks), rtol=1e-9, atol=0),
      "Freedman–Lane linear com R de seleção = rotacoes_freedman_lane")
yr = d987.R_log.to_numpy(float)
Zrest = np.column_stack([np.ones(len(D)), ridge.to_numpy(float) * (1 - D), D])      # β + γ = 0 reparametrizado
aj = Zrest @ np.linalg.lstsq(Zrest, yr, rcond=None)[0]
ZZi_r = np.linalg.inv(Zr.T @ Zr)
manual = np.array([wu.hac_wald_lin(aj + np.roll(yr - aj, k), Zr, (31,), {"s": R_soma}, ZZi_r)[0][0] for k in ks])
check(np.allclose(wu.rotacoes_freedman_lane_lin(yr, Zr, (31,), R_soma, ks), manual, rtol=1e-9, atol=0),
      "Freedman–Lane em β + γ = 0 = modelo restrito reparametrizado à mão")
W1, Wb1 = wu.bootstrap_pares(yr, Zr, (31,), 60, 199, [9, 9])
W2, Wb2 = wu.bootstrap_pares_lin(yr, Zr, (31,), {"t": R_todas}, 60, 199, [9, 9])
check(np.allclose(W1, W2[0], rtol=1e-12, atol=0) and np.allclose(Wb1, Wb2[:, 0, :], rtol=1e-12, atol=0, equal_nan=True),
      "bootstrap_pares_lin com R de todas as inclinações = bootstrap_pares (mesmos W*)")

print(f"\n{'Tudo OK' if not falhas else f'{len(falhas)} FALHA(S)'}")
sys.exit(1 if falhas else 0)
