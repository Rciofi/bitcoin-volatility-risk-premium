"""
test_T4_vazamento.py — teste de vazamento de informação futura (T4)

Para cada data t de uma amostra (25 sorteadas com semente fixa + datas-limite),
corta os dados BRUTOS (btc_prices.csv e dvol_30d_full.csv) em t, recalcula
todas as variáveis explicativas com construir_features e confere que o valor
em t é igual ao de data/ml_dataset_T4.csv (tolerância 1e-12: as diferenças
restantes são de arredondamento de ponto flutuante, ~1e-14). Se alguma
variável usasse dado posterior a t, o valor recalculado com a série cortada
seria diferente -- o item 3 confirma que o teste detecta esse caso.

Confere também:
  - as colunas do dataset são exatamente date, o alvo e as variáveis previstas;
  - nenhuma coluna prospectiva além do alvo (nome com "fut", rv_30d, rv_30d_fut);
  - N = 1.776, de 23/04/2021 a 03/03/2026, sem NaN;
  - vrp_30d == vh_30d - iv_30d (colinearidade exata documentada);
  - bvrp_realizado_defasado(t) == alvo de t-30 (o prêmio que se realiza em t).

Uso:  python code/test_T4_vazamento.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ml_dataset_T4 import ALVO, FEATURES, carregar_brutos, construir_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_SORTEIO = 25
SEMENTE = 20260930
DATAS_LIMITE = ["2021-04-23", "2023-03-10", "2023-04-30", "2026-03-03"]  # 1ª data, SVB, fim do mês que era buraco, última
N_ESPERADO = 1776
TOL = 1e-12

falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


ds = pd.read_csv(ROOT / "data" / "ml_dataset_T4.csv", parse_dates=["date"]).set_index("date")

print("1) Estrutura do dataset")
check(list(ds.columns) == [ALVO] + FEATURES, f"colunas = alvo + {len(FEATURES)} variáveis explicativas")
proib = [c for c in ds.columns if c != ALVO and ("fut" in c or c.startswith("rv_30d"))]
check(not proib, f"sem colunas prospectivas além do alvo {proib or ''}")
check(len(ds) == N_ESPERADO, f"N = {len(ds)} (esperado {N_ESPERADO}); {ds.index.min().date()} a {ds.index.max().date()}")
check(not ds.isna().any().any(), "sem NaN")
check(float((ds.vrp_30d - (ds.vh_30d - ds.iv_30d)).abs().max()) < 1e-9, "vrp_30d == vh_30d - iv_30d")
alvo_t30 = ds[ALVO].shift(30)  # série diária contínua: 30 linhas = 30 dias
m30 = alvo_t30.notna()
check(float((ds.loc[m30, "bvrp_realizado_defasado"] - alvo_t30[m30]).abs().max()) < 1e-9,
      f"bvrp_realizado_defasado(t) == alvo(t-30) em {int(m30.sum())} datas")

print(f"2) Vazamento: dados brutos cortados em t ({N_SORTEIO} datas sorteadas + {len(DATAS_LIMITE)} datas-limite)")
precos, dvol = carregar_brutos()
rng = np.random.default_rng(SEMENTE)
datas = sorted(set(pd.to_datetime(DATAS_LIMITE)) |
               set(pd.DatetimeIndex(rng.choice(ds.index.to_numpy(), N_SORTEIO, replace=False))))
pior = 0.0
for t in datas:
    x = construir_features(precos[precos.date <= t], dvol[dvol.date <= t])
    recalc = x.loc[t, FEATURES].to_numpy(dtype=float)
    orig = ds.loc[t, FEATURES].to_numpy(dtype=float)
    dif = float(np.max(np.abs(recalc - orig)))
    pior = max(pior, dif)
    ruins = [f for f, a, b in zip(FEATURES, recalc, orig) if abs(a - b) > TOL]
    check(not ruins, f"{t.date()}: {len(FEATURES)} variáveis iguais até {TOL:.0e} (dif. máx. {dif:.1e}) {ruins or ''}")
print(f"  maior diferença absoluta entre todas as datas e variáveis: {pior:.1e}")

print(f"3) Controle positivo: o teste detecta vazamento")
# variável falsa que usa o retorno de t+1: deve DIFERIR entre série cortada e completa
p_full = precos.set_index("date")["close"]
t = datas[len(datas) // 2]
falsa_full = np.log(p_full).diff().shift(-1).loc[t]
falsa_cort = np.log(p_full[p_full.index <= t]).diff().shift(-1).loc[t]
check(np.isnan(falsa_cort) and not np.isnan(falsa_full),
      f"{t.date()}: variável com r(t+1) muda ao cortar em t (completa {falsa_full:.5f}, cortada NaN)")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
