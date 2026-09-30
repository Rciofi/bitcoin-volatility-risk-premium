"""
test_T1_alinhamento.py — testes do T1 (definição prospectiva do BVRP)

Confere, depois de rodar o pipeline (build_vrp_dataset, build_targets,
analyze_vrp_regimes, build_ml_dataset):

  1. Alinhamento: rv_30d_fut(t) == rv_30d(t + 30 dias) em TODAS as datas
     válidas da série longa de preços, com 3 datas impressas explicitamente.
     rv_30d_fut é calculada com janela para frente e rv_30d com janela para
     trás, por caminhos independentes -- a igualdade não é tautológica.
  2. As últimas 30 linhas da série de preços têm rv_30d_fut NaN (e só elas).
  3. bvrp_30d_fut == rv_30d_fut - iv_30d no dataset consolidado.
  4. rv_30d e vrp_30d (proxy retrospectiva) não mudaram em relação ao commit
     HEAD -- o T1 não pode alterar o significado das colunas antigas.
  5. N de referência (vrp_with_targets.csv) = 1.806, sem NaN em bvrp_30d_fut.
  6. ml_dataset.csv não contém as colunas prospectivas (vazariam em t).

Uso:  python code/test_T1_alinhamento.py   (sai com código 1 se algo falhar)
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vrp_dataset import load_btc_prices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_REFERENCIA = 1806
DATAS_EXEMPLO = ["2021-03-24", "2023-03-10", "2026-04-02"]  # início da amostra, crise SVB, última válida
TOL = 1e-9

falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


# 1-2) série longa de preços
print("1-2) Alinhamento na série longa de preços")
p = load_btc_prices(str(ROOT)).copy()
p["date"] = pd.to_datetime(p["date"])
p = p.set_index("date")
rv_mais_30 = p["rv_30d"].reindex(p.index + pd.Timedelta(days=30)).to_numpy()
validos = p["rv_30d_fut"].notna().to_numpy()
dif = np.abs(p["rv_30d_fut"].to_numpy()[validos] - rv_mais_30[validos])
check(np.nanmax(dif) < TOL and not np.isnan(dif).any(),
      f"rv_30d_fut(t) == rv_30d(t+30 dias) em {validos.sum()} datas (dif. máx. {np.nanmax(dif):.2e})")
for d in DATAS_EXEMPLO:
    t = pd.Timestamp(d)
    a, b = p.at[t, "rv_30d_fut"], p.at[t + pd.Timedelta(days=30), "rv_30d"]
    check(abs(a - b) < TOL, f"{d}: rv_30d_fut = {a:.6f} | rv_30d({(t + pd.Timedelta(days=30)).date()}) = {b:.6f}")
nan_final = p["rv_30d_fut"].isna()
check(nan_final.sum() == 30 and nan_final.iloc[-30:].all(),
      f"rv_30d_fut NaN só nas últimas 30 linhas ({nan_final.sum()} NaN, a partir de {p.index[-30].date()})")

# 3) dataset consolidado
print("3) Dataset consolidado (vrp_30d_dataset.csv)")
ds = pd.read_csv(DATA / "vrp_30d_dataset.csv", parse_dates=["date"])
m = ds["bvrp_30d_fut"].notna()
check(np.allclose(ds.loc[m, "bvrp_30d_fut"], ds.loc[m, "rv_30d_fut"] - ds.loc[m, "iv_30d"], atol=TOL, rtol=0),
      "bvrp_30d_fut == rv_30d_fut - iv_30d")

# 4) colunas antigas inalteradas
print("4) Colunas antigas inalteradas em relação ao commit HEAD")
try:
    txt = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:data/vrp_30d_dataset.csv"],
                         capture_output=True, text=True, check=True).stdout
    head = pd.read_csv(io.StringIO(txt), parse_dates=["date"])
    j = ds.merge(head, on="date", suffixes=("", "_head"))
    check(len(j) == len(head) == len(ds), f"mesmas {len(ds)} datas")
    for c in ["rv_30d", "iv_30d", "vrp_30d", "ret", "close"]:
        check(np.allclose(j[c], j[c + "_head"], atol=TOL, rtol=0, equal_nan=True), f"{c} idêntica ao HEAD")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    check(False, f"não foi possível ler HEAD:data/vrp_30d_dataset.csv via git ({e})")

# 5) N de referência
print("5) Amostra de referência (vrp_with_targets.csv)")
ref = pd.read_csv(DATA / "vrp_with_targets.csv", parse_dates=["date"])
check(len(ref) == N_REFERENCIA, f"N = {len(ref)} (esperado {N_REFERENCIA}); "
      f"{ref['date'].min().date()} a {ref['date'].max().date()}")
check(ref["bvrp_30d_fut"].notna().all(), "sem NaN em bvrp_30d_fut na amostra de referência")

# 6) ml_dataset sem colunas prospectivas
print("6) ml_dataset.csv sem colunas prospectivas")
ml_cols = pd.read_csv(DATA / "ml_dataset.csv", nrows=1).columns
check(not {"rv_30d_fut", "bvrp_30d_fut"} & set(ml_cols), "rv_30d_fut e bvrp_30d_fut ausentes")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
