"""
test_regimes_T9.py — testes dos regimes do T9

  1. Corte fixo: recalculado a partir dos DADOS BRUTOS cortados no fim da
     primeira janela (22/04/2023), reconstruindo vh_30d com
     build_ml_dataset_T4.construir_features, é idêntico ao gravado. Nenhuma
     data posterior influencia o corte; o teste também altera toda a série
     depois da primeira janela e confirma que o corte não muda. O corte é
     estável entre sementes (diferença < 0,1), coincide com a antimoda da
     densidade de núcleo da 1ª janela (tolerância 1,0) e é 37,3. Toda
     estimação da mistura exige convergência (assert em ajustar_mistura).
  2. data/regimes_T9.csv: mesmas datas de ml_dataset_T4.csv; regime_alta_fixo
     = 1 se vh_30d > corte_fixo; nenhuma coluna prospectiva.
  3. Corte expansivo (robustez): em cada uma das 33 origens, recalculado com os
     dados brutos cortados na origem, é idêntico a cortes_por_origem_T9.csv; o
     corte vigente em cada data é o da origem mais recente <= data (antes da
     1ª origem, o fixo).
  4. Previsões com interação: origens iguais às do split_utils, 0 <= date -
     origem <= 29 dias, 987 datas por regime e modelo.

Uso:  python code/test_regimes_T9.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ml_dataset_T4 import carregar_brutos, construir_features  # noqa: E402
from regimes_T9 import antimoda_kde, corte_mistura_log  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


ds = pd.read_csv(ROOT / "data" / "ml_dataset_T4.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
reg = pd.read_csv(ROOT / "data" / "regimes_T9.csv", parse_dates=["date"])
cortes = pd.read_csv(ROOT / "outputs" / "T9" / "cortes_por_origem_T9.csv", parse_dates=["origem"])
div = gerar_divisoes(len(ds), h=30)
inicio = ds.date.iloc[0]
precos, dvol = carregar_brutos()


def vh_bruto_ate(t):
    """vh_30d reconstruída a partir dos dados brutos cortados em t, de 'inicio' até t."""
    x = construir_features(precos[precos.date <= t], dvol[dvol.date <= t])
    return x.loc[(x.index >= inicio) & (x.index <= t), "vh_30d"].to_numpy()


print("1) Corte fixo (primeira janela)")
fim_pj = ds.date.iloc[div.primeira_janela[-1]]
c_bruto = corte_mistura_log(vh_bruto_ate(fim_pj))
c_arq = float(reg.corte_fixo.iloc[0])
check(len(vh_bruto_ate(fim_pj)) == 730, f"primeira janela com 730 obs. até {fim_pj.date()}")
check(abs(c_bruto - c_arq) < 1e-9, f"corte com dados brutos cortados em {fim_pj.date()} = {c_bruto:.6f} (gravado {c_arq:.6f})")
vh_alt = ds.vh_30d.to_numpy().copy()
vh_alt[730:] *= np.random.default_rng(1).uniform(0.2, 5.0, len(vh_alt) - 730)
check(abs(corte_mistura_log(vh_alt[:730]) - c_arq) < 1e-12, "alterar toda a série depois da 1ª janela não muda o corte")
check(bool(reg.corte_fixo.nunique() == 1), "corte fixo único para toda a amostra")
vh_pj = ds.vh_30d.to_numpy()[:730]
por_semente = [corte_mistura_log(vh_pj, random_state=r) for r in range(6)]
check(max(por_semente) - min(por_semente) < 0.1,
      f"estável entre 6 sementes: {min(por_semente):.4f} a {max(por_semente):.4f} (diferença < 0,1)")
am = antimoda_kde(vh_pj)
check(abs(am - c_arq) < 1.0, f"coincide com a antimoda da densidade de núcleo: {am:.2f} vs. corte {c_arq:.2f} (tolerância 1,0)")
check(abs(c_arq - 37.3) < 0.05, f"corte = {c_arq:.2f} (decisão: 37,3)")

print("2) Arquivo de regimes")
check(list(reg.date) == list(ds.date), f"mesmas {len(ds)} datas de ml_dataset_T4.csv")
check(bool((reg.regime_alta_fixo == (reg.vh_30d > c_arq).astype(int)).all()), "regime_alta_fixo = 1 se vh_30d > corte")
check(not [c for c in reg.columns if "fut" in c or "alvo" in c], "nenhuma coluna prospectiva")

print("3) Corte expansivo por origem (robustez)")
ok = 0
for o, linha in zip(div.origens, cortes.itertuples()):
    t = ds.date[o.t]
    ok += int(linha.origem == t and abs(corte_mistura_log(vh_bruto_ate(t)) - linha.corte) < 1e-9)
check(ok == len(div.origens) == len(cortes), f"{ok} de {len(div.origens)} cortes recalculados com dados brutos cortados na origem")
vig = pd.Series(cortes.corte.to_numpy(), index=cortes.origem).reindex(reg.date, method="ffill")
esperado = np.where(vig.isna(), c_arq, vig.to_numpy())
check(np.allclose(reg.corte_expansivo_vigente, esperado), "corte vigente = o da origem mais recente <= data (antes, o fixo)")
check(bool((reg.regime_alta_expansivo == (reg.vh_30d > reg.corte_expansivo_vigente).astype(int)).all()),
      "regime_alta_expansivo coerente com o corte vigente")

print("4) Previsões com interação")
pv = pd.read_csv(ROOT / "outputs" / "T9" / "previsoes_interacao_T9.csv", parse_dates=["date", "origem"])
origens = [ds.date[o.t] for o in div.origens]
for (r, m), g in pv.groupby(["regime", "modelo"]):
    lag = (g.date - g.origem).dt.days
    check(sorted(g.origem.unique()) == origens and bool(lag.between(0, 29).all()) and len(g) == 987 and g.date.is_unique,
          f"{r}/{m}: 33 origens do split_utils, 0 <= date − origem <= 29, 987 datas")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
