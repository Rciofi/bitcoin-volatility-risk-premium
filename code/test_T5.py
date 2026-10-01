"""
test_T5.py — testes da previsão do BVRP (T5/T7)

  1. avaliacao_utils: R² fora da amostra, Clark–West e Diebold–Mariano
     conferidos contra cálculo manual (pesos de Bartlett 1 - j/(L+1),
     L = h+1, sem correção de amostra pequena, normal).
  2. data/previsoes_bvrp_T5.csv:
     - colunas date, origem, janela, modelo, previsao; SEM o alvo;
     - origens iguais às de split_utils; 0 <= date - origem <= 29 dias;
       cada origem cobre exatamente o seu bloco de teste; todas as datas
       fora da amostra (987) em todos os modelos (salvo --parcial);
     - embargo: n_treino e treino_fim de hiperparametros_T5.csv conferem
       com s <= origem - 30 (e, na móvel, com 730 obs.);
     - referências: média histórica = média do alvo com s <= origem - 30
       (na móvel, só a janela de 730); proxy = vrp_30d(date);
       persistência viável = bvrp_realizado_defasado(date).
  3. Placebo (outputs/T5/placebo_T5.csv; 10 permutações nas árvores e 200
     nos lineares), só como verificação de vazamento: mediana do R² placebo
     fora da amostra <= 0 em todos os modelos (mínimo e máximo impressos).

Uso:  python code/test_T5.py [--parcial] [--out-dir DIR] [--previsoes ARQ]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H = 30
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


def t_manual(serie, L):
    """t da média com Newey–West manual (Bartlett, L defasagens, sem correção)."""
    x = np.asarray(serie, float)
    u = x - x.mean()
    n = len(u)
    s = u @ u + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) for j in range(1, L + 1))
    return x.mean() / (np.sqrt(s) / n)


ap = argparse.ArgumentParser()
ap.add_argument("--parcial", action="store_true", help="aceita só as primeiras origens (teste de fumaça)")
ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "T5"))
ap.add_argument("--previsoes", default=str(ROOT / "data" / "previsoes_bvrp_T5.csv"))
args = ap.parse_args()

print("1) Fórmulas de avaliação")
rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(size=400)) * 0.1 + rng.normal(size=400)
p1, p2 = y + rng.normal(size=400), np.full(400, y.mean())
check(abs(r2_fora_da_amostra(y, p1, p2) - (1 - np.sum((y - p1) ** 2) / np.sum((y - p2) ** 2))) < 1e-12, "R² fora da amostra")
f = (y - p2) ** 2 - ((y - p1) ** 2 - (p2 - p1) ** 2)
cw = clark_west(y, p1, p2, h=H)
check(abs(cw["cw_t"] - t_manual(f, H + 1)) < 1e-8 and abs(cw["cw_p_unilateral"] - stats.norm.sf(cw["cw_t"])) < 1e-12,
      f"Clark–West: t = {cw['cw_t']:.6f} (manual {t_manual(f, H + 1):.6f}), p unilateral pela normal")
d = (y - p1) ** 2 - (y - p2) ** 2
dm = diebold_mariano(y, p1, p2, h=H)
check(abs(dm["dm_t"] - t_manual(d, H + 1)) < 1e-8 and abs(dm["dm_p_bilateral"] - 2 * stats.norm.sf(abs(dm["dm_t"]))) < 1e-12,
      f"Diebold–Mariano: t = {dm['dm_t']:.6f} (manual {t_manual(d, H + 1):.6f}), p bilateral pela normal")

print("2) Arquivo de previsões")
ds = pd.read_csv(ROOT / "data" / "ml_dataset_T4.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
pv = pd.read_csv(args.previsoes, parse_dates=["date", "origem"])
hp = pd.read_csv(Path(args.out_dir) / "hiperparametros_T5.csv", parse_dates=["origem", "treino_fim"])
check(list(pv.columns) == ["date", "origem", "janela", "modelo", "previsao"], "colunas: date, origem, janela, modelo, previsao")
check("alvo_bvrp_30d_fut" not in pv.columns, "o alvo não está no arquivo de previsões")
alvo = ds.set_index("date")["alvo_bvrp_30d_fut"]
d30 = pd.Timedelta(days=H)
for janela, g in pv.groupby("janela"):
    div = gerar_divisoes(len(ds), h=H, janela=janela)
    origens_split = [ds.date[o.t] for o in div.origens]
    origens_arq = sorted(g.origem.unique())
    esperado = origens_split[:len(origens_arq)] if args.parcial else origens_split
    check(list(pd.to_datetime(origens_arq)) == list(esperado), f"{janela}: {len(origens_arq)} origens iguais às do split_utils")
    lag = (g.date - g.origem).dt.days
    check(bool(lag.between(0, H - 1).all()), f"{janela}: 0 <= date - origem <= 29 dias")
    for m, gm in g.groupby("modelo"):
        esperado_n = sum(len(o.teste) for o in div.origens[:len(origens_arq)])
        ok = len(gm) == esperado_n and gm.date.is_unique
        if not args.parcial:
            ok = ok and esperado_n == 987
        check(ok, f"{janela}/{m}: {len(gm)} datas fora da amostra, sem repetição")
    # embargo: tamanho e fim do treino de cada origem
    h = hp[hp.janela == janela]
    fim_ok = bool((h.treino_fim == h.origem - d30).all())
    n_esp = h.origem.map(lambda t: int((ds.date <= t - d30).sum()) if janela == "expansiva" else 730)
    check(fim_ok and bool((h.n_treino == n_esp).all()), f"{janela}: treino termina em origem − 30 dias e tem o tamanho esperado")
    # referências
    w = g.pivot(index="date", columns="modelo", values="previsao")
    orig = g.drop_duplicates("date").set_index("date")["origem"]
    ini = {t: (ds.date.iloc[0] if janela == "expansiva" else t - d30 - pd.Timedelta(days=729)) for t in orig.unique()}
    media_esp = orig.map(lambda t: alvo[(alvo.index >= ini[t]) & (alvo.index <= t - d30)].mean())
    check(np.allclose(w["media_historica"], media_esp.loc[w.index]), f"{janela}: média histórica só com alvos de s <= origem − 30")
    check(np.allclose(w["proxy_retrospectiva"], ds.set_index("date").loc[w.index, "vrp_30d"]), f"{janela}: proxy = vrp_30d(date)")
    check(np.allclose(w["persistencia_viavel"], ds.set_index("date").loc[w.index, "bvrp_realizado_defasado"]),
          f"{janela}: persistência viável = bvrp_realizado_defasado(date)")

print("3) Placebo")
pl = Path(args.out_dir) / "placebo_T5.csv"
if pl.exists():
    p = pd.read_csv(pl)
    for r in p.itertuples():
        check(r.r2_placebo_mediana <= 0,
              f"placebo {r.modelo} ({r.n_permutacoes} permutações): mediana do R² = {r.r2_placebo_mediana:.4f} <= 0 "
              f"(mín. {r.r2_placebo_min:.4f}, máx. {r.r2_placebo_max:.4f})")
else:
    check(False, f"{pl} não encontrado")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
