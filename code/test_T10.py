"""
test_T10.py — testes do T10 (retorno sobre o BVRP previsto) e do T12

  1. Embargo: toda previsão usada (Ridge e floresta, janela expansiva) vem de
     uma origem a 0–29 dias, cujo treino termina em origem − 30 dias
     (hiperparametros_T5.csv); nenhuma previsão usa treino com s > t − 30.
  2. Fidelidade do primeiro estágio do bootstrap: com os dados originais,
     primeiro_estagio reproduz exatamente as previsões do T5 (Ridge com α
     reescolhido e com α fixo; floresta com hiperparâmetros fixos).
  3. Bootstrap em dois níveis: blocos contíguos de tamanho b; o 1º nível só
     reamostra posições de 0 a t − 30 em cada origem (embargo exato); o 2º
     nível só reamostra datas do período fora da amostra; mesma semente ->
     mesma reamostragem.
  4. Alinhamento: o β principal de cada h é igual ao MQO de ret_fut_h(t) sobre
     a previsão de t, montado à mão a partir dos arquivos.
  5. Saídas: nº de reamostragens por variante (999 x 5 e 199) e T12 com
     N = 1.806 e h = 30.
  6. (a) O bootstrap só do 2º nível (previsões fixas) é centrado:
     |média dos β* − β̂| < 0,5 × EP, em cada h -- valida a reamostragem.
     (b) Com os dois níveis (principal), a média dos β* fica entre 0 e β̂
     (atenuação pelo regressor gerado, sem inversão de sinal). A razão
     média/β̂ das demais variantes é impressa, sem limiar.
     (c) No principal, o EP corrigido da atenuação (EP_boot / λ) é maior que
     o EP só do 2º nível, em cada h: o 1º nível acrescenta incerteza.
  7. Reprodutibilidade: o resumo recalculado de bootstrap_betas_T10.csv com
     resumo_bootstrap é idêntico a bootstrap_resumo_T10.csv.

Uso:  python code/test_T10.py   (sai com código 1 se algo falhar)
"""
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # β, − no console mesmo com saída redirecionada (Windows)

import numpy as np  # noqa: E402
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retorno_bvrp_T10 as T  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "T10"
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


df, vt, prev, hp = T.carregar()
y = df[T.P.ALVO].to_numpy(float)
cols = [c for c in df.columns if c not in ("date", T.P.ALVO) and not c.startswith(("ret_fut", "regime_alta"))]
feats_arv = cols
feats_lin = [c for c in cols if c != "vh_30d"]
origens = gerar_divisoes(len(y), h=T.H_ALVO, janela="expansiva").origens

print("1) Embargo")
for m in ["Ridge", "floresta_aleatoria"]:
    g = prev[prev.modelo == m]
    h = hp[hp.modelo == m]
    lag = (g.date - g.origem).dt.days
    fim = dict(zip(h.origem, pd.to_datetime(h.treino_fim)))
    ok_fim = all(fim[o] == o - pd.Timedelta(days=30) for o in g.origem.unique())
    check(len(g) == 987 and bool(lag.between(0, 29).all()) and ok_fim,
          f"{m}: 987 previsões, origem a 0–29 dias, treino termina em origem − 30 (s <= t − 30)")

print("2) Fidelidade do primeiro estágio")
ridge_t5 = prev[prev.modelo == "Ridge"].sort_values("date").previsao.to_numpy()
rf_t5 = prev[prev.modelo == "floresta_aleatoria"].sort_values("date").previsao.to_numpy()
p1 = T.primeiro_estagio("Ridge", df[feats_lin].to_numpy(float), y, origens)
check(np.allclose(p1, ridge_t5, atol=1e-8, rtol=0), f"Ridge, α reescolhido: reproduz o T5 (dif. máx. {np.abs(p1 - ridge_t5).max():.1e})")
alfa = [T._params(s) for s in hp[hp.modelo == "Ridge"].sort_values("origem").params]
p2 = T.primeiro_estagio("Ridge", df[feats_lin].to_numpy(float), y, origens, alfa)
check(np.allclose(p2, ridge_t5, atol=1e-8, rtol=0), "Ridge, α fixo do T5: reproduz o T5")
rf = [T._params(s) for s in hp[hp.modelo == "floresta_aleatoria"].sort_values("origem").params]
p3 = T.primeiro_estagio("floresta_aleatoria", df[feats_arv].to_numpy(float), y, origens, rf)
check(np.allclose(p3, rf_t5, atol=1e-8, rtol=0), f"floresta, hiperparâmetros fixos: reproduz o T5 (dif. máx. {np.abs(p3 - rf_t5).max():.1e})")

print("3) Bootstrap em dois níveis: blocos e posições sorteadas")
idx = T.indices_blocos(1776, 60, np.random.default_rng(0))
quebras = np.where(np.diff(idx) != 1)[0] + 1
blocos = np.split(idx, quebras)
check(len(idx) == 1776 and all(len(b) == 60 for b in blocos[:-1]) and idx.min() >= 0 and idx.max() <= 1775,
      f"blocos contíguos de 60 ({len(blocos)} blocos em 1.776 posições)")
R = df[[f"ret_fut_{h}d" for h in T.HS]].to_numpy(float)
oos = np.concatenate([o.teste for o in origens])
ok1 = ok2 = True
for r in range(20):
    for b in (30, 60, 90):
        treinos, j = T.sortear_indices(r, b, origens, len(oos))
        ok1 &= all(len(tr) == len(o.treino) and tr.min() >= 0 and tr.max() <= o.t - 30 for tr, o in zip(treinos, origens))
        ok2 &= len(j) == len(oos) and j.min() >= 0 and j.max() < len(oos) and oos[j].min() >= origens[0].t
check(ok1, "1º nível: em cada origem, só posições de 0 a t − 30 (embargo exato) — 20 sementes × 3 blocos")
check(ok2, f"2º nível: só datas fora da amostra (posições >= {origens[0].t}, a 1ª origem) — 20 sementes × 3 blocos")
a = T.uma_reamostragem(7, "Ridge", df[feats_lin].to_numpy(float), y, R, origens, oos, 60, alfa)
b = T.uma_reamostragem(7, "Ridge", df[feats_lin].to_numpy(float), y, R, origens, oos, 60, alfa)
check(np.array_equal(a, b), "mesma semente -> mesma reamostragem")

print("4) Alinhamento das regressões")
pr = pd.read_csv(OUT / "principal_T10.csv")
p = prev[prev.modelo == "Ridge"].set_index("date").previsao
d = df.set_index("date").loc[p.index]
for h in T.HS:
    Z = np.column_stack([np.ones(len(p)), p.to_numpy()])
    bh = np.linalg.lstsq(Z, d[f"ret_fut_{h}d"].to_numpy(), rcond=None)[0][1]
    check(abs(bh - pr[(pr.modelo == "Ridge") & (pr.h == h)].beta.iloc[0]) < 1e-12, f"h = {h}: β = MQO de ret_fut_h(t) sobre a previsão de t")

print("5) Saídas")
bt = pd.read_csv(OUT / "bootstrap_betas_T10.csv")
n = bt.groupby(["variante", "bloco"]).size().to_dict()
esperado = {("Ridge_alpha_reescolhido", 60): 999, ("Ridge_alpha_fixo", 30): 999, ("Ridge_alpha_fixo", 60): 999,
            ("Ridge_alpha_fixo", 90): 999, ("floresta_hiperparametros_fixos", 60): 199,
            ("Ridge_so_segundo_nivel", 60): 999}
check(n == esperado, f"reamostragens por variante: {n}")
t12 = pd.read_csv(ROOT / "outputs" / "T12" / "regressao_T12.csv")
check(int(t12.n.iloc[0]) == 1806 and int(t12.h.iloc[0]) == 30, "T12: N = 1.806, h = 30")

rb = pd.read_csv(OUT / "bootstrap_resumo_T10.csv")
print("6a) Contraprova: bootstrap só do 2º nível (previsões fixas) é centrado em β̂")
for r in rb[rb.variante == "Ridge_so_segundo_nivel"].itertuples():
    d = abs(r.media_boot - r.beta)
    check(d < 0.5 * r.ep_boot, f"h = {r.h}: |média dos β* − β̂| = {d:.2e} < 0,5 × EP = {0.5 * r.ep_boot:.2e}")
print(f"6b) Dois níveis (principal: {T.VARIANTE_PRINCIPAL}, bloco de 60): média dos β* entre 0 e β̂ (atenuação)")
for r in rb[(rb.variante == T.VARIANTE_PRINCIPAL) & (rb.bloco == 60)].itertuples():
    check(min(0, r.beta) <= r.media_boot <= max(0, r.beta),
          f"h = {r.h}: média {r.media_boot:.2e} entre 0 e β̂ = {r.beta:.2e} (razão média/β̂ = {r.razao_media_beta:.2f})")
print(f"6c) Principal: EP corrigido (EP_boot / λ) > EP só do 2º nível (o 1º nível acrescenta incerteza)")
for r in rb[(rb.variante == T.VARIANTE_PRINCIPAL) & (rb.bloco == 60)].itertuples():
    check(r.ep_corrigido > r.ep_so_segundo_nivel,
          f"h = {r.h}: EP corrigido {r.ep_corrigido:.2e} > EP só do 2º nível {r.ep_so_segundo_nivel:.2e} "
          f"(razão {r.razao_ep_corrigido_so2:.2f}; EP bruto / 2º nível = {r.razao_ep_boot_so2:.2f})")
for r in rb[~rb.variante.isin(["Ridge_so_segundo_nivel"]) &
            ~((rb.variante == T.VARIANTE_PRINCIPAL) & (rb.bloco == 60))].itertuples():
    print(f"  (info) {r.variante}, bloco {r.bloco}, h = {r.h}: razão média/β̂ = {r.razao_media_beta:.2f}")

print("7) Reprodutibilidade do resumo: recalculado de bootstrap_betas_T10.csv com resumo_bootstrap")
bt_h = bt.rename(columns={str(h): h for h in T.HS})
rec = T.resumo_bootstrap(bt_h, pr)
rec = pd.read_csv(io.StringIO(rec.to_csv(index=False)))   # mesma ida e volta do CSV
try:
    pd.testing.assert_frame_equal(rec, rb, check_exact=True)
    check(True, f"resumo recalculado idêntico ao gravado ({rb.shape[0]} linhas x {rb.shape[1]} colunas)")
except AssertionError as e:
    check(False, f"resumo recalculado difere do gravado: {str(e).splitlines()[0]}")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
