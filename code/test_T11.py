"""
test_T11.py — testes do T11 (BVRP previsto e retornos, versão não linear)

  1. Identidade: rv_30d_fut = alvo + iv_30d coincide com vh_30d de t+30.
  2. Equivalência das formas no MQO: prever rv_30d_fut e subtrair iv_30d dá as
     mesmas previsões que o MQO direto do T5 (com iv_30d entre os regressores).
  3. Embargo: previsões da forma f(variáveis) − IV vêm de origens a 0–29 dias,
     com treino terminando em origem − 30; na leitura B, em origem − h.
  4. Fidelidade do 1º estágio do bootstrap: com os dados originais,
     T10.primeiro_estagio (alvo RV, hiperparâmetros fixos) − IV reproduz as
     previsões gravadas (floresta e Ridge).
  5. Mesmas reamostragens do T10: as réplicas do Ridge direto (dois níveis e
     só 2º nível, bloco de 60) são idênticas às do T10; β̂ do Ridge direto
     igual ao do T10; com B = 999, resumo igual ao do T10.
  6. Teste conjunto: (a) forma coerente β̄*'Σ*⁻¹β̄* = β̂'(Λ⁻¹Σ*Λ⁻¹)⁻¹β̂;
     (b) com 1 horizonte, W = z²; (c) tamanho por Monte Carlo (Σ com
     correlação 0,9^|i−j|): rejeição a 5% entre 2,5% e 8% no Wald e no
     max-|t|; (d) HAC empilhado com 1 horizonte e maxlags h+1 reproduz o EP
     de hac_utils.mqo_newey_west.
  7. Leitura B: média histórica de cada origem = média de ret_fut_h no treino
     (s <= t − h).
  8. Saídas: réplicas por variante; teste conjunto do T10 com as 6 variantes;
     placebo (verificação de vazamento) com mediana <= 0,01 em todos os modelos/h.

Uso:  python code/test_T11.py [--out-dir <dir> --saida-dados <arq>]   (sai com 1 se algo falhar)
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retorno_bvrp_T11 as T  # noqa: E402
from hac_utils import mqo_newey_west  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "T11"))
ap.add_argument("--saida-dados", default=str(ROOT / "data" / "previsoes_bvrp_T11.csv"))
args = ap.parse_args()
OUT = Path(args.out_dir)
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


df, vt, prev_t5, hp_t5, feats_arv, feats_lin = T.carregar()
origens = gerar_divisoes(len(df), h=T.H_ALVO, janela="expansiva").origens
oos = np.concatenate([o.teste for o in origens])
iv = df["iv_30d"].to_numpy(float)
prev = pd.read_csv(args.saida_dados, parse_dates=["date", "origem"])
hiper = pd.read_csv(OUT / "hiperparametros_T11.csv", parse_dates=["origem", "treino_fim"])

print("1) Identidade da RV futura")
d = (df[T.ALVO_RV] - df["vh_30d"].shift(-30)).abs().dropna()
check(len(d) == len(df) - 30 and d.max() < 1e-9, f"rv_30d_fut = vh_30d(t+30) em {len(d)} datas (dif. máx. {d.max():.1e})")

print("2) Equivalência das formas no MQO")
fixos_mqo = {T.chave(df.date[o.t]): {} for o in origens}
p_mqo, _ = T.prever(df, feats_lin, "MQO", "expansiva", None, hiper_fixos=fixos_mqo, verboso=False)
mqo_t5 = prev_t5[prev_t5.modelo == "MQO"].sort_values("date").previsao.to_numpy()
dif = np.abs(p_mqo.sort_values("date").previsao.to_numpy() - mqo_t5).max()
check(dif < 1e-6, f"MQO: f(variáveis) − IV = forma direta do T5 (dif. máx. {dif:.1e})")

print("3) Embargo")
for (rot, jan), g in prev.groupby(["rotulo", "janela"]):
    lag = (g.date - g.origem).dt.days
    h = hiper[(hiper.rotulo == rot) & (hiper.janela == jan)]
    check(len(g) == 987 and bool(lag.between(0, 29).all())
          and bool(((h.origem - h.treino_fim).dt.days == 30).all()),
          f"{rot}, {jan}: 987 previsões, origem a 0–29 dias, treino termina em origem − 30")
hb = pd.read_csv(OUT / "hiperparametros_arvore_retorno_T11.csv", parse_dates=["origem", "treino_fim"])
pb = pd.read_csv(OUT / "previsoes_arvore_retorno_T11.csv", parse_dates=["date", "origem"])
for h in T.HS:
    ok = bool(((hb[hb.h == h].origem - hb[hb.h == h].treino_fim).dt.days == h).all())
    check(ok and (pb.h == h).sum() == 987, f"leitura B, h = {h}: 987 previsões, treino termina em origem − {h}")

print("4) Fidelidade do primeiro estágio (forma f(variáveis) − IV)")
y_rv = df[T.ALVO_RV].to_numpy(float)
for rot, feats in (("floresta_aleatoria", feats_arv), ("Ridge", feats_lin)):
    fixos = T.params_fixos(hiper, rot, hp_t5, rot, "rv_menos_iv")
    p = T.T10.primeiro_estagio(rot, df[feats].to_numpy(float), y_rv, origens, fixos) - iv[oos]
    g = prev[(prev.rotulo == rot) & (prev.janela == "expansiva")].sort_values("date").previsao.to_numpy()
    check(np.allclose(p, g, atol=1e-8, rtol=0), f"{rot}: reproduz as previsões gravadas (dif. máx. {np.abs(p - g).max():.1e})")

print("5) Mesmas reamostragens do T10")
bt = pd.read_csv(OUT / "bootstrap_betas_T11.csv").rename(columns={str(h): h for h in T.HS})
b10 = pd.read_csv(ROOT / "outputs" / "T10" / "bootstrap_betas_T10.csv").rename(columns={str(h): h for h in T.HS})
for v11, v10 in (("dois_niveis", "Ridge_alpha_fixo"), ("so_segundo_nivel", "Ridge_so_segundo_nivel")):
    a = bt[(bt.regressor == "Ridge_direto") & (bt.variante == v11) & (bt.bloco == 60)][T.HS].to_numpy()
    b = b10[(b10.variante == v10) & (b10.bloco == 60)][T.HS].to_numpy()[:len(a)]
    check(np.allclose(a, b, atol=1e-12, rtol=0), f"Ridge direto, {v11}: {len(a)} réplicas idênticas às do T10 ({v10})")
pr = pd.read_csv(OUT / "principal_T11.csv")
p10 = pd.read_csv(ROOT / "outputs" / "T10" / "principal_T10.csv")
check(np.allclose(pr[pr.regressor == "Ridge_direto"].sort_values("h").beta,
                  p10[p10.modelo == "Ridge"].sort_values("h").beta, atol=1e-14, rtol=0), "β̂ do Ridge direto = T10")
rb = pd.read_csv(OUT / "bootstrap_resumo_T11.csv")
r11 = rb[(rb.regressor == "Ridge_direto") & (rb.variante == "dois_niveis") & (rb.bloco == 60)].sort_values("h")
r10 = pd.read_csv(ROOT / "outputs" / "T10" / "bootstrap_resumo_T10.csv")
r10 = r10[(r10.variante == "Ridge_alpha_fixo") & (r10.bloco == 60)].sort_values("h")
if int(r11.B.iloc[0]) == 999:
    cols = ["beta", "media_boot", "ep_boot", "ep_corrigido", "ep_so_segundo_nivel", "p_boot", "ic_inf", "ic_sup"]
    check(np.allclose(r11[cols].to_numpy(), r10[cols].to_numpy(), atol=1e-12, rtol=1e-10),
          "resumo do Ridge direto (B = 999) igual ao do T10")
else:
    print(f"  (info) B = {int(r11.B.iloc[0])}: comparação do resumo com o T10 só na rodada completa")

print("6) Teste conjunto")
rng = np.random.default_rng(1)
Sig = 0.9 ** np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
L = np.linalg.cholesky(Sig)
bh = np.array([-1.0, -0.8, -0.5, -0.4, -0.3, -0.2])
Bs = 0.6 * bh + rng.standard_normal((999, 6)) @ L.T * 0.3
G, ph, _ = T.wald_conjunto(Bs, bh)
m, S = Bs.mean(0), np.cov(Bs, rowvar=False)
Li = np.diag(1 / (m / bh))
W_lam = bh @ np.linalg.solve(Li @ S @ Li, bh)
check(abs(G["W"] - W_lam) < 1e-8 * W_lam, f"(a) forma coerente = forma com Λ (W = {G['W']:.3f})")
G1, ph1, _ = T.wald_conjunto(Bs[:, :1], bh[:1])
check(abs(G1["W"] - ph1.z.iloc[0] ** 2) < 1e-10, "(b) com 1 horizonte, W = z²")
rej_w = rej_t = 0
n_sim = 400
for s in range(n_sim):
    b_hat = L @ rng.standard_normal(6)
    Bstar = b_hat + rng.standard_normal((499, 6)) @ L.T
    Gs, _, _ = T.wald_conjunto(Bstar, b_hat, coerente=False)
    rej_w += Gs["p_wald"] < 0.05
    rej_t += Gs["p_max_t"] < 0.05
check(0.025 <= rej_w / n_sim <= 0.08 and 0.025 <= rej_t / n_sim <= 0.08,
      f"(c) tamanho a 5% sob H0 ({n_sim} simulações): Wald {rej_w / n_sim:.3f}, max-|t| {rej_t / n_sim:.3f}")
x = prev_t5[prev_t5.modelo == "Ridge"].set_index("date").previsao.sort_index()
dd = df.set_index("date").loc[x.index]
ok = True
for h in T.HS:
    w = T.wald_hac_empilhado(x.to_numpy(), dd[f"ret_fut_{h}d"].to_numpy(), h + 1)
    ep = mqo_newey_west(dd[f"ret_fut_{h}d"], x, h=h).bse.iloc[1]
    ok &= abs(w["ep"][0] - ep) < 1e-10 * ep
check(ok, "(d) HAC empilhado, 1 horizonte e maxlags h+1: EP = hac_utils.mqo_newey_west, em todo h")

print("7) Leitura B: média histórica embargada")
ok = True
for h in (1, 30, 60):
    for o in gerar_divisoes(len(df), h=h, janela="expansiva").origens[::10]:
        mh = df[f"ret_fut_{h}d"].to_numpy()[o.treino].mean()
        g = pb[(pb.h == h) & (pb.origem == df.date[o.t])]
        ok &= np.allclose(g.media_historica, mh) and o.treino[-1] == o.t - h
check(ok, "média histórica = média de ret_fut_h com s <= t − h (h = 1, 30, 60; 4 origens cada)")

print("8) Saídas")
n = bt.groupby(["regressor", "variante", "bloco"]).size()
print("  (info) réplicas: " + "; ".join(f"{a}/{b}/{c}: {v}" for (a, b, c), v in n.items()))
check(set(T.REGRESSORES) == set(bt.regressor) and (n > 0).all(), "bootstrap para os quatro regressores")
g10 = pd.read_csv(OUT / "teste_conjunto_T10.csv")
check(len(g10) == 6, f"teste conjunto do T10: {len(g10)} variantes")
pl = pd.read_csv(OUT / "placebo_T11.csv")
check(bool(pl.passa.all()), f"placebo (espaço da RV): mediana do R² <= {T.TOL_PLACEBO} em "
      + ", ".join(f"{r.modelo} ({r.r2_placebo_mediana:.3f})" for r in pl.itertuples()))
plb = pd.read_csv(OUT / "placebo_arvore_retorno_T11.csv")
check(bool(plb.passa.all()), f"placebo da leitura B: mediana <= {T.TOL_PLACEBO} em todo h ("
      + ", ".join(f"{r.h}: {r.median:.3f}" for r in plb.itertuples()) + ")")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
