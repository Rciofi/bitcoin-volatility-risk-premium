"""
test_A1.py — testes das estratégias do apêndice (A1)

  1. Sem look-ahead: trocar o sinal e os preços depois de t0 por ruído não muda
     nenhum limiar nem posição até t0 (Ridge e proxy).
  2. Limiar: q(t) = np.quantile(sinal[:t+1], q) (observações até t, inclusive);
     NaN antes da 252ª observação.
  3. Alinhamento: o retorno bruto da estratégia em t é pos(t) × (close(t+1)/close(t) − 1),
     montado à mão a partir de data/btc_prices.csv; nenhuma data avaliada usa preço
     depois do fim da amostra.
  4. Embargo: toda previsão do Ridge usada vem de uma origem a 0–29 dias.
  5. Custos: líquido = bruto − |Δpos| × custo, com a entrada inicial cobrada; o Sharpe
     bruto vendido/neutro é o negativo do comprado/neutro; Sortino usual conferido à
     mão; na decomposição, a última etapa só troca a fórmula do Sortino.
  6. Períodos: Ridge com 735 dias (burn-in 252) e 861 (burn-in 126); proxy com 1.554;
     ponte nas mesmas datas do Ridge; regimes só a partir de 21/06/2023.
  7. Réplica: a lógica antiga com os dados pré-T0 reproduz as Tabs. 7.1 a 7.4
     publicadas (4 casas decimais).
  8. Equivalência com o código antigo: com os dados pós-T0, a posição antiga em t
     (sinal de t−1) é a posição nova de t−1 -- a regra publicada não tinha vazamento.
     A coluna `ret` antiga é o log-retorno; com expm1(ret) (etapa 3 da decomposição),
     o código antigo fora do burn-in dá exatamente os retornos novos da proxy.

Uso:  python code/test_A1.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # −, × no console mesmo com saída redirecionada (Windows)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estrategias_A1 as E  # noqa: E402

falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


vt, prev, reg, btc = E.carregar()
ridge, proxy, ret_seg, fim = E.sinais(vt, prev)
m, series, datas = E.calcular(vt, prev, reg)
rng = np.random.default_rng(20261001)

print("1) Sem look-ahead")
for nome, s in [("Ridge", ridge), ("proxy", proxy)]:
    t0 = s.index[len(s) * 2 // 3]
    s2 = s.copy()
    s2[s2.index > t0] = rng.normal(0, 50, (s2.index > t0).sum())
    ok = True
    for q in E.QUANTIS:
        for b in (E.BURNIN, E.BURNIN_SENS):
            a = E.posicoes(s, q, b, 1.0)[: t0]
            c = E.posicoes(s2, q, b, 1.0)[: t0]
            la = E.limiar_expansivo(s, q, b)[: t0]
            lc = E.limiar_expansivo(s2, q, b)[: t0]
            ok &= a.equals(c) and la.equals(lc)
    check(ok, f"{nome}: sinal após {t0.date()} trocado por ruído -> limiares e posições até t0 iguais")
t0 = datas["ridge"][300]
r2 = ret_seg.copy()
r2[r2.index > t0] = rng.normal(0, 0.05, (r2.index > t0).sum())
pos = E.posicoes(ridge, E.Q_REF, E.BURNIN, 1.0)
a, _ = E.avaliar(pos, ret_seg, datas["ridge"], 30)
c, _ = E.avaliar(pos, r2, datas["ridge"], 30)
check(a[a.index <= t0].equals(c[c.index <= t0]),
      "retornos depois de t0 trocados por ruído -> retornos da estratégia até a posição de t0 iguais")

print("2) Limiar em janela expansiva, com t incluído")
ok = True
for s in (ridge, proxy):
    v = s.to_numpy()
    for q in E.QUANTIS:
        lim = E.limiar_expansivo(s, q, E.BURNIN).to_numpy()
        ok &= bool(np.isnan(lim[: E.BURNIN - 1]).all())
        for i in rng.choice(np.arange(E.BURNIN - 1, len(v)), 40, replace=False):
            ok &= bool(np.isclose(lim[i], np.quantile(v[: i + 1], q), atol=1e-10, rtol=0))
check(ok, "q(t) = np.quantile(sinal[:t+1], q); NaN antes da 252ª observação")

print("3) Alinhamento com os preços")
close = btc.set_index("date").close
for v in ["ridge", "proxy"]:
    s = ridge if v == "ridge" else proxy
    for direcao, sentido in E.DIRECOES.items():
        p = E.posicoes(s, E.Q_REF, E.BURNIN, sentido)
        r = series[(v, E.Q_REF, direcao)]
        mao = [p[t] * (close[t + pd.Timedelta(days=1)] / close[t] - 1) for t in r.index]
        check(np.allclose(r.to_numpy(), mao, atol=1e-14, rtol=0),
              f"{v}, {direcao}: retorno(t) = pos(t) × (close(t+1)/close(t) − 1), montado à mão")
ultimo = max(d[-1] for d in datas.values())
check(ultimo + pd.Timedelta(days=1) <= fim,
      f"última posição avaliada em {ultimo.date()}: o retorno usado termina em {fim.date()} (fim da amostra)")

print("4) Embargo das previsões")
lag = (prev.date - prev.origem).dt.days
check(len(prev) == 987 and bool(lag.between(0, 29).all()), "987 previsões do Ridge, origem a 0–29 dias")

print("5) Custos e direções")
p = E.posicoes(ridge, E.Q_REF, E.BURNIN, 1.0).loc[datas["ridge"]]
bruto, _ = E.avaliar(p, ret_seg, datas["ridge"], 0)
liq, _ = E.avaliar(p, ret_seg, datas["ridge"], 30)
dp = np.abs(np.diff(np.r_[0.0, p.to_numpy()]))
check(np.allclose(liq.to_numpy(), bruto.to_numpy() - dp * 0.003, atol=1e-15, rtol=0),
      "30 bps: líquido = bruto − |Δpos| × 0,003 (entrada inicial cobrada)")
x = m[(m.regime == "todos") & (m.custo_bps == 0) & m.quantil.notna()]
pares = x.pivot_table(index=["variante", "quantil"], columns="direcao", values="sharpe")
check(np.allclose(pares.comprado, -pares.vendido, atol=1e-12), "Sharpe bruto vendido/neutro = −comprado/neutro")
r = bruto.to_numpy()
mao = r.mean() * 365 / (np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(365))
check(np.isclose(E.metricas(bruto, p)["sortino"], mao, atol=1e-12, rtol=0),
      "Sortino usual = média × 365 / (√média(min(r, 0)²) × √365), todos os dias")
dq = E.decomposicao_q80(E.ler_publicado(), *[E.replica_cap7(*E.carregar_pre_t0())] * 3, m)
ok = dq.iloc[-1].drop(["etapa", "sortino"]).equals(dq.iloc[-2].drop(["etapa", "sortino"]))
check(ok and dq.etapa.iloc[-1].startswith("7.") and np.isclose(dq.sortino.iloc[-1], mao),
      "decomposição: a última etapa (7) só troca o Sortino (fórmula antiga até a 6)")

print("6) Períodos")
n = {v: len(d) for v, d in datas.items()}
check(n["ridge"] == 735 and n["ridge_b126"] == 861 and n["proxy"] == 1554,
      f"dias avaliados: Ridge {n['ridge']}, Ridge burn-in 126 {n['ridge_b126']}, proxy {n['proxy']}")
check(datas["ponte"].equals(datas["ridge"]), "ponte avaliada nas mesmas datas do Ridge")
rg = m[m.regime != "todos"]
check(bool((pd.to_datetime(rg.inicio) >= E.INICIO_REGIME).all()), "regimes só a partir de 21/06/2023")

print("7) Réplica do publicado")
pub = E.ler_publicado()
vr_pre, btc_pre = E.carregar_pre_t0()
rep = E.replica_cap7(vr_pre, btc_pre)
for t in ["7.1", "7.2", "7.2b", "7.3", "7.4"]:
    ok, nn = set(pub[t]) == set(rep[t]), 0
    for k in pub[t]:
        for a, b in zip(rep[t][k], pub[t][k]):
            if np.isnan(a) and np.isnan(b):
                continue
            ok &= abs(a - b) <= 5e-5 + 1e-12
            nn += 1
    check(ok, f"Tab. {t}: {nn} números reproduzidos (4 casas)")

print("8) Equivalência com a regra antiga (dados pós-T0)")
vr = pd.read_csv(Path(E.ROOT) / "data" / "vrp_with_regimes.csv", parse_dates=["date"]).sort_values("date")
check(np.allclose(vr.vrp_30d, vr.rv_30d - vr.iv_30d, atol=1e-10), "vrp_30d = rv_30d − iv_30d (proxy retrospectiva)")
for q in E.QUANTIS:
    antigo = E._sinal_antigo(vr.vrp_30d.reset_index(drop=True), q).to_numpy()
    novo = E.posicoes(proxy, q, E.BURNIN, 1.0).fillna(0.0).to_numpy()
    check(np.array_equal(antigo[1:], novo[:-1]), f"q{int(q * 100)}: posição antiga em t = posição nova em t−1")
c = vr.close.to_numpy()
check(np.allclose(vr.ret.to_numpy()[1:], np.log(c[1:] / c[:-1]), atol=1e-12),
      "a coluna `ret` antiga é o log-retorno, ln(close(t)/close(t−1))")
d = vr.reset_index(drop=True)
sig = E._sinal_antigo(d.vrp_30d, E.Q_REF)
antigo = (sig * np.expm1(d.ret)).to_numpy()[E.BURNIN:]
novo = series[("proxy", E.Q_REF, "comprado")].to_numpy()
check(np.allclose(antigo, novo, atol=1e-12, rtol=0),
      "com expm1(ret), o q80 antigo fora do burn-in = q80 novo da proxy (mesmos 1.554 retornos)")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
