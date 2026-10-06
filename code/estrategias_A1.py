"""
estrategias_A1.py — estratégias de negociação do apêndice (A1 do plano de revisão)

Refaz as estratégias do antigo Cap. 7 (scripts/regenerate_cap7.py e scripts/cap7/,
intactos) com o BVRP previsto e sem look-ahead. O BVRP prospectivo,
RV(t+1 a t+30) − IV_t, só é conhecido em t+30 e não entra como sinal.

Sinais (todos conhecidos no fim de t)
  PRINCIPAL  previsão do Ridge do T5, janela expansiva (data/previsoes_bvrp_T5.csv;
             987 datas, 21/06/2023 a 03/03/2026; treino com s <= origem − 30).
  ROBUSTEZ   proxy retrospectiva vrp_30d = RV(t−29 a t) − IV_t, amostra completa
             (24/03/2021 a 03/03/2026). É o sinal do Cap. 7 publicado.
  PONTE      a proxy avaliada nas mesmas datas do principal (limiar com o histórico
             da proxy desde 2021): isola o efeito do sinal do efeito do período.

Regra
  q(t) = quantil q de {sinal_s : s <= t} (janela expansiva; o valor de t entra,
  pois é conhecido no fim de t), definido a partir da 252ª observação do sinal
  (126 como sensibilidade no Ridge). Sinal alto: sinal(t) >= q(t), com
  q = 60, 70, 80 e 90% (80% é a referência, como no publicado). As duas direções
  são reportadas, fixadas antes dos resultados:
    comprado/neutro  posição +1 com sinal alto, 0 caso contrário (regra publicada);
    vendido/neutro   posição −1 com sinal alto, 0 caso contrário.
  A posição de t rende ret(t+1) = close(t+1)/close(t) − 1. A posição da última
  data é descartada: seu retorno cairia fora da amostra.
  Custos: 0, 10 e 30 bps unilaterais sobre |pos(t) − pos(t−1)|, com pos = 0 antes
  da primeira data avaliada, descontados do retorno de t+1.

Regimes: regime_alta_fixo do T9 (corte de 37,3, vigente em t), só a partir de
21/06/2023 (o corte usa a 1ª janela de estimação, até 22/04/2023). A estratégia
fica neutra nos dias do outro regime.

Métricas (365 dias por ano): retorno anualizado (geométrico), volatilidade, Sharpe
aritmético (média × 365 / (dp × √365); principal), Sortino usual (média × 365 /
(√média(min(r, 0)²) × √365), com todos os dias), drawdown máximo (a partir do
valor inicial 1), giro (média de |Δpos| × 365, operações por ano), número de
operações e % do tempo posicionado. O Sharpe geométrico (retorno anualizado /
volatilidade) e o Sortino do publicado (dp só dos retornos negativos) aparecem
apenas nas decomposições; a troca do Sortino é a última etapa de cada uma.

Decomposição publicado → novo: réplica do publicado com os dados anteriores ao T0
(git show 11a364f^), depois uma mudança por etapa. Vale também para a Tab. 7.1.
Achado do A1: a coluna `ret` de vrp_with_regimes.csv é o log-retorno, e o código
antigo a compunha como retorno simples nas Tabs. 7.2–7.4; a troca é uma etapa própria.
Aqui o retorno vem dos preços: close(t+1)/close(t) − 1.

Saídas em outputs/A1/ (nada em figs/ nem em tables/).

Uso:  python code/estrategias_A1.py
"""

import io
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")   # −, √ no console mesmo com saída redirecionada (Windows)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "A1")
ANN = 365
QUANTIS = [0.60, 0.70, 0.80, 0.90]
Q_REF = 0.80
CUSTOS = [0, 10, 30]                 # bps unilaterais
BURNIN = 252
BURNIN_SENS = 126
INICIO_REGIME = pd.Timestamp("2023-06-21")   # 1ª origem do T5; o corte do T9 usa dados até 22/04/2023
COMMIT_PRE_T0 = "11a364f^"           # último estado dos dados antes do T0 (Cap. 7 publicado)
DIRECOES = {"comprado": 1.0, "vendido": -1.0}
ROTULO_DIRECAO = {"comprado": "Comprado/neutro", "vendido": "Vendido/neutro",
                  "compra_e_manutencao": "Compra e manutenção"}
ROTULO_VARIANTE = {"ridge": "Ridge (T5), burn-in 252", "ridge_b126": "Ridge (T5), burn-in 126",
                   "proxy": "Proxy, amostra completa", "ponte": "Proxy, datas do Ridge"}
AZUL, LARANJA, VERDE, ROXO, CINZA = "#2b6cb0", "#d95f02", "#1b9e77", "#7570b3", "#7f7f7f"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    vt = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    vt = vt.sort_values("date").reset_index(drop=True)
    prev = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
    prev = prev[(prev.janela == "expansiva") & (prev.modelo == "Ridge")].sort_values("date")
    reg = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])
    btc = pd.read_csv(os.path.join(ROOT, "data", "btc_prices.csv"), parse_dates=["date"]).sort_values("date")
    return vt, prev, reg, btc


def carregar_pre_t0():
    """Dados usados no Cap. 7 publicado (antes do preenchimento do buraco de mar/2023)."""
    def ler(caminho):
        txt = subprocess.run(["git", "show", f"{COMMIT_PRE_T0}:{caminho}"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        return pd.read_csv(io.BytesIO(txt), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return ler("data/vrp_with_regimes.csv"), ler("data/btc_prices.csv")


# ---------------------------------------------------------------------------
# Regra
# ---------------------------------------------------------------------------
def limiar_expansivo(sinal, q, burnin):
    """q(t) = quantil q de {sinal_s : s <= t}; NaN enquanto houver menos de burnin observações."""
    return sinal.expanding(min_periods=burnin).quantile(q)


def posicoes(sinal, q, burnin, direcao):
    """Posição definida no fim de t: direcao × 1{sinal(t) >= q(t)}; NaN durante o burn-in."""
    lim = limiar_expansivo(sinal, q, burnin)
    return direcao * (sinal >= lim).astype(float).where(lim.notna())


def avaliar(pos, ret_seguinte, datas, custo_bps):
    """Retorno líquido de cada posição: pos(t) × ret(t+1) − custo × |Δpos(t)|, indexado por t."""
    p = pos.loc[datas].astype(float)
    dp = p.diff().abs()
    dp.iloc[0] = abs(p.iloc[0])
    return p * ret_seguinte.loc[datas] - dp * custo_bps / 1e4, p


def metricas(r, p):
    n = len(r)
    vol = r.std() * np.sqrt(ANN)
    ret_a = (1 + r).prod() ** (ANN / n) - 1
    desvio_neg = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(ANN)   # usual: todos os dias
    neg = r[r < 0]
    dp_neg = neg.std() * np.sqrt(ANN) if len(neg) > 1 else np.nan        # fórmula do publicado
    valor = (1 + r).cumprod()
    pico = np.maximum(valor.cummax(), 1.0)
    dp = p.diff().abs()
    dp.iloc[0] = abs(p.iloc[0])
    return {"inicio": r.index.min().date(), "fim": r.index.max().date(), "n_dias": n,
            "ret_anual": ret_a, "vol_anual": vol,
            "sharpe": r.mean() * ANN / vol if vol > 0 else np.nan,
            "sharpe_geom": ret_a / vol if vol > 0 else np.nan,
            "sortino": r.mean() * ANN / desvio_neg if desvio_neg > 0 else np.nan,
            "sortino_antigo": r.mean() * ANN / dp_neg if dp_neg > 0 else np.nan,
            "max_drawdown": float((valor / pico - 1).min()),
            "giro_anual": dp.mean() * ANN, "n_operacoes": int(dp.sum()),
            "pct_tempo": float((p != 0).mean()), "n_dias_posicionado": int((p != 0).sum())}


def sinais(vt, prev):
    base = vt.set_index("date")
    ret_seguinte = (base.close.shift(-1) / base.close - 1).rename("ret_seguinte")
    ridge = prev.set_index("date").previsao.rename("ridge")
    return ridge, base.vrp_30d.rename("proxy"), ret_seguinte, base.index.max()


def calcular(vt, prev, reg):
    """Todas as combinações (variante × q × direção × custo × regime) em formato longo."""
    ridge, proxy, ret_seg, fim = sinais(vt, prev)
    regime = reg.set_index("date").regime_alta_fixo
    spec = {"ridge": (ridge, BURNIN), "ridge_b126": (ridge, BURNIN_SENS),
            "proxy": (proxy, BURNIN), "ponte": (proxy, BURNIN)}
    datas = {}
    for v, (s, b) in spec.items():
        d = posicoes(s, Q_REF, b, 1.0).dropna().index      # o burn-in não depende de q
        datas[v] = d[d < fim]
    datas["ponte"] = datas["ridge"]

    linhas, series = [], {}
    for v, (s, b) in spec.items():
        d = datas[v]
        recortes = {"todos": d}
        if v in ("ridge", "proxy"):
            dr = d[d >= INICIO_REGIME]
            recortes.update({"alta": dr, "baixa": dr})
        for recorte, dd in recortes.items():
            if recorte != "todos":
                assert regime.reindex(dd).notna().all()
            filtro = (pd.Series(1.0, index=dd) if recorte == "todos"
                      else (regime.reindex(dd) == (1 if recorte == "alta" else 0)).astype(float))
            bh = pd.Series(1.0, index=dd) * filtro
            r, p = avaliar(bh, ret_seg, dd, 0)
            linhas.append({"variante": v, "regime": recorte, "direcao": "compra_e_manutencao",
                           "quantil": np.nan, "custo_bps": 0, **metricas(r, p)})
            for q in QUANTIS:
                for nome, sentido in DIRECOES.items():
                    pos = posicoes(s, q, b, sentido).reindex(dd) * filtro
                    for c in CUSTOS:
                        r, p = avaliar(pos, ret_seg, dd, c)
                        linhas.append({"variante": v, "regime": recorte, "direcao": nome,
                                       "quantil": q, "custo_bps": c, **metricas(r, p)})
                        if recorte == "todos" and c == 0:
                            series[(v, q, nome)] = r
            if recorte == "todos":
                series[(v, None, "compra_e_manutencao")] = avaliar(bh, ret_seg, dd, 0)[0]
    return pd.DataFrame(linhas), series, datas


# ---------------------------------------------------------------------------
# Réplica do Cap. 7 publicado (lógica de scripts/regenerate_cap7.py, sem gravar nada)
# ---------------------------------------------------------------------------
def _perf_antigo(r):
    r = r.dropna()
    n = len(r)
    ret_a = (1 + r).prod() ** (ANN / n) - 1
    vol = r.std() * np.sqrt(ANN)
    cum = (1 + r).cumprod()
    return ret_a, vol, ret_a / vol, float(((cum / cum.cummax()) - 1).min())


def _sortino_antigo(r):
    r = r.dropna()
    return float(r.mean() * ANN / (r[r < 0].std() * np.sqrt(ANN)))


def _sinal_antigo(x, q):
    th = x.expanding(min_periods=BURNIN).quantile(q)
    sig = (x.shift(1) >= th.shift(1)).astype(float).fillna(0.0)
    sig.iloc[:BURNIN] = 0.0
    return sig


def _custo_antigo(sr, sig, bps):
    return sr - sig.diff().abs().fillna(abs(sig.iloc[0])) * bps / 1e4


def replica_cap7(vr, btc, simples=False):
    """Tabs. 7.1 a 7.4 pela lógica antiga. Rótulos e colunas iguais aos .tex publicados.

    A coluna `ret` de vrp_with_regimes.csv é o LOG-retorno (build_vrp_dataset.py:53), e o
    código antigo a compunha como retorno simples ((1 + r).prod(), custos subtraídos de r).
    simples=True troca só isso: ret -> expm1(ret), o retorno simples exato, nas mesmas datas.
    A Tab. 7.1 já usava retorno simples (pct_change de btc_prices.csv) e não muda.
    """
    fim = vr.date.max()
    b = btc[btc.date <= fim].copy()
    b["ret"] = b.close.pct_change()
    b = b.dropna(subset=["ret"])
    r, v, s, m = _perf_antigo(b.ret)
    t71 = {"Retorno Anualizado": [r], "Volatilidade Anualizada": [v], "Sharpe Ratio": [s],
           "Sortino Ratio": [_sortino_antigo(b.ret)], "Max Drawdown": [m]}
    extra71 = {"n_dias": len(b), "maior_ret_diario": b.ret.max(),
               "data_maior_ret": b.date[b.ret.idxmax()].date(),
               "sharpe_arit": b.ret.mean() * ANN / (b.ret.std() * np.sqrt(ANN))}

    d = vr.dropna(subset=["vrp_30d", "ret"]).reset_index(drop=True)
    if simples:
        d["ret"] = np.expm1(d.ret)
    sub = d.ret.iloc[BURNIN:]
    rb, vb, sb, mb = _perf_antigo(sub)
    linha_bh = [rb, vb, sb, np.nan, np.nan, mb, _sortino_antigo(sub)]

    t72, t72b, t73, sinais_q = {}, {}, {}, {}
    for q in QUANTIS:
        sig = _sinal_antigo(d.vrp_30d, q)
        sr = sig * d.ret
        r, v, s, m = _perf_antigo(sr)
        sh = {c: _perf_antigo(_custo_antigo(sr, sig, c))[2] for c in (0, 5, 10)}
        t73[f"q{int(q * 100)}\\%"] = [r, v, sh[0], sh[5], sh[10], m, _sortino_antigo(sr),
                                      sig.diff().abs().mean(), sig.mean()]
        sinais_q[q] = sig
        if q == Q_REF:
            t72["BVRP q80\\%"] = [r, v, sh[0], sh[5], sh[10], m, _sortino_antigo(sr), sig.mean()]
            for c in (0, 5, 10):
                rn, vn, snn, mn = _perf_antigo(_custo_antigo(sr, sig, c))
                t72b[f"{c}\\,bps"] = [rn, vn, snn, _sortino_antigo(_custo_antigo(sr, sig, c)), mn,
                                      int(sig.diff().abs().sum())]
    t72["Buy \\& Hold"] = linha_bh + [1.0]
    t73["Buy \\& Hold"] = linha_bh + [np.nan, 1.0]

    sig80, th80 = sinais_q[Q_REF], d.vrp_30d.expanding(min_periods=BURNIN).quantile(Q_REF)
    terc = pd.qcut(d.rv_30d, q=3, labels=["Baixo", "Medio", "Alto"])
    t74, dias74 = {}, {}
    for reg_, rot in [("Baixo", "Baixa RV"), ("Medio", "Media RV"), ("Alto", "Alta RV")]:
        sig = ((d.vrp_30d.shift(1) >= th80.shift(1)) & (terc == reg_)).astype(float).fillna(0.0)
        sig.iloc[:BURNIN] = 0.0
        sr = sig * d.ret
        r, v, s, m = _perf_antigo(sr)
        t74[rot] = [r, v, s, m, _sortino_antigo(sr), sig.diff().abs().mean(), sig.mean()]
        dias74[rot] = int(sig.sum())

    # q80 também com as métricas novas sobre a mesma série antiga (para a decomposição)
    sr80 = sig80 * d.ret
    q80 = {}
    for c in (0, 10, 30):
        x = _custo_antigo(sr80, sig80, c)
        r, v, s, m = _perf_antigo(x)
        q80[c] = {"n_dias": len(x), "ret_anual": r, "vol_anual": v, "sharpe_geom": s,
                  "sharpe": x.mean() * ANN / (x.std() * np.sqrt(ANN)), "sortino": _sortino_antigo(x),
                  "max_drawdown": m, "pct_tempo": sig80.mean(), "n_operacoes": int(sig80.diff().abs().sum())}
    q80_bh = {"n_dias": len(sub), "sharpe_geom": sb, "sharpe": sub.mean() * ANN / (sub.std() * np.sqrt(ANN)),
              "max_drawdown": mb}
    return {"7.1": t71, "7.2": t72, "7.2b": t72b, "7.3": t73, "7.4": t74,
            "extra71": extra71, "dias74": dias74, "q80": q80, "q80_bh": q80_bh,
            "sig80": sig80, "datas": d.date, "inicio": d.date.iloc[BURNIN].date(), "fim": fim.date()}


def ler_publicado():
    """Linhas numéricas das tabelas publicadas (hoje em archive/tables/estrategias/; rótulo -> valores)."""
    arquivos = {"7.1": "tab_perf_buy_hold.tex", "7.2": "tab_perf_vrp_quantile.tex",
                "7.2b": "tab_custos.tex", "7.3": "tab_perf_bvrp_multi_quantile.tex",
                "7.4": "tab_perf_bvrp_regimes.tex"}
    out = {}
    for tab, arq in arquivos.items():
        with open(os.path.join(ROOT, "archive", "tables", "estrategias", arq), encoding="utf-8") as fh:
            corpo = fh.read().split(r"\midrule", 1)[1]
        linhas = {}
        for ln in corpo.splitlines():
            if "&" not in ln:
                continue
            ln = ln.replace(r"\\", "").replace("}", "").replace(r"\&", "\x00")   # "Buy \& Hold"
            cel = [c.strip().replace("\x00", r"\&") for c in ln.split("&")]
            linhas[cel[0]] = [np.nan if c == "--" else float(c) for c in cel[1:]]
        out[tab] = linhas
    return out


# ---------------------------------------------------------------------------
# Decomposições
# ---------------------------------------------------------------------------
def decomposicao_q80(pub, rep_pre, rep_pos, rep_simples, m):
    """Comprado/neutro, q80: uma mudança por etapa, do publicado ao Ridge."""
    cols = ["n_dias", "ret_anual", "vol_anual", "sharpe", "sharpe_geom", "sortino", "max_drawdown",
            "pct_tempo", "n_operacoes"]
    p72, p72b = pub["7.2"]["BVRP q80\\%"], pub["7.2b"]
    linhas = [{"etapa": "0. Publicado (Tabs. 7.2 e 7.2b)", "sinal": "proxy vrp_30d", "inicio": "2021-12-01",
               "fim": "2026-03-03", "n_dias": np.nan, "ret_anual": p72[0], "vol_anual": p72[1], "sharpe": np.nan,
               "sharpe_geom": p72[2], "sortino": p72[6], "max_drawdown": p72[5], "pct_tempo": p72[7],
               "n_operacoes": p72b["0\\,bps"][5], "sharpe_10bps": np.nan, "sharpe_geom_10bps": p72[4],
               "bh_sharpe": np.nan, "bh_sharpe_geom": pub["7.2"]["Buy \\& Hold"][2],
               "bh_max_drawdown": pub["7.2"]["Buy \\& Hold"][5]}]
    for etapa, rep in [("1. Réplica: código antigo, dados pré-T0", rep_pre),
                       ("2. Código antigo, dados pós-T0", rep_pos),
                       ("3. Código antigo, pós-T0, retorno simples no lugar do log-retorno", rep_simples)]:
        q = rep["q80"]
        linhas.append({"etapa": etapa, "sinal": "proxy vrp_30d", "inicio": rep["inicio"], "fim": rep["fim"],
                       **{c: q[0][c] for c in cols}, "sharpe_10bps": q[10]["sharpe"],
                       "sharpe_geom_10bps": q[10]["sharpe_geom"], "bh_sharpe": rep["q80_bh"]["sharpe"],
                       "bh_sharpe_geom": rep["q80_bh"]["sharpe_geom"],
                       "bh_max_drawdown": rep["q80_bh"]["max_drawdown"]})
    for etapa, v in [("4. Regra nova: avaliação sem os dias de burn-in, drawdown desde 1", "proxy"),
                     ("5. Proxy só nas datas do Ridge (ponte)", "ponte"),
                     ("6. Ridge do T5 (principal)", "ridge")]:
        def sel(direcao, custo):
            x = m[(m.variante == v) & (m.regime == "todos") & (m.direcao == direcao) & (m.custo_bps == custo)]
            if direcao != "compra_e_manutencao":
                x = x[x.quantil == Q_REF]
            return x.iloc[0]
        a, a10, bh = sel("comprado", 0), sel("comprado", 10), sel("compra_e_manutencao", 0)
        linhas.append({"etapa": etapa, "sinal": "Ridge (T5)" if v == "ridge" else "proxy vrp_30d",
                       "inicio": a.inicio, "fim": a.fim, **{c: a[c] for c in cols},
                       "sortino": a.sortino_antigo,      # a troca de fórmula é a última etapa
                       "sharpe_10bps": a10.sharpe, "sharpe_geom_10bps": a10.sharpe_geom,
                       "bh_sharpe": bh.sharpe, "bh_sharpe_geom": bh.sharpe_geom, "bh_max_drawdown": bh.max_drawdown})
    ultima = dict(linhas[-1])
    ultima.update({"etapa": "7. Ridge do T5, Sortino usual (desvio abaixo de zero, todos os dias)",
                   "sortino": a.sortino})
    linhas.append(ultima)
    return pd.DataFrame(linhas)


def decomposicao_regimes(pub, rep_pre, rep_pos, rep_simples, m):
    cols = ["ret_anual", "vol_anual", "sharpe_geom", "max_drawdown", "sortino", "turnover", "pct_tempo"]
    linhas = []
    for etapa, fonte, dias in [("0. Publicado (Tab. 7.4)", pub["7.4"], None),
                               ("1. Réplica: código antigo, dados pré-T0", rep_pre["7.4"], rep_pre["dias74"]),
                               ("2. Código antigo, dados pós-T0 (tercis por qcut)", rep_pos["7.4"], rep_pos["dias74"]),
                               ("3. Código antigo, pós-T0, retorno simples", rep_simples["7.4"],
                                rep_simples["dias74"])]:
        for rot, vals in fonte.items():
            linhas.append({"etapa": etapa, "regime": rot, **dict(zip(cols, vals)),
                           "n_dias_posicionado": np.nan if dias is None else dias[rot]})
    for etapa, v in [("4. Proxy, regime do T9 (corte 37,3), desde 21/06/2023", "proxy"),
                     ("5. Ridge do T5, regime do T9", "ridge")]:
        for rg in ("baixa", "alta"):
            a = m[(m.variante == v) & (m.regime == rg) & (m.direcao == "comprado") & (m.quantil == Q_REF)
                  & (m.custo_bps == 0)].iloc[0]
            linhas.append({"etapa": etapa, "regime": f"{rg.capitalize()} volatilidade",
                           "ret_anual": a.ret_anual, "vol_anual": a.vol_anual, "sharpe": a.sharpe,
                           "sharpe_geom": a.sharpe_geom, "max_drawdown": a.max_drawdown,
                           "sortino": a.sortino_antigo, "sortino_usual": a.sortino,
                           "turnover": a.giro_anual / ANN, "pct_tempo": a.pct_tempo,
                           "n_dias_posicionado": a.n_dias_posicionado})
    return pd.DataFrame(linhas)


def decomposicao_bh(pub, rep_pre, rep_pos, vt, btc):
    p = pub["7.1"]
    linhas = [{"etapa": "0. Publicado (Tab. 7.1)", "n_dias": np.nan, "ret_anual": p["Retorno Anualizado"][0],
               "vol_anual": p["Volatilidade Anualizada"][0], "sharpe": np.nan, "sharpe_geom": p["Sharpe Ratio"][0],
               "sortino": p["Sortino Ratio"][0], "max_drawdown": p["Max Drawdown"][0]}]
    for etapa, rep in [("1. Réplica: código antigo, dados pré-T0", rep_pre),
                       ("2. Código antigo, dados pós-T0", rep_pos)]:
        t, e = rep["7.1"], rep["extra71"]
        linhas.append({"etapa": etapa, "n_dias": e["n_dias"], "ret_anual": t["Retorno Anualizado"][0],
                       "vol_anual": t["Volatilidade Anualizada"][0], "sharpe": e["sharpe_arit"],
                       "sharpe_geom": t["Sharpe Ratio"][0], "sortino": t["Sortino Ratio"][0],
                       "max_drawdown": t["Max Drawdown"][0], "maior_ret_diario": e["maior_ret_diario"],
                       "data_maior_ret": e["data_maior_ret"]})
    # Métricas novas: posição de t rende ret(t+1), de 17/08/2017 (1º preço) a 03/03/2026
    b = btc[btc.date <= vt.date.max()].set_index("date").close
    ret_seg = b.shift(-1) / b - 1
    datas = b.index[:-1]
    r, pp = avaliar(pd.Series(1.0, index=datas), ret_seg, datas, 0)
    mm = metricas(r, pp)
    base = {k: mm[k] for k in ["n_dias", "ret_anual", "vol_anual", "sharpe", "sharpe_geom", "max_drawdown"]}
    extra = {"maior_ret_diario": r.max(), "data_maior_ret": (r.idxmax() + pd.Timedelta(days=1)).date()}
    linhas.append({"etapa": "3. Métricas novas (Sharpe aritmético, drawdown desde 1), dados pós-T0",
                   **base, "sortino": mm["sortino_antigo"], **extra})
    linhas.append({"etapa": "4. Sortino usual (desvio abaixo de zero, todos os dias)",
                   **base, "sortino": mm["sortino"], **extra})
    return pd.DataFrame(linhas), r


# ---------------------------------------------------------------------------
# Tabelas LaTeX
# ---------------------------------------------------------------------------
def _n(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    x = 100 * x if pct else x
    s = f"{abs(x):.{d}f}".replace(".", "{,}")
    return f"$-${s}" if round(x, d) < 0 else s


def _int(x):
    return "--" if pd.isna(x) else f"{int(x):,}".replace(",", ".")


def _tabela(nome, legenda, rotulo, colfmt, cabecalho, linhas, nota):
    corpo = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}",
             rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", cabecalho + r" \\", r"\midrule"]
    for ln in linhas:
        corpo.append(r"\midrule" if ln == "MIDRULE" else " & ".join(ln) + r" \\")
    corpo += [r"\bottomrule", r"\end{tabular}", "",
              rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {nota}}}", r"\end{table}", ""]
    with open(os.path.join(OUT, nome), "w", encoding="utf-8") as fh:
        fh.write("\n".join(corpo))


def tabelas_tex(m, dq, db, n_ridge):
    def sel(v, direcao, q=Q_REF, custo=0, regime="todos"):
        x = m[(m.variante == v) & (m.regime == regime) & (m.direcao == direcao) & (m.custo_bps == custo)]
        return x.iloc[0] if direcao == "compra_e_manutencao" else x[x.quantil == q].iloc[0]

    nota_comum = (r"Posição definida com informação até o fim de $t$, aplicada ao retorno de $t$ a $t+1$; "
                  r"limiar no quantil $q$ do sinal em janela expansiva (observações até $t$). "
                  r"Custos unilaterais sobre $|\Delta \text{posição}|$. Sharpe aritmético "
                  r"(média $\times$ 365 / desvio-padrão $\times \sqrt{365}$), sem taxa livre de risco; Sortino com "
                  r"o desvio abaixo de zero, $\sqrt{\text{média}(\min(r, 0)^2)}$ sobre todos os dias, anualizado. "
                  r"Com sinal alto, a regra vendida/neutra é a espelho da comprada/neutra.")
    r0 = sel("ridge", "comprado")
    periodo = f"{pd.Timestamp(r0.inicio):%d/%m/%Y} a {pd.Timestamp(r0.fim):%d/%m/%Y}"

    # 1. Principal
    linhas = []
    for direcao in DIRECOES:
        for c in CUSTOS:
            a = sel("ridge", direcao, custo=c)
            linhas.append([ROTULO_DIRECAO[direcao] if c == 0 else "", f"{c}", _n(a.ret_anual, 1, True),
                           _n(a.vol_anual, 1, True), _n(a.sharpe), _n(a.sortino), _n(a.max_drawdown, 1, True),
                           _n(a.giro_anual, 1), _int(a.n_operacoes), _n(a.pct_tempo, 1, True)])
        linhas.append("MIDRULE")
    bh = sel("ridge", "compra_e_manutencao")
    linhas.append([ROTULO_DIRECAO["compra_e_manutencao"], "--", _n(bh.ret_anual, 1, True), _n(bh.vol_anual, 1, True),
                   _n(bh.sharpe), _n(bh.sortino), _n(bh.max_drawdown, 1, True), "--", "--", "100{,}0"])
    _tabela("tab_A1_principal.tex",
            rf"Estratégias com o BVRP previsto pelo Ridge (quantil 80\%, {periodo}, {n_ridge} dias).",
            "tab:A1-principal", "llrrrrrrrr",
            r"Regra & Custo (bps) & Ret. anual (\%) & Vol. (\%) & Sharpe & Sortino & DD máx. (\%) & "
            r"Giro (op./ano) & Operações & Tempo posic. (\%)", linhas,
            nota_comum + r" Burn-in de 252 previsões: a negociação efetiva dura " + f"{n_ridge} dias.")

    # 2. Quantis (Ridge)
    linhas = []
    for direcao in DIRECOES:
        for q in QUANTIS:
            a = [sel("ridge", direcao, q, c) for c in CUSTOS]
            linhas.append([ROTULO_DIRECAO[direcao] if q == QUANTIS[0] else "", f"q{int(q * 100)}"]
                          + [_n(x.sharpe) for x in a]
                          + [_n(a[0].sortino), _n(a[0].max_drawdown, 1, True), _n(a[0].giro_anual, 1),
                             _n(a[0].pct_tempo, 1, True)])
        linhas.append("MIDRULE")
    linhas.append([ROTULO_DIRECAO["compra_e_manutencao"], "--", _n(bh.sharpe), "--", "--", _n(bh.sortino),
                   _n(bh.max_drawdown, 1, True), "--", "100{,}0"])
    _tabela("tab_A1_quantis.tex", rf"Estratégias com o BVRP previsto pelo Ridge, por quantil ({periodo}).",
            "tab:A1-quantis", "llrrrrrrr",
            r"Regra & Quantil & Sharpe (0) & Sharpe (10) & Sharpe (30) & Sortino & DD máx. (\%) & "
            r"Giro (op./ano) & Tempo posic. (\%)", linhas,
            nota_comum + r" Entre parênteses, o custo em bps; Sortino, drawdown, giro e tempo sem custos.")

    # 3. Robustez (q80)
    linhas = []
    for v in ["ridge", "ridge_b126", "proxy", "ponte"]:
        b_ = sel(v, "compra_e_manutencao")
        per = f"{pd.Timestamp(b_.inicio):%d/%m/%Y}--{pd.Timestamp(b_.fim):%d/%m/%Y}"
        for i, direcao in enumerate(["comprado", "vendido", "compra_e_manutencao"]):
            a = [sel(v, direcao, custo=c) for c in (CUSTOS if direcao != "compra_e_manutencao" else [0])]
            sh = [_n(x.sharpe) for x in a] + ["--"] * (3 - len(a))
            linhas.append([ROTULO_VARIANTE[v] if i == 0 else "", per if i == 0 else "",
                           _int(b_.n_dias) if i == 0 else "", ROTULO_DIRECAO[direcao]] + sh
                          + [_n(a[0].max_drawdown, 1, True), _n(a[0].pct_tempo, 1, True)])
        linhas.append("MIDRULE")
    _tabela("tab_A1_robustez.tex", r"Robustez: burn-in, proxy retrospectiva e ponte (quantil 80\%).",
            "tab:A1-robustez", "lllrrrrrr",
            r"Sinal & Período & Dias & Regra & Sharpe (0) & Sharpe (10) & Sharpe (30) & DD máx. (\%) & "
            r"Tempo posic. (\%)", linhas[:-1],
            nota_comum + r" A proxy é $\text{vrp\_30d} = \text{RV}(t-29 \text{ a } t) - \text{IV}_t$, conhecida em $t$; "
            r"na ponte, o limiar da proxy usa o histórico desde 2021 e a avaliação usa as datas do Ridge.")

    # 4. Regimes (q80, desde 21/06/2023)
    linhas = []
    for v in ["ridge", "proxy"]:
        for rg in ("baixa", "alta"):
            for i, direcao in enumerate(["comprado", "vendido"]):
                a, a30 = sel(v, direcao, regime=rg), sel(v, direcao, custo=30, regime=rg)
                linhas.append([("Ridge (T5)" if v == "ridge" else "Proxy") if (rg == "baixa" and i == 0) else "",
                               f"{rg.capitalize()} volatilidade" if i == 0 else "", ROTULO_DIRECAO[direcao],
                               _int(a.n_dias_posicionado), _n(a.ret_anual, 1, True), _n(a.sharpe), _n(a30.sharpe),
                               _n(a.max_drawdown, 1, True)])
        linhas.append("MIDRULE")
    _tabela("tab_A1_regimes.tex",
            r"Estratégias por regime de volatilidade do T9 (quantil 80\%, a partir de 21/06/2023).",
            "tab:A1-regimes", "lllrrrrr",
            r"Sinal & Regime & Regra & Dias posic. & Ret. anual (\%) & Sharpe (0) & Sharpe (30) & DD máx. (\%)",
            linhas[:-1],
            nota_comum + r" Regime de alta se $\text{vh\_30d}(t) > 37{,}3$ (corte do T9, estimado na 1ª janela, "
            r"até 22/04/2023); a estratégia fica neutra nos dias do outro regime. A proxy só é avaliada "
            r"a partir de 21/06/2023.")

    # 5. Decomposição q80
    linhas = []
    for _, a in dq.iterrows():
        linhas.append([a.etapa, _int(a.n_dias), _n(a.ret_anual, 1, True), _n(a.sharpe_geom), _n(a.sharpe),
                       _n(a.sharpe_10bps), _n(a.sortino), _n(a.max_drawdown, 1, True), _n(a.pct_tempo, 1, True),
                       _n(a.bh_sharpe_geom), _n(a.bh_sharpe), _n(a.bh_max_drawdown, 1, True)])
    _tabela("tab_A1_decomposicao.tex",
            r"Do Cap. 7 publicado ao apêndice: comprado/neutro, quantil 80\%, uma mudança por etapa.",
            "tab:A1-decomposicao", "lrrrrrrrrrrr",
            r"Etapa & Dias & Ret. anual (\%) & Sharpe geom. & Sharpe & Sharpe (10) & Sortino & DD máx. (\%) & "
            r"Tempo (\%) & C\&M Sharpe geom. & C\&M Sharpe & C\&M DD (\%)", linhas,
            r"Sharpe geom.: retorno anualizado / volatilidade (fórmula do publicado). Sharpe: aritmético. "
            r"C\&M: compra e manutenção no mesmo período. Nas etapas 0 a 2, as métricas da estratégia "
            r"incluem os 252 dias de burn-in (posição zero), como no código antigo; a partir da 3, não. "
            r"Até a etapa 2, o log-retorno era composto como retorno simples (código antigo); a 3 troca "
            r"só isso. Sortino: fórmula do publicado (desvio-padrão só dos retornos negativos) nas "
            r"etapas 0 a 6; na 7, desvio abaixo de zero sobre todos os dias.")

    # 6. Decomposição da Tab. 7.1
    linhas = []
    for _, a in db.iterrows():
        linhas.append([a.etapa, _int(a.n_dias), _n(a.ret_anual, 1, True), _n(a.vol_anual, 1, True),
                       _n(a.sharpe_geom), _n(a.sharpe), _n(a.sortino), _n(a.max_drawdown, 1, True),
                       _n(a.get("maior_ret_diario", np.nan), 1, True)])
    _tabela("tab_A1_compra_manutencao.tex",
            r"Compra e manutenção, 17/08/2017 a 03/03/2026: publicado, réplica e dados pós-T0.",
            "tab:A1-compra-manutencao", "lrrrrrrrr",
            r"Etapa & Dias & Ret. anual (\%) & Vol. (\%) & Sharpe geom. & Sharpe & Sortino & DD máx. (\%) & "
            r"Maior ret. diário (\%)", linhas,
            r"Antes do T0, o buraco de março/2023 entrava como um único retorno de 32 dias ("
            + _n(db.maior_ret_diario.iloc[1], 1, True) + r"\% em "
            + f"{pd.Timestamp(db.data_maior_ret.iloc[1]):%d/%m/%Y}"
            + r"). Sortino: fórmula do publicado nas etapas 0 a 3; na 4, desvio abaixo "
            r"de zero sobre todos os dias.")


# ---------------------------------------------------------------------------
# Figuras (sem título gravado: a legenda fica no LaTeX)
# ---------------------------------------------------------------------------
def _acumulado(r):
    v = (1 + r).cumprod()
    return pd.concat([pd.Series([1.0], index=[r.index[0]]), v.set_axis(r.index + pd.Timedelta(days=1))])


def _drawdown(r):
    v = _acumulado(r)
    return v / v.cummax() - 1


def figuras(series, r_bh_hist, datas):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3})
    # Principal: valor acumulado e drawdown (Ridge, q80, sem custos)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for chave, cor, rot in [(("ridge", Q_REF, "comprado"), AZUL, "Comprado/neutro"),
                            (("ridge", Q_REF, "vendido"), LARANJA, "Vendido/neutro"),
                            (("ridge", None, "compra_e_manutencao"), CINZA, "Compra e manutenção")]:
        ax[0].plot(_acumulado(series[chave]), color=cor, lw=2, label=rot)
        ax[1].plot(100 * _drawdown(series[chave]), color=cor, lw=2)
    ax[0].axhline(1, color="black", lw=0.8)
    ax[0].set_ylabel("Valor acumulado (início = 1)")
    ax[1].set_ylabel("Drawdown (%)")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_A1_principal.png"), dpi=150)
    plt.close(fig)

    # Quantis: (a) comprado/neutro, (b) vendido/neutro
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, direcao in enumerate(DIRECOES):
        for q, cor in zip(QUANTIS, [AZUL, LARANJA, VERDE, ROXO]):
            ax[k].plot(_acumulado(series[("ridge", q, direcao)]), color=cor, lw=2, label=f"q{int(q * 100)}")
        ax[k].plot(_acumulado(series[("ridge", None, "compra_e_manutencao")]), color=CINZA, lw=1.5,
                   label="Compra e manutenção")
        ax[k].axhline(1, color="black", lw=0.8)
        ax[k].text(0.02, 0.96, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, va="top")
        ax[k].tick_params(axis="x", rotation=30)
    ax[0].set_ylabel("Valor acumulado (início = 1)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_A1_quantis.png"), dpi=150)
    plt.close(fig)

    # Proxy na amostra completa (robustez), com o início das previsões do T5
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for chave, cor, rot in [(("proxy", Q_REF, "comprado"), AZUL, "Comprado/neutro"),
                            (("proxy", Q_REF, "vendido"), LARANJA, "Vendido/neutro"),
                            (("proxy", None, "compra_e_manutencao"), CINZA, "Compra e manutenção")]:
        ax.plot(_acumulado(series[chave]), color=cor, lw=2, label=rot)
    ax.axvline(datas["ridge"][0], color="black", lw=1, ls="--", label="Início da negociação com o Ridge")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylabel("Valor acumulado (início = 1)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_A1_proxy.png"), dpi=150)
    plt.close(fig)

    # Compra e manutenção no histórico completo (dados pós-T0), escala log
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(_acumulado(r_bh_hist), color=AZUL, lw=2)
    ax.set_yscale("log")
    ax.set_ylabel("Valor acumulado (início = 1, escala log)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_A1_compra_manutencao.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    vt, prev, reg, btc = carregar()
    vr_pre, btc_pre = carregar_pre_t0()
    vr_pos = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_regimes.csv"), parse_dates=["date"])

    pub = ler_publicado()
    rep_pre, rep_pos = replica_cap7(vr_pre, btc_pre), replica_cap7(vr_pos.sort_values("date"), btc)
    rep_simples = replica_cap7(vr_pos.sort_values("date"), btc, simples=True)
    difs = [abs(round(a, 4) - b) for t in ["7.1", "7.2", "7.2b", "7.3", "7.4"]
            for k in pub[t] for a, b in zip(rep_pre[t][k], pub[t][k]) if not (np.isnan(a) and np.isnan(b))]
    print(f"Réplica do publicado (dados pré-T0): {len(difs)} números, diferença máx. {max(difs):.1e}")

    m, series, datas = calcular(vt, prev, reg)
    m.to_csv(os.path.join(OUT, "metricas_A1.csv"), index=False)
    dq = decomposicao_q80(pub, rep_pre, rep_pos, rep_simples, m)
    dq.to_csv(os.path.join(OUT, "decomposicao_q80_A1.csv"), index=False)
    decomposicao_regimes(pub, rep_pre, rep_pos, rep_simples, m).to_csv(os.path.join(OUT, "decomposicao_regimes_A1.csv"),
                                                          index=False)
    db, r_bh_hist = decomposicao_bh(pub, rep_pre, rep_pos, vt, btc)
    db.to_csv(os.path.join(OUT, "decomposicao_compra_manutencao_A1.csv"), index=False)

    ridge, proxy, ret_seg, _ = sinais(vt, prev)
    d = pd.DataFrame({"ret_seguinte": ret_seg, "sinal_ridge": ridge,
                      "limiar_ridge_q80": limiar_expansivo(ridge, Q_REF, BURNIN),
                      "pos_ridge_q80_comprado": posicoes(ridge, Q_REF, BURNIN, 1.0),
                      "sinal_proxy": proxy, "limiar_proxy_q80": limiar_expansivo(proxy, Q_REF, BURNIN),
                      "pos_proxy_q80_comprado": posicoes(proxy, Q_REF, BURNIN, 1.0),
                      "regime_alta_fixo": reg.set_index("date").regime_alta_fixo})
    d.rename_axis("date").to_csv(os.path.join(OUT, "series_diarias_A1.csv"))

    tabelas_tex(m, dq, db, len(datas["ridge"]))
    figuras(series, r_bh_hist, datas)

    for v in ["ridge", "ridge_b126", "proxy", "ponte"]:
        print(f"{ROTULO_VARIANTE[v]}: {datas[v][0].date()} a {datas[v][-1].date()}, {len(datas[v])} dias")
    x = m[(m.regime == "todos") & ((m.quantil == Q_REF) | m.quantil.isna())]
    print(x[["variante", "direcao", "custo_bps", "ret_anual", "sharpe", "sortino", "max_drawdown",
             "giro_anual", "pct_tempo"]].round(4).to_string(index=False))
    print(f"\nSaídas em {OUT}")


if __name__ == "__main__":
    main()
