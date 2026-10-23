"""
passo7_regressor_gerado_b2.py — Rodada 3, B2, passo 7 (D7): BVRP previsto como regressor gerado

Bootstrap paramétrico do fator único R_log(t,h) = a + b · BVRP_previsto(t) + e, nas 987 datas fora da
amostra e nos seis horizontes. Nada vai para tables/ ou figs/; grava numa pasta NOVA,
outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo7/, ou retoma uma pasta (--out; pedaços gravados são pulados).

Modelos (hiperparâmetros fixos em cada origem, sem nova validação cruzada):
  ridge:    Ridge direto do T5 (16 variáveis, α de cada origem, alvo RBVRP);
  floresta: floresta f(X) − IV do T11 (17 variáveis, hiperparâmetros de cada origem, alvo RV de t+1 a t+30).
"Verdade" μ: o mesmo modelo ajustado na amostra de modelagem inteira (1.776 datas), com os hiperparâmetros
da última origem; e = RBVRP − μ. AR(p) no erro, p pelo BIC em {29, ..., 60} (amostra comum de 60
defasagens); inovações centradas e reamostradas iid; e* pela recursão do AR com 500 dias de aquecimento;
alvo simulado y* = μ + e* (na floresta, RV* = μ + IV + e*); o modelo é reestimado em cada origem e prevê
as 987 datas, BVRP*.

Variantes (células):
  fixo (só ridge): retorno real; β*_h = MQO de R_log em BVRP*. Três razões: média(β*)/β̂ (β̂ na amostra real),
      média(β*)/β_v (β_v: MQO de R_log em μ nas 987 datas) e média(corr(R_log, BVRP*))/corr(R_log, μ).
  conjunto: AR(q_h) no resíduo u_h = R_log − α_v − β_v·μ; q_h é o menor q em {max(h − 1, 1), ..., h + 59}
      com Ljung–Box > 0,05 em 30 e em 60 defasagens; se nenhum passa, o BIC no mesmo intervalo (registrado).
      A cada reamostragem, as datas sorteadas dão o par (η, ξ_h) junto. R*_h = α_v + β_v·μ + u*_h; sob H0,
      R0*_h = α_v + u*_h (mesmas inovações). EP: dp(β*), robusto (intervalo interquartil / 1,349) e dp do
      coeficiente padronizado β*·dp(BVRP*) (contra β̂·dp(BVRP previsto)); valor-p (i) fração de |β0*| >= |β̂|
      (principal); (ii) percentil, 2·min(P(β* <= 0), P(β* >= 0)), e o intervalo de 2,5% a 97,5%; (iii) como (i),
      com o coeficiente padronizado. Razão dp(BVRP*) / dp(BVRP previsto) por célula.
  Sorteio das datas: só entre as 987 datas fora da amostra (principal) ou na amostra inteira (1776, sensibilidade).
Sensibilidade do fixo: p = 29 e p = 60.

Ponte (ponte_ridge.csv): fator único do Ridge, retorno simples × log, com HAC (h+1), rotação
(g = max(90, 3h)) e bootstrap de pares em blocos circulares (max(30, 2h), B = 1.999), como na Tabela do
fator único do Cap. 6; no log, as chaves do passo 4 ([4, 2, h, bloco]) reproduzem aqueles valores-p.

Sanidade (antes de tudo): com o alvo real, a reestimação reproduz as previsões gravadas
(data/previsoes_bvrp_T5.csv, Ridge; data/previsoes_bvrp_T11.csv, floresta), com tolerância de 1e-9.

--previo: só os diagnósticos antes da rodada completa (previo_*.csv): (a) dp de R* simulado × R real nas
987 datas, 500 sorteios, nas duas formas de sorteio; (b) dp(β*) do bootstrap de pares em blocos (Cap. 6)
ao lado do EP HAC; as células conjuntas rodam com o --B dado.

Sementes: default_rng([20261020, 8, célula, r]); prévia (a) [20261020, 8, 90, sorteio, r]; bootstrap de
blocos do retorno simples [20261020, 8, 10, h, bloco]. Checkpoint: pedaços de 100 em
<pasta>/celulas/<célula>/pedaco_<k>.csv; previsões simuladas das 20 primeiras reamostragens em
<pasta>/celulas/<célula>/previsoes_r00_r19.csv.

Uso:  python code/diagnostico_b2/passo7_regressor_gerado_b2.py [--B 1999] [--B-floresta 499] [--out PASTA]
      [--celulas c1,c2] [--previo]
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed, cpu_count  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.ar_model import AutoReg  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
import previsao_bvrp_T5 as P  # noqa: E402
import retorno_bvrp_T10 as T10  # noqa: E402
import retorno_bvrp_T11 as T11  # noqa: E402
import wald_utils as W  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

HS = [1, 5, 10, 20, 30, 60]
SEMENTE, ETAPA = 20261020, 8
P_MIN, P_MAX, AQUEC, PEDACO, N_PREV, B_BLOCOS = 29, 60, 500, 100, 20, 1999
TOL = 1e-9
CELULAS = {  # nome: (id, modelo, variante, p fixo ou None = BIC, sorteio das datas)
    "ridge_fixo_bic": (1, "ridge", "fixo", None, "1776"),
    "ridge_fixo_p29": (2, "ridge", "fixo", 29, "1776"),
    "ridge_fixo_p60": (3, "ridge", "fixo", 60, "1776"),
    "ridge_conjunto_987": (6, "ridge", "conjunto", None, "987"),           # principal
    "ridge_conjunto_1776": (4, "ridge", "conjunto", None, "1776"),         # sensibilidade
    "floresta_conjunto_987": (7, "floresta", "conjunto", None, "987"),
}
AR_ERRO = {"ridge": "ridge_conjunto_987", "floresta": "floresta_conjunto_987"}   # AR do erro usado no par (η, ξ)


# ---------------------------------------------------------------------------
# Dados e modelos
# ---------------------------------------------------------------------------
def carregar():
    df, _, prev_t5, hp_t5 = T10.carregar()
    df[T11.ALVO_RV] = df[P.ALVO] + df["iv_30d"]
    cols_t4 = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), nrows=1).columns
    feats_arv = [c for c in cols_t4 if c not in ("date", P.ALVO)]
    feats_lin = [c for c in feats_arv if c != "vh_30d"]
    origens = gerar_divisoes(len(df), h=P.H, janela="expansiva").origens
    oos = np.concatenate([o.teste for o in origens])
    hip_t11 = pd.read_csv(os.path.join(ROOT, "outputs", "T11", "hiperparametros_T11.csv"), parse_dates=["origem"])
    prev_t11 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T11.csv"), parse_dates=["date", "origem"])
    iv = df["iv_30d"].to_numpy(float)
    modelos = {
        "ridge": dict(nome="Ridge", X=df[feats_lin].to_numpy(float), y=df[P.ALVO].to_numpy(float), soma_iv=None,
                      params=T11.params_fixos(None, None, hp_t5, "Ridge", "direta"),
                      prev=prev_t5[prev_t5.modelo == "Ridge"]),
        "floresta": dict(nome="floresta_aleatoria", X=df[feats_arv].to_numpy(float),
                         y=df[T11.ALVO_RV].to_numpy(float), soma_iv=iv,
                         params=T11.params_fixos(hip_t11, "floresta_aleatoria", hp_t5, "floresta_aleatoria", "rv_menos_iv"),
                         prev=prev_t11[(prev_t11.janela == "expansiva") & (prev_t11.rotulo == "floresta_aleatoria")]),
    }
    for m in modelos.values():
        assert len(m["params"]) == len(origens) == 33
        m["prev"] = m["prev"].set_index("date").loc[df.date.iloc[oos], "previsao"].to_numpy(float)
    R = 100 * np.log1p(df[[f"ret_fut_{h}d" for h in HS]].to_numpy(float))          # R_log em p.p., 1.776 × 6
    Rs = 100 * df[[f"ret_fut_{h}d" for h in HS]].to_numpy(float)                   # retorno simples (ponte)
    return df, origens, oos, iv, modelos, R, Rs


def prever_oos(m, y, origens, iv_oos):
    """Reestima o modelo em cada origem (hiperparâmetros fixos) com o alvo y e prevê as datas de teste."""
    prev = []
    for k, o in enumerate(origens):
        f, _ = P.ajustar(m["nome"], m["params"][k], m["X"][o.treino], y[o.treino], n_jobs=1)
        prev.append(f(m["X"][o.teste]))
    prev = np.concatenate(prev)
    return prev - iv_oos if m["soma_iv"] is not None else prev


def betas(x, Y):
    Z = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(Z, Y, rcond=None)[0]                                    # 2 × k: [α; β]


def corrs(x, Y):
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    return (xc @ Yc) / np.sqrt((xc @ xc) * (Yc ** 2).sum(axis=0))


# ---------------------------------------------------------------------------
# AR
# ---------------------------------------------------------------------------
def _lb(inov, p):
    lb = acorr_ljungbox(inov[p:], lags=[10, 30, 60])
    return [float(v) for v in lb.lb_pvalue]


def _ajuste(x, p):
    fit = AutoReg(x, lags=p, trend="c").fit()
    inov = np.full(len(x), np.nan)
    inov[p:] = fit.resid - fit.resid.mean()
    return fit, inov


def escolher_ar(x, p_min, p_max, p_fixo=None, regra="bic"):
    """
    regra = "bic": p pelo BIC na amostra comum (hold_back = p_max); "lb": menor p com Ljung–Box > 0,05 em 30
    e em 60 defasagens (se nenhum passa, o BIC). Ajuste final com hold_back = p; inovações alinhadas às datas.
    """
    criterio = "fixo"
    if p_fixo is None:
        p = None
        if regra == "lb":
            for q in range(p_min, p_max + 1):
                _, inov = _ajuste(x, q)
                lb = _lb(inov, q)
                if lb[1] > 0.05 and lb[2] > 0.05:
                    p, criterio = q, "Ljung-Box"
                    break
        if p is None:
            bics = {q: AutoReg(x, lags=q, trend="c", hold_back=p_max).fit().bic for q in range(p_min, p_max + 1)}
            p = min(bics, key=bics.get)
            criterio = "BIC" if regra == "bic" else "BIC (nenhum passa no Ljung-Box)"
    else:
        p = p_fixo
    fit, inov = _ajuste(x, p)
    const, phi = float(fit.params[0]), np.asarray(fit.params[1:], float)
    lb = _lb(inov, p)
    return dict(p=p, criterio=criterio, const=const, phi=phi, inov=inov, bic=fit.bic,
                modulo_max=float(np.abs(np.roots(np.r_[1.0, -phi])).max()),
                lb_p10=lb[0], lb_p30=lb[1], lb_p60=lb[2], dp_inov=float(np.nanstd(inov[p:], ddof=1)))


def simular(ar, choques, n):
    """Recursão do AR com aquecimento: os primeiros len(choques) − n valores são descartados."""
    return lfilter([1.0], np.r_[1.0, -ar["phi"]], ar["const"] + choques)[-n:]


# ---------------------------------------------------------------------------
# Reamostragem
# ---------------------------------------------------------------------------
def uma(r, cel, ctx):
    cid, modelo, variante, _, sorteio = CELULAS[cel]
    m, ar_e, n, oos = ctx["modelos"][modelo], ctx["ar_e"][cel], ctx["n"], ctx["oos"]
    rng = np.random.default_rng([SEMENTE, ETAPA, cid, r])
    if variante == "fixo":
        idx = rng.choice(np.flatnonzero(~np.isnan(ar_e["inov"])), size=n + AQUEC, replace=True)
    else:
        idx = rng.choice(ctx["pool"][modelo][sorteio], size=n + AQUEC, replace=True)
    e_s = simular(ar_e, ar_e["inov"][idx], n)
    y_s = ctx["mu"][modelo] + e_s + (m["soma_iv"] if m["soma_iv"] is not None else 0.0)
    prev = prever_oos(m, y_s, ctx["origens"], ctx["iv_oos"])
    linha = {"r": r, "dp_f": float(prev.std(ddof=1))}
    if variante == "fixo":
        Rr = ctx["R"][oos]
        b, c = betas(prev, Rr)[1], corrs(prev, Rr)
        linha.update({f"b_{h}": b[i] for i, h in enumerate(HS)})
        linha.update({f"c_{h}": c[i] for i, h in enumerate(HS)})
    else:
        Rs, R0 = [], []
        av, bv, mu_oos = ctx["alfa_v"][modelo], ctx["beta_v"][modelo], ctx["mu"][modelo][oos]
        for i, h in enumerate(HS):
            ar_u = ctx["ar_u"][modelo][h]
            u_s = simular(ar_u, ar_u["inov"][idx], n)[oos]
            Rs.append(av[i] + bv[i] * mu_oos + u_s)
            R0.append(av[i] + u_s)
        b, b0 = betas(prev, np.column_stack(Rs))[1], betas(prev, np.column_stack(R0))[1]
        linha.update({f"b_{h}": b[i] for i, h in enumerate(HS)})
        linha.update({f"b0_{h}": b0[i] for i, h in enumerate(HS)})
    return linha, (prev if r < N_PREV else None)


def rodar_celula(cel, B, ctx, out):
    d = os.path.join(out, "celulas", cel)
    os.makedirs(d, exist_ok=True)
    for k in range(int(np.ceil(B / PEDACO))):
        arq = os.path.join(d, f"pedaco_{k:03d}.csv")
        if os.path.exists(arq):
            continue
        rs = range(k * PEDACO, min(B, (k + 1) * PEDACO))
        t0 = time.perf_counter()
        res = Parallel(n_jobs=-1)(delayed(uma)(r, cel, ctx) for r in rs)
        prevs = {f"r{lin['r']:02d}": p for lin, p in res if p is not None}
        if prevs:
            arq_p = os.path.join(d, f"previsoes_r00_r{N_PREV - 1:02d}.csv")
            ant = pd.read_csv(arq_p) if os.path.exists(arq_p) else pd.DataFrame({"date": ctx["datas_oos"]})
            pd.concat([ant, pd.DataFrame(prevs)], axis=1).to_csv(arq_p, index=False, lineterminator="\n")
        pd.DataFrame([lin for lin, _ in res]).to_csv(arq, index=False, lineterminator="\n")   # por último
        seg = time.perf_counter() - t0
        arq_t = os.path.join(out, "tempo.csv")
        pd.DataFrame([{"celula": cel, "pedaco": k, "B_pedaco": len(rs), "segundos": seg, "nucleos": cpu_count()}]).to_csv(
            arq_t, mode="a", index=False, lineterminator="\n", header=not os.path.exists(arq_t))
        print(f"  {cel}: pedaço {k} ({len(rs)} reamostragens) em {seg:.1f}s", flush=True)
    return pd.concat([pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("pedaco_")])


def resumir(cel, bs, ctx):
    _, modelo, variante, _, sorteio = CELULAS[cel]
    linhas = []
    s = bs["dp_f"].to_numpy()
    s0 = ctx["dp_f"][modelo]
    for i, h in enumerate(HS):
        b = bs[f"b_{h}"].to_numpy()
        bh = ctx["beta_hat"][modelo][i]
        bv = ctx["beta_v"][modelo][i]
        q25, q75 = np.percentile(b, [25, 75])
        linha = {"celula": cel, "modelo": modelo, "variante": variante, "sorteio": sorteio, "h": h, "B": len(b),
                 "beta_hat": bh, "beta_v": bv, "media_bstar": b.mean(), "dp_bstar": b.std(ddof=1),
                 "ep_robusto": (q75 - q25) / 1.349, "q025": np.percentile(b, 2.5), "q975": np.percentile(b, 97.5),
                 "razao_dp_f": s.mean() / s0, "dp_f_min_sobre_dp_f": s.min() / s0,
                 "beta_hat_padronizado": bh * s0, "media_bstar_padronizado": (b * s).mean(),
                 "dp_bstar_padronizado": (b * s).std(ddof=1)}
        if variante == "fixo":
            c_rmu = ctx["corr_r_mu"][modelo][i]
            lam = b.mean() / bh
            linha.update({"razao_beta_hat": lam, "razao_beta_v": b.mean() / bv,
                          "razao_corr": bs[f"c_{h}"].mean() / c_rmu, "corr_r_mu": c_rmu,
                          "media_corr_bstar": bs[f"c_{h}"].mean(),
                          "beta_corrigido": bh / lam, "dp_bstar_sobre_razao": b.std(ddof=1) / abs(lam)})
        else:
            b0 = bs[f"b0_{h}"].to_numpy()
            linha.update({"p_h0": float(np.mean(np.abs(b0) >= abs(bh))),
                          "p_percentil": float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0)))),
                          "p_h0_padronizado": float(np.mean(np.abs(b0 * s) >= abs(bh * s0))),
                          "media_b0": b0.mean(), "dp_b0": b0.std(ddof=1), "ep_hac": ctx["ep_hac"][modelo][i]})
        linhas.append(linha)
    return linhas


# ---------------------------------------------------------------------------
# Ponte e bootstrap de blocos do Cap. 6
# ---------------------------------------------------------------------------
def fator_unico(x, Y, chave_boot):
    """Por h: β, EP HAC, p HAC, p rotação, p bootstrap (Cap. 6) e dp(β*) do bootstrap de pares em blocos."""
    from scipy import stats
    n, Z = len(x), np.column_stack([np.ones(len(x)), x])
    linhas = []
    for i, h in enumerate(HS):
        y = Y[:, i]
        Wh, b = W.hac_wald(y, Z, (h + 1,))
        g, bl = max(90, 3 * h), max(30, 2 * h)
        p_rot, _ = W.p_mc(Wh[0], W.rotacoes(y, Z, (h + 1,), np.arange(g, n - g + 1))[:, 0])
        Wb, Wstar = W.bootstrap_pares(y, Z, (h + 1,), bl, B_BLOCOS, [*chave_boot, h, bl])
        p_boot, _ = W.p_mc(Wb[0], Wstar[:, 0])
        rng = np.random.default_rng([SEMENTE, ETAPA, 91, h, bl])
        bst = np.array([betas(x[j], y[j])[1] for j in (W.idx_circ(n, bl, rng) for _ in range(B_BLOCOS))])
        e = y - Z @ b
        linhas.append({"h": h, "beta": b[1], "ep_hac": abs(b[1]) / np.sqrt(Wh[0]),
                       "p_hac": float(stats.chi2.sf(Wh[0], 1)), "p_rotacao": p_rot, "p_bootstrap": p_boot,
                       "dp_bstar_blocos": float(bst.std(ddof=1)),
                       "r2": 1 - (e @ e) / np.sum((y - y.mean()) ** 2)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1999)
    ap.add_argument("--B-floresta", type=int, default=499)
    ap.add_argument("--out")
    ap.add_argument("--celulas", help="lista separada por vírgulas (padrão: todas)")
    ap.add_argument("--previo", action="store_true", help="diagnósticos (a)-(b) e só as células conjuntas")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = args.out or os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo7")
    os.makedirs(out, exist_ok=bool(args.out))
    print(f"Saídas em {out}; núcleos: {cpu_count()}", flush=True)
    t_ini = time.perf_counter()
    df, origens, oos, iv, modelos, R, Rsimp = carregar()
    n, iv_oos = len(df), iv[oos]

    # 1. Sanidade
    san = []
    for nome, m in modelos.items():
        t0 = time.perf_counter()
        p = prever_oos(m, m["y"], origens, iv_oos)
        san.append({"item": f"{nome}: reestimação com o alvo real = previsões gravadas (máx. |dif|)",
                    "dif": float(np.max(np.abs(p - m["prev"]))), "tol": TOL, "segundos": time.perf_counter() - t0})

    # 2. Verdade μ, erro e, β̂, β_v, variâncias (c), AR
    bvrp = df[P.ALVO].to_numpy(float)
    mu, e, beta_hat, alfa_v, beta_v, corr_r_mu, ep_hac, ar_u, pool, var = {}, {}, {}, {}, {}, {}, {}, {}, {}, []
    eh_oos = np.zeros(n, bool)
    eh_oos[oos] = True
    for nome, m in modelos.items():
        f, _ = P.ajustar(m["nome"], m["params"][-1], m["X"], m["y"], n_jobs=1)   # n_jobs=1: soma das árvores em ordem fixa
        mu[nome] = f(m["X"]) - (iv if m["soma_iv"] is not None else 0.0)
        e[nome] = bvrp - mu[nome]
        e_fora = bvrp[oos] - m["prev"]
        var.append({"modelo": nome, "var_e_dentro_1776": float(np.var(e[nome], ddof=1)),
                    "var_e_dentro_987": float(np.var(e[nome][oos], ddof=1)), "var_erro_fora_987": float(np.var(e_fora, ddof=1)),
                    "razao_fora_dentro_987": float(np.var(e_fora, ddof=1) / np.var(e[nome][oos], ddof=1)),
                    "corr_mu_prev_987": float(np.corrcoef(mu[nome][oos], m["prev"])[0, 1])})
        beta_hat[nome] = betas(m["prev"], R[oos])[1]
        ab = betas(mu[nome][oos], R[oos])
        alfa_v[nome], beta_v[nome] = ab[0], ab[1]
        corr_r_mu[nome] = corrs(mu[nome][oos], R[oos])
        Z = np.column_stack([np.ones(len(oos)), m["prev"]])
        ep_hac[nome] = [abs(beta_hat[nome][i]) / np.sqrt(W.hac_wald(R[oos, i], Z, (h + 1,))[0][0]) for i, h in enumerate(HS)]
        ar_u[nome] = {}
        for i, h in enumerate(HS):
            u = R[:, i] - alfa_v[nome][i] - beta_v[nome][i] * mu[nome]
            q0 = max(h - 1, 1)
            ar_u[nome][h] = escolher_ar(u, q0, h + 59, regra="lb")
    ar_e = {cel: escolher_ar(e[mod], P_MIN, P_MAX, p_fixo) for cel, (_, mod, _, p_fixo, _) in CELULAS.items()}
    for nome in modelos:
        cel_c = AR_ERRO[nome]
        ok = ~np.isnan(ar_e[cel_c]["inov"]) & np.all([~np.isnan(ar_u[nome][h]["inov"]) for h in HS], axis=0)
        pool[nome] = {"1776": np.flatnonzero(ok), "987": np.flatnonzero(ok & eh_oos)}
    linhas_ar = []
    for cel, a in ar_e.items():
        linhas_ar.append({"modelo": CELULAS[cel][1], "serie": f"erro ({cel})", "ordem": a["p"], "criterio": a["criterio"],
                          "modulo_raiz_max": a["modulo_max"], "lb_p10": a["lb_p10"], "lb_p30": a["lb_p30"],
                          "lb_p60": a["lb_p60"], "dp_inov": a["dp_inov"], "corr_inov_erro": np.nan})
    for nome in modelos:
        eta = ar_e[AR_ERRO[nome]]["inov"]
        for h in HS:
            a = ar_u[nome][h]
            ok = ~np.isnan(eta) & ~np.isnan(a["inov"])
            linhas_ar.append({"modelo": nome, "serie": f"retorno h = {h}", "ordem": a["p"], "criterio": a["criterio"],
                              "modulo_raiz_max": a["modulo_max"], "lb_p10": a["lb_p10"], "lb_p30": a["lb_p30"],
                              "lb_p60": a["lb_p60"], "dp_inov": a["dp_inov"],
                              "corr_inov_erro": float(np.corrcoef(eta[ok], a["inov"][ok])[0, 1])})
    ar = pd.DataFrame(linhas_ar)
    san = pd.DataFrame(san).assign(bate=lambda x: x.dif <= x.tol)
    for nome_arq, d in (("sanidade", san), ("variancias", pd.DataFrame(var)), ("ar", ar)):
        d.to_csv(os.path.join(out, f"{nome_arq}.csv"), index=False, lineterminator="\n")
    pd.DataFrame([{"modelo": nome, "h": h, "beta_hat": beta_hat[nome][i], "alfa_v": alfa_v[nome][i], "beta_v": beta_v[nome][i],
                   "corr_r_mu": corr_r_mu[nome][i], "ep_hac": ep_hac[nome][i]}
                  for nome in modelos for i, h in enumerate(HS)]).to_csv(os.path.join(out, "verdade.csv"), index=False,
                                                                        lineterminator="\n")
    pd.set_option("display.width", 250)
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    print("=== Sanidade ===\n" + san.to_string(index=False, float_format=lambda v: f"{v:.3g}"), flush=True)
    if not san.bate.all():
        sys.exit(2)
    print("\n=== AR ===\n" + ar.to_string(index=False, float_format=f4), flush=True)

    ctx = dict(modelos=modelos, ar_e=ar_e, ar_u=ar_u, pool=pool, mu=mu, origens=origens, oos=oos, iv_oos=iv_oos, R=R,
               n=n, alfa_v=alfa_v, beta_v=beta_v, beta_hat=beta_hat, dp_f={k: float(v["prev"].std(ddof=1)) for k, v in modelos.items()}, corr_r_mu=corr_r_mu, ep_hac=ep_hac,
               datas_oos=df.date.iloc[oos].dt.strftime("%Y-%m-%d").to_numpy())

    # 3. Ponte (Ridge, simples × log) e bootstrap de blocos do Cap. 6
    ponte = pd.concat([fator_unico(modelos["ridge"]["prev"], M[oos], chave).assign(retorno=ret)
                       for ret, M, chave in (("simples", Rsimp, [ETAPA, 10]), ("log", R, [4, 2]))])
    ponte.to_csv(os.path.join(out, "ponte_ridge.csv"), index=False, lineterminator="\n")
    print("\n=== Ponte (Ridge) ===\n" + ponte.to_string(index=False, float_format=f4), flush=True)

    # 4. Prévia (a): dp de R* × R real nas 987 datas
    celulas = list(CELULAS) if not args.celulas else args.celulas.split(",")
    if args.previo:
        linhas = []
        for nome in modelos:
            ae = ar_e[AR_ERRO[nome]]
            for sorteio, cod in (("1776", 1), ("987", 2)):
                sims = []
                for r in range(500):
                    rng = np.random.default_rng([SEMENTE, ETAPA, 90, cod, r])
                    idx = rng.choice(pool[nome][sorteio], size=n + AQUEC, replace=True)
                    sims.append([np.std(alfa_v[nome][i] + beta_v[nome][i] * mu[nome][oos]
                                        + simular(ar_u[nome][h], ar_u[nome][h]["inov"][idx], n)[oos], ddof=1)
                                 for i, h in enumerate(HS)]
                                + [np.std(simular(ae, ae["inov"][idx], n)[oos], ddof=1)])
                sims = np.array(sims)
                for i, h in enumerate(HS):
                    linhas.append({"modelo": nome, "sorteio": sorteio, "h": h, "dp_R_real_987": float(np.std(R[oos, i], ddof=1)),
                                   "dp_Rstar_media": sims[:, i].mean(), "dp_Rstar_q05": np.percentile(sims[:, i], 5),
                                   "dp_Rstar_q95": np.percentile(sims[:, i], 95),
                                   "razao": sims[:, i].mean() / np.std(R[oos, i], ddof=1)})
                linhas.append({"modelo": nome, "sorteio": sorteio, "h": "erro e", "dp_R_real_987": float(np.std(e[nome][oos], ddof=1)),
                               "dp_Rstar_media": sims[:, -1].mean(), "dp_Rstar_q05": np.percentile(sims[:, -1], 5),
                               "dp_Rstar_q95": np.percentile(sims[:, -1], 95),
                               "razao": sims[:, -1].mean() / np.std(e[nome][oos], ddof=1)})
        prev_a = pd.DataFrame(linhas)
        prev_a.to_csv(os.path.join(out, "previo_a_dp.csv"), index=False, lineterminator="\n")
        print("\n=== Prévia (a): dp de R* simulado × R real, 987 datas ===\n" + prev_a.to_string(index=False, float_format=f4))
        celulas = [c for c in celulas if CELULAS[c][2] == "conjunto"] if not args.celulas else celulas

    # 5. Células
    resumo = []
    for cel in celulas:
        B = args.B_floresta if CELULAS[cel][1] == "floresta" else args.B
        resumo += resumir(cel, rodar_celula(cel, B, ctx, out), ctx)
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(out, "atenuacao.csv"), index=False, lineterminator="\n")
    print("\n=== Variâncias (c) ===\n" + pd.DataFrame(var).to_string(index=False, float_format=f4))
    print("\n=== Atenuação ===\n" + resumo.to_string(index=False, float_format=f4))
    print(f"\nTempo total: {(time.perf_counter() - t_ini) / 60:.1f} min")


if __name__ == "__main__":
    main()
