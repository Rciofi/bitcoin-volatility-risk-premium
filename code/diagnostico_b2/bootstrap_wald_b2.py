"""
bootstrap_wald_b2.py — Rodada 3, B2, passo 2: Wald por bootstrap em blocos, tamanho por simulação,
intervalo de corr(e, u) e checagem do Ridge com α no topo da grade

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é
alterado. Grava CSVs numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo2/
(criada só depois da conferência do HAC), além do console.

Conferência (antes de tudo; para se não bater): o HAC em numpy deste script
(Bartlett, sem correção de amostra pequena, como hac_utils) contra
hac_utils.mqo_newey_west / sensibilidade_defasagens e contra W = 20,01
(p χ² = 0,220) da amostra completa (R_log, 16 previsores) do passo 1; e o
p HAC de 0,091 do Ridge em h = 30 (T10).

1. Wald por bootstrap em blocos circulares (Politis e Romano, 1992), pares (X, y)
   reamostrados juntos; W* = (β̂* − β̂)' V̂*⁻¹ (β̂* − β̂), V̂* = HAC com L
   defasagens na série reamostrada (a mesma estatística de W); p = (1 + #{W* >= W})
   / (B_válidas + 1). Blocos de 60 (principal), 30 e 90; L = 31 e 61; B = 1.999.
   Casos: q = 16 (R_log) na amostra completa (N = 1.776) e nas 987 datas;
   q = 1 (constante + BVRP previsto pelo Ridge do T5, 987 datas) com R_simples e
   R_log. O caso q = 1 trata a previsão como dada (como o HAC): não corrige o
   regressor gerado.
2. Tamanho por simulação sob H0, X real fixo, y com sobreposição de 30 dias:
   S1 = log-retornos diários reais centrados sorteados iid; S2 = log-retornos
   reais em deslocamento circular de k dias (90 <= k <= M − 90) em relação a X.
   R = 2.000; rejeição do Wald HAC (χ²) a 10%, 5% e 1%, L = 31 e 61.
   Tamanho do próprio bootstrap: S2, R = 300 (500 se o tempo permitir),
   B = 199, bloco de 60, L = 31.
3. IC de 95% de corr(e, u_A2) (R_log; Ridge, floresta e as duas formas
   f(X) − IV), bootstrap em blocos circulares (30, 60, 90), B = 9.999:
   percentis e básico; ao lado, corr ± 1,96 EP HAC do passo 1. Sensibilidade:
   linhas (e, X, y) reamostradas juntas e A2 reestimada em cada réplica.
4. Ridge (α = 1e5) × média histórica nas previsões diretas do passo 1.

Uso:  python code/diagnostico_b2/bootstrap_wald_b2.py
"""
import os
import sys
import time
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed, cpu_count  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import previsao_bvrp_T5 as P  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

H = 30
LS = (31, 61)
BLOCOS = (60, 30, 90)            # 60 = principal
B_WALD = 1999
B_CORR = 9999
R_TAM = 2000
R_TAM_BOOT, R_TAM_BOOT_MAX, B_TAM_BOOT, BLOCO_TAM_BOOT, L_TAM_BOOT = 300, 500, 199, 60, 31
LIMITE_EXTRA_S = 20 * 60         # estende o tamanho do bootstrap a R = 500 se o acréscimo previsto couber nisto
COND_MAX = 1e12                  # acima disto, V̂* conta como mal condicionada
SEMENTE = 20261020
N_PEDACOS = 32                   # nº fixo de pedaços paralelos: resultado não depende do nº de núcleos
ALFAS = (0.10, 0.05, 0.01)
PASSO1 = os.path.join(ROOT, "outputs", "diagnostico_b2", "20261020_200636")
MODELOS_CORR = ["Ridge", "floresta_aleatoria", "Ridge_rv_menos_iv", "floresta_rv_menos_iv"]
NOTA_S = ("S1 e S2 geram y independente de X; portanto não reproduzem a dependência de Stambaugh "
          "(correlação entre a inovação do retorno e as inovações futuras dos previsores)")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("date", P.ALVO, "vh_30d")]          # 16, como no T5
    vt = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    df = df.merge(vt[["date", "ret_fut_30d"]], on="date", how="left")
    assert df.ret_fut_30d.notna().all()
    df["R_simples"] = 100 * df.ret_fut_30d
    df["R_log"] = 100 * np.log1p(df.ret_fut_30d)
    px = pd.read_csv(os.path.join(ROOT, "data", "btc_prices.csv"), parse_dates=["date"]).set_index("date").close
    px = px.sort_index()
    assert (px.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    r = 100 * np.log(px / px.shift(1))
    t5 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date"])
    ridge = t5[(t5.janela == "expansiva") & (t5.modelo == "Ridge")].set_index("date").previsao.sort_index()
    assert len(ridge) == 987
    return df, feats, r, ridge


def retornos_do_caso(r, datas):
    """Log-retornos diários das datas d0 .. d_fim + 30 (comprimento n + 30), para gerar y_t = Σ r(t+1..t+30)."""
    rc = r.loc[datas[0]: datas[-1] + pd.Timedelta(days=H)].to_numpy()
    assert len(rc) == len(datas) + H
    return rc


def somas_30(eps, n):
    c = np.concatenate([[0.0], np.cumsum(eps)])
    t = np.arange(n)
    return c[t + H + 1] - c[t + 1]


# ---------------------------------------------------------------------------
# MQO + HAC (numpy) e bootstrap
# ---------------------------------------------------------------------------
def hac_wald(y, Z, Ls, b0=None):
    """
    W das inclinações (contra b0[1:], ou 0) com HAC Bartlett sem correção para cada L em Ls.
    Retorna (W, cond, b); W = nan se Z'Z ou V for singular.
    """
    k = Z.shape[1]
    try:
        ZZi = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return np.full(len(Ls), np.nan), np.full(len(Ls), np.inf), np.full(k, np.nan)
    b = ZZi @ (Z.T @ y)
    s = Z * (y - Z @ b)[:, None]
    S0 = s.T @ s
    G = [s[l:].T @ s[:-l] for l in range(1, max(Ls) + 1)]
    d = b[1:] - (0.0 if b0 is None else b0[1:])
    W, cond = [], []
    for L in Ls:
        S = S0.copy()
        for l in range(1, L + 1):
            S += (1 - l / (L + 1)) * (G[l - 1] + G[l - 1].T)
        V = (ZZi @ S @ ZZi)[1:, 1:]
        c = float(np.linalg.cond(V))
        try:
            w = float(d @ np.linalg.solve(V, d))
        except np.linalg.LinAlgError:
            w = np.nan
        W.append(w if np.isfinite(w) else np.nan)
        cond.append(c)
    return np.array(W), np.array(cond), b


def idx_circ(n, bloco, rng):
    ini = rng.integers(0, n, size=-(-n // bloco))
    return ((ini[:, None] + np.arange(bloco)).ravel()[:n]) % n


def dividir(total, partes=N_PEDACOS):
    base = [total // partes + (1 if i < total % partes else 0) for i in range(partes)]
    return [x for x in base if x > 0]


def _pedaco_boot(y, Z, Ls, bloco, b_hat, reps, semente):
    rng = np.random.default_rng(semente)
    W, C = [], []
    for _ in range(reps):
        j = idx_circ(len(y), bloco, rng)
        w, c, _ = hac_wald(y[j], Z[j], Ls, b0=b_hat)
        W.append(w)
        C.append(c)
    return np.array(W), np.array(C)


def wald_bootstrap(y, Z, Ls, bloco, B, chave):
    W, _, b_hat = hac_wald(y, Z, Ls)
    res = Parallel(n_jobs=-1)(delayed(_pedaco_boot)(y, Z, Ls, bloco, b_hat, reps, [SEMENTE, *chave, k])
                              for k, reps in enumerate(dividir(B)))
    return W, np.vstack([a for a, _ in res]), np.vstack([c for _, c in res])


# ---------------------------------------------------------------------------
# Tamanho por simulação
# ---------------------------------------------------------------------------
def gerar_y(proc, rc, n, rng):
    if proc == "S1":
        eps = rng.choice(rc - rc.mean(), size=n + H, replace=True)
    else:
        eps = np.roll(rc, int(rng.integers(90, len(rc) - 90)))
    return somas_30(eps, n)


def _pedaco_tamanho(proc, rc, Z, reps, semente):
    rng = np.random.default_rng(semente)
    return np.array([hac_wald(gerar_y(proc, rc, len(Z), rng), Z, LS)[0] for _ in range(reps)])


def _uma_tamanho_boot(rc, Z, semente):
    """Uma réplica S2: W HAC (L = 31) e p do bootstrap (B = 199, bloco 60) na amostra simulada."""
    rng = np.random.default_rng(semente)
    y = gerar_y("S2", rc, len(Z), rng)
    W, _, b_hat = hac_wald(y, Z, [L_TAM_BOOT])
    Wb, _ = _pedaco_boot(y, Z, [L_TAM_BOOT], BLOCO_TAM_BOOT, b_hat, B_TAM_BOOT, [*semente, 1])
    Wb = Wb[:, 0][np.isfinite(Wb[:, 0])]
    return float(W[0]), float((1 + np.sum(Wb >= W[0])) / (len(Wb) + 1)), int(B_TAM_BOOT - len(Wb))


# ---------------------------------------------------------------------------
# IC de corr(e, u)
# ---------------------------------------------------------------------------
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _pedaco_corr(E, u, y, Z, bloco, reps, semente):
    """Réplicas de corr(e*, u*) com u fixo e com A2 reestimada (mesmos índices)."""
    rng = np.random.default_rng(semente)
    fixo, reest = [], []
    for _ in range(reps):
        j = idx_circ(len(u), bloco, rng)
        b = np.linalg.lstsq(Z[j], y[j], rcond=None)[0]
        u_r = y[j] - Z[j] @ b
        fixo.append([corr(E[j, m], u[j]) for m in range(E.shape[1])])
        reest.append([corr(E[j, m], u_r) for m in range(E.shape[1])])
    return np.array(fixo), np.array(reest)


# ---------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    t_ini = time.perf_counter()
    tempos = {}
    nucleos = cpu_count()
    df, feats, r, ridge = carregar()
    d987 = df.set_index("date").loc[ridge.index]
    Zfull = np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)])
    Z16_987 = np.column_stack([np.ones(len(d987)), d987[feats].to_numpy(float)])
    Z1 = np.column_stack([np.ones(len(d987)), ridge.to_numpy(float)])

    # ---------------- Conferência ----------------
    conf = []
    W_np, _, _ = hac_wald(df.R_log.to_numpy(), Zfull, LS)
    res31 = mqo_newey_west(df.R_log, df[feats], h=H)
    res61 = sensibilidade_defasagens(df.R_log, df[feats], lags=[61])[61]
    for L, res, wn in ((31, res31, W_np[0]), (61, res61, W_np[1])):
        ws = float(np.squeeze(res.wald_test(np.eye(17)[1:], use_f=False, scalar=True).statistic))
        conf.append({"item": f"q16 completa, R_log, L = {L}: W numpy × hac_utils", "numpy": wn, "referencia": ws,
                     "bate": abs(wn - ws) <= 1e-8 * max(1, ws)})
    p31 = float(stats.chi2.sf(W_np[0], 16))
    conf.append({"item": "q16 completa, L = 31: W arredondado = 20,01", "numpy": W_np[0], "referencia": 20.01,
                 "bate": round(W_np[0], 2) == 20.01})
    conf.append({"item": "q16 completa, L = 31: p χ² arredondado = 0,220", "numpy": p31, "referencia": 0.220,
                 "bate": round(p31, 3) == 0.220})
    W1, _, _ = hac_wald(d987.R_simples.to_numpy(), Z1, [31])
    p1 = float(stats.chi2.sf(W1[0], 1))
    conf.append({"item": "q1 Ridge, R_simples, L = 31: p = p HAC do T10 (0,0909109)", "numpy": p1,
                 "referencia": 0.0909109, "bate": abs(p1 - 0.0909109) < 1e-6})
    conf = pd.DataFrame(conf)
    print("=== Conferência do HAC ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
    if not conf.bate.all():
        raise SystemExit("CONFERÊNCIA FALHOU: nada mais foi calculado e nenhuma pasta foi criada.")
    out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo2")
    os.makedirs(out, exist_ok=False)
    conf.to_csv(os.path.join(out, "conferencia_hac.csv"), index=False)
    print(f"Núcleos: {nucleos} (joblib n_jobs = -1); pedaços paralelos fixos: {N_PEDACOS}")

    # ---------------- 1. Wald por bootstrap ----------------
    t0 = time.perf_counter()
    casos = {"q16_completa": ("R_log", df.R_log.to_numpy(), Zfull),
             "q16_987": ("R_log", d987.R_log.to_numpy(), Z16_987),
             "q1_ridge_R_simples": ("R_simples", d987.R_simples.to_numpy(), Z1),
             "q1_ridge_R_log": ("R_log", d987.R_log.to_numpy(), Z1)}
    linhas, dist = [], []
    for cid, (caso, (ycol, y, Z)) in enumerate(casos.items()):
        q = Z.shape[1] - 1
        for bloco in BLOCOS:
            W, Wb, Cb = wald_bootstrap(y, Z, LS, bloco, B_WALD, [1, cid, bloco])
            for i, L in enumerate(LS):
                wb = Wb[:, i]
                ok = np.isfinite(wb)
                v = wb[ok]
                linhas.append({"caso": caso, "y": ycol, "n": len(y), "q": q, "bloco": bloco,
                               "principal": bloco == 60, "L": L, "W": W[i], "p_chi2": float(stats.chi2.sf(W[i], q)),
                               "p_boot": float((1 + np.sum(v >= W[i])) / (len(v) + 1)),
                               "crit_boot_90": float(np.quantile(v, 0.90)), "crit_boot_95": float(np.quantile(v, 0.95)),
                               "crit_chi2_90": float(stats.chi2.ppf(0.90, q)), "crit_chi2_95": float(stats.chi2.ppf(0.95, q)),
                               "media_W_boot": float(v.mean()), "B": B_WALD, "n_validas": int(ok.sum()),
                               "n_singular": int((~ok).sum()), "n_mal_condicionada": int(np.sum(Cb[:, i] > COND_MAX))})
                dist.append(pd.DataFrame({"caso": caso, "bloco": bloco, "L": L, "r": np.arange(len(wb)), "W_boot": wb}))
            print(f"  Wald bootstrap {caso}, bloco {bloco}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    wald = pd.DataFrame(linhas)
    wald.to_csv(os.path.join(out, "wald_bootstrap.csv"), index=False)
    pd.concat(dist).to_csv(os.path.join(out, "wald_bootstrap_distribuicoes.csv"), index=False)
    tempos["1_wald_bootstrap"] = time.perf_counter() - t0

    # ---------------- 2. Tamanho por simulação ----------------
    t0 = time.perf_counter()
    sim_casos = {"q16_completa": (df.date, Zfull), "q16_987": (d987.index, Z16_987), "q1_ridge": (d987.index, Z1)}
    rcs = {}
    for caso, (datas, Z) in sim_casos.items():
        rc = retornos_do_caso(r, pd.DatetimeIndex(datas))
        y_real = df.set_index("date").loc[datas, "R_log"].to_numpy()
        assert np.allclose(somas_30(rc, len(Z)), y_real, atol=1e-8), f"{caso}: soma de 30 dias ≠ R_log"
        rcs[caso] = rc
    tam = []
    for cid, (caso, (_, Z)) in enumerate(sim_casos.items()):
        q = Z.shape[1] - 1
        for pid, proc in enumerate(("S1", "S2")):
            W = np.vstack(Parallel(n_jobs=-1)(delayed(_pedaco_tamanho)(proc, rcs[caso], Z, reps, [SEMENTE, 2, cid, pid, k])
                                              for k, reps in enumerate(dividir(R_TAM))))
            for i, L in enumerate(LS):
                w = W[:, i]
                ok = np.isfinite(w)
                tam.append({"teste": "Wald HAC (χ²)", "caso": caso, "n": len(Z), "q": q, "processo": proc, "L": L,
                            "bloco": np.nan, "R": len(w), "n_singular": int((~ok).sum()),
                            **{f"rejeicao_{int(a * 100)}pct": float(np.mean(w[ok] > stats.chi2.ppf(1 - a, q)))
                               for a in ALFAS}, "observacao": NOTA_S})
        print(f"  tamanho (HAC) {caso}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    tempos["2a_tamanho_hac"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tb = {caso: [] for caso in sim_casos}

    def rodar_tb(r_ini, r_fim):
        for cid, (caso, (_, Z)) in enumerate(sim_casos.items()):
            tb[caso] += Parallel(n_jobs=-1)(delayed(_uma_tamanho_boot)(rcs[caso], Z, [SEMENTE, 3, cid, rr])
                                            for rr in range(r_ini, r_fim))
            print(f"  tamanho (bootstrap) {caso}, réplicas {r_ini}–{r_fim - 1}: ok "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    rodar_tb(0, R_TAM_BOOT)
    t300 = time.perf_counter() - t0
    extra_previsto = t300 * (R_TAM_BOOT_MAX - R_TAM_BOOT) / R_TAM_BOOT
    if extra_previsto <= LIMITE_EXTRA_S:
        rodar_tb(R_TAM_BOOT, R_TAM_BOOT_MAX)
    else:
        print(f"  tamanho (bootstrap): mantido R = {R_TAM_BOOT} (acréscimo previsto {extra_previsto:.0f}s)")
    for caso, (_, Z) in sim_casos.items():
        q = Z.shape[1] - 1
        a = np.array(tb[caso])
        Wh, pb, nsing = a[:, 0], a[:, 1], a[:, 2]
        tam.append({"teste": "Wald bootstrap (blocos circulares)", "caso": caso, "n": len(Z), "q": q, "processo": "S2",
                    "L": L_TAM_BOOT, "bloco": BLOCO_TAM_BOOT, "R": len(a), "B": B_TAM_BOOT,
                    "n_singular": int(nsing.sum()),
                    **{f"rejeicao_{int(al * 100)}pct": float(np.mean(pb <= al)) for al in ALFAS}, "observacao": NOTA_S})
        tam.append({"teste": "Wald HAC (χ²), mesmas réplicas", "caso": caso, "n": len(Z), "q": q, "processo": "S2",
                    "L": L_TAM_BOOT, "bloco": np.nan, "R": len(a),
                    "n_singular": int(np.sum(~np.isfinite(Wh))),
                    **{f"rejeicao_{int(al * 100)}pct": float(np.mean(Wh[np.isfinite(Wh)] > stats.chi2.ppf(1 - al, q)))
                       for al in ALFAS}, "observacao": NOTA_S})
    tam = pd.DataFrame(tam)
    tam.to_csv(os.path.join(out, "tamanho_simulacao.csv"), index=False)
    tempos["2b_tamanho_bootstrap"] = time.perf_counter() - t0

    # ---------------- 3. IC de corr(e, u_A2) ----------------
    t0 = time.perf_counter()
    eu = pd.read_csv(os.path.join(PASSO1, "p1_series_e_u.csv"), parse_dates=["date"])
    eu = eu[(eu.y == "R_log") & eu.modelo.isin(MODELOS_CORR)]
    E = eu.pivot(index="date", columns="modelo", values="e")[MODELOS_CORR].sort_index()
    u = eu[eu.modelo == MODELOS_CORR[0]].set_index("date").u_A2.sort_index()
    for m in MODELOS_CORR:
        assert np.allclose(eu[eu.modelo == m].set_index("date").u_A2.sort_index(), u, atol=1e-12)
    dc = df.set_index("date").loc[E.index]
    yc = dc.R_log.to_numpy()
    Zc = np.column_stack([np.ones(len(dc)), dc[feats].to_numpy(float)])
    u_re = yc - Zc @ np.linalg.lstsq(Zc, yc, rcond=None)[0]
    assert np.allclose(u_re, u.to_numpy(), atol=1e-8), "u_A2 do CSV ≠ resíduo reestimado"
    p1c = pd.read_csv(os.path.join(PASSO1, "p1_corr_erro_inovacao.csv"))
    p1c = p1c[(p1c.y == "R_log") & (p1c.inovacao == "A2_resid_MQO_16_previsores")].set_index("modelo")
    En, un = E.to_numpy(), u.to_numpy()
    ic = []
    for bloco in BLOCOS:
        res = Parallel(n_jobs=-1)(delayed(_pedaco_corr)(En, un, yc, Zc, bloco, reps, [SEMENTE, 4, bloco, k])
                                  for k, reps in enumerate(dividir(B_CORR)))
        fixo, reest = np.vstack([a for a, _ in res]), np.vstack([b for _, b in res])
        for versao, M in (("u fixo", fixo), ("A2 reestimada", reest)):
            for i, m in enumerate(MODELOS_CORR):
                rho = corr(En[:, i], un)
                q025, q975 = np.percentile(M[:, i], [2.5, 97.5])
                ic.append({"modelo": m, "versao": versao, "bloco": bloco, "principal": bloco == 60, "B": len(M),
                           "corr": rho, "corr_passo1": float(p1c.loc[m, "corr"]),
                           "media_boot": float(M[:, i].mean()), "dp_boot": float(M[:, i].std(ddof=1)),
                           "ic_pct_inf": q025, "ic_pct_sup": q975,
                           "ic_basico_inf": 2 * rho - q975, "ic_basico_sup": 2 * rho - q025,
                           "ic_hac_inf": float(p1c.loc[m, "corr"] - 1.96 * p1c.loc[m, "ep_hac"]),
                           "ic_hac_sup": float(p1c.loc[m, "corr"] + 1.96 * p1c.loc[m, "ep_hac"])})
        print(f"  IC corr, bloco {bloco}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    ic = pd.DataFrame(ic)
    ic.to_csv(os.path.join(out, "ic_corr_e_u.csv"), index=False)
    tempos["3_ic_corr"] = time.perf_counter() - t0

    # ---------------- 4. Ridge (α = 1e5) × média histórica ----------------
    origens = gerar_divisoes(len(df), h=H, janela="expansiva").origens
    origem_de = pd.Series({df.date[u_]: df.date[o.t] for o in origens for u_ in o.teste})
    rv = []
    for ycol in ("R_log", "R_simples"):
        w = pd.read_csv(os.path.join(PASSO1, f"p2b_previsoes_{ycol}.csv"), index_col=0, parse_dates=True)
        dif = (w.Ridge - w.media_historica).abs()
        dp_y = float(df.set_index("date").loc[w.index, ycol].std(ddof=1))
        por_origem = dif.groupby(origem_de.loc[w.index].to_numpy()).max()
        rv.append({"y": ycol, "n": len(w), "max_abs_dif_pp": float(dif.max()), "media_abs_dif_pp": float(dif.mean()),
                   "dp_y_pp": dp_y, "max_abs_dif_sobre_dp_y": float(dif.max()) / dp_y,
                   "max_por_origem_mediana": float(por_origem.median()), "max_por_origem_max": float(por_origem.max()),
                   "corr_ridge_media": float(np.corrcoef(w.Ridge, w.media_historica)[0, 1])})
    rv = pd.DataFrame(rv)
    rv.to_csv(os.path.join(out, "ridge_vs_media.csv"), index=False)

    tempos["total"] = time.perf_counter() - t_ini
    tp = pd.DataFrame([{"etapa": k, "segundos": v} for k, v in tempos.items()]).assign(nucleos=nucleos)
    tp.to_csv(os.path.join(out, "tempo.csv"), index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 40)
    f = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== 1. Wald por bootstrap ===\n" + wald.to_string(index=False, float_format=f))
    print("\n=== 2. Tamanho por simulação ===\n" + tam.drop(columns="observacao").to_string(index=False, float_format=f))
    print(f"   Observação: {NOTA_S}.")
    print("\n=== 3. IC de corr(e, u_A2), R_log ===\n" + ic.to_string(index=False, float_format=f))
    print("\n=== 4. Ridge (α = 1e5) × média histórica ===\n" + rv.to_string(index=False, float_format=f))
    print("\n=== Tempo ===\n" + tp.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print(f"\nSaídas em {out}")


if __name__ == "__main__":
    main()
