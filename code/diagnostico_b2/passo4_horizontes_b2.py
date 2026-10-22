"""
passo4_horizontes_b2.py — Rodada 3, B2, passo 4, Etapa A: os seis horizontes com R_log

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é
alterado. Grava CSVs numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo4/
(criada só depois da conferência), além do console.

y = R_log(t, h) = 100·log1p(ret_fut_hd) = soma dos log-retornos diários de t+1 a t+h
(p.p.), h ∈ {1, 5, 10, 20, 30, 60}. R_simples = 100·ret_fut_hd só nas conferências.

Conferência (antes de tudo; para se não bater, sem criar a pasta):
  (i)   R_log(t, h) = soma dos log-retornos diários (btc_prices), 1.776 datas, 6 h;
  (ii)  W numpy (Bartlett, sem correção) × hac_utils, L = h+1 e 2h+1, q16, 2 amostras, 6 h;
  (iii) h = 30, passos 1–3: W = 20,008 / 33,386 (L = 31); p da rotação (g = 90) 0,4218 /
        0,1953 / 0,2361 (um fator Ridge, R_log); p do bootstrap 0,5895 (bloco 60, chave do passo 2);
  (iv)  um fator com R_simples, 6 h: p HAC do Ridge = principal_T10.csv e da floresta
        f(X) − IV = principal_T11.csv;
  (v)   regime com R_simples, corte fixo, h = 30 e 60: p da interação e do conjunto = regime_T10.csv;
        Wald de todas as inclinações por wald_sub = wald do passo 3;
  (vi)  fora da amostra em h = 30 (R_log): previsões = p2b_previsoes_R_log.csv e métricas =
        p2b_metricas_oos.csv (passo 1);
  (vii) sem sobreposição em h = 30 (R_log, completa): mediana do p F = p2a_wald_direto.csv.

Desenhos (987 = datas da janela expansiva do T5; 1.776 = amostra de modelagem):
  q16_completa, q16_987   R_log nos 16 previsores (17 menos vh_30d, como no T5)
  q1_ridge, q1_floresta   R_log no BVRP previsto pelo Ridge (T5) e pela floresta f(X) − IV (T11)
  regime (h = 30 e 60)    R_log = a + β·B̂ + δ·D + γ·D·B̂ (B̂ do Ridge; D = regime_alta_fixo,
                          principal, e _expansivo); H0: γ = 0; δ = γ = 0; β = δ = γ = 0
Testes, por h (L = h+1 principal; 2h+1 sensibilidade):
  Wald HAC com χ² (e F = W/q com F(q, n − q − 1));
  rotação: p = (1 + #{Wₖ >= W}) / (K + 1), k = g .. n − g, g_h = max(90, 3h)
    (sensibilidade max(60, 2h) e max(180, 6h));
  bootstrap em blocos circulares (pares (X, y)), centrado em β̂, B = 1.999,
    bloco ℓ_h = max(30, 2h) (sensibilidade max(15, h) e max(45, 3h)).
  Holm sobre os 6 p de cada (desenho, L, teste), na variante principal.
  Entre horizontes, pela rotação, com o MESMO k em todos os h, k = g* .. n − g*,
    g* = 180 = max_h g_h (sensibilidade 120 e 360): principal T = min_h p_h, em que p_h de
    cada elemento (observado ou deslocamento k) é o posto no conjunto {W_h} ∪ {W_h(k)}
    do próprio horizonte, #{s >= x} / (K + 1); p = (1 + #{T(k) <= T}) / (K + 1).
    Sensibilidade: T = max_h W_h. Registra o h do mínimo (e do máximo).
  No regime, rotação de Freedman–Lane: ajusta o modelo restrito (sem as colunas da
  hipótese), rotaciona circularmente o resíduo dele, y* = ajustado + resíduo rotacionado,
  e calcula o Wald da hipótese na regressão completa; em β = δ = γ = 0 o modelo restrito
  é só a constante e equivale à rotação de y. O bootstrap de pares, centrado em β̂, vale
  nas três hipóteses.
Fora da amostra (33 origens, embargo s <= t − h nas origens e nas dobras da validação
  cruzada): MQO, Ridge e LASSO (grades do T5) contra a média histórica e contra zero;
  R², Clark–West e Diebold–Mariano (HAC h+1); pontas da grade. Sem aleatoriedade.
Subamostras sem sobreposição (q16_completa, h <= 30, HC0): só no CSV.
Sementes do bootstrap: default_rng([20261020, 4 (q16/q1) ou 5 (regime), desenho, h, bloco,
pedaço]); 32 pedaços fixos: o resultado não depende do número de núcleos.

Uso:  python code/diagnostico_b2/passo4_horizontes_b2.py
"""
import os
import sys
import time
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from joblib import Parallel, delayed, cpu_count  # noqa: E402
from scipy import stats  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, AQUI)
import previsao_bvrp_T5 as P  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from bootstrap_wald_b2 import N_PEDACOS, SEMENTE, dividir, idx_circ, wald_bootstrap  # noqa: E402
from diagnostico_b2 import pontas_da_grade, wald_inclinacoes  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402
from mc_rotacao_poder_b2 import inv_ou_none, p_mc, wald  # noqa: E402
from split_utils import divisoes_validacao_cruzada, gerar_divisoes  # noqa: E402

HS = [1, 5, 10, 20, 30, 60]
N_OOS = 987
B = 1999
K_MIN = 60                       # menor g usado (sensibilidade max(60, 2h)); as rotações vão de K_MIN a n − K_MIN
G_MAX = {"principal": 180, "menor": 120, "maior": 360}
SUBS = {"interacao": [3], "regime_e_interacao": [2, 3], "todas": [1, 2, 3]}
DESENHOS = {"q16_completa": 0, "q16_987": 1, "q1_ridge": 2, "q1_floresta": 3}
PASSO1 = os.path.join(ROOT, "outputs", "diagnostico_b2", "20261020_200636")
NOTA = ("rotação: H0 = y independente dos regressores; não é exata com mudança de regime de volatilidade. "
        "Bootstrap de pares centrado em β̂. Um fator: a previsão é tratada como dada")
NOTA_REGIME = ("rotação de Freedman–Lane: resíduo do modelo restrito (sem as colunas da hipótese) rotacionado "
               "circularmente, y* = ajustado restrito + resíduo rotacionado, Wald da hipótese na regressão "
               "completa; em 'todas' o restrito é só a constante (= rotação de y). Não é exata com mudança de "
               "regime de volatilidade. Bootstrap de pares centrado em β̂ nas três hipóteses. B̂ tratado como dado")
NOTA_MAX = ("principal: T = min_h p_h, p_h pelo posto no conjunto {W_h} ∪ {W_h(k)} do próprio h, mesmo k em "
            "todos os h; sensibilidade: T = max_h W_h")


def ls(h):
    return (h + 1, 2 * h + 1)


def gs(h):
    return {"principal": max(90, 3 * h), "menor": max(60, 2 * h), "maior": max(180, 6 * h)}


def blocos(h):
    return {"principal": max(30, 2 * h), "menor": max(15, h), "maior": max(45, 3 * h)}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar():
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("date", P.ALVO, "vh_30d")]          # 16, como no T5
    vt = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    reg = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])
    df = (df.merge(vt[["date"] + [f"ret_fut_{h}d" for h in HS]], on="date", how="left")
            .merge(reg[["date", "regime_alta_fixo", "regime_alta_expansivo"]], on="date", how="left"))
    assert df[[f"ret_fut_{h}d" for h in HS] + ["regime_alta_fixo", "regime_alta_expansivo"]].notna().all().all()
    for h in HS:
        df[f"R_log_{h}"] = 100 * np.log1p(df[f"ret_fut_{h}d"])
        df[f"R_simples_{h}"] = 100 * df[f"ret_fut_{h}d"]
    px = pd.read_csv(os.path.join(ROOT, "data", "btc_prices.csv"), parse_dates=["date"]).set_index("date").close
    px = px.sort_index()
    assert (px.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    r = 100 * np.log(px / px.shift(1))
    t5 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date"])
    ridge = t5[(t5.janela == "expansiva") & (t5.modelo == "Ridge")].set_index("date").previsao.sort_index()
    t11 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T11.csv"), parse_dates=["date"])
    flor = (t11[(t11.janela == "expansiva") & (t11.rotulo == "floresta_aleatoria")]
            .set_index("date").previsao.sort_index())                                 # f(X) − IV
    assert len(ridge) == N_OOS and len(flor) == N_OOS and (ridge.index == flor.index).all()
    return df, feats, r, ridge, flor


# ---------------------------------------------------------------------------
# Wald de subconjuntos (regime), rotação e bootstrap
# ---------------------------------------------------------------------------
def wald_sub(y, Z, Ls, subs, ZZi=None, b0=None):
    """W de H0: b[idx] = b0[idx] (ou 0) para cada idx de subs e cada L (Bartlett, sem correção)."""
    if ZZi is None:
        ZZi = inv_ou_none(Z)
        if ZZi is None:
            return np.full((len(subs), len(Ls)), np.nan), None
    b = ZZi @ (Z.T @ y)
    s = Z * (y - Z @ b)[:, None]
    S0 = s.T @ s
    G = [s[l:].T @ s[:-l] for l in range(1, max(Ls) + 1)]
    d = b - (0.0 if b0 is None else b0)
    W = np.full((len(subs), len(Ls)), np.nan)
    for j, L in enumerate(Ls):
        S = S0.copy()
        for l in range(1, L + 1):
            S += (1 - l / (L + 1)) * (G[l - 1] + G[l - 1].T)
        V = ZZi @ S @ ZZi
        for i, idx in enumerate(subs.values()):
            try:
                w = float(d[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], d[idx]))
            except np.linalg.LinAlgError:
                w = np.nan
            W[i, j] = w if np.isfinite(w) else np.nan
    return W, b


def pedacos(ks):
    return [c for c in np.array_split(np.asarray(ks), N_PEDACOS) if len(c)]


def _rot(y, Z, ZZi, Ls, ks):
    return np.array([wald(np.roll(y, k), Z, Ls, ZZi)[0] for k in ks])


def _rot_sub(y, Z, ZZi, Ls, ks):
    return np.array([wald_sub(np.roll(y, k), Z, Ls, SUBS, ZZi)[0] for k in ks])


def _rot_fl(ajustado, resid, Z, ZZi, Ls, idx, ks):
    """Freedman–Lane: W da hipótese idx com y* = ajustado restrito + roll(resíduo restrito, k)."""
    return np.array([wald_sub(ajustado + np.roll(resid, k), Z, Ls, {"h": idx}, ZZi)[0][0] for k in ks])


def rotacoes_fl(y, Z, ZZi, Ls, idx, ks):
    """K × L. Modelo restrito: Z sem as colunas idx (com idx = todas as inclinações, só a constante)."""
    Zr = np.delete(Z, idx, axis=1)
    ajustado = Zr @ np.linalg.lstsq(Zr, y, rcond=None)[0]
    resid = y - ajustado
    return np.concatenate(Parallel(n_jobs=-1)(delayed(_rot_fl)(ajustado, resid, Z, ZZi, Ls, idx, p)
                                              for p in pedacos(ks)))


def postos(Wobs, Wk):
    """p do observado e de cada deslocamento no conjunto {W_obs} ∪ {W_k}: #{s >= x} / (K + 1)."""
    S = np.concatenate([[Wobs], Wk])
    p = (len(S) - np.searchsorted(np.sort(S), S, side="left")) / len(S)
    return p[0], p[1:]


def rotacoes(func, y, Z, ZZi, Ls, ks):
    return np.concatenate(Parallel(n_jobs=-1)(delayed(func)(y, Z, ZZi, Ls, p) for p in pedacos(ks)))


def _boot_sub(y, Z, Ls, bloco, b_hat, reps, semente):
    rng = np.random.default_rng(semente)
    out = []
    for _ in range(reps):
        j = idx_circ(len(y), bloco, rng)
        out.append(wald_sub(y[j], Z[j], Ls, SUBS, b0=b_hat)[0])
    return np.array(out)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[o]), 1.0)
    out = np.empty(m)
    out[o] = adj
    return out


# ---------------------------------------------------------------------------
# Testes por desenho e horizonte
# ---------------------------------------------------------------------------
def testes(nome, y, Z, h):
    """χ², F, rotação (3 g) e bootstrap (3 blocos) para L = h+1 e 2h+1. Retorna linhas, ks, Wk, W, b."""
    n, q, Ls = len(y), Z.shape[1] - 1, ls(h)
    ZZi = np.linalg.inv(Z.T @ Z)
    W, b = wald(y, Z, Ls, ZZi)
    ks = np.arange(K_MIN, n - K_MIN + 1)
    Wk = rotacoes(_rot, y, Z, ZZi, Ls, ks)
    boots = {v: wald_bootstrap(y, Z, Ls, bl, B, [4, DESENHOS[nome], h, bl])[1] for v, bl in blocos(h).items()}
    linhas = []
    for i, (regra, L) in enumerate(zip(("h+1", "2h+1"), Ls)):
        base = {"desenho": nome, "n": n, "q": q, "h": h, "L_regra": regra, "L": L, "W": W[i]}
        linhas.append({**base, "teste": "chi2", "variante": "principal", "p": float(stats.chi2.sf(W[i], q))})
        linhas.append({**base, "teste": "F", "variante": "principal",
                       "p": float(stats.f.sf(W[i] / q, q, n - q - 1))})
        for v, g in gs(h).items():
            sel = (ks >= g) & (ks <= n - g)
            p, ns = p_mc(W[i], Wk[sel, i])
            linhas.append({**base, "teste": "rotacao", "variante": v, "parametro": g, "reps": int(sel.sum()),
                           "n_invalidas": ns, "p": p})
        for v, bl in blocos(h).items():
            p, ns = p_mc(W[i], boots[v][:, i])
            linhas.append({**base, "teste": "bootstrap", "variante": v, "parametro": bl, "reps": B,
                           "n_blocos": round(n / bl, 1), "n_invalidas": ns, "p": p})
    return linhas, ks, Wk, W, b


def um_fator(nome, y, Z, h, W, b):
    """β, EP HAC (= |β|/√W), t e R² do um fator, para L = h+1 e 2h+1."""
    e = y - Z @ b
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return [{"desenho": nome, "h": h, "L_regra": regra, "L": L, "beta": b[1], "ep_hac": abs(b[1]) / np.sqrt(W[i]),
             "t_hac": np.sign(b[1]) * np.sqrt(W[i]), "r2": r2} for i, (regra, L) in enumerate(zip(("h+1", "2h+1"), ls(h)))]


def entre_horizontes(nome, n, res):
    """
    Rotação com o mesmo k em todos os h; res[h] = (ks, Wk, W). Principal: T = min_h p_h (postos no
    conjunto do próprio h); sensibilidade: T = max_h W_h. Só entram os k com W finito em todos os h.
    """
    linhas = []
    ks = res[HS[0]][0]
    for i, regra in enumerate(("h+1", "2h+1")):
        Wobs = np.array([res[h][2][i] for h in HS])
        Wk = np.column_stack([res[h][1][:, i] for h in HS])                             # K × 6
        for v, g in G_MAX.items():
            sel = (ks >= g) & (ks <= n - g)
            ok = sel & np.isfinite(Wk).all(axis=1)
            pp = [postos(Wobs[j], Wk[ok, j]) for j in range(len(HS))]
            pobs = np.array([a for a, _ in pp])
            pk = np.column_stack([b for _, b in pp])
            T, Tk = pobs.min(), pk.min(axis=1)
            K = int(ok.sum())
            Tw, Twk = Wobs.max(), Wk[ok].max(axis=1)
            linhas.append({"desenho": nome, "L_regra": regra, "variante": v, "g": g, "K": K,
                           "n_invalidas": int(sel.sum() - K),
                           **{f"p_h{h}": pobs[j] for j, h in enumerate(HS)},
                           "T_min_p": T, "h_do_min_p": ",".join(str(HS[j]) for j in np.flatnonzero(pobs == T)),
                           "p_min_p": float((1 + np.sum(Tk <= T)) / (K + 1)),
                           "T_max_W": Tw, "h_do_max_W": HS[int(np.argmax(Wobs))],
                           "p_max_W": float((1 + np.sum(Twk >= Tw)) / (K + 1)), "observacao": NOTA_MAX})
    return linhas


def regime(d987, b_hat, ycol, h, corte):
    """Regime do Cap. 6 com B̂ do Ridge: W de SUBS para L = h+1 e 2h+1, HAC/rotação/bootstrap."""
    D = d987[f"regime_alta_{corte}"].to_numpy(float)
    Z = np.column_stack([np.ones(len(D)), b_hat, D, D * b_hat])
    y = d987[ycol].to_numpy(float)
    n, Ls = len(y), ls(h)
    ZZi = np.linalg.inv(Z.T @ Z)
    W, b = wald_sub(y, Z, Ls, SUBS, ZZi)
    return Z, y, n, Ls, ZZi, W, b


def regime_testes(d987, b_hat, h, corte, cid):
    Z, y, n, Ls, ZZi, W, b = regime(d987, b_hat, f"R_log_{h}", h, corte)
    ks = np.arange(K_MIN, n - K_MIN + 1)
    Wk = np.stack([rotacoes_fl(y, Z, ZZi, Ls, idx, ks) for idx in SUBS.values()], axis=1)   # K × subs × L
    boots = {}
    for v, bl in blocos(h).items():
        res = Parallel(n_jobs=-1)(delayed(_boot_sub)(y, Z, Ls, bl, b, reps, [SEMENTE, 5, cid, h, bl, k])
                                  for k, reps in enumerate(dividir(B)))
        boots[v] = np.concatenate(res)
    linhas = []
    for s, (hip, idx) in enumerate(SUBS.items()):
        for i, (regra, L) in enumerate(zip(("h+1", "2h+1"), Ls)):
            base = {"corte": corte, "h": h, "n": n, "pct_alta": 100 * float(Z[:, 2].mean()),
                    "b_bvrp_previsto": b[1], "b_regime_alta": b[2], "b_interacao": b[3],
                    "hipotese": hip, "gl": len(idx), "L_regra": regra, "L": L, "W": W[s, i]}
            linhas.append({**base, "teste": "chi2", "variante": "principal",
                           "p": float(stats.chi2.sf(W[s, i], len(idx)))})
            for v, g in gs(h).items():
                sel = (ks >= g) & (ks <= n - g)
                p, ns = p_mc(W[s, i], Wk[sel, s, i])
                linhas.append({**base, "teste": "rotacao_freedman_lane", "variante": v, "parametro": g,
                               "reps": int(sel.sum()), "n_invalidas": ns, "p": p})
            for v, bl in blocos(h).items():
                p, ns = p_mc(W[s, i], boots[v][:, s, i])
                linhas.append({**base, "teste": "bootstrap", "variante": v, "parametro": bl, "reps": B,
                               "n_invalidas": ns, "p": p})
    return linhas


# ---------------------------------------------------------------------------
# Fora da amostra e sem sobreposição
# ---------------------------------------------------------------------------
def escolher_lin_h(nome, grade, X, y, idx_treino, h):
    """P.escolher com embargo h nas dobras (P.escolher fixa h = 30); só modelos lineares."""
    dobras = divisoes_validacao_cruzada(idx_treino, h=h, n_dobras=P.N_DOBRAS)
    erros = [P._mse_dobra(nome, p, X, y, tr, va, -1) for p in grade for tr, va in dobras]
    mses = np.asarray(erros).reshape(len(grade), len(dobras)).mean(axis=1)
    i = int(np.argmin(mses))
    return grade[i], float(mses[i])


def fora_da_amostra(df, feats, ycol, h, datas_oos):
    X, y, datas = df[feats].to_numpy(float), df[ycol].to_numpy(float), df.date.to_numpy()
    grades = {"MQO": None, "Ridge": P.GRADE_RIDGE, "LASSO": P.GRADE_LASSO}
    linhas, hiper = [], []
    for o in gerar_divisoes(len(df), h=h, janela="expansiva").origens:
        prevs = {"media_historica": np.full(len(o.teste), y[o.treino].mean()), "zero": np.zeros(len(o.teste))}
        for nome, grade in grades.items():
            p = {} if grade is None else escolher_lin_h(nome, grade, X, y, o.treino, h)[0]
            f, m = P.ajustar(nome, p, X[o.treino], y[o.treino])
            prevs[nome] = f(X[o.teste])
            hiper.append({"y": ycol, "h": h, "origem": datas[o.t], "modelo": nome, "alpha": p.get("alpha", np.nan),
                          "n_coef_nao_nulos": int(np.sum(m.coef_ != 0))})
        linhas += [{"date": datas[u], "modelo": nome, "previsao": float(v)}
                   for nome, vs in prevs.items() for u, v in zip(o.teste, vs)]
    w = pd.DataFrame(linhas).pivot(index="date", columns="modelo", values="previsao").sort_index()
    assert len(w) == N_OOS and (w.index == datas_oos).all()
    yo = df.set_index("date").loc[w.index, ycol].to_numpy()
    met = []
    for nome in ("MQO", "Ridge", "LASSO", "media_historica"):
        linha = {"y": ycol, "h": h, "modelo": nome, "r2_oos_vs_zero": r2_fora_da_amostra(yo, w[nome], w.zero)}
        if nome != "media_historica":
            linha.update({"r2_oos_vs_media": r2_fora_da_amostra(yo, w[nome], w.media_historica),
                          **clark_west(yo, w[nome], w.media_historica, h=h),
                          **diebold_mariano(yo, w[nome], w.media_historica, h=h)})  # dm_media > 0: a média erra menos
        met.append(linha)
    return w, pd.DataFrame(hiper), pd.DataFrame(met)


def sem_sobreposicao(df, feats, ycol, h):
    ps = []
    for k in range(h):
        d = df.iloc[k::h]
        res = sm.OLS(d[ycol].astype(float), sm.add_constant(d[feats].astype(float))).fit(cov_type="HC0")
        ps.append(wald_inclinacoes(res)["p_F"])
    ps = np.array(ps)
    return {"y": ycol, "h": h, "n_subamostras": h, "n_por_subamostra": len(df.iloc[0::h]), "q": len(feats),
            "p_F_mediana": float(np.median(ps)), "p_F_min": float(ps.min()), "p_F_max": float(ps.max()),
            "frac_p_F_abaixo_5pct": float(np.mean(ps < 0.05))}


# ---------------------------------------------------------------------------
# Conferência
# ---------------------------------------------------------------------------
def item(nome, valor, ref, tol, relativa=False):
    dif = abs(valor - ref)
    return {"item": nome, "valor": valor, "referencia": ref, "dif": dif,
            "bate": bool(dif <= tol * (max(1.0, abs(ref)) if relativa else 1.0))}


def conferencia(df, feats, r, ridge, flor, d987, Zs):
    conf = []
    c = r.fillna(0).cumsum()
    pos = c.index.get_indexer(df.date)
    cv = c.to_numpy()
    for h in HS:
        assert (pos >= 0).all() and pos.max() + h < len(cv)
        dif = float(np.max(np.abs(cv[pos + h] - cv[pos] - df[f"R_log_{h}"].to_numpy())))
        conf.append(item(f"(i) h = {h}: R_log = soma dos log-retornos diários (máx. |dif|)", dif, 0.0, 1e-8))
    for h in HS:
        for amostra, d in (("completa", df), ("987", d987)):
            Z = Zs["q16_completa" if amostra == "completa" else "q16_987"]
            W, _ = wald(d[f"R_log_{h}"].to_numpy(float), Z, ls(h))
            res = {h + 1: mqo_newey_west(d[f"R_log_{h}"], d[feats], h=h),
                   2 * h + 1: sensibilidade_defasagens(d[f"R_log_{h}"], d[feats], lags=[2 * h + 1])[2 * h + 1]}
            for i, L in enumerate(ls(h)):
                ws = float(np.squeeze(res[L].wald_test(np.eye(17)[1:], use_f=False, scalar=True).statistic))
                conf.append(item(f"(ii) q16 {amostra}, h = {h}, L = {L}: W numpy × hac_utils", W[i], ws, 1e-8, True))
    y30, y30_987 = df.R_log_30.to_numpy(float), d987.R_log_30.to_numpy(float)
    W, _ = wald(y30, Zs["q16_completa"], (31,))
    conf.append(item("(iii) q16 completa, h = 30, L = 31: W do passo 1", W[0], 20.008077684062627, 1e-8, True))
    W, _ = wald(y30_987, Zs["q16_987"], (31,))
    conf.append(item("(iii) q16 987, h = 30, L = 31: W do passo 1", W[0], 33.385841144236366, 1e-8, True))
    for nome, y, Z, ref in (("q16 completa", y30, Zs["q16_completa"], 0.42177722152690866),
                            ("q16 987", y30_987, Zs["q16_987"], 0.19530284301606923),
                            ("q1 Ridge R_log", y30_987, Zs["q1_ridge"], 0.2360939431396786)):
        ZZi = np.linalg.inv(Z.T @ Z)
        ks = np.arange(90, len(y) - 90 + 1)
        W, _ = wald(y, Z, (31,), ZZi)
        p, _ = p_mc(W[0], rotacoes(_rot, y, Z, ZZi, (31,), ks)[:, 0])
        conf.append(item(f"(iii) {nome}, h = 30, g = 90: p da rotação do passo 3", p, ref, 1e-12))
    W, Wb, _ = wald_bootstrap(y30, Zs["q16_completa"], (31, 61), 60, B, [1, 0, 60])
    conf.append(item("(iii) q16 completa, h = 30, bloco 60: p do bootstrap do passo 2",
                     p_mc(W[0], Wb[:, 0])[0], 0.5895, 1e-12))
    t10 = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "principal_T10.csv")).query("modelo == 'Ridge'").set_index("h")
    t11 = pd.read_csv(os.path.join(ROOT, "outputs", "T11", "principal_T11.csv")).query(
        "regressor == 'floresta_rv_menos_iv'").set_index("h")
    for h in HS:
        for nome, ref in (("q1_ridge", t10.p_hac[h]), ("q1_floresta", t11.p_hac[h])):
            W, _ = wald(d987[f"R_simples_{h}"].to_numpy(float), Zs[nome], (h + 1,))
            conf.append(item(f"(iv) {nome}, R_simples, h = {h}: p HAC = T10/T11", float(stats.chi2.sf(W[0], 1)),
                             float(ref), 1e-8))
    rg = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "regime_T10.csv")).query("corte == 'fixo'").set_index("h")
    for h in (30, 60):
        Z, _, _, _, ZZi, W, _ = regime(d987, ridge.to_numpy(float), f"R_simples_{h}", h, "fixo")
        conf.append(item(f"(v) regime fixo, R_simples, h = {h}: p da interação = T10",
                         float(stats.chi2.sf(W[0, 0], 1)), float(rg.p_interacao[h]), 1e-8))
        conf.append(item(f"(v) regime fixo, R_simples, h = {h}: p do conjunto = T10",
                         float(stats.chi2.sf(W[1, 0], 2)), float(rg.p_conjunto[h]), 1e-8))
        Wall, _ = wald(d987[f"R_simples_{h}"].to_numpy(float), Z, ls(h), ZZi)
        conf.append(item(f"(v) regime fixo, h = {h}, L = {h + 1}: wald_sub(todas) × wald", W[2, 0], Wall[0], 1e-10, True))
    Z, y, n, Ls, ZZi, W, _ = regime(d987, ridge.to_numpy(float), "R_log_30", 30, "fixo")
    ks = np.arange(90, n - 90 + 1)
    p_fl, _ = p_mc(W[2, 0], rotacoes_fl(y, Z, ZZi, Ls, SUBS["todas"], ks)[:, 0])
    p_y, _ = p_mc(W[2, 0], rotacoes(_rot_sub, y, Z, ZZi, Ls, ks)[:, 2, 0])
    conf.append(item("(v) regime fixo, h = 30, g = 90: Freedman–Lane em 'todas' = rotação de y", p_fl, p_y, 1e-12))
    w30, h30, m30 = fora_da_amostra(df, feats, "R_log_30", 30, ridge.index)
    ref = pd.read_csv(os.path.join(PASSO1, "p2b_previsoes_R_log.csv"), parse_dates=["date"]).set_index("date")
    cols = ["MQO", "Ridge", "LASSO", "media_historica", "zero"]
    conf.append(item("(vi) h = 30, R_log: previsões fora da amostra = passo 1 (máx. |dif|)",
                     float(np.max(np.abs(w30[cols].to_numpy() - ref.loc[w30.index, cols].to_numpy()))), 0.0, 1e-9))
    mref = pd.read_csv(os.path.join(PASSO1, "p2b_metricas_oos.csv")).query("y == 'R_log'").set_index("modelo")
    for nome in ("MQO", "Ridge"):
        linha = m30.set_index("modelo").loc[nome]
        for col in ("r2_oos_vs_media", "cw_p_unilateral", "dm_p_bilateral"):
            conf.append(item(f"(vi) h = 30, {nome}: {col} = passo 1", float(linha[col]), float(mref.loc[nome, col]), 1e-9))
    ss = sem_sobreposicao(df, feats, "R_log_30", 30)
    conf.append(item("(vii) h = 30, completa: mediana do p F sem sobreposição = passo 1", ss["p_F_mediana"],
                     0.013899966008457762, 1e-9))
    return pd.DataFrame(conf), (w30, h30, m30)


# ---------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    t_ini = time.perf_counter()
    tempos = {}
    df, feats, r, ridge, flor = carregar()
    d987 = df.set_index("date").loc[ridge.index]
    um = np.ones(N_OOS)
    Zs = {"q16_completa": np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)]),
          "q16_987": np.column_stack([um, d987[feats].to_numpy(float)]),
          "q1_ridge": np.column_stack([um, ridge.to_numpy(float)]),
          "q1_floresta": np.column_stack([um, flor.to_numpy(float)])}
    ys = {"q16_completa": df, "q16_987": d987, "q1_ridge": d987, "q1_floresta": d987}

    t0 = time.perf_counter()
    conf, oos30 = conferencia(df, feats, r, ridge, flor, d987, Zs)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 90)
    print("=== Conferência ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
    if not conf.bate.all():
        raise SystemExit("CONFERÊNCIA FALHOU: nada mais foi calculado e nenhuma pasta foi criada.")
    tempos["0_conferencia"] = time.perf_counter() - t0
    out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo4")
    os.makedirs(out, exist_ok=False)
    conf.to_csv(os.path.join(out, "conferencia.csv"), index=False)
    print(f"Núcleos: {cpu_count()}; pedaços paralelos fixos: {N_PEDACOS}", flush=True)

    # ---------------- Wald direto, um fator, Holm e máximo ----------------
    t0 = time.perf_counter()
    linhas, uf, mx = [], [], []
    for nome, Z in Zs.items():
        res = {}
        for h in HS:
            y = ys[nome][f"R_log_{h}"].to_numpy(float)
            lin, ks, Wk, W, b = testes(nome, y, Z, h)
            linhas += lin
            res[h] = (ks, Wk, W)
            if nome in ("q1_ridge", "q1_floresta"):
                uf += um_fator(nome, y, Z, h, W, b)
            print(f"  {nome}, h = {h}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
        mx += entre_horizontes(nome, Z.shape[0], res)
    tst = pd.DataFrame(linhas).assign(observacao=NOTA)
    prin = tst[tst.variante == "principal"].copy()
    prin["p_holm"] = prin.groupby(["desenho", "L_regra", "teste"]).p.transform(lambda s: holm(s.to_numpy()))
    tst.to_csv(os.path.join(out, "testes_por_horizonte.csv"), index=False)
    prin.drop(columns="observacao").to_csv(os.path.join(out, "holm.csv"), index=False)
    uf = pd.DataFrame(uf)
    assert set(uf.desenho) == {"q1_ridge", "q1_floresta"}
    assert (uf.groupby(["h", "L_regra"]).desenho.apply(lambda s: tuple(sorted(s))) == ("q1_floresta", "q1_ridge")).all()
    assert len(uf) == 2 * len(HS) * 2
    uf.to_csv(os.path.join(out, "um_fator.csv"), index=False)
    mx = pd.DataFrame(mx)
    sens = (mx.pivot_table(index=["desenho", "L_regra"], columns="g", values="p_min_p")
              .rename(columns=lambda g: f"p_min_p_g{g}").reset_index()
              .assign(variante="sensibilidade_g*", observacao="p do min-p com g* = 120, 180 (principal) e 360"))
    mx = pd.concat([mx, sens], ignore_index=True)
    mx.to_csv(os.path.join(out, "entre_horizontes.csv"), index=False)
    tempos["1_wald_direto"] = time.perf_counter() - t0

    # ---------------- Regime (h = 30 e 60) ----------------
    t0 = time.perf_counter()
    rg = []
    for cid, corte in enumerate(("fixo", "expansivo")):
        for h in (30, 60):
            rg += regime_testes(d987, ridge.to_numpy(float), h, corte, cid)
            print(f"  regime {corte}, h = {h}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    rg = pd.DataFrame(rg).assign(observacao=NOTA_REGIME)
    rg.to_csv(os.path.join(out, "regime.csv"), index=False)
    tempos["2_regime"] = time.perf_counter() - t0

    # ---------------- Fora da amostra ----------------
    t0 = time.perf_counter()
    ws, hs_, ms = [], [], []
    for h in HS:
        w, hp, m = oos30 if h == 30 else fora_da_amostra(df, feats, f"R_log_{h}", h, ridge.index)
        ws.append(w.assign(h=h))
        hs_.append(hp)
        ms.append(m)
        print(f"  fora da amostra, h = {h}: ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    met, hiper = pd.concat(ms), pd.concat(hs_)
    met.to_csv(os.path.join(out, "oos_metricas.csv"), index=False)
    pd.concat(ws).to_csv(os.path.join(out, "oos_previsoes.csv"))
    hiper.to_csv(os.path.join(out, "oos_hiperparametros.csv"), index=False)
    pontas_da_grade(hiper).to_csv(os.path.join(out, "oos_pontas_grade.csv"), index=False)
    tempos["3_fora_da_amostra"] = time.perf_counter() - t0

    # ---------------- Sem sobreposição (só CSV) ----------------
    pd.DataFrame([sem_sobreposicao(df, feats, f"R_log_{h}", h) for h in HS if h <= 30]).to_csv(
        os.path.join(out, "sem_sobreposicao.csv"), index=False)

    tempos["total"] = time.perf_counter() - t_ini
    pd.DataFrame([{"etapa": k, "segundos": v, "nucleos": cpu_count()} for k, v in tempos.items()]).to_csv(
        os.path.join(out, "tempo.csv"), index=False)

    # ---------------- Console ----------------
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    tab = (prin[prin.teste != "F"].pivot_table(index=["desenho", "L_regra", "h"], columns="teste",
                                               values=["p", "p_holm"]).round(4))
    print("\n=== Wald direto e um fator: p (variante principal) e Holm ===\n" + tab.to_string())
    print("\n=== Entre horizontes (rotação: min p principal, max W sensibilidade) ===\n"
          + mx.drop(columns="observacao").to_string(index=False, float_format=fmt))
    print("\n=== Um fator ===\n" + uf.to_string(index=False, float_format=fmt))
    print("\n=== Regime (principal) ===\n" + rg[rg.variante == "principal"].drop(columns="observacao")
          .to_string(index=False, float_format=fmt))
    print("\n=== Fora da amostra ===\n" + met.to_string(index=False, float_format=fmt))
    print("\n=== Pontas da grade ===\n" + pontas_da_grade(hiper).to_string(index=False))
    print(f"\nTempo total: {tempos['total'] / 60:.1f} min; saídas em {out}")


if __name__ == "__main__":
    main()
