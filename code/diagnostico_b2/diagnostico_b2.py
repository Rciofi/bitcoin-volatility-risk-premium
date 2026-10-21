"""
diagnostico_b2.py — Rodada 3, B2, passo 1: diagnóstico para D3 e D7

Só lê o que já existe; nada dos Caps. 5–7 é reestimado e nada vai para tables/
ou figs/. Grava CSVs numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>/
(sai com erro se a pasta já existir), além do console.

Entradas: data/ml_dataset_T4.csv (N = 1.776, 17 variáveis, alvo RBVRP),
data/vrp_with_targets.csv (ret_fut_30d), data/btc_prices.csv (log-retorno
diário até t+60), data/previsoes_bvrp_T5.csv e data/previsoes_bvrp_T11.csv
(janela expansiva, 987 datas), outputs/T10/principal_T10.csv,
outputs/T11/arvore_retorno_T11.csv e outputs/T11/previsoes_arvore_retorno_T11.csv.

Sanidade (antes de tudo; para se não bater): com as previsões gravadas,
refaz o β do Ridge em h = 30 (retorno simples, HAC 31) do T10 e o R² fora da
amostra e o Clark–West da leitura B do T11 (h = 30).

Parte 1 — erro de previsão × inovação do retorno (pergunta 3 do Marcelo, D7)
  e_t = RBVRP_t − BVRP previsto_t, por modelo (previsões gravadas).
  u_t = inovação de R_{t+30}: A2 (principal) e A1, A3, A4 (ver inovacoes()).
  corr(e, u) = inclinação de u padronizado sobre e padronizado; EP de
  Newey–West com 31 defasagens (hac_utils.mqo_newey_west(h=30), como no H1).
  Correlograma cruzado diário: corr(e_t, r_{t+j}), j = −29..60, r = log-retorno
  diário. j = 1..30: os mesmos retornos que formam RBVRP_t e R_{t+30} (canal
  mecânico); j <= 0 e j >= 31: controle. Com o log-retorno,
  cov(e, R^log) = Σ_{j=1}^{30} cov(e, r_{t+j}) exatamente.
  Subamostras sem sobreposição: uma data a cada 30, nos 30 deslocamentos.
Parte 2 — retorno de 30 dias direto nos previsores (D3)
  (a) MQO de R_{t+30} nas 16 variáveis lineares do T5 (17 menos vh_30d, pois
      vrp_30d = vh_30d − iv_30d); Wald HAC de todas as inclinações = 0:
      χ²(16) e F = W/16 com F(16, n − 17). N = 1.776 e 987 datas; 31
      defasagens (principal) e 61 (sensibilidade); 30 subamostras sem
      sobreposição com HC0.
  (b) Fora da amostra, nas 33 origens do T5 (embargo s <= t − 30): MQO, Ridge
      e LASSO (α por validação cruzada embargada, grades do T5) contra a média
      histórica e contra zero; R², Clark–West e Diebold–Mariano
      (avaliacao_utils, HAC 31). Retorno em p.p. (× 100), escala do alvo do
      T5, para as grades de α valerem. Sem aleatoriedade (não há semente).
      Ao lado, sem reestimar: floresta direto no retorno (T11, leitura B).
Variável dependente: log-retorno (principal) e retorno simples (ponte com os
Caps. 6 e 7).

Uso:  python code/diagnostico_b2/diagnostico_b2.py
"""
import os
import sys
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import previsao_bvrp_T5 as P  # noqa: E402
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402
from split_utils import gerar_divisoes  # noqa: E402

H = 30
N_OOS = 987
JS = list(range(-29, 61))
TOL = 1e-9
MODELOS_T5 = ["media_historica", "proxy_retrospectiva", "persistencia_viavel",
              "MQO", "Ridge", "LASSO", "floresta_aleatoria", "XGBoost"]
MODELOS_T11 = {"floresta_aleatoria": "floresta_rv_menos_iv", "Ridge": "Ridge_rv_menos_iv"}
YS = {"R_log": "log-retorno acumulado de t+1 a t+30 (principal)",
      "R_simples": "retorno simples de t a t+30 (ponte com os Caps. 6 e 7)"}


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
    assert (px.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), "preços com lacuna"
    assert px.index.max() >= df.date.max() + pd.Timedelta(days=max(JS))
    r = 100 * np.log(px / px.shift(1))                                              # p.p.
    # conferências com o pipeline (só imprimem; não interrompem)
    soma = r.rolling(H).sum().shift(-H).reindex(df.date).to_numpy()
    rv = np.sqrt((r ** 2).rolling(H).mean().shift(-H) * 365).reindex(df.date).to_numpy()
    print(f"Conferência: max|Σr(t+1..t+30) − R_log| = {np.nanmax(np.abs(soma - df.R_log)):.2e}; "
          f"max|RV refeita − (RBVRP + IV)| = {np.nanmax(np.abs(rv - df[P.ALVO] - df.iv_30d)):.2e}")
    return df, feats, r


def previsoes():
    t5 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date"])
    t5 = t5[t5.janela == "expansiva"].pivot(index="date", columns="modelo", values="previsao")[MODELOS_T5]
    t11 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T11.csv"), parse_dates=["date"])
    t11 = t11[(t11.janela == "expansiva") & t11.rotulo.isin(list(MODELOS_T11))]
    t11 = t11.pivot(index="date", columns="rotulo", values="previsao").rename(columns=MODELOS_T11)
    w = t5.join(t11, how="inner").sort_index()
    assert len(t5) == len(w) == N_OOS
    return w


# ---------------------------------------------------------------------------
# Sanidade
# ---------------------------------------------------------------------------
def sanidade(df, w):
    """Refaz, com as previsões gravadas, números publicados; SystemExit se não baterem."""
    linhas = []
    d = df.set_index("date").loc[w.index]
    res = mqo_newey_west(d.ret_fut_30d, w["Ridge"].rename("bvrp_previsto"), h=H)
    t10 = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "principal_T10.csv"))
    t10 = t10[(t10.modelo == "Ridge") & (t10.h == H)].iloc[0]
    linhas += [{"item": "T10 Ridge h=30: beta", "refeito": float(res.params["bvrp_previsto"]), "gravado": t10.beta},
               {"item": "T10 Ridge h=30: p HAC", "refeito": float(res.pvalues["bvrp_previsto"]), "gravado": t10.p_hac}]
    pb = pd.read_csv(os.path.join(ROOT, "outputs", "T11", "previsoes_arvore_retorno_T11.csv"))
    pb = pb[pb.h == H]
    t11 = pd.read_csv(os.path.join(ROOT, "outputs", "T11", "arvore_retorno_T11.csv"))
    t11 = t11[t11.h == H].iloc[0]
    cw = clark_west(pb.realizado, pb.previsao, pb.media_historica, h=H)
    linhas += [{"item": "T11 leitura B h=30: R2 fora da amostra",
                "refeito": r2_fora_da_amostra(pb.realizado, pb.previsao, pb.media_historica),
                "gravado": t11.r2_oos_vs_media_historica},
               {"item": "T11 leitura B h=30: p Clark-West", "refeito": cw["cw_p_unilateral"],
                "gravado": t11.cw_p_unilateral}]
    # a leitura B usa o mesmo retorno: confere o realizado com ret_fut_30d
    real = pd.Series(pb.realizado.to_numpy(), index=pd.to_datetime(pb.date))
    linhas.append({"item": "T11 leitura B: max|realizado − ret_fut_30d|",
                   "refeito": float(np.max(np.abs(real - df.set_index("date").ret_fut_30d.loc[real.index]))),
                   "gravado": 0.0})
    s = pd.DataFrame(linhas).assign(dif=lambda t: (t.refeito - t.gravado).abs())
    s["bate"] = s.dif <= TOL
    print("\n=== Sanidade ===\n" + s.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    if not s.bate.all():
        raise SystemExit("SANIDADE FALHOU: ver tabela acima. Nada mais foi calculado.")
    return s


# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------
def corr_hac(a, b):
    """corr(a, b) com EP de Newey–West (31 defasagens): b padronizado sobre a padronizado."""
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    za = ((a - a.mean()) / a.std(ddof=0)).rename("x")
    zb = (b - b.mean()) / b.std(ddof=0)
    res = mqo_newey_west(zb, za, h=H)
    return {"n": int(res.nobs), "corr": float(res.params["x"]), "ep_hac": float(res.bse["x"]),
            "p_hac": float(res.pvalues["x"])}


def sem_sobreposicao(a, b):
    """corr(a, b) nas 30 subamostras de uma data a cada 30 dias."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.array([np.corrcoef(a[k::H], b[k::H])[0, 1] for k in range(H)])
    return {"corr_sem_sobrep_mediana": float(np.median(c)), "corr_sem_sobrep_min": float(c.min()),
            "corr_sem_sobrep_max": float(c.max()), "n_por_subamostra": len(a[0::H])}


def wald_inclinacoes(res):
    """H0: todas as inclinações = 0, com a covariância do próprio ajuste (HAC ou HC0)."""
    q = len(res.params) - 1
    W = float(np.squeeze(res.wald_test(np.eye(q + 1)[1:], use_f=False, scalar=True).statistic))
    n = int(res.nobs)
    return {"n": n, "q": q, "W": W, "p_chi2": float(stats.chi2.sf(W, q)), "F": W / q,
            "p_F": float(stats.f.sf(W / q, q, n - q - 1)), "r2": res.rsquared, "r2_aj": res.rsquared_adj}


# ---------------------------------------------------------------------------
# Parte 2: retorno direto nos previsores
# ---------------------------------------------------------------------------
def regressao_direta(df, feats, ycol, datas_oos):
    linhas = []
    d_oos = df.set_index("date").loc[datas_oos].reset_index()
    for amostra, d in (("completa", df), ("987 datas", d_oos)):
        y, X = d[ycol], d[feats]
        linhas.append({"y": ycol, "amostra": amostra, "maxlags": H + 1,
                       **wald_inclinacoes(mqo_newey_west(y, X, h=H))})
        L = 2 * H + 1
        linhas.append({"y": ycol, "amostra": amostra, "maxlags": L,
                       **wald_inclinacoes(sensibilidade_defasagens(y, X, lags=[L])[L])})
    ps = []
    for k in range(H):
        d = df.iloc[k::H]
        res = sm.OLS(d[ycol].astype(float), sm.add_constant(d[feats].astype(float))).fit(cov_type="HC0")
        ps.append(wald_inclinacoes(res)["p_F"])
    ps = np.array(ps)
    linhas.append({"y": ycol, "amostra": "completa, sem sobreposição (30 subamostras, HC0)", "maxlags": 0,
                   "n": len(df.iloc[0::H]), "q": len(feats), "p_F": float(np.median(ps)),
                   "p_F_min": float(ps.min()), "p_F_max": float(ps.max()),
                   "frac_p_F_abaixo_5pct": float(np.mean(ps < 0.05))})
    return pd.DataFrame(linhas)


def fora_da_amostra(df, feats, ycol):
    X, y, datas = df[feats].to_numpy(float), df[ycol].to_numpy(float), df.date.to_numpy()
    grades = {"MQO": None, "Ridge": P.GRADE_RIDGE, "LASSO": P.GRADE_LASSO}
    linhas, hiper = [], []
    for o in gerar_divisoes(len(df), h=H, janela="expansiva").origens:
        prevs = {"media_historica": np.full(len(o.teste), y[o.treino].mean()), "zero": np.zeros(len(o.teste))}
        for nome, grade in grades.items():
            p = {} if grade is None else P.escolher(nome, grade, X, y, o.treino)[0]
            f, m = P.ajustar(nome, p, X[o.treino], y[o.treino])
            prevs[nome] = f(X[o.teste])
            hiper.append({"y": ycol, "origem": datas[o.t], "modelo": nome, "alpha": p.get("alpha", np.nan),
                          "n_coef_nao_nulos": int(np.sum(m.coef_ != 0))})
        linhas += [{"date": datas[u], "modelo": nome, "previsao": float(v)}
                   for nome, vs in prevs.items() for u, v in zip(o.teste, vs)]
    w = pd.DataFrame(linhas).pivot(index="date", columns="modelo", values="previsao").sort_index()
    assert len(w) == N_OOS
    yo = df.set_index("date").loc[w.index, ycol].to_numpy()
    met = []
    for nome in ("MQO", "Ridge", "LASSO", "media_historica"):
        linha = {"y": ycol, "modelo": nome, "r2_oos_vs_zero": r2_fora_da_amostra(yo, w[nome], w.zero)}
        if nome != "media_historica":
            linha.update({"r2_oos_vs_media": r2_fora_da_amostra(yo, w[nome], w.media_historica),
                          **clark_west(yo, w[nome], w.media_historica, h=H),
                          **diebold_mariano(yo, w[nome], w.media_historica, h=H)})  # dm_media > 0: a média erra menos
        met.append(linha)
    return w, pd.DataFrame(hiper), pd.DataFrame(met)


def pontas_da_grade(hiper):
    linhas = []
    for (y, m), g in hiper[hiper.modelo != "MQO"].groupby(["y", "modelo"], sort=False):
        grade = [p["alpha"] for p in (P.GRADE_RIDGE if m == "Ridge" else P.GRADE_LASSO)]
        linhas.append({"y": y, "modelo": m, "n_origens": len(g), "alpha_min_grade": grade[0],
                       "n_no_menor": int(np.isclose(g.alpha, grade[0]).sum()),
                       "alpha_max_grade": grade[-1], "n_no_maior": int(np.isclose(g.alpha, grade[-1]).sum()),
                       "coef_nao_nulos_mediana": float(g.n_coef_nao_nulos.median()),
                       "coef_nao_nulos_min": int(g.n_coef_nao_nulos.min()),
                       "coef_nao_nulos_max": int(g.n_coef_nao_nulos.max())})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Parte 1: erro de previsão × inovação
# ---------------------------------------------------------------------------
def inovacoes(d, feats, ycol, w, w_dir):
    """u_t nas 987 datas. A1 depende do modelo; A2–A4 são comuns."""
    y = d[ycol]
    comuns = {"A2_resid_MQO_16_previsores": mqo_newey_west(y, d[feats], h=H).resid.to_numpy(),
              "A3_menos_media_historica_oos": (y - w_dir.media_historica.loc[d.index]).to_numpy(),
              "A4_menos_ridge_direto_oos": (y - w_dir.Ridge.loc[d.index]).to_numpy()}
    a1 = {m: mqo_newey_west(y, w[m].rename("x"), h=H).resid.to_numpy() for m in w.columns}
    return comuns, a1


def parte1(df, feats, r, w, w_dir):
    d = df.set_index("date").loc[w.index]
    rbvrp = d[P.ALVO]
    corr, decomp, series_eu = [], [], []
    for ycol in YS:
        comuns, a1 = inovacoes(d, feats, ycol, w, w_dir[ycol])
        for m in w.columns:
            e = (rbvrp - w[m]).to_numpy()
            for nome_u, u in [("A1_resid_propria_previsao", a1[m]), *comuns.items()]:
                corr.append({"y": ycol, "modelo": m, "inovacao": nome_u, **corr_hac(e, u), **sem_sobreposicao(e, u)})
            series_eu.append(pd.DataFrame({"date": w.index, "y": ycol, "modelo": m, "rbvrp": rbvrp.to_numpy(),
                                           "previsao": w[m].to_numpy(), "e": e, "u_A1": a1[m],
                                           **{f"u_{k[:2]}": v for k, v in comuns.items()}}))
        u = comuns["A2_resid_MQO_16_previsores"]          # e = RBVRP − previsão: de onde vem a correlação
        decomp.append({"y": ycol, "serie": "RBVRP", **corr_hac(rbvrp, u)})
        decomp += [{"y": ycol, "serie": f"previsao_{m}", **corr_hac(w[m], u)} for m in w.columns]

    series = {"RBVRP": rbvrp.to_numpy(), **{f"erro_{m}": (rbvrp - w[m]).to_numpy() for m in w.columns}}
    cc = []
    for nome, s in series.items():
        for j in JS:
            cc.append({"serie": nome, "j": j, **corr_hac(s, r.reindex(w.index + pd.Timedelta(days=j)).to_numpy())})
        cov_soma = sum(np.cov(s, r.reindex(w.index + pd.Timedelta(days=j)))[0, 1] for j in range(1, H + 1))
        assert np.isclose(cov_soma, np.cov(s, d.R_log)[0, 1]), f"{nome}: decomposição não fecha"
    return pd.DataFrame(corr), pd.DataFrame(decomp), pd.DataFrame(cc), pd.concat(series_eu)


# ---------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S"))
    df, feats, r = carregar()
    w = previsoes()
    print(f"N = {len(df)}; {len(feats)} variáveis; {len(w)} datas fora da amostra; modelos: {list(w.columns)}")
    san = sanidade(df, w)
    os.makedirs(out, exist_ok=False)

    w_dir, hiper, met, wald = {}, [], [], []
    for ycol in YS:                                         # parte 2 antes: A3 e A4 usam as previsões diretas
        w_dir[ycol], h_, m_ = fora_da_amostra(df, feats, ycol)
        hiper.append(h_)
        met.append(m_)
        wald.append(regressao_direta(df, feats, ycol, w.index))
    hiper, met, wald = pd.concat(hiper), pd.concat(met), pd.concat(wald)
    pontas = pontas_da_grade(hiper)
    corr, decomp, cc, eu = parte1(df, feats, r, w, w_dir)
    bloco = np.where(cc.j.between(1, H), "j=1..30 (mecanico)", np.where(cc.j <= 0, "j<=0 (controle)", "j>=31 (controle)"))
    cc_bloco = (cc.assign(bloco=bloco).groupby(["serie", "bloco"])["corr"]
                .agg(media="mean", minimo="min", maximo="max", soma="sum").reset_index())

    for nome, t in [("sanidade", san), ("p1_corr_erro_inovacao", corr), ("p1_decomposicao", decomp),
                    ("p1_correlograma_diario", cc), ("p1_correlograma_por_bloco", cc_bloco),
                    ("p1_series_e_u", eu), ("p2a_wald_direto", wald), ("p2b_metricas_oos", met),
                    ("p2b_hiperparametros", hiper), ("p2b_pontas_da_grade", pontas)]:
        t.to_csv(os.path.join(out, f"{nome}.csv"), index=False)
    for ycol, wd in w_dir.items():
        wd.to_csv(os.path.join(out, f"p2b_previsoes_{ycol}.csv"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    f = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== Parte 1: corr(erro, inovação), HAC 31; mediana sem sobreposição ===\n" + corr[
        ["y", "modelo", "inovacao", "corr", "ep_hac", "p_hac", "corr_sem_sobrep_mediana"]].to_string(index=False, float_format=f))
    print("\n=== Parte 1: corr com a inovação A2 — RBVRP × previsão ===\n" + decomp.to_string(index=False, float_format=f))
    print("\n=== Correlograma cruzado diário por bloco ===\n" + cc_bloco.to_string(index=False, float_format=f))
    print("\n=== Parte 2(a): Wald, todas as inclinações = 0 ===\n" + wald.to_string(index=False, float_format=f))
    print("\n=== Parte 2(b): fora da amostra ===\n" + met.to_string(index=False, float_format=f))
    print("\n=== Parte 2(b): α nas pontas da grade ===\n" + pontas.to_string(index=False, float_format=f))
    t11 = pd.read_csv(os.path.join(ROOT, "outputs", "T11", "arvore_retorno_T11.csv"))
    print("\n=== Já existente (T11, leitura B): floresta direto no retorno, h = 30 ===\n"
          + t11[t11.h == H].to_string(index=False, float_format=f))
    print(f"\nSaídas em {out}")


if __name__ == "__main__":
    main()
