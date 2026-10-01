"""
previsao_bvrp_T5.py — previsão do BVRP prospectivo (T5) e ajuste das árvores (T7)

Alvo: alvo_bvrp_30d_fut = RV(t+1 a t+30) - IV_t, conhecido só em t+30.
Dados: data/ml_dataset_T4.csv (N = 1.776, 17 variáveis explicativas).
Divisão: split_utils.gerar_divisoes(h = 30): origens a cada 30 dias a partir de
21/06/2023, treino com s <= t - 30, 987 datas fora da amostra. Janela
expansiva (principal) e móvel de 730 (robustez).

Modelos
  Referências (sem ajuste):
    media_historica      média do alvo no treino da origem (s <= t - 30)
    proxy_retrospectiva  vrp_30d em t (passeio aleatório)
    persistencia_viavel  bvrp_realizado_defasado em t (= alvo de t - 30)
  Lineares (16 variáveis: todas menos vh_30d, pela colinearidade exata
  vrp_30d = vh_30d - iv_30d): MQO, Ridge, LASSO. Padronização no treino.
  Árvores (17 variáveis): floresta aleatória e XGBoost.
  α de Ridge/LASSO e hiperparâmetros das árvores escolhidos em CADA
  reestimação por validação cruzada embargada (split_utils.
  divisoes_validacao_cruzada: 5 dobras expansivas, treino da dobra com
  s <= v - 30), pelo menor MSE médio entre as dobras.

Avaliação (avaliacao_utils): R² fora da amostra contra as três referências;
RMSE; MAE; Clark–West contra a referência principal (REFERENCIA, parâmetro)
e Diebold–Mariano entre todos os pares, com HAC de h+1 = 31 defasagens; R²
dentro da amostra (média entre origens, só referência); % de previsões
negativas; média das previsões nos dias com realizado > 0; LASSO x MQO.
Placebo (janela expansiva), SÓ COMO VERIFICAÇÃO DE VAZAMENTO: alvo
embaralhado dentro do treino de cada origem; 10 permutações nas árvores
(hiperparâmetros da rodada principal) e 200 nos lineares (α reescolhido;
em paralelo). Critério: mediana do R² placebo fora da amostra (contra a
média histórica) <= 0 em cada modelo. NÃO é teste de significância: as
permutações destroem a autocorrelação dos alvos sobrepostos (distribuição
placebo estreita demais) -- a significância vem do Clark–West e do
Diebold–Mariano com HAC.

Saídas: data/previsoes_bvrp_T5.csv (date, origem, janela, modelo, previsao;
sem o alvo, que vira regressor no T10) e tabelas/figuras em outputs/T5/.

Uso:  python code/previsao_bvrp_T5.py
      python code/previsao_bvrp_T5.py --rapido --out-dir <dir> --saida-dados <arq>   (teste de fumaça)
"""

import argparse
import itertools
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
from split_utils import divisoes_validacao_cruzada, gerar_divisoes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALVO = "alvo_bvrp_30d_fut"
H = 30
SEMENTE = 42
N_DOBRAS = 5

REFERENCIAS = ["media_historica", "proxy_retrospectiva", "persistencia_viavel"]
REFERENCIA_PADRAO = "media_historica"
MODELOS = ["MQO", "Ridge", "LASSO", "floresta_aleatoria", "XGBoost"]

# Grades estendidas depois da 1ª rodada, em que o escolhido caía no limite
# (LASSO em α = 10; floresta e XGBoost no canto mais regularizado).
GRADE_RIDGE = [{"alpha": a} for a in np.logspace(-3, 5, 17)]   # 10^-3 a 10^5, passo de meia década
GRADE_LASSO = [{"alpha": a} for a in np.logspace(-3, 2, 16)]   # 10^-3 a 10^2, passo de 1/3 de década
GRADE_RF = [{"max_depth": d, "min_samples_leaf": l, "max_features": m}
            for d, l, m in itertools.product([1, 2, 3, 6, None], [5, 20, 50, 100, 200], [0.2, 1 / 3, 1.0])]
GRADE_XGB = [{"n_estimators": n, "learning_rate": lr, "max_depth": d, "min_child_weight": w}
             for n, lr, d, w in itertools.product([50, 100, 200, 500], [0.01, 0.03, 0.1], [1, 2, 4], [1, 10])]
# Valores de cada hiperparâmetro, para o relatório de escolhas no limite da grade
EIXOS = {"Ridge": {"alpha": np.logspace(-3, 5, 17)}, "LASSO": {"alpha": np.logspace(-3, 2, 16)},
         "floresta_aleatoria": {"max_depth": [1, 2, 3, 6, None], "min_samples_leaf": [5, 20, 50, 100, 200],
                                "max_features": [0.2, 1 / 3, 1.0]},
         "XGBoost": {"n_estimators": [50, 100, 200, 500], "learning_rate": [0.01, 0.03, 0.1],
                     "max_depth": [1, 2, 4], "min_child_weight": [1, 10]}}

COR_REAL, COR_PREV = "#2b6cb0", "#d95f02"  # par validado (skill dataviz, T1)
NOMES = {"media_historica": "Média histórica", "proxy_retrospectiva": "Proxy retrospectiva",
         "persistencia_viavel": "Persistência viável", "MQO": "MQO", "Ridge": "Ridge", "LASSO": "LASSO",
         "floresta_aleatoria": "Floresta aleatória", "XGBoost": "XGBoost"}


# ---------------------------------------------------------------------------
# Fábricas de modelos
# ---------------------------------------------------------------------------
def fabricar(nome, p, n_jobs=-1):
    if nome == "MQO":
        return LinearRegression()
    if nome == "Ridge":
        return Ridge(alpha=p["alpha"])
    if nome == "LASSO":
        return Lasso(alpha=p["alpha"], max_iter=50_000)
    if nome == "floresta_aleatoria":
        return RandomForestRegressor(n_estimators=500, random_state=SEMENTE, n_jobs=n_jobs, **p)
    if nome == "XGBoost":
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEMENTE, n_jobs=n_jobs,
                            verbosity=0, **p)
    raise ValueError(nome)


LINEARES = {"MQO", "Ridge", "LASSO"}


def ajustar(nome, p, X_tr, y_tr, n_jobs=-1):
    """Ajusta; nos lineares, padroniza com média e desvio do treino."""
    if nome in LINEARES:
        sc = StandardScaler().fit(X_tr)
        m = fabricar(nome, p).fit(sc.transform(X_tr), y_tr)
        return lambda X: m.predict(sc.transform(X)), m
    m = fabricar(nome, p, n_jobs=n_jobs).fit(X_tr, y_tr)
    return m.predict, m


def _mse_dobra(nome, p, X, y, tr, va, n_jobs):
    prever, _ = ajustar(nome, p, X[tr], y[tr], n_jobs=n_jobs)
    return mean_squared_error(y[va], prever(X[va]))


def escolher(nome, grade, X, y, idx_treino):
    """
    Hiperparâmetro de menor MSE médio na validação cruzada embargada.
    Nas árvores, os ajustes (combinação x dobra) são distribuídos entre os
    núcleos, cada um com n_jobs = 1 (mesmo resultado, com random_state fixo;
    ~6x mais rápido que n_jobs = -1 em cada ajuste de árvores rasas).
    """
    dobras = divisoes_validacao_cruzada(idx_treino, h=H, n_dobras=N_DOBRAS)
    if nome in LINEARES:
        erros = [_mse_dobra(nome, p, X, y, tr, va, -1) for p in grade for tr, va in dobras]
    else:
        erros = Parallel(n_jobs=-1)(delayed(_mse_dobra)(nome, p, X, y, tr, va, 1)
                                    for p in grade for tr, va in dobras)
    mses = np.asarray(erros).reshape(len(grade), len(dobras)).mean(axis=1)
    i = int(np.argmin(mses))
    return grade[i], float(mses[i])


def limites_da_grade(hiper):
    """Por janela, modelo e hiperparâmetro: nº de origens com o escolhido no menor/maior valor da grade."""
    linhas = []
    for (janela, modelo), g in hiper[hiper.modelo.isin(list(EIXOS))].groupby(["janela", "modelo"]):
        for eixo, valores in EIXOS[modelo].items():
            esc = [p[eixo] for p in g.params_obj]
            ordem = list(valores)  # None (sem limite de profundidade) é o maior valor de max_depth
            igual = (lambda a, b: (a is None and b is None) or (a is not None and b is not None and np.isclose(a, b)))
            no_min = sum(igual(e, ordem[0]) for e in esc)
            no_max = sum(igual(e, ordem[-1]) for e in esc)
            linhas.append({"janela": janela, "modelo": modelo, "hiperparametro": eixo, "n_origens": len(esc),
                           "menor_valor_grade": ordem[0], "n_no_menor": no_min,
                           "maior_valor_grade": ordem[-1], "n_no_maior": no_max})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Laço de previsão
# ---------------------------------------------------------------------------
def rodar(df, feats_lin, feats_arv, janela, grades, max_origens=None, placebo=False, hiper_fixos=None,
          modelos=MODELOS, verboso=True):
    datas = df["date"].to_numpy()
    y_real = df[ALVO].to_numpy(dtype=float)
    X_lin = df[feats_lin].to_numpy(dtype=float)
    X_arv = df[feats_arv].to_numpy(dtype=float)
    div = gerar_divisoes(len(df), h=H, janela=janela)
    origens = div.origens[:max_origens] if max_origens else div.origens

    linhas, hiper = [], []
    for k, o in enumerate(origens):
        t0 = time.perf_counter()
        y = y_real.copy()
        if placebo:  # placebo = número da permutação (1, 2, ...): semente distinta por permutação e origem
            y[o.treino] = np.random.default_rng([SEMENTE, int(placebo), o.t]).permutation(y_real[o.treino])
        base = {"origem": datas[o.t], "janela": janela}

        prevs = {}
        if not placebo:
            prevs["media_historica"] = np.full(len(o.teste), y[o.treino].mean())
            prevs["proxy_retrospectiva"] = df["vrp_30d"].to_numpy()[o.teste]
            prevs["persistencia_viavel"] = df["bvrp_realizado_defasado"].to_numpy()[o.teste]

        for nome in modelos:
            X = X_lin if nome in LINEARES else X_arv
            if nome == "MQO":
                p, mse_cv = {}, np.nan
            elif hiper_fixos is not None and nome not in LINEARES:
                p, mse_cv = hiper_fixos[(janela, str(datas[o.t])[:10], nome)], np.nan
            else:
                p, mse_cv = escolher(nome, grades[nome], X, y, o.treino)
            prever, _ = ajustar(nome, p, X[o.treino], y[o.treino])
            prevs[nome] = prever(X[o.teste])
            hiper.append({**base, "modelo": nome, "n_treino": len(o.treino),
                          "treino_fim": datas[o.treino[-1]], "params": repr(p), "params_obj": p, "mse_cv": mse_cv,
                          "r2_dentro_amostra": r2_score(y[o.treino], prever(X[o.treino]))})

        for nome, v in prevs.items():
            for u, val in zip(o.teste, v):
                linhas.append({"date": datas[u], **base, "modelo": nome, "previsao": float(val)})
        if verboso:
            print(f"  [{janela}{f' placebo {placebo}' if placebo else ''}] origem {k + 1}/{len(origens)} "
                  f"{str(datas[o.t])[:10]}: treino {len(o.treino)} obs., {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(hiper)


def placebo_uma_permutacao(b, modelos, *, df, feats_lin, feats_arv, grades, max_origens, hiper_fixos, media):
    """Uma permutação do placebo (janela expansiva): R² fora da amostra contra a média histórica."""
    pp, _ = rodar(df, feats_lin, feats_arv, "expansiva", grades, max_origens=max_origens, placebo=b,
                  hiper_fixos=hiper_fixos, modelos=modelos, verboso=False)
    alvo = df.set_index("date")[ALVO]
    out = []
    for m, g in pp.groupby("modelo"):
        g = g.set_index("date").sort_index()
        out.append({"permutacao": b, "modelo": m, "r2_oos_vs_media_historica":
                    r2_fora_da_amostra(alvo.loc[g.index], g["previsao"], media.loc[g.index])})
    return out


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def avaliar(prev, hiper, df, referencia):
    alvo = df.set_index("date")[ALVO]
    metricas, dm_rows, lxm = [], [], []
    for janela, g in prev.groupby("janela"):
        w = g.pivot(index="date", columns="modelo", values="previsao").sort_index()
        y = alvo.loc[w.index].to_numpy()
        pos = y > 0
        for m in w.columns:
            p = w[m].to_numpy()
            row = {"janela": janela, "modelo": m, "n": len(p),
                   "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
                   "pct_negativas": 100 * float((p < 0).mean()),
                   "media_prev_dias_realizado_pos": float(p[pos].mean())}
            for ref in REFERENCIAS:
                row[f"r2_oos_vs_{ref}"] = r2_fora_da_amostra(y, p, w[ref]) if m != ref else np.nan
            if m != referencia:
                row.update({f"{k}_vs_{referencia}": v for k, v in clark_west(y, p, w[referencia], h=H).items()})
            r2in = hiper[(hiper.janela == janela) & (hiper.modelo == m)]["r2_dentro_amostra"]
            row["r2_dentro_amostra_media"] = float(r2in.mean()) if len(r2in) else np.nan
            metricas.append(row)
        for m1, m2 in itertools.combinations(w.columns, 2):
            dm_rows.append({"janela": janela, "modelo_1": m1, "modelo_2": m2,
                            **diebold_mariano(y, w[m1], w[m2], h=H)})
        if {"LASSO", "MQO"} <= set(w.columns):
            d = w["LASSO"] - w["MQO"]
            lxm.append({"janela": janela, "max_dif_abs": float(d.abs().max()),
                        "corr": float(np.corrcoef(w["LASSO"], w["MQO"])[0, 1])})
    return pd.DataFrame(metricas), pd.DataFrame(dm_rows), pd.DataFrame(lxm)


def figura(prev, df, caminho):
    alvo = df.set_index("date")[ALVO]
    w = prev[prev.janela == "expansiva"].pivot(index="date", columns="modelo", values="previsao").sort_index()
    y = alvo.loc[w.index]
    paineis = [m for m in ["media_historica", "persistencia_viavel", "proxy_retrospectiva",
                           "LASSO", "floresta_aleatoria", "XGBoost"] if m in w.columns]
    fig, axes = plt.subplots(len(paineis), 1, figsize=(11, 2.6 * len(paineis)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, paineis):
        r2 = r2_fora_da_amostra(y, w[m], w["media_historica"]) if m != "media_historica" else 0.0
        ax.plot(w.index, y, color=COR_REAL, linewidth=1.2, label="BVRP prospectivo realizado")
        ax.plot(w.index, w[m], color=COR_PREV, linewidth=1.0, linestyle="--",
                label=f"{NOMES.get(m, m)} (R² fora da amostra vs. média = {f'{r2:.3f}'.replace('.', ',')})")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylabel("p.p.")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(v, d=3):
    return "--" if pd.isna(v) else f"{v:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def tabela_tex(met, referencia, caminho):
    m = met[met.janela == "expansiva"].set_index("modelo")
    ordem = [x for x in REFERENCIAS + MODELOS if x in m.index]
    cw = f"cw_p_unilateral_vs_{referencia}"
    linhas = [r"\begin{table}[H]", r"\centering", r"\small",
              r"\caption{Previsão do BVRP prospectivo fora da amostra (janela expansiva, 21/06/2023 a 03/03/2026).}",
              r"\label{tab:T5-previsao}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Modelo & $R^2$ vs. média & $R^2$ vs. persist. & RMSE & MAE & CW ($p$) & $R^2$ dentro \\",
              r"\midrule"]
    for x in ordem:
        r = m.loc[x]
        linhas.append(f"{NOMES.get(x, x)} & {fmt(r['r2_oos_vs_media_historica'])} & "
                      f"{fmt(r['r2_oos_vs_persistencia_viavel'])} & {fmt(r['rmse'], 2)} & {fmt(r['mae'], 2)} & "
                      f"{fmt(r.get(cw, np.nan))} & {fmt(r['r2_dentro_amostra_media'])} \\\\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
               r"\footnotesize\textit{Nota}: $R^2$ fora da amostra $= 1 - \text{SSE}_{\text{modelo}}/\text{SSE}_{\text{referência}}$; "
               r"média histórica calculada só com alvos conhecidos em cada origem ($s \leq t - 30$). "
               r"CW: Clark--West contra a média histórica, unilateral, HAC com $h+1 = 31$ defasagens. "
               r"$R^2$ dentro da amostra: média entre as 33 reestimações.",
               r"\end{table}"]
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--janelas", nargs="+", default=["expansiva", "movel"])
    ap.add_argument("--referencia", default=REFERENCIA_PADRAO, choices=REFERENCIAS)
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T5"))
    ap.add_argument("--saida-dados", default=os.path.join("data", "previsoes_bvrp_T5.csv"))
    ap.add_argument("--sem-placebo", action="store_true")
    ap.add_argument("--permutacoes-arvores", type=int, default=10, help="permutações do placebo nas árvores")
    ap.add_argument("--permutacoes-lineares", type=int, default=200, help="permutações do placebo nos lineares")
    ap.add_argument("--rapido", action="store_true", help="teste de fumaça: 2 origens e grades mínimas")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    absoluto = (lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p))
    out_dir, saida = absoluto(args.out_dir), absoluto(args.saida_dados)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    assert (df["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "datas não contínuas"
    feats_arv = [c for c in df.columns if c not in ("date", ALVO)]
    feats_lin = [c for c in feats_arv if c != "vh_30d"]
    print(f"N = {len(df)} ({df.date.iloc[0].date()} a {df.date.iloc[-1].date()}); "
          f"lineares: {len(feats_lin)} variáveis; árvores: {len(feats_arv)}")

    grades = {"Ridge": GRADE_RIDGE, "LASSO": GRADE_LASSO, "floresta_aleatoria": GRADE_RF, "XGBoost": GRADE_XGB}
    max_origens = None
    if args.rapido:
        grades = {k: v[:2] for k, v in grades.items()}
        max_origens = 2

    prevs, hipers = [], []
    for janela in args.janelas:
        p, h = rodar(df, feats_lin, feats_arv, janela, grades, max_origens=max_origens)
        prevs.append(p)
        hipers.append(h)
    prev, hiper = pd.concat(prevs), pd.concat(hipers)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    prev[["date", "origem", "janela", "modelo", "previsao"]].to_csv(saida, index=False)
    hiper.drop(columns="params_obj").to_csv(os.path.join(out_dir, "hiperparametros_T5.csv"), index=False)
    lim = limites_da_grade(hiper)
    lim.to_csv(os.path.join(out_dir, "limites_grade_T5.csv"), index=False)

    met, dm, lxm = avaliar(prev, hiper, df, args.referencia)
    met.to_csv(os.path.join(out_dir, "metricas_T5.csv"), index=False)
    dm.to_csv(os.path.join(out_dir, "diebold_mariano_T5.csv"), index=False)
    lxm.to_csv(os.path.join(out_dir, "lasso_vs_mqo_T5.csv"), index=False)

    if not args.sem_placebo and "expansiva" in args.janelas:
        fixos = {(r.janela, str(r.origem)[:10], r.modelo): r.params_obj
                 for r in hiper[hiper.janela == "expansiva"].itertuples() if r.modelo not in LINEARES}
        media = prev[(prev.janela == "expansiva") & (prev.modelo == "media_historica")].set_index("date")["previsao"]
        comum = dict(df=df, feats_lin=feats_lin, feats_arv=feats_arv, grades=grades,
                     max_origens=max_origens, hiper_fixos=fixos, media=media)
        arvores = [m for m in MODELOS if m not in LINEARES]
        lineares = [m for m in MODELOS if m in LINEARES]
        t0 = time.perf_counter()
        linhas = [r for b in range(1, args.permutacoes_arvores + 1)
                  for r in placebo_uma_permutacao(b, arvores, **comum)]
        print(f"  placebo das árvores: {args.permutacoes_arvores} permutações em {time.perf_counter() - t0:.0f}s", flush=True)
        t0 = time.perf_counter()
        par = Parallel(n_jobs=-1)(delayed(placebo_uma_permutacao)(b, lineares, **comum)
                                   for b in range(1, args.permutacoes_lineares + 1))
        linhas += [r for bloco in par for r in bloco]
        print(f"  placebo dos lineares: {args.permutacoes_lineares} permutações em {time.perf_counter() - t0:.0f}s", flush=True)
        pl = pd.DataFrame(linhas)
        pl.to_csv(os.path.join(out_dir, "placebo_permutacoes_T5.csv"), index=False)
        resumo = pl.groupby("modelo")["r2_oos_vs_media_historica"].agg(["count", "median", "mean", "min", "max"])
        resumo.columns = ["n_permutacoes"] + [f"r2_placebo_{c}" for c in ["mediana", "media", "min", "max"]]
        resumo.reset_index().to_csv(os.path.join(out_dir, "placebo_T5.csv"), index=False)

    if "expansiva" in args.janelas:
        figura(prev, df, os.path.join(out_dir, "fig_T5_previsoes.png"))
        tabela_tex(met, args.referencia, os.path.join(out_dir, "tab_T5_previsao.tex"))

    pd.set_option("display.width", 250)
    cols = ["janela", "modelo", "r2_oos_vs_media_historica", "r2_oos_vs_persistencia_viavel",
            "r2_oos_vs_proxy_retrospectiva", "rmse", "mae", f"cw_p_unilateral_vs_{args.referencia}",
            "r2_dentro_amostra_media", "pct_negativas", "media_prev_dias_realizado_pos"]
    print(met[[c for c in cols if c in met.columns]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaídas: {saida} e {out_dir}")


if __name__ == "__main__":
    main()
