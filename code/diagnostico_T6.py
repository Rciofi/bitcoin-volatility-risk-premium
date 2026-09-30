"""
diagnostico_T6.py — diagnóstico do erro da Figura 6.1 (T6 do plano de revisão)

Sintomas (Prof. Marcelo, PDF anotado p. 58 e reunião de 29/09/2026, 1:15):
"parece estar sempre indicando BVRP negativo" e "o laranja [LASSO] nunca vai
para cima... não sobe do 0".

Reproduz a especificação de scripts/regenerate_cap8.py na versão que gerou a
figura publicada (commit 1e1f146), com os dados daquele commit (lidos via
`git show`, sem alterar data/), e compara variantes. NÃO altera nenhum script
antigo e grava só em outputs/T6/.

Especificação original (1e1f146): alvo = vrp_30d (proxy retrospectiva) em t+1;
11 variáveis (ret, month, weekday, is_month_start, is_month_end, ret_lag_1,
ret_lag_5, ret_lag_20, d_iv_1d, d_vrp_1d, vrp_regime_num); vrp_30d, rv_30d e
iv_30d em t EXCLUÍDAS; StandardScaler no treino; divisão 70/30 fixa;
LASSO com alpha = 0,001; MQO sem regularização.

Variantes:
  A original: com vrp_regime_num, sem BVRP defasado;
  B sem vrp_regime_num e sem BVRP defasado;
  C com vrp_30d em t (nível contínuo) e sem vrp_regime_num;
  persistência pura: previsão = vrp_30d em t.

Uso:  python code/diagnostico_T6.py
"""

import io
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "T6")
COMMIT_FIGURA = "1e1f146"
ALPHA_LASSO = 0.001

TARGET = "vrp_30d"
COLS_EXCLUIR = [TARGET, "close", "iv_30d", "rv_30d", "ret_fut_1d", "ret_fut_5d", "ret_fut_20d",
                "ret_fut_10d", "ret_fut_30d", "ret_fut_60d", "vrp_regime"]

# Tabs. 6.1 e 6.2 publicadas (PDF anotado): conferência da reprodução
COEF_PUBLICADO = {"vrp_regime_num": 9.1810, "d_iv_1d": 0.7761, "d_vrp_1d": 0.7665, "ret_lag_5": 0.5625,
                  "ret": 0.4919, "ret_lag_20": 0.4370, "weekday": -0.2670, "is_month_end": 0.1945,
                  "ret_lag_1": 0.1358, "is_month_start": 0.0291, "month": -0.0028}
R2_PUBLICADO = {"LASSO": 0.388, "MQO": 0.388, "Persistência": 0.937}

COR_REAL, COR_PREV = "#2b6cb0", "#d95f02"  # par validado (skill dataviz, T1)


def git_csv(caminho, commit=COMMIT_FIGURA):
    txt = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:{caminho}"],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    return pd.read_csv(io.StringIO(txt), parse_dates=["date"])


def ajustar(X_tr, y_tr, X_te):
    """LASSO (alpha fixo) e MQO com padronização ajustada só no treino."""
    sc = StandardScaler().fit(X_tr)
    Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
    lasso = Lasso(alpha=ALPHA_LASSO, max_iter=50_000).fit(Xtr, y_tr)
    mqo = LinearRegression().fit(Xtr, y_tr)
    return {"LASSO": (lasso, lasso.predict(Xte)), "MQO": (mqo, mqo.predict(Xte))}


def resumo(nome, prev, real):
    pos = real > 0
    return {"serie": nome, "media": prev.mean(), "desvio_padrao": prev.std(ddof=1),
            "minimo": prev.min(), "maximo": prev.max(), "pct_negativo": 100 * (prev < 0).mean(),
            "r2_oos": r2_score(real, prev) if nome != "Realizado" else np.nan,
            "rmse": np.sqrt(mean_squared_error(real, prev)) if nome != "Realizado" else np.nan,
            "corr_com_realizado": np.corrcoef(prev, real)[0, 1] if nome != "Realizado" else np.nan,
            "media_nos_dias_realizado_pos": prev[pos].mean(),
            "pct_prev_pos_nos_dias_realizado_pos": 100 * (prev[pos] > 0).mean()}


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------- dados
    df = git_csv("data/ml_dataset.csv").sort_values("date").set_index("date")
    y_level = df[TARGET]
    y = y_level.shift(-1).rename("alvo_t1")
    X_orig = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns])
    base = pd.concat([y, y_level.rename("vrp_30d_t"), X_orig], axis=1).dropna()
    split = int(0.7 * len(base))
    tr, te = base.iloc[:split], base.iloc[split:]
    print(f"Dados de {COMMIT_FIGURA}: N = {len(base)} | treino {len(tr)} "
          f"({tr.index[0].date()} a {tr.index[-1].date()}) | teste {len(te)} "
          f"({te.index[0].date()} a {te.index[-1].date()})")

    feats_A = list(X_orig.columns)
    feats_B = [c for c in feats_A if c != "vrp_regime_num"]
    feats_C = feats_B + ["vrp_30d_t"]
    variantes = {"A_original": feats_A, "B_sem_regime": feats_B, "C_nivel_continuo": feats_C}

    real = te["alvo_t1"].to_numpy()
    prev = pd.DataFrame({"realizado": real}, index=te.index)
    coefs, stats, dif = [], [resumo("Realizado", real, real)], []
    for v, feats in variantes.items():
        fits = ajustar(tr[feats], tr["alvo_t1"], te[feats])
        for mod, (est, p) in fits.items():
            prev[f"{v}_{mod}"] = p
            stats.append(resumo(f"{v}_{mod}", p, real))
            coefs += [{"variante": v, "modelo": mod, "variavel": "intercepto", "coef": est.intercept_}] + \
                     [{"variante": v, "modelo": mod, "variavel": f, "coef": c} for f, c in zip(feats, est.coef_)]
        pl, pm = fits["LASSO"][1], fits["MQO"][1]
        dif.append({"variante": v, "n_variaveis": len(feats),
                    "lasso_coef_nao_nulos": int(np.sum(fits["LASSO"][0].coef_ != 0)),
                    "max_dif_abs_lasso_mqo": float(np.max(np.abs(pl - pm))),
                    "corr_lasso_mqo": float(np.corrcoef(pl, pm)[0, 1])})
    prev["persistencia"] = te["vrp_30d_t"].to_numpy()
    stats.append(resumo("Persistência (vrp_30d em t)", prev["persistencia"].to_numpy(), real))

    stats, coefs, dif = pd.DataFrame(stats), pd.DataFrame(coefs), pd.DataFrame(dif)

    # --------------------------------------------- conferência com o publicado
    cA = coefs[(coefs.variante == "A_original") & (coefs.modelo == "LASSO")].set_index("variavel")["coef"]
    dmax = max(abs(cA[k] - v) for k, v in COEF_PUBLICADO.items())
    assert dmax < 5e-5, f"coeficientes do LASSO diferem da Tab. 6.1 publicada (dif. máx. {dmax})"
    r2 = stats.set_index("serie")["r2_oos"]
    for k, alvo in {"A_original_LASSO": R2_PUBLICADO["LASSO"], "A_original_MQO": R2_PUBLICADO["MQO"],
                    "Persistência (vrp_30d em t)": R2_PUBLICADO["Persistência"]}.items():
        assert abs(round(r2[k], 3) - alvo) < 1e-9, f"R² de {k} = {r2[k]:.4f} difere do publicado {alvo}"
    print(f"Reprodução conferida: coeficientes da Tab. 6.1 (dif. máx. {dmax:.1e}) e R² da Tab. 6.2.")

    # ------------------------------------ regime: janela expansiva ou amostra inteira?
    reg = git_csv("data/vrp_with_regimes.csv").sort_values("date").set_index("date")
    x = reg["vrp_30d"]

    def codigo(q1, q2):
        return pd.Series(np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), index=x.index).where(pd.Series(q1, index=x.index).notna())

    exp = codigo(x.expanding(min_periods=252).quantile(1 / 3), x.expanding(min_periods=252).quantile(2 / 3))
    full = codigo(pd.Series(x.quantile(1 / 3), index=x.index), pd.Series(x.quantile(2 / 3), index=x.index))
    r = df["vrp_regime_num"]
    regime = pd.DataFrame([{"criterio": "janela expansiva (min. 252), como em analyze_vrp_regimes.py",
                            "coincidencia": float((r == exp.loc[r.index]).mean())},
                           {"criterio": "tercis da amostra inteira",
                            "coincidencia": float((r == full.loc[r.index]).mean())}])

    # ---------------------------------------------------------------- saídas
    prev.to_csv(os.path.join(OUT, "previsoes_T6.csv"))
    stats.to_csv(os.path.join(OUT, "estatisticas_T6.csv"), index=False)
    coefs.to_csv(os.path.join(OUT, "coeficientes_T6.csv"), index=False)
    dif.to_csv(os.path.join(OUT, "lasso_vs_mqo_T6.csv"), index=False)
    regime.to_csv(os.path.join(OUT, "regime_T6.csv"), index=False)

    paineis = [("A_original_LASSO", "(a) Original: com regime, sem BVRP defasado"),
               ("B_sem_regime_LASSO", "(b) Sem regime e sem BVRP defasado"),
               ("C_nivel_continuo_LASSO", "(c) Com vrp_30d em t (nível contínuo), sem regime"),
               ("persistencia", "(d) Persistência pura: previsão = vrp_30d em t")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, sharey=True)
    for ax, (col, titulo) in zip(axes, paineis):
        r2v = r2_score(real, prev[col])
        ax.plot(prev.index, prev["realizado"], color=COR_REAL, linewidth=1.4, label="BVRP realizado (t+1)")
        ax.plot(prev.index, prev[col], color=COR_PREV, linewidth=1.1, linestyle="--",
                label=f"Previsto (R² fora da amostra = {r2v:.3f})".replace(".", ","))
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(titulo, loc="left", fontsize=10)
        ax.set_ylabel("BVRP (p.p.)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_T6_variantes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 220)
    print("\n=== Estatísticas das previsões no teste ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== LASSO x MQO ===")
    print(dif.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Regime ===")
    print(regime.to_string(index=False))
    print(f"\nSaídas em: {OUT}")


if __name__ == "__main__":
    main()
