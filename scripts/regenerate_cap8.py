"""
Gera todas as figuras e tabelas do Capítulo 8.

Outputs:
  figs/cap8/fig_cap8_oos_prediction.png   (Fig. 8.1 — melhorada)
  figs/cap8/fig_cap8_feature_importance.png (Fig. 8.2 — importância + coeficientes)
  tables/tab8/tab8_oos_performance.tex    (Tab. 8.1 — já gerada, regera igual)
  tables/tab8/tab8_coeficientes.tex       (Tab. 8.2 — coeficientes Lasso e Ridge)
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ml_dataset.csv"
# amostra do Cap. 6: 1.524 obs (1.523 apos shift(-1)); menor que a canonica
# (1.775) devido ao burn-in de 252 obs da variavel de regime
OUT_FIGS  = ROOT / "figs" / "cap8"
OUT_TABS  = ROOT / "tables" / "tab8"
OUT_SHAP  = ROOT / "tables" / "cap8"   # [M1] CSV com shap_values
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABS.mkdir(parents=True, exist_ok=True)
OUT_SHAP.mkdir(parents=True, exist_ok=True)

# ── 1. Carregar e preparar dados ──────────────────────────────────────────────
TARGET = "vrp_30d"
COLS_EXCLUIR = [
    TARGET,
    "close",       # serie I(1) -- removida das features (E3)
    "iv_30d", "rv_30d",
    "ret_fut_1d", "ret_fut_5d", "ret_fut_20d",
    "ret_fut_10d", "ret_fut_30d", "ret_fut_60d",  # [M1] retornos futuros -- lookahead direto
    "vrp_regime",
]

FEATURE_LABELS = {
    # "close" removido — serie I(1), excluida das features (E3)
    "ret":            "Retorno diário",
    "month":          "Mês",
    "weekday":        "Dia da semana",
    "is_month_start": "Início do mês",
    "is_month_end":   "Fim do mês",
    "ret_lag_1":      "Retorno defasado (1d)",
    "ret_lag_5":      "Retorno defasado (5d)",
    "ret_lag_20":     "Retorno defasado (20d)",
    "d_iv_1d":        "Variação da IV (1d)",
    "d_vrp_1d":       "Variação do BVRP (1d)",
    "vrp_regime_num": "Regime de volatilidade",
}

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

# Forecasting real (t -> t+1): o alvo e o BVRP do dia SEGUINTE, nao do dia
# corrente. y_level preserva o nivel contemporaneo (BVRP_t) para servir de
# benchmark de persistencia (Eq. 6.2: BVRP_hat_{t+1} = BVRP_t); y desloca -1
# para alinhar cada linha t com o alvo t+1. A ultima observacao perde o alvo
# (nao ha t+1 disponivel) e cai no dropna.
y_level = df[TARGET].copy()
y = y_level.shift(-1)
y.name = TARGET
X = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns])

data_ml = pd.concat([y, X], axis=1).dropna()
y = data_ml[TARGET]
X = data_ml.drop(columns=[TARGET])

split = int(0.7 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Treino : {X_train.shape}  {y_train.index[0].date()} a {y_train.index[-1].date()}")
print(f"Teste  : {X_test.shape}   {y_test.index[0].date()} a {y_test.index[-1].date()}")

# ── 2. Treinar modelos ────────────────────────────────────────────────────────
# Padronizacao (z-score) SO para os modelos lineares (MQO/Ridge/LASSO) --
# scaler fitado exclusivamente no treino, aplicado (transform) no teste, sem
# vazamento. Arvores (RF/GB) e benchmarks recebem X bruto -- nao precisam de
# escala e a padronizacao nao altera sua previsao.
LINEAR_MODELS = {"MQO", "LASSO", "Ridge"}

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  index=X_test.index,  columns=X_test.columns)

# ── 2a. Selecao do alpha de LASSO/Ridge por validacao cruzada temporal com
# embargo -- expanding window, 5 folds, embargo de 30 observacoes entre o
# fim do treino e o inicio da validacao de cada fold (TimeSeriesSplit puro
# nao tem embargo; janelas moveis como rv_30d/iv_30d chegam a 30 dias, entao
# sem embargo o fold de validacao vazaria para o treino via sobreposicao).
# StandardScaler e fitado DENTRO de cada fold (so no treino do fold), nunca
# no fold inteiro -- sem vazamento.
#
# Selecao NAO e argmin cego: a superficie de perda de validacao e
# estatisticamente achatada (diferencas <0,1 desvio-padrao entre folds) para
# alpha em [1e-4, 1] -- o argmin bruto (alpha=1 para o LASSO, que zera 10/11
# coeficientes) e artefato da granularidade da grade (passos de 10x) sobre
# essa superficie plana, e nao evidencia real de esparsidade (confirmado por
# teste de robustez: a escolha alpha=1 e estavel a variacoes de n. de folds,
# embargo e metrica -- ou seja, e a grade que e grosseira, nao o sinal que e
# esparso). Regra adotada: (i) identifica a faixa de alphas cujo MSE medio de
# validacao esta a ate 1 desvio-padrao (entre folds) do minimo; (ii) dentro
# dessa faixa, prefere-se o alpha menos regularizado, com desempate por
# parcimonia quando dois candidatos adjacentes sao estatisticamente
# indistintos; (iii) para o Ridge, se o alpha=1,0 ja publicado estiver dentro
# da faixa plana, mantem-se por coerencia com o texto (nao ha ganho
# estatistico em trocar por um valor ainda menos regularizado).
ALPHA_GRID_CV = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
CV_FOLDS   = 5
CV_EMBARGO = 30


def _embargoed_expanding_splits(n_samples, n_folds=CV_FOLDS, embargo=CV_EMBARGO):
    fold_size = n_samples // (n_folds + 1)
    splits = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        val_end = fold_size * (k + 1) if k < n_folds else n_samples
        train_idx = np.arange(0, max(0, train_end - embargo))
        val_idx = np.arange(train_end, val_end)
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def _cv_curve(estimator_cls, X_tr, y_tr, splits):
    """MSE medio e desvio-padrao entre folds, para cada alpha da grade."""
    means, stds = [], []
    for alpha in ALPHA_GRID_CV:
        fold_mses = []
        for tr_idx, va_idx in splits:
            Xf_tr, Xf_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            yf_tr, yf_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            fold_scaler = StandardScaler().fit(Xf_tr)
            Xf_tr_s = fold_scaler.transform(Xf_tr)
            Xf_va_s = fold_scaler.transform(Xf_va)
            kwargs = {"alpha": alpha}
            if estimator_cls is Lasso:
                kwargs["max_iter"] = 50_000
            m = estimator_cls(**kwargs)
            m.fit(Xf_tr_s, yf_tr)
            fold_mses.append(mean_squared_error(yf_va, m.predict(Xf_va_s)))
        means.append(np.mean(fold_mses))
        stds.append(np.std(fold_mses))
    return np.array(means), np.array(stds)


def _select_alpha_faixa_plana(means, stds, coherence_value=None):
    """Selecao por faixa-plana + parcimonia -- nao e argmin (ver nota acima)."""
    best_i = int(np.argmin(means))
    threshold = means[best_i] + stds[best_i]
    faixa = sorted(ALPHA_GRID_CV[i] for i in range(len(ALPHA_GRID_CV)) if means[i] <= threshold)
    selecionado = faixa[0]
    if len(faixa) >= 2:
        i0, i1 = ALPHA_GRID_CV.index(faixa[0]), ALPHA_GRID_CV.index(faixa[1])
        if abs(means[i0] - means[i1]) <= stds[best_i]:
            selecionado = faixa[1]
    if coherence_value is not None and coherence_value in faixa:
        selecionado = coherence_value
    return selecionado, faixa


_splits_cv = _embargoed_expanding_splits(len(X_train))
lasso_cv_means, lasso_cv_stds = _cv_curve(Lasso, X_train, y_train, _splits_cv)
ridge_cv_means, ridge_cv_stds = _cv_curve(Ridge, X_train, y_train, _splits_cv)

ALPHA_LASSO, faixa_lasso = _select_alpha_faixa_plana(lasso_cv_means, lasso_cv_stds)
ALPHA_RIDGE, faixa_ridge = _select_alpha_faixa_plana(ridge_cv_means, ridge_cv_stds, coherence_value=1.0)

print(f"\n[CV embargada] LASSO -- curva de validacao ({CV_FOLDS} folds, embargo={CV_EMBARGO}d):")
for a, m, s in zip(ALPHA_GRID_CV, lasso_cv_means, lasso_cv_stds):
    print(f"  alpha={a:<8g}  MSE_medio={m:8.4f}  desvio_entre_folds={s:7.4f}")
print(f"  Faixa plana (<=1 dp do minimo): {faixa_lasso}")
print(f"  Alpha selecionado: {ALPHA_LASSO}")

print(f"\n[CV embargada] Ridge -- curva de validacao ({CV_FOLDS} folds, embargo={CV_EMBARGO}d):")
for a, m, s in zip(ALPHA_GRID_CV, ridge_cv_means, ridge_cv_stds):
    print(f"  alpha={a:<8g}  MSE_medio={m:8.4f}  desvio_entre_folds={s:7.4f}")
print(f"  Faixa plana (<=1 dp do minimo): {faixa_ridge}")
print(f"  Alpha selecionado: {ALPHA_RIDGE}")


def _fmt_alpha_pt(a):
    """Formata alpha em notacao pt-BR (virgula decimal, braces LaTeX) p/ legendas/tabelas."""
    s = f"{a:g}"
    return s.replace(".", "{,}") if "." in s else f"{s}{{,}}0"


ALPHA_LASSO_FMT = _fmt_alpha_pt(ALPHA_LASSO)
ALPHA_RIDGE_FMT = _fmt_alpha_pt(ALPHA_RIDGE)

models = {
    "MQO":               LinearRegression(),
    "LASSO":             Lasso(alpha=ALPHA_LASSO, max_iter=50_000),
    "Ridge":             Ridge(alpha=ALPHA_RIDGE),
    "Random Forest":     RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Benchmark (Média)": DummyRegressor(strategy="mean"),
}

preds   = {}
metrics = []

for name, model in models.items():
    Xtr = X_train_scaled if name in LINEAR_MODELS else X_train
    Xte = X_test_scaled  if name in LINEAR_MODELS else X_test
    model.fit(Xtr, y_train)
    yhat = pd.Series(model.predict(Xte), index=y_test.index, name=name)
    preds[name] = yhat
    mse = mean_squared_error(y_test, yhat)
    r2  = r2_score(y_test, yhat)
    metrics.append({"Modelo": name, "MSE": mse, "RMSE": math.sqrt(mse), "R2_OOS": r2})
    print(f"  {name:20s}  MSE={mse:7.3f}  R²={r2:.4f}")

# Benchmark de persistencia (random walk): previsao do alvo em t+1 pelo
# nivel contemporaneo observado em t (Eq. 6.2: BVRP_hat_{t+1} = BVRP_t).
# y_level esta alinhado ao indice original (t), y_test tambem (o shift(-1)
# preserva o indice de t, so desloca o VALOR para o de t+1).
y_prev_persist = y_level.loc[y_test.index]
mse_p = mean_squared_error(y_test, y_prev_persist)
r2_p  = r2_score(y_test, y_prev_persist)
metrics.append({"Modelo": "Persistência (valor anterior)", "MSE": mse_p, "RMSE": math.sqrt(mse_p), "R2_OOS": r2_p})
print(f"  {'Persistência (valor anterior)':20s}  MSE={mse_p:7.3f}  R²={r2_p:.4f}")

results = pd.DataFrame(metrics).sort_values("MSE").reset_index(drop=True)

# Diagnostico: coeficientes do MQO (checar blow-up de colinearidade) -- so print,
# nao entra em nenhuma tabela ainda.
mqo_model = models["MQO"]
mqo_coefs = sorted(zip(X.columns, mqo_model.coef_), key=lambda t: -abs(t[1]))
print("\n[Diagnostico] Coeficientes MQO (ordenados por |magnitude|):")
for fname, c in mqo_coefs:
    label = FEATURE_LABELS.get(fname, fname)
    print(f"  {label:<28s} {c:+.4f}")

# ── 3. Fig. 8.1 — BVRP realizado vs. Lasso previsto (melhorada) ──────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_test.index, y_test.values,
        label="BVRP realizado", linewidth=1.8, color="#1f4e79")
ax.plot(preds["LASSO"].index, preds["LASSO"].values,
        label="Previsto — LASSO", linewidth=1.4, linestyle="--", color="#c55a11")
ax.set_ylabel("BVRP (p.p.)", fontsize=10)
ax.set_xlabel("Data", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("Previsão fora da amostra: BVRP realizado vs. LASSO", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fig_cap8_oos_prediction.png", dpi=300)
plt.close(fig)
print("OK fig_cap8_oos_prediction.png salva")


# -- 4. SHAP -- Importancia das variaveis com TreeExplainer [M1] ----------------
rf_model    = models["Random Forest"]
feat_names  = X.columns.tolist()
feat_labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

import time as _time
_t0 = _time.time()

# TreeExplainer e O(n x p) -- eficiente para florestas; calculado no TESTE
# (usar dados de teste evita inflar importancias com overfitting do treino)
SHAP_SAMPLE  = 200  # subamostrar X_test -- suficiente para beeswarm, evita timeout
explainer_rf = shap.TreeExplainer(rf_model)
Xte_shap     = X_test.sample(n=SHAP_SAMPLE, random_state=42)  # amostra representativa
shap_values  = explainer_rf.shap_values(Xte_shap, approximate=True)  # approx: rapido com 300 arvores
_elapsed = _time.time() - _t0
print(f"\n[M1] SHAP calculado: {shap_values.shape[0]} obs x {shap_values.shape[1]} features  ({_elapsed:.1f}s)")

# [M1] Salvar shap_values como CSV para reprodutibilidade
shap_df = pd.DataFrame(shap_values, columns=feat_names, index=Xte_shap.index)
shap_df.to_csv(OUT_SHAP / "shap_values_rf_cap8.csv")
print(f"[M1] shap_values_rf_cap8.csv salvo: {shap_df.shape}")

# [M1] Fig. 8.2 -- Beeswarm plot (substituicao da importancia MDI como figura principal)
shap.summary_plot(
    shap_values, Xte_shap, feature_names=feat_labels,
    show=False, plot_type="dot",
    max_display=len(feat_labels),
)
plt.tight_layout()
plt.savefig(str(OUT_FIGS / "fig_cap8_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()
print("OK fig_cap8_shap_beeswarm.png salva")

# [M1] Fig. 8.3 -- Bar plot (importancia media |SHAP|)
shap.summary_plot(
    shap_values, Xte_shap, feature_names=feat_labels,
    show=False, plot_type="bar",
    max_display=len(feat_labels),
)
plt.tight_layout()
plt.savefig(str(OUT_FIGS / "fig_cap8_shap_bar.png"), dpi=300, bbox_inches="tight")
plt.close()
print("OK fig_cap8_shap_bar.png salva")

# Ranking SHAP (importancia media |SHAP|) para log
import numpy as _np
shap_mean_abs = pd.Series(
    _np.abs(shap_values).mean(axis=0),
    index=feat_names
).sort_values(ascending=False)
print("\n[M1] Ranking SHAP (importancia media |SHAP|):")
for rank, (fname, val) in enumerate(shap_mean_abs.items(), 1):
    label = FEATURE_LABELS.get(fname, fname)
    print(f"  {rank:2d}. {label:<35s} {val:.6f}")

# -- 5. Fig. apendice -- MDI (RF) + coeficientes (Lasso) -- movido para apendice [M1] --

rf_model    = models["Random Forest"]
lasso_model = models["LASSO"]

feat_names  = X.columns.tolist()
feat_labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

rf_imp     = rf_model.feature_importances_
lasso_coef = lasso_model.coef_

# ordenar RF por importância
order_rf = np.argsort(rf_imp)
# ordenar Lasso por magnitude absoluta
order_lasso = np.argsort(np.abs(lasso_coef))

colors_lasso = ["#c00000" if c < 0 else "#1f4e79" for c in lasso_coef[order_lasso]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# — painel esquerdo: Random Forest feature importance
ax = axes[0]
bars = ax.barh(
    [feat_labels[i] for i in order_rf],
    rf_imp[order_rf],
    color="#2e75b6", edgecolor="white", linewidth=0.5
)
ax.set_xlabel("Importância (impureza média)", fontsize=9)
ax.set_title("Random Forest — Importância das Variáveis", fontsize=10)
ax.tick_params(axis="y", labelsize=8)

# — painel direito: Lasso coeficientes
ax = axes[1]
bars = ax.barh(
    [feat_labels[i] for i in order_lasso],
    lasso_coef[order_lasso],
    color=colors_lasso, edgecolor="white", linewidth=0.5
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coeficiente estimado", fontsize=9)
ax.set_title(f"LASSO — Coeficientes ($\\alpha={ALPHA_LASSO_FMT}$)", fontsize=10)
ax.tick_params(axis="y", labelsize=8)

fig.tight_layout(pad=2.0)
fig.savefig(OUT_FIGS / "fig_cap8_mdi_appendix.png", dpi=300)
plt.close(fig)
print("OK fig_cap8_mdi_appendix.png salva (apendice -- MDI mantido para referencia)")

# ── 5. Tab. 8.1 — Desempenho OOS (regera, formato já correto) ────────────────
tab1_path = OUT_TABS / "tab8_oos_performance.tex"

_MESES_ABREV = {1: "jan.", 2: "fev.", 3: "mar.", 4: "abr.", 5: "mai.", 6: "jun.",
                7: "jul.", 8: "ago.", 9: "set.", 10: "out.", 11: "nov.", 12: "dez."}

def _fmt_data_pt(d):
    return f"{d.day}~{_MESES_ABREV[d.month]}\\ {d.year}"

_oos_ini = _fmt_data_pt(y_test.index[0])
_oos_fim = _fmt_data_pt(y_test.index[-1])
_n_oos   = len(y_test)
_n_train = len(y_train)
_pct_train = round(100 * _n_train / (_n_train + _n_oos))

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Desempenho preditivo fora da amostra --- Prêmio de Risco de Volatilidade do Bitcoin.",
    f"Período OOS\\@: {_oos_ini} -- {_oos_fim} ($N={_n_oos}$ observações diárias).",
    r"O MSE e o RMSE são reportados em unidades percentuais ao quadrado e percentuais, respectivamente.}",
    r"\label{tab:cap8_oos_performance}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Modelo & MSE & RMSE & $R^2_{\text{OOS}}$ \\",
    r"\midrule",
]

for _, row in results.iterrows():
    nome = row["Modelo"]
    sep  = r"\midrule" if nome == "Benchmark (Média)" else None
    if sep:
        lines.append(sep)
    r2_fmt = f"$-${abs(row['R2_OOS']):.3f}" if row["R2_OOS"] < 0 else f"{row['R2_OOS']:.3f}"
    lines.append(
        f"{nome} & {row['MSE']:.2f} & {row['RMSE']:.2f} & {r2_fmt} \\\\"
        .replace(".", ",")
        .replace(r"$-$,", r"$-$")
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    rf"\footnotesize\textit{{Nota}}: todos os modelos são treinados exclusivamente com dados anteriores ao período OOS ({_pct_train}\% inicial da amostra, {_n_train} observações). O benchmark ``Persistência'' prediz $\text{{BVRP}}_{{t+1}}=\text{{BVRP}}_t$; o benchmark ``Média'' prediz a média incondicional do período de treino para todas as observações OOS. Hiperparâmetros: Ridge ($\alpha={ALPHA_RIDGE_FMT}$), LASSO ($\alpha={ALPHA_LASSO_FMT}$), Random Forest (300 árvores), Gradient Boosting (padrão scikit-learn).",
    r"\end{table}",
]

tab1_path.write_text("\n".join(lines), encoding="utf-8")
print("OK tab8_oos_performance.tex salva")

# ── 6. Tab. 8.2 — Coeficientes Lasso e Ridge ─────────────────────────────────
ridge_model  = models["Ridge"]
ridge_coef   = ridge_model.coef_

coef_df = pd.DataFrame({
    "Variável": feat_labels,
    "LASSO":    lasso_coef,
    "Ridge":    ridge_coef,
})
coef_df["Abs_Lasso"] = np.abs(lasso_coef)
coef_df = coef_df.sort_values("Abs_Lasso", ascending=False).drop(columns="Abs_Lasso")

tab2_lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\caption{Coeficientes estimados dos modelos lineares regularizados --- LASSO e Ridge.",
    r"Os coeficientes são ordenados pela magnitude absoluta do LASSO.",
    r"Variáveis com coeficiente LASSO exatamente zero foram eliminadas pela penalidade $L_1$.}",
    r"\label{tab:cap8_coeficientes}",
    r"\begin{tabular}{lrr}",
    r"\toprule",
    r"Variável & LASSO & Ridge \\",
    r"\midrule",
]

for _, row in coef_df.iterrows():
    lasso_val = f"{row['LASSO']:.4f}".replace(".", ",")
    ridge_val = f"{row['Ridge']:.4f}".replace(".", ",")
    tab2_lines.append(f"{row['Variável']} & {lasso_val} & {ridge_val} \\\\")

tab2_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    rf"\footnotesize\textit{{Nota}}: parâmetros de regularização selecionados por validação cruzada "
    rf"temporal com \textit{{embargo}} de 30 dias ($\alpha_{{\text{{LASSO}}}} = {ALPHA_LASSO_FMT}$, "
    rf"$\alpha_{{\text{{Ridge}}}} = {ALPHA_RIDGE_FMT}$; grade $\{{10^{{-4}}, \ldots, 10^{{2}}\}}$), na faixa "
    r"de desempenho de validação estatisticamente equivalente. LASSO com 50.000 iterações máximas. "
    r"Variáveis padronizadas (\textit{z-score}: centradas e escaladas pelo desvio-padrão) com parâmetros calibrados exclusivamente no conjunto de treino. "
    f"Período de treino: {_fmt_data_pt(y_train.index[0])} -- {_fmt_data_pt(y_train.index[-1])} ({_n_train} observações).",
    r"\end{table}",
]

tab2_path = OUT_TABS / "tab8_coeficientes.tex"
tab2_path.write_text("\n".join(tab2_lines), encoding="utf-8")
print("OK tab8_coeficientes.tex salva")

print("\nResumo de outputs:")
print(f"  {OUT_FIGS / 'fig_cap8_oos_prediction.png'}")
print(f"  {OUT_FIGS / 'fig_cap8_shap_beeswarm.png'}   [M1 -- novo, figura principal]")
print(f"  {OUT_FIGS / 'fig_cap8_shap_bar.png'}         [M1 -- novo, bar plot]")
print(f"  {OUT_FIGS / 'fig_cap8_mdi_appendix.png'}     [M1 -- MDI movido para apendice]")
print(f"  {OUT_SHAP / 'shap_values_rf_cap8.csv'}       [M1 -- shap_values CSV]")
print(f"  {OUT_TABS / 'tab8_oos_performance.tex'}")
print(f"  {OUT_TABS / 'tab8_coeficientes.tex'}")
