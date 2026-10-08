"""
publicar_cap5.py — tabelas e figura do Cap. 5 (previsão do BVRP), sem reestimar nada

Lê as saídas do T4, do T5 e do T9 e grava, na notação da seção 14 de
docs/pendencias_T1.md (VH, BVRP, BVRP^proxy):
  tables/previsao_bvrp/tab_dicionario.tex       dicionário das 17 variáveis (T4)
  tables/previsao_bvrp/tab_hiperparametros.tex  grades e escolhas da validação cruzada (T5/T7)
  tables/previsao_bvrp/tab_resultados.tex       desempenho fora da amostra (T5)
  tables/previsao_bvrp/tab_regime.tex           interações com o regime (T9)
  figs/previsao_bvrp/fig_previsoes.png          BVRP e previsões fora da amostra (T5)

Conferência (sai com erro se algo não bater): recalcula, a partir de
data/previsoes_bvrp_T5.csv e do alvo de data/ml_dataset_T4.csv, o R² fora da
amostra, o Clark–West e o Diebold–Mariano contra a média histórica nas duas
janelas e confere com outputs/T5/metricas_T5.csv e diebold_mariano_T5.csv; refaz
o model confidence set da tabela de resultados (T_max, bloco de 60, B = 9.999,
mcs_utils) e confere os valores-p com outputs/MCS/mcs_T5.csv (M2, seção 25); o
mesmo para o R² com regime contra outputs/T9/metricas_interacao_T9.csv; e
confere a contagem de escolhas e de origens no limite da grade com
outputs/T5/limites_grade_T5.csv. Imprime os números citados no texto.

Uso:  python code/publicar_cap5.py [--raiz DIR]
      --raiz: onde gravar tables/ e figs/ (padrão: a raiz do repositório)
"""
import argparse
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliacao_utils import clark_west, diebold_mariano, r2_fora_da_amostra  # noqa: E402
import mcs_utils  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALVO = "alvo_bvrp_30d_fut"
H = 30
TOL = 1e-9
MCS_BLOCO, MCS_B, MCS_ALFA = 60, 9999, 0.10   # especificação principal do MCS (seção 25.2)
COR_REAL, COR_PREV = "#2b6cb0", "#d95f02"   # mesmo par das figuras do T5
MODELOS = ["MQO", "Ridge", "LASSO", "floresta_aleatoria", "XGBoost"]
REFERENCIAS = ["media_historica", "proxy_retrospectiva", "persistencia_viavel"]
NOMES = {"media_historica": "Média histórica", "proxy_retrospectiva": r"$\text{BVRP}^{\text{proxy}}_t$",
         "persistencia_viavel": r"$\text{BVRP}_{t-30}$", "MQO": "MQO",
         "Ridge": "Ridge", "LASSO": "LASSO", "floresta_aleatoria": "Floresta aleatória", "XGBoost": "XGBoost"}

ROTULOS_FIG = {"Ridge": "Ridge", "floresta_aleatoria": "Floresta aleatória",
               "proxy_retrospectiva": r"$\mathrm{BVRP}^{\mathrm{proxy}}_t$",
               "persistencia_viavel": r"Prêmio realizado defasado, $\mathrm{BVRP}_{t-30}$"}

# Dicionário: grupo, símbolo e definição na notação do texto (a janela e o ADF vêm do T4)
DICIONARIO = [
    ("Volatilidade histórica", "vh_1d", r"$\text{VH}_{t:t}$", r"$\sqrt{365}\,|r_t|$"),
    ("", "vh_5d", r"$\text{VH}_{t-4:t}$", r"$\sqrt{(365/5)\sum_{j=0}^{4} r_{t-j}^2}$"),
    ("", "vh_30d", r"$\text{VH}_{t-29:t}$", r"$\sqrt{(365/30)\sum_{j=0}^{29} r_{t-j}^2}$"),
    ("", "vh_60d", r"$\text{VH}_{t-59:t}$", "idem, 60 dias"),
    ("", "vh_90d", r"$\text{VH}_{t-89:t}$", "idem, 90 dias"),
    ("Volatilidade implícita", "iv_30d", r"$\text{IV}_t$", "DVOL no fim de $t$"),
    ("", "d_iv_1d", r"$\Delta_1\text{IV}_t$", r"$\text{IV}_t - \text{IV}_{t-1}$"),
    ("", "d_iv_5d", r"$\Delta_5\text{IV}_t$", r"$\text{IV}_t - \text{IV}_{t-5}$"),
    ("", "iv_menos_ma5d", r"$\text{IV}_t - \overline{\text{IV}}_{t-4:t}$", "desvio da média móvel de 5 dias"),
    ("", "iv_menos_ma30d", r"$\text{IV}_t - \overline{\text{IV}}_{t-29:t}$", "desvio da média móvel de 30 dias"),
    ("Preço", "ret_1d", r"$r_t$", r"$\ln(P_t/P_{t-1})$"),
    ("", "ret_acum_5d", r"$r_{t-4:t}$", r"$\sum_{j=0}^{4} r_{t-j}$"),
    ("", "ret_acum_30d", r"$r_{t-29:t}$", r"$\sum_{j=0}^{29} r_{t-j}$"),
    ("", "log_close_ma30d", r"$\ln(P_t/\overline{P}_{t-29:t})$", "preço relativo à média móvel de 30 dias"),
    ("Prêmio", "vrp_30d", r"$\text{BVRP}^{\text{proxy}}_t$", r"$\text{VH}_{t-29:t} - \text{IV}_t$"),
    ("", "d_vrp_1d", r"$\Delta_1\text{BVRP}^{\text{proxy}}_t$",
     r"$\text{BVRP}^{\text{proxy}}_t - \text{BVRP}^{\text{proxy}}_{t-1}$"),
    ("", "bvrp_realizado_defasado", r"$\text{BVRP}_{t-30}$", r"$\text{VH}_{t-29:t} - \text{IV}_{t-30}$"),
]
TRATAMENTO = {"nível": "nível", "nível (longa memória)": "nível", "diferença": "diferença",
              "desvio da média móvel (média em nível é I(1))": "desvio da média móvel",
              "log-diferença de close (I(1))": "log-diferença do preço",
              "soma de log-diferenças": "soma de log-diferenças",
              "preço relativo à média móvel (close é I(1))": "razão com a média móvel"}


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{d}f}".replace(".", "{,}")
    return s.replace("-", "$-$") if v < 0 else s


def fmt_p(p):
    if p < 0.001:
        return r"$<$0{,}001"
    return fmt(p, 3)


def fmt_cientifico(p):
    """3,1e-13 -> $3 \\times 10^{-13}$ (uma casa significativa)."""
    mant, exp = f"{p:.0e}".split("e")
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def tabela(legenda, rotulo, colunas, cabecalho, linhas, nota, tamanho=r"\small"):
    corpo = [r"\begin{table}[H]", r"\centering", tamanho, rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}",
             rf"\begin{{tabular}}{{{colunas}}}", r"\toprule", cabecalho + r" \\", r"\midrule"]
    corpo += linhas
    corpo += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{0.95\linewidth}}{{\footnotesize\textit{{Nota}}: {nota}}}", r"\end{table}", ""]
    return "\n".join(corpo)


def conferir(cond, msg):
    if not cond:
        raise SystemExit(f"CONFERÊNCIA FALHOU: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=ROOT)
    args = ap.parse_args()
    dir_tab = os.path.join(args.raiz, "tables", "previsao_bvrp")
    dir_fig = os.path.join(args.raiz, "figs", "previsao_bvrp")
    os.makedirs(dir_tab, exist_ok=True)
    os.makedirs(dir_fig, exist_ok=True)

    t4 = os.path.join(ROOT, "outputs", "T4")
    t5 = os.path.join(ROOT, "outputs", "T5")
    t9 = os.path.join(ROOT, "outputs", "T9")
    df = pd.read_csv(os.path.join(ROOT, "data", "ml_dataset_T4.csv"), parse_dates=["date"]).set_index("date")
    prev = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date"])
    met = pd.read_csv(os.path.join(t5, "metricas_T5.csv")).set_index(["janela", "modelo"])
    dm = pd.read_csv(os.path.join(t5, "diebold_mariano_T5.csv"))
    hip = pd.read_csv(os.path.join(t5, "hiperparametros_T5.csv"), parse_dates=["origem"])
    lim = pd.read_csv(os.path.join(t5, "limites_grade_T5.csv"))
    pla = pd.read_csv(os.path.join(t5, "placebo_T5.csv")).set_index("modelo")
    lvm = pd.read_csv(os.path.join(t5, "lasso_vs_mqo_T5.csv")).set_index("janela")
    dic = pd.read_csv(os.path.join(t4, "dicionario_adf_kpss_T4.csv")).set_index("variavel")
    tf = pd.read_csv(os.path.join(t9, "teste_F_T9.csv"))
    mi = pd.read_csv(os.path.join(t9, "metricas_interacao_T9.csv")).set_index(["regime", "modelo"])
    pi = pd.read_csv(os.path.join(t9, "previsoes_interacao_T9.csv"), parse_dates=["date"])
    mcs = pd.read_csv(os.path.join(ROOT, "outputs", "MCS", "mcs_T5.csv"))
    mcs = mcs[(mcs.bloco == MCS_BLOCO) & (mcs.estatistica == "T_max")].set_index(["janela", "modelo"])

    def dm_media(janela, modelo):
        r = dm[(dm.janela == janela) & (((dm.modelo_1 == modelo) & (dm.modelo_2 == "media_historica"))
                                        | ((dm.modelo_2 == modelo) & (dm.modelo_1 == "media_historica")))]
        conferir(len(r) == 1, f"DM {janela} {modelo}")
        return float(r.dm_p_bilateral.iloc[0])

    # ---------------- Conferência: recálculo das métricas do T5 ----------------
    largo = {j: prev[prev.janela == j].pivot(index="date", columns="modelo", values="previsao").sort_index()
             for j in ("expansiva", "movel")}
    for j, w in largo.items():
        y = df.loc[w.index, ALVO].to_numpy()
        conferir(len(w) == 987, f"{j}: {len(w)} datas")
        for m in MODELOS + REFERENCIAS:
            if m == "media_historica":
                continue
            r2 = r2_fora_da_amostra(y, w[m], w["media_historica"])
            conferir(abs(r2 - met.loc[(j, m), "r2_oos_vs_media_historica"]) < TOL, f"R² {j} {m}")
            cw = clark_west(y, w[m], w["media_historica"], h=H)["cw_p_unilateral"]
            conferir(abs(cw - met.loc[(j, m), "cw_p_unilateral_vs_media_historica"]) < TOL, f"CW {j} {m}")
            d = diebold_mariano(y, w[m], w["media_historica"], h=H)["dm_p_bilateral"]
            conferir(abs(d - dm_media(j, m)) < TOL, f"DM {j} {m}")
        # mesma ordem de colunas do mcs_previsao_T5.py (M2)
        perdas = w[MODELOS + ["media_historica", "persistencia_viavel", "proxy_retrospectiva"]].sub(y, axis=0) ** 2
        idx = mcs_utils.sortear_indices(len(perdas), bloco=MCS_BLOCO, B=MCS_B)
        r = mcs_utils.mcs(perdas, medias_boot=mcs_utils.medias_bootstrap(perdas, idx),
                          estatistica="max").set_index("modelo")
        for m in MODELOS + REFERENCIAS:
            conferir(int(mcs.loc[(j, m), "B"]) == MCS_B, f"MCS {j} {m}: B")
            conferir(abs(r.loc[m, "p_mcs"] - mcs.loc[(j, m), "p_mcs"]) < TOL, f"MCS {j} {m}")
    w = largo["expansiva"]
    y = df.loc[w.index, ALVO]
    for (reg, mod), row in mi.iterrows():
        p = pi[(pi.regime == reg) & (pi.modelo == mod)].set_index("date").previsao.sort_index()
        r2 = r2_fora_da_amostra(y.loc[p.index], p, w.loc[p.index, "media_historica"])
        conferir(abs(r2 - row.r2_oos_vs_media) < TOL, f"R² com regime {reg} {mod}")
    print("Conferência: R², Clark–West, Diebold–Mariano e MCS recalculados batem com outputs/T5, "
          "outputs/MCS e outputs/T9.")

    # ---------------- Dicionário (T4) ----------------
    linhas = []
    for grupo, cod, simb, defin in DICIONARIO:
        r = dic.loc[cod]
        if grupo:
            if linhas:
                linhas.append(r"\addlinespace")
            linhas.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{grupo}}}}} \\")
        linhas.append(rf"{simb} & {defin} & \texttt{{{cod.replace('_', chr(92) + '_')}}} & "
                      rf"{fmt_p(r.adf_p)} & {TRATAMENTO[r.transformacao]} \\")
    conferir(len(DICIONARIO) == 17 and set(c for _, c, _, _ in DICIONARIO) == set(dic.index) - {ALVO},
             "dicionário com as 17 variáveis do T4")
    nota = (r"$r_t$ é o log-retorno diário e $P_t$, o preço de fechamento. Volatilidades em \% a.a. "
            r"Valor-$p$ do teste ADF na amostra de modelagem ($N = 1.776$). As séries em que o ADF não rejeita "
            r"a raiz unitária e o KPSS rejeita a estacionariedade (o preço e as médias móveis da IV) entram "
            r"transformadas; as volatilidades "
            r"entram em nível, inclusive $\text{VH}_{t-89:t}$ e $\text{IV}_t$, cujo ADF fica na fronteira. "
            r"Os modelos lineares excluem $\text{VH}_{t-29:t}$, porque "
            r"$\text{BVRP}^{\text{proxy}}_t = \text{VH}_{t-29:t} - \text{IV}_t$.")
    with open(os.path.join(dir_tab, "tab_dicionario.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tabela("Variáveis explicativas, todas conhecidas no fim do dia $t$.", "tab:prev-dicionario",
                        "lp{4.2cm}lrl", r"Símbolo & Definição & Código & ADF ($p$) & Tratamento",
                        linhas, nota, tamanho=r"\scriptsize"))

    # ---------------- Hiperparâmetros (T5/T7) ----------------
    he = hip[hip.janela == "expansiva"].copy()
    he["p"] = he.params.str.replace("np.float64(", "", regex=False).str.replace(")", "", regex=False) \
        .map(ast.literal_eval)

    def mais_frequente(modelo, chave):
        v = he[he.modelo == modelo].p.map(lambda d: d[chave]).value_counts()
        return v.index[0], int(v.iloc[0])

    def no_limite(modelo, chave, lado):
        r = lim[(lim.janela == "expansiva") & (lim.modelo == modelo) & (lim.hiperparametro == chave)]
        return int(r[f"n_no_{lado}"].iloc[0])

    a_r, n_r = mais_frequente("Ridge", "alpha")
    a_l, n_l = mais_frequente("LASSO", "alpha")
    n_lasso_zero = int((he[he.modelo == "LASSO"].r2_dentro_amostra == 0).sum())
    n_d_rf = no_limite("floresta_aleatoria", "max_depth", "menor")
    n_lr = no_limite("XGBoost", "learning_rate", "menor")
    n_d_xgb = no_limite("XGBoost", "max_depth", "menor")
    n_n_xgb = no_limite("XGBoost", "n_estimators", "menor")
    conferir(no_limite("Ridge", "alpha", "menor") + no_limite("Ridge", "alpha", "maior") == 0, "Ridge interior")
    conferir(no_limite("LASSO", "alpha", "menor") + no_limite("LASSO", "alpha", "maior") == 0, "LASSO interior")
    conferir(len(he.origem.unique()) == 33, "33 origens")

    def pot10(a):
        e = np.log10(a)
        return rf"10^{{{e:.0f}}}" if abs(e - round(e)) < 1e-9 else fmt(a, 2)

    linhas = [
        rf"Ridge & $\alpha$: 17 valores de $10^{{-3}}$ a $10^{{5}}$ & $\alpha = {pot10(a_r)}$ ({n_r}) & 0 \\",
        rf"LASSO & $\alpha$: 16 valores de $10^{{-3}}$ a $10^{{2}}$ & $\alpha = {pot10(a_l)}$ ({n_l}) & 0 \\",
        r"\addlinespace",
        r"Floresta aleatória & profundidade máxima: 1, 2, 3, 6, sem limite & 1 (" + str(n_d_rf) + r") & "
        + str(n_d_rf) + r" \\",
        r" & folha mínima: 5, 20, 50, 100, 200 & & \\",
        r" & fração de variáveis por corte: 0{,}2; 1/3; 1 & & \\",
        r"\addlinespace",
        r"XGBoost & taxa de aprendizado: 0{,}01; 0{,}03; 0{,}1 & 0{,}01 (" + str(n_lr) + r") & "
        + str(n_lr) + r" \\",
        r" & profundidade máxima: 1, 2, 4 & 1 (" + str(n_d_xgb) + r") & " + str(n_d_xgb) + r" \\",
        r" & número de árvores: 50, 100, 200, 500 & 50 (" + str(n_n_xgb) + r") & " + str(n_n_xgb) + r" \\",
        r" & peso mínimo por folha: 1, 10 & & \\",
    ]
    nota = (r"Janela expansiva, 33 reestimações. Em cada uma, validação cruzada com cinco dobras expansivas "
            r"e embargo de 30 dias; escolhe-se a combinação de menor erro quadrático médio. Entre parênteses, "
            r"o número de reestimações com a escolha indicada. Última coluna: reestimações com o "
            r"hiperparâmetro no extremo de menor capacidade da grade. Floresta com 500 árvores; XGBoost "
            r"com 80\% das observações e das variáveis em cada árvore. Os modelos lineares usam variáveis "
            r"padronizadas com a média e o desvio-padrão do treino.")
    with open(os.path.join(dir_tab, "tab_hiperparametros.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tabela("Hiperparâmetros: grades e escolhas da validação cruzada embargada.",
                        "tab:prev-hiperparametros", "lp{6.2cm}lc",
                        r"Modelo & Grade & Mais frequente & No limite", linhas, nota))

    # ---------------- Resultados (T5) ----------------
    def p_mcs(janela, m):
        s = mcs.loc[(janela, m)]
        marca = r"$^{\dagger}$" if s.p_mcs >= MCS_ALFA else ""
        return fmt_p(s.p_mcs) + marca

    linhas = []
    for m in REFERENCIAS + MODELOS:
        e, mv = met.loc[("expansiva", m)], met.loc[("movel", m)]
        if m == "media_historica":
            linhas.append(rf"{NOMES[m]} & -- & -- & -- & -- & {p_mcs('expansiva', m)} & {p_mcs('movel', m)} & -- \\")
            continue
        linhas.append(rf"{NOMES[m]} & {fmt(e.r2_oos_vs_media_historica)} & {fmt(mv.r2_oos_vs_media_historica)} & "
                      rf"{fmt_p(e.cw_p_unilateral_vs_media_historica)} & {fmt_p(mv.cw_p_unilateral_vs_media_historica)} & "
                      rf"{p_mcs('expansiva', m)} & {p_mcs('movel', m)} & {fmt(e.r2_dentro_amostra_media)} \\")
        if m == "persistencia_viavel":
            linhas.append(r"\addlinespace")
    cab = (r" & \multicolumn{2}{c}{$R^2_{\text{fora}}$} & \multicolumn{2}{c}{Clark--West ($p$)} & "
           r"\multicolumn{2}{c}{MCS ($p$)} & $R^2$ dentro \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
           "\n"
           r"Modelo & expansiva & móvel & expansiva & móvel & expansiva & móvel & expansiva")
    nota = (r"$\text{BVRP}^{\text{proxy}}_t$: proxy retrospectiva; $\text{BVRP}_{t-30}$: prêmio realizado "
            r"defasado. 987 previsões, de 21/06/2023 a 03/03/2026. $R^2_{\text{fora}}$: contra a média histórica dos "
            r"alvos conhecidos em cada origem ($s \le t - 30$). Clark--West: unilateral, contra a média "
            r"histórica, com erro-padrão HAC de $h + 1 = 31$ defasagens. MCS: valor-$p$ do "
            r"\textit{model confidence set} de \citet{hansen2011model}, com perda quadrática, estatística "
            r"$T_{\max}$ e bootstrap em blocos móveis de 60 dias (9.999 reamostragens), sobre os oito "
            r"modelos de cada janela; $^{\dagger}$: no conjunto a 10\%. $R^2$ dentro da amostra: média das "
            r"33 reestimações.")
    with open(os.path.join(dir_tab, "tab_resultados.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tabela("Previsão do BVRP fora da amostra.", "tab:prev-resultados", "lrrrrrrr", cab, linhas, nota,
                        tamanho=r"\small\setlength{\tabcolsep}{4pt}"))

    # ---------------- Regime (T9) ----------------
    linhas = [r"\multicolumn{5}{l}{\textit{Painel A: interações dentro da amostra (MQO, $N = 1.776$)}} \\"]
    for _, r in tf.iterrows():
        corte = "fixo" if r.regime == "fixo" else "expansivo"
        h0 = "16 interações" if r.q == 16 else "16 interações e $D_t$"
        linhas.append(rf"Corte {corte} & {h0} & {fmt(r.qui2, 1)} ({int(r.q)}) & {fmt_cientifico(r.p_qui2)} & "
                      rf"{fmt(r.r2_dentro_amostra)} \\")
    linhas += [r"\addlinespace",
               r"\multicolumn{5}{l}{\textit{Painel B: previsão fora da amostra ($R^2_{\text{fora}}$, janela expansiva)}} \\",
               r"Modelo & Sem regime & Corte fixo & Corte expansivo & DM ($p$), fixo \\", r"\cmidrule(lr){1-5}"]
    for mod, nome in [("Ridge_regime", "Ridge"), ("MQO_regime", "MQO")]:
        f, x = mi.loc[("fixo", mod)], mi.loc[("expansivo", mod)]
        linhas.append(rf"{nome} & {fmt(f.r2_oos_sem_regime_vs_media)} & {fmt(f.r2_oos_vs_media)} & "
                      rf"{fmt(x.r2_oos_vs_media)} & {fmt(f.dm_p_bilateral_vs_sem_regime, 2)} \\")
    nota = (r"$D_t = 1$ se $\text{VH}_{t-29:t} > 37{,}3\%$ a.a. (corte fixo, estimado na primeira janela); "
            r"no corte expansivo, reestimado em cada origem. Painel A: MQO com as 16 variáveis, $D_t$ e as 16 "
            r"interações $D_t X_t$; estatística de Wald $\chi^2(q)$ com erro-padrão HAC de 31 defasagens. "
            r"Painel B: modelos com as interações, mesmas origens da Tabela~\ref{tab:prev-resultados}; DM: Diebold--Mariano bilateral "
            r"contra o mesmo modelo sem regime.")
    with open(os.path.join(dir_tab, "tab_regime.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tabela("Regime de volatilidade e previsão do BVRP.", "tab:prev-regime", "llrrr",
                        r"Corte & $H_0$ & $\chi^2$ ($q$) & $p$ & $R^2$ dentro", linhas, nota))

    # ---------------- Figura (T5) ----------------
    paineis = ["Ridge", "floresta_aleatoria", "proxy_retrospectiva", "persistencia_viavel"]
    fig, axes = plt.subplots(len(paineis), 1, figsize=(10, 2.5 * len(paineis)), sharex=True, sharey=True)
    for ax, m in zip(axes, paineis):
        r2 = met.loc[("expansiva", m), "r2_oos_vs_media_historica"]
        r2_txt = f"{r2:.3f}".replace(".", ",").replace("-", "−")
        ax.plot(w.index, y, color=COR_REAL, linewidth=1.1, label=r"$\mathrm{BVRP}$")
        ax.plot(w.index, w[m], color=COR_PREV, linewidth=1.0, linestyle="--",
                label=rf"{ROTULOS_FIG[m]} ($R^2_{{\mathrm{{fora}}}}$ = {r2_txt})")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylabel("p.p.")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_previsoes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Números citados no texto ----------------
    pos = y > 0
    print(f"Realizado fora da amostra: média {y.mean():.2f}; positivo em {int(pos.sum())} de {len(y)} dias")
    for m in MODELOS:
        print(f"  {m}: % negativas {100 * (w[m] < 0).mean():.1f}; média nos dias positivos {w.loc[pos, m].mean():.2f}; "
              f"dp das previsões {w[m].std():.2f}")
    print(f"Ridge α mais frequente {a_r:g} ({n_r}); LASSO α {a_l:.3f} ({n_l}); LASSO zera tudo em {n_lasso_zero} de 33")
    print(f"Floresta prof. 1: {n_d_rf}; XGBoost taxa 0,01: {n_lr}, prof. 1: {n_d_xgb}, 50 árvores: {n_n_xgb}")
    print(f"Treino: {he.n_treino.min()} a {he.n_treino.max()} obs.")
    print(f"LASSO × MQO: dif. máx. {lvm.loc['expansiva', 'max_dif_abs']:.1f}; corr. {lvm.loc['expansiva', 'corr']:.2f}")
    print(f"DM MQO × Ridge (expansiva): p = {float(dm[(dm.janela == 'expansiva') & (dm.modelo_1 == 'MQO') & (dm.modelo_2 == 'Ridge')].dm_p_bilateral.iloc[0]):.4f}")
    for j in ("expansiva", "movel"):
        s = mcs.loc[j].sort_values("ordem_eliminacao")
        print(f"MCS ({j}, T_max, bloco {MCS_BLOCO}): " + "; ".join(f"{m} {p:.4f}" for m, p in s.p_mcs.items()))
    print(f"Placebo, medianas: {pla.r2_placebo_mediana.min():.4f} a {pla.r2_placebo_mediana.max():.4f}")
    print(f"Gravado em {dir_tab} e {dir_fig}")


if __name__ == "__main__":
    main()
