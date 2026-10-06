"""
publicar_cap7.py — tabelas e figuras do Cap. 7 (BVRP previsto e retornos: modelo não linear), sem reestimar nada

Lê as saídas do T11 (outputs/T11, data/previsoes_bvrp_T11.csv) e das
descritivas do capítulo (outputs/C7, gravadas por code/descritivas_C7.py) e
grava, na notação da seção 14 de docs/pendencias_T1.md:
  tables/retornos_nao_linear/tab_previsao_formas.tex   previsão do BVRP: forma direta × f(X) − IV
  tables/retornos_nao_linear/tab_principal.tex         β, HAC e bootstrap por h (floresta, f(X) − IV)
  tables/retornos_nao_linear/tab_robustez.tex          β e p HAC dos três regressores de robustez
  tables/retornos_nao_linear/tab_conjunto.tex          teste conjunto nos seis horizontes (F2)
  tables/retornos_nao_linear/tab_regime.tex            teste F por horizonte com o regime (F1)
  tables/retornos_nao_linear/tab_arvore_retorno.tex    floresta direto no retorno
  tables/retornos_nao_linear/tab_retorno_regime.tex    retorno futuro médio por regime
  tables/retornos_nao_linear/tab_sinal.tex             BVRP e variação da IV pelo sinal do retorno
  figs/retornos_nao_linear/fig_betas.png               β por h, quatro regressores, IC HAC
  figs/retornos_nao_linear/fig_importancia.png         importância por impureza e por permutação
  figs/retornos_nao_linear/fig_cortes.png              limiares escolhidos pelas árvores nas VH
  figs/retornos_nao_linear/fig_dependencia_parcial.png dependência parcial do BVRP previsto na VH de 30 dias
β em pontos percentuais de retorno por p.p. de BVRP previsto (β × 100).

Conferência (sai com erro se algo não bater; tolerância 1e-9): recalcula, com
as funções do T11 e do avaliacao_utils (só MQO e médias), o R² fora da amostra
e o Clark–West das previsões gravadas (exceto a referência "média da VH − IV",
cuja média histórica da VH não está gravada), a regressão principal, o F1, o
resumo do bootstrap, o F2 (de bootstrap_betas_T11.csv) e o HAC empilhado, e
confere com os CSVs de outputs/T11.

Uso:  python code/publicar_cap7.py [--raiz DIR]
      --raiz: onde gravar tables/ e figs/ (padrão: a raiz do repositório)
"""
import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import previsao_bvrp_T5 as P  # noqa: E402
import retorno_bvrp_T11 as T11  # noqa: E402
from avaliacao_utils import clark_west, r2_fora_da_amostra  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = T11.HS
TOL = 1e-9
AZUL, LARANJA, CINZA = T11.AZUL, T11.LARANJA, T11.CINZA
PRINCIPAL = T11.PRINCIPAL
REGS = ["floresta_rv_menos_iv", "floresta_direta", "Ridge_rv_menos_iv", "Ridge_direto"]
NOME_REG = {"floresta_rv_menos_iv": r"Floresta, $\hat f(X_t) - \text{IV}_t$ (principal)",
            "floresta_direta": "Floresta, direta",
            "Ridge_rv_menos_iv": r"Ridge, $\hat f(X_t) - \text{IV}_t$",
            "Ridge_direto": "Ridge, direta"}
NOME_FIG = {"floresta_rv_menos_iv": r"Floresta, $\hat f(X_t)-\mathrm{IV}_t$ (principal)",
            "floresta_direta": "Floresta, direta", "Ridge_rv_menos_iv": r"Ridge, $\hat f(X_t)-\mathrm{IV}_t$",
            "Ridge_direto": "Ridge, direta"}
FORMA_FIG = {"rv_menos_iv": r"Floresta, $\hat f(X_t)-\mathrm{IV}_t$", "direta": "Floresta, direta"}
MODELOS_PREV = [("Ridge_direto", "Ridge, direta"), ("floresta_direta", "Floresta, direta"),
                ("floresta_rv_menos_iv", r"Floresta, $\hat f(X_t) - \text{IV}_t$ (principal)"),
                ("floresta_rv_menos_iv_com_regime", r"Floresta, $\hat f(X_t) - \text{IV}_t$, com $D_t$"),
                ("Ridge_rv_menos_iv", r"Ridge, $\hat f(X_t) - \text{IV}_t$"),
                ("media_rv_menos_iv", r"Média histórica da VH $-\ \text{IV}_t$")]
# símbolos das variáveis nas figuras (mathtext), como no dicionário do Cap. 5
SIMBOLO = {"vh_1d": r"$\mathrm{VH}_{t:t}$", "vh_5d": r"$\mathrm{VH}_{t-4:t}$", "vh_30d": r"$\mathrm{VH}_{t-29:t}$",
           "vh_60d": r"$\mathrm{VH}_{t-59:t}$", "vh_90d": r"$\mathrm{VH}_{t-89:t}$", "iv_30d": r"$\mathrm{IV}_t$",
           "d_iv_1d": r"$\Delta_1\mathrm{IV}_t$", "d_iv_5d": r"$\Delta_5\mathrm{IV}_t$",
           "iv_menos_ma5d": r"$\mathrm{IV}_t-\overline{\mathrm{IV}}_{t-4:t}$",
           "iv_menos_ma30d": r"$\mathrm{IV}_t-\overline{\mathrm{IV}}_{t-29:t}$", "ret_1d": r"$r_t$",
           "ret_acum_5d": r"$r_{t-4:t}$", "ret_acum_30d": r"$r_{t-29:t}$",
           "log_close_ma30d": r"$\ln(P_t/\overline{P}_{t-29:t})$", "vrp_30d": r"$\mathrm{BVRP}^{\mathrm{proxy}}_t$",
           "d_vrp_1d": r"$\Delta_1\mathrm{BVRP}^{\mathrm{proxy}}_t$", "bvrp_realizado_defasado": r"$\mathrm{BVRP}_{t-30}$"}
fmt_eixo = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{d}f}".replace(".", "{,}")
    return s.replace("-", "$-$") if v < 0 else s


def fmt_p(p):
    return r"$<$0{,}001" if p < 0.001 else fmt(p, 3)


def tabela(legenda, rotulo, colunas, cabecalho, linhas, nota, tamanho=r"\small"):
    corpo = [r"\begin{table}[H]", r"\centering", tamanho, rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}",
             rf"\begin{{tabular}}{{{colunas}}}", r"\toprule", cabecalho + r" \\", r"\midrule"]
    corpo += linhas
    corpo += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{0.95\linewidth}}{{\footnotesize\textit{{Nota}}: {nota}}}", r"\end{table}", ""]
    return "\n".join(corpo)


def gravar(caminho, texto):
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto)


def conferir(cond, msg):
    if not cond:
        raise SystemExit(f"CONFERÊNCIA FALHOU: {msg}")


def conferir_df(novo, gravado, chaves, colunas, nome):
    m = novo.merge(gravado, on=chaves, suffixes=("_n", "_g"))
    conferir(len(m) == len(gravado) == len(novo), f"{nome}: linhas diferentes")
    for c in colunas:
        a, b = m[f"{c}_n"].to_numpy(float), m[f"{c}_g"].to_numpy(float)
        ok = np.isclose(a, b, rtol=0, atol=TOL) | (np.isnan(a) & np.isnan(b))
        conferir(ok.all(), f"{nome}, coluna {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=ROOT)
    args = ap.parse_args()
    dir_tab = os.path.join(args.raiz, "tables", "retornos_nao_linear")
    dir_fig = os.path.join(args.raiz, "figs", "retornos_nao_linear")
    os.makedirs(dir_tab, exist_ok=True)
    os.makedirs(dir_fig, exist_ok=True)
    t11, c7 = os.path.join(ROOT, "outputs", "T11"), os.path.join(ROOT, "outputs", "C7")

    met = pd.read_csv(os.path.join(t11, "metricas_previsao_T11.csv")).set_index(["janela", "modelo"])
    dm = pd.read_csv(os.path.join(t11, "diebold_mariano_T11.csv"))
    pr = pd.read_csv(os.path.join(t11, "principal_T11.csv"))
    rb = pd.read_csv(os.path.join(t11, "bootstrap_resumo_T11.csv"))
    f1 = pd.read_csv(os.path.join(t11, "f1_regime_T11.csv"))
    tc = pd.read_csv(os.path.join(t11, "teste_conjunto_T11.csv"))
    th = pd.read_csv(os.path.join(t11, "teste_conjunto_hac_T11.csv"))
    boot = pd.read_csv(os.path.join(t11, "bootstrap_betas_T11.csv")).rename(columns={str(h): h for h in HS})
    imp = pd.read_csv(os.path.join(t11, "importancia_T11.csv"))
    ag = pd.read_csv(os.path.join(t11, "cortes_T11.csv"))
    cvh = pd.read_csv(os.path.join(t11, "cortes_vh30d_por_origem_T11.csv"))
    pdp = pd.read_csv(os.path.join(t11, "dependencia_parcial_T11.csv"), parse_dates=["origem"])
    ar = pd.read_csv(os.path.join(t11, "arvore_retorno_T11.csv")).set_index("h")
    rr = pd.read_csv(os.path.join(c7, "retorno_por_regime_C7.csv"))
    sn = pd.read_csv(os.path.join(c7, "sinal_retorno_C7.csv")).set_index("variavel")
    ri = pd.read_csv(os.path.join(c7, "resposta_iv_C7.csv")).set_index("coef")
    reg9 = pd.read_csv(os.path.join(ROOT, "data", "regimes_T9.csv"), parse_dates=["date"])

    def p_dm(a, b, janela="expansiva"):
        r = dm[(dm.janela == janela) & (((dm.modelo_1 == a) & (dm.modelo_2 == b))
                                        | ((dm.modelo_1 == b) & (dm.modelo_2 == a)))]
        conferir(len(r) == 1, f"DM {a} × {b}")
        return float(r.dm_p_bilateral.iloc[0])

    # ---------------- Conferência ----------------
    warnings.filterwarnings("ignore")
    df, _, prev_t5, _, _, _ = T11.carregar()
    prev_t11 = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T11.csv"), parse_dates=["date", "origem"])
    prev_t5_todas = pd.read_csv(os.path.join(ROOT, "data", "previsoes_bvrp_T5.csv"), parse_dates=["date", "origem"])
    alvo = df.set_index("date")[P.ALVO]
    rot = {"floresta_aleatoria": "floresta_rv_menos_iv", "Ridge": "Ridge_rv_menos_iv",
           "floresta_aleatoria_com_regime": "floresta_rv_menos_iv_com_regime"}
    for janela in ("expansiva", "movel"):
        w5 = prev_t5_todas[prev_t5_todas.janela == janela].pivot(index="date", columns="modelo", values="previsao").sort_index()
        w11 = prev_t11[prev_t11.janela == janela].pivot(index="date", columns="rotulo", values="previsao").sort_index()
        conferir(len(w5) == len(w11) == 987 and (w5.index == w11.index).all(), f"datas {janela}")
        y, ref = alvo.loc[w5.index].to_numpy(), w5["media_historica"]
        cols = {"floresta_direta": w5["floresta_aleatoria"], "Ridge_direto": w5["Ridge"],
                **{rot[c]: w11[c] for c in w11.columns}}
        for m, p in cols.items():
            conferir(abs(r2_fora_da_amostra(y, p, ref) - met.loc[(janela, m), "r2_oos_vs_media_historica"]) < TOL,
                     f"R² {janela} {m}")
            conferir(abs(clark_west(y, p, ref, h=T11.H_ALVO)["cw_p_unilateral"]
                         - met.loc[(janela, m), "cw_p_unilateral_vs_media_historica"]) < TOL, f"CW {janela} {m}")
    regs = T11.regressores(prev_t11, prev_t5)
    pr_n = T11.principal(df, regs)
    conferir_df(pr_n, pr, ["regressor", "h"], ["beta", "ep_hac", "p_hac", "r2"], "principal")
    conferir_df(T11.regime_f1(df, regs), f1, ["regressor", "corte", "h"],
                ["b_bvrp_previsto", "b_interacao", "p_3coef", "p_2coef"], "F1")
    conferir_df(T11.resumo_bootstrap(boot, pr), rb, ["regressor", "variante", "bloco", "h"],
                ["razao_media_beta", "ep_corrigido", "p_boot"], "resumo do bootstrap")
    est = lambda c: pr[pr.regressor == c["regressor"]].sort_values("h").beta.to_numpy()  # noqa: E731
    g, _, _ = T11.conjunto_de(boot, est, ["regressor", "variante", "bloco"])
    conferir_df(g, tc, ["regressor", "variante", "bloco"], ["W", "p_wald", "p_max_t", "num_condicao_corr"], "F2")
    for reg, p in regs.items():
        R = df.set_index("date").loc[p.index, [f"ret_fut_{h}d" for h in HS]].to_numpy(float)
        w = T11.wald_hac_empilhado(p.to_numpy(), R, max(HS) + 1)
        conferir(abs(w["p_wald"] - th.set_index("regressor").loc[reg, "p_wald"]) < TOL, f"HAC empilhado {reg}")
    print("Conferência: R², Clark–West, regressões, F1, bootstrap, F2 e HAC empilhado batem com outputs/T11.")

    # ---------------- Previsão do BVRP nas duas formas ----------------
    linhas = []
    for m, nome in MODELOS_PREV:
        e = met.loc[("expansiva", m)]
        mov = fmt(met.loc[("movel", m), "r2_oos_vs_media_historica"]) if ("movel", m) in met.index else "--"
        linhas.append(rf"{nome} & {fmt(e.r2_oos_vs_media_historica)} & {mov} & "
                      rf"{fmt_p(e.cw_p_unilateral_vs_media_historica)} & {fmt(e.pct_negativas, 0)}\% & "
                      rf"{fmt(e.media_prev_dias_realizado_pos, 1)} \\")
    nota = (r"Alvo: $\text{BVRP}_t = \text{VH}_{t+1:t+30} - \text{IV}_t$; 987 previsões, de 21/06/2023 a "
            r"03/03/2026. Direta: o modelo prevê o BVRP. $\hat f(X_t) - \text{IV}_t$: o modelo prevê "
            r"$\text{VH}_{t+1:t+30}$, e a IV de $t$ é subtraída. $D_t$: indicadora do regime de alta volatilidade "
            r"(corte fixo) como variável a mais. Média histórica da VH $-\ \text{IV}_t$: média de "
            r"$\text{VH}_{t+1:t+30}$ nos alvos conhecidos na origem, menos a IV de $t$. $R^2_{\text{fora}}$ "
            r"contra a média histórica do BVRP, nas janelas expansiva e móvel (730 dias); Clark--West unilateral "
            r"(expansiva), com HAC de 31 defasagens. Última coluna: média do BVRP previsto nas datas em que o "
            r"BVRP realizado foi positivo, em p.p.")
    gravar(os.path.join(dir_tab, "tab_previsao_formas.tex"),
           tabela(r"Previsão do BVRP: forma direta e forma $\hat f(X_t) - \text{IV}_t$.", "tab:retnl-previsao",
                  "lrrrrr", r"Modelo & $R^2_{\text{fora}}$ & $R^2_{\text{fora}}$ (móvel) & $p$ (CW) & "
                  r"Negativas & BVRP $> 0$", linhas, nota))

    # ---------------- Tabela principal ----------------
    e = pr[pr.regressor == PRINCIPAL].set_index("h")
    b = rb[(rb.regressor == PRINCIPAL) & (rb.variante == "dois_niveis") & (rb.bloco == 60)].set_index("h")
    linhas = [rf"{h} & {fmt(100 * e.loc[h, 'beta'])} & {fmt(100 * e.loc[h, 'ep_hac'])} & {fmt_p(e.loc[h, 'p_hac'])} & "
              rf"{fmt(e.loc[h, 'r2'], 4)} & {fmt(b.loc[h, 'razao_media_beta'], 2)} & "
              rf"{fmt_p(b.loc[h, 'p_boot']) if np.isfinite(b.loc[h, 'p_boot']) else '--'} \\" for h in HS]
    cab = (r" & & \multicolumn{3}{c}{HAC} & \multicolumn{2}{c}{Bootstrap em dois níveis} \\ "
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}" "\n"
           r"$h$ & $\hat\beta_h$ & EP & $p$ & $R^2$ & $\lambda$ & $p$")
    nota = (r"$R_{t+h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, com "
            r"$\widehat{\text{BVRP}}_t = \hat f(X_t) - \text{IV}_t$ e $\hat f$ a floresta aleatória para "
            r"$\text{VH}_{t+1:t+30}$ (janela expansiva); 987 datas. $\hat\beta_h$ e EP em pontos percentuais de "
            r"retorno por ponto percentual de BVRP previsto. HAC: Newey--West com $h+1$ defasagens. Bootstrap: 999 "
            r"reamostragens em blocos de 60 dias; $\lambda$ é a razão entre a média dos $\beta^*$ e $\hat\beta_h$, "
            r"e o valor-$p$ usa o EP do bootstrap dividido por $\lambda$; com $\lambda \le 0$, a correção não é "
            r"definida (--).")
    gravar(os.path.join(dir_tab, "tab_principal.tex"),
           tabela("Retorno futuro sobre o BVRP previsto pela floresta: estimativas e inferência por horizonte.",
                  "tab:retnl-principal", "rrrrrrr", cab, linhas, nota))

    # ---------------- Robustez: outros regressores ----------------
    linhas = []
    for reg in REGS[1:]:
        e = pr[pr.regressor == reg].set_index("h")
        b = rb[(rb.regressor == reg) & (rb.variante == "dois_niveis") & (rb.bloco == 60)].set_index("h")
        linhas.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{NOME_REG[reg]}}}}} \\")
        linhas.append(r"\quad $\hat\beta_h$ & " + " & ".join(fmt(100 * e.loc[h, "beta"]) for h in HS) + r" \\")
        linhas.append(r"\quad $p$ HAC & " + " & ".join(fmt_p(e.loc[h, "p_hac"]) for h in HS) + r" \\")
        linhas.append(r"\quad $p$ bootstrap & " + " & ".join(
            fmt_p(b.loc[h, "p_boot"]) if np.isfinite(b.loc[h, "p_boot"]) else "--" for h in HS) + r" \\")
    nota = (r"Mesma especificação da Tabela~\ref{tab:retnl-principal}, com o BVRP previsto por outros modelos. "
            r"Forma direta: previsões do Capítulo~\ref{chap:previsao_bvrp}. Bootstrap em dois níveis, 999 "
            r"reamostragens em blocos de 60 dias, as mesmas em todos os regressores; -- quando $\lambda \le 0$.")
    gravar(os.path.join(dir_tab, "tab_robustez.tex"),
           tabela("Testes de robustez: retorno futuro sobre o BVRP previsto por outros modelos.",
                  "tab:retnl-robustez", "l" + "r" * 6, " & " + " & ".join(rf"$h = {h}$" for h in HS), linhas, nota))

    # ---------------- Teste conjunto (F2) ----------------
    linhas = []
    tcx, thx = tc.set_index(["regressor", "variante", "bloco"]), th.set_index("regressor")
    for reg in REGS:
        r, s2 = tcx.loc[(reg, "dois_niveis", 60)], tcx.loc[(reg, "so_segundo_nivel", 60)]
        linhas.append(rf"{NOME_REG[reg]} & {fmt(r.W, 2)} & {fmt_p(r.p_wald)} & {fmt(r.p_max_t, 3)} & "
                      rf"{fmt_p(thx.loc[reg, 'p_wald'])} & {fmt_p(s2.p_wald)} \\")
        if reg == PRINCIPAL:
            for bl in (30, 90):
                r = tcx.loc[(reg, "dois_niveis", bl)]
                linhas.append(rf"\quad blocos de {bl} dias & {fmt(r.W, 2)} & {fmt_p(r.p_wald)} & "
                              rf"{fmt(r.p_max_t, 3)} & & \\")
    nota = (r"$H_0$: $\beta_h = 0$ nos seis horizontes. Wald: $W = \bar\beta^{*\prime}\,\Sigma^{*-1}\,\bar\beta^*$, "
            r"com a média e a covariância dos $\beta^*$ do bootstrap em dois níveis (blocos de 60 dias, salvo "
            r"indicação), $\chi^2(6)$; continua definido quando $\lambda_h \le 0$. max-$|t|$: valor-$p$ global do "
            r"maior $|z_h|$. HAC empilhado: as seis regressões em conjunto, Newey--West com 61 defasagens. As duas "
            r"últimas colunas ignoram a estimação do BVRP previsto.")
    gravar(os.path.join(dir_tab, "tab_conjunto.tex"),
           tabela("Teste conjunto nos seis horizontes.", "tab:retnl-conjunto", "lrrrrr",
                  r"Regressor & $W$ & $p$ (Wald) & $p$ (max-$|t|$) & $p$ (HAC empilhado) & $p$ (só 2º nível)",
                  linhas, nota, tamanho=r"\footnotesize"))

    # ---------------- F1: regime ----------------
    g = f1[f1.regressor == PRINCIPAL].set_index(["corte", "h"])
    linhas = [rf"{h} & {fmt(g.loc[('fixo', h), 'qui2_3coef'], 2)} & {fmt_p(g.loc[('fixo', h), 'p_3coef'])} & "
              rf"{fmt_p(g.loc[('fixo', h), 'p_2coef'])} & {fmt(g.loc[('expansivo', h), 'qui2_3coef'], 2)} & "
              rf"{fmt_p(g.loc[('expansivo', h), 'p_3coef'])} & {fmt_p(g.loc[('expansivo', h), 'p_2coef'])} \\"
              for h in HS]
    n_abaixo = int((f1[["p_3coef"]].to_numpy() < T11.BONFERRONI).sum())
    cab = (r" & \multicolumn{3}{c}{Corte fixo} & \multicolumn{3}{c}{Corte expansivo} \\ "
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
           r"$h$ & $\chi^2(3)$ & $p$ & $p$ (regime) & $\chi^2(3)$ & $p$ & $p$ (regime)")
    nota = (r"$R_{t+h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \delta_h D_t + \gamma_h D_t\,"
            r"\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, com o BVRP previsto da Tabela~\ref{tab:retnl-principal}. "
            r"$\chi^2(3)$ e $p$: Wald de $\beta_h = \delta_h = \gamma_h = 0$; $p$ (regime): Wald de "
            r"$\delta_h = \gamma_h = 0$. HAC com $h+1$ defasagens, que ignora a estimação do BVRP previsto. "
            r"Corte fixo: $37{,}3\%$ a.a.; expansivo: reestimado em cada origem.")
    gravar(os.path.join(dir_tab, "tab_regime.tex"),
           tabela("Teste de significância conjunta por horizonte, com o regime de volatilidade.", "tab:retnl-regime",
                  "rrrrrrr", cab, linhas, nota))

    # ---------------- Floresta direto no retorno ----------------
    linhas = [rf"{h} & {fmt(ar.loc[h, 'r2_oos_vs_media_historica'])} & {fmt_p(ar.loc[h, 'cw_p_unilateral'])} & "
              rf"{fmt_p(ar.loc[h, 'dm_p_bilateral'])} & {fmt(ar.loc[h, 'r2_dentro_amostra_media'], 2)} \\" for h in HS]
    nota = (r"$R_{t+h} = g_h(X_t) + \varepsilon_{t+h}$, com $g_h$ uma floresta aleatória por horizonte, as mesmas "
            r"variáveis, origens e janela expansiva, e embargo de $h$ dias; 987 previsões. $R^2_{\text{fora}}$ "
            r"contra a média histórica do retorno; Clark--West unilateral e Diebold--Mariano bilateral contra a "
            r"média histórica, com HAC de $h+1$ defasagens. $R^2$ dentro: média nas 33 origens.")
    gravar(os.path.join(dir_tab, "tab_arvore_retorno.tex"),
           tabela("Floresta aleatória direto no retorno futuro.", "tab:retnl-arvore-retorno", "rrrrr",
                  r"$h$ & $R^2_{\text{fora}}$ & $p$ (CW) & $p$ (DM) & $R^2$ dentro", linhas, nota))

    # ---------------- Retorno futuro por regime ----------------
    linhas = []
    for defin, titulo, b0, b1 in (("corte_fixo_37_3", r"Corte fixo ($37{,}3\%$ a.a.)", "Baixa", "Alta"),
                                  ("quartil_superior_expansivo", r"Quartil superior em janela expansiva",
                                   "Demais", "Extrema")):
        g = rr[rr.definicao == defin].set_index("h")
        n0, n1 = int(g.n_baixa.iloc[0]), int(g.n_alta.iloc[0])
        linhas.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{titulo}}}: {b0}, $N = {n0}$; {b1}, $N = {n1}$}} \\")
        for h in HS:
            r = g.loc[h]
            linhas.append(rf"{h} & {fmt(r.media_baixa, 2)} & {fmt(r.media_alta, 2)} & "
                          rf"{fmt(r.diferenca_alta_menos_baixa, 2)} & {fmt_p(r.p_hac)} \\")
        if defin == "corte_fixo_37_3":
            linhas.append(r"\addlinespace")
    nota = (r"Retorno simples médio de $t$ a $t+h$, em \%. Diferença: regime de alta (ou volatilidade extrema) "
            r"menos o outro grupo, estimada pela regressão do retorno numa indicadora do regime, com HAC de $h+1$ "
            r"defasagens. Corte fixo: amostra de modelagem, 23/04/2021 a 03/03/2026. Quartil superior: "
            r"$\text{VH}_{t-29:t}$ acima do terceiro quartil da própria série com dados até $t$, a partir de "
            r"730 observações (22/04/2023 a 03/03/2026).")
    gravar(os.path.join(dir_tab, "tab_retorno_regime.tex"),
           tabela("Retorno futuro médio por regime de volatilidade.", "tab:retnl-retorno-regime", "rrrrr",
                  r"$h$ & Baixa / demais & Alta / extrema & Diferença & $p$", linhas, nota))

    # ---------------- Sinal do retorno ----------------
    b, d = sn.loc["alvo_bvrp_30d_fut"], sn.loc["d_iv_1d"]
    linhas = [rf"$\text{{BVRP}}_t$ (p.p.) & {fmt(b.media_pos, 2)} & {fmt(b.media_neg, 2)} & "
              rf"{fmt(b.diferenca_neg_menos_pos, 2)} & {fmt_p(b.p_hac)} \\",
              rf"$\Delta_1\text{{IV}}_t$ (p.p.) & {fmt(d.media_pos, 2)} & {fmt(d.media_neg, 2)} & "
              rf"{fmt(d.diferenca_neg_menos_pos, 2)} & {fmt_p(d.p_hac)} \\",
              r"\addlinespace",
              rf"\multicolumn{{5}}{{l}}{{$\Delta_1\text{{IV}}_t$ sobre $r_t^+$ e $r_t^-$: "
              rf"$\hat b^+ = {fmt(ri.loc['r_pos', 'estimativa'], 3)}$ ($p = {fmt_p(ri.loc['r_pos', 'p_hac'])}$); "
              rf"$\hat b^- = {fmt(ri.loc['r_neg', 'estimativa'], 3)}$ ($p = {fmt_p(ri.loc['r_neg', 'p_hac'])}$); "
              rf"$b^+ + b^- = 0$: $p = {fmt_p(ri.loc['simetria_b_pos_mais_b_neg', 'p_hac'])}$}} \\"]
    nota = (rf"Amostra de modelagem; {int(b.n_pos)} dias com $r_t > 0$ e {int(b.n_neg)} com $r_t < 0$. "
            r"Diferença: dias de queda menos dias de alta, pela regressão numa indicadora de $r_t < 0$, com HAC "
            r"de 31 defasagens para o BVRP (alvos de 30 dias sobrepostos) e de 2 para $\Delta_1\text{IV}_t$. "
            r"Última linha: $\Delta_1\text{IV}_t = a + b^+ r_t^+ + b^- r_t^- + u_t$, com "
            r"$r_t^+ = \max(r_t, 0)$ e $r_t^- = \min(r_t, 0)$ em \%, HAC de 2 defasagens; a simetria é "
            r"$b^+ = -b^-$.")
    gravar(os.path.join(dir_tab, "tab_sinal.tex"),
           tabela("BVRP e variação da IV pelo sinal do retorno do dia.", "tab:retnl-sinal", "lrrrr",
                  r"Variável & $r_t > 0$ & $r_t < 0$ & Diferença & $p$", linhas, nota))

    # ---------------- Figura: β por h ----------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(HS))
    estilos = {"floresta_rv_menos_iv": (AZUL, "o", -0.24), "floresta_direta": (LARANJA, "s", -0.08),
               "Ridge_rv_menos_iv": (CINZA, "^", 0.08), "Ridge_direto": ("#555555", "D", 0.24)}
    for reg in REGS:
        e = pr[pr.regressor == reg].set_index("h").loc[HS]
        cor, mk, dx = estilos[reg]
        ax.errorbar(x + dx, 100 * e.beta, yerr=1.96 * 100 * e.ep_hac, fmt=mk, color=cor, capsize=3,
                    markersize=5, label=NOME_FIG[reg])
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xticks(x, [str(h) for h in HS])
    ax.set_xlabel("Horizonte $h$ (dias)")
    ax.set_ylabel(r"$\hat\beta_h$ (p.p. de retorno por p.p. de BVRP previsto)")
    ax.yaxis.set_major_formatter(fmt_eixo)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_betas.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figura: importância ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, forma in enumerate(["rv_menos_iv", "direta"]):
        g = imp[imp.forma == forma].sort_values("mdi_media")
        rotulos = [SIMBOLO[v] for v in g.variavel]
        for ax, col, tit in ((axes[i, 0], "mdi_media", "redução de impureza (média das 33 origens)"),
                             (axes[i, 1], "perm_delta_mse_media", r"permutação fora da amostra ($\Delta$ EQM)")):
            ax.barh(rotulos, g[col], color=[AZUL if v >= 0 else LARANJA for v in g[col]], height=0.6)
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_title(f"{FORMA_FIG[forma]}: {tit}", fontsize=9)
            ax.xaxis.set_major_formatter(fmt_eixo)
            ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_importancia.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figura: limiares nas VH ----------------
    vols = ["vh_30d", "vh_60d", "vh_90d"]
    oos = reg9[reg9.date >= pd.Timestamp("2023-06-21")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for i, forma in enumerate(["rv_menos_iv", "direta"]):
        for jj, var in enumerate(vols):
            ax = axes[i, jj]
            g = ag[(ag.forma == forma) & (ag.variavel == var)]
            ax.bar(g.faixa + 0.5, g.ganho, width=0.9, color=AZUL)
            if var == "vh_30d":
                ax.axvline(T11.CORTE_T9, color="black", linewidth=1, linestyle=":", label="corte fixo (37,3)")
                ax.axvspan(oos.corte_expansivo_vigente.min(), oos.corte_expansivo_vigente.max(), color=LARANJA,
                           alpha=0.15, label="faixa do corte expansivo")
                ax.legend(fontsize=7)
            ax.set_title(f"{FORMA_FIG[forma]}: {SIMBOLO[var]}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(fmt_eixo)
            ax.yaxis.set_major_formatter(fmt_eixo)
        axes[i, 0].set_ylabel("redução de impureza\n(soma nas origens)")
    for ax in axes[-1]:
        ax.set_xlabel("limiar (% a.a.)")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_cortes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figura: dependência parcial na VH de 30 dias ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ultima = pdp.origem.max()
    for ax, forma in zip(axes, ["rv_menos_iv", "direta"]):
        g = pdp[(pdp.forma == forma) & (pdp.variavel == "vh_30d") & (pdp.versao == "coerente")]
        for _, gg in g.groupby("origem"):
            ax.plot(gg.valor, gg.bvrp_previsto, color=CINZA, linewidth=0.6, alpha=0.6)
        u = g[g.origem == ultima]
        ax.plot(u.valor, u.bvrp_previsto, color=AZUL, linewidth=2, label="última origem")
        ax.axvline(T11.CORTE_T9, color="black", linewidth=1, linestyle=":", label="corte fixo (37,3)")
        ax.set_xlabel(r"$\mathrm{VH}_{t-29:t}$ (% a.a.)")
        ax.set_ylabel("BVRP previsto (p.p.)")
        ax.set_title(FORMA_FIG[forma], fontsize=9)
        ax.xaxis.set_major_formatter(fmt_eixo)
        ax.yaxis.set_major_formatter(fmt_eixo)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_dependencia_parcial.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Números citados no texto ----------------
    e = pr[pr.regressor == PRINCIPAL]
    print(f"Principal: β (p.p.) {100 * e.beta.min():.3f} a {100 * e.beta.max():.3f}; "
          f"p HAC {e.p_hac.min():.2f} a {e.p_hac.max():.2f}; R² máx {e.r2.max():.4f}")
    b = rb[(rb.regressor == PRINCIPAL) & (rb.variante == "dois_niveis")]
    print(f"λ do principal: {b.razao_media_beta.min():.2f} a {b.razao_media_beta.max():.2f}; "
          f"λ ≤ 0 em {int((b.razao_media_beta <= 0).sum())} de {len(b)} células (blocos 30/60/90)")
    for reg in REGS:
        r = tcx.loc[(reg, "dois_niveis", 60)]
        print(f"F2 {reg}: W = {r.W:.2f}, p = {r.p_wald:.3f}; max-|t| {r.p_max_t:.3f}; "
              f"HAC {thx.loc[reg, 'p_wald']:.3f}; só 2º {tcx.loc[(reg, 'so_segundo_nivel', 60)].p_wald:.3f}")
    print(f"F2 principal blocos 30/90: {tcx.loc[(PRINCIPAL, 'dois_niveis', 30)].p_wald:.3f} / "
          f"{tcx.loc[(PRINCIPAL, 'dois_niveis', 90)].p_wald:.3f}")
    g = f1[f1.regressor == PRINCIPAL]
    for corte in ("fixo", "expansivo"):
        gg = g[g.corte == corte]
        print(f"F1 principal {corte}: p 3 coef {gg.p_3coef.min():.2f}–{gg.p_3coef.max():.2f}; "
              f"p 2 coef {gg.p_2coef.min():.2f}–{gg.p_2coef.max():.2f}")
    print(f"F1: {n_abaixo} de {len(f1)} células com p (3 coef.) < 0,0083:")
    print(f1[f1.p_3coef < T11.BONFERRONI][["regressor", "corte", "h", "p_3coef"]].to_string(index=False))
    print("DM forma direta × f(X) − IV: floresta p = %.3f; Ridge p = %.3f; com regime × sem: p = %.3f; "
          "f(X) − IV × média histórica: p = %.3f" % (p_dm("floresta_direta", "floresta_rv_menos_iv"),
                                                    p_dm("Ridge_direto", "Ridge_rv_menos_iv"),
                                                    p_dm("floresta_rv_menos_iv", "floresta_rv_menos_iv_com_regime"),
                                                    p_dm("floresta_rv_menos_iv", "media_historica")))
    for forma in ("rv_menos_iv", "direta"):
        g = imp[imp.forma == forma].set_index("variavel")
        print(f"Importância {forma}: maior impureza {g.mdi_media.idxmax()} ({g.mdi_media.max():.3f}); "
              f"maior permutação {g.perm_delta_mse_media.idxmax()} ({g.perm_delta_mse_media.max():.2f}); "
              f"vh_90d: impureza {g.loc['vh_90d', 'mdi_media']:.3f}, permutação {g.loc['vh_90d', 'perm_delta_mse_media']:.2f}; "
              f"vh_30d: impureza {g.loc['vh_30d', 'mdi_media']:.3f}, permutação {g.loc['vh_30d', 'perm_delta_mse_media']:.2f}")
        c = cvh[cvh.forma == forma]
        print(f"  vh_30d: mediana ponderada do limiar {c.limiar_mediana_ponderada.median():.1f} "
              f"({c.limiar_mediana_ponderada.min():.1f}–{c.limiar_mediana_ponderada.max():.1f}); "
              f"fração do ganho entre 35 e 40: {c.fracao_ganho_vh_30d_entre_35_e_40.median():.3f}; "
              f"origens com vh_30d: {int((c.fracao_ganho_vh_30d > 0).sum())}")
        for var in vols:
            gg = ag[(ag.forma == forma) & (ag.variavel == var)].sort_values("ganho", ascending=False)
            print(f"  {var}: faixas de maior ganho {', '.join(f'{v:g}' for v in gg.faixa.head(4))}")
    print(f"Faixa do corte expansivo fora da amostra: {oos.corte_expansivo_vigente.min():.1f}–"
          f"{oos.corte_expansivo_vigente.max():.1f}")
    print(f"Leitura B: R² {ar.r2_oos_vs_media_historica.min():.3f}–{ar.r2_oos_vs_media_historica.max():.3f}; "
          f"CW p {ar.cw_p_unilateral.min():.2f}–{ar.cw_p_unilateral.max():.2f}; DM h=60 p {ar.loc[60, 'dm_p_bilateral']:.3f}")
    print(f"Gravado em {dir_tab} e {dir_fig}")


if __name__ == "__main__":
    main()
