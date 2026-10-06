"""
publicar_cap6.py — tabelas e figura do Cap. 6 (BVRP previsto e retornos: modelo linear), sem reestimar nada

Lê as saídas do T10 e o teste conjunto do T10 calculado no T11 e grava, na
notação da seção 14 de docs/pendencias_T1.md:
  tables/retornos_linear/tab_principal.tex     β, HAC e bootstrap em dois níveis por h (Ridge)
  tables/retornos_linear/tab_conjunto.tex      teste conjunto nos seis horizontes e max-|t|
  tables/retornos_linear/tab_robustez.tex      valores-p das variantes do bootstrap e da floresta
  tables/retornos_linear/tab_componentes.tex   retorno sobre VH e IV separadas
  tables/retornos_linear/tab_regime.tex        interação com o regime de volatilidade
  figs/retornos_linear/fig_betas.png           β por h com os dois intervalos (Ridge)
β em pontos percentuais de retorno por p.p. de BVRP previsto (β × 100).

Conferência (sai com erro se algo não bater): com as funções do T10, que só
estimam MQO, recalcula a regressão principal, os componentes e o regime a
partir dos dados e das previsões gravadas, e o resumo do bootstrap a partir de
outputs/T10/bootstrap_betas_T10.csv; confere tudo com os CSVs de outputs/T10.

Uso:  python code/publicar_cap6.py [--raiz DIR]
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
import retorno_bvrp_T10 as T10  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = T10.HS
TOL = 1e-9
AZUL, LARANJA = T10.AZUL, T10.LARANJA
PRINCIPAL = ("Ridge_alpha_fixo", 60)
VARIANTES = [(("Ridge_alpha_fixo", 30), "Blocos de 30 dias"), (("Ridge_alpha_fixo", 90), "Blocos de 90 dias"),
             (("Ridge_alpha_reescolhido", 60), r"$\alpha$ reescolhido"),
             (("floresta_hiperparametros_fixos", 60), "Floresta aleatória"),
             (("Ridge_so_segundo_nivel", 60), r"Só o 2º nível (ignora o 1º estágio)")]


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


def conferir_df(novo, gravado, chaves, colunas, nome):
    m = novo.merge(gravado, on=chaves, suffixes=("_n", "_g"))
    if len(m) != len(gravado) or len(m) != len(novo):
        raise SystemExit(f"CONFERÊNCIA FALHOU: {nome}: linhas diferentes")
    for c in colunas:
        a, b = m[f"{c}_n"].to_numpy(float), m[f"{c}_g"].to_numpy(float)
        ok = np.isclose(a, b, rtol=0, atol=TOL) | (np.isnan(a) & np.isnan(b))
        if not ok.all():
            raise SystemExit(f"CONFERÊNCIA FALHOU: {nome}, coluna {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=ROOT)
    args = ap.parse_args()
    dir_tab = os.path.join(args.raiz, "tables", "retornos_linear")
    dir_fig = os.path.join(args.raiz, "figs", "retornos_linear")
    os.makedirs(dir_tab, exist_ok=True)
    os.makedirs(dir_fig, exist_ok=True)
    t10, t11 = os.path.join(ROOT, "outputs", "T10"), os.path.join(ROOT, "outputs", "T11")

    pr = pd.read_csv(os.path.join(t10, "principal_T10.csv"))
    rb = pd.read_csv(os.path.join(t10, "bootstrap_resumo_T10.csv"))
    co = pd.read_csv(os.path.join(t10, "componentes_T10.csv"))
    rg = pd.read_csv(os.path.join(t10, "regime_T10.csv"))
    boot = pd.read_csv(os.path.join(t10, "bootstrap_betas_T10.csv"))
    boot.columns = [int(c) if c.isdigit() else c for c in boot.columns]
    tc = pd.read_csv(os.path.join(t11, "teste_conjunto_T10.csv"))
    th = pd.read_csv(os.path.join(t11, "teste_conjunto_hac_T11.csv")).set_index("regressor")
    mt = pd.read_csv(os.path.join(t11, "max_t_por_horizonte_T10.csv"))

    # ---------------- Conferência ----------------
    warnings.filterwarnings("ignore")
    df, _, prev, _ = T10.carregar()
    conferir_df(T10.principal(df, prev), pr, ["modelo", "h"], ["beta", "ep_hac", "p_hac", "r2"], "principal")
    conferir_df(T10.componentes(df, prev), co, ["h"], ["b_vh", "b_iv", "b_soma", "p_soma", "r2"], "componentes")
    conferir_df(T10.regime(df, prev), rg, ["corte", "h"],
                ["b_bvrp_previsto", "b_interacao", "p_interacao", "p_conjunto"], "regime")
    conferir_df(T10.resumo_bootstrap(boot, pr), rb, ["variante", "bloco", "h"],
                ["razao_media_beta", "ep_corrigido", "p_boot", "ic_inf", "ic_sup"], "resumo do bootstrap")
    print("Conferência: regressões e resumo do bootstrap recalculados batem com outputs/T10.")

    rp = rb[(rb.variante == PRINCIPAL[0]) & (rb.bloco == PRINCIPAL[1])].set_index("h")
    ridge = pr[pr.modelo == "Ridge"].set_index("h")
    flor = pr[pr.modelo == "floresta_aleatoria"].set_index("h")

    # ---------------- Tabela principal ----------------
    linhas = []
    for h in HS:
        r, e = rp.loc[h], ridge.loc[h]
        linhas.append(rf"{h} & {fmt(100 * e.beta)} & {fmt(100 * e.ep_hac)} & {fmt_p(e.p_hac)} & "
                      rf"{fmt(r.razao_media_beta, 2)} & {fmt(100 * r.ep_corrigido)} & {fmt_p(r.p_boot)} & "
                      rf"[{fmt(100 * r.ic_inf)}; {fmt(100 * r.ic_sup)}] & {fmt(e.r2)} \\")
    cab = (r" & & \multicolumn{2}{c}{HAC} & \multicolumn{4}{c}{Bootstrap em dois níveis} & \\ "
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}" "\n"
           r"$h$ & $\hat\beta_h$ & EP & $p$ & $\lambda$ & EP corrigido & $p$ & IC 95\% & $R^2$")
    nota = (r"$R_{t+h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, com o BVRP previsto "
            r"pelo Ridge (janela expansiva); 987 datas, de 21/06/2023 a 03/03/2026. $\hat\beta_h$ e EP em pontos "
            r"percentuais de retorno por ponto percentual de BVRP previsto. HAC: Newey--West com $h+1$ "
            r"defasagens, que ignora a estimação do BVRP previsto. Bootstrap: 999 reamostragens em blocos de "
            r"60 dias; $\lambda$ é a razão entre a média dos $\beta^*$ e $\hat\beta_h$; EP corrigido $=$ "
            r"EP do bootstrap$/\lambda$; o valor-$p$ e o IC usam o EP corrigido. Limite de Bonferroni para "
            r"seis horizontes: $0{,}05/6 \approx 0{,}0083$.")
    gravar(os.path.join(dir_tab, "tab_principal.tex"),
           tabela("Retorno futuro sobre o BVRP previsto: estimativas e inferência por horizonte.",
                  "tab:retlin-principal", "rrrrrrrcr", cab, linhas, nota, tamanho=r"\footnotesize"))

    # ---------------- Teste conjunto ----------------
    nomes = {("Ridge_alpha_fixo", 60): r"Dois níveis, blocos de 60 dias (principal)",
             ("Ridge_alpha_fixo", 30): "Dois níveis, blocos de 30 dias",
             ("Ridge_alpha_fixo", 90): "Dois níveis, blocos de 90 dias",
             ("Ridge_alpha_reescolhido", 60): r"Dois níveis, $\alpha$ reescolhido",
             ("Ridge_so_segundo_nivel", 60): r"Só o 2º nível (ignora o 1º estágio)"}
    linhas = []
    for (var, bl), nome in nomes.items():
        r = tc[(tc.variante == var) & (tc.bloco == bl)].iloc[0]
        linhas.append(rf"{nome} & {fmt(r.W, 2)} & {fmt_p(r.p_wald)} & {fmt(r.p_max_t, 3)} \\")
    hac = th.loc["Ridge_direto"]
    linhas.append(rf"HAC empilhado, {int(hac.maxlags)} defasagens (ignora o 1º estágio) & {fmt(hac.W, 2)} & "
                  rf"{fmt_p(hac.p_wald)} & -- \\")
    linhas += [r"\midrule", r"\multicolumn{4}{l}{\textit{max-$|t|$ por horizonte (principal), valor-$p$ ajustado}} \\"]
    mp = mt[(mt.variante == PRINCIPAL[0]) & (mt.bloco == PRINCIPAL[1])].set_index("h")
    linhas.append(r"\multicolumn{4}{l}{" + "; ".join(
        rf"$h = {h}$: {fmt(mp.loc[h, 'p_max_t'], 3)}" for h in HS) + r"} \\")
    p0 = tc[(tc.variante == PRINCIPAL[0]) & (tc.bloco == PRINCIPAL[1])].iloc[0]
    nota = (r"$H_0$: $\beta_h = 0$ nos seis horizontes. Wald: $W = \bar\beta^{*\prime}\,\Sigma^{*-1}\,"
            r"\bar\beta^*$, com a média e a covariância dos $\beta^*$ do bootstrap, $\chi^2(6)$; equivale ao "
            r"teste na escala corrigida da atenuação. max-$|t|$: o maior $|z_h|$ entre os horizontes, com valor "
            r"crítico tirado da distribuição do máximo nas reamostragens (passo único). HAC empilhado: as seis "
            r"regressões estimadas em conjunto, com Newey--West. Os $\beta^*$ de horizontes vizinhos têm "
            rf"correlação de 0,8 a 0,97, e o número de condição da matriz de correlação é {fmt(p0.num_condicao_corr, 0)}.")
    gravar(os.path.join(dir_tab, "tab_conjunto.tex"),
           tabela("Teste conjunto nos seis horizontes.", "tab:retlin-conjunto", "lrrr",
                  r"Inferência & $W$ & $p$ (Wald) & $p$ (max-$|t|$)", linhas, nota))

    # ---------------- Robustez ----------------
    linhas = []
    for (chave, nome) in VARIANTES:
        g = rb[(rb.variante == chave[0]) & (rb.bloco == chave[1])].set_index("h")
        linhas.append(nome + " & " + " & ".join(fmt_p(g.loc[h, "p_boot"]) for h in HS) + r" \\")
    linhas += [r"\addlinespace",
               r"Floresta aleatória: $\hat\beta_h$ & " + " & ".join(fmt(100 * flor.loc[h, "beta"]) for h in HS) + r" \\",
               r"Floresta aleatória: $p$ HAC & " + " & ".join(fmt_p(flor.loc[h, "p_hac"]) for h in HS) + r" \\"]
    nota = (r"Valores-$p$ do bootstrap em dois níveis na escala corrigida da atenuação (999 reamostragens; "
            r"199 na floresta). Principal: Ridge com o $\alpha$ de cada origem fixo e blocos de 60 dias "
            r"(Tabela~\ref{tab:retlin-principal}). $\alpha$ reescolhido: validação cruzada refeita em cada "
            r"treino reamostrado. Só o 2º nível: previsões originais fixas, sem correção. Floresta: $\hat\beta_h$ "
            r"em p.p. por p.p.")
    gravar(os.path.join(dir_tab, "tab_robustez.tex"),
           tabela("Testes de robustez: valores-$p$ por horizonte.", "tab:retlin-robustez", "l" + "r" * 6,
                  "Variante & " + " & ".join(rf"$h = {h}$" for h in HS), linhas, nota))

    # ---------------- Componentes ----------------
    linhas = []
    for _, r in co.iterrows():
        linhas.append(rf"{int(r.h)} & {fmt(100 * r.b_vh)} & {fmt_p(r.p_vh)} & {fmt(100 * r.b_iv)} & "
                      rf"{fmt_p(r.p_iv)} & {fmt(100 * r.b_soma)} & {fmt_p(r.p_soma)} \\")
    nota = (r"$R_{t+h} = \alpha_h + \beta_{1,h}\,\text{VH}_{t-29:t} + \beta_{2,h}\,\text{IV}_t + "
            r"\varepsilon_{t+h}$, nas mesmas 987 datas; coeficientes em p.p. de retorno por p.p. de volatilidade. "
            r"Valores-$p$ HAC com $h+1$ defasagens; a última coluna testa $\beta_{1,h} + \beta_{2,h} = 0$.")
    gravar(os.path.join(dir_tab, "tab_componentes.tex"),
           tabela("Retorno futuro sobre a volatilidade histórica e a implícita.", "tab:retlin-componentes",
                  "rrrrrrr", r"$h$ & $\hat\beta_{1,h}$ (VH) & $p$ & $\hat\beta_{2,h}$ (IV) & $p$ & "
                  r"$\hat\beta_{1,h} + \hat\beta_{2,h}$ & $p$", linhas, nota))

    # ---------------- Regime ----------------
    linhas = []
    for corte, nome in [("fixo", "Corte fixo"), ("expansivo", "Corte expansivo")]:
        g = rg[rg.corte == corte].set_index("h")
        linhas.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{nome}}} ({fmt(100 - g.pct_alta.iloc[0], 1)}\% das datas "
                      r"no regime de baixa)} \\")
        for h in HS:
            r = g.loc[h]
            linhas.append(rf"{h} & {fmt(100 * r.b_bvrp_previsto)} & {fmt_p(r.p_bvrp_previsto)} & "
                          rf"{fmt(100 * (r.b_bvrp_previsto + r.b_interacao))} & {fmt(100 * r.b_interacao)} & "
                          rf"{fmt_p(r.p_interacao)} & {fmt_p(r.p_conjunto)} \\")
        if corte == "fixo":
            linhas.append(r"\addlinespace")
    nota = (r"$R_{t+h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \delta_h D_t + \gamma_h D_t\,"
            r"\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, com $D_t = 1$ no regime de alta volatilidade. "
            r"Inclinação no regime de baixa: $\hat\beta_h$; no de alta: $\hat\beta_h + \hat\gamma_h$; em p.p. "
            r"por p.p. Valores-$p$ HAC com $h+1$ defasagens; Wald: $\delta_h = \gamma_h = 0$. Corte fixo: "
            r"$37{,}3\%$ a.a., da primeira janela; expansivo: reestimado em cada origem.")
    gravar(os.path.join(dir_tab, "tab_regime.tex"),
           tabela("Retorno futuro, BVRP previsto e regime de volatilidade.", "tab:retlin-regime", "rrrrrrr",
                  r"$h$ & Baixa: $\hat\beta_h$ & $p$ & Alta & $\hat\gamma_h$ & $p$ & $p$ (Wald)", linhas, nota))

    # ---------------- Figura ----------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(HS))
    b = 100 * ridge.loc[HS, "beta"].to_numpy()
    ep_h = 100 * ridge.loc[HS, "ep_hac"].to_numpy()
    lo, hi = 100 * rp.loc[HS, "ic_inf"].to_numpy(), 100 * rp.loc[HS, "ic_sup"].to_numpy()
    ax.errorbar(x - 0.12, b, yerr=1.96 * ep_h, fmt="o", color=AZUL, capsize=4,
                label=r"IC 95%, HAC com $h+1$ defasagens (ignora o 1º estágio)")
    ax.errorbar(x + 0.12, b, yerr=[b - lo, hi - b], fmt="s", color=LARANJA, capsize=4,
                label=r"IC 95%, bootstrap em dois níveis (EP corrigido, EP$/\lambda$)")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xticks(x, [str(h) for h in HS])
    ax.set_xlabel("Horizonte $h$ (dias)")
    ax.set_ylabel(r"$\hat\beta_h$ (p.p. de retorno por p.p. de BVRP previsto)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−")))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_fig, "fig_betas.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Números citados no texto ----------------
    print("Ridge, β (p.p./p.p.), p HAC, λ, p boot:")
    for h in HS:
        print(f"  h={h}: {100 * ridge.loc[h, 'beta']:.3f}  {ridge.loc[h, 'p_hac']:.3f}  "
              f"{rp.loc[h, 'razao_media_beta']:.2f}  {rp.loc[h, 'p_boot']:.3f}  "
              f"EPcorr/EPhac {rp.loc[h, 'razao_ep_corrigido_hac']:.2f}  EPcorr/EPso2 {rp.loc[h, 'razao_ep_corrigido_so2']:.2f}")
    print(f"F2 principal p = {p0.p_wald:.3f}; max-|t| global p = {p0.p_max_t:.3f}; HAC empilhado p = {hac.p_wald:.3f}")
    for (chave, nome) in VARIANTES:
        g = rb[(rb.variante == chave[0]) & (rb.bloco == chave[1])]
        print(f"  {nome}: p boot {g.p_boot.min():.3f}–{g.p_boot.max():.3f}; λ {g.razao_media_beta.min():.2f}–"
              f"{g.razao_media_beta.max():.2f}")
    print(f"Floresta: p HAC {flor.p_hac.min():.2f}–{flor.p_hac.max():.2f}")
    print(f"Gravado em {dir_tab} e {dir_fig}")


if __name__ == "__main__":
    main()
