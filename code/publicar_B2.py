"""
publicar_B2.py — tabelas do B2 (Caps. 6 e 7 e apêndice) a partir de outputs/B2/, sem estimar nada

Lê as cópias das saídas agregadas dos diagnósticos do B2 em outputs/B2/ e grava:
  tables/retornos_linear/tab_wald_direto.tex       Wald direto nos 16 previsores, por h (Cap. 6)
  tables/retornos_linear/tab_um_fator.tex          BVRP previsto como fator único (Cap. 6)
  tables/retornos_linear/tab_oos_retorno.tex       MQO e Ridge fora da amostra (Cap. 6)
  tables/retornos_nao_linear/tab_oos_floresta.tex  floresta aleatória direto no retorno (Cap. 7)
  tables/apendice_b2/tab_tamanho_poder.tex         tamanho e poder por simulação (apêndice)
  tables/apendice_b2/tab_regime.tex                regime de volatilidade (apêndice)
  tables/apendice_b2/tab_corr_e_u.tex              corr(erro de previsão do RBVRP, inovação do retorno)
Não toca nos .tex dos capítulos nem nas tabelas do publicar_cap6.py e do publicar_cap7.py.

Conferência (sai com erro na primeira falha):
  1. Procedência (outputs/B2/procedencia_B2.csv, uma linha por arquivo): SHA-256 do conteúdo
     normalizado para LF de cada cópia = o registrado = o da origem em outputs/diagnostico_b2/
     = o do blob da origem no commit registrado; blob do script no commit = o registrado;
     nenhum arquivo de outputs/B2/ fora da procedência.
  2. Cópia × origem: mesmas linhas e colunas; colunas numéricas com tolerância de 1e-9 (NaN só
     casa com NaN); demais colunas iguais.
  3. Coerência interna, sem estimar: p da χ² = chi2.sf(W, q) e p do F; Holm refeito
     (wald_utils.holm); W do um fator = t²; EP = |β|/√W; p do min-p da linha
     sensibilidade_g* = o da variante; p da χ² do regime = chi2.sf(W, gl); corr(e, u) do
     passo 2 = a do passo 1; IC HAC = corr ± 1,96·EP; rejeições em [0, 1]; placebo da
     floresta com mediana <= 0,01. Tolerância de 1e-9.
Cada número impresso vem dos DataFrames já conferidos.

Uso:  python code/publicar_B2.py [--raiz DIR]
      --raiz: onde gravar tables/ (padrão: a raiz do repositório)
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wald_utils import holm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B2 = os.path.join(ROOT, "outputs", "B2")
TOL = 1e-9
HS = [1, 5, 10, 20, 30, 60]
MODELOS = {"Ridge": "Ridge", "floresta_aleatoria": "Floresta aleatória",
           "Ridge_rv_menos_iv": r"Ridge, $\hat f(X_t)-\text{IV}_t$",
           "floresta_rv_menos_iv": r"Floresta, $\hat f(X_t)-\text{IV}_t$",
           "MQO": "MQO", "LASSO": "LASSO", "XGBoost": "XGBoost", "media_historica": "Média histórica",
           "proxy_retrospectiva": "HBVRP", "persistencia_viavel": "RBVRP defasado"}
CASOS = {"q16_completa": "16 previsores, 1.776 datas", "q16_987": "16 previsores, 987 datas",
         "q1_ridge": r"$\widehat{\text{BVRP}}$ do Ridge", "q1_floresta": r"$\widehat{\text{BVRP}}$ da floresta"}


# ---------------------------------------------------------------------------
# Formatação (como em publicar_cap6.py)
# ---------------------------------------------------------------------------
def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{d}f}"
    if float(s) == 0:                                    # arredondado para zero: sem sinal ("0{,}0000")
        s = s.lstrip("-")
    s = s.replace(".", "{,}")
    return s.replace("-", "$-$")                          # fora do modo matemático: nunca dentro de $...$


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return r"$<$0{,}001" if p < 0.001 else fmt(p, 3)


def tabela(legenda, rotulo, colunas, cabecalho, linhas, nota, tamanho=r"\small"):
    corpo = [r"\begin{table}[H]", r"\centering", tamanho, rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}",
             rf"\begin{{tabular}}{{{colunas}}}", r"\toprule", cabecalho + r" \\", r"\midrule"]
    corpo += linhas
    corpo += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{0.95\linewidth}}{{\footnotesize\textit{{Nota}}: {nota}}}", r"\end{table}", ""]
    return "\n".join(corpo)


def gravar(caminho, texto):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto)


def falha(msg):
    raise SystemExit(f"CONFERÊNCIA FALHOU: {msg}")


def checar(cond, msg):
    if not bool(np.all(cond)):
        falha(msg)


# ---------------------------------------------------------------------------
# 1. Procedência e 2. cópia × origem
# ---------------------------------------------------------------------------
def sha_lf(dados):
    return hashlib.sha256(dados.replace(b"\r\n", b"\n")).hexdigest()


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        falha(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def mesmo_df(a, b, nome):
    checar(a.shape == b.shape and list(a.columns) == list(b.columns), f"{nome}: formato ≠ origem")
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            checar(np.isclose(x, y, rtol=0, atol=TOL) | (np.isnan(x) & np.isnan(y)), f"{nome}, coluna {c}")
        else:
            checar(a[c].equals(b[c]), f"{nome}, coluna {c}")                         # NaN = NaN


def carregar_e_conferir_procedencia():
    proc = pd.read_csv(os.path.join(B2, "procedencia_B2.csv"), dtype=str, keep_default_na=False)
    listados = {os.path.basename(a) for a in proc.arquivo}
    presentes = {f for f in os.listdir(B2) if f != "procedencia_B2.csv"}
    checar(listados == presentes, f"outputs/B2 × procedência: {sorted(listados ^ presentes)}")
    dados = {}
    for _, r in proc.iterrows():
        checar(r.sha256_regra == "conteudo_lf", f"{r.arquivo}: regra do SHA")
        copia = open(os.path.join(ROOT, r.arquivo), "rb").read()
        checar(sha_lf(copia) == r.sha256_conteudo_lf, f"{r.arquivo}: SHA-256 ≠ procedência")
        checar(copia.replace(b"\r\n", b"\n").count(b"\n") == int(r.linhas), f"{r.arquivo}: nº de linhas")
        origem = os.path.join(ROOT, r.origem)
        checar(os.path.isfile(origem), f"{r.origem}: origem ausente")
        checar(sha_lf(open(origem, "rb").read()) == r.sha256_conteudo_lf, f"{r.origem}: SHA-256 da origem ≠ cópia")
        checar(git("rev-parse", f"{r.commit_origem}:{r.origem}").decode().strip() == r.git_blob_origem,
               f"{r.origem}: blob ≠ procedência")
        checar(sha_lf(git("show", f"{r.commit_origem}:{r.origem}")) == r.sha256_conteudo_lf,
               f"{r.origem}: SHA-256 do blob ≠ cópia")
        checar(git("rev-parse", f"{r.commit_origem}:{r.script}").decode().strip() == r.git_blob_script,
               f"{r.script}: blob do script ≠ procedência")
        nome = os.path.basename(r.arquivo).removesuffix("_B2.csv")
        d = pd.read_csv(os.path.join(ROOT, r.arquivo))
        mesmo_df(d, pd.read_csv(origem), nome)
        dados[nome] = d
    print(f"Procedência: {len(proc)} arquivos; SHA-256 (LF), blobs e cópia × origem conferidos.")
    return dados


# ---------------------------------------------------------------------------
# 3. Coerência interna
# ---------------------------------------------------------------------------
def conferir_coerencia(d):
    for nome in ("holm", "testes_por_horizonte"):
        t = d[nome]
        c = t[t.teste == "chi2"]
        checar(np.isclose(c.p, stats.chi2.sf(c.W, c.q), rtol=0, atol=TOL), f"{nome}: p da χ²")
        f = t[t.teste == "F"]
        checar(np.isclose(f.p, stats.f.sf(f.W / f.q, f.q, f.n - f.q - 1), rtol=0, atol=TOL), f"{nome}: p do F")
    h = d["holm"]
    for _, g in h.groupby(["desenho", "L_regra", "teste"]):
        g = g.sort_values("h")
        checar(len(g) == len(HS), "holm: seis horizontes por grupo")
        checar(np.isclose(g.p_holm, holm(g.p.to_numpy()), rtol=0, atol=TOL), "holm: p ajustado")
    uf = d["um_fator"]
    w = h[h.teste == "chi2"].set_index(["desenho", "h", "L_regra"]).W
    for _, r in uf.iterrows():
        W = w[(r.desenho, r.h, r.L_regra)]
        checar(abs(r.t_hac ** 2 - W) <= TOL * max(1.0, W), f"um fator {r.desenho} h = {r.h}: t² ≠ W")
        checar(abs(r.ep_hac - abs(r.beta) / np.sqrt(W)) <= TOL, f"um fator {r.desenho} h = {r.h}: EP")
    eh = d["entre_horizontes"]
    sens = eh[eh.variante == "sensibilidade_g*"]
    for _, r in sens.iterrows():
        for g in (120, 180, 360):
            v = eh[(eh.desenho == r.desenho) & (eh.L_regra == r.L_regra) & (eh.g == g)].p_min_p
            checar(len(v) == 1 and abs(v.iloc[0] - r[f"p_min_p_g{g}"]) <= TOL, f"entre horizontes: g* = {g}")
    rg = d["regime"]
    c = rg[rg.teste == "chi2"]
    checar(np.isclose(c.p, stats.chi2.sf(c.W, c.gl), rtol=0, atol=TOL), "regime: p da χ²")
    ic, p1 = d["ic_corr_e_u"], d["corr_erro_inovacao"]
    a2 = p1[(p1.y == "R_log") & (p1.inovacao == "A2_resid_MQO_16_previsores")].set_index("modelo")
    for _, r in ic.iterrows():
        e = a2.loc[r.modelo]
        checar(abs(r["corr"] - e["corr"]) <= TOL and abs(r["corr_passo1"] - e["corr"]) <= TOL, f"corr(e, u) {r.modelo}")
        checar(abs(r.ic_hac_inf - (e["corr"] - 1.96 * e.ep_hac)) <= TOL
               and abs(r.ic_hac_sup - (e["corr"] + 1.96 * e.ep_hac)) <= TOL, f"IC HAC {r.modelo}")
    for nome in ("tamanho_poder", "tamanho_4b"):
        t = d[nome]
        for col in ("rejeicao_10pct", "rejeicao_5pct", "rejeicao_1pct"):
            checar(t[col].between(0, 1).all(), f"{nome}: {col} fora de [0, 1]")
    checar(d["placebo_floresta"].passa.all() & (d["placebo_floresta"]["median"] <= 0.01).all(), "placebo da floresta")
    print("Coerência interna: χ², F, Holm, um fator, min-p, regime, corr(e, u), tamanho e placebo conferidos.")


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def p_de(h_df, desenho, h, L_regra, teste, col="p"):
    v = h_df[(h_df.desenho == desenho) & (h_df.h == h) & (h_df.L_regra == L_regra) & (h_df.teste == teste)][col]
    checar(len(v) == 1, f"{desenho} h = {h} {L_regra} {teste}: linha ausente ou repetida")
    return float(v.iloc[0])


def linha_entre(eh, desenho, ncol):
    r = eh[(eh.desenho == desenho) & (eh.L_regra == "h+1") & (eh.variante == "principal")].iloc[0]
    hs = " e ".join(str(r.h_do_min_p).split(","))
    return (rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Entre horizontes (rotação, $g^* = 180$)}}: mín.\ $p$ em "
            rf"$h = {hs}$, $p$ = {fmt(r.p_min_p, 3)}; máx.\ $W$ em $h = {int(r.h_do_max_W)}$, "
            rf"$p$ = {fmt(r.p_max_W, 3)}}} \\")


def tab_wald_direto(d, raiz):
    h, eh = d["holm"], d["entre_horizontes"]
    linhas = []
    for painel, des, n in (("A", "q16_completa", "1.776"), ("B", "q16_987", "987")):
        linhas.append(rf"\multicolumn{{8}}{{l}}{{\textit{{Painel {painel}: {n} datas, $L = h+1$}}}} \\")
        for hh in HS:
            linhas.append(
                rf"{hh} & {fmt(p_de(h, des, hh, 'h+1', 'chi2', 'W'), 2)} & {fmt_p(p_de(h, des, hh, 'h+1', 'chi2'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'rotacao'))} & {fmt_p(p_de(h, des, hh, 'h+1', 'bootstrap'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'chi2', 'p_holm'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'rotacao', 'p_holm'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'bootstrap', 'p_holm'))} \\")
        linhas.append(linha_entre(eh, des, 8))
        linhas.append(r"\midrule")
    linhas.append(r"\multicolumn{8}{l}{\textit{Painel C: sensibilidade, $L = 2h+1$ (rotação e bootstrap)}} \\")
    linhas.append(r"$h$ & \multicolumn{2}{c}{1.776 datas: rotação} & \multicolumn{2}{c}{bootstrap} & "
                  r"\multicolumn{2}{c}{987 datas: rotação} & bootstrap \\")
    for hh in HS:
        linhas.append(rf"{hh} & \multicolumn{{2}}{{c}}{{{fmt_p(p_de(h, 'q16_completa', hh, '2h+1', 'rotacao'))}}} & "
                      rf"\multicolumn{{2}}{{c}}{{{fmt_p(p_de(h, 'q16_completa', hh, '2h+1', 'bootstrap'))}}} & "
                      rf"\multicolumn{{2}}{{c}}{{{fmt_p(p_de(h, 'q16_987', hh, '2h+1', 'rotacao'))}}} & "
                      rf"{fmt_p(p_de(h, 'q16_987', hh, '2h+1', 'bootstrap'))} \\")
    cab = (r" & & & & & \multicolumn{3}{c}{Holm} \\ \cmidrule(lr){6-8}" "\n"
           r"$h$ & $W$ & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap")
    nota = (r"$R^{\log}_{t,h} = \alpha_h + X_t'\beta_h + \varepsilon_{t+h}$, com $R^{\log}_{t,h}$ o log-retorno de "
            r"$t$ a $t+h$ em pontos percentuais e $X_t$ as 16 variáveis explicativas: as 17 do "
            r"Capítulo~\ref{chap:previsao_bvrp} menos a VH de 30 dias, colinear com o HBVRP e a IV. "
            r"$H_0$: as 16 inclinações são nulas. $W$: Wald com erro-padrão HAC de Newey--West (núcleo de "
            r"Bartlett, $h+1$ defasagens, sem correção de amostra pequena); $p_{\chi^2}$ pela $\chi^2(16)$. "
            r"Rotação: $W$ recalculado com $y$ deslocado circularmente em $k$ dias, $g_h \le k \le n - g_h$, "
            r"$g_h = \max(90, 3h)$. Bootstrap: pares $(X_t, y_t)$ em blocos circulares de $\max(30, 2h)$ dias, "
            r"1.999 reamostragens, centrado em $\hat\beta$. Holm sobre os seis horizontes. Entre horizontes: "
            r"mínimo dos valores-$p$ da rotação com o mesmo deslocamento em todos os $h$. Com $2h+1$ "
            r"defasagens, o $p_{\chi^2}$ fica muito pequeno e não é confiável: nas simulações de tamanho "
            r"(Apêndice), o teste $\chi^2$ rejeita muito acima do nível nominal; por isso o painel C traz só a "
            r"rotação e o bootstrap.")
    gravar(os.path.join(raiz, "tables", "retornos_linear", "tab_wald_direto.tex"),
           tabela("Retorno de $h$ dias sobre os 16 previsores: teste de Wald de significância conjunta, por horizonte.",
                  "tab:retlin-wald-direto", "rrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_um_fator(d, raiz):
    h, uf, eh = d["holm"], d["um_fator"], d["entre_horizontes"]
    linhas = []
    for painel, des, nome in (("A", "q1_ridge", "Ridge"),
                              ("B", "q1_floresta", r"floresta aleatória, $\hat f(X_t)-\text{IV}_t$")):
        linhas.append(rf"\multicolumn{{9}}{{l}}{{\textit{{Painel {painel}: {nome}}}}} \\")
        for hh in HS:
            u = uf[(uf.desenho == des) & (uf.h == hh) & (uf.L_regra == "h+1")].iloc[0]
            linhas.append(
                rf"{hh} & {fmt(u.beta)} & {fmt(u.ep_hac)} & {fmt_p(p_de(h, des, hh, 'h+1', 'chi2'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'rotacao'))} & {fmt_p(p_de(h, des, hh, 'h+1', 'bootstrap'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'rotacao', 'p_holm'))} & "
                rf"{fmt_p(p_de(h, des, hh, 'h+1', 'bootstrap', 'p_holm'))} & {fmt(u.r2, 4)} \\")
        linhas.append(linha_entre(eh, des, 9))
        if painel == "A":
            linhas.append(r"\midrule")
    sens = eh[(eh.variante == "sensibilidade_g*") & (eh.L_regra == "h+1")].set_index("desenho")
    sg = "; ".join(rf"{n}: {fmt(sens.loc[des, 'p_min_p_g120'], 3)}, {fmt(sens.loc[des, 'p_min_p_g180'], 3)} e "
                   rf"{fmt(sens.loc[des, 'p_min_p_g360'], 3)}" for des, n in (("q1_ridge", "Ridge"),
                                                                             ("q1_floresta", "floresta")))
    cab = (r" & & & & & & \multicolumn{2}{c}{Holm} & \\ \cmidrule(lr){7-8}" "\n"
           r"$h$ & $\hat\beta_h$ & EP HAC & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap & $p$ rotação & "
           r"$p$ bootstrap & $R^2$")
    nota = (r"$R^{\log}_{t,h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, nas 987 datas fora "
            r"da amostra; $\hat\beta_h$ e EP em pontos percentuais de retorno por ponto percentual de "
            r"$\widehat{\text{BVRP}}$. EP HAC de Newey--West com $h+1$ defasagens; $p_{\chi^2}$ pela $\chi^2(1)$. "
            r"Rotação e bootstrap como na Tabela~\ref{tab:retlin-wald-direto}; Holm sobre os seis horizontes. "
            r"O $\widehat{\text{BVRP}}$ é tratado como dado: os testes não incorporam a estimação do primeiro "
            r"estágio (ver o apêndice do regressor gerado). Entre horizontes, $p$ do mínimo dos valores-$p$ da "
            rf"rotação com $g^* = 120$, 180 e 360: {sg}.")
    gravar(os.path.join(raiz, "tables", "retornos_linear", "tab_um_fator.tex"),
           tabela(r"Retorno de $h$ dias sobre o $\widehat{\text{BVRP}}$ como fator único.",
                  "tab:retlin-um-fator", "rrrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_oos_retorno(d, raiz):
    m = d["oos_metricas"].set_index(["h", "modelo"])
    pg = d["oos_pontas_grade"]
    linhas = []
    for hh in HS:
        a, b, c = m.loc[(hh, "MQO")], m.loc[(hh, "Ridge")], m.loc[(hh, "media_historica")]
        linhas.append(rf"{hh} & {fmt(a.r2_oos_vs_media, 4)} & {fmt_p(a.cw_p_unilateral)} & {fmt_p(a.dm_p_bilateral)} & "
                      rf"{fmt(b.r2_oos_vs_media, 4)} & {fmt_p(b.cw_p_unilateral)} & {fmt_p(b.dm_p_bilateral)} & "
                      rf"{fmt(c.r2_oos_vs_zero, 4)} \\")
    lasso = m.xs("LASSO", level="modelo")
    checar(np.isclose(lasso.r2_oos_vs_media, 0, atol=TOL).all(), "LASSO ≠ média histórica")
    checar((pg[pg.modelo == "LASSO"].coef_nao_nulos_max == 0).all(), "LASSO com coeficientes não nulos")
    hm = {mod: (holm(m.xs(mod, level="modelo").loc[HS, "cw_p_unilateral"].to_numpy()).min(),
                holm(m.xs(mod, level="modelo").loc[HS, "dm_p_bilateral"].to_numpy()).min()) for mod in ("MQO", "Ridge")}
    rid = pg[pg.modelo == "Ridge"].assign(h=lambda x: x.y.str.removeprefix("R_log_").astype(int)).set_index("h")
    topo = "; ".join(rf"{int(rid.loc[hh, 'n_no_maior'])} de {int(rid.loc[hh, 'n_origens'])} em $h = {hh}$" for hh in HS)
    cab = (r" & \multicolumn{3}{c}{MQO} & \multicolumn{3}{c}{Ridge} & Média \\ "
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
           r"$h$ & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM & "
           r"$R^2$ contra zero")
    nota = (r"Previsão de $R^{\log}_{t,h}$ com as 16 variáveis explicativas nas 987 datas fora da amostra, 33 "
            r"origens em janela expansiva, treino e validação cruzada com embargo $s \le t-h$. "
            r"$R^2_{\text{fora}}$ contra o modelo de referência, a média histórica dos alvos conhecidos na "
            r"origem. CW: Clark--West, unilateral; DM: Diebold--Mariano, bilateral; ambos com HAC de $h+1$ "
            r"defasagens. Nenhum valor-$p$ da tabela está ajustado por comparações múltiplas; o menor valor-$p$ "
            r"ajustado por Holm sobre os seis horizontes, contra a média histórica, é "
            rf"{fmt_p(hm['MQO'][0])} (CW) e {fmt_p(hm['MQO'][1])} (DM) no MQO e "
            rf"{fmt_p(hm['Ridge'][0])} (CW) e {fmt_p(hm['Ridge'][1])} (DM) no Ridge. "
            r"A última coluna é o $R^2_{\text{fora}}$ da média histórica contra a previsão zero. O "
            r"LASSO zera todos os coeficientes em todas as origens e coincide com a média histórica. O "
            rf"$\alpha$ do Ridge fica no maior valor da grade ($10^5$) em {topo}.")
    gravar(os.path.join(raiz, "tables", "retornos_linear", "tab_oos_retorno.tex"),
           tabela("Previsão do retorno de $h$ dias fora da amostra: MQO e Ridge contra a média histórica.",
                  "tab:retlin-oos", "rrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_oos_floresta(d, raiz):
    f = d["floresta_retorno_metricas"].set_index("h")
    pl = d["placebo_floresta"].set_index("h")
    hf = (holm(f.loc[HS, "cw_p_unilateral"].to_numpy()).min(), holm(f.loc[HS, "dm_p_bilateral"].to_numpy()).min())
    linhas =[rf"{hh} & {fmt(f.loc[hh, 'r2_oos_vs_media'], 4)} & {fmt(f.loc[hh, 'r2_oos_vs_zero'], 4)} & "
              rf"{fmt_p(f.loc[hh, 'cw_p_unilateral'])} & {fmt_p(f.loc[hh, 'dm_p_bilateral'])} & "
              rf"{fmt_p(f.loc[hh, 'dm_vs_Ridge_p_bilateral'])} & {fmt(f.loc[hh, 'r2_dentro_amostra_media'], 3)} & "
              rf"{fmt(pl.loc[hh, 'median'], 4)} \\" for hh in HS]
    cab = (r" & \multicolumn{2}{c}{$R^2_{\text{fora}}$} & & \multicolumn{2}{c}{$p$ DM} & & \\ "
           r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}" "\n"
           r"$h$ & contra a média & contra zero & $p$ CW & contra a média & contra o Ridge & $R^2$ dentro & placebo")
    nota = (r"$R^{\log}_{t,h} = g_h(X_t) + \varepsilon_{t+h}$, com uma floresta aleatória por horizonte e as 17 "
            r"variáveis explicativas em nível; 500 árvores, hiperparâmetros por validação cruzada com embargo "
            r"$h$, 33 origens em janela expansiva, 987 datas fora da amostra. Modelo de referência: a média "
            r"histórica. CW: Clark--West, unilateral; DM: Diebold--Mariano, bilateral; ambos com HAC de $h+1$ "
            r"defasagens; DM contra o Ridge da Tabela~\ref{tab:retlin-oos}. Nenhum valor-$p$ da tabela está "
            r"ajustado por comparações múltiplas; o menor valor-$p$ ajustado por Holm sobre os seis horizontes, "
            rf"contra a média histórica, é {fmt_p(hf[0])} (CW) e {fmt_p(hf[1])} (DM). $R^2$ dentro: média, "
            r"entre as origens, do $R^2$ no treino. Placebo: mediana do $R^2_{\text{fora}}$ em 10 permutações do "
            r"alvo dentro do treino.")
    gravar(os.path.join(raiz, "tables", "retornos_nao_linear", "tab_oos_floresta.tex"),
           tabela("Floresta aleatória direto no retorno de $h$ dias, fora da amostra.",
                  "tab:retnl-oos-floresta", "rrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_tamanho_poder(d, raiz):
    t3, t4 = d["tamanho_poder"], d["tamanho_4b"]
    nome3 = {"Wald HAC (χ²)": "chi2", "Monte Carlo por rotação (g = 90)": "rot", "Wald bootstrap (bloco 60, B = 199)": "boot"}
    nome4 = {"Wald HAC (χ²)": "chi2", "rotação": "rot", "Wald bootstrap": "boot"}
    caso3 = {"q16_completa": "q16_completa", "q16_987": "q16_987", "q1_R_log": "q1_ridge"}

    def tam(caso, proc, h, teste):
        des = "um_fator" if caso in ("q1_ridge", "q1_floresta") else "q16"
        if h == 30:
            c = {v: k for k, v in caso3.items()}.get(caso)
            if c is None:
                return np.nan
            r = t3[(t3.caso == c) & (t3.desenho == des) & (t3.processo == proc) & (t3.rho == 0) & (t3.L == 31)
                   & (t3.teste.map(nome3) == teste)]
        else:
            r = t4[(t4.caso == caso) & (t4.desenho == des) & (t4.processo == proc) & (t4.h == h)
                   & (t4.L_regra == "h+1") & (t4.teste.map(nome4) == teste)]
        checar(len(r) == 1, f"tamanho {caso} {proc} h = {h} {teste}")
        return float(r.rejeicao_5pct.iloc[0])

    linhas = [r"\multicolumn{11}{l}{\textit{Painel A: tamanho a 5\% ($\rho = 0$), $L = h+1$}} \\"]
    for caso in CASOS:
        for proc in ("S1", "S3"):
            v = [tam(caso, proc, hh, t) for hh in (1, 30, 60) for t in ("chi2", "rot", "boot")]
            linhas.append(rf"{CASOS[caso]} & {proc} & " + " & ".join(fmt(x, 3) for x in v) + r" \\")
    linhas += [r"\midrule", r"\multicolumn{11}{l}{\textit{Painel B: poder a 5\% em $h = 30$, ruído S1}} \\",
               r"Caso & Teste & $\rho = 0{,}5\%$ & 1\% & 2\% & 5\% & & & & & \\"]
    rhos = (0.005, 0.01, 0.02, 0.05)
    for c, des, rot in (("q16_completa", "q16", CASOS["q16_completa"]), ("q16_987", "q16", CASOS["q16_987"]),
                        ("q1_R_log", "um_fator", CASOS["q1_ridge"])):
        for teste, col, rotulo in (("chi2", "poder_ajustado_5pct", r"$\chi^2$ ajustado ao tamanho"),
                                   ("rot", "rejeicao_5pct", "rotação"), ("boot", "rejeicao_5pct", "bootstrap")):
            vals = []
            for rho in rhos:
                r = t3[(t3.caso == c) & (t3.desenho == des) & (t3.processo == "S1") & np.isclose(t3.rho, rho)
                       & (t3.L == 31) & (t3.teste.map(nome3) == teste)]
                checar(len(r) <= 1, f"poder {c} ρ = {rho} {teste}")
                vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
            linhas.append(rf"{rot} & {rotulo} & " + " & ".join(fmt(x, 3) for x in vals) + r" & & & & & \\")
    cab = (r" & & \multicolumn{3}{c}{$h = 1$} & \multicolumn{3}{c}{$h = 30$} & \multicolumn{3}{c}{$h = 60$} \\ "
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n"
           r"Caso & Ruído & $\chi^2$ & rotação & bootstrap & $\chi^2$ & rotação & bootstrap & $\chi^2$ & rotação & bootstrap")
    nota = (r"Simulação com as variáveis explicativas reais fixas e $y$ = constante + ruído (painel A) ou + sinal "
            r"com $R^2$ populacional $\rho$ (painel B). Ruído: soma de $h$ log-retornos diários reais, S1 sorteados "
            r"com reposição, S3 em blocos circulares de $\max(60, 2h)$ dias. 1.000 réplicas com $\rho = 0$ e 500 "
            r"com $\rho > 0$. Testes como nas Tabelas~\ref{tab:retlin-wald-direto} e~\ref{tab:retlin-um-fator}; "
            r"o bootstrap usa 199 reamostragens. Poder do $\chi^2$ ajustado ao tamanho: valor crítico igual ao "
            r"quantil de 95\% de $W$ com $\rho = 0$. Em $h = 30$, o $\widehat{\text{BVRP}}$ da floresta não foi "
            r"simulado (--). O ruído S2 (rotação sorteada da série real) fica só nos arquivos de saída: suas "
            r"réplicas são rotações da mesma série e não são independentes, de modo que o tamanho sob S2 não é "
            r"estimado com a precisão nominal. O poder foi simulado só com o ruído S1; o S3 entra só com "
            r"$\rho = 0$. Os ruídos não reproduzem a dependência de Stambaugh, entre a inovação do retorno e as "
            r"inovações futuras das variáveis explicativas.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_tamanho_poder.tex"),
           tabela("Tamanho e poder dos testes por simulação.", "tab:apb2-tamanho-poder", "llrrrrrrrrr",
                  cab, linhas, nota, tamanho=r"\scriptsize"))


def tab_regime(d, raiz):
    rg = d["regime"]
    rg = rg[(rg.variante == "principal") & (rg.L_regra == "h+1")]
    hips = (("interacao", r"$\gamma = 0$"), ("regime_e_interacao", r"$\delta = \gamma = 0$"),
            ("todas", r"$\beta = \delta = \gamma = 0$"))
    linhas = []
    for corte, nome in (("fixo", "corte fixo"), ("expansivo", "corte reestimado em cada origem")):
        for hh in (30, 60):
            g = rg[(rg.corte == corte) & (rg.h == hh)]
            r0 = g.iloc[0]
            linhas.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{nome.capitalize()}, $h = {hh}$}}: $\hat\beta$ = "
                          rf"{fmt(r0.b_bvrp_previsto, 2)}, $\hat\delta$ = {fmt(r0.b_regime_alta, 2)}, $\hat\gamma$ = "
                          rf"{fmt(r0.b_interacao, 2)}; {fmt(r0.pct_alta, 1)}\% das datas no regime de alta}} \\")
            for hip, rot in hips:
                x = g[g.hipotese == hip].set_index("teste")
                linhas.append(rf"{rot} & {int(x.loc['chi2', 'gl'])} & {fmt(x.loc['chi2', 'W'], 2)} & "
                              rf"{fmt_p(x.loc['chi2', 'p'])} & {fmt_p(x.loc['rotacao_freedman_lane', 'p'])} & "
                              rf"{fmt_p(x.loc['bootstrap', 'p'])} \\")
    cab = r"$H_0$ & gl & $W$ & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap"
    nota = (r"$R^{\log}_{t,h} = a + \beta\,\widehat{\text{BVRP}}_t + \delta D_t + \gamma D_t\,\widehat{\text{BVRP}}_t "
            r"+ \varepsilon_{t+h}$, com o $\widehat{\text{BVRP}}$ do Ridge, nas 987 datas fora da amostra. $D_t = 1$ "
            r"se $\text{VH}_{t-29:t}$ supera o corte de $37{,}3\%$ a.a. (fixo) ou o corte vigente na origem "
            r"(reestimado). $W$ com HAC de $h+1$ defasagens. Rotação de Freedman--Lane: o resíduo do modelo "
            r"restrito, sem as colunas da hipótese, é deslocado circularmente e somado ao ajustado restrito; em "
            r"$\beta = \delta = \gamma = 0$ equivale à rotação de $y$. $g_h = \max(90, 3h)$. Bootstrap de pares "
            r"em blocos circulares de $\max(30, 2h)$ dias, 1.999 reamostragens, centrado em $\hat\beta$. O "
            r"$\widehat{\text{BVRP}}$ é tratado como dado.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_regime.tex"),
           tabela(r"Retorno de $h$ dias, $\widehat{\text{BVRP}}$ do Ridge e regime de volatilidade.",
                  "tab:apb2-regime", "lrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_corr_e_u(d, raiz):
    p1, ic = d["corr_erro_inovacao"], d["ic_corr_e_u"]
    p1 = p1[p1.y == "R_log"]
    a2 = p1[p1.inovacao == "A2_resid_MQO_16_previsores"].set_index("modelo")
    icp = ic[ic.principal & (ic.versao == "u fixo")].set_index("modelo")
    principais = ["Ridge", "floresta_aleatoria", "Ridge_rv_menos_iv", "floresta_rv_menos_iv"]
    linhas = [r"\multicolumn{7}{l}{\textit{Painel A: inovação A2}} \\"]
    for m in principais:
        e, b = a2.loc[m], icp.loc[m]
        linhas.append(rf"{MODELOS[m]} & {fmt(e['corr'])} & {fmt(e.ep_hac)} & {fmt_p(e.p_hac)} & "
                      rf"[{fmt(b.ic_hac_inf)}; {fmt(b.ic_hac_sup)}] & [{fmt(b.ic_pct_inf)}; {fmt(b.ic_pct_sup)}] & "
                      rf"{fmt(e.corr_sem_sobrep_mediana)} [{fmt(e.corr_sem_sobrep_min)}; {fmt(e.corr_sem_sobrep_max)}] \\")
    inov = ["A1_resid_propria_previsao", "A2_resid_MQO_16_previsores", "A3_menos_media_historica_oos",
            "A4_menos_ridge_direto_oos"]
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel B: correlação por inovação}} \\",
               r"Modelo & A1 & A2 & A3 & A4 & & \\"]
    w = p1.pivot(index="modelo", columns="inovacao", values="corr")
    for m in principais + ["MQO", "LASSO", "XGBoost", "media_historica", "proxy_retrospectiva", "persistencia_viavel"]:
        linhas.append(rf"{MODELOS[m]} & " + " & ".join(fmt(w.loc[m, i]) for i in inov) + r" & & \\")
    cab = r"Modelo & corr & EP HAC & $p$ HAC & IC 95\% HAC & IC 95\% bootstrap & Sem sobreposição"
    nota = (r"$e_t = \text{RBVRP}_t - \widehat{\text{BVRP}}_t$, o erro de previsão do RBVRP de cada modelo, e "
            r"$u_t$ a inovação de $R^{\log}_{t,30}$, nas 987 datas fora da amostra. A2 (principal): resíduo do MQO "
            r"de $R^{\log}_{t,30}$ nas 16 variáveis explicativas; A1: resíduo da regressão de $R^{\log}_{t,30}$ na "
            r"própria previsão; A3: $R^{\log}_{t,30}$ menos a média histórica fora da amostra; A4: $R^{\log}_{t,30}$ "
            r"menos a previsão do Ridge direto fora da amostra. EP HAC com 31 defasagens; IC HAC $=$ corr "
            r"$\pm 1{,}96$ EP. IC bootstrap: percentis de 9.999 reamostragens em blocos circulares de 60 dias, com "
            r"$u$ fixo. Sem sobreposição: mediana [mínimo; máximo] da correlação nas 30 subamostras de uma data a "
            r"cada 30 dias. Média histórica, HBVRP e RBVRP defasado são os modelos de referência do "
            r"Capítulo~\ref{chap:previsao_bvrp}.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_corr_e_u.tex"),
           tabela("Correlação entre o erro de previsão do RBVRP e a inovação do retorno de 30 dias.",
                  "tab:apb2-corr-e-u", "lrrrccc", cab, linhas, nota, tamanho=r"\footnotesize"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=ROOT)
    args = ap.parse_args()
    d = carregar_e_conferir_procedencia()
    conferir_coerencia(d)
    for f in (tab_wald_direto, tab_um_fator, tab_oos_retorno, tab_oos_floresta, tab_tamanho_poder, tab_regime,
              tab_corr_e_u):
        f(d, args.raiz)
        print(f"  {f.__name__}: ok")
    print(f"Tabelas gravadas em {os.path.join(args.raiz, 'tables')}")


if __name__ == "__main__":
    main()
