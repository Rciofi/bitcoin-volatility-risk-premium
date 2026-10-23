"""
publicar_B2.py — tabelas do B2 (Caps. 6 e 7 e apêndice) a partir de outputs/B2/, sem estimar nada

Lê as cópias das saídas agregadas dos diagnósticos do B2 em outputs/B2/ e grava:
  tables/retornos_linear/tab_wald_direto.tex       Wald direto nos 16 previsores, por h (Cap. 6)
  tables/retornos_linear/tab_um_fator.tex          BVRP previsto como fator único (Cap. 6)
  tables/retornos_linear/tab_oos_retorno.tex       MQO e Ridge fora da amostra (Cap. 6)
  tables/retornos_nao_linear/tab_oos_floresta.tex  floresta aleatória direto no retorno (Cap. 7)
  tables/retornos_linear/tab_componentes_wald.tex  VH e IV separadas: Wald β1 = β2 = 0 e β1 + β2 = 0 (Cap. 6)
  tables/apendice_b2/tab_tamanho_poder.tex         tamanho e poder por simulação (apêndice)
  tables/apendice_b2/tab_regime.tex                regime de volatilidade, com α, β, δ, γ e β + γ (apêndice)
  tables/apendice_b2/tab_corr_e_u.tex              corr(erro de previsão do RBVRP, inovação do retorno)
  tables/apendice_b2/tab_ponte_retorno_simples.tex floresta direto no retorno e fator único do Ridge: simples × log
                                                   (passos 6 e 7)
  tables/apendice_b2/tab_atenuacao.tex             regressor gerado: atenuação com o retorno real fixo (passo 7)
  tables/apendice_b2/tab_regressor_gerado.tex      regressor gerado: EP e valores-p, retorno e erro simulados (passo 7)
  tables/apendice_b2/tab_ar.tex                    AR do erro de previsão e do retorno (passo 7)
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
     floresta com mediana <= 0,01; componentes e regime do passo 5: p da χ² = chi2.sf(W, gl), Holm
     refeito e W de cada restrição de uma linha = (estimativa/EP)² dos coeficientes; ponte do passo 6:
     conferência do passo 6 toda verdadeira, linhas do log-retorno = floresta_retorno_metricas,
     n_50pct <= n_80pct <= n_datas e log-retorno = 100·log1p(retorno simples/100). Tolerância de 1e-9.
Cada número impresso vem dos DataFrames já conferidos.

Uso:  python code/publicar_B2.py [--raiz DIR] [--b2 DIR]
      --raiz: onde gravar tables/ (padrão: a raiz do repositório)
      --b2: pasta com as cópias e a procedência (padrão: outputs/B2)
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


def tabela(legenda, rotulo, colunas, cabecalho, linhas, nota, tamanho=r"\small", caber=False, colsep=None):
    """caber: reduz a tabela só se ela passar da largura da linha; colsep: \\tabcolsep em pt."""
    corpo = [r"\begin{table}[H]", r"\centering", tamanho, rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}"]
    if colsep is not None:
        corpo.append(rf"\setlength{{\tabcolsep}}{{{colsep}pt}}")
    if caber:
        corpo.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    corpo += [rf"\begin{{tabular}}{{{colunas}}}", r"\toprule", cabecalho + r" \\", r"\midrule"]
    corpo += linhas
    corpo += [r"\bottomrule", r"\end{tabular}" + ("}" if caber else ""), r"\par\smallskip",
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
        copia = open(os.path.join(B2, os.path.basename(r.arquivo)), "rb").read()
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
        d = pd.read_csv(os.path.join(B2, os.path.basename(r.arquivo)))
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
    for nome in ("componentes_testes", "regime_testes"):
        c = d[nome][d[nome].teste == "chi2"]
        checar(np.isclose(c.p, stats.chi2.sf(c.W, c.gl), rtol=0, atol=TOL), f"{nome}: p da χ²")
    ch = d["componentes_holm"]
    for _, g in ch.groupby(["hipotese", "L_regra", "teste"]):
        g = g.sort_values("h")
        checar(len(g) == len(HS) and np.isclose(g.p_holm, holm(g.p.to_numpy()), rtol=0, atol=TOL).all(),
               "componentes: Holm")
    # W de uma restrição de uma linha = (estimativa / EP)², com o mesmo L
    pares_c = {"soma_zero": ("b1_mais_b2", "ep_b1_mais_b2")}
    pares_r = {"beta_zero": ("beta", "ep_beta"), "gama_zero": ("gama", "ep_gama"),
               "beta_mais_gama_zero": ("beta_mais_gama", "ep_beta_mais_gama")}
    for nome_t, nome_c, chaves, pares in (("componentes_testes", "componentes_coeficientes", ["h", "L_regra"], pares_c),
                                          ("regime_testes", "regime_coeficientes", ["corte", "h", "L_regra"], pares_r)):
        t = d[nome_t][(d[nome_t].teste == "chi2")]
        m = t.merge(d[nome_c], on=chaves)
        for hip, (est, ep) in pares.items():
            x = m[m.hipotese == hip]
            checar(len(x) > 0 and np.allclose(x.W, (x[est] / x[ep]) ** 2, rtol=TOL, atol=0), f"{nome_t}: W de {hip} = (est/EP)²")
    # ponte do passo 6
    checar(d["conferencia_passo6"].bate.all(), "passo 6: conferência")
    pl = d["ponte_dois_retornos"].query("retorno == 'log'").set_index("h").loc[HS]
    fm = d["floresta_retorno_metricas"].set_index("h").loc[HS]
    for a, b in (("r2_fora", "r2_oos_vs_media"), ("p_cw", "cw_p_unilateral"), ("p_dm", "dm_p_bilateral")):
        checar(np.isclose(pl[a], fm[b], rtol=0, atol=TOL), f"ponte: {a} do log ≠ floresta_retorno_metricas")
    cc = d["ponte_concentracao_h60"].dropna(subset=["n_50pct"])
    checar((cc.n_50pct <= cc.n_80pct) & (cc.n_80pct <= cc.n_datas), "ponte: concentração")
    mr = d["ponte_maiores_retornos_h60"]
    checar(np.isclose(mr.log_retorno_pct, 100 * np.log1p(mr.retorno_simples_pct / 100), rtol=0, atol=TOL),
           "ponte: log-retorno ≠ log1p do simples")
    # passo 7: sanidade, β̂ e EP HAC = fator único do passo 4, ponte em log = Cap. 6, Holm
    checar(d["regressor_gerado_sanidade"].bate.all(), "passo 7: sanidade")
    uf = d["um_fator"][d["um_fator"].L_regra == "h+1"]
    at = d["regressor_gerado_atenuacao"]
    for modelo, des in (("ridge", "q1_ridge"), ("floresta", "q1_floresta")):
        u = uf[uf.desenho == des].set_index("h").loc[HS]
        for cel in at[at.modelo == modelo].celula.unique():
            g = at[at.celula == cel].set_index("h").loc[HS]
            checar(np.isclose(g.beta_hat, u.beta, rtol=0, atol=TOL), f"passo 7, {cel}: β̂ ≠ fator único")
            if "ep_hac" in g and g.ep_hac.notna().all():
                checar(np.isclose(g.ep_hac, u.ep_hac, rtol=0, atol=TOL), f"passo 7, {cel}: EP HAC ≠ fator único")
    pl = d["ponte_ridge"][d["ponte_ridge"].retorno == "log"].set_index("h").loc[HS]
    u = uf[uf.desenho == "q1_ridge"].set_index("h").loc[HS]
    checar(np.isclose(pl.beta, u.beta, rtol=0, atol=TOL), "ponte Ridge: β do log ≠ fator único")
    for teste, col in (("rotacao", "p_rotacao"), ("bootstrap", "p_bootstrap")):
        ref = [p_de(d["holm"], "q1_ridge", hh, "h+1", teste) for hh in HS]
        checar(np.isclose(pl[col].to_numpy(), ref, rtol=0, atol=TOL), f"ponte Ridge: p {teste} do log ≠ Cap. 6")
    # passo 8 (#135): sanidade; as linhas originais reproduzem o T5 (R², CW, R² dentro e MCS)
    checar((d["sens_profundidade_sanidade"].dif <= TOL).all(), "passo 8: sanidade")
    met5 = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "metricas_T5.csv")).set_index(["janela", "modelo"])
    mcs5 = pd.read_csv(os.path.join(ROOT, "outputs", "MCS", "mcs_T5.csv"))
    mcs5 = mcs5[(mcs5.bloco == 60) & (mcs5.estatistica == "T_max")].set_index(["janela", "modelo"])
    for x in d["sens_profundidade_resultados"].query("versao == 'original'").itertuples():
        checar(abs(x.mcs_p - mcs5.loc[(x.janela, x.modelo), "p_mcs"]) <= TOL, f"passo 8: MCS {x.janela} {x.modelo}")
        if x.modelo != "media_historica":
            checar(abs(x.r2_fora - met5.loc[(x.janela, x.modelo), "r2_oos_vs_media_historica"]) <= TOL
                   and abs(x.p_cw - met5.loc[(x.janela, x.modelo), "cw_p_unilateral_vs_media_historica"]) <= TOL
                   and abs(x.r2_dentro - met5.loc[(x.janela, x.modelo), "r2_dentro_amostra_media"]) <= TOL,
                   f"passo 8: original {x.janela} {x.modelo} ≠ T5")
    print("Coerência interna: χ², F, Holm, um fator, min-p, regime, corr(e, u), tamanho, placebo, componentes, "
          "regime do passo 5, ponte do passo 6, regressor gerado do passo 7 e profundidade do passo 8 conferidos.")


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

    sigla = {"q16_completa": "P16-M", "q16_987": "P16-F", "q1_ridge": "FU-R", "q1_floresta": "FU-F"}
    linhas = [r"\multicolumn{11}{l}{\textit{Painel A: tamanho a 5\% ($\rho = 0$), $L = h+1$}} \\"]
    for caso in CASOS:
        for proc in ("S1", "S3"):
            v = [tam(caso, proc, hh, t) for hh in (1, 30, 60) for t in ("chi2", "rot", "boot")]
            linhas.append(rf"{sigla[caso]} & {proc} & " + " & ".join(fmt(x, 3) for x in v) + r" \\")
    linhas += [r"\midrule", r"\multicolumn{11}{l}{\textit{Painel B: poder a 5\% em $h = 30$, ruído S1}} \\",
               r"Caso & Teste & $\rho = 0{,}5\%$ & 1\% & 2\% & 5\% & & & & & \\"]
    rhos = (0.005, 0.01, 0.02, 0.05)
    for c, des, rot in (("q16_completa", "q16", sigla["q16_completa"]), ("q16_987", "q16", sigla["q16_987"]),
                        ("q1_R_log", "um_fator", sigla["q1_ridge"])):
        for teste, col, rotulo in (("chi2", "poder_ajustado_5pct", r"$\chi^2$ aj."),
                                   ("rot", "rejeicao_5pct", "rot."), ("boot", "rejeicao_5pct", "boot.")):
            vals = []
            for rho in rhos:
                r = t3[(t3.caso == c) & (t3.desenho == des) & (t3.processo == "S1") & np.isclose(t3.rho, rho)
                       & (t3.L == 31) & (t3.teste.map(nome3) == teste)]
                checar(len(r) <= 1, f"poder {c} ρ = {rho} {teste}")
                vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
            linhas.append(rf"{rot} & {rotulo} & " + " & ".join(fmt(x, 3) for x in vals) + r" & & & & & \\")
    cab = (r" & & \multicolumn{3}{c}{$h = 1$} & \multicolumn{3}{c}{$h = 30$} & \multicolumn{3}{c}{$h = 60$} \\ "
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n"
           r"Caso & Ruído & $\chi^2$ & rot. & boot. & $\chi^2$ & rot. & boot. & $\chi^2$ & rot. & boot.")
    nota = (r"Casos: P16-M, 16 previsores na amostra de modelagem (1.776 datas); P16-F, 16 previsores nas 987 "
            r"datas fora da amostra; FU-R e FU-F, fator único com o $\widehat{\text{BVRP}}$ do Ridge e da "
            r"floresta. rot.: rotação; boot.: bootstrap; $\chi^2$ aj.: $\chi^2$ com o valor crítico ajustado ao "
            r"tamanho. "
            r"Simulação com as variáveis explicativas reais fixas e $y$ = constante + ruído (painel A) ou + sinal "
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
                  cab, linhas, nota, tamanho=r"\footnotesize", caber=True))


def p5(t, filtro, hip, teste):
    """p (e W) da variante principal com L = h+1 no passo 5."""
    x = t[filtro & (t.hipotese == hip) & (t.teste == teste) & (t.variante == "principal") & (t.L_regra == "h+1")]
    checar(len(x) == 1, f"passo 5: {hip} {teste}: linha ausente ou repetida")
    return x.iloc[0]


def tab_componentes(d, raiz):
    t, c, hm = d["componentes_testes"], d["componentes_coeficientes"], d["componentes_holm"]
    c = c[c.L_regra == "h+1"].set_index("h")
    hm = hm[hm.L_regra == "h+1"]
    cab = (r"\multicolumn{8}{l}{\textit{Painel A: coeficientes (EP HAC) e $H_0$: $\beta_{1,h} + \beta_{2,h} = 0$}} \\" "\n"
           r"$h$ & $\hat\alpha_h$ & $\hat\beta_{1,h}$ & $\hat\beta_{2,h}$ & $\hat\beta_{1,h} + \hat\beta_{2,h}$ & "
           r"$p_{\chi^2}$ & $p$ rotação & $p$ bootstrap")
    linhas = []
    for hh in HS:
        r = c.loc[hh]
        f = t.h == hh
        linhas.append(rf"{hh} & {fmt(r.alfa, 2)} & {fmt(r.b1_vh, 3)} & {fmt(r.b2_iv, 3)} & {fmt(r.b1_mais_b2, 3)} & "
                      rf"{fmt_p(p5(t, f, 'soma_zero', 'chi2').p)} & {fmt_p(p5(t, f, 'soma_zero', 'rotacao_freedman_lane').p)} & "
                      rf"{fmt_p(p5(t, f, 'soma_zero', 'bootstrap').p)} \\")
        linhas.append(rf" & ({fmt(r.ep_alfa, 2)}) & ({fmt(r.ep_b1_vh, 3)}) & ({fmt(r.ep_b2_iv, 3)}) & "
                      rf"({fmt(r.ep_b1_mais_b2, 3)}) & & & \\")
    linhas += [r"\midrule", r"\multicolumn{8}{l}{\textit{Painel B: $H_0$: $\beta_{1,h} = \beta_{2,h} = 0$}} \\",
               r" & & & & & \multicolumn{3}{c}{Holm} \\ \cmidrule(lr){6-8}",
               r"$h$ & $W$ & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap \\"]
    for hh in HS:
        f = t.h == hh
        hb = hm[(hm.h == hh) & (hm.hipotese == "b1_b2_zero")].set_index("teste").p_holm
        linhas.append(rf"{hh} & {fmt(p5(t, f, 'b1_b2_zero', 'chi2').W, 2)} & {fmt_p(p5(t, f, 'b1_b2_zero', 'chi2').p)} & "
                      rf"{fmt_p(p5(t, f, 'b1_b2_zero', 'rotacao').p)} & {fmt_p(p5(t, f, 'b1_b2_zero', 'bootstrap').p)} & "
                      rf"{fmt_p(hb['chi2'])} & {fmt_p(hb['rotacao'])} & {fmt_p(hb['bootstrap'])} \\")
    hs_soma = hm[hm.hipotese == "soma_zero"].groupby("teste").p_holm.min()
    nota = (r"$R^{\log}_{t,h} = \alpha_h + \beta_{1,h}\,\text{VH}_{t-29:t} + \beta_{2,h}\,\text{IV}_t + \varepsilon_{t+h}$, "
            r"nas 987 datas fora da amostra; coeficientes em pontos percentuais de retorno por ponto percentual de "
            r"volatilidade, EP HAC entre parênteses. Se $\beta_{1,h} + \beta_{2,h} = 0$, a diferença "
            r"$\text{VH}_{t-29:t} - \text{IV}_t$, isto é, o HBVRP, resume o conteúdo das duas medidas. HAC de "
            r"Newey--West com $h+1$ defasagens; $p_{\chi^2}$ pela $\chi^2(2)$ no painel B e pela $\chi^2(1)$ no A. "
            r"Rotação: de $y$ no painel B; de Freedman--Lane (resíduo do modelo restrito deslocado circularmente) no "
            r"A; $g_h = \max(90, 3h)$. Bootstrap de pares em blocos circulares de $\max(30, 2h)$ dias, 1.999 "
            r"reamostragens, centrado em $\hat\beta$. Holm sobre os seis horizontes; em $\beta_{1,h} + \beta_{2,h} = 0$, "
            rf"o menor valor-$p$ ajustado é {fmt_p(hs_soma['chi2'])} ($\chi^2$), "
            rf"{fmt_p(hs_soma['rotacao_freedman_lane'])} (rotação) e {fmt_p(hs_soma['bootstrap'])} (bootstrap).")
    gravar(os.path.join(raiz, "tables", "retornos_linear", "tab_componentes_wald.tex"),
           tabela("Retorno de $h$ dias sobre a volatilidade histórica e a implícita, separadas.",
                  "tab:retlin-componentes-wald", "rrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_regime(d, raiz):
    t, c = d["regime_testes"], d["regime_coeficientes"]
    c = c[c.L_regra == "h+1"]
    cortes = (("fixo", "fixo"), ("expansivo", "reestimado"))
    cab = (r"\multicolumn{8}{l}{\textit{Painel A: coeficientes (EP HAC)}} \\" "\n"
           r"Corte & $h$ & \% alta & $\hat\alpha$ & $\hat\beta$ & $\hat\delta$ & $\hat\gamma$ & $\hat\beta + \hat\gamma$")
    linhas = []
    for corte, rot in cortes:
        for hh in (30, 60):
            r = c[(c.corte == corte) & (c.h == hh)].iloc[0]
            linhas.append(rf"{rot} & {hh} & {fmt(r.pct_alta, 1)} & {fmt(r.alfa, 2)} & {fmt(r.beta, 2)} & "
                          rf"{fmt(r.delta, 2)} & {fmt(r.gama, 2)} & {fmt(r.beta_mais_gama, 2)} \\")
            linhas.append(rf" & & & ({fmt(r.ep_alfa, 2)}) & ({fmt(r.ep_beta, 2)}) & ({fmt(r.ep_delta, 2)}) & "
                          rf"({fmt(r.ep_gama, 2)}) & ({fmt(r.ep_beta_mais_gama, 2)}) \\")
    hips = (("beta_zero", r"$\beta = 0$"), ("beta_mais_gama_zero", r"$\beta + \gamma = 0$"), ("gama_zero", r"$\gamma = 0$"))
    linhas += [r"\midrule", r"\multicolumn{8}{l}{\textit{Painel B: testes}} \\",
               r"Corte & $h$ & $H_0$ & $W$ & $p_{\chi^2}$ & $p$ rotação & $p$ bootstrap & \\"]
    for corte, rot in cortes:
        for hh in (30, 60):
            f = (t.corte == corte) & (t.h == hh)
            for hip, rot_h in hips:
                linhas.append(rf"{rot} & {hh} & {rot_h} & {fmt(p5(t, f, hip, 'chi2').W, 2)} & "
                              rf"{fmt_p(p5(t, f, hip, 'chi2').p)} & {fmt_p(p5(t, f, hip, 'rotacao_freedman_lane').p)} & "
                              rf"{fmt_p(p5(t, f, hip, 'bootstrap').p)} & \\")
    nota = (r"$R^{\log}_{t,h} = \alpha + \beta\,\widehat{\text{BVRP}}_t + \delta D_t + \gamma D_t\,\widehat{\text{BVRP}}_t "
            r"+ \varepsilon_{t+h}$, com o $\widehat{\text{BVRP}}$ do Ridge, nas 987 datas fora da amostra. $D_t = 1$ "
            r"se $\text{VH}_{t-29:t}$ supera o corte de $37{,}3\%$ a.a. (fixo) ou o corte vigente na origem "
            r"(reestimado); \% alta: porcentagem das datas com $D_t = 1$. $\beta$ é a inclinação no regime de baixa "
            r"volatilidade, e $\beta + \gamma$, no de alta. $\hat\beta$, $\hat\gamma$ e $\hat\beta + \hat\gamma$ em "
            r"pontos percentuais de retorno por ponto percentual de $\widehat{\text{BVRP}}$; $\hat\alpha$ e "
            r"$\hat\delta$ em pontos percentuais; EP HAC de $h+1$ defasagens entre parênteses. Painel B: valores-$p$ "
            r"de $\beta = 0$ (inclinação no regime de baixa), $\beta + \gamma = 0$ (inclinação no de alta) e "
            r"$\gamma = 0$ (diferença entre as inclinações), por HAC, rotação de Freedman--Lane e bootstrap, nos "
            r"cortes fixo e reestimado. $W$ com o mesmo HAC; "
            r"$p_{\chi^2}$ pela $\chi^2(1)$. Rotação de Freedman--Lane: o resíduo do modelo restrito pela hipótese é "
            r"deslocado circularmente e somado ao ajustado restrito; $g_h = \max(90, 3h)$. Bootstrap de pares em "
            r"blocos circulares de $\max(30, 2h)$ dias, 1.999 reamostragens, centrado em $\hat\beta$. O "
            r"$\widehat{\text{BVRP}}$ é tratado como dado.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_regime.tex"),
           tabela(r"Retorno de $h$ dias, $\widehat{\text{BVRP}}$ do Ridge e regime de volatilidade.",
                  "tab:apb2-regime", "llrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_corr_e_u(d, raiz):
    p1, ic = d["corr_erro_inovacao"], d["ic_corr_e_u"]
    p1 = p1[p1.y == "R_log"]
    a2 = p1[p1.inovacao == "A2_resid_MQO_16_previsores"].set_index("modelo")
    icp = ic[ic.principal & (ic.versao == "u fixo")].set_index("modelo")
    principais = ["Ridge", "floresta_aleatoria", "Ridge_rv_menos_iv", "floresta_rv_menos_iv"]
    curto = {**MODELOS, "floresta_aleatoria": "Floresta", "Ridge_rv_menos_iv": r"Ridge, $\hat f - \text{IV}$",
             "floresta_rv_menos_iv": r"Floresta, $\hat f - \text{IV}$"}
    linhas = [r"\multicolumn{7}{l}{\textit{Painel A: inovação A2}} \\"]
    for m in principais:
        e, b = a2.loc[m], icp.loc[m]
        linhas.append(rf"{curto[m]} & {fmt(e['corr'])} & {fmt(e.ep_hac)} & {fmt_p(e.p_hac)} & "
                      rf"[{fmt(b.ic_hac_inf)}; {fmt(b.ic_hac_sup)}] & [{fmt(b.ic_pct_inf)}; {fmt(b.ic_pct_sup)}] & "
                      rf"{fmt(e.corr_sem_sobrep_mediana)} [{fmt(e.corr_sem_sobrep_min)}; {fmt(e.corr_sem_sobrep_max)}] \\")
    inov = ["A1_resid_propria_previsao", "A2_resid_MQO_16_previsores", "A3_menos_media_historica_oos",
            "A4_menos_ridge_direto_oos"]
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel B: correlação por inovação}} \\",
               r"Modelo & A1 & A2 & A3 & A4 & & \\"]
    w = p1.pivot(index="modelo", columns="inovacao", values="corr")
    for m in principais + ["MQO", "LASSO", "XGBoost", "media_historica", "proxy_retrospectiva", "persistencia_viavel"]:
        linhas.append(rf"{curto[m]} & " + " & ".join(fmt(w.loc[m, i]) for i in inov) + r" & & \\")
    cab = r"Modelo & corr & EP HAC & $p$ HAC & IC 95\% HAC & IC 95\% boot. & Sem sobrep."
    nota = (r"$\hat f - \text{IV}$: forma $\hat f(X_t) - \text{IV}_t$; boot.: bootstrap; sem sobrep.: sem "
            r"sobreposição. "
            r"$e_t = \text{RBVRP}_t - \widehat{\text{BVRP}}_t$, o erro de previsão do RBVRP de cada modelo, e "
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
                  "tab:apb2-corr-e-u", "lrrrccc", cab, linhas, nota, tamanho=r"\footnotesize", caber=True, colsep=4))


def tab_ponte(d, raiz):
    a = d["ponte_dois_retornos"].set_index(["h", "retorno"])
    cv = d["ponte_conversao"].query("h == 60").set_index("variante")
    cc = d["ponte_concentracao_h60"].set_index(["retorno", "perda"])
    mr = d["ponte_maiores_retornos_h60"].query("amostra == 'fora_da_amostra_987'").head(5)
    hp = d["ponte_hiperparametros"].query("h == 60").set_index("retorno")
    linhas = [r"\multicolumn{7}{l}{\textit{Painel A: floresta treinada em cada retorno, contra a média histórica do mesmo retorno}} \\"]
    for hh in HS:
        s, l_ = a.loc[(hh, "simples")], a.loc[(hh, "log")]
        linhas.append(rf"{hh} & {fmt(s.r2_fora, 4)} & {fmt_p(s.p_cw)} & {fmt_p(s.p_dm)} & "
                      rf"{fmt(l_.r2_fora, 4)} & {fmt_p(l_.p_cw)} & {fmt_p(l_.p_dm)} \\")
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel B: $h = 60$, avaliação no retorno simples}} \\",
               r"\multicolumn{4}{l}{Previsão $\times$ referência} & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM \\"]
    s60, l60 = a.loc[(60, "simples")], a.loc[(60, "log")]
    for rot, r in ((r"Floresta no retorno simples $\times$ média do simples", s60),
                   (r"$\exp(\text{previsão log}) - 1$ $\times$ média do simples", cv.loc["expm1_log_vs_media_simples"]),
                   (r"$\exp(\text{previsão log}) - 1$ $\times$ $\exp(\text{média do log}) - 1$",
                    cv.loc["expm1_log_vs_expm1_media_log"])):
        linhas.append(rf"\multicolumn{{4}}{{l}}{{{rot}}} & {fmt(r.r2_fora, 3)} & {fmt_p(r.p_cw)} & {fmt_p(r.p_dm)} \\")
    linhas.append(rf"\multicolumn{{4}}{{l}}{{Floresta no log-retorno $\times$ média do log (painel A)}} & "
                  rf"{fmt(l60.r2_fora, 3)} & {fmt_p(l60.p_cw)} & {fmt_p(l60.p_dm)} \\")
    pr = d["ponte_ridge"].set_index(["retorno", "h"])
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel C: fator único do Ridge, $\widehat{\text{BVRP}}$ como regressor}} \\",
               r"$h$ & \multicolumn{1}{c}{$\hat\beta_h$} & \multicolumn{1}{c}{EP HAC} & \multicolumn{1}{c}{$p_{\chi^2}$} & "
               r"\multicolumn{1}{c}{$p$ rotação} & \multicolumn{1}{c}{$p$ bootstrap} & \multicolumn{1}{c}{$R^2$} \\",
               r"\cmidrule(lr){2-7}"]
    for ret, rot in (("simples", "Retorno simples"), ("log", "Log-retorno")):
        linhas.append(rf"\multicolumn{{7}}{{l}}{{{rot}}} \\")
        for hh in HS:
            r = pr.loc[(ret, hh)]
            linhas.append(rf"{hh} & {fmt(r.beta)} & {fmt(r.ep_hac)} & {fmt_p(r.p_hac)} & {fmt_p(r.p_rotacao)} & "
                          rf"{fmt_p(r.p_bootstrap)} & {fmt(r.r2, 4)} \\")
    w = cv.loc["winsor_1_99"]
    datas = "; ".join(rf"{r.date[8:10]}/{r.date[5:7]}/{r.date[:4]}, {fmt(r.retorno_simples_pct, 1)}\%" for r in mr.itertuples())
    n = lambda ret, perda, c: int(cc.loc[(ret, perda), c])  # noqa: E731
    cab = (r" & \multicolumn{3}{c}{Retorno simples} & \multicolumn{3}{c}{Log-retorno} \\ "
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
           r"$h$ & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM")
    nota = (r"Floresta aleatória direto no retorno de $h$ dias, nas 987 datas fora da amostra, com o mesmo "
            r"protocolo nos dois retornos (17 variáveis, mesma grade, 500 árvores, validação cruzada de cinco dobras "
            r"com embargo $h$, 33 origens, mesma semente); só o alvo muda. Retorno simples: as previsões da versão "
            r"anterior do Capítulo~\ref{chap:retornos_nao_linear}; "
            r"log-retorno: Tabela~\ref{tab:retnl-oos-floresta}. CW unilateral e DM bilateral contra a média "
            r"histórica, com HAC de $h+1$ defasagens. Painel B: a previsão do "
            r"log-retorno convertida para retorno simples. Com o realizado winsorizado a 1\% e 99\% e as "
            rf"previsões inalteradas, o $R^2_{{\text{{fora}}}}$ em $h = 60$ fica em {fmt(w.r2_fora, 3)} ($p$ DM "
            rf"{fmt_p(w.p_dm)}). Em $h = 60$, metade da soma dos erros quadráticos da floresta vem de "
            rf"{n('simples', 'e2_floresta', 'n_50pct')} das 987 datas e 80\%, de "
            rf"{n('simples', 'e2_floresta', 'n_80pct')} no retorno simples "
            rf"({n('log', 'e2_floresta', 'n_50pct')} e {n('log', 'e2_floresta', 'n_80pct')} no log-retorno). "
            rf"Maiores retornos simples de 60 dias: {datas}. Em $h = 60$, a validação cruzada escolhe árvores de "
            rf"profundidade 1 em {int(hp.loc['simples', 'max_depth_1'])} das 33 origens com o retorno simples e em "
            rf"{int(hp.loc['log', 'max_depth_1'])} com o log-retorno, e árvores sem limite de profundidade em "
            rf"{int(hp.loc['simples', 'max_depth_None'])} e {int(hp.loc['log', 'max_depth_None'])}; o $R^2$ "
            rf"dentro da amostra, em média nas 33 origens, é {fmt(hp.loc['simples', 'r2_dentro_media'], 3)} e "
            rf"{fmt(hp.loc['log', 'r2_dentro_media'], 3)}. Painel C: $R_{{t,h}} = \alpha_h + \beta_h\,"
            r"\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$, com o BVRP previsto pelo Ridge, nas 987 datas, com o "
            r"retorno simples e com o log-retorno em pontos percentuais; HAC com $h+1$ defasagens, rotação e "
            r"bootstrap de pares em blocos circulares como na Tabela~\ref{tab:retlin-um-fator}; no log-retorno, os "
            r"números são os daquela tabela.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_ponte_retorno_simples.tex"),
           tabela("Floresta direto no retorno e fator único do Ridge: retorno simples e log-retorno.",
                  "tab:apb2-ponte-retorno",
                  "rrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def milhar(n):
    """1999 -> '1{.}999' (separador de milhar no padrão do texto)."""
    return f"{int(n):,}".replace(",", "{.}")


def tab_atenuacao(d, raiz):
    a = d["regressor_gerado_atenuacao"]
    f = a[a.celula == "ridge_fixo_bic"].set_index("h")
    sens = {c: a[a.celula == c].set_index("h").razao_beta_hat for c in ("ridge_fixo_p29", "ridge_fixo_p60")}
    linhas = [rf"{hh} & {fmt(f.loc[hh, 'beta_hat'])} & {fmt(f.loc[hh, 'beta_v'])} & {fmt(f.loc[hh, 'media_bstar'])} & "
              rf"{fmt(f.loc[hh, 'razao_beta_hat'], 2)} & {fmt(f.loc[hh, 'razao_corr'], 2)} & "
              rf"{fmt(f.loc[hh, 'razao_beta_v'], 2)} & {fmt(f.loc[hh, 'dp_bstar'])} & {fmt(f.loc[hh, 'ep_robusto'])} \\"
              for hh in HS]
    cab = (r"$h$ & $\hat\beta_h$ & $\beta^{\text{v}}_h$ & $\overline{\beta^*_h}$ & $\overline{\beta^*_h}/\hat\beta_h$ & "
           r"$\rho^*/\rho^{\text{v}}$ & $\overline{\beta^*_h}/\beta^{\text{v}}_h$ & dp($\beta^*$) & EP robusto")
    r0 = f.iloc[0]
    nota = (r"Ridge, $R^{\log}_{t,h} = \alpha_h + \beta_h\,\widehat{\text{BVRP}}_t + \varepsilon_{t+h}$ nas 987 datas, com o "
            r"retorno real fixo e o BVRP previsto regenerado em cada reamostragem (Seção~\ref{sec:apb2-regressor-gerado}); "
            rf"{milhar(r0.B)} reamostragens. "
            r"$\hat\beta_h$: coeficiente na amostra real; $\beta^{\text{v}}_h$: coeficiente do retorno na verdade "
            r"$\mu$ (o modelo ajustado nas 1.776 datas); $\overline{\beta^*_h}$: média dos coeficientes com o BVRP "
            r"previsto simulado; $\rho^*/\rho^{\text{v}}$: média de corr($R^{\log}$, BVRP previsto simulado) sobre "
            r"corr($R^{\log}$, $\mu$), livre de escala. $\overline{\beta^*_h}/\beta^{\text{v}}_h$ mistura a atenuação "
            r"com a diferença de escala entre $\mu$ e a previsão original, cujo coeficiente é maior em módulo que o de "
            r"$\mu$. EP robusto: intervalo interquartil dos $\beta^*$ dividido por 1,349. Razão "
            rf"$\overline{{\beta^*_h}}/\hat\beta_h$ com $p = 29$: {fmt(sens['ridge_fixo_p29'].min(), 2)} a "
            rf"{fmt(sens['ridge_fixo_p29'].max(), 2)}; com $p = 60$: {fmt(sens['ridge_fixo_p60'].min(), 2)} a "
            rf"{fmt(sens['ridge_fixo_p60'].max(), 2)}. Nesta variante (retorno fixo, inovações sorteadas entre as "
            rf"1.776 datas), as previsões simuladas do BVRP têm, em média, {fmt(100 * r0.razao_dp_f, 0)}\% do "
            rf"desvio-padrão da previsão original (mínimo de {fmt(100 * r0.dp_f_min_sobre_dp_f, 0)}\%); trata-se da "
            r"dispersão da previsão, e não do erro simulado da Tabela~\ref{tab:apb2-regressor-gerado}. A variante não é um teste limpo de erro de medida: o Ridge "
            r"com o $\alpha$ fixo é reestimado sobre um alvo simulado a partir de uma previsão já encolhida e com as "
            r"mesmas variáveis.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_atenuacao.tex"),
           tabela(r"BVRP previsto pelo Ridge como regressor gerado: atenuação, com o retorno real fixo.",
                  "tab:apb2-atenuacao", "rrrrrrrrr", cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_regressor_gerado(d, raiz):
    a = d["regressor_gerado_atenuacao"]
    linhas = []
    rot = {"ridge_conjunto_987": "Ridge, datas sorteadas entre as 987 (principal)",
           "ridge_conjunto_1776": "Ridge, datas sorteadas entre as 1.776 (sensibilidade)",
           "floresta_conjunto_987": r"Floresta, $\hat f(X_t) - \text{IV}_t$, datas sorteadas entre as 987"}
    for k, (cel, nome) in enumerate(rot.items()):
        g = a[a.celula == cel].set_index("h").loc[HS]
        h1, h3 = holm(g.p_h0.to_numpy()), holm(g.p_h0_padronizado.to_numpy())
        if k:
            linhas.append(r"\midrule")
        linhas.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{nome}}}; B = {milhar(g.B.iloc[0])}}} \\")
        for i, hh in enumerate(HS):
            r = g.loc[hh]
            linhas.append(rf"{hh} & {fmt(r.beta_hat)} & {fmt(r.dp_bstar)} & {fmt(r.ep_robusto)} & {fmt(r.ep_hac)} & "
                          rf"{fmt_p(r.p_h0)} & {fmt_p(h1[i])} & {fmt_p(r.p_percentil)} & {fmt_p(r.p_h0_padronizado)} & "
                          rf"{fmt_p(h3[i])} \\")
    pv = d["regressor_gerado_previo"]
    pr = pv[(pv.modelo == "ridge") & (pv.sorteio.astype(str) == "987")]
    eh_erro = pr.h.astype(str) == "erro e"
    checar(len(pr) == 7 and eh_erro.sum() == 1, "prévia (a): linhas do Ridge com sorteio nas 987 datas")
    rr, re_ = pr[~eh_erro].razao.astype(float), float(pr[eh_erro].razao.iloc[0])
    va = d["regressor_gerado_variancias"].set_index("modelo")
    fr = {c: a[a.celula == c].iloc[0] for c in rot}
    cab = (r" & & & & & \multicolumn{2}{c}{(i) sob $H_0$} & (ii) & \multicolumn{2}{c}{(iii) padronizado} \\ "
           r"\cmidrule(lr){6-7}\cmidrule(lr){9-10}" "\n"
           r"$h$ & $\hat\beta_h$ & dp($\beta^*$) & EP robusto & EP HAC & $p$ & Holm & $p$ percentil & $p$ & Holm")
    nota = (r"Retorno e erro de previsão simulados juntos (Seção~\ref{sec:apb2-regressor-gerado}): em cada "
            r"reamostragem, as datas sorteadas dão o par de inovações do erro de previsão e do retorno; o retorno "
            r"simulado é $\alpha^{\text{v}}_h + \beta^{\text{v}}_h\mu_t + u^*_t$ e, sob $H_0$, $\alpha^{\text{v}}_h + u^*_t$. "
            r"(i) fração de $|\beta^*_0| \ge |\hat\beta_h|$ sob $H_0$, o critério principal; (ii) $2\min\{P(\beta^* \le 0), "
            r"P(\beta^* \ge 0)\}$; (iii) como (i), com o coeficiente padronizado $\beta^*\,\text{dp}(\widehat{\text{BVRP}}^*)$ "
            r"contra $\hat\beta_h\,\text{dp}(\widehat{\text{BVRP}})$, livre do encolhimento do Ridge. Holm sobre os seis "
            r"horizontes. EP robusto: intervalo interquartil dividido por 1,349. Desvio-padrão médio das previsões "
            rf"simuladas sobre o da original: {fmt(fr['ridge_conjunto_987'].razao_dp_f, 2)} (mínimo "
            rf"{fmt(fr['ridge_conjunto_987'].dp_f_min_sobre_dp_f, 2)}), {fmt(fr['ridge_conjunto_1776'].razao_dp_f, 2)} "
            rf"e {fmt(fr['floresta_conjunto_987'].razao_dp_f, 2)}, nos três painéis. Com as datas sorteadas entre as "
            rf"987 (painel principal), o retorno simulado tem de {fmt(rr.min(), 2)} a {fmt(rr.max(), 2)} vez o "
            rf"desvio-padrão do real, e o erro de previsão simulado, $e^*$, {fmt(re_, 2)} vez o desvio-padrão do "
            r"resíduo dentro da amostra; essa razão é a do erro, e não a da dispersão das previsões simuladas da "
            r"Tabela~\ref{tab:apb2-atenuacao}, que coincide por acaso. Na floresta, a variância "
            rf"do erro fora da amostra é {fmt(va.loc['floresta', 'razao_fora_dentro_987'], 2)} vez a do resíduo dentro "
            rf"dela (no Ridge, {fmt(va.loc['ridge', 'razao_fora_dentro_987'], 2)}), e o erro simulado subestima o "
            r"erro de previsão.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_regressor_gerado.tex"),
           tabela(r"Fator único com o BVRP previsto tratado como regressor gerado: erro-padrão e valores-$p$.",
                  "tab:apb2-regressor-gerado", "rrrrrrrrrr", cab, linhas, nota, tamanho=r"\scriptsize"))


def tab_ar(d, raiz):
    ar = d["regressor_gerado_ar"]
    nomes = {"erro (ridge_conjunto_987)": "Erro de previsão, Ridge", "erro (ridge_fixo_p29)": r"\quad $p = 29$ (sensibilidade)",
             "erro (ridge_fixo_p60)": r"\quad $p = 60$ (sensibilidade)", "erro (floresta_conjunto_987)": "Erro de previsão, floresta"}
    linhas = []
    for serie, rotulo in nomes.items():
        r = ar[ar.serie == serie].iloc[0]
        linhas.append(rf"{rotulo} & {int(r.ordem)} & {r.criterio.replace('fixo', 'fixa')} & {fmt(r.modulo_raiz_max, 3)} & "
                      rf"{fmt_p(r.lb_p30)} & {fmt_p(r.lb_p60)} & -- \\")
    for modelo, nome in (("ridge", "Ridge"), ("floresta", "floresta")):
        linhas.append(r"\addlinespace")
        for hh in HS:
            r = ar[(ar.modelo == modelo) & (ar.serie == f"retorno h = {hh}")].iloc[0]
            linhas.append(rf"Retorno, {nome}, $h = {hh}$ & {int(r.ordem)} & {r.criterio} & {fmt(r.modulo_raiz_max, 3)} & "
                          rf"{fmt_p(r.lb_p30)} & {fmt_p(r.lb_p60)} & {fmt(r.corr_inov_erro)} \\")
    cab = r"Série & Ordem & Critério & Raiz máx. & LB $p$ (30) & LB $p$ (60) & corr$(\eta, \xi)$"
    nota = (r"Processos autorregressivos do erro de previsão do BVRP, $e_t = \text{RBVRP}_t - \mu_t$, e do resíduo do "
            r"retorno em $\mu$, $u_{t,h}$, nas 1.776 datas. Erro: ordem pelo BIC entre 29 e 60 (os alvos de 30 dias se "
            r"sobrepõem em 29). Retorno: a menor ordem entre $\max(h-1, 1)$ e $h+59$ com Ljung--Box acima de 0,05 em "
            r"30 e em 60 defasagens; em todos os horizontes uma ordem passou, e o BIC não foi usado. Raiz máx.: maior "
            r"módulo das raízes do polinômio característico. corr$(\eta, \xi)$: correlação entre as inovações do erro e "
            r"do retorno da mesma data, preservada no sorteio conjunto.")
    gravar(os.path.join(raiz, "tables", "apendice_b2", "tab_ar.tex"),
           tabela("Processos autorregressivos do erro de previsão e do retorno.", "tab:apb2-ar", "lrlrrrr",
                  cab, linhas, nota, tamanho=r"\footnotesize"))


def tab_profundidade(d, raiz):
    """Apêndice F (#135): floresta e XGBoost com a grade do T5 restrita a profundidade >= 3."""
    r = d["sens_profundidade_resultados"]
    pr = d["sens_profundidade_profundidades"]
    hip5 = pd.read_csv(os.path.join(ROOT, "outputs", "T5", "hiperparametros_T5.csv"))
    hip5 = hip5[hip5.modelo.isin(["floresta_aleatoria", "XGBoost"])].assign(
        d=lambda x: x.params.str.extract(r"'max_depth': (\w+)")[0])
    nome = {"floresta_aleatoria": "Floresta", "XGBoost": "XGBoost"}
    jan = {"expansiva": "expansiva", "movel": "móvel"}
    linhas = [r"\multicolumn{7}{l}{\textit{Painel A: desempenho fora da amostra}} \\"]
    for j, jn in jan.items():
        linhas.append(rf"\multicolumn{{7}}{{l}}{{Janela {jn}}} \\")
        for m in ("floresta_aleatoria", "XGBoost"):
            for versao, rot in (("original", "original"), ("profundidade >= 3", r"prof.\ $\ge 3$")):
                x = r[(r.janela == j) & (r.modelo == m) & (r.versao == versao)].iloc[0]
                linhas.append(rf"\quad {nome[m]}, {rot} & {fmt(x.r2_fora)} & {fmt_p(x.p_cw)} & {fmt_p(x.p_dm_vs_media)} & "
                              rf"{fmt_p(x.p_dm_vs_original) if versao != 'original' else '--'} & {fmt(x.r2_dentro)} & "
                              rf"{fmt_p(x.mcs_p)} \\")
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel B: profundidade escolhida (nº de origens, de 33)}} \\",
               r"Modelo, janela & \multicolumn{3}{l}{Grade completa} & \multicolumn{3}{l}{Profundidade $\ge 3$} \\"]

    def contagem(g):
        def chave(v):
            return 99 if v == "None" else int(v)
        return "; ".join(f"{('sem limite' if v == 'None' else v)}: {int(n)}" for v, n in sorted(g, key=lambda t: chave(t[0])))

    for j, jn in jan.items():
        for m in ("floresta_aleatoria", "XGBoost"):
            orig = hip5[(hip5.janela == j) & (hip5.modelo == m)].groupby("d").size()
            rest = pr[(pr.janela == j) & (pr.modelo == m)]
            checar(orig.sum() == 33 and rest.n_origens.sum() == 33, f"profundidades {j} {m}")
            linhas.append(rf"{nome[m]}, {jn} & \multicolumn{{3}}{{l}}{{{contagem(orig.items())}}} & "
                          rf"\multicolumn{{3}}{{l}}{{{contagem(zip(rest.d.astype(str), rest.n_origens))}}} \\")
    linhas += [r"\midrule", r"\multicolumn{7}{l}{\textit{Painel C: valor-$p$ do MCS dos demais modelos}} \\",
               r"Janela & Ridge & MQO & LASSO & Média histórica & & \\"]
    for j, jn in jan.items():
        for versao, rot in (("original", "originais"), ("MCS com os restritos", r"com prof.\ $\ge 3$")):
            vals = [r[(r.janela == j) & (r.modelo == m) & (r.versao == versao)].mcs_p.iloc[0]
                    for m in ("Ridge", "MQO", "LASSO", "media_historica")]
            linhas.append(rf"{jn}, {rot} & " + " & ".join(fmt_p(v) for v in vals) + r" & & \\")
    orig_in = r[(r.versao == "original") & r.modelo.isin(["floresta_aleatoria", "XGBoost"])].r2_dentro
    cab = (r"Modelo & $R^2_{\text{fora}}$ & $p$ CW & $p$ DM, média & $p$ DM, original & $R^2$ dentro & $p$ MCS")
    nota = (r"Floresta aleatória e XGBoost com a grade do Capítulo~\ref{chap:previsao_bvrp} restrita a profundidade "
            r"$\ge 3$, no mesmo protocolo: 33 origens por janela, validação cruzada embargada de cinco dobras e a "
            r"mesma semente. Floresta: profundidade 3, 6 ou sem limite, 45 das 75 combinações. XGBoost: a grade só "
            r"tem a profundidade 4 acima de 3, 24 das 72 combinações. $R^2_{\text{fora}}$, Clark--West e "
            r"Diebold--Mariano contra a média histórica, com HAC de 31 defasagens; DM, original: a versão restrita "
            r"contra a original. $R^2$ dentro: média das 33 reestimações; nos originais, de "
            rf"{fmt(orig_in.min(), 2)} a {fmt(orig_in.max(), 2)}. MCS: $T_{{\max}}$, blocos de 60 dias e 9.999 "
            r"reamostragens, sobre os oito modelos do Capítulo~\ref{chap:previsao_bvrp}; nas linhas restritas e no "
            r"painel C, com a floresta e o XGBoost originais trocados pelos restritos.")
    gravar(os.path.join(raiz, "tables", "apendice_profundidade", "tab_profundidade.tex"),
           tabela(r"Floresta aleatória e XGBoost com profundidade mínima de 3.", "tab:apf-profundidade", "lrrrrrr",
                  cab, linhas, nota, tamanho=r"\footnotesize", caber=True))


def main():
    global B2
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=ROOT)
    ap.add_argument("--b2", default=B2, help="pasta com as cópias e o procedencia_B2.csv")
    args = ap.parse_args()
    B2 = os.path.abspath(args.b2)
    d = carregar_e_conferir_procedencia()
    conferir_coerencia(d)
    for f in (tab_wald_direto, tab_um_fator, tab_oos_retorno, tab_oos_floresta, tab_componentes, tab_tamanho_poder,
              tab_regime, tab_corr_e_u, tab_ponte, tab_atenuacao, tab_regressor_gerado, tab_ar, tab_profundidade):
        f(d, args.raiz)
        print(f"  {f.__name__}: ok")
    print(f"Tabelas gravadas em {os.path.join(args.raiz, 'tables')}")


if __name__ == "__main__":
    main()
