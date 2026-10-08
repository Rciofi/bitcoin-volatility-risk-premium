"""
publicar_apendice.py — tabelas e figuras do Apêndice A (estratégias de negociação), sem reestimar nada

Lê as saídas do A1 (outputs/A1/series_diarias_A1.csv e metricas_A1.csv, gravadas
por code/estrategias_A1.py) e grava, na notação da seção 14 de docs/pendencias_T1.md:
  tables/estrategias/tab_principal.tex   Ridge, q80: comprada/neutra e vendida/neutra (0, 10, 30 bps) e C&M
  tables/estrategias/tab_quantis.tex     Ridge, q60 a q90, nas duas direções
  tables/estrategias/tab_robustez.tex    burn-in 126, proxy na amostra completa e ponte (datas do Ridge)
  tables/estrategias/tab_regimes.tex     por regime de volatilidade, com o C&M de cada regime
  figs/estrategias/fig_principal.png     valor acumulado e drawdown (Ridge, q80, sem custos)
  figs/estrategias/fig_quantis.png       valor acumulado por quantil, (a) comprada e (b) vendida
  figs/estrategias/fig_proxy.png         proxy na amostra completa, com o início da negociação do Ridge
A decomposição do publicado ao atual (tab_A1_decomposicao, tab_A1_compra_manutencao)
fica em outputs/A1, para o relatório ao orientador, e não entra no apêndice.

Conferência (sai com erro se algo não bater; tolerância 1e-9): com o sinal do
Ridge, a proxy, o retorno seguinte e o regime gravados em series_diarias_A1.csv,
refaz as regras com as funções do estrategias_A1 (só quantis expansivos e médias;
o Ridge não é reestimado) e confere todas as linhas de metricas_A1.csv e as
posições e limiares do q80 gravados. Confere também, sem as funções do A1, as
métricas do compra e manutenção e da regra comprada q80 a partir das posições
gravadas, e a simetria do Sharpe entre as duas direções sem custos.

Uso:  python code/publicar_apendice.py [--raiz DIR]
      --raiz: onde gravar tables/ e figs/ (padrão: a raiz do repositório)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estrategias_A1 as A  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(ROOT, "outputs", "A1")
TOL = 1e-9
CHAVES = ["variante", "regime", "direcao", "quantil", "custo_bps"]
# rótulos no feminino ("regra"), como no texto; o estrategias_A1.py (saídas do relatório) não muda
ROTULO_DIRECAO = {**A.ROTULO_DIRECAO, "comprado": "Comprada/neutra", "vendido": "Vendida/neutra"}
ROTULO_SINAL = {"ridge": r"$\widehat{\text{BVRP}}_t$ (Ridge), burn-in de 252",
                "ridge_b126": r"$\widehat{\text{BVRP}}_t$ (Ridge), burn-in de 126",
                "proxy": r"$\text{BVRP}^{\text{proxy}}_t$, amostra completa",
                "ponte": r"$\text{BVRP}^{\text{proxy}}_t$, datas do Ridge"}
NOTA_COMUM = (r"Posição definida com informação até o fim de $t$ e aplicada ao retorno simples de $t$ a $t+1$; "
              r"limiar no quantil $q$ do sinal em janela expansiva (observações até $t$). Custos unilaterais "
              r"sobre $|\Delta \text{posição}|$. Sharpe: média $\times$ 365 / (desvio-padrão $\times \sqrt{365}$), "
              r"sem taxa livre de risco; Sortino: média $\times$ 365 / "
              r"($\sqrt{\text{média}(\min(r_t, 0)^2)} \times \sqrt{365}$), com todos os dias. DD máx.: drawdown "
              r"máximo a partir do valor inicial 1. Compra e manutenção nas mesmas datas, sem custo. Sem custos, "
              r"o retorno diário da regra vendida/neutra é o da comprada/neutra com o sinal trocado.")
fmt_eixo = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


# ---------------------------------------------------------------------------
# Leitura e conferência
# ---------------------------------------------------------------------------
def ler():
    s = pd.read_csv(os.path.join(ENTRADA, "series_diarias_A1.csv"), parse_dates=["date"]).set_index("date")
    m = pd.read_csv(os.path.join(ENTRADA, "metricas_A1.csv"))
    return s, m


def calcular_de_series(s):
    """A.calcular com os sinais, o retorno seguinte e o regime gravados no CSV diário."""
    sinais = (s.sinal_ridge.dropna().rename("ridge"), s.sinal_proxy.rename("proxy"), s.ret_seguinte,
              s.index.max())
    original, A.sinais = A.sinais, lambda vt, prev: sinais
    try:
        return A.calcular(None, None, s.regime_alta_fixo.reset_index())
    finally:
        A.sinais = original


def _maxdif(a, b):
    a, b = pd.Series(a, dtype=float).reset_index(drop=True), pd.Series(b, dtype=float).reset_index(drop=True)
    if not (a.isna() == b.isna()).all():
        return np.inf
    return float((a - b).abs().max()) if a.notna().any() else 0.0


def _metricas_diretas(p, r_seg, c):
    """Sharpe, Sortino e drawdown sem as funções do A1."""
    dp = np.abs(np.diff(np.concatenate([[0.0], p])))
    r = p * r_seg - dp * c / 1e4
    sharpe = r.mean() * 365 / (r.std(ddof=1) * np.sqrt(365))
    sortino = r.mean() * 365 / (np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(365))
    v = np.cumprod(1 + r)
    dd = np.min(v / np.maximum.accumulate(np.maximum(v, 1.0)) - 1)
    return {"sharpe": sharpe, "sortino": sortino, "max_drawdown": dd, "n_operacoes": dp.sum()}


def conferir(s, m_csv):
    m, series, datas = calcular_de_series(s)
    erros = []
    if len(m) != len(m_csv):
        erros.append(f"{len(m)} linhas recalculadas contra {len(m_csv)} no CSV")
    j = m.merge(m_csv, on=CHAVES, suffixes=("", "_csv"), how="outer", validate="1:1", indicator=True)
    if (j._merge != "both").any():
        erros.append(f"{(j._merge != 'both').sum()} linhas sem par entre o recálculo e o CSV")
    for c in m.columns.difference(CHAVES):
        if c in ("inicio", "fim"):
            if not (j[c].astype(str) == j[c + "_csv"].astype(str)).all():
                erros.append(f"{c}: datas diferentes")
        elif (d := _maxdif(j[c], j[c + "_csv"])) > TOL:
            erros.append(f"{c}: diferença máx. {d:.1e}")

    # posições e limiares do q80 gravados
    for nome, sinal in (("ridge", s.sinal_ridge.dropna()), ("proxy", s.sinal_proxy)):
        for col, valor in ((f"limiar_{nome}_q80", A.limiar_expansivo(sinal, A.Q_REF, A.BURNIN)),
                           (f"pos_{nome}_q80_comprado", A.posicoes(sinal, A.Q_REF, A.BURNIN, 1.0))):
            if (d := _maxdif(valor.reindex(s.index), s[col])) > TOL:
                erros.append(f"{col}: diferença máx. {d:.1e}")

    # sem as funções do A1: C&M e comprado q80 das posições gravadas, nas datas de negociação
    for v, col in (("ridge", "pos_ridge_q80_comprado"), ("proxy", "pos_proxy_q80_comprado")):
        d = datas[v]
        r_seg = s.ret_seguinte.loc[d].to_numpy()
        for direcao, p, custos in (("compra_e_manutencao", np.ones(len(d)), [0]),
                                   ("comprado", s[col].loc[d].to_numpy(), A.CUSTOS)):
            for c in custos:
                x = m_csv[(m_csv.variante == v) & (m_csv.regime == "todos") & (m_csv.direcao == direcao)
                          & (m_csv.custo_bps == c)]
                x = x if direcao == "compra_e_manutencao" else x[x.quantil == A.Q_REF]
                for k, valor in _metricas_diretas(p, r_seg, c).items():
                    if abs(valor - x[k].iloc[0]) > TOL:
                        erros.append(f"{v}/{direcao}/{c} bps, {k}: {valor} contra {x[k].iloc[0]}")

    # simetria sem custos: Sharpe da vendida = − Sharpe da comprada (o Sortino não é simétrico)
    sem = m_csv[(m_csv.custo_bps == 0) & m_csv.quantil.notna()].set_index(["variante", "regime", "quantil",
                                                                          "direcao"]).sharpe
    dif = (sem.xs("comprado", level="direcao") + sem.xs("vendido", level="direcao")).abs()
    if dif.max() > TOL:
        erros.append(f"simetria do Sharpe: {dif.max():.1e}")

    if erros:
        sys.exit("Conferência falhou:\n  " + "\n  ".join(erros))
    print(f"Conferência: {len(m_csv)} linhas de metricas_A1.csv, limiares e posições do q80, "
          f"métricas diretas e simetria — tudo bate (tolerância {TOL:g})")
    return series, datas


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def _periodo(a, sep=" a "):
    return f"{pd.Timestamp(a.inicio):%d/%m/%Y}{sep}{pd.Timestamp(a.fim):%d/%m/%Y}"


def _tabela(raiz, nome, legenda, rotulo, colfmt, cabecalho, linhas, nota):
    corpo = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{legenda}}}", rf"\label{{{rotulo}}}",
             r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", cabecalho + r" \\",
             r"\midrule"]
    for ln in linhas:
        corpo.append(r"\midrule" if ln == "MIDRULE" else " & ".join(ln) + r" \\")
    corpo += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
              rf"\parbox{{0.95\linewidth}}{{\footnotesize\textit{{Nota}}: {nota}}}", r"\end{table}", ""]
    with open(os.path.join(raiz, "tables", "estrategias", nome), "w", encoding="utf-8") as fh:
        fh.write("\n".join(corpo))


def tabelas(m, raiz):
    def sel(v, direcao, q=A.Q_REF, custo=0, regime="todos"):
        x = m[(m.variante == v) & (m.regime == regime) & (m.direcao == direcao) & (m.custo_bps == custo)]
        return x.iloc[0] if direcao == "compra_e_manutencao" else x[x.quantil == q].iloc[0]

    n, pct = A._n, lambda x: A._n(x, 1, True)
    bh = sel("ridge", "compra_e_manutencao")
    periodo = _periodo(bh)

    # 1. Principal
    linhas = []
    for direcao in A.DIRECOES:
        for c in A.CUSTOS:
            a = sel("ridge", direcao, custo=c)
            linhas.append([ROTULO_DIRECAO[direcao] if c == 0 else "", f"{c}", pct(a.ret_anual), pct(a.vol_anual),
                           n(a.sharpe), n(a.sortino), pct(a.max_drawdown), A._n(a.giro_anual, 1),
                           A._int(a.n_operacoes), pct(a.pct_tempo)])
        linhas.append("MIDRULE")
    linhas.append([ROTULO_DIRECAO["compra_e_manutencao"], "--", pct(bh.ret_anual), pct(bh.vol_anual),
                   n(bh.sharpe), n(bh.sortino), pct(bh.max_drawdown), "--", "--", pct(bh.pct_tempo)])
    _tabela(raiz, "tab_principal.tex",
            rf"Estratégias com o BVRP previsto pelo Ridge (quantil 80\%, {periodo}, {A._int(bh.n_dias)} dias).",
            "tab:estr-principal", "llrrrrrrrr",
            r"Regra & Custo (bps) & Ret. anual (\%) & Vol. (\%) & Sharpe & Sortino & DD máx. (\%) & "
            r"Giro (op./ano) & Operações & Tempo posic. (\%)", linhas,
            NOTA_COMUM + r" Burn-in de 252 previsões do Ridge.")

    # 2. Quantis
    linhas = []
    for direcao in A.DIRECOES:
        for q in A.QUANTIS:
            a = [sel("ridge", direcao, q, c) for c in A.CUSTOS]
            linhas.append([ROTULO_DIRECAO[direcao] if q == A.QUANTIS[0] else "", f"q{int(q * 100)}"]
                          + [n(x.sharpe) for x in a]
                          + [n(a[0].sortino), pct(a[0].max_drawdown), A._n(a[0].giro_anual, 1), pct(a[0].pct_tempo)])
        linhas.append("MIDRULE")
    linhas.append([ROTULO_DIRECAO["compra_e_manutencao"], "--", n(bh.sharpe), "--", "--", n(bh.sortino),
                   pct(bh.max_drawdown), "--", pct(bh.pct_tempo)])
    _tabela(raiz, "tab_quantis.tex", rf"Estratégias com o BVRP previsto pelo Ridge, por quantil ({periodo}).",
            "tab:estr-quantis", "llrrrrrrr",
            r"Regra & Quantil & Sharpe (0) & Sharpe (10) & Sharpe (30) & Sortino & DD máx. (\%) & "
            r"Giro (op./ano) & Tempo posic. (\%)", linhas,
            NOTA_COMUM + r" Entre parênteses, o custo em bps; Sortino, drawdown, giro e tempo sem custos.")

    # 3. Robustez
    linhas = []
    for v in ["ridge", "ridge_b126", "proxy", "ponte"]:
        b_ = sel(v, "compra_e_manutencao")
        for i, direcao in enumerate(["comprado", "vendido", "compra_e_manutencao"]):
            a = [sel(v, direcao, custo=c) for c in (A.CUSTOS if direcao != "compra_e_manutencao" else [0])]
            sh = [n(x.sharpe) for x in a] + ["--"] * (3 - len(a))
            linhas.append([ROTULO_SINAL[v] if i == 0 else "", _periodo(b_, "--") if i == 0 else "",
                           A._int(b_.n_dias) if i == 0 else "", ROTULO_DIRECAO[direcao]] + sh
                          + [pct(a[0].max_drawdown), pct(a[0].pct_tempo)])
        linhas.append("MIDRULE")
    _tabela(raiz, "tab_robustez.tex", r"Testes de robustez: burn-in e proxy retrospectiva (quantil 80\%).",
            "tab:estr-robustez", "lllrrrrrr",
            r"Sinal & Período & Dias & Regra & Sharpe (0) & Sharpe (10) & Sharpe (30) & DD máx. (\%) & "
            r"Tempo posic. (\%)", linhas[:-1],
            NOTA_COMUM + r" $\text{BVRP}^{\text{proxy}}_t = \text{VH}_{t-29:t} - \text{IV}_t$, conhecida em $t$. "
            r"Nas datas do Ridge, o limiar da proxy usa todo o histórico dela desde 2021.")

    # 4. Regimes, com o C&M de cada regime
    linhas, per = [], {}
    for v in ["ridge", "proxy"]:
        per[v] = _periodo(sel(v, "compra_e_manutencao", regime="baixa"))
        for rg in ("baixa", "alta"):
            for i, direcao in enumerate(["comprado", "vendido", "compra_e_manutencao"]):
                a = sel(v, direcao, regime=rg)
                s30 = n(sel(v, direcao, custo=30, regime=rg).sharpe) if direcao != "compra_e_manutencao" else "--"
                linhas.append([(r"$\widehat{\text{BVRP}}_t$ (Ridge)" if v == "ridge"
                                else r"$\text{BVRP}^{\text{proxy}}_t$") if (rg == "baixa" and i == 0) else "",
                               f"{rg.capitalize()} volatilidade" if i == 0 else "", ROTULO_DIRECAO[direcao],
                               A._int(a.n_dias_posicionado), pct(a.ret_anual), n(a.sharpe), s30,
                               pct(a.max_drawdown)])
        linhas.append("MIDRULE")
    _tabela(raiz, "tab_regimes.tex",
            rf"Estratégias por regime de volatilidade (quantil 80\%): Ridge de {per['ridge']} e proxy de "
            rf"{per['proxy']}.", "tab:estr-regimes", "lllrrrrr",
            r"Sinal & Regime & Regra & Dias posic. & Ret. anual (\%) & Sharpe (0) & Sharpe (30) & DD máx. (\%)",
            linhas[:-1],
            NOTA_COMUM + r" Regime de alta volatilidade: $\text{VH}_{t-29:t} > 37{,}3\%$ a.a. "
            r"(Seção~\ref{sec:met-regimes}). Em cada regime, a regra e o compra e manutenção ficam neutros nos "
            r"dias do outro regime; no compra e manutenção, os dias posicionados são os dias do regime. Com o "
            r"Ridge, o período é o da negociação (depois do burn-in de 252 previsões); a proxy é avaliada desde "
            r"a primeira data das previsões do Ridge. Sem dias posicionados, Sharpe indefinido (--).")


# ---------------------------------------------------------------------------
# Figuras (sem título gravado: a legenda fica no LaTeX)
# ---------------------------------------------------------------------------
def figuras(series, datas, raiz):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3})
    destino = os.path.join(raiz, "figs", "estrategias")
    trio = lambda v: [((v, A.Q_REF, "comprado"), A.AZUL, "Comprada/neutra"),  # noqa: E731
                      ((v, A.Q_REF, "vendido"), A.LARANJA, "Vendida/neutra"),
                      ((v, None, "compra_e_manutencao"), A.CINZA, "Compra e manutenção")]

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for chave, cor, rot in trio("ridge"):
        ax[0].plot(A._acumulado(series[chave]), color=cor, lw=2, label=rot)
        ax[1].plot(100 * A._drawdown(series[chave]), color=cor, lw=2)
    ax[0].axhline(1, color="black", lw=0.8)
    ax[0].set_ylabel("Valor acumulado (início = 1)")
    ax[1].set_ylabel("Drawdown (%)")
    for a in ax:
        a.yaxis.set_major_formatter(fmt_eixo)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(destino, "fig_principal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, direcao in enumerate(A.DIRECOES):
        for q, cor in zip(A.QUANTIS, [A.AZUL, A.LARANJA, A.VERDE, A.ROXO]):
            ax[k].plot(A._acumulado(series[("ridge", q, direcao)]), color=cor, lw=2, label=f"q{int(q * 100)}")
        ax[k].plot(A._acumulado(series[("ridge", None, "compra_e_manutencao")]), color=A.CINZA, lw=1.5,
                   label="Compra e manutenção")
        ax[k].axhline(1, color="black", lw=0.8)
        ax[k].text(0.02, 0.96, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, va="top")
        ax[k].tick_params(axis="x", rotation=30)
        ax[k].yaxis.set_major_formatter(fmt_eixo)
    ax[0].set_ylabel("Valor acumulado (início = 1)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(destino, "fig_quantis.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for chave, cor, rot in trio("proxy"):
        ax.plot(A._acumulado(series[chave]), color=cor, lw=2, label=rot)
    ax.axvline(datas["ridge"][0], color="black", lw=1, ls="--", label="Início da negociação com o Ridge")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylabel("Valor acumulado (início = 1)")
    ax.yaxis.set_major_formatter(fmt_eixo)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(destino, "fig_proxy.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default=ROOT)
    raiz = p.parse_args().raiz
    s, m = ler()
    series, datas = conferir(s, m)
    for sub in ("tables", "figs"):
        os.makedirs(os.path.join(raiz, sub, "estrategias"), exist_ok=True)
    tabelas(m, raiz)
    figuras(series, datas, raiz)
    print(f"Gravados: 4 tabelas em {os.path.join(raiz, 'tables', 'estrategias')} e 3 figuras em "
          f"{os.path.join(raiz, 'figs', 'estrategias')}")


if __name__ == "__main__":
    main()
