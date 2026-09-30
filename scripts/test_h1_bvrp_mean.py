"""
Teste da hipotese H1 (existencia do BVRP): a media do BVRP e diferente de zero?

T2 do plano de revisao (set/2026). Compara, lado a lado, na amostra de
referencia (data/vrp_with_targets.csv, N = 1.806):
  - bvrp_30d_fut: RV(t+1 a t+30) - IV_t   (definicao prospectiva, principal)
  - vrp_30d:      RV(t-29 a t) - IV_t     (proxy retrospectiva)

Testes:
  1. Media por MQO numa constante, erro-padrao HAC (Newey-West) com
     maxlags = h+1 = 31 (principal; h = 30 pela sobreposicao das janelas, +1
     conforme o T3). Sensibilidade: maxlags = 7 (regra automatica do pacote,
     floor(4(T/100)^(2/9))), 30 (valor anterior), 60 e 90.
     Convencao do statsmodels: nucleo de Bartlett, pesos w_j = 1 - j/(L+1),
     L = maior defasagem incluida (Newey e West, 1987); sem correcao de
     amostra pequena; p-valor e IC pela distribuicao NORMAL.
  2. Robustez: amostra sem sobreposicao (uma observacao a cada 30 dias),
     teste t simples com distribuicao t de Student com N-1 graus de
     liberdade. Principal: comeca na primeira data da amostra; tambem roda
     os 30 pontos de partida possiveis e resume a estatistica t.
  3. Diagnosticos de persistencia: ACF (BVRP nas duas definicoes, RV nas
     duas janelas e IV) e Ljung-Box (BVRP nas duas definicoes).

Nao faz parte do pipeline de regeneracao dos capitulos. NAO grava em tables/:
todas as saidas vao para --out-dir (csv e tex).

Uso:  python scripts/test_h1_bvrp_mean.py --out-dir outputs/T2
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "vrp_with_targets.csv")

DEFS = {
    "bvrp_30d_fut": "BVRP prospectivo",
    "vrp_30d":      "BVRP retrospectivo (proxy)",
}
H = 30                              # janela da RV (dias)
MAXLAGS_PRINCIPAL = H + 1           # T3: h+1
MAXLAGS_SENS = [7, 30, 31, 60, 90]  # 7 = regra automatica do pacote para T = 1.806
PASSO = H                           # amostra sem sobreposicao: uma obs a cada 30 dias
LAGS_ACF = [1, 5, 10, 20, 25, 29, 30, 31, 35, 40, 60]
LAGS_LB = [1, 5, 10, 20, 30]
ACF_SERIES = {
    "bvrp_30d_fut": "BVRP prospectivo",
    "vrp_30d":      "BVRP retrospectivo (proxy)",
    "rv_30d_fut":   "RV 30d prospectiva",
    "rv_30d":       "RV 30d retrospectiva",
    "iv_30d":       "IV 30d",
}
Z975 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------
def teste_hac(x, maxlags):
    """Media por MQO numa constante; EP HAC (Bartlett), p e IC pela normal."""
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    assert res.use_t is False  # convencao do pacote: inferencia pela normal
    m, se = float(res.params[0]), float(res.bse[0])
    return {"N": len(x), "media": m, "ep": se, "t": float(res.tvalues[0]),
            "p": float(res.pvalues[0]), "ic_inf": m - Z975 * se, "ic_sup": m + Z975 * se}


def teste_t_simples(x):
    """Teste t de uma amostra, EP = s/sqrt(N), distribuicao t com N-1 g.l."""
    n = len(x)
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n))
    t = m / se
    q = stats.t.ppf(0.975, n - 1)
    return {"N": n, "media": m, "ep": se, "t": t, "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "ic_inf": m - q * se, "ic_sup": m + q * se}


# ---------------------------------------------------------------------------
# Formatacao LaTeX (virgula decimal; {,} em modo matematico)
# ---------------------------------------------------------------------------
def fm(x, d):
    return f"{x:.{d}f}".replace(".", "{,}")


def fp(p):
    """p-valor em modo matematico: decimal se >= 0,001; senao notacao cientifica."""
    if p >= 0.001:
        return f"${fm(p, 3)}$"
    if p == 0:
        return r"$< 10^{-300}$"
    e = int(np.floor(np.log10(p)))
    return rf"${fm(p / 10**e, 2)} \times 10^{{{e}}}$"


def fic(r):
    return f"$[{fm(r['ic_inf'], 2)};\\ {fm(r['ic_sup'], 2)}]$"


def fmt_n(n):
    return f"{n:,}".replace(",", "{.}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T2"))
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Amostra de referencia: N={len(df)}  {df['date'].min().date()} a {df['date'].max().date()}")

    # 1) HAC: principal + sensibilidade as defasagens
    rows_hac = []
    for col, label in DEFS.items():
        x = df[col].dropna().to_numpy()
        for L in MAXLAGS_SENS:
            rows_hac.append({"definicao": col, "rotulo": label, "maxlags": L,
                             "principal": L == MAXLAGS_PRINCIPAL, **teste_hac(x, L)})
    hac = pd.DataFrame(rows_hac)
    hac.to_csv(os.path.join(out_dir, "h1_hac.csv"), index=False)

    # 2) Amostra sem sobreposicao: 30 pontos de partida
    rows_ns = []
    for col, label in DEFS.items():
        s = df[["date", col]].dropna().reset_index(drop=True)
        for k in range(PASSO):
            sub = s.iloc[k::PASSO]
            rows_ns.append({"definicao": col, "rotulo": label, "inicio": k + 1,
                            "primeira_data": sub["date"].iloc[0].date(),
                            **teste_t_simples(sub[col].to_numpy())})
    ns = pd.DataFrame(rows_ns)
    ns.to_csv(os.path.join(out_dir, "h1_nao_sobreposto.csv"), index=False)
    resumo_ns = (ns.groupby("definicao", sort=False)
                   .agg(t_min=("t", "min"), t_mediana=("t", "median"), t_max=("t", "max"),
                        p_max=("p", "max"), N_min=("N", "min"), N_max=("N", "max"))
                   .reset_index())
    resumo_ns.to_csv(os.path.join(out_dir, "h1_nao_sobreposto_resumo.csv"), index=False)

    # 3) Diagnosticos de persistencia
    rows_acf = []
    for col, label in ACF_SERIES.items():
        s = df[col].dropna()
        a = acf(s, nlags=max(LAGS_ACF), fft=True)
        rows_acf.append({"serie": col, "rotulo": label, **{f"rho_{l}": a[l] for l in LAGS_ACF}})
    pd.DataFrame(rows_acf).to_csv(os.path.join(out_dir, "h1_acf.csv"), index=False)
    lb = []
    for col in DEFS:
        t = acorr_ljungbox(df[col].dropna(), lags=LAGS_LB, return_df=True)
        t.insert(0, "defasagem", t.index)
        t.insert(0, "definicao", col)
        lb.append(t)
    pd.concat(lb).to_csv(os.path.join(out_dir, "h1_ljungbox.csv"), index=False)

    # ---------------------------------------------------------------------
    # Tabela principal (tex)
    # ---------------------------------------------------------------------
    prin = {c: hac[(hac.definicao == c) & hac.principal].iloc[0] for c in DEFS}
    ns1 = {c: ns[(ns.definicao == c) & (ns.inicio == 1)].iloc[0] for c in DEFS}
    rs = {c: resumo_ns[resumo_ns.definicao == c].iloc[0] for c in DEFS}
    cols = list(DEFS)

    def linha(rotulo, f):
        return rotulo + " & " + " & ".join(f(c) for c in cols) + r" \\"

    periodo = f"{df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Teste de H1 --- média do BVRP nas definições prospectiva e retrospectiva (proxy).}",
        r"\label{tab:h1-teste-media}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & " + " & ".join(DEFS[c] for c in cols) + r" \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Amostra completa, erro-padrão HAC (Newey--West, "
        + f"{MAXLAGS_PRINCIPAL}" + r" defasagens)}} \\",
        linha("N", lambda c: f"${fmt_n(int(prin[c]['N']))}$"),
        linha("Média (p.p.)", lambda c: f"${fm(prin[c]['media'], 2)}$"),
        linha("Erro-padrão HAC", lambda c: f"${fm(prin[c]['ep'], 3)}$"),
        linha("Estatística $t$", lambda c: f"${fm(prin[c]['t'], 2)}$"),
        linha("Valor-$p$", lambda c: fp(prin[c]["p"])),
        linha(r"IC de 95\%", lambda c: fic(prin[c])),
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Amostra sem sobreposição (uma observação a cada "
        + f"{PASSO}" + r" dias), teste $t$ simples}} \\",
        linha("N", lambda c: f"${int(ns1[c]['N'])}$"),
        linha("Média (p.p.)", lambda c: f"${fm(ns1[c]['media'], 2)}$"),
        linha("Erro-padrão", lambda c: f"${fm(ns1[c]['ep'], 3)}$"),
        linha("Estatística $t$", lambda c: f"${fm(ns1[c]['t'], 2)}$"),
        linha("Valor-$p$", lambda c: fp(ns1[c]["p"])),
        linha(r"IC de 95\%", lambda c: fic(ns1[c])),
        linha(r"$t$ nos " + f"{PASSO}" + r" pontos de partida (mín.; mediana; máx.)",
              lambda c: f"${fm(rs[c]['t_min'], 2)};\\ {fm(rs[c]['t_mediana'], 2)};\\ {fm(rs[c]['t_max'], 2)}$"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize\textit{Nota}: período " + periodo + ". "
        r"BVRP prospectivo: $\text{RV}_{t+1:t+30} - \text{IV}_t$; retrospectivo (proxy): "
        r"$\text{RV}_{t-29:t} - \text{IV}_t$. "
        r"Painel superior: média estimada por MQO numa constante; erro-padrão HAC de Newey--West "
        r"(núcleo de Bartlett, pesos $1 - j/(L+1)$, $L = h+1 = " + f"{MAXLAGS_PRINCIPAL}" + r"$), "
        r"sem correção de amostra pequena; valor-$p$ e intervalo de confiança pela distribuição normal. "
        r"Painel inferior: observações espaçadas em " + f"{PASSO}" + r" dias, a partir da primeira data "
        r"da amostra (janelas da RV sem sobreposição); valor-$p$ e intervalo de confiança pela "
        r"distribuição $t$ de Student com $N-1$ graus de liberdade. A última linha resume a "
        r"estatística $t$ nos " + f"{PASSO}" + r" pontos de partida possíveis.",
        r"\end{table}",
    ]
    with open(os.path.join(out_dir, "tab_h1_teste_media.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------------
    # Tabela de sensibilidade as defasagens (tex)
    # ---------------------------------------------------------------------
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Teste de H1 --- sensibilidade do erro-padrão HAC ao número de defasagens.}",
        r"\label{tab:h1-sensibilidade-defasagens}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{" + DEFS[cols[0]] + r"} & \multicolumn{3}{c}{" + DEFS[cols[1]] + r"} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Defasagens ($L$) & EP & $t$ & Valor-$p$ & EP & $t$ & Valor-$p$ \\",
        r"\midrule",
    ]
    for L in MAXLAGS_SENS:
        cel = []
        for c in cols:
            r = hac[(hac.definicao == c) & (hac.maxlags == L)].iloc[0]
            cel += [f"${fm(r['ep'], 3)}$", f"${fm(r['t'], 2)}$", fp(r["p"])]
        marca = r"$^{a}$" if L == MAXLAGS_PRINCIPAL else (r"$^{b}$" if L == 7 else "")
        lines.append(f"{L}{marca} & " + " & ".join(cel) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize\textit{Nota}: N = " + f"${fmt_n(len(df))}$" + r"; médias de "
        f"${fm(prin[cols[0]]['media'], 2)}$ e ${fm(prin[cols[1]]['media'], 2)}$" + r" p.p. "
        r"Núcleo de Bartlett, sem correção de amostra pequena, valor-$p$ pela normal. "
        r"$^{a}$~Especificação principal ($h+1$). $^{b}$~Regra automática do statsmodels, "
        r"$\lfloor 4(T/100)^{2/9} \rfloor$. 30: valor usado na versão anterior.",
        r"\end{table}",
    ]
    with open(os.path.join(out_dir, "tab_h1_sensibilidade_defasagens.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------------
    # Console
    # ---------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n=== HAC (principal: maxlags = %d) ===" % MAXLAGS_PRINCIPAL)
    print(hac.drop(columns=["rotulo"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Sem sobreposicao: inicio na primeira data ===")
    print(ns[ns.inicio == 1].drop(columns=["rotulo"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Sem sobreposicao: 30 pontos de partida ===")
    print(resumo_ns.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== ACF ===")
    print(pd.read_csv(os.path.join(out_dir, "h1_acf.csv")).drop(columns=["rotulo"]).round(3).to_string(index=False))
    print(f"\nSaidas em: {out_dir}")


if __name__ == "__main__":
    main()
