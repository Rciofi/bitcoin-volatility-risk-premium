"""
Script isolado de diagnostico -- NAO faz parte do pipeline de regeneracao
dos capitulos. Testa a hipotese H1 (existencia do BVRP) sobre a media
amostral, comparando teste t simples vs. erro-padrao HAC (Newey-West),
e investiga a persistencia (ACF / Ljung-Box) do BVRP, RV e IV.

Nao gera figura nem tabela -- so imprime no console.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA_PATH = "data/vrp_with_targets.csv"
LAGS_ACF = [1, 5, 10, 20, 25, 30, 35, 40]
LAGS_LB = [1, 5, 10, 20, 30]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

bvrp = df["vrp_30d"].dropna()
rv = df["rv_30d"].dropna()
iv = df["iv_30d"].dropna()

print("=" * 70)
print("H1 -- BVRP: estatisticas descritivas")
print("=" * 70)
n = len(bvrp)
mean = bvrp.mean()
std = bvrp.std()
print(f"N = {n}")
print(f"Media = {mean:.4f}")
print(f"Desvio-padrao = {std:.4f}")

print()
print("=" * 70)
print("Teste t SIMPLES (scipy ttest_1samp, H0: media = 0)")
print("=" * 70)
t_simple, p_simple = stats.ttest_1samp(bvrp, popmean=0.0)
se_simple = std / np.sqrt(n)
print(f"Erro-padrao (naive, N efetivo = N) = {se_simple:.4f}")
print(f"t = {t_simple:.4f}")
print(f"p = {p_simple:.6e}")

print()
print("=" * 70)
print("Teste t com erro-padrao HAC (Newey-West, maxlags=30)")
print("regressao: vrp_30d ~ const, cov_type='HAC', cov_kwds={'maxlags': 30}")
print("=" * 70)
X = np.ones((len(bvrp), 1))
model = sm.OLS(bvrp.values, X)
result = model.fit(cov_type="HAC", cov_kwds={"maxlags": 30})
coef = result.params[0]
se_hac = result.bse[0]
t_hac = result.tvalues[0]
p_hac = result.pvalues[0]
print(f"Coeficiente (media) = {coef:.4f}")
print(f"Erro-padrao HAC = {se_hac:.4f}")
print(f"t (HAC) = {t_hac:.4f}")
print(f"p (HAC) = {p_hac:.6e}")

print()
print(f"Razao SE_HAC / SE_naive = {se_hac / se_simple:.4f}")

print()
print("=" * 70)
print("ACF do BVRP (vrp_30d)")
print("=" * 70)
acf_bvrp = acf(bvrp, nlags=max(LAGS_ACF), fft=True)
for lag in LAGS_ACF:
    print(f"  rho_{lag:2d} = {acf_bvrp[lag]:.4f}")

print()
print("=" * 70)
print("Ljung-Box (BVRP)")
print("=" * 70)
lb = acorr_ljungbox(bvrp, lags=LAGS_LB, return_df=True)
print(lb)

print()
print("=" * 70)
print("ACF da RV (rv_30d) -- separada")
print("=" * 70)
acf_rv = acf(rv, nlags=max(LAGS_ACF), fft=True)
for lag in LAGS_ACF:
    print(f"  rho_{lag:2d} = {acf_rv[lag]:.4f}")

print()
print("=" * 70)
print("ACF da IV (iv_30d) -- separada")
print("=" * 70)
acf_iv = acf(iv, nlags=max(LAGS_ACF), fft=True)
for lag in LAGS_ACF:
    print(f"  rho_{lag:2d} = {acf_iv[lag]:.4f}")

print()
print("=" * 70)
print("Escrevendo tabela: tables/cap5/tab_cap5_h1_teste_media.tex")
print("=" * 70)
p_hac_exp = int(np.floor(np.log10(p_hac)))
p_hac_mantissa = p_hac / (10 ** p_hac_exp)
p_hac_latex = r"${:.2f} \times 10^{{{}}}$".format(p_hac_mantissa, p_hac_exp)

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Teste de H1 --- média amostral do BVRP, com erro-padrão HAC (Newey--West, 30 defasagens).}",
    r"\label{tab:cap5-h1-teste-media}",
    r"\begin{tabular}{lr}",
    r"\toprule",
    r"Estatística & Valor \\",
    r"\midrule",
    "N & {} \\\\".format(n),
    "Média (p.p.) & {:.4f} \\\\".format(mean),
    "Desvio-padrão & {:.4f} \\\\".format(std),
    "Erro-padrão HAC (30 defasagens) & {:.4f} \\\\".format(se_hac),
    "Estatística $t$ & {:.4f} \\\\".format(t_hac),
    "Valor-$p$ & {} \\\\".format(p_hac_latex),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
table_tex = "\n".join(lines) + "\n"
with open("tables/cap5/tab_cap5_h1_teste_media.tex", "w", encoding="utf-8") as fh:
    fh.write(table_tex)
print(table_tex)

print("=" * 70)
print("FIM")
print("=" * 70)
