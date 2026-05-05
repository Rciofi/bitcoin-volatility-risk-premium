# -*- coding: utf-8 -*-
"""Regenera todas as tabelas do Capítulo 7 com dados atualizados e metodologia consistente."""

import pandas as pd
import numpy as np
import os

os.makedirs('tables/tab7', exist_ok=True)

ANN = 365  # Bitcoin: mercado 24/7


def perf(rets):
    rets = rets.dropna()
    n = len(rets)
    if n == 0 or rets.std() == 0:
        return 0.0, 0.0, 0.0, 0.0
    ret_a = (1 + rets).prod() ** (ANN / n) - 1
    vol_a = rets.std() * np.sqrt(ANN)
    sharpe = ret_a / vol_a
    cum = (1 + rets).cumprod()
    mdd = float(((cum / cum.cummax()) - 1).min())
    return float(ret_a), float(vol_a), float(sharpe), mdd


# ============================================================
# Dataset 1: BTC historico completo (benchmark Buy-and-Hold)
# ============================================================
df_btc = pd.read_csv('data/btc_prices.csv', parse_dates=['date'])
df_btc = df_btc.sort_values('date').reset_index(drop=True)
df_btc['ret'] = df_btc['close'].pct_change()
df_btc = df_btc[df_btc['date'] <= '2024-12-31'].dropna(subset=['ret'])

r, v, s, m = perf(df_btc['ret'])
print(f"BH 2017-2024 (N={len(df_btc)}): Return={r:.4f}, Vol={v:.4f}, Sharpe={s:.4f}, MDD={m:.4f}")

tab1_content = (
    r"\begin{tabular}{lr}" + "\n"
    r"\toprule" + "\n"
    r" & Valor \\" + "\n"
    r"M\'etrica &  \\" + "\n"
    r"\midrule" + "\n"
    f"Retorno Anualizado & {r:.4f} \\\\\n"
    f"Volatilidade Anualizada & {v:.4f} \\\\\n"
    f"Sharpe Ratio & {s:.4f} \\\\\n"
    f"Max Drawdown & {m:.4f} \\\\\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}"
)
with open('tables/tab7/tab7_1_perf_buy_hold.tex', 'w', encoding='utf-8') as f:
    f.write(tab1_content)
print("Tabela 1 salva: tables/tab7/tab7_1_perf_buy_hold.tex")


# ============================================================
# Dataset 2: BVRP (consistente com cap5/cap6: 2023-02 a 2024-12)
# ============================================================
df = pd.read_csv('data/vrp_with_regimes.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['vrp_30d', 'ret'])
print(f"\nDataset BVRP: {df['date'].min().date()} a {df['date'].max().date()}, N={len(df)}")

# Limiar q80 fixado no periodo amostral
q80_th = df['vrp_30d'].quantile(0.80)
print(f"Limiar BVRP q80: {q80_th:.3f}")

# BH no mesmo periodo
r_bh2, v_bh2, s_bh2, m_bh2 = perf(df['ret'])
print(f"BH 2023-2024: Return={r_bh2:.4f}, Vol={v_bh2:.4f}, Sharpe={s_bh2:.4f}, MDD={m_bh2:.4f}")

# ============================================================
# Tab7_2: Estrategia condicional q80
# ============================================================
sig80 = (df['vrp_30d'].shift(1) > q80_th).astype(float)
sr80 = sig80 * df['ret']
r2, v2, s2, m2 = perf(sr80)
ti2 = float(sig80.mean())
to2 = float(sig80.diff().abs().mean())
print(f"\nEstrategia q80: Return={r2:.4f}, Vol={v2:.4f}, Sharpe={s2:.4f}, MDD={m2:.4f}, Time={ti2:.4f}")

tab2_content = (
    r"\begin{tabular}{lr}" + "\n"
    r"\toprule" + "\n"
    r" & BVRP Quantil 80\% \\" + "\n"
    r"\midrule" + "\n"
    f"Retorno Anualizado & {r2:.4f} \\\\\n"
    f"Volatilidade Anualizada & {v2:.4f} \\\\\n"
    f"Sharpe Ratio & {s2:.4f} \\\\\n"
    f"Max Drawdown & {m2:.4f} \\\\\n"
    r"\% Tempo Investido & " + f"{ti2:.4f} \\\\\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}"
)
with open('tables/tab7/tab7_2_perf_vrp_quantile.tex', 'w', encoding='utf-8') as f:
    f.write(tab2_content)
print("Tabela 2 salva.")

# ============================================================
# Tab7_3: Multi-quantis
# ============================================================
print("\nEstrategias por quantil:")
rows3 = []
for q in [0.60, 0.70, 0.80, 0.90]:
    th = df['vrp_30d'].quantile(q)
    sig = (df['vrp_30d'].shift(1) > th).astype(float)
    sr = sig * df['ret']
    r, v, s, m = perf(sr)
    ti = float(sig.mean())
    to = float(sig.diff().abs().mean())
    rows3.append((f"q{int(q*100)}\\%", r, v, s, m, to, ti))
    print(f"  q{int(q*100)}: Return={r:.4f}, Vol={v:.4f}, Sharpe={s:.4f}, MDD={m:.4f}, Time={ti:.4f}")

lines3 = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Quantil & Retorno Anual & Volatilidade Anual & Sharpe Ratio & Max Drawdown & Turnover & \% Tempo Investido \\",
    r"\midrule",
]
for row in rows3:
    lines3.append(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.4f} & {row[6]:.4f} \\\\")
lines3 += [r"\bottomrule", r"\end{tabular}"]

with open('tables/tab7/tab7_3_perf_bvrp_multi_quantile.tex', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines3))
print("Tabela 3 salva.")

# ============================================================
# Tab7_4: Regimes por tercis de RV_30d
# ============================================================
print("\nRegimes por RV (tercis):")
df['rv_regime'] = pd.qcut(df['rv_30d'], q=3, labels=['Baixo', 'Medio', 'Alto'])
rv_stats = df.groupby('rv_regime')['rv_30d'].agg(['min', 'mean', 'max'])
print(rv_stats)
vrp_stats = df.groupby('rv_regime')['vrp_30d'].agg(['mean'])
print(vrp_stats)

rows4 = []
for regime, label in [('Baixo', 'Baixa RV'), ('Medio', 'Media RV'), ('Alto', 'Alta RV')]:
    mask = df['rv_regime'] == regime
    sig = ((df['vrp_30d'].shift(1) > q80_th) & mask).astype(float)
    sr = sig * df['ret']
    r, v, s, m = perf(sr)
    ti = float(sig.mean())
    to = float(sig.diff().abs().mean())
    rows4.append((label, r, v, s, m, to, ti))
    print(f"  {regime}: Return={r:.4f}, Vol={v:.4f}, Sharpe={s:.4f}, MDD={m:.4f}, Time={ti:.4f}")

lines4 = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r" & Retorno Anual & Volatilidade Anual & Sharpe Ratio & Max Drawdown & Turnover & \% Tempo Investido \\",
    r"Regime de RV &  &  &  &  &  &  \\",
    r"\midrule",
]
for row in rows4:
    lines4.append(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.4f} & {row[6]:.4f} \\\\")
lines4 += [r"\bottomrule", r"\end{tabular}"]

with open('tables/tab7/tab7_4_perf_bvrp_regimes.tex', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines4))
print("Tabela 4 salva.")

print("\nTodas as tabelas do cap7 regeneradas com sucesso.")
