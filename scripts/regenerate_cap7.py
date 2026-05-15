# -*- coding: utf-8 -*-
"""Regenera todas as tabelas do Capitulo 7 com dados atualizados e metodologia consistente."""

import pandas as pd
import numpy as np
import os

os.makedirs('tables/tab7', exist_ok=True)
os.makedirs('tables/cap7', exist_ok=True)  # [E6] para strategy_costs_cap7.csv

ANN = 365  # Bitcoin: mercado 24/7
MIN_PERIODS_Q80 = 252  # [E6] janela minima para limiar expansivo -- elimina lookahead


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
# Dataset 1: BTC historico completo (referencia secundaria)
# ============================================================
df_btc = pd.read_csv('data/btc_prices.csv', parse_dates=['date'])
df_btc = df_btc.sort_values('date').reset_index(drop=True)
df_btc['ret'] = df_btc['close'].pct_change()
# [E6] removido filtro <= '2024-12-31' -- usa serie historica completa (2017-2026)
df_btc = df_btc.dropna(subset=['ret'])

r, v, s, m = perf(df_btc['ret'])
print(f"BH historico completo (N={len(df_btc)}): Return={r:.4f}, Vol={v:.4f}, Sharpe={s:.4f}, MDD={m:.4f}")

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
# Dataset 2: BVRP (2021-2026)
# ============================================================
df = pd.read_csv('data/vrp_with_regimes.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['vrp_30d', 'ret'])
print(f"\nDataset BVRP: {df['date'].min().date()} a {df['date'].max().date()}, N={len(df)}")

# [E6] Limiar q80 via janela expansiva (min_periods=252) -- elimina lookahead
# O sinal usa vrp_30d_{t-1} > q80_{t-1}; o shift(1) garante que
# o limiar do dia t so usa dados ate t-1.
q80_series = df['vrp_30d'].expanding(min_periods=MIN_PERIODS_Q80).quantile(0.80)
n_warmup = int(q80_series.isna().sum()) + 1  # +1 pelo shift(1) aplicado ao sinal
print(f"Limiar q80 expansivo: NaN iniciais={n_warmup - 1}, "
      f"primeiro limiar em {df['date'].iloc[n_warmup - 1].date()}, "
      f"valor final={float(q80_series.iloc[-1]):.4f}")

# [E6] BH subperiodo: referencia principal -- mesmo periodo operacional da estrategia
df_sub = df.iloc[n_warmup:].copy()
r_bh2, v_bh2, s_bh2, m_bh2 = perf(df_sub['ret'])
print(f"BH subperiodo ({df_sub['date'].min().date()} -> {df_sub['date'].max().date()}, "
      f"N={len(df_sub)}): Sharpe={s_bh2:.4f}, MDD={m_bh2:.4f}")

# ============================================================
# Tab7_2: Estrategia condicional q80
# ============================================================
# [E6] sinal usa q80_series expansivo com shift para evitar lookahead
sig80 = (df['vrp_30d'].shift(1) > q80_series.shift(1)).astype(float).fillna(0.0)
sr80 = sig80 * df['ret']
r2, v2, s2, m2 = perf(sr80)
ti2 = float(sig80.mean())
to2 = float(sig80.diff().abs().mean())
ops2 = int(sig80.diff().abs().sum())
print(f"\nEstrategia q80 exp: Return={r2:.4f}, Vol={v2:.4f}, Sharpe={s2:.4f}, MDD={m2:.4f}, "
      f"Time={ti2:.4f}, Ops={ops2}")

# [E6] tabela passa de 1 coluna para 2: estrategia + BH subperiodo
tab2_content = (
    r"\begin{tabular}{lrr}" + "\n"
    r"\toprule" + "\n"
    r" & BVRP Quantil 80\% & Buy \& Hold \\" + "\n"
    r"\midrule" + "\n"
    f"Retorno Anualizado & {r2:.4f} & {r_bh2:.4f} \\\\\n"
    f"Volatilidade Anualizada & {v2:.4f} & {v_bh2:.4f} \\\\\n"
    f"Sharpe Ratio & {s2:.4f} & {s_bh2:.4f} \\\\\n"
    f"Max Drawdown & {m2:.4f} & {m_bh2:.4f} \\\\\n"
    r"\% Tempo Investido & " + f"{ti2:.4f} & 1.0000 \\\\\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}"
)
with open('tables/tab7/tab7_2_perf_vrp_quantile.tex', 'w', encoding='utf-8') as f:
    f.write(tab2_content)
print("Tabela 2 salva.")

# ============================================================
# [E6 NOVO] Analise de sensibilidade a custos de transacao
# ============================================================
print("\nSensibilidade a custos (estrategia q80 expansiva):")
cost_rows = []
for cost_bp in [0, 10, 30]:
    cost = cost_bp / 10000
    cost_series = (sig80.diff().abs() * cost).fillna(0.0)
    sr_net = sr80 - cost_series
    r_c, v_c, s_c, m_c = perf(sr_net)
    cost_rows.append({
        'custos_bp': cost_bp,
        'cagr': round(r_c, 6),
        'vol_anualizada': round(v_c, 6),
        'sharpe': round(s_c, 6),
        'mdd': round(m_c, 6),
        'n_operacoes': ops2,
    })
    print(f"  {cost_bp}bp -> Sharpe={s_c:.4f}, CAGR={r_c:.4f}, MDD={m_c:.4f}")

df_costs = pd.DataFrame(cost_rows)
df_costs.to_csv('tables/cap7/strategy_costs_cap7.csv', index=False)
print("Sensibilidade a custos salva: tables/cap7/strategy_costs_cap7.csv")

# ============================================================
# Tab7_3: Multi-quantis
# ============================================================
print("\nEstrategias por quantil:")
rows3 = []
for q in [0.60, 0.70, 0.80, 0.90]:
    # [E6] expanding window para todos os quantis -- consistencia metodologica
    th_series = df['vrp_30d'].expanding(min_periods=MIN_PERIODS_Q80).quantile(q)
    sig = (df['vrp_30d'].shift(1) > th_series.shift(1)).astype(float).fillna(0.0)
    sr = sig * df['ret']
    r, v, s, m = perf(sr)
    ti = float(sig.mean())
    to = float(sig.diff().abs().mean())
    qval = int(q * 100)
    q_label = "q" + str(qval) + r"\%"
    rows3.append((q_label, r, v, s, m, to, ti))
    print(f"  q{qval}: Return={r:.4f}, Vol={v:.4f}, Sharpe={s:.4f}, MDD={m:.4f}, Time={ti:.4f}")

lines3 = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Estrat\'egia & Retorno Anual & Volatilidade Anual & Sharpe Ratio & Max Drawdown & Turnover & \% Tempo Investido \\",
    r"\midrule",
]
for row in rows3:
    lines3.append(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.4f} & {row[6]:.4f} \\\\")
# [E6] linha BH subperiodo como referencia contemporanea
lines3.append(r"\midrule")
lines3.append(f"Buy \\& Hold & {r_bh2:.4f} & {v_bh2:.4f} & {s_bh2:.4f} & {m_bh2:.4f} & -- & 1.0000 \\\\")
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
    # [E6] usa q80_series expansivo -- consistente com Tab7_2
    sig = ((df['vrp_30d'].shift(1) > q80_series.shift(1)) & mask).astype(float).fillna(0.0)
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
