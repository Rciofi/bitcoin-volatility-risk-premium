# -*- coding: utf-8 -*-
"""Regenera todas as tabelas do Capitulo 7.

Melhorias implementadas (Fase 1):
  [P1] apply_transaction_costs(): cenarios 0, 5 e 10 bps nas Tabs 7.2 e 7.3
  [P2] expanding_quantile_signal(): encapsula logica de janela expansiva
  [P3] sortino_ratio(): adicionado em todas as tabelas (7.1 a 7.4)
  Tab 7.2b (auxiliar): tres cenarios de custo lado a lado
"""

import pandas as pd
import numpy as np
import os

os.makedirs('tables/estrategias', exist_ok=True)

ANN = 365        # Bitcoin: mercado 24/7
MIN_PERIODS = 252  # janela minima para limiar expansivo (elimina lookahead)

# Constante LaTeX para terminador de linha (evita f-string terminando com \\)
_NL = r"\\"


# ============================================================
# Funcoes auxiliares
# ============================================================

def perf(rets):
    """Retorno anualizado, volatilidade anualizada, Sharpe e MaxDD."""
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


def sortino_ratio(returns, target=0, periods_per_year=365):
    """[P3] Sortino ratio anualizado. Target padrao = 0."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess = returns - target / periods_per_year
    downside_returns = returns[returns < target / periods_per_year]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float((excess.mean() * periods_per_year) / downside_std)


def apply_transaction_costs(returns_series, position_series, cost_bps=10):
    """[P1] Aplica custo proporcional ao turnover diario.

    cost_bps: custo unilateral em pontos-base (10 bps = 0.10%).
    Custo so incide quando position(t) != position(t-1).
    """
    position_change = position_series.diff().abs().fillna(
        abs(position_series.iloc[0])
    )
    cost_per_day = position_change * (cost_bps / 10_000)
    return returns_series - cost_per_day


def expanding_quantile_signal(bvrp_series, quantile=0.80, min_obs=MIN_PERIODS):
    """[P2] Sinal binario com quantil em janela expansiva (vetorizado).

    Em cada t, o limiar usa APENAS dados de [0, t-1] (ex-ante).
    O sinal usa BVRP defasado em 1 dia.
    Burn-in: primeiros min_obs dias sem sinal (signal=0).

    Retorna: (signal Series, threshold Series)
    """
    th_series = bvrp_series.expanding(min_periods=min_obs).quantile(quantile)
    signal = (bvrp_series.shift(1) >= th_series.shift(1)).astype(float)
    signal = signal.fillna(0.0)
    signal.iloc[:min_obs] = 0.0
    return signal, th_series


def _fmt(val, d=4):
    """Formata numero para LaTeX; retorna '--' se NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return "{:.{}f}".format(val, d)


# ============================================================
# Dataset 1: BTC historico completo (referencia secundaria)
# ============================================================
df_btc = pd.read_csv('data/btc_prices.csv', parse_dates=['date'])
df_btc = df_btc.sort_values('date').reset_index(drop=True)
# Trunca no cutoff canonico da dissertacao (mesma data de vrp_with_targets.csv).
# btc_prices.csv se estende ate mai/2026; sem este corte a Tab 7.1 usaria ~2 meses
# que nenhum outro capitulo enxerga.
AMOSTRA_FIM_BTC = "2026-03-03"
df_btc = df_btc[df_btc['date'] <= AMOSTRA_FIM_BTC].reset_index(drop=True)
df_btc['ret'] = df_btc['close'].pct_change()
df_btc = df_btc.dropna(subset=['ret'])

r_bh, v_bh, s_bh, m_bh = perf(df_btc['ret'])
sortino_bh = sortino_ratio(df_btc['ret'])
print("BH historico completo (N={}): Ret={:.4f}, Vol={:.4f}, Sharpe={:.4f}, Sortino={}, MDD={:.4f}".format(
    len(df_btc), r_bh, v_bh, s_bh, _fmt(sortino_bh), m_bh))

# Tab 7.1: metricas do Buy & Hold (formato vertical, + Sortino)
tab1_rows = [
    r"\begin{tabular}{lr}",
    r"\toprule",
    r"M\'etrica & Valor \\",
    r"\midrule",
    "Retorno Anualizado & {:.4f} {}".format(r_bh, _NL),
    "Volatilidade Anualizada & {:.4f} {}".format(v_bh, _NL),
    "Sharpe Ratio & {:.4f} {}".format(s_bh, _NL),
    "Sortino Ratio & {} {}".format(_fmt(sortino_bh), _NL),
    "Max Drawdown & {:.4f} {}".format(m_bh, _NL),
    r"\bottomrule",
    r"\end{tabular}",
]
with open('tables/estrategias/tab_perf_buy_hold.tex', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(tab1_rows))
print("Tabela 1 salva: tables/estrategias/tab_perf_buy_hold.tex")


# ============================================================
# Dataset 2: BVRP (2021-2026)
# ============================================================
df = pd.read_csv('data/vrp_with_regimes.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['vrp_30d', 'ret'])
print("\nDataset BVRP: {} a {}, N={}".format(
    df['date'].min().date(), df['date'].max().date(), len(df)))

# [P2] Sinal q80 via expanding_quantile_signal encapsulada
sig80, q80_series = expanding_quantile_signal(df['vrp_30d'], quantile=0.80, min_obs=MIN_PERIODS)

n_nan = int(q80_series.isna().sum())
n_warmup = n_nan + 1  # +1 pelo shift(1) no sinal
q80_primeiro = float(q80_series.iloc[n_warmup]) if n_warmup < len(q80_series) else np.nan
q80_ultimo = float(q80_series.iloc[-1])
print("Limiar q80 expansivo: NaN iniciais={}, primeiro valido em {} = {:.4f}, ultimo = {:.4f}".format(
    n_nan, df['date'].iloc[n_warmup].date(), q80_primeiro, q80_ultimo))

# BH subperiodo: referencia principal (mesmo periodo operacional)
df_sub = df.iloc[n_warmup:].copy()
r_bh2, v_bh2, s_bh2, m_bh2 = perf(df_sub['ret'])
sortino_bh2 = sortino_ratio(df_sub['ret'])
print("BH subperiodo ({} -> {}, N={}): Sharpe={:.4f}, Sortino={}, MDD={:.4f}".format(
    df_sub['date'].min().date(), df_sub['date'].max().date(), len(df_sub),
    s_bh2, _fmt(sortino_bh2), m_bh2))


# ============================================================
# Tab 7.2 -- Estrategia q80% com cenarios de custo (0, 5, 10 bps)
# ============================================================
sr80 = sig80 * df['ret']

# Serie diaria (date, retorno bruto, retorno da estrategia q80 expansiva) --
# exportada em CSV para as figuras de retorno acumulado (regenerate_cap7_figs.py)
# reusarem em vez de recalcular o sinal com quantil fixo (look-ahead bias).
daily_series = pd.DataFrame({'date': df['date'], 'ret': df['ret'], 'sr_q80': sr80})

r80, v80, s80, m80 = perf(sr80)
sortino_80 = sortino_ratio(sr80)
ti80 = float(sig80.mean())
to80 = float(sig80.diff().abs().mean())
ops80 = int(sig80.diff().abs().sum())
turnover_anual_80 = to80 * ANN

print("\nEstrategia q80: Ret={:.4f}, Vol={:.4f}, Sharpe={:.4f}, Sortino={}, MDD={:.4f}, Ops={}, Turnover/ano={:.2f}".format(
    r80, v80, s80, _fmt(sortino_80), m80, ops80, turnover_anual_80))

# [P1] Cenarios de custo: 0, 5 e 10 bps
sharpe_cost80 = {}
mdd_cost80 = {}
for cost_bp in [0, 5, 10]:
    sr_net = apply_transaction_costs(sr80, sig80, cost_bps=cost_bp)
    r_c, v_c, s_c, m_c = perf(sr_net)
    sharpe_cost80[cost_bp] = s_c
    mdd_cost80[cost_bp] = m_c
    print("  q80 {}bps -> Sharpe={:.4f}, MDD={:.4f}".format(cost_bp, s_c, m_c))

# Tab 7.2: uma linha por estrategia
# Colunas: Estrategia | Ret.Anual | Vol.Anual | Sharpe(bruto) | Sharpe(5bps) | Sharpe(10bps) | MaxDD | Sortino(bruto) | %Tempo
row_q80 = "BVRP q80\\% & {:.4f} & {:.4f} & {} & {} & {} & {:.4f} & {} & {:.4f} {}".format(
    r80, v80,
    _fmt(sharpe_cost80[0]), _fmt(sharpe_cost80[5]), _fmt(sharpe_cost80[10]),
    m80, _fmt(sortino_80), ti80, _NL)

row_bh2 = "Buy \\& Hold & {:.4f} & {:.4f} & {:.4f} & -- & -- & {:.4f} & {} & 1.0000 {}".format(
    r_bh2, v_bh2, s_bh2, m_bh2, _fmt(sortino_bh2), _NL)

lines2 = [
    r"\begin{tabular}{lrrrrrrrr}",
    r"\toprule",
    r"Estrat\'egia & Ret.\ Anual & Vol.\ Anual & Sharpe & Sharpe & Sharpe & Max & Sortino & \%\ Tempo \\",
    r" &  &  & (bruto) & (5\,bps) & (10\,bps) & Drawdown & (bruto) &  \\",
    r"\midrule",
    row_q80,
    r"\midrule",
    row_bh2,
    r"\bottomrule",
    r"\end{tabular}",
]
with open('tables/estrategias/tab_perf_vrp_quantile.tex', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(lines2))
print("Tabela 2 salva: tables/estrategias/tab_perf_vrp_quantile.tex")

# Tab 7.2b -- tabela auxiliar: tres cenarios de custo (q80)
cost_rows_csv = []
lines2b = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Custo & Ret.\ Anual & Vol.\ Anual & Sharpe & Sortino & Max Drawdown & N.\ Opera\c{c}\~oes \\",
    r"\midrule",
]
for cost_bp in [0, 5, 10]:
    sr_net = apply_transaction_costs(sr80, sig80, cost_bps=cost_bp)
    r_c, v_c, s_c, m_c = perf(sr_net)
    sortino_c = sortino_ratio(sr_net)
    lines2b.append("{:d}\\,bps & {:.4f} & {:.4f} & {:.4f} & {} & {:.4f} & {:d} {}".format(
        cost_bp, r_c, v_c, s_c, _fmt(sortino_c), m_c, ops80, _NL))
    sortino_val = float(sortino_c) if not (isinstance(sortino_c, float) and np.isnan(sortino_c)) else None
    cost_rows_csv.append({
        'custos_bp': cost_bp,
        'cagr': round(r_c, 6),
        'vol_anualizada': round(v_c, 6),
        'sharpe': round(s_c, 6),
        'sortino': round(sortino_c, 6) if sortino_val is not None else np.nan,
        'mdd': round(m_c, 6),
        'n_operacoes': ops80,
    })
lines2b += [r"\bottomrule", r"\end{tabular}"]
with open('tables/estrategias/tab_custos.tex', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(lines2b))
print("Tabela 2b (custos) salva: tables/estrategias/tab_custos.tex")

pd.DataFrame(cost_rows_csv).to_csv('tables/estrategias/strategy_costs.csv', index=False)
print("CSV de custos salvo: tables/estrategias/strategy_costs.csv")


# ============================================================
# Tab 7.3 -- Multi-quantis (q60, q70, q80, q90) com custos e Sortino
# ============================================================
print("\nEstrategias por quantil:")
rows3 = []
for q in [0.60, 0.70, 0.80, 0.90]:
    sig_q, _ = expanding_quantile_signal(df['vrp_30d'], quantile=q, min_obs=MIN_PERIODS)
    sr_q = sig_q * df['ret']
    r_q, v_q, s_q, m_q = perf(sr_q)
    sortino_q = sortino_ratio(sr_q)
    to_q = float(sig_q.diff().abs().mean())
    ti_q = float(sig_q.mean())
    qval = int(q * 100)

    sharpe_q = {}
    for cost_bp in [0, 5, 10]:
        sr_net = apply_transaction_costs(sr_q, sig_q, cost_bps=cost_bp)
        _, _, s_c, _ = perf(sr_net)
        sharpe_q[cost_bp] = s_c

    q_label = "q{}\\%".format(qval)
    rows3.append((q_label, r_q, v_q, sharpe_q[0], sharpe_q[5], sharpe_q[10], m_q, sortino_q, to_q, ti_q))
    daily_series[f'sr_q{qval}'] = sr_q.values
    print("  q{}: Ret={:.4f}, Vol={:.4f}, Sharpe={:.4f}, Sortino={}, MDD={:.4f}, Tempo={:.4f}".format(
        qval, r_q, v_q, s_q, _fmt(sortino_q), m_q, ti_q))

lines3 = [
    r"\begin{tabular}{lrrrrrrrrr}",
    r"\toprule",
    r"Estrat\'egia & Ret.\ Anual & Vol.\ Anual & Sharpe & Sharpe & Sharpe & Max & Sortino & Turnover & \%\ Tempo \\",
    r" &  &  & (bruto) & (5\,bps) & (10\,bps) & Drawdown & (bruto) &  &  \\",
    r"\midrule",
]
for row in rows3:
    lines3.append("{} & {:.4f} & {:.4f} & {} & {} & {} & {:.4f} & {} & {:.4f} & {:.4f} {}".format(
        row[0], row[1], row[2],
        _fmt(row[3]), _fmt(row[4]), _fmt(row[5]),
        row[6], _fmt(row[7]), row[8], row[9], _NL))
lines3.append(r"\midrule")
lines3.append("Buy \\& Hold & {:.4f} & {:.4f} & {:.4f} & -- & -- & {:.4f} & {} & -- & 1.0000 {}".format(
    r_bh2, v_bh2, s_bh2, m_bh2, _fmt(sortino_bh2), _NL))
lines3 += [r"\bottomrule", r"\end{tabular}"]
with open('tables/estrategias/tab_perf_bvrp_multi_quantile.tex', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(lines3))
print("Tabela 3 salva: tables/estrategias/tab_perf_bvrp_multi_quantile.tex")

# CSV com as mesmas metricas, para o script de figuras (regenerate_cap7_figs.py)
# ler em vez de recalcular o sinal -- garante que heatmap e tabela venham da
# mesma fonte (evita divergencia como a do look-ahead bias corrigido em c02e22a).
pd.DataFrame(rows3, columns=[
    "quantil", "ret_anual", "vol_anual", "sharpe_bruto", "sharpe_5bps",
    "sharpe_10bps", "max_drawdown", "sortino", "turnover", "pct_tempo",
]).to_csv('tables/estrategias/perf_multi_quantile.csv', index=False)
print("CSV salvo: tables/estrategias/perf_multi_quantile.csv")


# ============================================================
# Tab 7.4 -- Regimes por tercis de RV_30d (+ Sortino)
# ============================================================
print("\nRegimes por RV (tercis):")
df['rv_regime'] = pd.qcut(df['rv_30d'], q=3, labels=['Baixo', 'Medio', 'Alto'])

rows4 = []
regime_key = {'Baixo': 'baixo', 'Medio': 'medio', 'Alto': 'alto'}
for regime, label in [('Baixo', 'Baixa RV'), ('Medio', 'Media RV'), ('Alto', 'Alta RV')]:
    mask = df['rv_regime'] == regime
    sig = ((df['vrp_30d'].shift(1) >= q80_series.shift(1)) & mask).astype(float).fillna(0.0)
    sig.iloc[:MIN_PERIODS] = 0.0
    sr = sig * df['ret']
    r, v, s, m = perf(sr)
    sortino_r = sortino_ratio(sr)
    ti = float(sig.mean())
    to = float(sig.diff().abs().mean())
    rows4.append((label, r, v, s, m, sortino_r, to, ti))
    daily_series[f'sr_regime_{regime_key[regime]}'] = sr.values
    print("  {}: Ret={:.4f}, Vol={:.4f}, Sharpe={:.4f}, Sortino={}, MDD={:.4f}, Tempo={:.4f}".format(
        regime, r, v, s, _fmt(sortino_r), m, ti))

lines4 = [
    r"\begin{tabular}{lrrrrrrrr}",
    r"\toprule",
    r" & Ret.\ Anual & Vol.\ Anual & Sharpe & Max & Sortino & Turnover & \%\ Tempo \\",
    r"Regime de RV &  &  &  & Drawdown &  &  &  \\",
    r"\midrule",
]
for row in rows4:
    lines4.append("{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {} & {:.4f} & {:.4f} {}".format(
        row[0], row[1], row[2], row[3], row[4], _fmt(row[5]), row[6], row[7], _NL))
lines4 += [r"\bottomrule", r"\end{tabular}"]
with open('tables/estrategias/tab_perf_bvrp_regimes.tex', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(lines4))
print("Tabela 4 salva: tables/estrategias/tab_perf_bvrp_regimes.tex")

# CSV com as mesmas metricas, para o script de figuras ler (ver nota acima).
pd.DataFrame(rows4, columns=[
    "regime", "ret_anual", "vol_anual", "sharpe", "max_drawdown",
    "sortino", "turnover", "pct_tempo",
]).to_csv('tables/estrategias/perf_regimes.csv', index=False)
print("CSV salvo: tables/estrategias/perf_regimes.csv")

# Serie diaria completa (date, ret, sr_q80, sr_q60/70/80/90, sr_regime_*) --
# fonte unica para as figuras de retorno acumulado (Fig 7.2, 7.3, 7.4, 7.6).
daily_series.to_csv('tables/estrategias/daily_returns.csv', index=False)
print("CSV salvo: tables/estrategias/daily_returns.csv")

print("\nTodas as tabelas do cap7 regeneradas com sucesso.")
print("\n--- RESUMO PASSO 5 ---")
print("Sortino q80 bruto:          {}".format(_fmt(sortino_80)))
print("Turnover medio anual q80:   {:.2f} operacoes/ano".format(turnover_anual_80))
print("Sharpe liquido 10bps q80:   {}".format(_fmt(sharpe_cost80[10])))
print("Limiar q80 primeiro dia:    {:.4f}".format(q80_primeiro))
print("Limiar q80 ultimo dia:      {:.4f}".format(q80_ultimo))
