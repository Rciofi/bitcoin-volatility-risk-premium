# -*- coding: utf-8 -*-
"""Regenera todas as figuras do Capitulo 7 com dados e metodologia consistentes."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

os.makedirs('archive/figs/estrategias', exist_ok=True)

# ---- Estetica global ----
FIGSIZE_WIDE  = (10, 4.5)
FIGSIZE_COMP  = (10, 5)
FIGSIZE_HEAT  = (9, 3.5)
DPI           = 150
ANN           = 365
COLOR_BH      = '#2c7bb6'
COLOR_Q80     = '#d7191c'
COLOR_PALETTE = ['#2c7bb6', '#fdae61', '#d7191c', '#1a9641']
REGIME_COLORS = {'Baixo': '#2c7bb6', 'Medio': '#fdae61', 'Alto': '#d7191c'}

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': DPI,
})


def perf(rets):
    rets = rets.dropna()
    n = len(rets)
    if n == 0 or rets.std() == 0:
        return dict(ret=0, vol=0, sharpe=0, mdd=0, n=0)
    ret_a = (1 + rets).prod() ** (ANN / n) - 1
    vol_a = rets.std() * np.sqrt(ANN)
    sharpe = ret_a / vol_a
    cum = (1 + rets).cumprod()
    mdd = float(((cum / cum.cummax()) - 1).min())
    return dict(ret=float(ret_a), vol=float(vol_a), sharpe=float(sharpe), mdd=mdd, n=n)


def fmt_pct(ax, axis='y'):
    fmt = mticker.FuncFormatter(lambda x, _: f'{x:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ====================================================================
# Dados
# ====================================================================
# BTC historico completo (benchmark)
df_btc = pd.read_csv('data/btc_prices.csv', parse_dates=['date'])
df_btc = df_btc.sort_values('date').reset_index(drop=True)
df_btc['ret'] = df_btc['close'].pct_change()
df_btc = df_btc[df_btc['date'] <= '2026-03-03'].dropna(subset=['ret'])

# Dataset BVRP (periodo consistente com cap5/cap6)
df = pd.read_csv('data/vrp_with_regimes.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['vrp_30d', 'ret'])

# Regimes por tercis de RV_30d
df['rv_regime'] = pd.qcut(df['rv_30d'], q=3, labels=['Baixo', 'Medio', 'Alto'])

# Series diarias de retorno da estrategia (sinal expansivo, mesma fonte que
# gera as Tabelas 7.2-7.4) -- substitui o quantil fixo com look-ahead bias
# (q80_th = df['vrp_30d'].quantile(0.80)) usado anteriormente nas figuras de
# retorno acumulado. Ver regenerate_cap7.py para a definicao do sinal.
df_sig = pd.read_csv('archive/tables/estrategias/daily_returns.csv', parse_dates=['date'])
df = df.merge(df_sig.drop(columns=['ret']), on='date', how='left')

# Periodo real plotado (eixo X das figuras de retorno acumulado do BVRP) --
# usado nos titulos em vez de datas hardcoded, que ficavam fosseis quando a
# amostra mudava (ex.: "fev 2023 - dez 2024" nao correspondia ao range real).
_MESES_ABREV_FIG = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun',
                     'jul', 'ago', 'set', 'out', 'nov', 'dez']


def _fmt_periodo_fig(d):
    return f"{_MESES_ABREV_FIG[d.month - 1]} {d.year}"


periodo_bvrp = f"{_fmt_periodo_fig(df['date'].min())} – {_fmt_periodo_fig(df['date'].max())}"

print(f"BTC historico: {df_btc['date'].min().date()} a {df_btc['date'].max().date()} (N={len(df_btc)})")
print(f"Dataset BVRP:  {df['date'].min().date()} a {df['date'].max().date()} (N={len(df)})")


# ====================================================================
# Fig 7-01: Buy-and-Hold retorno acumulado (historico completo)
# ====================================================================
print("\nGerando fig_7_01...")
cum_bh = (1 + df_btc['ret']).cumprod()

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
ax.plot(df_btc['date'], cum_bh, color=COLOR_BH, lw=1.5, label='Buy-and-Hold')
ax.fill_between(df_btc['date'], 1, cum_bh, alpha=0.15, color=COLOR_BH)
ax.axhline(1, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Data')
ax.set_ylabel('Retorno acumulado (base 1)')
_MESES_PT = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun',
             'jul', 'ago', 'set', 'out', 'nov', 'dez']
def _mes_ano_pt(ts):
    return f"{_MESES_PT[ts.month - 1]} {ts.year}"
_ini = _mes_ano_pt(df_btc['date'].min())
_fim = _mes_ano_pt(df_btc['date'].max())
ax.set_title(f'Retorno acumulado — Buy-and-Hold (Bitcoin, {_ini} – {_fim})')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}x'))
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_cum_returns_buy_hold.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_cum_returns_buy_hold.png")


# ====================================================================
# Fig 7-02: Estrategia BVRP q80 — retorno acumulado
# ====================================================================
print("Gerando fig_7_02...")
cum80 = (1 + df['sr_q80']).cumprod()

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
ax.plot(df['date'], cum80, color=COLOR_Q80, lw=1.8, label='Estratégia BVRP q80')
ax.fill_between(df['date'], 1, cum80, alpha=0.15, color=COLOR_Q80)
ax.axhline(1, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Data')
ax.set_ylabel('Retorno acumulado (base 1)')
ax.set_title('Retorno acumulado — Estratégia condicional BVRP (quantil 80%)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}x'))
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_cum_returns_bvrp_quantile.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_cum_returns_bvrp_quantile.png")


# ====================================================================
# Fig 7-03a: BH vs BVRP q80 (mesmo subperiodo)
# ====================================================================
print("Gerando fig_7_03 (BH vs q80)...")
# BH no mesmo subperiodo do BVRP
df_bh_sub = df_btc[df_btc['date'].isin(df['date'])].copy()
# Melhor: usar df diretamente
cum_bh_sub = (1 + df['ret']).cumprod()

fig, ax = plt.subplots(figsize=FIGSIZE_COMP)
ax.plot(df['date'], cum_bh_sub, color=COLOR_BH,  lw=1.5, label='Buy-and-Hold', alpha=0.85)
ax.plot(df['date'], cum80,      color=COLOR_Q80, lw=1.8, label='BVRP q80 (20% do tempo)', ls='-')
ax.fill_between(df['date'], 1, cum80, alpha=0.10, color=COLOR_Q80)
ax.axhline(1, color='black', lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Data')
ax.set_ylabel('Retorno acumulado (base 1)')
ax.set_title(f'Buy-and-Hold vs Estratégia BVRP q80 ({periodo_bvrp})')
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}x'))
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_cum_returns_bh_vs_bvrp_quantile.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_cum_returns_bh_vs_bvrp_quantile.png")


# ====================================================================
# Fig 7-03b: Multi-quantis — retorno acumulado
# ====================================================================
print("Gerando fig_7_03 (multi-quantis)...")
quantiles = [0.60, 0.70, 0.80, 0.90]
fig, ax = plt.subplots(figsize=FIGSIZE_COMP)
for q, color in zip(quantiles, COLOR_PALETTE):
    qval = int(q * 100)
    cum = (1 + df[f'sr_q{qval}']).cumprod()
    ax.plot(df['date'], cum, color=color, lw=1.6, label=f'q{qval}%')
ax.plot(df['date'], cum_bh_sub, color='grey', lw=1.2, ls='--', alpha=0.7, label='Buy-and-Hold')
ax.axhline(1, color='black', lw=0.7, ls=':', alpha=0.4)
ax.set_xlabel('Data')
ax.set_ylabel('Retorno acumulado (base 1)')
ax.set_title(f'Retorno acumulado por quantil do BVRP ({periodo_bvrp})')
ax.legend(framealpha=0.9, ncol=3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}x'))
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_cum_returns_bvrp_multi_quantile.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_cum_returns_bvrp_multi_quantile.png")


# ====================================================================
# Fig 7-03c: Heatmap multi-quantis (z-score das metricas)
# ====================================================================
# Le as metricas do CSV gerado por regenerate_cap7.py (mesma fonte da Tab 7.3),
# em vez de recalcular o sinal aqui -- garante que a figura bata exatamente
# com a tabela (evita divergencia por look-ahead bias no limiar, corrigido em
# c02e22a mas nunca propagado para este script de figuras).
print("Gerando fig_7_03 (heatmap quantis)...")
df_q3 = pd.read_csv('archive/tables/estrategias/perf_multi_quantile.csv', index_col='quantil')
metrics_data = {}
for idx in df_q3.index:
    metrics_data[idx.replace('\\%', '%')] = {
        'Retorno\nAnual':   df_q3.loc[idx, 'ret_anual'],
        'Sharpe\nRatio':    df_q3.loc[idx, 'sharpe_bruto'],
        'Max\nDrawdown':    -df_q3.loc[idx, 'max_drawdown'],
        'Tempo\nInvestido': df_q3.loc[idx, 'pct_tempo'],
    }

df_heat = pd.DataFrame(metrics_data).T
# z-score por coluna
df_z = (df_heat - df_heat.mean()) / df_heat.std()
# drawdown: z-score invertido (maior drawdown = menor z-score)
df_z['Max\nDrawdown'] = -df_z['Max\nDrawdown']

fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
sns.heatmap(
    df_z, annot=df_heat.map(lambda x: f'{x:.2f}'),
    fmt='', cmap='RdYlGn', center=0,
    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.7},
)
ax.set_title('Métricas de desempenho por quantil do BVRP (z-score, valores originais anotados)')
ax.set_xlabel('')
ax.set_ylabel('Quantil')
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_heatmap_bvrp_quantile_metrics.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_heatmap_bvrp_quantile_metrics.png")


# ====================================================================
# Fig 7-04: Retorno acumulado por regime de RV
# ====================================================================
print("Gerando fig_7_04 (regimes)...")
regime_key = {'Baixo': 'baixo', 'Medio': 'medio', 'Alto': 'alto'}
fig, ax = plt.subplots(figsize=FIGSIZE_COMP)
for regime, color in REGIME_COLORS.items():
    cum = (1 + df[f'sr_regime_{regime_key[regime]}']).cumprod()
    label_map = {'Baixo': 'Baixa RV', 'Medio': 'Média RV', 'Alto': 'Alta RV'}
    ax.plot(df['date'], cum, color=color, lw=1.8, label=label_map[regime])
ax.axhline(1, color='black', lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Data')
ax.set_ylabel('Retorno acumulado (base 1)')
ax.set_title('Retorno acumulado — BVRP q80 por regime de volatilidade realizada')
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}x'))
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_cum_returns_bvrp_by_regime.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_cum_returns_bvrp_by_regime.png")


# ====================================================================
# Fig 7-05: Heatmap metricas por regime
# ====================================================================
# Le as metricas do CSV gerado por regenerate_cap7.py (mesma fonte da Tab 7.4) --
# ver nota acima sobre o heatmap de quantis.
print("Gerando fig_7_05 (heatmap regimes)...")
df_r4 = pd.read_csv('archive/tables/estrategias/perf_regimes.csv', index_col='regime')
label_map_csv = {'Baixa RV': 'Baixa RV', 'Media RV': 'Média RV', 'Alta RV': 'Alta RV'}
regime_metrics = {}
for idx in df_r4.index:
    regime_metrics[label_map_csv[idx]] = {
        'Retorno\nAnual':   df_r4.loc[idx, 'ret_anual'],
        'Sharpe\nRatio':    df_r4.loc[idx, 'sharpe'],
        'Max\nDrawdown':   -df_r4.loc[idx, 'max_drawdown'],
        'Tempo\nInvestido': df_r4.loc[idx, 'pct_tempo'],
    }

df_rh = pd.DataFrame(regime_metrics).T
df_rz = (df_rh - df_rh.mean()) / df_rh.std().replace(0, 1)
df_rz['Max\nDrawdown'] = -df_rz['Max\nDrawdown']

fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
sns.heatmap(
    df_rz, annot=df_rh.map(lambda x: f'{x:.2f}'),
    fmt='', cmap='RdYlGn', center=0,
    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.7},
)
ax.set_title('Métricas de desempenho por regime de RV (z-score, valores originais anotados)')
ax.set_xlabel('')
ax.set_ylabel('Regime de Volatilidade')
fig.tight_layout()
fig.savefig('archive/figs/estrategias/fig_heatmap_bvrp_regimes.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  OK: fig_heatmap_bvrp_regimes.png")

print("\nTodas as figuras do cap7 geradas com sucesso.")
