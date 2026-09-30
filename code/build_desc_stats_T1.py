"""
build_desc_stats_T1.py
======================
Gera as Tabs. 3.1 (estatisticas descritivas) e 3.2 (ADF/KPSS) e a tabela
comparativa das definicoes do BVRP, SEM sobrescrever tables/ (sincronizado
com o Overleaf). Todas as saidas vao para --out-dir.

Criado no T0/T1 do plano de revisao (set/2026): as Tabs. 3.1 e 3.2 nao
tinham gerador versionado. A partir do T1, as duas definicoes do BVRP saem
lado a lado: prospectiva (bvrp_30d_fut, RV de t+1 a t+30 - IV_t) e
retrospectiva (vrp_30d, proxy; RV de t-29 a t - IV_t). So entram as colunas
presentes no dataset, entao o script continua rodando sobre dados pre-T1.

Convencoes (conferidas contra a Tab. 3.1 publicada, reproduzida exatamente):
  - retorno diario = coluna `ret` (log-retorno), em fracao decimal
  - desvio-padrao amostral (ddof=1)
  - assimetria e curtose de Fisher (excesso) SEM correcao de vies (scipy)
  - ADF: autolag="AIC"; KPSS: regression="c", nlags="auto"
    (mesma convencao de code/validate_tab_3_adf_kpss.py, cuja funcao de
    classificacao e reaproveitada)

Entradas:
  data/vrp_with_targets.csv  -> amostra de referencia (apos corte de h=60)
  data/vrp_30d_dataset.csv   -> serie completa (antes do corte de h=60)
  data/btc_prices.csv, data/dvol_30d_full.csv, data/vrp_with_regimes.csv,
  data/ml_dataset.csv        -> apenas para a contagem de N por etapa

Uso:
  python code/build_desc_stats_T1.py --out-dir outputs/T1
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_tab_3_adf_kpss import classifica  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Definicoes do BVRP comparadas lado a lado. So entram as colunas presentes
# no dataset.
BVRP_DEFS = {
    "bvrp_30d_fut": "BVRP prospectivo (p.p.)",
    "vrp_30d":      "BVRP retrospectivo, proxy (p.p.)",
}

# Demais linhas da Tab. 3.1: (coluna, rotulo, casas decimais). So entram as
# colunas presentes no dataset.
VOL_ROWS = [
    ("rv_30d_fut", "RV 30d prospectiva (p.p.)", 2),
    ("rv_30d",     "RV 30d retrospectiva (p.p.)", 2),
    ("iv_30d",     "IV 30d (p.p.)", 2),
    ("ret",        "Retorno diário", 4),
]

ADF_LABELS = {
    "I(1)": "I(1)",
    "I(0)": "I(0)",
    "persistente": "Longa memória",
    "inconclusivo": "Inconclusivo",
}

_MESES = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.",
          "jul.", "ago.", "set.", "out.", "nov.", "dez."]


# ---------------------------------------------------------------------------
# Formatacao
# ---------------------------------------------------------------------------
def fmt_br(x, d):
    return f"{x:.{d}f}".replace(".", ",")


def fmt_m(x, d):
    """Número para modo matemático: vírgula decimal protegida ({,}) sem espaço."""
    return fmt_br(x, d).replace(",", "{,}")


def fmt_n(n):
    s = f"{n:,}".replace(",", "{.}")
    return s


def fmt_periodo(d0, d1):
    return f"{_MESES[d0.month - 1]}\\ {d0.year} -- {_MESES[d1.month - 1]}\\ {d1.year}"


# ---------------------------------------------------------------------------
# Estatisticas
# ---------------------------------------------------------------------------
def desc_row(s):
    s = s.dropna()
    q = s.quantile([0.05, 0.25, 0.75, 0.95])
    return {
        "N": len(s),
        "Média": s.mean(),
        "Desv. Pad.": s.std(ddof=1),
        "Mínimo": s.min(),
        "p5": q[0.05], "p25": q[0.25], "p75": q[0.75], "p95": q[0.95],
        "Máximo": s.max(),
        "Assimetria": stats.skew(s),
        "Curtose": stats.kurtosis(s),
    }


def comparativo_row(df, col, amostra):
    sub = df[["date", col]].dropna()
    s = sub[col]
    return {
        "definicao": col,
        "amostra": amostra,
        "N": len(s),
        "data_inicial": sub["date"].min().date(),
        "data_final": sub["date"].max().date(),
        "media": s.mean(),
        "mediana": s.median(),
        "desvio_padrao": s.std(ddof=1),
        "minimo": s.min(),
        "maximo": s.max(),
        "pct_negativo": 100.0 * (s < 0).mean(),
        "assimetria": stats.skew(s),
        "curtose_excesso": stats.kurtosis(s),
        "ac1": acf(s, nlags=1, fft=True)[1],
    }


def adf_kpss_row(s):
    s = s.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return adf_stat, adf_p, kpss_stat, kpss_p, classifica(adf_p, kpss_p)


# ---------------------------------------------------------------------------
# Saidas
# ---------------------------------------------------------------------------
def tab_3_1(df, out_dir, bvrp_cols):
    rows, tex_rows = [], []
    for col in bvrp_cols:
        r = desc_row(df[col])
        rows.append({"Variável": BVRP_DEFS[col], "coluna": col, **r})
        tex_rows.append((BVRP_DEFS[col], r, 2))
    for col, label, d in VOL_ROWS:
        if col not in df.columns:
            continue
        r = desc_row(df[col])
        rows.append({"Variável": label, "coluna": col, **r})
        tex_rows.append((label, r, d))

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tab_3_1_desc_stats.csv"), index=False)

    n = len(df)
    periodo = fmt_periodo(df["date"].min(), df["date"].max())
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Estatísticas descritivas das variáveis principais.",
        f"Período: {periodo} ($N={fmt_n(n)}$ observações diárias).",
        r"BVRP e volatilidades em pontos percentuais anualizados;",
        r"retorno diário em fração decimal.",
        r"Curtose reportada como excesso (Fisher): distribuição normal $= 0$.}",
        r"\label{tab:cap3_desc_stats}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Variável & Média & D.P. & Mín. & p5 & p25 & p75 & p95 & Máx. & Assim. & Curtose \\",
        r"\midrule",
    ]
    for label, r, d in tex_rows:
        vals = [r[k] for k in ["Média", "Desv. Pad.", "Mínimo", "p5", "p25", "p75", "p95", "Máximo"]]
        cells = " & ".join(fmt_br(v, d) for v in vals)
        lines.append(f"{label} & {cells} & {fmt_br(r['Assimetria'], 3)} & {fmt_br(r['Curtose'], 3)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip",
        r"\footnotesize\textit{Nota}: RV 30d calculada como raiz da média dos retornos logarítmicos"
        r" diários quadráticos, anualizada ($\times\sqrt{365}$): retrospectiva sobre os retornos de"
        r" $t-29$ a $t$; prospectiva sobre os retornos de $t+1$ a $t+30$."
        r" IV 30d: índice DVOL da Deribit, observado em $t$."
        + (r" BVRP prospectivo: $\text{BVRP}_{t+1:t+30\,|\,t} = \text{RV}_{t+1:t+30} - \text{IV}_t$;"
           r" BVRP retrospectivo (proxy): $\text{RV}_{t-29:t} - \text{IV}_t$."
           if "bvrp_30d_fut" in bvrp_cols else ""),
        r"\end{table}",
    ]
    with open(os.path.join(out_dir, "tab_3_1_desc_stats.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def tab_3_2(df, out_dir, bvrp_cols):
    series = []
    if "close" in df.columns:
        series.append(("close", r"Preço (\textit{close})", df["close"]))
    series.append(("ret", "Retorno diário", df["ret"]))
    for col, label, _ in VOL_ROWS:
        if col in df.columns and col != "ret":
            series.append((col, label.replace(" (p.p.)", ""), df[col]))
    for col in bvrp_cols:
        series.append((col, BVRP_DEFS[col].replace(" (p.p.)", ""), df[col]))

    rows, lines_body = [], []
    for col, label, s in series:
        a, ap, k, kp, concl = adf_kpss_row(s)
        rows.append({"variavel": col, "adf_stat": a, "adf_p": ap,
                     "kpss_stat": k, "kpss_p": kp, "classificacao": concl})
        ap_s = r"$< 0{,}001$" if ap < 0.001 else f"${fmt_m(ap, 3)}$"
        if kp <= 0.01:
            kp_s = r"${\leq}0{,}01$"
        elif kp >= 0.10:
            kp_s = r"${\geq}0{,}10$"
        else:
            kp_s = f"${fmt_m(kp, 3)}$"
        lines_body.append(
            f"{label} & ${fmt_m(a, 2)}$ & {ap_s} & ${fmt_m(k, 2)}$ & {kp_s} & {ADF_LABELS[concl]} \\\\")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tab_3_2_adf_kpss.csv"), index=False)

    n = len(df)
    periodo = fmt_periodo(df["date"].min(), df["date"].max())
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Testes de estacionariedade ADF e KPSS --- variáveis principais da dissertação."
        r" Hipótese nula do ADF: presença de raiz unitária; hipótese nula do KPSS: estacionariedade."
        f" Período: {periodo} ($N = {fmt_n(n)}$ observações diárias).}}",
        r"\label{tab:adf_kpss}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Variável & \multicolumn{2}{c}{ADF} & \multicolumn{2}{c}{KPSS} & Conclusão \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & Estat. & $p$-valor & Estat. & $p$-valor & \\",
        r"\midrule",
        *lines_body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize\textit{Nota}: ADF com seleção de defasagens por AIC; KPSS com constante e"
        r" defasagens automáticas. \textit{Longa memória}: ADF rejeita (ou está na fronteira) e"
        r" KPSS rejeita a estacionariedade.",
        r"\end{table}",
    ]
    with open(os.path.join(out_dir, "tab_3_2_adf_kpss.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def comparativo(df_ref, df_full, out_dir, bvrp_cols):
    rows = []
    for col in bvrp_cols:
        rows.append(comparativo_row(df_ref, col, "referencia (apos corte h=60)"))
        if col in df_full.columns:
            rows.append(comparativo_row(df_full, col, "serie completa"))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "comparativo_bvrp.csv"), index=False)
    return out


def n_por_etapa(out_dir):
    def info(path, date_col="date", tz=False):
        full = os.path.join(DATA, path)
        if not os.path.exists(full):
            return None
        d = pd.read_csv(full)
        dates = pd.to_datetime(d[date_col])
        if tz:
            dates = dates.dt.tz_localize(None)
        dates = dates.dt.normalize()
        esperado = (dates.max() - dates.min()).days + 1
        return {"arquivo": path, "N": len(d),
                "data_inicial": dates.min().date(), "data_final": dates.max().date(),
                "dias_corridos": esperado, "dias_faltando": esperado - dates.nunique()}

    etapas = [
        ("preços BTC brutos", info("btc_prices.csv")),
        ("DVOL bruto", info("dvol_30d_full.csv", "timestamp", tz=True)),
        ("merge preço x DVOL", info("vrp_30d_dataset.csv")),
        ("após corte h=60 (referência)", info("vrp_with_targets.csv")),
        ("com coluna de regime", info("vrp_with_regimes.csv")),
        ("dataset de AM (burn-in de regime)", info("ml_dataset.csv")),
    ]
    rows = [{"etapa": e, **i} for e, i in etapas if i is not None]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "n_por_etapa.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T1"))
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_ref = pd.read_csv(os.path.join(DATA, "vrp_with_targets.csv"), parse_dates=["date"])
    df_full = pd.read_csv(os.path.join(DATA, "vrp_30d_dataset.csv"), parse_dates=["date"])
    df_ref = df_ref.sort_values("date").reset_index(drop=True)
    df_full = df_full.sort_values("date").reset_index(drop=True)

    bvrp_cols = [c for c in BVRP_DEFS if c in df_ref.columns]
    if not bvrp_cols:
        raise ValueError("Nenhuma coluna de BVRP encontrada em vrp_with_targets.csv")

    tab_3_1(df_ref, out_dir, bvrp_cols)
    tab_3_2(df_ref, out_dir, bvrp_cols)
    comp = comparativo(df_ref, df_full, out_dir, bvrp_cols)
    etapas = n_por_etapa(out_dir)

    pd.set_option("display.width", 200)
    print(f"Saidas em: {out_dir}\n")
    print("=== N por etapa ===")
    print(etapas.to_string(index=False))
    print("\n=== Comparativo do BVRP ===")
    print(comp.T.to_string(header=False))


if __name__ == "__main__":
    main()
