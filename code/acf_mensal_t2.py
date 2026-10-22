"""
acf_mensal_t2.py — autocorrelação do prêmio e de seus insumos: série diária × série mensal sem
sobreposição (Cap. 3, #76/#78)

Lê data/vrp_with_targets.csv (amostra completa, 1.806 datas, a do Cap. 3) e grava em --out-dir
(padrão outputs/T2):
  acf_mensal_T2.csv            resumo por série (diária nas defasagens 1 e 30; mensal de 1ª ordem:
                               mediana, mínimo e máximo entre os 30 pontos de partida; N; EP ≈ 1/√N)
  acf_mensal_partidas_T2.csv   autocorrelação mensal de cada ponto de partida k = 0..29
  tab_acf_mensal_T2.tex        tabela do Cap. 3
Com --publicar-cap3, copia a tabela para tables/dados/acf_mensal.tex (.tex e .csv), o nome usado no
Cap. 3, como os demais geradores do capítulo.

Séries: RBVRP (bvrp_30d_fut), HBVRP (vrp_30d), RV(t+1:t+30) (rv_30d_fut), VH(t−29:t) (rv_30d) e
IV (iv_30d). Estimador: statsmodels.tsa.stattools.acf, o mesmo de scripts/test_h1_bvrp_mean.py.
Série mensal: s[k], s[k+30], s[k+60], ... para k = 0..29; janelas de 30 dias de datas consecutivas
não se sobrepõem. Sem aleatoriedade.

Conferência (sai com erro se não bater): autocorrelação diária nas defasagens 1 e 30 =
outputs/T2/h1_acf.csv (1e-12); a mensal de 1ª ordem de cada ponto de partida = a correlação de
Pearson recalculada à mão com a média e a variância da própria série (definição do acf), 1e-12.

Uso:  python code/acf_mensal_t2.py [--out-dir outputs/T2] [--publicar-cap3]
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSO = 30
SERIES = [("bvrp_30d_fut", r"RBVRP"), ("vrp_30d", r"HBVRP"), ("rv_30d_fut", r"$\text{RV}_{t+1:t+30}$"),
          ("rv_30d", r"$\text{VH}_{t-29:t}$"), ("iv_30d", r"IV")]
TOL = 1e-12


def fmt(v, d=2):
    s = f"{v:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "$-$")


def acf1_manual(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[1:] @ x[:-1]) / (x @ x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("outputs", "T2"))
    ap.add_argument("--publicar-cap3", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ROOT, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) != 1806 or not (df.date.diff().dropna() == pd.Timedelta(days=1)).all():
        sys.exit("amostra diferente da do Cap. 3 (1.806 datas contíguas)")
    ref = pd.read_csv(os.path.join(ROOT, "outputs", "T2", "h1_acf.csv")).set_index("serie")

    resumo, partidas = [], []
    for col, rotulo in SERIES:
        s = df[col].dropna().to_numpy(float)
        a = acf(s, nlags=PASSO, fft=True)
        for lag in (1, PASSO):
            if abs(a[lag] - ref.loc[col, f"rho_{lag}"]) > TOL:
                sys.exit(f"CONFERÊNCIA FALHOU: {col}, defasagem {lag} ≠ outputs/T2/h1_acf.csv")
        m = []
        for k in range(PASSO):
            x = s[k::PASSO]
            r1 = float(acf(x, nlags=1, fft=False)[1])
            if abs(r1 - acf1_manual(x)) > TOL:
                sys.exit(f"CONFERÊNCIA FALHOU: {col}, k = {k}: acf ≠ cálculo à mão")
            m.append({"serie": col, "k": k, "n": len(x), "rho1": r1, "ep_aprox": 1 / np.sqrt(len(x))})
        m = pd.DataFrame(m)
        partidas.append(m)
        resumo.append({"serie": col, "rotulo": rotulo, "n_diaria": len(s), "diaria_rho1": a[1],
                       "diaria_rho30": a[PASSO], "mensal_rho1_mediana": m.rho1.median(),
                       "mensal_rho1_min": m.rho1.min(), "mensal_rho1_max": m.rho1.max(),
                       "mensal_n_min": int(m.n.min()), "mensal_n_max": int(m.n.max()),
                       "ep_aprox_min": m.ep_aprox.min(), "ep_aprox_max": m.ep_aprox.max()})
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(out_dir, "acf_mensal_T2.csv"), index=False)
    pd.concat(partidas).to_csv(os.path.join(out_dir, "acf_mensal_partidas_T2.csv"), index=False)

    nmin, nmax = int(resumo.mensal_n_min.min()), int(resumo.mensal_n_max.max())
    ep = fmt(1 / np.sqrt(nmin), 2)
    linhas = [rf"{r.rotulo} & {fmt(r.diaria_rho1)} & {fmt(r.diaria_rho30)} & {fmt(r.mensal_rho1_mediana)} & "
              rf"{fmt(r.mensal_rho1_min)} & {fmt(r.mensal_rho1_max)} \\" for r in resumo.itertuples()]
    tex = "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Autocorrelação do prêmio e de seus insumos: série diária e série mensal sem sobreposição.}",
        r"\label{tab:dados-acf-mensal}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{Diária} & \multicolumn{3}{c}{Mensal, 1ª ordem} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Série & $\rho_1$ & $\rho_{30}$ & Mediana & Mín. & Máx. \\", r"\midrule", *linhas,
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        rf"\parbox{{0.95\linewidth}}{{\footnotesize\textit{{Nota}}: Diária: autocorrelação da série diária "
        rf"({f'{len(df):,}'.replace(',', '.')} datas, de 24/03/2021 a 03/03/2026) nas defasagens de 1 e 30 dias. Mensal: autocorrelação "
        rf"de primeira ordem da série com uma data a cada 30 dias, em que as janelas de 30 dias de datas "
        rf"consecutivas não se sobrepõem, em cada um dos 30 pontos de partida ($N$ = {nmin} ou {nmax}); "
        rf"mediana, mínimo e máximo entre os pontos de partida. Erro-padrão aproximado da autocorrelação "
        rf"mensal: $1/\sqrt{{N}} \approx {ep}$.}}",
        r"\end{table}", ""])
    caminho = os.path.join(out_dir, "tab_acf_mensal_T2.tex")
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)
    print(resumo.drop(columns="rotulo").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.publicar_cap3:
        dest = os.path.join(ROOT, "tables", "dados")
        shutil.copyfile(caminho, os.path.join(dest, "acf_mensal.tex"))
        shutil.copyfile(os.path.join(out_dir, "acf_mensal_T2.csv"), os.path.join(dest, "acf_mensal.csv"))
        print(f"publicado em {dest}: acf_mensal.tex e acf_mensal.csv")


if __name__ == "__main__":
    main()
