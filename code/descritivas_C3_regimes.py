"""
descritivas_C3_regimes.py — BVRP e proxy por regime de volatilidade (Fase 3, F4/C3)

Amostra descritiva do Cap. 3 (data/vrp_with_targets.csv, N = 1.806,
24/03/2021 a 03/03/2026). Regime de alta: vh_30d (coluna rv_30d, VH de t-29 a
t) acima do corte fixo da mistura de duas normais estimada só na 1ª janela
(corte_fixo de data/regimes_T9.csv, 37,3% a.a.). As 30 datas anteriores à
amostra de modelagem (24/03 a 22/04/2021) usam o mesmo corte.

Para cada série (bvrp_30d_fut, o BVRP prospectivo; vrp_30d, a proxy
retrospectiva): média, desvio-padrão e N por regime, e a diferença entre
regimes por MQO da série numa dummy (1 = alta), EP de Newey–West com h = 30
(31 defasagens; hac_utils.mqo_newey_west).

Convenção de sinal: BVRP = VH − IV. Prêmio maior = BVRP mais negativo
(sinal oposto ao de Almeida et al., que usam variância neutra ao risco menos
física).

Saídas em <out>/: bvrp_por_regime_C3.csv e log_C3_regimes.txt.
Uso: python code/descritivas_C3_regimes.py [--out outputs/C3]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = a.root
    out = a.out or os.path.join(root, "outputs", "C3")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.join(root, "code"))
    from hac_utils import mqo_newey_west  # noqa: E402

    df = pd.read_csv(os.path.join(root, "data", "vrp_with_targets.csv"), parse_dates=["date"])
    reg = pd.read_csv(os.path.join(root, "data", "regimes_T9.csv"), parse_dates=["date"])
    assert len(df) == 1806, len(df)
    assert (df["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "calendário com lacunas"
    corte = float(reg["corte_fixo"].iloc[0])
    assert np.allclose(reg["corte_fixo"], corte)
    df["alta"] = (df["rv_30d"] > corte).astype(int)
    chk = df.merge(reg[["date", "regime_alta_fixo"]], on="date", how="inner")
    assert len(chk) == 1776 and (chk["alta"] == chk["regime_alta_fixo"]).all(), "regime diverge do T9"
    assert df[["bvrp_30d_fut", "vrp_30d", "rv_30d"]].notna().all().all()

    linhas, log = [], [f"Corte fixo (T9): {corte:.6f}% a.a.; N = {len(df)}; "
                       f"regime de alta: {df.alta.sum()} datas ({100 * df.alta.mean():.1f}%)"]
    for col, nome in [("bvrp_30d_fut", "BVRP"), ("vrp_30d", "BVRP_proxy")]:
        res = mqo_newey_west(df[col], df["alta"], h=30)
        b = res.params.iloc[1]
        assert np.isclose(b, df.loc[df.alta == 1, col].mean() - df.loc[df.alta == 0, col].mean())
        linha = {"serie": nome, "corte": corte}
        for r, rot in [(0, "baixa"), (1, "alta")]:
            s = df.loc[df.alta == r, col]
            linha.update({f"n_{rot}": len(s), f"media_{rot}": s.mean(), f"dp_{rot}": s.std(ddof=1),
                          f"frac_neg_{rot}": (s < 0).mean()})
        linha.update({"dif_alta_menos_baixa": b, "ep_hac31": res.bse.iloc[1], "z": res.tvalues.iloc[1],
                      "p_valor": res.pvalues.iloc[1], "maxlags": res.cov_kwds["maxlags"]})
        linhas.append(linha)
        log.append(f"{nome}: baixa {linha['media_baixa']:.2f} (N={linha['n_baixa']}), alta "
                   f"{linha['media_alta']:.2f} (N={linha['n_alta']}); dif {b:.2f}, EP {linha['ep_hac31']:.2f}, "
                   f"z {linha['z']:.2f}, p {linha['p_valor']:.4f}")
    # Componentes por regime: VH retrospectiva, VH prospectiva e IV (médias, % a.a.)
    for col, nome in [("rv_30d", "VH_retrospectiva"), ("rv_30d_fut", "VH_prospectiva"), ("iv_30d", "IV")]:
        linha = {"serie": nome, "corte": corte}
        for r, rot in [(0, "baixa"), (1, "alta")]:
            s = df.loc[df.alta == r, col]
            linha.update({f"n_{rot}": len(s), f"media_{rot}": s.mean(), f"dp_{rot}": s.std(ddof=1)})
        linhas.append(linha)
        log.append(f"{nome}: baixa {linha['media_baixa']:.2f}, alta {linha['media_alta']:.2f}")
    pd.DataFrame(linhas).to_csv(os.path.join(out, "bvrp_por_regime_C3.csv"), index=False)
    with open(os.path.join(out, "log_C3_regimes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
