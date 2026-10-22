"""
passo5_componentes_regime_b2.py — Rodada 3, B2, passo 5: componentes (#151) e regime (#155/#158), R_log

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é alterado. Grava numa
pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo5/, criada só depois da conferência, ou, com
--out, retoma uma pasta existente (uma célula gravada em <pasta>/celulas/ é pulada).
Usa code/wald_utils.py (com as funções de restrição linear) e os g_h e ℓ_h do passo 4.

(b) Componentes (#151): R_log(t, h) = α + β1·VH(t−29:t) + β2·IV(t) + ε, nas 987 datas, h ∈ {1, 5, 10,
    20, 30, 60}. H0 principal: β1 = β2 = 0 (Wald, χ²(2)); ao lado, a restrição antiga do Cap. 6,
    β1 + β2 = 0 (χ²(1)). Testes: HAC (L = h+1 e 2h+1); rotação (β1 = β2 = 0: rotação de y;
    β1 + β2 = 0: Freedman–Lane), g_h = max(90, 3h) e sensibilidades; bootstrap de pares centrado,
    ℓ_h = max(30, 2h) e sensibilidades, B = 1.999. Holm sobre os 6 h (principal). Coeficientes
    α, β1, β2 e β1 + β2 com EP HAC.
(c) Regime (#155/#158): R_log(t, h) = α + β·B̂ + δ·D + γ·D·B̂ + ε, B̂ do Ridge, h ∈ {30, 60}, cortes
    fixo (37,3% a.a.) e expansivo. Coeficientes α, β, δ, γ e β + γ (inclinação no regime de alta), com
    EP HAC. H0: γ = 0, β + γ = 0 (inclinação no regime de alta) e β = 0 (inclinação no regime de
    baixa), χ²(1). Rotação de Freedman–Lane e bootstrap de pares centrado.

Conferência (antes de tudo; para se não bater, sem criar a pasta):
  (i)  componentes com R_simples, L = h+1, 6 h: b_vh, b_iv (× 100), p_vh, p_iv e p_soma =
       outputs/T10/componentes_T10.csv (1e-8);
  (ii) regime com R_log: W e p (HAC, rotação e bootstrap, variante principal, L = h+1) de γ = 0 =
       outputs/diagnostico_b2/20261021_101927_passo4/regime.csv (passo 4; W 1e-10 relativo, p 1e-12),
       com as mesmas sementes do passo 4;
  (iii) β + γ = 0 e β1 + β2 = 0: W HAC = t² do statsmodels (t_test), h = 30 e 60 (1e-8 relativo).
Sementes: componentes default_rng([20261020, 6, h, bloco, pedaço]); regime as do passo 4,
default_rng([20261020, 5, corte, h, bloco, pedaço]).

Uso:  python code/diagnostico_b2/passo5_componentes_regime_b2.py [--out <pasta>]
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import cpu_count  # noqa: E402
from scipy import stats  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, AQUI)
import wald_utils as wu  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402
from passo4_horizontes_b2 import HS, blocos, carregar, gs, ls  # noqa: E402

B = 1999
K_MIN = 60
PASSO4 = os.path.join(ROOT, "outputs", "diagnostico_b2", "20261021_101927_passo4")
H_REG = (30, 60)
CORTES = ("fixo", "expansivo")                      # índice = id do corte nas sementes (como no passo 4)
R_COMP = {"b1_b2_zero": [[0, 1.0, 0], [0, 0, 1.0]], "soma_zero": [[0, 1.0, 1.0]]}
R_REG = {"gama_zero": [[0, 0, 0, 1.0]], "beta_mais_gama_zero": [[0, 1.0, 0, 1.0]], "beta_zero": [[0, 1.0, 0, 0]]}
NOTA_COMP = ("β1 = β2 = 0: rotação de y; β1 + β2 = 0: Freedman–Lane; bootstrap de pares centrado em β̂. "
             "Rotação não exata com mudança de regime de volatilidade")
NOTA_REG = ("Freedman–Lane: resíduo do modelo restrito (Z·N, N base do núcleo de R) rotacionado; bootstrap de pares "
            "centrado em β̂; B̂ tratado como dado")


def coeficientes(y, Z, L, nomes, combos):
    """b, EP HAC (L) e, para cada combinação linear c, c'b e seu EP."""
    ZZi = np.linalg.inv(Z.T @ Z)
    b, Vs = wu._hac(y, Z, (L,), ZZi)
    V = Vs[0]
    out = {f"{n}": b[i] for i, n in enumerate(nomes)} | {f"ep_{n}": np.sqrt(V[i, i]) for i, n in enumerate(nomes)}
    for nome, c in combos.items():
        c = np.asarray(c, float)
        out[nome], out[f"ep_{nome}"] = float(c @ b), float(np.sqrt(c @ V @ c))
    return out


def testes(y, Z, h, Rs, rot_simples, chave_boot):
    """Linhas de W e p (χ², rotação, bootstrap) para cada restrição de Rs, L = h+1 e 2h+1, todas as variantes."""
    n, Ls = len(y), ls(h)
    ZZi = np.linalg.inv(Z.T @ Z)
    W, _ = wu.hac_wald_lin(y, Z, Ls, Rs, ZZi)
    ks = np.arange(K_MIN, n - K_MIN + 1)
    rot = {}
    for nome, R in Rs.items():
        rot[nome] = (wu.rotacoes(y, Z, Ls, ks, ZZi) if nome in rot_simples
                     else wu.rotacoes_freedman_lane_lin(y, Z, Ls, R, ks, ZZi))
    boots = {v: wu.bootstrap_pares_lin(y, Z, Ls, Rs, bl, B, [*chave_boot, h, bl])[1] for v, bl in blocos(h).items()}
    linhas = []
    for r, (nome, R) in enumerate(Rs.items()):
        gl = np.atleast_2d(R).shape[0]
        for i, (regra, L) in enumerate(zip(("h+1", "2h+1"), Ls)):
            base = {"h": h, "n": n, "hipotese": nome, "gl": gl, "L_regra": regra, "L": L, "W": W[r, i]}
            linhas.append({**base, "teste": "chi2", "variante": "principal", "p": float(stats.chi2.sf(W[r, i], gl))})
            for v, g in gs(h).items():
                sel = (ks >= g) & (ks <= n - g)
                p, ns = wu.p_mc(W[r, i], rot[nome][sel, i])
                linhas.append({**base, "teste": "rotacao" if nome in rot_simples else "rotacao_freedman_lane",
                               "variante": v, "parametro": g, "reps": int(sel.sum()), "n_invalidas": ns, "p": p})
            for v, bl in blocos(h).items():
                p, ns = wu.p_mc(W[r, i], boots[v][:, r, i])
                linhas.append({**base, "teste": "bootstrap", "variante": v, "parametro": bl, "reps": B,
                               "n_invalidas": ns, "p": p})
    return linhas


def item(nome, valor, ref, tol, relativa=False):
    dif = abs(valor - ref)
    return {"item": nome, "valor": valor, "referencia": ref, "dif": dif,
            "bate": bool(dif <= tol * (max(1.0, abs(ref)) if relativa else 1.0))}


def zs(d987, ridge):
    Zc = np.column_stack([np.ones(len(d987)), d987.vh_30d.to_numpy(float), d987.iv_30d.to_numpy(float)])
    b = ridge.to_numpy(float)
    Zr = {c: np.column_stack([np.ones(len(b)), b, d987[f"regime_alta_{c}"].to_numpy(float),
                              d987[f"regime_alta_{c}"].to_numpy(float) * b]) for c in CORTES}
    return Zc, Zr


def conferencia(df, d987, ridge, Zc, Zr):
    conf = []
    t10 = pd.read_csv(os.path.join(ROOT, "outputs", "T10", "componentes_T10.csv")).set_index("h")
    for h in HS:
        y = d987[f"R_simples_{h}"].to_numpy(float)
        cf = coeficientes(y, Zc, h + 1, ["a", "b_vh", "b_iv"], {"b_soma": [0, 1, 1]})
        W, _ = wu.hac_wald_lin(y, Zc, (h + 1,), {"vh": [[0, 1.0, 0]], "iv": [[0, 0, 1.0]], "soma": R_COMP["soma_zero"]})
        conf += [item(f"(i) h = {h}: b_vh = 100 × T10", cf["b_vh"], 100 * t10.b_vh[h], 1e-8, True),
                 item(f"(i) h = {h}: b_iv = 100 × T10", cf["b_iv"], 100 * t10.b_iv[h], 1e-8, True),
                 item(f"(i) h = {h}: p_vh = T10", float(stats.chi2.sf(W[0, 0], 1)), float(t10.p_vh[h]), 1e-8),
                 item(f"(i) h = {h}: p_iv = T10", float(stats.chi2.sf(W[1, 0], 1)), float(t10.p_iv[h]), 1e-8),
                 item(f"(i) h = {h}: p_soma = T10", float(stats.chi2.sf(W[2, 0], 1)), float(t10.p_soma[h]), 1e-8)]
    p4 = pd.read_csv(os.path.join(PASSO4, "regime.csv"))
    p4 = p4[(p4.hipotese == "interacao") & (p4.variante == "principal") & (p4.L_regra == "h+1")]
    for cid, c in enumerate(CORTES):
        for h in H_REG:
            lin = pd.DataFrame(testes(d987[f"R_log_{h}"].to_numpy(float), Zr[c], h, {"gama_zero": R_REG["gama_zero"]},
                                      (), [5, cid]))
            lin = lin[(lin.variante == "principal") & (lin.L_regra == "h+1")].set_index("teste")
            ref = p4[(p4.corte == c) & (p4.h == h)].set_index("teste")
            conf.append(item(f"(ii) regime {c}, h = {h}: W de γ = passo 4", lin.loc["chi2", "W"], ref.loc["chi2", "W"],
                             1e-10, True))
            for t in ("chi2", "rotacao_freedman_lane", "bootstrap"):
                conf.append(item(f"(ii) regime {c}, h = {h}: p {t} de γ = passo 4", lin.loc[t, "p"], ref.loc[t, "p"], 1e-12))
    for h in H_REG:
        y = pd.Series(d987[f"R_log_{h}"].to_numpy(float), index=d987.index)
        b = ridge.to_numpy(float)
        D = d987.regime_alta_fixo.to_numpy(float)
        X = pd.DataFrame({"b": b, "D": D, "DB": D * b}, index=d987.index)
        tt = mqo_newey_west(y, X, h=h).t_test("b + DB = 0")
        W, _ = wu.hac_wald_lin(y.to_numpy(), Zr["fixo"], (h + 1,), {"s": R_REG["beta_mais_gama_zero"]})
        conf.append(item(f"(iii) regime fixo, h = {h}: W de β + γ = t² statsmodels", W[0, 0],
                         float(np.squeeze(tt.tvalue)) ** 2, 1e-8, True))
        yc = pd.Series(d987[f"R_log_{h}"].to_numpy(float), index=d987.index)
        tc = mqo_newey_west(yc, d987[["vh_30d", "iv_30d"]], h=h).t_test("vh_30d + iv_30d = 0")
        W, _ = wu.hac_wald_lin(yc.to_numpy(), Zc, (h + 1,), {"s": R_COMP["soma_zero"]})
        conf.append(item(f"(iii) componentes, h = {h}: W de β1 + β2 = t² statsmodels", W[0, 0],
                         float(np.squeeze(tc.tvalue)) ** 2, 1e-8, True))
    return pd.DataFrame(conf)


def celula(dir_cel, nome, func):
    arq = os.path.join(dir_cel, f"{nome}.csv")
    if os.path.exists(arq):
        print(f"  {nome}: já gravada, pulada", flush=True)
        return pd.read_csv(arq)
    t0 = time.perf_counter()
    d = pd.DataFrame(func())
    d.to_csv(arq, index=False)
    print(f"  {nome}: {time.perf_counter() - t0:.0f}s", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="pasta do passo 5 a retomar")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t_ini = time.perf_counter()
    df, feats, r, ridge, flor = carregar()
    d987 = df.set_index("date").loc[ridge.index]
    Zc, Zr = zs(d987, ridge)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 80)
    if args.out:
        out = args.out
    else:
        conf = conferencia(df, d987, ridge, Zc, Zr)
        print("=== Conferência ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
        if not conf.bate.all():
            raise SystemExit("CONFERÊNCIA FALHOU: nada mais foi calculado e nenhuma pasta foi criada.")
        out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo5")
        os.makedirs(out, exist_ok=False)
        conf.to_csv(os.path.join(out, "conferencia.csv"), index=False)
    dir_cel = os.path.join(out, "celulas")
    os.makedirs(dir_cel, exist_ok=True)
    print(f"Núcleos: {cpu_count()}; saídas em {out}", flush=True)

    # (b) componentes
    comp, coef_c = [], []
    for h in HS:
        y = d987[f"R_log_{h}"].to_numpy(float)
        comp.append(celula(dir_cel, f"componentes_h{h}",
                           lambda: testes(y, Zc, h, R_COMP, ("b1_b2_zero",), [6])))
        for regra, L in zip(("h+1", "2h+1"), ls(h)):
            coef_c.append({"h": h, "L_regra": regra, "L": L,
                           **coeficientes(y, Zc, L, ["alfa", "b1_vh", "b2_iv"], {"b1_mais_b2": [0, 1, 1]})})
    comp = pd.concat(comp).assign(observacao=NOTA_COMP)
    prin = comp[comp.variante == "principal"].copy()
    prin["p_holm"] = prin.groupby(["hipotese", "L_regra", "teste"]).p.transform(lambda s: wu.holm(s.to_numpy()))
    comp.to_csv(os.path.join(out, "componentes_testes.csv"), index=False)
    prin.drop(columns="observacao").to_csv(os.path.join(out, "componentes_holm.csv"), index=False)
    pd.DataFrame(coef_c).to_csv(os.path.join(out, "componentes_coeficientes.csv"), index=False)

    # (c) regime
    reg, coef_r = [], []
    for cid, c in enumerate(CORTES):
        for h in H_REG:
            y = d987[f"R_log_{h}"].to_numpy(float)
            reg.append(celula(dir_cel, f"regime_{c}_h{h}",
                              lambda: [x | {"corte": c} for x in testes(y, Zr[c], h, R_REG, (), [5, cid])]))
            for regra, L in zip(("h+1", "2h+1"), ls(h)):
                coef_r.append({"corte": c, "h": h, "L_regra": regra, "L": L,
                               "pct_alta": 100 * float(Zr[c][:, 2].mean()),
                               **coeficientes(y, Zr[c], L, ["alfa", "beta", "delta", "gama"],
                                              {"beta_mais_gama": [0, 1, 0, 1]})})
    reg = pd.concat(reg).assign(observacao=NOTA_REG)
    reg.to_csv(os.path.join(out, "regime_testes.csv"), index=False)
    pd.DataFrame(coef_r).to_csv(os.path.join(out, "regime_coeficientes.csv"), index=False)

    seg = time.perf_counter() - t_ini
    pd.DataFrame([{"etapa": "total", "segundos": seg, "nucleos": cpu_count()}]).to_csv(
        os.path.join(out, "tempo.csv"), index=False)
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    print("\n=== Componentes: coeficientes (L = h+1) ===\n" + pd.DataFrame(coef_c).query("L_regra == 'h+1'")
          .to_string(index=False, float_format=fmt))
    print("\n=== Componentes: p (principal) e Holm ===\n" + prin.pivot_table(
        index=["hipotese", "L_regra", "h"], columns="teste", values=["p", "p_holm"]).round(4).to_string())
    print("\n=== Regime: coeficientes (L = h+1) ===\n" + pd.DataFrame(coef_r).query("L_regra == 'h+1'")
          .to_string(index=False, float_format=fmt))
    print("\n=== Regime: p (principal) ===\n" + reg[reg.variante == "principal"].pivot_table(
        index=["corte", "h", "hipotese", "L_regra"], columns="teste", values="p").round(4).to_string())
    print(f"\nTempo: {seg / 60:.1f} min; saídas em {out}")


if __name__ == "__main__":
    main()
