"""
passo4b_tamanho_b2.py — Rodada 3, B2, passo 4b: tamanho dos testes em h = 1 e h = 60 (R_log)

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é
alterado. Grava numa pasta NOVA, outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo4b/
(criada só depois da conferência), ou, com --out, retoma uma pasta existente.

É o desenho do passo 3 (mc_rotacao_poder_b2.py) com o horizonte como parâmetro, só sob H0
(ρ = 0): y = c + ruído, X real fixo, c = média do y real do caso; ruído = soma de h dias de
log-retornos diários reais, centrada: S1 sorteio iid (centrados), S2 rotação sorteada
(90 <= k <= M − 90, M = n + h), S3 blocos circulares de max(60, 2h).
Em cada réplica, ao mesmo y: Wald HAC com χ² (L = h+1 e 2h+1); Wald por bootstrap de
pares em blocos circulares (bloco ℓ_h = max(30, 2h), B = 199, L = h+1, centrado em β̂);
rotação (g_h = max(90, 3h); L = h+1 e 2h+1). Mesmos g_h e ℓ_h da Etapa A.
Casos: q16_completa, q16_987 (16 previsores), q1_ridge (BVRP previsto do Ridge; também o
Wald omnibus de 16 variáveis ao mesmo y, como no passo 3) e q1_floresta (f(X) − IV, T11).
h ∈ {1, 60}; R = 1.000 por célula; 2 h × 4 casos × 3 ruídos = 24 células, uma por CSV em
<pasta>/celulas/ (uma célula gravada é pulada).
Agregado: rejeição a 10%, 5% e 1% e valor crítico empírico de 95% do W HAC por célula.

Conferência (antes de tudo; para se não bater): com h = 30 e as chaves do passo 3, as
funções generalizadas reproduzem as réplicas r = 0, 1, 2 gravadas em
outputs/diagnostico_b2/20261020_203512_passo3/celulas/ (q1_R_log S1, q16_987 S3 e
q16_completa S2, ρ = 0), 1e-12.

Sementes: default_rng([20261020, 7, caso, ruído, h, réplica]); o resultado não depende do
número de núcleos.

Uso:  python code/diagnostico_b2/passo4b_tamanho_b2.py [--out <pasta>]
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
from joblib import Parallel, delayed, cpu_count  # noqa: E402
from scipy import stats  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, AQUI)
from bootstrap_wald_b2 import SEMENTE, idx_circ  # noqa: E402
from mc_rotacao_poder_b2 import p_mc, wald  # noqa: E402
from passo4_horizontes_b2 import blocos, carregar, gs, ls  # noqa: E402

HS_4B = (1, 60)
R = 1000
B_BOOT = 199
PROC_ID = {"S1": 1, "S2": 2, "S3": 3}
ALFAS = (0.10, 0.05, 0.01)
PASSO3 = os.path.join(ROOT, "outputs", "diagnostico_b2", "20261020_203512_passo3")
NOTA = ("ρ = 0: y independente de X; S1, S2 e S3 não reproduzem a dependência de Stambaugh. "
        "Rotação não exata com mudança de regime de volatilidade")


def bloco_s3(h):
    return max(60, 2 * h)


def retornos_h(r, datas, h):
    rc = r.loc[datas[0]: datas[-1] + pd.Timedelta(days=h)].to_numpy()
    assert len(rc) == len(datas) + h
    return rc


def somas_h(eps, n, h):
    c = np.concatenate([[0.0], np.cumsum(eps)])
    t = np.arange(n)
    return c[t + h + 1] - c[t + 1]


def montar(df, feats, r, ridge, flor, h):
    """Casos do passo 3 (id 0..3, os mesmos) e q1_floresta (id 4), com y = R_log de h dias."""
    d987 = df.set_index("date").loc[ridge.index]
    Zf = np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)])
    Z16 = np.column_stack([np.ones(len(d987)), d987[feats].to_numpy(float)])
    Z1 = np.column_stack([np.ones(len(d987)), ridge.to_numpy(float)])
    Zfl = np.column_stack([np.ones(len(d987)), flor.to_numpy(float)])
    base = [(0, "q16_completa", df, Zf, {"q16": Zf}),
            (1, "q16_987", d987, Z16, {"q16": Z16}),
            (3, "q1_ridge", d987, Z1, {"um_fator": Z1, "omnibus16": Z16}),            # id 3 = q1_R_log do passo 3
            (4, "q1_floresta", d987, Zfl, {"um_fator": Zfl})]
    casos = {}
    for cid, nome, d, Z, desenhos in base:
        datas = pd.DatetimeIndex(d["date"] if "date" in d else d.index)
        y = d[f"R_log_{h}"].to_numpy(float)
        rc = retornos_h(r, datas, h)
        assert np.allclose(somas_h(rc, len(y), h), y, atol=1e-8), f"{nome}: soma de {h} dias ≠ R_log"
        casos[nome] = {"id": cid, "nome": nome, "y": y, "rc": rc, "c": float(y.mean()),
                       "desenhos": {k: (Zd, np.linalg.inv(Zd.T @ Zd)) for k, Zd in desenhos.items()}}
    return casos


def gerar_y(c, proc, h, rng):
    rc, n = c["rc"], len(c["y"])
    if proc == "S1":
        eps = rng.choice(rc - rc.mean(), size=n + h, replace=True)
    elif proc == "S2":
        eps = np.roll(rc, int(rng.integers(90, len(rc) - 90)))
    else:
        eps = rc[idx_circ(len(rc), bloco_s3(h), rng)]
    ruido = somas_h(eps, n, h)
    ruido = ruido - ruido.mean()
    return c["c"] + ruido                                  # mesma ordem de operações do passo 3 com s = 0


def p_bootstrap(y, Z, L, bloco, rng):
    W0, b = wald(y, Z, [L])
    Wb = np.array([wald(y[j], Z[j], [L], b0=b)[0][0]
                   for j in (idx_circ(len(y), bloco, rng) for _ in range(B_BOOT))])
    v = Wb[np.isfinite(Wb)]
    return float((1 + np.sum(v >= W0[0])) / (len(v) + 1)), int(B_BOOT - len(v))


def uma_replica(c, proc, h, rr, chave):
    rng = np.random.default_rng([*chave, rr])
    y = gerar_y(c, proc, h, rng)
    Ls, g = ls(h), gs(h)["principal"]
    linhas = []
    for nome_d, (Z, ZZi) in c["desenhos"].items():
        q = Z.shape[1] - 1
        W, _ = wald(y, Z, Ls, ZZi)
        pb, nsb = p_bootstrap(y, Z, Ls[0], blocos(h)["principal"], rng)
        Wk = np.array([wald(np.roll(y, k), Z, Ls, ZZi)[0] for k in range(g, len(y) - g + 1)])
        p1, ns1 = p_mc(W[0], Wk[:, 0])
        p2, ns2 = p_mc(W[1], Wk[:, 1])
        linhas.append({"caso": c["nome"], "desenho": nome_d, "q": q, "processo": proc, "h": h, "r": rr,
                       "W_hac_h1": W[0], "p_hac_h1": float(stats.chi2.sf(W[0], q)),
                       "W_hac_2h1": W[1], "p_hac_2h1": float(stats.chi2.sf(W[1], q)),
                       "p_boot_h1": pb, "p_rot_h1": p1, "p_rot_2h1": p2,
                       "n_singular": int(np.isnan(W).sum()) + nsb + ns1 + ns2})
    return linhas


def conferencia(df, feats, r, ridge, flor):
    """h = 30 com as chaves do passo 3 ([SEMENTE, 5, id, ruído, ρ = 0]) reproduz as células gravadas."""
    casos = montar(df, feats, r, ridge, flor, 30)
    conf = []
    for nome, proc, arq in (("q1_ridge", "S1", "q1_R_log_S1_rho0000.csv"),
                            ("q16_987", "S3", "q16_987_S3_rho0000.csv"),
                            ("q16_completa", "S2", "q16_completa_S2_rho0000.csv")):
        c = casos[nome]
        ref = pd.read_csv(os.path.join(PASSO3, "celulas", arq))
        for rr in range(3):
            novo = pd.DataFrame(uma_replica(c, proc, 30, rr, [SEMENTE, 5, c["id"], PROC_ID[proc], 0]))
            for _, a in novo.iterrows():
                b = ref[(ref.r == rr) & (ref.desenho == a.desenho)].iloc[0]
                for col_n, col_r in (("W_hac_h1", "W_hac31"), ("W_hac_2h1", "W_hac61"), ("p_boot_h1", "p_boot31"),
                                     ("p_rot_h1", "p_rot31"), ("p_rot_2h1", "p_rot61")):
                    dif = abs(a[col_n] - b[col_r])
                    conf.append({"item": f"{nome} {proc} r = {rr} {a.desenho}: {col_n} = passo 3 ({col_r})",
                                 "novo": a[col_n], "passo3": b[col_r], "dif": dif,
                                 "bate": bool(dif <= 1e-12 * max(1.0, abs(b[col_r])))})
    return pd.DataFrame(conf)


def agregar(raw):
    linhas = []
    testes = [("Wald HAC (χ²)", "p_hac_h1", "h+1"), ("Wald HAC (χ²)", "p_hac_2h1", "2h+1"),
              ("Wald bootstrap", "p_boot_h1", "h+1"), ("rotação", "p_rot_h1", "h+1"), ("rotação", "p_rot_2h1", "2h+1")]
    for (caso, des, proc, h), g in raw.groupby(["caso", "desenho", "processo", "h"]):
        for teste, col, regra in testes:
            linha = {"caso": caso, "desenho": des, "q": int(g.q.iloc[0]), "processo": proc, "h": h, "teste": teste,
                     "L_regra": regra, "L": ls(h)[0 if regra == "h+1" else 1], "R": len(g),
                     "n_singular": int(g.n_singular.sum()),
                     **{f"rejeicao_{int(a * 100)}pct": float(np.mean(g[col] <= a)) for a in ALFAS}}
            if col.startswith("p_hac"):
                linha["crit_empirico_95"] = float(np.quantile(g[col.replace("p_", "W_")], 0.95))
                linha["crit_chi2_95"] = float(stats.chi2.ppf(0.95, linha["q"]))
            linha["observacao"] = NOTA
            linhas.append(linha)
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="pasta do 4b a retomar")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t_ini = time.perf_counter()
    df, feats, r, ridge, flor = carregar()
    if args.out:
        out = args.out
    else:
        conf = conferencia(df, feats, r, ridge, flor)
        print("=== Conferência (h = 30 × passo 3) ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
        if not conf.bate.all():
            raise SystemExit("CONFERÊNCIA FALHOU: nada mais foi calculado e nenhuma pasta foi criada.")
        out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo4b")
        os.makedirs(out, exist_ok=False)
        conf.to_csv(os.path.join(out, "conferencia.csv"), index=False)
    final = os.path.join(out, "tamanho_4b.csv")
    if os.path.exists(final):
        raise SystemExit(f"já existe: {final} (não sobrescrevo)")
    dir_cel = os.path.join(out, "celulas")
    os.makedirs(dir_cel, exist_ok=True)
    tempos_csv = os.path.join(out, "tempo_celulas.csv")
    casos_h = {h: montar(df, feats, r, ridge, flor, h) for h in HS_4B}
    lista = [(h, c, proc) for h in HS_4B for c in casos_h[h].values() for proc in PROC_ID]
    print(f"Núcleos: {cpu_count()}; {len(lista)} células", flush=True)
    for i, (h, c, proc) in enumerate(lista, 1):
        arq = os.path.join(dir_cel, f"{c['nome']}_{proc}_h{h}.csv")
        if os.path.exists(arq):
            print(f"  [{i}/{len(lista)}] {os.path.basename(arq)} já existe: pulada", flush=True)
            continue
        t0 = time.perf_counter()
        chave = [SEMENTE, 7, c["id"], PROC_ID[proc], h]
        res = Parallel(n_jobs=-1)(delayed(uma_replica)(c, proc, h, rr, chave) for rr in range(R))
        pd.DataFrame([x for bloco in res for x in bloco]).to_csv(arq, index=False)
        seg = time.perf_counter() - t0
        pd.DataFrame([{"caso": c["nome"], "processo": proc, "h": h, "R": R, "segundos": seg,
                       "nucleos": cpu_count()}]).to_csv(tempos_csv, mode="a", index=False,
                                                         header=not os.path.exists(tempos_csv))
        print(f"  [{i}/{len(lista)}] {c['nome']} {proc} h = {h}: {seg:.0f}s "
              f"(total {(time.perf_counter() - t_ini) / 60:.1f} min)", flush=True)
    raw = pd.concat([pd.read_csv(os.path.join(dir_cel, f)) for f in sorted(os.listdir(dir_cel))])
    tp = agregar(raw)
    tp.to_csv(final, index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 1000)
    print("\n=== Tamanho (ρ = 0) ===\n" + tp.drop(columns="observacao").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nTempo total: {(time.perf_counter() - t_ini) / 60:.1f} min; saídas em {out}")


if __name__ == "__main__":
    main()
