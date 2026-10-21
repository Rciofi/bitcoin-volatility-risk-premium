"""
mc_rotacao_poder_b2.py — Rodada 3, B2, passo 3: teste de Monte Carlo por rotação; tamanho e poder

Só lê arquivos existentes; nada vai para tables/ ou figs/ e nada do pipeline é
alterado. Duas etapas, na mesma pasta nova outputs/diagnostico_b2/<AAAAMMDD_HHMMSS>_passo3/:

  --etapa observado   (segundos) conferência do HAC em numpy contra hac_utils e
                      W = 20,01 (p χ² = 0,220) e p HAC 0,091 do T10 (para se não
                      bater; a pasta só é criada depois); teste de rotação nos
                      dados observados: Wₖ = Wald HAC(roll(y, k), X), k = g .. n − g
                      (todos), p = (1 + #{Wₖ >= W}) / (K + 1), g = 90 (principal),
                      60 e 180; L = 31 e 61. Sensibilidade: rotação dos log-retornos
                      DIÁRIOS (k = g .. M − g, M = n + 30) e y refeito pela soma de
                      30 dias. Determinístico (sem semente).
  --etapa simulacao --out <pasta>   (em segundo plano) tamanho e poder:
                      y = c + s·(X·b̂ − média) + ruído, X real fixo, b̂ = MQO do
                      próprio caso, s tal que o R² populacional = ρ (Var(ruído) =
                      variância do y real do caso). Ruído = soma de 30 dias de
                      log-retornos diários reais: S1 sorteio iid (centrados), S2
                      rotação sorteada (90 <= k <= M − 90), S3 blocos circulares de
                      60 (só ρ = 0). R = 1.000 (ρ = 0) e 500 (ρ > 0). Em cada
                      réplica, ao mesmo y: Wald HAC com χ² (L = 31, 61), Wald por
                      bootstrap circular (bloco 60, B = 199, L = 31, centrado em β̂)
                      e Monte Carlo por rotação (g = 90; L = 31, 61). Casos q16
                      (amostra completa e 987 datas, R_log; ρ = 0, 1%, 2%, 5%) e
                      q = 1 (BVRP previsto do Ridge do T5, 987 datas, R_simples e
                      R_log; sinal na direção de f̂; ρ = 0, 0,5%, 1%, 2%, 5%), no
                      q = 1 também o Wald omnibus de 16 variáveis ao mesmo y.
                      CSV parcial por célula em <pasta>/celulas/ (uma célula já
                      gravada é pulada); agregado em tamanho_poder.csv, com o poder
                      ajustado ao tamanho do Wald HAC (valor crítico = quantil de
                      95% de W nas réplicas de ρ = 0 do mesmo ruído).
Sementes: default_rng([20261020, 5, caso, ruído, ρ em 1/10.000, réplica]); o
resultado não depende do número de núcleos.

Uso:  python code/diagnostico_b2/mc_rotacao_poder_b2.py --etapa observado
      python code/diagnostico_b2/mc_rotacao_poder_b2.py --etapa simulacao --out <pasta do passo 3>
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
from bootstrap_wald_b2 import carregar, idx_circ, retornos_do_caso, somas_30  # noqa: E402
from hac_utils import mqo_newey_west, sensibilidade_defasagens  # noqa: E402

H = 30
LS = (31, 61)
G_PRINCIPAL = 90
GS = (90, 60, 180)
B_BOOT, BLOCO_BOOT, L_BOOT = 199, 60, 31
R_NULO, R_ALT = 1000, 500
RHOS = {"q16": (0.0, 0.01, 0.02, 0.05), "q1": (0.0, 0.005, 0.01, 0.02, 0.05)}
PROC_ID = {"S1": 1, "S2": 2, "S3": 3}
ALFAS = (0.10, 0.05, 0.01)
SEMENTE = 20261020
NOTAS = ("H0 do teste de rotação: y independente de X (mais forte que β = 0); o teste não é exato com mudança "
         "de regime de volatilidade (y não é circularmente estacionário); S1, S2 e S3 geram ruído independente "
         "de X e não reproduzem a dependência de Stambaugh")


# ---------------------------------------------------------------------------
# Wald HAC (numpy; Bartlett sem correção, como hac_utils)
# ---------------------------------------------------------------------------
def inv_ou_none(Z):
    try:
        return np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None


def wald(y, Z, Ls, ZZi=None, b0=None):
    """W das inclinações (contra b0[1:], ou 0) para cada L; nan se singular. Retorna (W, b)."""
    if ZZi is None:
        ZZi = inv_ou_none(Z)
        if ZZi is None:
            return np.full(len(Ls), np.nan), None
    b = ZZi @ (Z.T @ y)
    s = Z * (y - Z @ b)[:, None]
    S0 = s.T @ s
    G = [s[l:].T @ s[:-l] for l in range(1, max(Ls) + 1)]
    d = b[1:] - (0.0 if b0 is None else b0[1:])
    W = []
    for L in Ls:
        S = S0.copy()
        for l in range(1, L + 1):
            S += (1 - l / (L + 1)) * (G[l - 1] + G[l - 1].T)
        V = (ZZi @ S @ ZZi)[1:, 1:]
        try:
            w = float(d @ np.linalg.solve(V, d))
        except np.linalg.LinAlgError:
            w = np.nan
        W.append(w if np.isfinite(w) else np.nan)
    return np.array(W), b


def p_mc(W0, Wk):
    v = Wk[np.isfinite(Wk)]
    return float((1 + np.sum(v >= W0)) / (len(v) + 1)), int(len(Wk) - len(v))


def W_rotacoes_y(y, Z, ZZi, ks):
    return np.array([wald(np.roll(y, k), Z, LS, ZZi)[0] for k in ks])


def W_rotacoes_diarias(rc, n, simples, Z, ZZi, ks):
    out = []
    for k in ks:
        y = somas_30(np.roll(rc, k), n)
        if simples:
            y = 100 * np.expm1(y / 100)
        out.append(wald(y, Z, LS, ZZi)[0])
    return np.array(out)


def em_paralelo(func, ks, *args):
    pedacos = [list(c) for c in np.array_split(np.asarray(list(ks)), 32) if len(c)]
    return np.vstack(Parallel(n_jobs=-1)(delayed(func)(*args, p) for p in pedacos))


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------
def montar(df, feats, r, ridge):
    d987 = df.set_index("date").loc[ridge.index]
    Zf = np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)])
    Z16 = np.column_stack([np.ones(len(d987)), d987[feats].to_numpy(float)])
    Z1 = np.column_stack([np.ones(len(d987)), ridge.to_numpy(float)])
    rc_f = retornos_do_caso(r, pd.DatetimeIndex(df.date))
    rc_9 = retornos_do_caso(r, pd.DatetimeIndex(d987.index))
    base = [("q16_completa", "q16", "R_log", df.R_log.to_numpy(), Zf, rc_f, {"q16": Zf}),
            ("q16_987", "q16", "R_log", d987.R_log.to_numpy(), Z16, rc_9, {"q16": Z16}),
            ("q1_R_simples", "q1", "R_simples", d987.R_simples.to_numpy(), Z1, rc_9, {"um_fator": Z1, "omnibus16": Z16}),
            ("q1_R_log", "q1", "R_log", d987.R_log.to_numpy(), Z1, rc_9, {"um_fator": Z1, "omnibus16": Z16})]
    casos = {}
    for cid, (nome, tipo, ycol, y, Z, rc, desenhos) in enumerate(base):
        y0 = somas_30(rc, len(y))
        y0 = 100 * np.expm1(y0 / 100) if ycol == "R_simples" else y0
        assert np.allclose(y0, y, atol=1e-8), f"{nome}: soma de 30 dias ≠ {ycol}"
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        sig = Z[:, 1:] @ b[1:]                        # q16: X·b̂ (MQO do caso); q1: β̂·f̂
        casos[nome] = {"id": cid, "nome": nome, "tipo": tipo, "ycol": ycol, "y": y, "rc": rc,
                       "sinal": sig - sig.mean(), "var_ruido": float(np.var(y, ddof=1)), "c": float(y.mean()),
                       "desenhos": {k: (Zd, np.linalg.inv(Zd.T @ Zd)) for k, Zd in desenhos.items()}}
    return casos


# ---------------------------------------------------------------------------
# Etapa 1: conferência e rotação no observado
# ---------------------------------------------------------------------------
def conferencia(df, feats, casos):
    conf = []
    Zf, ZZi = casos["q16_completa"]["desenhos"]["q16"]
    W, _ = wald(df.R_log.to_numpy(), Zf, LS, ZZi)
    res = {31: mqo_newey_west(df.R_log, df[feats], h=H),
           61: sensibilidade_defasagens(df.R_log, df[feats], lags=[61])[61]}
    for i, L in enumerate(LS):
        ws = float(np.squeeze(res[L].wald_test(np.eye(17)[1:], use_f=False, scalar=True).statistic))
        conf.append({"item": f"q16 completa, R_log, L = {L}: W numpy × hac_utils", "numpy": W[i], "referencia": ws,
                     "bate": abs(W[i] - ws) <= 1e-8 * max(1, ws)})
    p = float(stats.chi2.sf(W[0], 16))
    conf += [{"item": "q16 completa, L = 31: W arredondado = 20,01", "numpy": W[0], "referencia": 20.01,
              "bate": round(W[0], 2) == 20.01},
             {"item": "q16 completa, L = 31: p χ² arredondado = 0,220", "numpy": p, "referencia": 0.220,
              "bate": round(p, 3) == 0.220}]
    c = casos["q1_R_simples"]
    W1, _ = wald(c["y"], c["desenhos"]["um_fator"][0], [31])
    p1 = float(stats.chi2.sf(W1[0], 1))
    conf.append({"item": "q1 Ridge, R_simples, L = 31: p = p HAC do T10 (0,0909109)", "numpy": p1,
                 "referencia": 0.0909109, "bate": abs(p1 - 0.0909109) < 1e-6})
    return pd.DataFrame(conf)


def etapa_observado():
    t0 = time.perf_counter()
    df, feats, r, ridge = carregar()
    casos = montar(df, feats, r, ridge)
    conf = conferencia(df, feats, casos)
    print("=== Conferência do HAC ===\n" + conf.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
    if not conf.bate.all():
        raise SystemExit("CONFERÊNCIA FALHOU: nada mais foi calculado e nenhuma pasta foi criada.")
    out = os.path.join(ROOT, "outputs", "diagnostico_b2", datetime.now().strftime("%Y%m%d_%H%M%S") + "_passo3")
    os.makedirs(out, exist_ok=False)
    conf.to_csv(os.path.join(out, "conferencia_hac.csv"), index=False)

    # (caso, desenho, y): os quatro casos e, a mais, o omnibus de 16 variáveis com R_simples nas 987 datas
    alvos = [(c["nome"], nome_d, c) for c in casos.values() for nome_d in c["desenhos"]
             if not (c["tipo"] == "q1" and nome_d == "omnibus16" and c["ycol"] == "R_log")]
    linhas, dist = [], []
    for nome, nome_d, c in alvos:
        Z, ZZi = c["desenhos"][nome_d]
        y, n, q = c["y"], len(c["y"]), Z.shape[1] - 1
        W0, _ = wald(y, Z, LS, ZZi)
        Wy = em_paralelo(W_rotacoes_y, range(1, n), y, Z, ZZi)                    # k = 1 .. n − 1
        M = len(c["rc"])
        Wd = em_paralelo(W_rotacoes_diarias, range(1, M), c["rc"], n, c["ycol"] == "R_simples", Z, ZZi)
        for versao, Wk, ks, tam in (("rotacao_y", Wy, np.arange(1, n), n), ("rotacao_retornos_diarios", Wd, np.arange(1, M), M)):
            for g in GS:
                sel = (ks >= g) & (ks <= tam - g)
                for i, L in enumerate(LS):
                    p, ns = p_mc(W0[i], Wk[sel, i])
                    linhas.append({"caso": nome, "desenho": nome_d, "y": c["ycol"], "n": n, "q": q, "versao": versao,
                                   "principal": versao == "rotacao_y" and g == G_PRINCIPAL, "g": g, "L": L,
                                   "W": W0[i], "p_chi2": float(stats.chi2.sf(W0[i], q)), "K": int(sel.sum()),
                                   "n_singular": ns, "p_mc": p, "observacao": NOTAS})
            dist.append(pd.DataFrame({"caso": nome, "desenho": nome_d, "versao": versao, "k": ks,
                                      "W_L31": Wk[:, 0], "W_L61": Wk[:, 1]}))
        print(f"  rotação no observado: {nome} / {nome_d} ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    obs = pd.DataFrame(linhas)
    obs.to_csv(os.path.join(out, "mc_rotacao_observado.csv"), index=False)
    pd.concat(dist).to_csv(os.path.join(out, "distribuicoes_rotacao_observado.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    print("\n=== Teste de rotação no observado ===\n"
          + obs.drop(columns="observacao").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nNúcleos: {cpu_count()}; tempo: {time.perf_counter() - t0:.0f}s\nSaídas em {out}")


# ---------------------------------------------------------------------------
# Etapa 2: tamanho e poder
# ---------------------------------------------------------------------------
def gerar_y(c, proc, rho, rng):
    rc, n = c["rc"], len(c["y"])
    if proc == "S1":
        eps = rng.choice(rc - rc.mean(), size=n + H, replace=True)
    elif proc == "S2":
        eps = np.roll(rc, int(rng.integers(90, len(rc) - 90)))
    else:
        eps = rc[idx_circ(len(rc), 60, rng)]
    ruido = somas_30(eps, n)
    if c["ycol"] == "R_simples":
        ruido = 100 * np.expm1(ruido / 100)
    ruido = ruido - ruido.mean()
    s = 0.0 if rho == 0 else np.sqrt(rho / (1 - rho) * c["var_ruido"] / np.var(c["sinal"], ddof=1))
    return c["c"] + s * c["sinal"] + ruido


def p_bootstrap(y, Z, rng):
    W0, b = wald(y, Z, [L_BOOT])
    Wb = np.array([wald(y[j], Z[j], [L_BOOT], b0=b)[0][0]
                   for j in (idx_circ(len(y), BLOCO_BOOT, rng) for _ in range(B_BOOT))])
    v = Wb[np.isfinite(Wb)]
    return float((1 + np.sum(v >= W0[0])) / (len(v) + 1)), int(B_BOOT - len(v))


def uma_replica(c, proc, rho, rr):
    rng = np.random.default_rng([SEMENTE, 5, c["id"], PROC_ID[proc], int(round(rho * 10000)), rr])
    y = gerar_y(c, proc, rho, rng)
    linhas = []
    for nome_d, (Z, ZZi) in c["desenhos"].items():
        q = Z.shape[1] - 1
        W, _ = wald(y, Z, LS, ZZi)
        pb, nsb = p_bootstrap(y, Z, rng)
        Wk = W_rotacoes_y(y, Z, ZZi, range(G_PRINCIPAL, len(y) - G_PRINCIPAL + 1))
        p31, ns31 = p_mc(W[0], Wk[:, 0])
        p61, ns61 = p_mc(W[1], Wk[:, 1])
        linhas.append({"caso": c["nome"], "desenho": nome_d, "q": q, "processo": proc, "rho": rho, "r": rr,
                       "W_hac31": W[0], "p_hac31": float(stats.chi2.sf(W[0], q)),
                       "W_hac61": W[1], "p_hac61": float(stats.chi2.sf(W[1], q)),
                       "p_boot31": pb, "p_rot31": p31, "p_rot61": p61,
                       "n_singular": int(np.isnan(W).sum()) + nsb + ns31 + ns61})
    return linhas


def celulas(casos):
    for c in casos.values():
        for rho in RHOS[c["tipo"]]:
            for proc in (("S1", "S2", "S3") if rho == 0 else ("S1", "S2")):
                yield c, proc, rho, (R_NULO if rho == 0 else R_ALT)


TESTES = [("Wald HAC (χ²)", "p_hac31", 31), ("Wald HAC (χ²)", "p_hac61", 61),
          ("Wald bootstrap (bloco 60, B = 199)", "p_boot31", 31),
          ("Monte Carlo por rotação (g = 90)", "p_rot31", 31), ("Monte Carlo por rotação (g = 90)", "p_rot61", 61)]


def agregar(raw):
    linhas = []
    for (caso, des, proc, rho), g in raw.groupby(["caso", "desenho", "processo", "rho"]):
        nulo = raw[(raw.caso == caso) & (raw.desenho == des) & (raw.processo == proc) & (raw.rho == 0)]
        for teste, col, L in TESTES:
            linha = {"caso": caso, "desenho": des, "q": int(g.q.iloc[0]), "processo": proc, "rho": rho,
                     "teste": teste, "L": L, "R": len(g), "n_singular": int(g.n_singular.sum()),
                     **{f"rejeicao_{int(a * 100)}pct": float(np.mean(g[col] <= a)) for a in ALFAS}}
            if col.startswith("p_hac") and len(nulo):
                crit = float(np.quantile(nulo[f"W_hac{L}"], 0.95))
                linha.update({"crit_empirico_95_rho0": crit,
                              "poder_ajustado_5pct": float(np.mean(g[f"W_hac{L}"] > crit))})
            linha["observacao"] = NOTAS
            linhas.append(linha)
    return pd.DataFrame(linhas)


def etapa_simulacao(out):
    assert os.path.isfile(os.path.join(out, "conferencia_hac.csv")), "rode antes a etapa observado nesta pasta"
    final = os.path.join(out, "tamanho_poder.csv")
    if os.path.exists(final):
        raise SystemExit(f"já existe: {final} (não sobrescrevo)")
    dir_cel = os.path.join(out, "celulas")
    os.makedirs(dir_cel, exist_ok=True)
    tempos_csv = os.path.join(out, "tempo_celulas.csv")
    warnings.filterwarnings("ignore")
    t_ini = time.perf_counter()
    df, feats, r, ridge = carregar()
    casos = montar(df, feats, r, ridge)
    lista = list(celulas(casos))
    print(f"Núcleos: {cpu_count()}; {len(lista)} células", flush=True)
    for i, (c, proc, rho, R) in enumerate(lista, 1):
        arq = os.path.join(dir_cel, f"{c['nome']}_{proc}_rho{int(round(rho * 10000)):04d}.csv")
        if os.path.exists(arq):
            print(f"  [{i}/{len(lista)}] {os.path.basename(arq)} já existe: pulada", flush=True)
            continue
        t0 = time.perf_counter()
        res = Parallel(n_jobs=-1)(delayed(uma_replica)(c, proc, rho, rr) for rr in range(R))
        pd.DataFrame([x for bloco in res for x in bloco]).to_csv(arq, index=False)
        seg = time.perf_counter() - t0
        pd.DataFrame([{"caso": c["nome"], "processo": proc, "rho": rho, "R": R, "segundos": seg,
                       "nucleos": cpu_count()}]).to_csv(tempos_csv, mode="a", index=False,
                                                         header=not os.path.exists(tempos_csv))
        print(f"  [{i}/{len(lista)}] {c['nome']} {proc} ρ = {rho}: R = {R}, {seg:.0f}s "
              f"(total {(time.perf_counter() - t_ini) / 60:.1f} min)", flush=True)
    raw = pd.concat([pd.read_csv(os.path.join(dir_cel, f)) for f in sorted(os.listdir(dir_cel))])
    tp = agregar(raw)
    tp.to_csv(final, index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 1000)
    print("\n=== Tamanho e poder ===\n" + tp.drop(columns="observacao").to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nTempo total: {(time.perf_counter() - t_ini) / 60:.1f} min; saídas em {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", required=True, choices=["observado", "simulacao"])
    ap.add_argument("--out", help="pasta do passo 3 (etapa simulacao)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.etapa == "observado":
        etapa_observado()
    else:
        if not args.out:
            raise SystemExit("--out é obrigatório na etapa simulacao")
        etapa_simulacao(args.out)


if __name__ == "__main__":
    main()
