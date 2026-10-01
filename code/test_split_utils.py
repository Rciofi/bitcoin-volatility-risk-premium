"""
test_split_utils.py — testes de code/split_utils.py (T8)

O verificador de embargo é independente do gerador: usa as DATAS reais de
data/ml_dataset_T4.csv (e não as posições) e confere, para toda origem e
toda data de teste u, que max(datas de treino) <= u - h dias.

Confere, para h em {1, 5, 10, 20, 30, 60} e janela expansiva e móvel:
  1. embargo: nenhuma data de treino com s > u - h, para toda data de teste
     u (em particular, para a origem t);
  2. treino e teste sem interseção; testes contíguos, sem sobreposição e
     cobrindo todo o período fora da amostra;
  3. expansiva: treinos começam na 1ª data e crescem estritamente;
     móvel: todos os treinos com o mesmo tamanho;
  4. primeira janela: 730 obs., 23/04/2021 a 22/04/2023, igual em todas as
     combinações e inteiramente conhecida na primeira origem (max <= t0 - h);
     na expansiva, contida no treino da primeira origem; primeira origem em
     21/06/2023 para todos os h;
  5. validação cruzada do T7: mesmo embargo entre treino e validação da
     dobra, dobras dentro do treino da origem, sem interseção;
  6. controle positivo: divisões sem embargo (geradas com h = 0) e uma
     divisão montada à mão com a origem no treino precisam FALHAR;
  7. entradas inválidas são rejeitadas.

Uso:  python code/test_split_utils.py   (sai com código 1 se algo falhar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import (DivisaoTemporal, Origem, divisoes_validacao_cruzada,  # noqa: E402
                         gerar_divisoes)

ROOT = Path(__file__).resolve().parent.parent
HS = [1, 5, 10, 20, 30, 60]
JANELAS = ["expansiva", "movel"]
falhas = []


def check(cond, msg):
    print(("  OK    " if cond else "  FALHA ") + msg)
    if not cond:
        falhas.append(msg)


datas = pd.to_datetime(pd.read_csv(ROOT / "data" / "ml_dataset_T4.csv", usecols=["date"])["date"]).to_numpy()
N = len(datas)
check(N == 1776 and bool(np.all(np.diff(datas).astype("timedelta64[D]").astype(int) == 1)),
      f"ml_dataset_T4.csv: N = {N}, datas diárias contínuas")
um_dia = np.timedelta64(1, "D")


def violacoes_embargo(div, h):
    """Nº de pares (origem, data de teste u) com max(data de treino) > u - h dias."""
    v = 0
    for o in div.origens:
        ult_treino = datas[o.treino].max()
        v += int(np.sum(ult_treino > datas[o.teste] - h * um_dia))
    return v


def violacoes_cv(treino, val, h):
    return int(np.sum(datas[treino].max() > datas[val] - h * um_dia))


# 1-4) divisões
divs = {}
for jan in JANELAS:
    for h in HS:
        d = gerar_divisoes(N, h=h, janela=jan)
        divs[(jan, h)] = d
        rot = f"{jan:9s} h={h:2d}"
        check(violacoes_embargo(d, h) == 0,
              f"{rot}: embargo em {len(d.origens)} origens e {d.n_fora_da_amostra} datas de teste")
        inter = sum(len(np.intersect1d(o.treino, o.teste)) for o in d.origens)
        testes = np.concatenate([o.teste for o in d.origens])
        cobre = np.array_equal(testes, np.arange(d.primeira_origem, N))
        check(inter == 0 and cobre, f"{rot}: treino ∩ teste vazio; testes contíguos cobrindo [t0, fim]")
        tam = np.array([len(o.treino) for o in d.origens])
        if jan == "expansiva":
            ok = all(o.treino[0] == 0 for o in d.origens) and bool(np.all(np.diff(tam) > 0))
            check(ok, f"{rot}: treinos começam na 1ª data e crescem ({tam[0]} → {tam[-1]})")
        else:
            check(bool(np.all(tam == d.n_movel)), f"{rot}: todos os treinos com {d.n_movel} obs.")

print("4) Primeira janela e primeira origem")
pj = [d.primeira_janela for d in divs.values()]
d30 = divs[("expansiva", 30)]
check(all(np.array_equal(p, pj[0]) for p in pj), "primeira janela idêntica em todas as combinações")
check(len(pj[0]) == 730 and str(datas[pj[0][0]])[:10] == "2021-04-23" and str(datas[pj[0][-1]])[:10] == "2023-04-22",
      f"primeira janela: {len(pj[0])} obs., {str(datas[pj[0][0]])[:10]} a {str(datas[pj[0][-1]])[:10]}")
t0s = {str(datas[d.primeira_origem])[:10] for d in divs.values()}
check(t0s == {"2023-06-21"}, f"primeira origem igual para todos os h: {sorted(t0s)}")
# O T9 precisa que a primeira janela inteira já seja conhecida na 1ª origem
# (alvos realizados): max(primeira janela) <= t0 - h, em todas as combinações.
check(all(datas[d.primeira_janela].max() <= datas[d.primeira_origem] - d.h * um_dia for d in divs.values()),
      "primeira janela inteira conhecida na primeira origem (max <= t0 - h), todas as combinações")
# Na expansiva, ela está contida no treino da 1ª origem. Na móvel com h < 60,
# a janela de 730 obs. já deslizou (começa em 60 - h) -- comportamento esperado.
check(all(np.isin(d.primeira_janela, d.origens[0].treino).all()
          for (jan, _), d in divs.items() if jan == "expansiva"),
      "expansiva: primeira janela contida no treino da primeira origem")

print("5) Validação cruzada do T7 (dentro do treino, mesmo embargo)")
for h in (30, 60):
    d = divs[("expansiva", h)]
    for o in (d.origens[0], d.origens[len(d.origens) // 2], d.origens[-1]):
        dobras = divisoes_validacao_cruzada(o.treino, h=h, n_dobras=5)
        v = sum(violacoes_cv(tr, va, h) for tr, va in dobras)
        dentro = all(np.isin(tr, o.treino).all() and np.isin(va, o.treino).all() for tr, va in dobras)
        disj = all(len(np.intersect1d(tr, va)) == 0 for tr, va in dobras)
        cresc = all(dobras[k][0][-1] < dobras[k + 1][0][-1] for k in range(len(dobras) - 1))
        check(v == 0 and dentro and disj and cresc,
              f"h={h}, origem {str(datas[o.t])[:10]}: 5 dobras embargadas, dentro do treino, "
              f"sem interseção e expansivas")

print("6) Controle positivo: divisões sem embargo precisam falhar")
for jan in JANELAS:
    sem = gerar_divisoes(N, h=0, janela=jan, h_primeira_origem=30)
    check(violacoes_embargo(sem, 30) > 0,
          f"{jan}: divisão gerada com h = 0, verificada com h = 30 -> {violacoes_embargo(sem, 30)} violações")
o = divs[("expansiva", 30)].origens[0]
vaz = DivisaoTemporal(h=30, janela="expansiva", n_inicial=730, n_movel=730, freq=30,
                      primeira_janela=np.arange(730),
                      origens=(Origem(t=o.t, treino=np.arange(0, o.t + 1), teste=o.teste),))
check(violacoes_embargo(vaz, 30) > 0, "divisão montada à mão com a origem t no treino é detectada")
tr, va = divisoes_validacao_cruzada(o.treino, h=0, n_dobras=5)[2]
check(violacoes_cv(tr, va, 30) > 0, "validação cruzada sem embargo (h = 0), verificada com h = 30, é detectada")

print("7) Entradas inválidas")
for kw in ({"h": -1}, {"h": 30, "janela": "outra"}, {"h": 61, "h_primeira_origem": 60},
           {"h": 30, "n_inicial": 1760}, {"h": 2.5}):
    try:
        gerar_divisoes(N, **kw)
        check(False, f"{kw} deveria ser rejeitado")
    except ValueError:
        check(True, f"{kw} rejeitado")

print("\nResumo (janela expansiva):")
for h in (30, 60):
    d = divs[("expansiva", h)]
    o0, o1 = d.origens[0], d.origens[-1]
    print(f"  h={h}: {len(d.origens)} origens, {d.n_fora_da_amostra} datas fora da amostra "
          f"({str(datas[o0.t])[:10]} a {str(datas[N - 1])[:10]}); treino da 1ª origem: {len(o0.treino)} obs. "
          f"até {str(datas[o0.treino[-1]])[:10]}; da última: {len(o1.treino)} obs.")

print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S)")
sys.exit(1 if falhas else 0)
