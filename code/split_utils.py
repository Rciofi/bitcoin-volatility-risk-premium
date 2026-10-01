"""
split_utils.py — janela de estimação e divisão treino/teste (T8 do plano de revisão)

Gera, para cada origem de previsão t, os índices de treino e de teste, com
EMBARGO obrigatório: o treino da origem t só usa datas s com s <= t - h,
porque o alvo de s (que olha h dias à frente) só é conhecido em s + h.
  - h = 30 no Cap. 5 (alvo bvrp_30d_fut);
  - h = horizonte do retorno nos Caps. 6 e 7 (até 60).

Desenho (docs/pendencias_T1.md, seção 7):
  - Janela EXPANSIVA como principal (N = 1.776 é pequeno); MÓVEL, de tamanho
    fixo, como teste de robustez.
  - Primeira janela de estimação: as n_inicial primeiras observações
    (padrão 730 = 2 anos: 23/04/2021 a 22/04/2023 em ml_dataset_T4.csv).
    É devolvida como objeto próprio porque o T9 calcula o corte dos regimes
    só com ela. Ela é a mesma para todos os capítulos.
  - Primeira origem: t0 = n_inicial - 1 + h_primeira_origem. Com o padrão
    h_primeira_origem = 60 (maior horizonte do plano), o período fora da
    amostra é o mesmo nos Caps. 5 a 7 (começa em 21/06/2023).
  - Reestimação a cada `freq` dias (padrão 30): o modelo estimado na origem
    t prevê as datas t, t+1, ..., até a véspera da próxima origem, sempre com
    as variáveis explicativas da própria data. Como s <= t - h <= u - h para
    toda data de teste u >= t, o embargo vale para todas elas.
  - Validação cruzada do T7 (divisoes_validacao_cruzada): dobras expansivas
    dentro do treino, com o MESMO embargo: numa dobra cuja validação começa
    em v, o treino da dobra usa s <= v - h.

Tudo em índices POSICIONAIS de uma série diária contínua (ml_dataset_T4.csv
passa pela checagem de continuidade do T0), de modo que k posições = k dias.
"""

from dataclasses import dataclass

import numpy as np

JANELAS = ("expansiva", "movel")
N_INICIAL_PADRAO = 730
FREQ_PADRAO = 30
H_PRIMEIRA_ORIGEM_PADRAO = 60


@dataclass(frozen=True)
class Origem:
    """Uma origem de previsão: estima com `treino` e prevê `teste`."""
    t: int
    treino: np.ndarray
    teste: np.ndarray


@dataclass(frozen=True)
class DivisaoTemporal:
    """Resultado de gerar_divisoes."""
    h: int
    janela: str
    n_inicial: int
    n_movel: int
    freq: int
    primeira_janela: np.ndarray  # índices da primeira janela de estimação (para o T9)
    origens: tuple               # tuple[Origem, ...]

    @property
    def primeira_origem(self):
        return self.origens[0].t

    @property
    def n_fora_da_amostra(self):
        return int(sum(len(o.teste) for o in self.origens))


def _inteiro(v, nome, minimo):
    if isinstance(v, (bool, np.bool_)) or int(v) != v or v < minimo:
        raise ValueError(f"{nome} deve ser inteiro >= {minimo}, recebido {v!r}")
    return int(v)


def gerar_divisoes(n, *, h, janela="expansiva", n_inicial=N_INICIAL_PADRAO,
                   freq=FREQ_PADRAO, n_movel=None, h_primeira_origem=H_PRIMEIRA_ORIGEM_PADRAO):
    """
    Origens de previsão com treino embargado.

    Parâmetros
    ----------
    n : int
        Número de observações da série (posições 0 .. n-1).
    h : int (keyword)
        Horizonte do alvo em dias; treino da origem t: s <= t - h.
    janela : {"expansiva", "movel"}
        Expansiva: treino = [0, t - h]. Móvel: treino = as n_movel últimas
        posições até t - h (tamanho fixo).
    n_inicial : int
        Tamanho da primeira janela de estimação (posições 0 .. n_inicial-1).
    freq : int
        Dias entre reestimações; teste de cada origem = [t, t + freq).
    n_movel : int ou None
        Tamanho da janela móvel (padrão: n_inicial).
    h_primeira_origem : int
        Horizonte usado para posicionar a primeira origem,
        t0 = n_inicial - 1 + h_primeira_origem. Deve ser >= h, para que o
        treino da primeira origem contenha toda a primeira janela.

    Retorna
    -------
    DivisaoTemporal
    """
    n = _inteiro(n, "n", 1)
    h = _inteiro(h, "h", 0)
    n_inicial = _inteiro(n_inicial, "n_inicial", 1)
    freq = _inteiro(freq, "freq", 1)
    h0 = _inteiro(h_primeira_origem, "h_primeira_origem", 0)
    n_movel = n_inicial if n_movel is None else _inteiro(n_movel, "n_movel", 1)
    if janela not in JANELAS:
        raise ValueError(f"janela deve ser um de {JANELAS}, recebido {janela!r}")
    if h0 < h:
        raise ValueError(f"h_primeira_origem ({h0}) deve ser >= h ({h})")
    if janela == "movel" and n_movel > n_inicial + (h0 - h):
        raise ValueError("n_movel maior que o treino disponível na primeira origem")

    t0 = n_inicial - 1 + h0
    if t0 >= n:
        raise ValueError(f"primeira origem ({t0}) fora da série (n = {n})")

    origens = []
    for t in range(t0, n, freq):
        fim = t - h  # última posição de treino (inclusive)
        inicio = 0 if janela == "expansiva" else fim - n_movel + 1
        origens.append(Origem(t=t,
                              treino=np.arange(inicio, fim + 1),
                              teste=np.arange(t, min(t + freq, n))))
    return DivisaoTemporal(h=h, janela=janela, n_inicial=n_inicial, n_movel=n_movel,
                           freq=freq, primeira_janela=np.arange(n_inicial),
                           origens=tuple(origens))


def divisoes_validacao_cruzada(idx_treino, *, h, n_dobras=5):
    """
    Validação cruzada temporal do T7, dentro de um conjunto de treino.

    Divide idx_treino (contíguo, crescente) em n_dobras + 1 blocos; a dobra k
    (k = 1..n_dobras) valida no bloco k e treina com as posições de idx_treino
    que satisfazem s <= v - h, em que v é a primeira posição da validação
    (mesmo embargo da previsão). Dobras expansivas; a última validação vai até
    o fim de idx_treino.

    Retorna
    -------
    list[tuple[np.ndarray, np.ndarray]]  -- (treino_da_dobra, validacao)
    """
    idx = np.asarray(idx_treino)
    h = _inteiro(h, "h", 0)
    n_dobras = _inteiro(n_dobras, "n_dobras", 1)
    if len(idx) == 0 or np.any(np.diff(idx) != 1):
        raise ValueError("idx_treino deve ser contíguo e crescente")
    tam = len(idx) // (n_dobras + 1)
    dobras = []
    for k in range(1, n_dobras + 1):
        ini = k * tam
        fim = (k + 1) * tam if k < n_dobras else len(idx)
        val = idx[ini:fim]
        tr = idx[idx <= val[0] - h]
        if len(tr) == 0:
            raise ValueError(f"dobra {k}: treino vazio com embargo h = {h}")
        dobras.append((tr, val))
    return dobras
