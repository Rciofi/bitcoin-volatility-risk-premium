"""
fix_btc_gap_2023_03.py — T0 do plano de revisao (set/2026)

Preenche o buraco de 01/03/2023 a 31/03/2023 em data/btc_prices.csv.

O buraco ja existia na primeira versao versionada do arquivo (commit 315d9fb);
causa provavel: o CSV mensal BTCUSDT-1d-2023-03 faltava em binance_raw/ quando
merge_binance_csvs.py rodou. Efeitos antes da correcao: o "retorno diario" de
01/04/2023 cobria 32 dias (log 28452,73/23141,57 = 0,2066, o maximo da Tab. 3.1),
as janelas de RV que atravessavam o buraco misturavam periodos e o merge com o
DVOL perdia 31 observacoes.

Mesma fonte e convencao de update_btc_prices.py: API publica da Binance,
BTCUSDT, velas de 1d, data = horario de ABERTURA da vela (00:00 UTC), close da vela.

Busca 27/02/2023 a 02/04/2023 (35 velas): as 4 datas das bordas ja existem no
arquivo e servem de controle -- os closes da API precisam bater com os do arquivo.

Uso:
  python code/fix_btc_gap_2023_03.py            # so consulta e valida (nao grava)
  python code/fix_btc_gap_2023_03.py --write    # insere as 31 linhas; as existentes nao sao tocadas
"""
import sys
from datetime import date
from pathlib import Path

from update_btc_prices import fetch_klines

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "btc_prices.csv"
GAP_START, GAP_END = date(2023, 3, 1), date(2023, 3, 31)
CONTROLE = [date(2023, 2, 27), date(2023, 2, 28), date(2023, 4, 1), date(2023, 4, 2)]


def main():
    api = dict(fetch_klines(date(2023, 2, 27), date(2023, 4, 2)))
    lines = CSV.read_text(encoding="utf-8").splitlines()
    existing = {l.split(",")[0]: l.split(",")[1] for l in lines[1:]}

    # 1) validacoes
    assert len(api) == 35, f"esperava 35 velas, recebi {len(api)}"
    for d in CONTROLE:
        assert float(existing[d.isoformat()]) == api[d], (
            f"controle diverge em {d}: arquivo={existing[d.isoformat()]} api={api[d]}")
    novos = {d: c for d, c in api.items() if GAP_START <= d <= GAP_END}
    assert len(novos) == 31, f"esperava 31 dias novos, recebi {len(novos)}"
    assert not any(d.isoformat() in existing for d in novos), "alguma data do buraco ja existe no arquivo"
    print("Controles OK (4 datas das bordas batem com o arquivo).")
    for d, c in sorted(novos.items()):
        print(f"  {d},{c}")

    # 2) gravacao (so com --write)
    if "--write" not in sys.argv:
        print("\n(dry-run: nada gravado; rode com --write para inserir)")
        return
    rows = [(l.split(",")[0], l) for l in lines[1:]]
    rows += [(d.isoformat(), f"{d.isoformat()},{c}") for d, c in novos.items()]
    rows.sort(key=lambda t: t[0])
    CSV.write_text("\n".join([lines[0]] + [r for _, r in rows]) + "\n", encoding="utf-8", newline="\n")
    print(f"Gravado: {CSV} ({len(rows)} linhas de dados)")


if __name__ == "__main__":
    main()
