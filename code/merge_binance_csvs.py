#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge dos CSVs de candles diários BTCUSDT (Binance) em um único btc_prices.csv.

Esperado:
- Arquivos CSV da Binance em: ../binance_raw/
  (todos os .csv extraídos dos .zip mensais BTCUSDT-1d-AAAA-MM.csv)

Saída:
- ../data/btc_prices.csv   com colunas: date, close
"""

import os
import glob
import pandas as pd


def merge_binance_klines():
    # pasta raiz (um nível acima da pasta code)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_dir = os.path.join(base_dir, "binance_raw")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    pattern = os.path.join(raw_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {raw_dir}. "
            "Verifique se você extraiu os .zip da Binance para essa pasta."
        )

    print(f"Encontrados {len(files)} arquivos CSV em {raw_dir}")
    frames = []

    for f in files:
        print(f"Lendo {os.path.basename(f)} ...")

        # CSV da Binance normalmente não tem header
        df = pd.read_csv(f, header=None)

        # checagem mínima de formato
        if df.shape[1] < 5:
            raise ValueError(f"Arquivo {f} tem menos de 5 colunas, formato inesperado.")

        # coluna 4 = close
        close = pd.to_numeric(df[4], errors="coerce")

        # coluna 0 = open time (epoch em ms)
        ts_raw = pd.to_numeric(df[0], errors="coerce")

        # converte para datetime, valores ruins viram NaT
        dt = pd.to_datetime(ts_raw, unit="ms", errors="coerce")

        tmp = pd.DataFrame({"date": dt, "close": close})
        tmp = tmp.dropna(subset=["date", "close"])

        # usa só a data (sem horário)
        tmp["date"] = tmp["date"].dt.date

        frames.append(tmp)

    # concatena tudo
    all_df = pd.concat(frames, ignore_index=True)

    # garante datetime e ordenação
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values("date").drop_duplicates(subset="date", keep="last")

    out_path = os.path.join(data_dir, "btc_prices.csv")
    all_df.to_csv(out_path, index=False)

    print("Merge concluído!")
    print(f"Arquivo final salvo em: {out_path}")
    print(f"Número de linhas (dias): {len(all_df)}")


if __name__ == "__main__":
    merge_binance_klines()
