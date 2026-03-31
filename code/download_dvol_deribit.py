import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


def fetch_dvol_chunk(start_ts, end_ts):
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

    params = {
        "currency": "BTC",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": "1D",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "result" not in data or "data" not in data["result"]:
        return []

    return data["result"]["data"]


def download_dvol_full():
    print("\nBaixando DVOL 30D HISTÓRICO COMPLETO da Deribit...\n")

    end_dt = datetime.now(timezone.utc)
    end_ts = int(end_dt.timestamp() * 1000)

    chunks = []
    total_rows = 0
    batch_size = 1000

    while True:
        start_dt = end_dt - timedelta(days=batch_size * 2)
        start_ts = int(start_dt.timestamp() * 1000)

        print(f"Baixando chunk: {start_dt.date()} → {end_dt.date()}")

        chunk = fetch_dvol_chunk(start_ts, end_ts)

        if not chunk:
            print("Fim dos dados ou chunk vazio.")
            break

        chunks.extend(chunk)
        total_rows += len(chunk)

        # Atualiza janela
        end_dt = start_dt

        # Se chegou antes de 2017, para
        if end_dt.year < 2017:
            break

    print(f"\nTotal baixado: {total_rows} linhas")

    # Converte para DataFrame
    df = pd.DataFrame(
        chunks, columns=["timestamp_ms", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    df = df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)
    df["dvol_30d"] = df["close"] * 100.0  # DVOL em %

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "dvol_30d_full.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✔ Arquivo completo salvo em:\n{out_path}")
    print(f"Linhas finais: {len(df)}")


if __name__ == "__main__":
    download_dvol_full()
