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
    print("\nBaixando DVOL 30D HISTORICO COMPLETO da Deribit...\n")

    # DVOL lancado em 2021-01-11; vai ate hoje em chunks de 180 dias
    start_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end_dt   = datetime.now(timezone.utc)
    chunk_days = 180

    all_rows = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=chunk_days), end_dt)
        start_ts = int(cur.timestamp() * 1000)
        end_ts   = int(nxt.timestamp() * 1000)

        print(f"Baixando chunk: {cur.date()} a {nxt.date()}")
        chunk = fetch_dvol_chunk(start_ts, end_ts)
        if chunk:
            all_rows.extend(chunk)
        cur = nxt

    print(f"\nTotal baixado: {len(all_rows)} linhas")

    # Converte para DataFrame
    df = pd.DataFrame(
        all_rows, columns=["timestamp_ms", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    df = df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)
    df["dvol_30d"] = df["close"] * 100.0  # DVOL em %

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "dvol_30d_full.csv")
    df.to_csv(out_path, index=False)

    print(f"\nArquivo completo salvo em:\n{out_path}")
    print(f"Linhas finais: {len(df)}")


if __name__ == "__main__":
    download_dvol_full()
