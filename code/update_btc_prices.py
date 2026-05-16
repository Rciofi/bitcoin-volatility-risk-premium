"""
Atualiza btc_prices.csv com preços diários BTCUSDT da API pública da Binance.

Comportamento:
- Se btc_prices.csv já existe, busca só os dados faltantes (a partir do dia seguinte ao último)
- Se não existe, faz download completo desde 2017-08-17

Uso:
    python code/update_btc_prices.py
"""

import csv
import os
import time
from datetime import datetime, timezone, date
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "btc_prices.csv"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
START_FULL = date(2017, 8, 17)


def fetch_klines(start: date, end: date) -> list[tuple[date, float]]:
    """Busca candles diários BTCUSDT entre start e end (inclusive)."""
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime(end.year,   end.month,   end.day,   23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)

    results = []
    while start_ms < end_ms:
        params = {
            "symbol":    "BTCUSDT",
            "interval":  "1d",
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000,
        }
        r = requests.get(BINANCE_URL, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()

        if not rows:
            break

        for row in rows:
            dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).date()
            close = float(row[4])
            results.append((dt, close))

        # avança janela para depois do último candle recebido
        last_ts = rows[-1][0]
        start_ms = last_ts + 86_400_000  # + 1 dia em ms

        if len(rows) < 1000:
            break

        time.sleep(0.2)  # respeita rate limit

    return results


def load_existing() -> dict[date, float]:
    if not OUT_PATH.exists():
        return {}
    data = {}
    with open(OUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
            data[dt] = float(row["close"])
    return data


def save(data: dict[date, float]):
    DATA_DIR.mkdir(exist_ok=True)
    rows = sorted(data.items())
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        for dt, close in rows:
            writer.writerow([dt.isoformat(), close])


def main():
    existing = load_existing()

    if existing:
        last_date = max(existing.keys())
        start = date(last_date.year, last_date.month, last_date.day)
        from datetime import timedelta
        start = last_date + timedelta(days=1)
        print(f"btc_prices.csv existente — último registro: {last_date}")
        print(f"Buscando dados a partir de {start}...\n")
    else:
        start = START_FULL
        print(f"Arquivo não encontrado — download completo desde {start}\n")

    today = datetime.now(timezone.utc).date()

    if start > today:
        print("Dados já estão atualizados.")
        return

    new_rows = fetch_klines(start, today)

    if not new_rows:
        print("Nenhum dado novo retornado pela Binance.")
        return

    print(f"Recebidos {len(new_rows)} novos registros.")
    existing.update({dt: close for dt, close in new_rows})
    save(existing)

    final_last = max(existing.keys())
    print(f"\n✔ btc_prices.csv atualizado: {min(existing.keys())} → {final_last}")
    print(f"  Total de registros: {len(existing)}")


if __name__ == "__main__":
    main()
