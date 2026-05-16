"""
build_bvrp_fut_target.py — Tarefa 3, Passo 4
Gera data/bvrp_ml_target_fut_1d.csv a partir de vrp_with_targets.csv.

Estrutura de saida:
  date, close, ret, rv_30d, iv_30d, bvrp_30d, ret_fut_1d,
  ret_fut_5d, ret_fut_20d, bvrp_fut_1d

Onde:
  bvrp_30d   = vrp_30d (RV - IV, sinal correto — Decisao 1, Fase 0)
  bvrp_fut_1d = bvrp_30d do dia seguinte (alvo de previsao 1-step-ahead)
"""
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")
    out_path = os.path.join(base_dir, "data", "bvrp_ml_target_fut_1d.csv")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {src_path}")

    df = pd.read_csv(src_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Lendo: {src_path}")
    print(f"  Entrada: {len(df)} obs | {df['date'].min().date()} -> {df['date'].max().date()}")

    # Renomear vrp_30d -> bvrp_30d (explicitar que e o BVRP = RV - IV)
    df = df.rename(columns={"vrp_30d": "bvrp_30d"})

    # bvrp_fut_1d = BVRP do proximo dia (alvo de previsao)
    df["bvrp_fut_1d"] = df["bvrp_30d"].shift(-1)

    # Selecionar colunas e remover ultima linha (NaN do shift(-1))
    cols = [
        "date", "close", "ret", "rv_30d", "iv_30d",
        "bvrp_30d", "ret_fut_1d", "ret_fut_5d", "ret_fut_20d", "bvrp_fut_1d"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna(subset=["bvrp_fut_1d"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)

    print(f"\n=== bvrp_ml_target_fut_1d.csv GERADO ===")
    print(f"Arquivo: {out_path}")
    print(f"Obs:     {len(df)}")
    print(f"Periodo: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"\nEstatisticas de bvrp_30d (deve ser negativo em media):")
    print(df["bvrp_30d"].describe().round(4))

if __name__ == "__main__":
    main()
