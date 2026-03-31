import os
import pandas as pd
import numpy as np


def classify_regime(vrp, q1, q2):
    """
    Classifica VRP em 3 regimes:
    - Baixo: vrp <= q1
    - Medio: q1 < vrp <= q2
    - Alto: vrp > q2
    """
    if vrp <= q1:
        return "Baixo"
    elif vrp <= q2:
        return "Médio"
    else:
        return "Alto"


def main():
    # Diretório base (um nível acima da pasta code)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo dataset com VRP e retornos futuros em: {data_path}")
    df = pd.read_csv(data_path)

    # Converte data
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Garante que colunas necessárias existem
    for col in ["vrp_30d", "ret_fut_1d", "ret_fut_5d", "ret_fut_20d"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada em vrp_with_targets.csv")

    # ===============================
    # 1) Definir tercis de VRP (Q1, Q2)
    # ===============================

    q1 = df["vrp_30d"].quantile(1/3)
    q2 = df["vrp_30d"].quantile(2/3)

    print("\nQuantis de VRP 30D (em p.p. de vol):")
    print(f"  Q1 (1/3): {q1:.2f}")
    print(f"  Q2 (2/3): {q2:.2f}")

    # Cria coluna de regime
    df["vrp_regime"] = df["vrp_30d"].apply(lambda x: classify_regime(x, q1, q2))

    # ===============================
    # 2) Estatísticas por regime
    # ===============================

    group_cols = ["vrp_regime"]

    stats = (
        df
        .groupby(group_cols)
        .agg(
            contagem=("vrp_30d", "count"),
            vrp_medio=("vrp_30d", "mean"),
            ret_fut_1d_medio=("ret_fut_1d", "mean"),
            ret_fut_5d_medio=("ret_fut_5d", "mean"),
            ret_fut_20d_medio=("ret_fut_20d", "mean"),
        )
        .reset_index()
        .sort_values("vrp_medio")
    )

    # Converte retornos para %
    for col in ["ret_fut_1d_medio", "ret_fut_5d_medio", "ret_fut_20d_medio"]:
        stats[col] = stats[col] * 100.0

    print("\n=== ESTATÍSTICAS POR REGIME DE VRP ===")
    print(stats)

    # Salva tabela em CSV
    out_stats_path = os.path.join(base_dir, "data", "vrp_regime_stats.csv")
    stats.to_csv(out_stats_path, index=False)
    print(f"\nTabela de regimes salva em: {out_stats_path}")

    # ===============================
    # 3) (Opcional) salva dataset com regimes
    # ===============================

    out_df_path = os.path.join(base_dir, "data", "vrp_with_regimes.csv")
    df.to_csv(out_df_path, index=False)
    print(f"Dataset com coluna 'vrp_regime' salvo em: {out_df_path}")


if __name__ == "__main__":
    main()
