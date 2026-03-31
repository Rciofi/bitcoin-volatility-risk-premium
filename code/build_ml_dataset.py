import os
import pandas as pd
import numpy as np


def main():
    # Caminho base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_with_regimes.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo dataset VRP+regimes: {data_path}")
    df = pd.read_csv(data_path)

    # =====================
    # PREPARAÇÃO BÁSICA
    # =====================
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # =====================
    # FEATURES TEMPORAIS
    # =====================
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday    # 0 = segunda, 6 = domingo
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # =====================
    # LAGS DE RETORNO
    # =====================
    df["ret_lag_1"] = df["ret"].shift(1)
    df["ret_lag_5"] = df["ret"].rolling(5).sum().shift(1)
    df["ret_lag_20"] = df["ret"].rolling(20).sum().shift(1)

    # =====================
    # MUDANÇAS EM VOLATILIDADE
    # =====================
    df["d_iv_1d"] = df["iv_30d"].diff()
    df["d_vrp_1d"] = df["vrp_30d"].diff()

    # =====================
    # REGIME NUMÉRICO
    # =====================
    regime_map = {"Baixo": 0, "Médio": 1, "Alto": 2}
    df["vrp_regime_num"] = df["vrp_regime"].map(regime_map)

    # =====================
    # REMOVE MISSING DATA
    # =====================
    df_clean = df.dropna().reset_index(drop=True)

    # =====================
    # SALVAR RESULTADO
    # =====================
    out_path = os.path.join(base_dir, "data", "ml_dataset.csv")
    df_clean.to_csv(out_path, index=False)

    print("\n=== DATASET DE MACHINE LEARNING GERADO COM SUCESSO ===")
    print(f"Arquivo salvo em: {out_path}")
    print(f"Número de linhas: {len(df_clean)}")
    print("\nColunas disponíveis (features + targets):")
    print(df_clean.columns.tolist())


if __name__ == "__main__":
    main()
