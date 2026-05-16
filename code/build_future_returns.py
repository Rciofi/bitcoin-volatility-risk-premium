import os
import pandas as pd
import numpy as np


def main():
    # Diretório base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_30d_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo dataset VRP em: {data_path}")
    df = pd.read_csv(data_path)

    # Converte data
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # =======================
    #  CRIA RETORNOS FUTUROS
    # =======================

    df["ret_fut_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["ret_fut_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["ret_fut_20d"] = df["close"].shift(-20) / df["close"] - 1

    # Remove linhas sem target futuro (últimos 20 dias)
    df_clean = df.dropna().reset_index(drop=True)

    # Salvar dataset expandido
    out_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")
    df_clean.to_csv(out_path, index=False)

    print("\n=== RETORNOS FUTUROS GERADOS COM SUCESSO ===")
    print(f"Arquivo salvo em: {out_path}")
    print(f"Linhas finais: {len(df_clean)}")
    print("\nColunas criadas: ret_fut_1d, ret_fut_5d, ret_fut_20d")
    print("\nResumo estatístico dos retornos futuros:")
    print(df_clean[['ret_fut_1d','ret_fut_5d','ret_fut_20d']].describe())


if __name__ == "__main__":
    main()
