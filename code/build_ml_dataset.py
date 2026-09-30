import os
import pandas as pd
import numpy as np


def main():
    # Caminho base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Tarefa 3 (Passo 3): fonte com regimes — vrp_with_regimes (regenerado com 1.775 obs)
    data_path = os.path.join(base_dir, "data", "vrp_with_regimes.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {data_path}")

    print(f"Lendo dataset VRP+regimes: {data_path}")
    df = pd.read_csv(data_path)

    # =====================
    # PREPARACAO BASICA
    # =====================
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Entrada: {len(df)} obs | {df['date'].min().date()} -> {df['date'].max().date()}")

    # =====================
    # FEATURES TEMPORAIS
    # =====================
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # =====================
    # LAGS DE RETORNO
    # =====================
    df["ret_lag_1"]  = df["ret"].shift(1)
    df["ret_lag_5"]  = df["ret"].rolling(5).sum().shift(1)
    df["ret_lag_20"] = df["ret"].rolling(20).sum().shift(1)

    # =====================
    # MUDANCAS EM VOLATILIDADE
    # =====================
    df["d_iv_1d"]  = df["iv_30d"].diff()
    df["d_vrp_1d"] = df["vrp_30d"].diff()

    # =====================
    # REGIME NUMERICO
    # =====================
    # Tarefa 3 (Passo 3): vrp_with_regimes.csv regenerado com dados atualizados
    regime_map = {"Baixo": 0, "Médio": 1, "Alto": 2}
    df["vrp_regime_num"] = df["vrp_regime"].map(regime_map)

    # =====================
    # REMOVE MISSING DATA
    # =====================
    # dropna apenas nas colunas de feature efetivamente usadas
    feature_cols = [
        "ret", "rv_30d", "iv_30d", "vrp_30d",
        "ret_fut_1d", "ret_fut_5d", "ret_fut_20d", "ret_fut_60d",
        "ret_lag_1", "ret_lag_5", "ret_lag_20",
        "d_iv_1d", "d_vrp_1d",
        "month", "weekday", "is_month_start", "is_month_end",
        "vrp_regime_num",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    # =====================
    # SALVAR RESULTADO
    # =====================
    # Remove close (serie I(1)) do CSV de saida — nao deve ser usada como feature (E3)
    # Remove tambem as colunas prospectivas do T1 (RV de t+1 a t+30): conhecidas
    # so em t+30. regenerate_cap8.py usa como feature toda coluna fora de
    # COLS_EXCLUIR, entao elas vazariam dados do futuro sem aviso. O alvo
    # prospectivo entra no dataset de AM no T5, de forma explicita.
    cols_drop = [c for c in ["close", "rv_30d_fut", "bvrp_30d_fut"] if c in df_clean.columns]
    df_out = df_clean.drop(columns=cols_drop)
    out_path = os.path.join(base_dir, "data", "ml_dataset.csv")
    df_out.to_csv(out_path, index=False)

    print("\n=== DATASET DE MACHINE LEARNING GERADO COM SUCESSO ===")
    print(f"Arquivo salvo em: {out_path}")
    print(f"Numero de linhas: {len(df_out)}")
    print(f"Periodo: {df_out['date'].min().date()} -> {df_out['date'].max().date()}")
    print("\nDistribuicao dos regimes:")
    print(df_out["vrp_regime"].value_counts().sort_index())
    print("\nColunas disponiveis (features + targets):")
    print(df_out.columns.tolist())


if __name__ == "__main__":
    main()
