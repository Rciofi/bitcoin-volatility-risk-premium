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
    # 1) Definir tercis de VRP — janela expansiva (E4)
    # ===============================
    # CORRECAO E4: limiares calculados via expanding window com min_periods=252
    # para eliminar vies de antecipacao (lookahead bias).
    # As primeiras 251 obs (2021-03-24 a 2021-11-29) ficam sem rotulo (NaN)
    # e sao removidas automaticamente pelo dropna em build_ml_dataset.py.
    # Amostra efetiva para modelos com regime: ~1.504 obs (a partir de 2021-11-30).
    # Analises sem regime (Cap. 4, Cap. 6 modelos lineares): usam 1.755 obs completas.
    MIN_PERIODS_REGIME = 252

    q1_series = df["vrp_30d"].expanding(min_periods=MIN_PERIODS_REGIME).quantile(1/3)
    q2_series = df["vrp_30d"].expanding(min_periods=MIN_PERIODS_REGIME).quantile(2/3)

    # Cria coluna de regime — NaN para obs sem historico suficiente
    df["vrp_regime"] = [
        classify_regime(v, r1, r2) if pd.notna(r1) else np.nan
        for v, r1, r2 in zip(df["vrp_30d"], q1_series, q2_series)
    ]

    # Relatorio de limiares em pontos representativos
    checkpoints = {252: "dia 252 (primeiro rotulo)", 500: "dia 500",
                   1000: "dia 1000", len(df): "dia final"}
    print("\nEvolucao dos limiares (janela expansiva, min_periods=252):")
    print(f"  {'Ponto':<28} {'Data':>12}  {'Q1':>10}  {'Q2':>10}")
    for idx_1, label in checkpoints.items():
        i = min(idx_1 - 1, len(df) - 1)
        print(f"  {label:<28} {str(df['date'].iloc[i].date()):>12}  "
              f"{q1_series.iloc[i]:>+10.4f}  {q2_series.iloc[i]:>+10.4f}")

    n_nan = df["vrp_regime"].isna().sum()
    print(f"\nObs sem rotulo (NaN): {n_nan} "
          f"({df['date'].iloc[0].date()} a "
          f"{df.loc[df['vrp_regime'].isna(), 'date'].max().date()})")
    print(f"Obs com rotulo valido: {len(df) - n_nan}")

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
