import os
import pandas as pd
import numpy as np


def build_targets():
    """
    Lê vrp_30d_dataset.csv, cria retornos futuros 1D, 5D, 20D e
    salva o dataset expandido + estatísticas dos retornos.
    """
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

    df["ret_fut_1d"]  = df["close"].shift(-1)  / df["close"] - 1
    df["ret_fut_5d"]  = df["close"].shift(-5)  / df["close"] - 1
    df["ret_fut_20d"] = df["close"].shift(-20) / df["close"] - 1
    df["ret_fut_10d"] = df["close"].shift(-10) / df["close"] - 1
    df["ret_fut_30d"] = df["close"].shift(-30) / df["close"] - 1
    df["ret_fut_60d"] = df["close"].shift(-60) / df["close"] - 1  # Tarefa 3 — h=60 (Decisão 2)

    # Remove linhas sem target futuro (últimas 60 linhas, critério h=60)
    df_clean = df.dropna().reset_index(drop=True)

    # Salvar dataset expandido
    out_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")
    df_clean.to_csv(out_path, index=False)

    print("\n=== RETORNOS FUTUROS GERADOS COM SUCESSO ===")
    print(f"Arquivo salvo em: {out_path}")
    print(f"Linhas finais: {len(df_clean)}")
    print("Colunas criadas: ret_fut_1d, ret_fut_5d, ret_fut_10d, ret_fut_20d, ret_fut_30d, ret_fut_60d")

    # Estatísticas descritivas dos retornos futuros
    print("\nResumo estatístico dos retornos futuros:")
    stats_ret = (
        df_clean[["ret_fut_1d", "ret_fut_5d", "ret_fut_20d"]]
        .describe()
        .T[["mean", "std", "min", "max"]]
    )

    stats_ret.index = [
        "Retorno futuro 1D",
        "Retorno futuro 5D",
        "Retorno futuro 20D",
    ]

    stats_ret = stats_ret.rename(
        columns={
            "mean": "Média",
            "std": "Desvio-padrão",
            "min": "Mínimo",
            "max": "Máximo",
        }
    ).round(6)

    print(stats_ret)

    # Salva CSV e LaTeX (se quiser usar separado)
    out_stats_csv = os.path.join(base_dir, "data", "estatisticas_retornos_futuros.csv")
    stats_ret.to_csv(out_stats_csv, sep=";", encoding="utf-8")
    print(f"\nEstatísticas dos retornos salvas em: {out_stats_csv}")

    out_stats_tex = os.path.join(
        base_dir, "data", "tabela_estatisticas_retornos_futuros.tex"
    )
    with open(out_stats_tex, "w", encoding="utf-8") as f:
        f.write(
            stats_ret.to_latex(
                column_format="lcccc",
                escape=False,
                float_format="%.6f".__mod__,
                caption="Estatísticas descritivas dos retornos futuros do Bitcoin",
                label="tab:estatisticas_retornos_futuros",
            )
        )
    print(f"Tabela LaTeX (retornos) salva em: {out_stats_tex}")


if __name__ == "__main__":
    build_targets()
