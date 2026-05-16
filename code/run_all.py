import os
import pandas as pd

import build_vrp_dataset
import build_targets


def main():
    # 1) Gera VRP dataset (IV30D, RV30D, VRP30D)
    build_vrp_dataset.build_vrp_dataset()

    # 2) Gera retornos futuros + vrp_with_targets.csv
    build_targets.build_targets()

    # 3) Lê dataset completo e monta Tabela 4.1 com as 6 variáveis
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Arquivo {full_path} não encontrado. Rode os passos anteriores."
        )

    print(f"\nLendo dataset completo em: {full_path}")
    df = pd.read_csv(full_path)

    # Monta estatísticas para as 6 variáveis
    cols = ["iv_30d", "rv_30d", "vrp_30d",
            "ret_fut_1d", "ret_fut_5d", "ret_fut_20d"]

    stats_all = (
        df[cols]
        .describe()
        .T[["mean", "std", "min", "max"]]
    )

    # Index legível para a Tabela 4.1
    stats_all.index = [
        "IV30D",
        "RV30D",
        "BVRP (RV30D - IV30D)",
        "Retorno futuro 1D",
        "Retorno futuro 5D",
        "Retorno futuro 20D",
    ]

    stats_all = stats_all.rename(
        columns={
            "mean": "Média",
            "std": "Desvio-padrão",
            "min": "Mínimo",
            "max": "Máximo",
        }
    ).round(6)

    print("\n=== Tabela 4.1 - Estatísticas descritivas (completa) ===\n")
    print(stats_all)

    # Salva CSV
    out_csv = os.path.join(base_dir, "data", "tabela_cap4_estatisticas_completas.csv")
    stats_all.to_csv(out_csv, sep=";", encoding="utf-8")
    print(f"\nTabela 4.1 em CSV salva em: {out_csv}")

    # Salva LaTeX
    out_tex = os.path.join(base_dir, "data", "tabela_cap4_estatisticas_completas.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(
            stats_all.to_latex(
                column_format="lcccc",
                escape=False,
                float_format="%.6f".__mod__,
                caption="Estatísticas descritivas das variáveis principais e retornos futuros",
                label="tab:estatisticas_cap4",
            )
        )
    print(f"Tabela 4.1 LaTeX salva em: {out_tex}")


if __name__ == "__main__":
    main()
