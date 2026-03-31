import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Cria coluna com o ano
    df["year"] = df["date"].dt.year

    # ============================
    #         PLOT ANUAL
    # ============================

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="year",
        y="vrp_30d",
        color="lightgray",
        linewidth=1,
        fliersize=3,
        ax=ax
    )

    ax.set_title("Boxplot do VRP 30D por Ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("VRP 30D (IV30D – RV30D)")
    ax.grid(True, axis="y", alpha=0.5)

    # Diretório para salvar figuras
    figs_dir = os.path.join(base_dir, "figuras")
    os.makedirs(figs_dir, exist_ok=True)

    out_path = os.path.join(figs_dir, "vrp_boxplot_ano.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print("\n=== FIGURA GERADA COM SUCESSO ===")
    print(f"Figura salva em: {out_path}")


if __name__ == "__main__":
    main()
