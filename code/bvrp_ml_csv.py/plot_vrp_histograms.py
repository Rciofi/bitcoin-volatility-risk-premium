import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Diretório base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_30d_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo VRP dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Converte data
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ============================
    #       FIGURA 2A
    # HISTOGRAMA + CURVA KDE
    # ============================

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histograma
    sns.histplot(
        df["vrp_30d"],
        bins=30,
        kde=True,
        stat="density",
        linewidth=0.5,
        color="steelblue",
        ax=ax,
    )

    ax.set_title("Distribuição do VRP 30D (Histograma + KDE)")
    ax.set_xlabel("VRP 30D (IV30D – RV30D)")
    ax.set_ylabel("Densidade")
    ax.grid(True, alpha=0.6)

    figs_dir = os.path.join(base_dir, "figuras")
    os.makedirs(figs_dir, exist_ok=True)

    out_path1 = os.path.join(figs_dir, "vrp_histogram_kde.png")
    plt.tight_layout()
    plt.savefig(out_path1, dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {out_path1}")

    # ============================
    #       FIGURA 2B
    #          BOXPLOT
    # ============================

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x="vrp_30d",
        color="lightgray",
        fliersize=3,
        linewidth=1,
        ax=ax,
    )

    ax.set_title("Boxplot do VRP 30D")
    ax.set_xlabel("VRP 30D (IV30D – RV30D)")
    ax.grid(True, axis="x", alpha=0.4)

    out_path2 = os.path.join(figs_dir, "vrp_boxplot.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {out_path2}")

    print("\n=== GRÁFICOS DO VRP GERADOS COM SUCESSO ===")


if __name__ == "__main__":
    main()
