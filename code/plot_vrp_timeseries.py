import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_30d_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo VRP dataset: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Trunca para a amostra canonica (mar/2021-mar/2026, N=1.775), igual a
    # vrp_with_targets.csv e as Tabelas 3.1/3.2. vrp_30d_dataset.csv tem 60
    # dias a mais (ate mai/2026) que ainda nao entraram na fonte canonica.
    AMOSTRA_FIM = "2026-03-03"
    df = df[df["date"] <= AMOSTRA_FIM].reset_index(drop=True)
    print(f"Truncado para amostra canonica: N={len(df)}  {df['date'].min().date()} a {df['date'].max().date()}")

    figs_dir = os.path.join(base_dir, "figs", "cap3")
    os.makedirs(figs_dir, exist_ok=True)

    # ================================================
    #  FIGURA 3.1 — SÉRIE TEMPORAL (eixo duplo)
    # ================================================
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["date"], df["rv_30d"],  color="steelblue",  linewidth=1.2,
             label="Vol. Realizada 30D (RV30D)")
    ax1.plot(df["date"], df["iv_30d"],  color="darkorange", linewidth=1.2,
             label="Vol. Implícita 30D (IV30D – DVOL)")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Volatilidade (% a.a.)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["vrp_30d"], color="steelblue", linewidth=1.0,
             linestyle="--", alpha=0.7, label="VRP 30D (RV30D – IV30D)")
    ax2.set_ylabel("VRP 30D (p.p. de vol.)")
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8, framealpha=0.8)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out1 = os.path.join(figs_dir, "vrp_timeseries.png")
    fig.savefig(out1, dpi=300)
    plt.close(fig)
    print(f"Figura salva em: {out1}")

    # ================================================
    #  FIGURA — HISTOGRAMA + KDE
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["vrp_30d"], bins=30, kde=True, stat="density",
                 linewidth=0.5, color="steelblue", ax=ax)
    ax.set_xlabel("VRP 30D (RV30D – IV30D)")
    ax.set_ylabel("Densidade")
    ax.grid(True, alpha=0.6)
    fig.tight_layout()

    out2 = os.path.join(figs_dir, "vrp_histogram_kde.png")
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"Figura salva em: {out2}")

    # ================================================
    #  FIGURA — BOXPLOT POR ANO
    # ================================================
    df["year"] = df["date"].dt.year.astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="year", y="vrp_30d", hue="year", legend=False,
                palette="dark:steelblue", fliersize=2, linewidth=0.8, ax=ax)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Ano")
    ax.set_ylabel("VRP 30D (p.p. de vol.)")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()

    out3 = os.path.join(figs_dir, "vrp_boxplot.png")
    fig.savefig(out3, dpi=300)
    plt.close(fig)
    print(f"Figura salva em: {out3}")

    print("\n=== GRAFICOS DO VRP GERADOS COM SUCESSO ===")


if __name__ == "__main__":
    main()
