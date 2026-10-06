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

    figs_dir = os.path.join(base_dir, "figs", "dados")
    os.makedirs(figs_dir, exist_ok=True)

    # ================================================
    #  FIGURA 3.1 — SÉRIE TEMPORAL (2 painéis com eventos)
    # ================================================
    plt.style.use("default")
    out1 = os.path.join(figs_dir, "vrp_timeseries.png")

    # === FIGURA 3.1 — SERIE TEMPORAL: 2 PAINEIS COM EVENTOS ===
    eventos = [
        ("2022-05-09", "LUNA"),
        ("2022-06-13", "3AC/Celsius"),
        ("2022-11-08", "FTX"),
        ("2023-03-10", "Crise banc. EUA"),
        ("2024-01-10", "ETF spot SEC"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Painel (a): RV e IV sobrepostas ---
    axes[0].plot(df["date"], df["rv_30d"], color="#1f4e79", linewidth=0.8,
                 label="Volatilidade Realizada (RV30D)")
    axes[0].plot(df["date"], df["iv_30d"], color="#c0392b", linewidth=0.8,
                 label="Volatilidade Implícita (IV30D — DVOL Deribit)")
    axes[0].set_ylabel("Volatilidade (% a.a.)")
    axes[0].set_title("(a) Volatilidade realizada e implícita", loc="left", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Painel (b): BVRP ---
    axes[1].plot(df["date"], df["vrp_30d"], color="#2e7d32", linewidth=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("BVRP 30D (p.p. de vol.)")
    axes[1].set_title("(b) Prêmio de Risco de Volatilidade (BVRP = RV30D − IV30D)",
                      loc="left", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Data")

    # --- Anotacoes verticais de eventos (nos dois paineis) ---
    # Altura do rotulo alternada (linha superior/inferior) para eventos
    # proximos no tempo nao se sobreporem (ex.: LUNA e 3AC/Celsius, 5
    # semanas de distancia).
    for i, (data_ev, rotulo) in enumerate(eventos):
        x = pd.to_datetime(data_ev)
        for ax in axes:
            ax.axvline(x, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)
        # rotulo so no painel de cima, perto do topo
        ymin, ymax = axes[0].get_ylim()
        y_offset = -4 if i % 2 == 0 else -20
        axes[0].annotate(rotulo, xy=(x, ymax), xytext=(0, y_offset),
                         textcoords="offset points", rotation=0, ha="center",
                         va="top", fontsize=7,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                   ec="gray", lw=0.4, alpha=0.85))

    fig.tight_layout()
    fig.savefig(out1, dpi=150, bbox_inches="tight")
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
