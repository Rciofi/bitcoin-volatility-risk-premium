import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def scatter_plot(df, x, y, title, out_path):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        alpha=0.5,
        s=25,
        color="steelblue",
        ax=ax
    )

    # Regressão linear
    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    line_x = df[x]
    line_y = intercept + slope * line_x

    ax.plot(
        line_x,
        line_y,
        color="darkred",
        linewidth=1.5,
        label=f"Regressão (corr = {r_value:.2f})"
    )

    ax.set_title(title)
    ax.set_xlabel("VRP 30D (RV30D – IV30D)")
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.6)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {out_path}")


def main():
    # Diretório base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vrp_with_targets.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    print(f"Lendo VRP dataset com retornos futuros em: {data_path}")
    df = pd.read_csv(data_path)

    # Converte date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Diretório de figuras
    figs_dir = os.path.join(base_dir, "figs", "dados")
    os.makedirs(figs_dir, exist_ok=True)

    # ====== VRP(t) vs Retorno Futuro ======

    scatter_plot(
        df,
        x="vrp_30d",
        y="ret_fut_1d",
        title="VRP 30D vs Retorno Futuro 1D",
        out_path=os.path.join(figs_dir, "vrp_vs_return_1d.png")
    )

    scatter_plot(
        df,
        x="vrp_30d",
        y="ret_fut_5d",
        title="VRP 30D vs Retorno Futuro 5D",
        out_path=os.path.join(figs_dir, "vrp_vs_return_5d.png")
    )

    scatter_plot(
        df,
        x="vrp_30d",
        y="ret_fut_20d",
        title="VRP 30D vs Retorno Futuro 20D",
        out_path=os.path.join(figs_dir, "vrp_vs_return_20d.png")
    )

    print("\n=== GRÁFICOS VRP VS RETORNO FUTURO GERADOS COM SUCESSO ===")


if __name__ == "__main__":
    main()
