import os
import numpy as np
import pandas as pd


def load_btc_prices(base_dir: str) -> pd.DataFrame:
    """
    Lê btc_prices.csv (Binance), ajusta datas, calcula retornos e RV30D.
    Espera arquivo em: data/btc_prices.csv, com colunas: date, close.
    """
    price_path = os.path.join(base_dir, "data", "btc_prices.csv")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {price_path}")

    print(f"Lendo preços do BTC de: {price_path}")
    df = pd.read_csv(price_path)

    # Garante nome das colunas
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("btc_prices.csv precisa ter colunas 'date' e 'close'.")

    # Converte datas e ordena
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # Retorno logarítmico diário
    df["close"] = df["close"].astype(float)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))

    # ===================================================
    # Volatilidade realizada 30D (forma clássica - RV)
    # ===================================================
    window = 30

    df["rv_30d"] = (
        (df["ret"] ** 2)
        .rolling(window=window)
        .mean()
        * 365
    ) ** 0.5 * 100

    return df


def load_dvol(base_dir: str) -> pd.DataFrame:
    """
    Lê dvol_30d_full.csv (Deribit) e prepara a série de IV30D em %.
    Espera arquivo em: data/dvol_30d_full.csv.
    Usa a coluna 'close' como IV 30D (% a.a.).
    """
    iv_path = os.path.join(base_dir, "data", "dvol_30d_full.csv")
    if not os.path.exists(iv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {iv_path}")

    print(f"Lendo DVOL 30D da Deribit de: {iv_path}")
    df_iv = pd.read_csv(iv_path)

    # Normaliza nomes
    df_iv.columns = [c.lower() for c in df_iv.columns]

    # Verifica colunas esperadas
    if "timestamp" not in df_iv.columns:
        raise ValueError("dvol_30d_full.csv precisa ter coluna 'timestamp'.")
    if "close" not in df_iv.columns:
        raise ValueError("dvol_30d_full.csv precisa ter coluna 'close' (DVOL em %).")

    # Converte timestamp em data (sem horário)
    df_iv["timestamp"] = pd.to_datetime(df_iv["timestamp"])
    df_iv["date"] = df_iv["timestamp"].dt.date

    # IV 30D em % (já está em % no 'close')
    df_iv["iv_30d"] = df_iv["close"].astype(float)

    df_iv = df_iv[["date", "iv_30d"]].sort_values("date").reset_index(drop=True)

    return df_iv


def build_vrp_dataset():
    """
    Constrói o dataset consolidado de VRP 30D:

    - Lê preços do BTC (Binance) e DVOL 30D (Deribit)
    - Calcula RV30D (realized vol 30 dias, anualizada)
    - Faz merge por data
    - Calcula VRP30D = IV30D - RV30D
    - Salva em data/vrp_30d_dataset.csv
    - Gera estatísticas descritivas de RV30D, IV30D e VRP30D
    """
    # base_dir = raiz do projeto (um nível acima da pasta code)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Diretório base do projeto: {base_dir}")

    # 1) Carrega preços e RV
    df_price = load_btc_prices(base_dir)

    # 2) Carrega IV 30D
    df_iv = load_dvol(base_dir)

    # 3) Faz o merge por data (inner -> período em comum)
    print("Fazendo merge entre preços (com RV30D) e IV30D (DVOL)...")
    df = pd.merge(df_price, df_iv, on="date", how="inner")

    # 4) Calcula BVRP 30D = RV30D - IV30D  (Decisão 1 — Fase 0: sinal correto per literatura)
    df["vrp_30d"] = df["rv_30d"] - df["iv_30d"]

    # Ordena e limpa colunas
    df = df.sort_values("date").reset_index(drop=True)

    # Reorganiza colunas para ficar mais legível
    cols_order = [
        "date",
        "close",      # preço BTC
        "ret",        # retorno diário
        "rv_30d",     # realized vol 30D (% a.a.)
        "iv_30d",     # implied vol 30D (DVOL, % a.a.)
        "vrp_30d",    # BVRP 30D = RV - IV  (negativo em média: mercado paga prêmio de seguro)
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]

    # 5) Salva resultado
    out_path = os.path.join(base_dir, "data", "vrp_30d_dataset.csv")
    df.to_csv(out_path, index=False)

    print("\n=== VRP DATASET GERADO COM SUCESSO ===")
    print(f"Arquivo salvo em: {out_path}")
    print(f"Número de linhas: {len(df)}")
    print("\nPeríodo coberto:")
    print(f"  De: {df['date'].min()}  Até: {df['date'].max()}")

    # 6) Estatísticas descritivas e exportação
    print("\nResumo estatístico de RV30D, IV30D e VRP30D:")

    stats = (
        df[["rv_30d", "iv_30d", "vrp_30d"]]
        .describe()
        .T[["mean", "std", "min", "max"]]
    )

    # deixa o índice bonito pra bater com a dissertação
    stats.index = ["RV30D", "IV30D", "BVRP (IV30D - RV30D)"]

    # renomeia colunas pra português e arredonda
    stats = stats.rename(
        columns={
            "mean": "Média",
            "std": "Desvio-padrão",
            "min": "Mínimo",
            "max": "Máximo",
        }
    ).round(4)

    print(stats)

    # salva em CSV (pra conferência)
    out_stats_csv = os.path.join(base_dir, "data", "estatisticas_vrp_30d.csv")
    stats.to_csv(out_stats_csv, sep=";", encoding="utf-8")
    print(f"\nEstatísticas salvas em: {out_stats_csv}")

    # salva em LaTeX (se quiser usar separado)
    out_stats_tex = os.path.join(base_dir, "data", "tabela_estatisticas_vrp_30d.tex")
    with open(out_stats_tex, "w", encoding="utf-8") as f:
        f.write(
            stats.to_latex(
                column_format="lcccc",
                escape=False,
                float_format="%.4f".__mod__,
                caption="Estatísticas descritivas das volatilidades",
                label="tab:estatisticas_vols",
            )
        )
    print(f"Tabela LaTeX (vols) salva em: {out_stats_tex}")


if __name__ == "__main__":
    build_vrp_dataset()