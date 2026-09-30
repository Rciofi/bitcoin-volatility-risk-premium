import os
import numpy as np
import pandas as pd


def check_daily_continuity(dates, name):
    """
    Interrompe o pipeline se faltar alguma data (ou houver data duplicada) na serie diaria.
    A RV usa rolling(30) sobre LINHAS: um buraco no calendario vira silenciosamente
    um retorno de varios dias e janelas que misturam periodos (ex.: mar/2023, T0).
    """
    d = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    dups = d[d.duplicated()].dt.date.unique().tolist()
    missing = pd.date_range(d.min(), d.max(), freq="D").difference(d)
    if dups or len(missing):
        blocos = []
        if len(missing):
            s = pd.Series(missing)
            for _, b in s.groupby((s.diff().dt.days != 1).cumsum()):
                blocos.append(f"{b.min().date()} a {b.max().date()} ({len(b)} dias)")
        raise ValueError(
            f"{name}: serie diaria descontinua -- faltando: {blocos or 'nenhuma'}; "
            f"duplicadas: {dups or 'nenhuma'}. Corrija os dados brutos antes de rodar o pipeline."
        )
    print(f"  {name}: continuidade diaria OK ({len(d)} dias, {d.min().date()} a {d.max().date()})")


def load_btc_prices(base_dir: str) -> pd.DataFrame:
    """
    Lê btc_prices.csv (Binance), ajusta datas, calcula retornos, RV30D
    retrospectiva (proxy) e RV30D prospectiva (T1).
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
    check_daily_continuity(df["date"], "btc_prices.csv")

    # Retorno logarítmico diário
    df["close"] = df["close"].astype(float)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))

    # ===================================================
    # Volatilidade realizada 30D (forma clássica - RV)
    # ===================================================
    window = 30

    # RV retrospectiva: retornos de t-29 a t. Informação disponível em t;
    # a partir do T1 é a PROXY do prêmio esperado (passeio aleatório sem deriva).
    df["rv_30d"] = (
        (df["ret"] ** 2)
        .rolling(window=window)
        .mean()
        * 365
    ) ** 0.5 * 100

    # RV prospectiva (T1): mesma fórmula, retornos de t+1 a t+30 -- só é
    # conhecida em t+30. Calculada na série longa de preços, antes do merge,
    # com janela para frente sobre ret deslocado em 1 dia; as últimas 30
    # linhas ficam NaN. Por construção rv_30d_fut(t) = rv_30d(t+30), conferido
    # em code/test_T1_alinhamento.py.
    fwd = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    df["rv_30d_fut"] = (
        (df["ret"].shift(-1) ** 2)
        .rolling(window=fwd, min_periods=window)
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
    check_daily_continuity(df_iv["date"], "dvol_30d_full.csv")

    return df_iv


def build_vrp_dataset():
    """
    Constrói o dataset consolidado de VRP 30D:

    - Lê preços do BTC (Binance) e DVOL 30D (Deribit)
    - Calcula RV30D (realized vol 30 dias, anualizada), retrospectiva e prospectiva
    - Faz merge por data
    - Calcula VRP30D = RV30D - IV30D (proxy retrospectiva)
    - Calcula BVRP30D_FUT = RV30D_FUT - IV30D (definição prospectiva, T1)
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
    check_daily_continuity(df["date"], "merge preço x DVOL")

    # 4) BVRP 30D.
    # vrp_30d: RV(t-29 a t) - IV_t -- PROXY retrospectiva, informação disponível
    #   em t. Mantida com o nome e o significado antigos de propósito: os scripts
    #   que a leem como regressor/sinal em t (docs/pendencias_T1.md, seção 1)
    #   continuam sem vazar dados do futuro até migrarem no T5/T10.
    # bvrp_30d_fut: RV(t+1 a t+30) - IV_t -- definição prospectiva do BVRP
    #   (reunião de 29/09/2026, T1). NÃO usar como informação disponível em t.
    df["vrp_30d"] = df["rv_30d"] - df["iv_30d"]
    df["bvrp_30d_fut"] = df["rv_30d_fut"] - df["iv_30d"]

    # Ordena e limpa colunas
    df = df.sort_values("date").reset_index(drop=True)

    # Reorganiza colunas para ficar mais legível
    cols_order = [
        "date",
        "close",      # preço BTC
        "ret",        # retorno diário
        "rv_30d",       # realized vol 30D retrospectiva, t-29 a t (% a.a.) -- proxy
        "iv_30d",       # implied vol 30D (DVOL, % a.a.)
        "vrp_30d",      # BVRP proxy = RV retrospectiva - IV (disponível em t)
        "rv_30d_fut",   # realized vol 30D prospectiva, t+1 a t+30 (% a.a.) -- T1
        "bvrp_30d_fut", # BVRP prospectivo = RV prospectiva - IV (conhecido só em t+30)
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
    stats.index = ["RV30D", "IV30D", "BVRP (RV30D - IV30D)"]

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