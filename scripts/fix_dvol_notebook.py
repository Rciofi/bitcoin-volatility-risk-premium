import json

NB_PATH = "c:/Users/ciofi/OneDrive/Documentos/FGV/Dissertação/bitcoin-volatility-risk-premium/code/dvol.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 6 — série temporal com eixo duplo + save
nb["cells"][6]["source"] = [
    "# celula 3 - grafico de series temporais (eixo duplo)\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "import os\n",
    "\n",
    "fig, ax1 = plt.subplots(figsize=(12, 5))\n",
    "\n",
    "ax1.plot(df_final['date'], df_final['rv_30d'], color='steelblue', linewidth=1.2,\n",
    "         label='Vol. Realizada 30D (RV30D)')\n",
    "ax1.plot(df_final['date'], df_final['iv_30d'], color='darkorange', linewidth=1.2,\n",
    "         label='Vol. Implicita 30D (IV30D - DVOL)')\n",
    "ax1.set_xlabel('Data')\n",
    "ax1.set_ylabel('Volatilidade (% a.a.)')\n",
    "ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))\n",
    "ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))\n",
    "plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')\n",
    "\n",
    "ax2 = ax1.twinx()\n",
    "ax2.plot(df_final['date'], df_final['vrp_30d'], color='steelblue', linewidth=1.0,\n",
    "         linestyle='--', alpha=0.7, label='VRP 30D (IV30D - RV30D)')\n",
    "ax2.set_ylabel('VRP 30D (p.p. de vol.)')\n",
    "ax2.axhline(0, color='gray', linewidth=0.7, linestyle=':')\n",
    "\n",
    "lines1, labels1 = ax1.get_legend_handles_labels()\n",
    "lines2, labels2 = ax2.get_legend_handles_labels()\n",
    "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)\n",
    "ax1.grid(True, alpha=0.3)\n",
    "\n",
    "os.makedirs('../figs/cap3', exist_ok=True)\n",
    "fig.tight_layout()\n",
    "fig.savefig('../figs/cap3/vrp_timeseries.png', dpi=300, bbox_inches='tight')\n",
    "print('Salvo em: ../figs/cap3/vrp_timeseries.png')\n",
    "plt.show()\n",
]
nb["cells"][6]["outputs"] = []
nb["cells"][6]["execution_count"] = None

# Cell 7 — histograma + KDE + save
nb["cells"][7]["source"] = [
    "# celula 4 - histograma e KDE do VRP\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "vrp = df_final['vrp_30d'].dropna()\n",
    "sns.histplot(vrp, bins=40, kde=True, stat='density', alpha=0.6,\n",
    "             color='steelblue', ax=ax)\n",
    "media = vrp.mean()\n",
    "ax.axvline(media, color='red', linestyle='--', label=f'Media: {media:.1f} p.p.')\n",
    "ax.set_xlabel('VRP 30D (p.p. de volatilidade)')\n",
    "ax.set_ylabel('Densidade')\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3)\n",
    "\n",
    "os.makedirs('../figs/cap3', exist_ok=True)\n",
    "fig.tight_layout()\n",
    "fig.savefig('../figs/cap3/vrp_histogram_kde.png', dpi=300, bbox_inches='tight')\n",
    "print('Salvo em: ../figs/cap3/vrp_histogram_kde.png')\n",
    "plt.show()\n",
]
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None

# Cell 8 — boxplot + save
nb["cells"][8]["source"] = [
    "# celula 5 - boxplot do VRP\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "sns.boxplot(data=df_final, x='vrp_30d', color='lightgray',\n",
    "            fliersize=3, linewidth=1, ax=ax)\n",
    "ax.set_xlabel('VRP 30D (p.p. de vol.)')\n",
    "ax.grid(True, axis='x', alpha=0.4)\n",
    "\n",
    "os.makedirs('../figs/cap3', exist_ok=True)\n",
    "fig.tight_layout()\n",
    "fig.savefig('../figs/cap3/vrp_boxplot.png', dpi=300, bbox_inches='tight')\n",
    "print('Salvo em: ../figs/cap3/vrp_boxplot.png')\n",
    "plt.show()\n",
]
nb["cells"][8]["outputs"] = []
nb["cells"][8]["execution_count"] = None

# Cell 9 — era save redundante, limpa
nb["cells"][9]["source"] = ["# (removido - save redundante)\n"]
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None

# Cell 30 — limpa (era plots com escala errada, salvava em code/figs/)
nb["cells"][30]["source"] = ["# (removido - plots com escala incorreta do pipeline antigo)\n"]
nb["cells"][30]["outputs"] = []
nb["cells"][30]["execution_count"] = None

# Cell 32 — limpa (era plots com escala errada, salvava em code/figs/)
nb["cells"][32]["source"] = ["# (removido - plots com escala incorreta do pipeline antigo)\n"]
nb["cells"][32]["outputs"] = []
nb["cells"][32]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook atualizado.")
