# Pendências registradas no T0/T1

Levantamento feito no mapeamento do T1 (30/09/2026). **Nenhum destes scripts foi alterado.**
As tarefas indicadas seguem o plano de revisão atual, `docs/Plano_Revisao_Dissertacao_BVRP.pdf`
(reunião com o Prof. Marcelo de 29/09/2026), que substitui a ordem de tarefas da Fase 0.

Regra (reunião de 29/09/2026): a partir do T1, o BVRP-alvo é prospectivo,
`RV(t+1 a t+30) − IV_t`. Nada que seja usado como informação disponível em t
(regressor, sinal de estratégia, classificação de regime, modelo de referência)
pode usar o BVRP prospectivo. Nesses casos, usa-se `bvrp_proxy` (retrospectivo)
ou o BVRP previsto (T10/T11).

Siglas das tarefas (plano): T2 teste H1 · T3 Newey–West com h+1 · T4 feature engineering ·
T5 BVRP defasado e persistência · T6 erro da previsão · T7 hiperparâmetros das árvores ·
T8 janela e divisão treino/teste · T9 regimes sem look-ahead · T10 retorno ~ BVRP previsto (linear) ·
T11 modelo único em árvore · T12 horizonte de 20 → 30 dias · A1 apêndice de estratégias ·
C2.x–C7.x reescrita por capítulo.

## 1. Vazamento de informação futura com o BVRP prospectivo (c.2)

Depois do T1, as tabelas e figuras abaixo ficam **inválidas** até serem refeitas com
`bvrp_proxy` ou com o BVRP previsto. Isso é esperado.

| # | Arquivo : linha | Uso do BVRP | Saída (no LaTeX?) | Gravidade | Tarefa |
|---|---|---|---|---|---|
| 1 | `code/regenerate_cap5.py:64,159,374` | regressor BVRP_t para `ret_fut_h` (modelo básico, bootstrap, Fig. 1) | Tabs. `tab_ols_basico_multihoriz`, `tab_bootstrap_ci` (sim) | crítica: a RV prospectiva é feita dos mesmos retornos que formam `ret_fut_h` | **T10** (com T3 e T12) |
| 2 | `code/analyze_magnitude_bvrp.py:56,93` | regressor em \|ret_fut_30d\| | console | crítica: correlação mecânica (mesma janela) | **T10** |
| 3 | `scripts/regenerate_cap9.py:270-273, 414` | regressor BVRP + dummy + interação (Tab. 9.5); reta da Fig. 9.5 | `tab_cap9_05_regressoes`, `fig_cap9_05` (sim) | crítica | **T9 / T11** (C7.1: o modelo único em árvore substitui os regimes construídos à mão) |
| 4 | `code/plot_vrp_vs_return.py:71-87` | dispersão BVRP × `ret_fut_20d` | `figs/cap3/vrp_vs_return_20d.png` (sim) | alta | **T10 / T12**; ver também C3.4 |
| 5 | `scripts/regenerate_cap7.py:139, 252, 312`; `scripts/cap7/strategy_vrp_quantile.py:25` | sinal de estratégia com `vrp_30d.shift(1)`: o BVRP de t−1 contém retornos até t+29 | Tabs. 7.2–7.4 e Figs. 7.2–7.5 (sim) | crítica | **A1** (apêndice, D2) |
| 6 | `code/analyze_vrp_regimes.py:52-58` | regime pelos tercis do BVRP → `vrp_regime` | `data/vrp_with_regimes.csv` | alta | **T9** (D4: dois regimes, corte sem look-ahead) |
| 7 | `code/build_ml_dataset.py:44, 51` | features `d_vrp_1d = vrp_30d.diff()` e `vrp_regime_num` | `data/ml_dataset.csv` | alta | **T4 / T5** (T4: justificar ou incluir a variação do prêmio) |
| 8 | `scripts/regenerate_cap8.py:46, 78-79` | alvo `vrp_30d.shift(-1)`; features `d_vrp_1d`, `vrp_regime_num` | Tabs. 8.1–8.2, Figs. SHAP (sim; novo Cap. 5) | alta; o alvo "t+1" precisa ser redefinido | **T5** (com T6, T7, T8 e C5.1) |
| 9 | `code/train_linear_cap6.py:44, 49, 54` | features `vrp_30d`, `d_vrp_1d`, `vrp_regime_num` → `ret_fut_1d` | `chapters/cap6_ml/tables/*.csv` (não) | alta | **candidato a descarte — decidir no T7** (saída não usada no LaTeX) |
| 10 | `code/train_ml_baseline.py:37`, `code/train_rf_gridsearch.py:25`, `code/train_xgboost.py:32` | feature `vrp_30d` → `ret_fut_5d` | `data/*feature_importance.csv` (não) | legado | **candidatos a descarte — decidir no T7** |

## 2. Modelo de referência de persistência e `bvrp_fut_1d` (c.3)

A Decisão 3 da Fase 0 (persistência, `BVRP_{t+1} = BVRP_t`) está **suspensa até o T5**:
com a definição prospectiva, BVRP_t só é conhecido em t+30. O T5 exige que o R² dentro
da amostra não fique abaixo do benchmark de persistência (0,94).

| Arquivo : linha | O que é | Tarefa |
|---|---|---|
| `scripts/regenerate_cap8.py:78` | `y_level = BVRP_t` (base da persistência) | T5 |
| `scripts/regenerate_cap8.py:238-242` | previsão de persistência nas métricas fora da amostra | T5 |
| `scripts/regenerate_cap8.py:431` | nota da Tab. 8.1 descrevendo a persistência | T5 / C5.1 |
| `code/build_bvrp_fut_target.py:33` | cria `bvrp_fut_1d = bvrp_30d.shift(-1)` | T5 / C5.1 (nova notação, sem o deslocamento t+1) |
| `scripts/regenerate_cap6_scatter.py:30` | alvo `bvrp_fut_1d` (Figs. 6.11/6.12, fora do LaTeX) | T5 |
| `chapters/Cap6_ML.ipynb` (seção 2.4.2; L1362) | persistência e `bvrp_fut_1d` no notebook legado | T5 (legado) |

## 3. Outros conflitos com a Fase 0

Onde o plano atual e a Fase 0 divergem, vale o plano (reunião de 29/09/2026).

| # | Arquivo : linha | Conflito | Fase 0 | Plano atual | Tarefa |
|---|---|---|---|---|---|
| 1 | `scripts/regenerate_cap9.py:62-63` | q25/q75 da RV calculados na amostra inteira (três regimes) | E4: janela expansiva | D4: **dois** regimes; T9: corte calculado só com a primeira janela de estimação (~20 no histograma) | **T9** |
| 2 | `scripts/regenerate_cap7.py:306` (e `regenerate_cap7_figs.py:72`) | `pd.qcut` na amostra inteira para o regime de RV da estratégia: look-ahead mesmo na definição antiga | E4 | A1: sinal por quantis em janela expansiva, sem look-ahead | **A1** |
| 3 | `scripts/regenerate_cap6_scatter.py:31` | `close` (I(1)) como feature | E3: eliminar `close` | T4: **transformar** variáveis não estacionárias (ex.: diferenças) em vez de eliminá-las | **T4** |
| 4 | `scripts/regenerate_cap7.py:180` | custos de 0/5/10 bps | E6: 10/30 bps | A1: "manter os custos de transação" | **A1** (confirmar se os cenários mudam) |
| 5 | `scripts/regenerate_cap7.py:100`, `scripts/regenerate_cap7_figs.py:64`, `code/plot_vrp_timeseries.py:23` | data final da amostra fixa em `"2026-03-03"` (continua correta depois do T0, mas é frágil) | Decisão 2 | — | sem tarefa própria; tratar quando o script for tocado (A1, T1) |
| 6 | `scripts/test_h1_bvrp_mean.py:55` | HAC com 30 defasagens fixas | IC5: lag = h | T3: **h+1** defasagens | **T2 / T3** (C3.3 e C6.8: o H1 vira parágrafo, sem tabela) |
| 7 | `code/regenerate_cap5.py:53`, `scripts/regenerate_cap9.py:100,274` | HAC com `maxlags=h` | IC5: lag = h | T3: **h+1** | **T3** |
| 8 | `code/bvrp_ml_csv.py/plot_vrp_histograms.py:42,73` | rótulo "IV30D – RV30D" (sinal antigo); script legado | E0 | — | fora do plano: candidato a descarte |

## 4. Texto (não é código)

| Item | Tarefa |
|---|---|
| A nota da Tab. 3.1 descreve a RV como "desvio padrão anualizado". O código (correto, padrão da literatura) usa a raiz da média dos retornos quadráticos, sem subtrair a média. **Manter o código; corrigir o texto.** O plano também pede chamar a medida de **volatilidade histórica** (janela móvel de h dias) e explicar a diferença para a volatilidade realizada. | **C2.4**, F7 |
| O texto usa N = 1.775 em vários lugares; depois do T0 o N de referência é **1.806** (24/03/2021 a 03/03/2026). | Fase 2 (C3 e seguintes) |
| **Cap. 3 — artefatos do buraco de mar/2023 (T0).** Eram artefatos: (i) o máximo do retorno diário da Tab. 3.1 (0,2066 em 01/04/2023, na verdade um retorno de 32 dias; o novo máximo é 0,1353, em 28/02/2022); (ii) o máximo do BVRP (32,49 em 14/04/2023; o novo é 31,66, em 14/02/2026); (iii) o prêmio positivo de abril/2023 (BVRP médio +28,2 → −4,6; RV média 84,1 → 51,3). Qualquer menção a esses episódios no texto deve ser revista, assim como descrições da cauda direita do BVRP (p95 15,96 → 11,71; assimetria 0,609 → 0,383). | **C3** (reescrita do Cap. 3; ver também C3.1, distribuição bimodal) |
| A Tab. 3.2 publicada não é reproduzível bit a bit (só o ADF do BVRP bate; as conclusões batem). O novo `code/build_desc_stats_T1.py` passa a ser o gerador. | T1 |
| C3.2 pede nota de que 2026 é ano incompleto "(dados até julho/agosto)"; a amostra de referência termina em **03/03/2026** (os dados brutos vão até 02/05/2026). Conferir a data na legenda. | **C3.2** |
