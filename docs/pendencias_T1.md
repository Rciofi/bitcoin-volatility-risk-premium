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
| 5 | `scripts/regenerate_cap7.py:139, 252, 312`; `scripts/cap7/strategy_vrp_quantile.py:25` | sinal de estratégia com `vrp_30d.shift(1)`. **Corrigido no A1: não há vazamento.** `vrp_30d` continua sendo a proxy retrospectiva, RV(t−29 a t) − IV_t, conhecida em t (o T1 criou `bvrp_30d_fut` em coluna separada), e `sinal(t−1) × ret(t)` equivale a "posição no fim de t rende t+1" (`test_A1.py`, teste 8). A redação anterior ("o BVRP de t−1 contém retornos até t+29") só valeria se a coluna tivesse sido redefinida. O único look-ahead do Cap. 7 era o `qcut` da Tab. 7.4 (seção 3, item 2) | Tabs. 7.2–7.4 e Figs. 7.2–7.5 (sim) | nenhuma (era a proxy) | **A1** (apêndice, D2): sinal principal passa a ser o BVRP previsto (Ridge do T5); a regra antiga com a proxy fica como robustez. Ver seção 13 |
| 6 | `code/analyze_vrp_regimes.py:52-58` | regime pelos tercis do BVRP → `vrp_regime` | `data/vrp_with_regimes.csv` | alta | **T9** (D4: dois regimes, corte sem look-ahead) |
| 7 | `code/build_ml_dataset.py:44, 51` | features `d_vrp_1d = vrp_30d.diff()` e `vrp_regime_num` | `data/ml_dataset.csv` | alta | **T5** — o T4 criou um dataset novo (`data/ml_dataset_T4.csv`, seção 6) com `d_vrp_1d` só da proxy e sem regime; `ml_dataset.csv` e este script ficam intactos para os scripts antigos |
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

### 2.1 Proxy × prêmio prospectivo: correlação ≈ 0 (achado do passo 5 do T1)

> **Respondido no T5 (decisão nossa, a validar com o Prof. Marcelo):** a "persistência viável em t" para o alvo prospectivo é `bvrp_realizado_defasado(t) = vh_30d(t) − iv_30d(t−30)`, o prêmio de t−30, que se realiza em t. Ele entra como variável explicativa em todos os modelos e como terceiro modelo de referência. Ver seção 9.

Na amostra de referência (N = 1.806), a correlação entre `bvrp_30d_fut`
(RV(t+1 a t+30) − IV_t) e `vrp_30d` (RV(t−29 a t) − IV_t) **na mesma data t** é
**−0,02**. Os extremos não coincidem: no mínimo do prospectivo (13/11/2022,
−75,83) o proxy era −26,84; no máximo do proxy (14/02/2026, +31,66) o
prospectivo era −0,68.

| Consequência | Tarefa |
|---|---|
| O critério do T5 ("R² dentro da amostra ≥ 0,94, benchmark de persistência") foi definido para o **alvo antigo** (BVRP retrospectivo, autocorrelação ≈ 0,94). Com o alvo prospectivo, ele precisa ser redefinido — inclusive qual é o benchmark de persistência viável em t, já que o BVRP prospectivo de t só é conhecido em t+30. **Decisão pendente com o Prof. Marcelo.** | **T5** |
| A proxy (passeio aleatório sem deriva) aproxima a **expectativa** do prêmio, E(RV_{t+1:t+30} \| I_t) − IV_t, e não sua **realização**. A correlação ≈ 0 com o prêmio realizado não invalida a proxy, mas o texto não pode apresentá-la como aproximação do BVRP realizado. | **C2.5 / C4.5** |

## 3. Outros conflitos com a Fase 0

Onde o plano atual e a Fase 0 divergem, vale o plano (reunião de 29/09/2026).

| # | Arquivo : linha | Conflito | Fase 0 | Plano atual | Tarefa |
|---|---|---|---|---|---|
| 1 | `scripts/regenerate_cap9.py:62-63` | q25/q75 da RV calculados na amostra inteira (três regimes) | E4: janela expansiva | D4: **dois** regimes; T9: corte calculado só com a primeira janela de estimação (~20 no histograma) | **T9** |
| 2 | `scripts/regenerate_cap7.py:306` (e `regenerate_cap7_figs.py:72`) | `pd.qcut` na amostra inteira para o regime de RV da estratégia: look-ahead mesmo na definição antiga | E4 | A1: sinal por quantis em janela expansiva, sem look-ahead | **resolvido no A1**: regime do T9 (`regime_alta_fixo`, corte de 37,3, vigente em t), só a partir de 21/06/2023 |
| 3 | `scripts/regenerate_cap6_scatter.py:31` | `close` (I(1)) como feature | E3: eliminar `close` | T4: **transformar** variáveis não estacionárias (ex.: diferenças) em vez de eliminá-las | **T4** |
| 4 | `scripts/regenerate_cap7.py:180` | custos de 0/5/10 bps | E6: 10/30 bps | A1: "manter os custos de transação" | **resolvido no A1**: 0, 10 e 30 bps unilaterais sobre \|Δposição\| |
| 5 | `scripts/regenerate_cap7.py:100`, `scripts/regenerate_cap7_figs.py:64`, `code/plot_vrp_timeseries.py:23` | data final da amostra fixa em `"2026-03-03"` (continua correta depois do T0, mas é frágil) | Decisão 2 | — | sem tarefa própria; tratar quando o script for tocado (T1). **A1:** `code/estrategias_A1.py` lê o fim da amostra de `vrp_with_targets.csv`; os scripts antigos do Cap. 7 ficam como estão |
| 6 | `scripts/test_h1_bvrp_mean.py:55` | HAC com 30 defasagens fixas | IC5: lag = h | T3: **h+1** defasagens | **resolvido no T2** (maxlags = 31; sensibilidade 7/30/60/90). O texto: C3.3 e C6.8 (o H1 vira parágrafo, sem tabela) |
| 7 | `code/regenerate_cap5.py:53`, `scripts/regenerate_cap9.py:100,274` | HAC com `maxlags=h` | IC5: lag = h | T3: **h+1** | **T10 / T11** (o T3 criou a função `mqo_newey_west`; ver seção 4.1) |
| 8 | `code/bvrp_ml_csv.py/plot_vrp_histograms.py:42,73` | rótulo "IV30D – RV30D" (sinal antigo); script legado | E0 | — | fora do plano: candidato a descarte |

## 4. Regra de defasagens do HAC no statsmodels (para o T3 e o C4.4)

Identificada no T2 (statsmodels 0.14.6, `regression/linear_model.py` e
`stats/sandwich_covariance.py`), para `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L})`:

| Item | Regra |
|---|---|
| Núcleo | Bartlett (`weights_bartlett`, padrão) |
| Pesos | $w_j = 1 - j/(L+1)$, $j = 0, \ldots, L$ — convenção de **Newey e West (1987)**; a mesma do `newey, lag(L)` do Stata |
| Significado de `maxlags = L` | maior defasagem incluída na janela do núcleo (a defasagem $L+1$ tem peso zero) |
| Texto ↔ código | "**h+1 defasagens**" no texto corresponde a `maxlags = h+1` no código (h = 30 → 31) |
| Correção de amostra pequena | **não aplicada** (`use_correction=False` é o padrão para `cov_type="HAC"`) |
| Distribuição de referência | **normal** (`use_t=False` no resultado): valor-$p$ e intervalo de confiança pela normal, não pela $t$ |
| Regra automática (se `maxlags` for omitido) | $L = \lfloor 4(T/100)^{2/9} \rfloor$ (Newey e West, 1994); para $T = 1.806$, $L = 7$ |

Observação para o C4.4: com o núcleo de Bartlett e $L = 31$, a defasagem 29 (a última com
sobreposição mecânica das janelas de 30 dias) recebe peso de apenas $1 - 29/32 \approx 0{,}09$.
O T2 reporta a sensibilidade a $L$ = 7, 30, 31, 60 e 90.

**Para o texto do C4.4 (resultado do T2, BVRP prospectivo, N = 1.806):** com a regra
automática do pacote ($L = 7$), a estatística $t$ seria **−8,10**, contra **−5,15** com
$L = h+1 = 31$ — o padrão do software subestima o erro-padrão numa série com
autocorrelação de 1ª ordem de 0,96, o que justifica usar $h+1$ em vez do padrão.
E o erro-padrão praticamente estabiliza a partir de $L = 30$: 1,70 com $L = 30$/31,
1,77 com $L = 60$ e 1,78 com $L = 90$ (a conclusão não muda em nenhuma janela).

### 4.1 Função única e inventário dos scripts com HAC (T3)

A partir do T3, toda regressão nova com erro-padrão HAC usa
`code/hac_utils.py::mqo_newey_west(y, X, h=...)`, que fixa a convenção da tabela acima
(Bartlett, `maxlags = h+1`, sem correção de amostra pequena, inferência pela normal).
Análises de sensibilidade a L usam `sensibilidade_defasagens(y, X, lags=[...])`, com a
mesma convenção; o resultado principal sempre sai de `mqo_newey_west`.
Testes em `code/test_hac_utils.py`. Os scripts antigos **não** foram ajustados: serão
refeitos nas tarefas indicadas.

| Arquivo : linha | Regressão | `maxlags` atual | Saída (no LaTeX?) | Tarefa |
|---|---|---|---|---|
| `scripts/test_h1_bvrp_mean.py` | teste H1 (média na constante) | h+1 = 31; sensibilidade 7/30/60/90 | `outputs/T2/` | **migrado no T3** para `mqo_newey_west` |
| `code/regenerate_cap5.py:50-53` (chamadas em `:64`, `:252`, `:374`) | `ret_fut_h` ~ BVRP (proxy); `ret_fut_h` ~ RV + IV; reta da figura com h = 30 | **h** (1, 5, 10, 20, 30, 60); 30 na figura | `tab_ols_basico_multihoriz`, `tab_ols_rv_iv_multihoriz`, figuras do Cap. 5 (sim) | **T10** |
| `code/analyze_magnitude_bvrp.py:33, 54` (chamadas em `:56`, `:109`) | \|ret_fut_30d\| ~ BVRP; \|ret_fut_30d\| ~ RV + IV | **30** fixo (h = 30) | console | **T10** |
| `scripts/regenerate_cap9.py:100` (`p_nw`) | `ret_fut_h` ~ dummy de regime (Tab. 9.4) | **h** | `tab_cap9_04_ttest` (sim) | **T11** (com T9) |
| `scripts/regenerate_cap9.py:274` | `ret_fut_h` ~ BVRP + dummy + interação (Tab. 9.5) | **h** | `tab_cap9_05_regressoes` (sim) | **T11** (com T9) |
| `code/test_log_transform_cap5.py:22, 40` | transformações de y (arcsinh, log, \|y\|) | **30** fixo | figura de teste em `figs/cap5` (não) | **candidato a descarte** — só marcado, nada apagado (usa `dataset_bvrp_with_skew.csv`, arquivado e incompatível; ver docstring de `analyze_magnitude_bvrp.py`) |
| `code/cap5_ols_vrp.ipynb` (`run_ols_hac`) | notebook original do Cap. 5 | **5** fixo | substituído por `regenerate_cap5.py` | **candidato a descarte** — só marcado, nada apagado |

Os scripts de estratégias (A1: `regenerate_cap7.py`, `scripts/cap7/`) não usam HAC.

## 5. Texto (não é código)

| Item | Tarefa |
|---|---|
| A nota da Tab. 3.1 descreve a RV como "desvio padrão anualizado". O código (correto, padrão da literatura) usa a raiz da média dos retornos quadráticos, sem subtrair a média. **Manter o código; corrigir o texto.** O plano também pede chamar a medida de **volatilidade histórica** (janela móvel de h dias) e explicar a diferença para a volatilidade realizada. | **C2.4**, F7 |
| O texto usa N = 1.775 em vários lugares; depois do T0 o N de referência é **1.806** (24/03/2021 a 03/03/2026). | Fase 2 (C3 e seguintes) |
| **Cap. 3 — artefatos do buraco de mar/2023 (T0).** Eram artefatos: (i) o máximo do retorno diário da Tab. 3.1 (0,2066 em 01/04/2023, na verdade um retorno de 32 dias; o novo máximo é 0,1353, em 28/02/2022); (ii) o máximo do BVRP (32,49 em 14/04/2023; o novo é 31,66, em 14/02/2026); (iii) o prêmio positivo de abril/2023 (BVRP médio +28,2 → −4,6; RV média 84,1 → 51,3). Qualquer menção a esses episódios no texto deve ser revista, assim como descrições da cauda direita do BVRP (p95 15,96 → 11,71; assimetria 0,609 → 0,383). | **C3** (reescrita do Cap. 3; ver também C3.1, distribuição bimodal) |
| **Cap. 6 novo (atual Cap. 5) — defasagens do HAC.** As tabelas publicadas do Cap. 5 (`tab_ols_basico_multihoriz`, `tab_ols_rv_iv_multihoriz`, `tab_bootstrap_ci`) usavam `maxlags = h`, e não h+1; as notas dizem "$h$ defasagens". Na reescrita, as tabelas refeitas no T10 usarão h+1 (`mqo_newey_west`) e o texto deve dizer "h+1 defasagens" (C4.4), sem comparar diretamente com os números antigos. | **C6** (com T10 e C4.4) |
| A Tab. 3.2 publicada não é reproduzível bit a bit (só o ADF do BVRP bate; as conclusões batem). O novo `code/build_desc_stats_T1.py` passa a ser o gerador. | T1 |
| C3.2 pede nota de que 2026 é ano incompleto "(dados até julho/agosto)"; a amostra de referência termina em **03/03/2026** (os dados brutos vão até 02/05/2026). Conferir a data na legenda. | **C3.2** |

## 6. T4 — variáveis explicativas (`data/ml_dataset_T4.csv`)

Gerado por `code/build_ml_dataset_T4.py`; teste de vazamento em `code/test_T4_vazamento.py`
(dados brutos cortados em t para 29 datas; controle positivo). Dicionário de variáveis,
ADF/KPSS e descritivas em `outputs/T4/`. O dataset tem o alvo `alvo_bvrp_30d_fut` e **17**
variáveis explicativas (16 do T4 + `bvrp_realizado_defasado`, acrescentada no T5); nenhuma
outra coluna prospectiva.

| Decisão / pendência | Tarefa |
|---|---|
| **"Médias" no plano** ("volatilidade histórica de 1, 5, 30, 60 e 90 dias, médias e IV") foi interpretado como médias móveis da IV e do preço, **transformadas** (`iv_menos_ma5d`, `iv_menos_ma30d`, `log_close_ma30d`), porque as médias em nível são I(1). **Interpretação a confirmar com o Prof. Marcelo.** | T4 → reunião |
| Texto do Cap. 5 novo: `vh_1d`, `vh_5d` e `vh_30d` seguem a estrutura do modelo HAR de **Corsi (2009)** (volatilidade diária, semanal e mensal). | C5 |
| **Critério de transformação (revisado):** transformar o que tem **evidência robusta de raiz unitária** — log do preço (`close` → `ret_1d`, `ret_acum_*`, `log_close_ma30d`) e médias móveis da IV em nível (→ `iv_menos_ma5d`, `iv_menos_ma30d`); **manter em nível as volatilidades persistentes, mesmo com ADF na fronteira**. Caso concreto: `vh_90d` passa de "longa memória" (ADF p = 0,089, N = 1.806) para "I(1)" (p = 0,194, N = 1.777) com a retirada de só 29 observações; a janela de 90 dias compartilha 89 de 90 dias entre observações vizinhas, o que gera persistência mecânica e reduz o poder do ADF. Isso não é evidência robusta de raiz unitária. `vh_90d` e `iv_30d` (p = 0,078) ficam classificadas como "longa memória (ADF instável/na fronteira)", em nível; o dicionário de `outputs/T4/` mostra a classe do teste e a classificação adotada. | T4 (C4) |
| **Colinearidade exata:** `vrp_30d = vh_30d − iv_30d`. As três ficam no dataset, mas **não podem entrar juntas num MQO**; cada script de modelo escolhe duas. | T5, T10 |
| `rv_30d` **não** entra no dataset novo: é idêntica a `vh_30d`, e o texto vai usar "volatilidade histórica". | C2.4 |
| Variação do prêmio: `d_vrp_1d` usa só a **proxy** (`vrp_30d`); a variação do prospectivo usaria retornos até t+29. | T5 |
| **Dois N de referência:** **1.806** (descritivo: Cap. 3 e teste H1, 24/03/2021 a 03/03/2026) e **1.776** (modelagem: Caps. 5 a 7, **23/04/2021** a 03/03/2026). A diferença (30 obs.) vem do início do DVOL (24/03/2021): `bvrp_realizado_defasado` exige a IV de t−30 (antes do T5 eram 1.777, a partir de 22/04/2021, limitados por `iv_menos_ma30d`). **O Cap. 4 deve explicar a diferença.** | C4.2 |
| **Corte de h = 60 também no Cap. 5.** A amostra de modelagem mantém o corte de h = 60 (termina em 03/03/2026), embora o alvo prospectivo do Cap. 5 precise só de 30 dias à frente (permitiria ir até 02/04/2026). Justificativa: **uma única amostra de modelagem (N = 1.776) para os Caps. 5 a 7**, o que torna os resultados comparáveis entre capítulos. Explicar no Cap. 4. | C4.2 |
| Variáveis de **calendário** (`month`, `weekday`, `is_month_start`, `is_month_end`) ficaram de fora: não constam do T4. **Possível teste de robustez.** | T5/T7 (robustez) |
| Variáveis de **regime** (`vrp_regime_num`, tercis da amostra inteira) ficaram de fora. | **T9** |
| **Variável acrescentada no T5:** `bvrp_realizado_defasado(t) = vh_30d(t) − iv_30d(t−30)`, o BVRP prospectivo de t−30, realizado em t (confere com o alvo de t−30 em todas as datas). **Para o texto:** ela é igual a `vrp_30d(t)` mais a variação da IV em 30 dias, `iv(t) − iv(t−30)`. | T5, C5 |
| **Candidata para o T5** (não criada): `vh_90d − vh_30d`, a inclinação da estrutura a termo da volatilidade histórica. | **T5** |

## 7. T8 — janela de estimação e divisão treino/teste (`code/split_utils.py`)

Testes em `code/test_split_utils.py` (verificador de embargo independente, sobre as datas;
controle positivo com divisões sem embargo).

| Decisão | Valor | Tarefa |
|---|---|---|
| **Embargo** | treino da origem t: s ≤ t − h (h = 30 no Cap. 5; h = horizonte do retorno, até 60, nos Caps. 6 e 7); vale para toda data de teste u ≥ t | T5, T10, T11 |
| **Janela** | **expansiva** (principal; N pequeno) e **móvel** de tamanho fixo (robustez) | C4.2 |
| **Primeira janela de estimação** | **730 obs. (2 anos): 23/04/2021 a 22/04/2023** (era 22/04 a 21/04 antes do T5), igual para todos os capítulos; o **T9 calcula o corte dos regimes só com ela** | **T9** |
| **Primeira origem** | **21/06/2023** para todos os h (t0 = 730 − 1 + 60): mesmo período fora da amostra (**987** datas, até 03/03/2026) nos Caps. 5 a 7. No Cap. 5 (h = 30), o treino da 1ª origem na janela expansiva tem 760 obs. (até 22/05/2023). Antes do T5: 20/06/2023 e 988 datas | C4.2 |
| **Janela móvel** | 730 obs.; com h < 60 ela já começa deslizada na 1ª origem (começa em 60 − h) | C4.2 |
| **Reestimação** | a cada **30 dias**: 33 origens; o modelo da origem t prevê de t até a véspera da próxima origem | T7 (custo das árvores) |
| **Validação cruzada do T7** | 5 dobras expansivas dentro do treino de cada origem, com o **mesmo embargo** (treino da dobra: s ≤ v − h, v = início da validação) | **T7** |

**Decisões confirmadas pelo autor:** (i) mesma primeira origem (21/06/2023 depois do T5,
`h_primeira_origem = 60`) para todos os capítulos — além da comparabilidade, o regressor do
Cap. 6 é o BVRP previsto no Cap. 5, então as duas séries precisam cobrir o mesmo período fora
da amostra; (ii) primeira janela de 730 obs., janela móvel de 730 e reestimação a cada 30 dias.

| Pendência para as próximas tarefas | Tarefa |
|---|---|
| O BVRP previsto é um **regressor gerado** (Pagan, 1984). O bootstrap em bloco deve **reestimar o modelo do Cap. 5 dentro de cada reamostragem**, para incorporar essa incerteza no erro-padrão. | **T10** |
| As 987 previsões fora da amostra têm alvos de 30 dias sobrepostos (~33 observações independentes). Testes de comparação de previsões (**Diebold–Mariano, Clark–West**) devem usar **HAC com h+1 defasagens, via `hac_utils`**. | **T5** |

### 7.1 Inventário das divisões treino/teste dos scripts antigos (para o texto do C4.2)

| Script | Dados | Alvo | Divisão | Datas | Embargo treino → teste | Validação interna | No LaTeX? |
|---|---|---|---|---|---|---|---|
| `scripts/regenerate_cap8.py` (Tabs. 8.1–8.2, SHAP) | `ml_dataset.csv` (1.524 obs. na versão publicada; 1.555 depois do T0) | BVRP retrospectivo em t+1 | **70/30, uma única divisão fixa**, sem reestimação | publicado: treino 30/11/2021 a 30/11/2024 (1.066); teste 01/12/2024 a 02/03/2026 (457) | **não** | 5 dobras expansivas com embargo de 30 obs. (seleção de α) | **sim** (Cap. 6 unificado, "Protocolo de avaliação fora da amostra": "divisão temporal 70%/30%") |
| `code/train_linear_cap6.py` | `ml_dataset.csv` | `ret_fut_1d` | 70/30 fixa | ~nov/2021 a nov/2024 / dez/2024 a mar/2026 | não | `TimeSeriesSplit(5, gap=30)` | não |
| `scripts/regenerate_cap6_scatter.py` (Figs. 6.11/6.12) | `bvrp_ml_target_fut_1d.csv` | BVRP em t+1 | **70/15/15** (treino/validação/teste) fixa | a partir de 24/03/2021 | não | busca do XGBoost no bloco de validação | não |
| `chapters/Cap6_ML.ipynb` (legado) | `bvrp_with_targets.csv` (623 obs., 27/02/2023 a 11/12/2024) | `bvrp_fut_1d` | 70/15/15 fixa | treino até 07/06/2024; validação até 08/09/2024 | não | bloco de validação | versão antiga do Cap. 6 |
| `chapters/cap8_ml/cap8_ml_bvrp_oos.ipynb` (legado) | `ml_dataset.csv` | BVRP | 70/30 fixa | — | não | — | versão antiga do Cap. 8 |
| `code/train_ml_baseline.py`, `code/train_xgboost.py` | `ml_dataset.csv` | `ret_fut_5d` | **80/20** fixa | — | não | nenhuma | não |
| `code/train_rf_gridsearch.py` | `ml_dataset.csv` | `ret_fut_5d` | 80/20 fixa | — | não | `TimeSeriesSplit(5)` **sem** gap | não |
| `code/regenerate_cap5.py`, `scripts/regenerate_cap9.py` | `vrp_with_targets.csv` / `vrp_with_regimes.csv` | retornos futuros | **amostra inteira** (regressões dentro da amostra) | — | — | — | sim |
| `scripts/regenerate_cap7.py`, `code/analyze_vrp_regimes.py` | `vrp_with_regimes.csv` / `vrp_with_targets.csv` | — | limiar em janela expansiva com mínimo de 252 obs. (não é treino/teste) | — | — | — | sim |

**Para o texto (C4.2) — dizer explicitamente:** nenhuma divisão antiga tinha embargo entre treino
e teste nem reestimação; todas eram uma única divisão fixa por proporção (70/30, 70/15/15 ou 80/20).
**O Cap. 8 publicado usava uma única divisão 70/30, sem intervalo entre treino e teste.** O protocolo
novo (T8) substitui isso por origens com reestimação a cada 30 dias, janela expansiva e embargo
s ≤ t − h.

## 8. T6 — diagnóstico da Figura 6.1 (previsão do LASSO)

Script: `code/diagnostico_T6.py` (reproduz `regenerate_cap8.py` na versão `1e1f146`, com os
dados daquele commit; confere os coeficientes da Tab. 6.1 e os R² da Tab. 6.2). Saídas em
`outputs/T6/`. Nenhum script antigo alterado. A imagem da Figura 6.1 no PDF anotado é idêntica,
pixel a pixel, à gerada por `1e1f146`.

**Sintomas** (PDF anotado, p. 58: *"parece estar sempre indicando BVRP negativo… tem certamente
algo errado"*; reunião, 1:15: *"o laranja… nunca vai para cima… não sobe do 0"*).

**Causa confirmada — hipótese (b):** o nível do prêmio em t (`vrp_30d`) estava **excluído** das
variáveis (`COLS_EXCLUIR` continha o próprio alvo), assim como `rv_30d` e `iv_30d`. A única
informação de nível era `vrp_regime_num`, o tercil do próprio BVRP em t (3 valores), com
coeficiente 9,18 por desvio-padrão, 12 vezes o segundo maior. A previsão era, na prática,
intercepto (−7,85) + 9,18 × regime padronizado: **três patamares (~−24, ~−11, ~+1), com teto
em ~+3,9**. Nos 100 dias de teste com BVRP realizado positivo (média +12,9; máximo +31,7), a
previsão média foi +0,6.

**Hipótese (a) descartada:** o alvo era o nível (não a variação), havia intercepto e o LASSO
(α = 0,001) manteve 11 de 11 coeficientes.

| Variante (LASSO, α = 0,001; divisão 70/30 original) | Média | Mín. | Máx. | % neg. | R² fora da amostra | Média nos 100 dias realizados > 0 |
|---|---|---|---|---|---|---|
| Realizado (BVRP retrospectivo em t+1) | −4,25 | −22,53 | 31,66 | 78,1% | — | +12,9 |
| (A) original: com regime, sem BVRP defasado | −7,62 | −25,91 | 3,85 | 62,8% | 0,388 | +0,6 |
| (B) sem regime e sem BVRP defasado | −7,79 | −15,45 | 6,13 | **99,8%** | −0,094 | −6,7 |
| (C) com `vrp_30d` em t contínuo, sem regime | −4,41 | −21,66 | 29,72 | 78,6% | **0,945** | +11,2 |
| Persistência pura (`vrp_30d` em t) | −4,29 | −22,53 | 31,66 | 78,1% | 0,937 | +12,0 |

- **"Sempre negativa"** vale literalmente para a variante (B): sem nenhuma informação de nível, a
  previsão colapsa na média do treino (−7,8) e é negativa em 99,8% dos dias. Em (A), o regime
  cria o patamar de cima (~0), que é o que o Prof. Marcelo viu como "fica no 0".
- Com o nível contínuo (C), as previsões acompanham os dias positivos e o R² sobe de 0,388 para
  0,945, só pouco acima da persistência (0,937).
- **Regime:** `vrp_regime_num` coincide 100% com tercis em **janela expansiva** (mín. 252 obs.)
  e 80% com tercis da amostra inteira. **Não houve vazamento do período de teste pelo regime**; o
  defeito é o nível do prêmio entrar só discretizado.
- **LASSO × MQO:** com 11 variáveis e α = 0,001, a regularização praticamente não atua:
  diferença máxima entre as previsões de 0,012 (A), 0,010 (B) e 0,026 (C) p.p., correlação 1,000
  (C5.2).

**Lições para o T5:**

| Lição | Tarefa |
|---|---|
| O **nível da proxy (`vrp_30d` em t) deve entrar como variável contínua** (já está em `ml_dataset_T4.csv`). | **T5** |
| **Nada de regime discretizado com cortes da amostra inteira**; o T9 refaz os regimes sem vazamento (corte só com a primeira janela de estimação do T8). Mesmo sem vazamento, um regime discretizado não substitui o nível contínuo. | **T5, T9** |
| **α do LASSO escolhido por validação cruzada embargada** (`split_utils.divisoes_validacao_cruzada`) e a **comparação LASSO × MQO reportada** (diferença máxima e correlação das previsões). | **T5, T7** (C5.2) |
| O alvo antigo era o BVRP **retrospectivo em t+1**, quase igual ao de t (autocorrelação ≈ 0,94); por isso a persistência tinha R² de 0,937. O alvo novo (`alvo_bvrp_30d_fut`) não tem essa propriedade (correlação com a proxy de −0,02), e o benchmark de persistência precisa ser redefinido (seção 2.1). | **T5** |

## 9. T5 e T7 — previsão do BVRP prospectivo e ajuste das árvores

Script: `code/previsao_bvrp_T5.py`; métricas em `code/avaliacao_utils.py`; testes em
`code/test_T5.py`. Previsões em `data/previsoes_bvrp_T5.csv` (date, origem, janela, modelo,
previsao; **sem o alvo**, que vira regressor no T10); tabelas e figuras em `outputs/T5/`.

**Critério de avaliação — decisão nossa, a validar com o Prof. Marcelo.** O critério do plano
("R² dentro da amostra ≥ 0,94, persistência") foi definido para o alvo antigo (retrospectivo em
t+1) e não se aplica ao alvo prospectivo. Adotado:
- **principal:** R² fora da amostra contra a **média histórica** (Campbell e Thompson, 2008),
  calculada só com alvos conhecidos em cada origem (s ≤ t − 30);
- **secundários:** R² contra a **proxy retrospectiva** (`vrp_30d` em t) e contra a
  **persistência viável** (`bvrp_realizado_defasado` em t, responde à seção 2.1);
- **testes:** Clark–West contra a média histórica (unilateral) e Diebold–Mariano entre todos os
  pares, com HAC de h+1 = 31 defasagens via `hac_utils`. 987 previsões com alvos sobrepostos
  equivalem a ~33 observações independentes: pouco poder;
- o modelo de referência é um parâmetro (`--referencia`).

| Decisão | Valor |
|---|---|
| Variáveis | lineares (MQO, Ridge, LASSO): 16 (todas menos `vh_30d`, pela colinearidade exata); árvores (floresta aleatória, XGBoost): 17 |
| Hiperparâmetros | em **cada reestimação** (33 origens), por validação cruzada embargada (5 dobras, treino da dobra com s ≤ v − 30), menor MSE médio. Ridge: α em 15 pontos de 10⁻³ a 10⁴; LASSO: 13 pontos de 10⁻³ a 10¹; floresta: 500 árvores × `max_depth` {3, 6, sem limite} × `min_samples_leaf` {5, 20, 50} × `max_features` {1/3, 1} (18); XGBoost: `n_estimators` {200, 500} × `learning_rate` {0,03; 0,1} × `max_depth` {2, 4} × `min_child_weight` {1, 10}, `subsample` = `colsample_bytree` = 0,8 (16) |
| Placebo | alvo embaralhado dentro do treino de cada origem: **10 permutações nas árvores** (hiperparâmetros da rodada principal) e **200 nos lineares** (α reescolhido). **Só como verificação de vazamento**, com o critério **mediana do R² placebo ≤ 0** em cada modelo. Uma permutação só era frágil (no teste de fumaça, com 60 obs., os lineares deram +0,03 por acaso). **Não é teste de significância:** permutações independentes destroem a autocorrelação dos alvos de 30 dias sobrepostos, e a distribuição placebo fica estreita demais; além disso, 10 permutações não dão resolução abaixo de ~0,09. **A significância vem do Clark–West e do Diebold–Mariano com HAC.** |
| Robustez | janela móvel de 730, com o mesmo ajuste |
| Grades (2ª rodada) | Na 1ª rodada, o escolhido caía no limite da grade (LASSO em α = 10; floresta e XGBoost no canto mais regularizado). As grades foram estendidas: Ridge até 10⁵; LASSO até 10²; floresta com profundidade {1, 2, 3, 6, sem limite}, folha mínima {5, 20, 50, 100, 200} e `max_features` {0,2; 1/3; 1} (75 combinações); XGBoost com `n_estimators` {50, 100, 200, 500}, `learning_rate` {0,01; 0,03; 0,1}, `max_depth` {1, 2, 4} e `min_child_weight` {1, 10} (72). A validação cruzada das árvores passou a distribuir os ajustes (combinação × dobra) entre os núcleos, com o mesmo resultado |

### 9.1 Resultados (2ª rodada, grades estendidas; 987 previsões, 21/06/2023 a 03/03/2026)

| Modelo | R² vs. média (expansiva) | R² vs. média (móvel) | CW p (expansiva) | DM p vs. média (expansiva) | R² dentro da amostra | % negativas | Média nos 290 dias com realizado > 0 |
|---|---|---|---|---|---|---|---|
| Ridge | **0,223** | 0,095 | 0,0005 | 0,042 | 0,21 | 93% | −5,1 |
| Floresta aleatória | 0,143 | 0,073 | 0,0008 | 0,051 | 0,19 | 100% | −6,8 |
| LASSO | 0,123 | 0,039 | 0,016 | 0,071 | 0,09 | 100% | −8,7 |
| XGBoost | 0,100 | 0,066 | 0,0033 | 0,064 | 0,16 | 100% | −8,5 |
| MQO | 0,097 | −0,037 | 0,0007 | 0,517 | 0,29 | 73% | −2,3 |
| Persistência viável | −1,003 | −1,197 | 0,53 | — (pior que a média) | — | 70% | −9,9 |
| Proxy retrospectiva | −0,506 | −0,652 | 0,23 | — (pior que a média) | — | 78% | −8,5 |

Realizado fora da amostra: média −5,3; positivo em 290 de 987 dias (29%).

| Achado | Tarefa |
|---|---|
| **Comparação múltipla (5 modelos contra a média; Bonferroni, limite 0,05/5 = 0,01).** Clark–West: sobrevivem Ridge, floresta aleatória, XGBoost e MQO (não o LASSO, p = 0,016), nas duas janelas. Diebold–Mariano: **nenhum sobrevive** (o menor é o do Ridge, p = 0,042). **O texto do Cap. 5 deve apoiar a conclusão principal no Clark–West, mencionando a correção.** Ressalva: o CW ajusta pelo ruído de estimação a favor do modelo maior e pode ser significante mesmo com R² fora da amostra negativo (MQO na móvel: R² = −0,037, CW p = 0,004); e 987 previsões com alvos sobrepostos equivalem a ~33 observações independentes. | **C5**, T5 |
| **XGBoost: o sobreajuste da 1ª rodada (R² dentro da amostra 0,67, fora −0,08) vinha da grade truncada.** Com a grade estendida, a validação cruzada escolhe árvores de profundidade 1 com taxa de 0,01, e a distância some (0,16 dentro, 0,10 fora). Registrar no texto que a grade inicial não permitia regularização suficiente. | **C5**, T7 |
| **LASSO × MQO: regularização efetiva** (diferença máxima de 25,4 p.p. e correlação de 0,34 entre as previsões, janela expansiva), ao contrário do T6. No limite, o LASSO **zera todos os coeficientes e prevê exatamente a média histórica** em 12 das 33 origens (26 de 33 na móvel) — daí 100% de previsões negativas e R² idêntico nas duas rodadas. | **C5.2** |
| **Nenhum modelo antecipa os episódios positivos** (ex.: fev/2026, +45). Nos 290 dias com prêmio positivo, todas as previsões médias são negativas (−2,3 a −8,7). As previsões **encolhem para a média por baixa previsibilidade, não por erro de especificação** (diferente do T6). | **C5**, F2 |
| **Limites da grade depois da extensão:** Ridge e LASSO escolhem valores interiores (0 origens nos limites). As árvores continuam no limite de **menor capacidade**: floresta com profundidade 1 em 33/33 origens (25/33 na móvel); XGBoost com taxa 0,01 em 33/33, 50 árvores em 22/33 e profundidade 1 em 28/33. Esse limite é estrutural: árvores de profundidade 1 com encolhimento máximo já se aproximam da média histórica (desvio-padrão das previsões de 2,0–2,3 contra 4,5 do Ridge); estender mais só aproximaria da média. É evidência de **sinal não linear fraco**. | **T7**, **T11**, C5 |
| As referências ingênuas são piores que a média: persistência viável (R² −1,00) e proxy (−0,51), coerente com a correlação ≈ 0 da seção 2.1. | C5, C2.5 |
| Placebo (verificação de vazamento): medianas entre −0,016 e 0,000 em todos os modelos. | T5 |

### 9.2 Decisões para os capítulos seguintes (fixadas antes de ver os resultados do T10/T11)

| Decisão | Tarefa |
|---|---|
| **Previsão que alimenta os capítulos seguintes**, escolhida agora para evitar escolher o modelo depois de ver os resultados: **T10** usa o **Ridge** como principal e a **floresta aleatória** como robustez; **T11** usa a **floresta aleatória** como principal (o plano pede árvore) e o **Ridge** como robustez. Fonte: `data/previsoes_bvrp_T5.csv`, janela expansiva. | **T10**, **T11** |
| **Forma do BVRP previsto no T11.** O plano define BVRP previsto = f(variáveis) − IV, isto é, prever a RV futura e subtrair a IV conhecida em t. O T5 previu o BVRP diretamente. No MQO as duas formas são equivalentes (com `iv_30d` entre os regressores, a regressão de RV − IV só desloca em 1 o coeficiente da IV); no Ridge e nas árvores, não (a penalidade e as quebras atuam sobre alvos diferentes). **O T11 deve testar as duas.** | **T11** |

## 10. T9 — regimes sem look-ahead

Scripts: `code/regimes_T9.py` (corte e regimes), `code/interacao_regime_T9.py` (teste F e
avaliação fora da amostra) e `code/test_regimes_T9.py`. Regimes em `data/regimes_T9.csv`
(`ml_dataset_T4.csv` intacto); cortes, figuras e resultados em `outputs/T9/`.

> **DESTAQUE — comunicar ao Prof. Marcelo junto com os resultados.** A bimodalidade do BVRP na
> Fig. 3.2 (anotações da p. 36: *"claramente bimodal"*, *"moda local na cauda positiva"*; reunião,
> 43:56–44:26: *"você vai precisar de 1 negócio acima de 20"*) era **em boa parte artefato do buraco
> de março/2023** corrigido no T0: a antimoda em +20,2 (o "~20" do plano) e a moda em +27 vinham
> de abril/2023 — **30 das 73 datas com BVRP > 20 eram de abril/2023**. **Depois do T0, a
> distribuição do BVRP é unimodal** (moda em −9 na amostra; em −17 na 1ª janela).
> **Consequência para o C3.1:** a evidência de dois regimes passa a ser a **bimodalidade da RV de
> 30 dias** (Fig. 8.2, p. 77, *"olha aqui a bimodalidade novamente"*), não a do BVRP.
> Figura: `outputs/T9/fig_T9_bimodalidade_bvrp.png`.

| Decisão | Valor | Tarefa |
|---|---|---|
| Variável de regime | **RV de 30 dias (`vh_30d`)**, conhecida em t; dois regimes (D4): alta se `vh_30d` > corte | T9 |
| Corte (principal) | **mistura de duas normais em log(RV), estimada só na 1ª janela** (23/04/2021 a 22/04/2023), convergida; **corte fixo de 37,3% a.a.** para toda a amostra | T9 |
| Corte (robustez) | o mesmo método reestimado em cada origem do `split_utils` (janela expansiva), só com dados até a origem | T9 |
| Busca do corte | ponto em que a posteriori do componente de alta passa de 1/2: a maior raiz crescente da quadrática de igual posteriori (o componente baixo domina entre as raízes) | T9 |

**Erro corrigido no próprio T9 (registrar):** o corte de **47,4** proposto no passo 1 vinha de uma
estimação interrompida — com a tolerância padrão do scikit-learn (`tol` = 10⁻³), a mistura parava
após **7 iterações**, com log-verossimilhança **−154,9**, contra **−134,2** da solução convergida
(77 iterações; mesma solução em todas as sementes). **Correção:** `tol` = 10⁻⁸, 10 inicializações,
`max_iter` = 10.000 e `assert converged_` em toda estimação, inclusive nas origens; o teste
automático confere a estabilidade entre sementes (diferença < 0,1) e a coincidência com a antimoda.

**Para o texto:** o corte de **37,3** coincide com a **antimoda da densidade de núcleo da 1ª janela
(37,5)** — dois métodos independentes. O componente baixo é um grupo de dias calmos (média de
31,2% a.a., peso 0,08 na mistura). O regime de baixa tem **9,0% das datas da 1ª janela e 31,4% das
datas fora da amostra** (o peso da mistura, 0,08, é a fração estimada; 9,0% é a fração observada
abaixo do corte), coerente com a queda da volatilidade do bitcoin a partir de 2023.

**Corte expansivo (robustez) — instável:** começa em 36,8 (21/06/2023), oscila entre 32 e 43 até
o fim de 2024, **salta entre ~29 e ~51 entre fev. e jul./2025** (a mistura troca de solução: o peso
do componente baixo pula de ~0,10 para ~0,49) e cai para **~27** a partir de out./2025, quando o
componente baixo vira um pico estreito em ~26% a.a. À medida que a volatilidade cai, a RV deixa
de ser claramente bimodal na amostra expansiva. Isso reforça o corte fixo da 1ª janela como
principal. Figura: `outputs/T9/fig_T9_cortes_por_origem.png`.

**Para o texto do Cap. 7:** a instabilidade do corte expansivo a partir de fevereiro/2025 indica
que a **bimodalidade da RV perde nitidez com a queda de volatilidade do bitcoin**: a estrutura de
dois regimes é clara em 2021–2024 e enfraquece depois. Isso reforça o corte fixo como principal, e
as conclusões (teste F e Ridge) não mudam com o corte expansivo. (C7)

### 10.1 Resultados do T9

**Corte:** 37,30% a.a. (mistura de duas normais em log(RV), 1ª janela, convergida: médias 31,2 e
66,7% a.a., pesos 0,08 e 0,92); antimoda da densidade de núcleo: 37,47; idêntico em 6 sementes.

| Regime de alta volatilidade (`vh_30d` > 37,3) | 1ª janela | Fora da amostra | Amostra inteira |
|---|---|---|---|
| Fração de dias | 91,0% | 68,6% | 77,8% |

**Teste F das interações** (MQO com X + D + D·X, 16 + 1 + 16 regressores; Wald com HAC de 31
defasagens via `hac_utils`; N = 1.776):

| Corte | H0 | χ² (q) | p | R² dentro da amostra |
|---|---|---|---|---|
| Fixo | 16 interações = 0 | 94,9 (16) | 3×10⁻¹³ | 0,328 |
| Fixo | interações + dummy = 0 | 94,9 (17) | 8×10⁻¹³ | 0,328 |
| Expansivo | 16 interações = 0 | 85,6 (16) | 2×10⁻¹¹ | 0,318 |
| Expansivo | interações + dummy = 0 | 87,0 (17) | 2×10⁻¹¹ | 0,318 |

**Fora da amostra** (987 previsões, mesmas origens do T5, janela expansiva):

| Modelo | R² vs. média, sem regime (T5) | Com regime (fixo) | Com regime (expansivo) | DM vs. sem regime: p (fixo) | CW vs. média: p (fixo) |
|---|---|---|---|---|---|
| Ridge | 0,223 | 0,229 | 0,227 | 0,81 | 0,0002 |
| MQO | 0,097 | −0,174 | −0,165 | 0,21 | 0,0005 |

**Leitura:** as interações regime × variáveis são muito significantes **dentro** da amostra, mas
**não melhoram a previsão fora dela**: no Ridge, +0,006 no R² (DM p = 0,81); no MQO, com 33
regressores, pioram (sobreajuste). Coerente com a anotação da p. 82 (*"teste-F … sem muita
esperança"*). Conclusões iguais com o corte fixo e o expansivo. (C7, T10, T11)

| Como a variável de regime entra nas próximas tarefas (proposta, a validar) | Tarefa |
|---|---|
| **T10 (linear):** retorno ~ BVRP previsto, com a dummy `regime_alta_fixo` e a interação BVRP previsto × regime; robustez com `regime_alta_expansivo`. O regime é conhecido em t (corte da 1ª janela; `vh_30d` de t). | **T10** |
| **T11 (árvore):** a árvore escolhe os cortes sozinha (C5.6, C7.1): entra `vh_30d` (e as demais volatilidades), **não** a dummy; robustez acrescentando `regime_alta_fixo`. Comparar os cortes escolhidos pela árvore com 37,3. | **T11** |
| Nada de regime calculado na amostra inteira (anotações das p. 76 e 82: *"look-ahead bias"*). | T10, T11, A1 |

## 11. T10 e T12 — retorno futuro sobre o BVRP previsto e dispersão do Cap. 3

Código: `code/retorno_bvrp_T10.py` (saídas em `outputs/T10/`), `code/plot_vrp_vs_return_T12.py`
(saídas em `outputs/T12/`), testes em `code/test_T10.py`. Nada em `figs/` ou `tables/`.

**Especificações (T10).** Amostra: as 987 datas fora da amostra do T5 (21/06/2023 a 03/03/2026,
janela expansiva). Horizontes h = 1, 5, 10, 20, 30, 60; EP de Newey–West com h+1 defasagens
(`hac_utils`); limite de Bonferroni 0,05/6 = 0,0083.
1. Principal: ret_fut_h ~ BVRP previsto. Regressor do Ridge (principal) e da floresta aleatória
   (robustez), como fixado na seção 9.2 antes de ver os resultados.
2. Componentes (C6.5): ret_fut_h ~ `vh_30d` + `iv_30d`, teste de b1 + b2 = 0.
3. Regime (T9): ret_fut_h ~ BVRP previsto + D + D × BVRP previsto, D = `regime_alta_fixo`
   (corte de 37,3); robustez com `regime_alta_expansivo`; Wald conjunto de D e da interação.
4. Comparação com o publicado: ret_fut_h ~ `vrp_30d` (N = 1.806), maxlags = h (como nas tabelas
   do Cap. 5) contra h+1 (T3).

### 11.1 Regressor gerado: bootstrap em blocos em dois níveis

- **Desenho abandonado (registrar no Cap. 6, nota metodológica).** O primeiro desenho reamostrava
  a série inteira em blocos e aplicava as origens de previsão por posição. Com isso, as "datas
  fora da amostra" de cada reamostragem misturavam épocas, e o bootstrap estimava outro
  parâmetro: a média dos β* ficou perto de zero, e o IC não continha β̂ em nenhuma das 24
  combinações (6 horizontes × 4 variantes) do teste de fumaça.
- **Desenho adotado.** 1º nível (incerteza do regressor gerado, Pagan, 1984): em cada uma das
  33 origens, o 1º estágio é reestimado num treino reamostrado em blocos **dentro** da janela de
  treino da própria origem (posições 0 a t − 30: embargo exato) e prevê as datas reais de teste.
  2º nível (incerteza amostral): os 987 pares (previsão reestimada, retorno) são reamostrados
  em blocos dentro do período fora da amostra, e β_h é reestimado.
- **Principal:** Ridge com o α de cada origem do T5 **fixo**, bloco de 60, B = 999. Sensibilidade:
  α fixo com blocos de 30 e 90 (B = 999); α **reescolhido** por validação cruzada embargada em
  cada treino reamostrado (B = 999); floresta aleatória com os hiperparâmetros de cada origem do
  T5 fixos (bloco de 60, B = 199). Contraprova: só o 2º nível, com as previsões originais
  fixas (B = 999). Sementes fixas (`SEMENTE = 20261001`, combinada com bloco e reamostragem).
- **Por que o α reescolhido não é o principal (mecanismo da distorção).** No bootstrap em
  blocos, blocos repetidos podem cair no treino e na validação da mesma dobra; a validação
  cruzada passa então a favorecer α menores (menos regularização), e a previsão fica mais
  ruidosa. É uma distorção do **procedimento**, não incerteza real da seleção de α. No teste de
  fumaça, a razão média dos β*/β̂ foi ~0,3–0,4 com α reescolhido, contra ~0,6–0,7 com α fixo.
- **Inferência, para cada h (escala coerente, decidida após a rodada completa; ver 11.4):**
  λ = β̄*/β̂ (atenuação); EP corrigido = EP_boot / λ; teste de H0: β = 0 por
  β̂ / (EP_boot / λ), equivalente a β̄* / EP_boot (normal); IC principal β̂ ± 1,96 × EP_boot / λ,
  coerente com o teste. Ao lado: EP HAC (ignora o 1º estágio), EP só do 2º nível e as razões
  entre os EPs (CSV). IC básico (2β̂ − q97,5; 2β̂ − q2,5) como complemento — ele se refere à
  relação **corrigida da atenuação**, não a β̂; viés (β̄* − β̂). Na contraprova só do 2º nível
  não há 1º estágio, e o EP não é corrigido.
- **Atenuação.** A reestimação do 1º estágio acrescenta ruído à previsão e atenua β no 2º
  estágio (erro nas variáveis); a contraprova só do 2º nível é centrada em β̂, e com os dois
  níveis a média dos β* fica entre 0 e β̂ (testes 6a e 6b de `test_T10.py`). **Para o texto do
  Cap. 6:** o β estimado com o BVRP previsto é uma estimativa **conservadora (atenuada)** da
  relação entre o prêmio prospectivo e o retorno futuro.

### 11.2 Ressalvas para o texto do Cap. 6

- **Junções dos blocos:** nas fronteiras entre blocos concatenados, a ordem temporal deixa de
  valer (limitação padrão do bootstrap em blocos móveis).
- **Poucos blocos no 2º nível:** 987 datas em blocos de 60 são ~16 blocos; os intervalos por
  percentis são grosseiros, por isso o EP HAC é reportado ao lado e o IC principal usa o EP.
- **Poder baixo nos horizontes longos:** em h = 60 há só ~16 períodos não sobrepostos na amostra
  fora da amostra; "não significante" nesses horizontes **não** é evidência de ausência de
  relação.
- **Comparações múltiplas:** 6 horizontes; reportar quais sobrevivem a Bonferroni (0,0083).
- **Tabelas publicadas do Cap. 5:** usavam maxlags = h; `publicado_h_vs_h1_T10.csv` mostra o
  efeito de passar a h+1 (T3) nos mesmos dados (N = 1.806, proxy `vrp_30d`).

### 11.3 T12 — dispersão BVRP × retorno de 30 dias

`outputs/T12/fig_T12_vrp_vs_ret30d.png` substitui, para o Cap. 3, `figs/cap3/vrp_vs_return_20d.png`
(h = 20 → h = 30, horizonte do BVRP). Figura **descritiva**: proxy `vrp_30d` (conhecida em t),
amostra descritiva N = 1.806; reta de MQO com β, EP HAC (31 defasagens), p e R² na legenda, no
lugar da correlação simples do script antigo (`code/plot_vrp_vs_return.py`, não alterado). A troca
do arquivo em `figs/` e a legenda no .tex ficam para quando o Cap. 3 for reescrito (sincronizado
com o Overleaf). (C3, T12)

### 11.4 Resultados do T10

Rodada completa (B = 999 no Ridge, 199 na floresta; 987 datas, 21/06/2023 a 03/03/2026;
`test_T10.py`: todos os testes passaram). β em pontos percentuais de retorno por p.p. de BVRP
previsto (β × 100 dos CSVs).

**Principal (Ridge, α fixo, bloco de 60; teste na escala coerente):**

| h | β | EP HAC | p HAC | EP só 2º nível | EP boot bruto | λ = β̄*/β̂ | EP corrigido (EP/λ) | p boot | IC 95% | fração β* ≥ 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | −0,045 | 0,018 | 0,011 | 0,015 | 0,013 | 0,59 | 0,022 | 0,043 | [−0,089; −0,001] | 0,012 |
| 5 | −0,144 | 0,064 | 0,025 | 0,063 | 0,056 | 0,68 | 0,082 | 0,080 | [−0,306; 0,017] | 0,034 |
| 10 | −0,250 | 0,114 | 0,029 | 0,132 | 0,117 | 0,64 | 0,182 | 0,170 | [−0,606; 0,107] | 0,074 |
| 20 | −0,459 | 0,230 | 0,046 | 0,285 | 0,249 | 0,63 | 0,392 | 0,242 | [−1,228; 0,310] | 0,099 |
| 30 | −0,586 | 0,347 | 0,091 | 0,435 | 0,379 | 0,67 | 0,570 | 0,303 | [−1,703; 0,530] | 0,135 |
| 60 | −1,433 | 0,654 | 0,029 | 0,817 | 0,688 | 0,70 | 0,979 | 0,143 | [−3,352; 0,485] | 0,057 |

- **Por que o teste β̂ / EP_boot foi trocado.** Com os dois níveis, os β* encolhem para zero
  (λ ≈ 0,6–0,7) e a dispersão encolhe junto: o EP bruto em dois níveis saiu 11–16% **menor**
  que o EP só do 2º nível, embora o 1º nível acrescente incerteza (o coeficiente de variação,
  EP/|β̄*|, é maior). O teste β̂ / EP_boot misturava escalas — β̂ não atenuado sobre EP
  atenuado — e dava p pequenos demais (h = 1: 0,0007, "sobrevivendo" a Bonferroni; h = 5:
  0,0097). Na escala coerente, o EP corrigido fica 1,20–1,49 vez o EP só do 2º nível e
  1,25–1,71 vez o EP HAC (teste 6c de `test_T10.py`: EP corrigido > EP só do 2º nível em todo h).
- **Sinal:** β < 0 em todos os horizontes e nos dois modelos (Ridge e floresta).
- **Significância:** p boot de 0,043 (h = 1) a 0,30; só h = 1 abaixo de 0,05, e **nenhum
  horizonte sobrevive a Bonferroni** (0,0083). HAC: p de 0,011 a 0,091, também nenhum.
  Sensibilidade (p boot): blocos de 30, 0,042–0,20; de 90, 0,050–0,42; α reescolhido,
  0,068–0,57; floresta, 0,50–0,84. Só a contraprova, que ignora o 1º estágio, tem h = 1 abaixo
  de 0,0083 (p = 0,0025).
- **Atenuação:** λ de 0,59 a 0,70 (blocos de 30: 0,72–0,83; de 90: 0,50–0,62; α reescolhido:
  0,27–0,43; floresta: 0,34–0,53). Contraprova só do 2º nível centrada (razão 0,90–1,09;
  |β̄* − β̂| < 0,5 EP em todos os h).
- **Conclusão para o Cap. 6:** sinal negativo estável em todos os horizontes e nos dois
  modelos, mas nenhum horizonte sobrevive a Bonferroni; **a versão linear não prevê os retornos
  de forma robusta**.
- **Para o texto (interpretação do sinal):** com BVRP = RV − IV, β < 0 equivale a "prêmio de
  variância maior (IV acima da RV futura) → retorno futuro maior", a mesma direção de
  Bollerslev, Tauchen e Zhou (2009). Aqui a evidência é **fraca** (só h = 1 com p < 0,05, sem
  sobreviver a Bonferroni).
- **Floresta (robustez):** mesmo sinal em todos os h, β de magnitude parecida, mas p HAC entre
  0,15 e 0,61.
- **Componentes:** b_vh + b_iv = 0 só é rejeitado em h = 1 (p = 0,039); nos demais, p ≥ 0,18.
- **Regime (corte fixo):** no regime de baixa volatilidade (31% das datas), a inclinação é mais
  negativa (h = 30: −2,92 p.p./p.p., p = 0,005); a interação é positiva (p < 0,05 de h = 10 a 60),
  e no regime de alta a inclinação fica perto de zero (h = 30: −0,34). Wald conjunto (D e
  interação): p < 0,05 só em h = 60 (0,035). **Com o corte expansivo, não se sustenta:** interação
  com p ≥ 0,06 e Wald com p ≥ 0,16. **Para o texto:** a diferença entre regimes é frágil (some
  com o corte expansivo) e deve ser apresentada como **sugestiva**, não como resultado.
- **Publicado (proxy `vrp_30d`, N = 1.806):** β ≈ 0 em todos os h (p de 0,17 a 0,93). Passar de
  maxlags = h para h+1 muda o EP em −3% a +1%: irrelevante aqui. A proxy na amostra inteira não
  mostra a relação que o BVRP previsto mostra fora da amostra; a comparação mistura amostra
  (1.806 × 987 datas) e regressor (proxy × previsão).
- **T12:** β = 0,088 p.p. de retorno de 30 dias por p.p. de proxy (EP HAC 0,119; p = 0,46;
  R² = 0,004), N = 1.806.

## 12. T11 — BVRP previsto e retornos, versão não linear (modelo único em árvore)

Código: `code/retorno_bvrp_T11.py` (saídas em `outputs/T11/`; previsões da forma f(variáveis) − IV em
`data/previsoes_bvrp_T11.csv`), testes em `code/test_T11.py`. O código e as saídas do T5 e do T10 não
foram alterados. Decisões abaixo **registradas antes de rodar** (01/10/2026).

**Leituras do "modelo único em árvore"**

| Leitura | Papel | Base |
|---|---|---|
| **A:** ret_fut_h(t) = a + β_h · BVRP_previsto(t) + e, com BVRP_previsto(t) = f(variáveis em t) − IV(t) e f = floresta aleatória | **Principal** (Cap. 7) | Fórmula da anotação da p. 75 (*"retorno(t+h) = beta·BVRP(t) + erro(t+h), em que BVRP(t) = função(todos os preditores) − IV… RF ou XGBoost"*) e estrutura **D1** fechada na reunião (1:19–1:21: prever o BVRP e depois testar se o BVRP previsto prevê o retorno; *"linearmente não prevê; se eu fizer não linear…"*) |
| **B:** ret_fut_h(t) = g_h(variáveis em t), floresta direto no retorno, por horizonte | **Complemento**, sem bootstrap: R² fora da amostra contra a média histórica, Clark–West com HAC h+1 e placebo de vazamento | Reunião, 1:17:29 (*"é retorno como função, essa função é dada pela árvore… de todas aquelas variáveis lá e deixa ele escolher onde é que está tendo quebra"*) e 1:18:42 (crítica ao caminho "prevê retorno → prevê BVRP → volta ao retorno") |

"Único", na leitura A: um só modelo de árvore com todas as volatilidades substitui o aparato do antigo
Cap. 8 (quantil q75, dummy, interações); as quebras são escolhidas na previsão do BVRP (C7.1).

**Decisões**

| Decisão | Valor |
|---|---|
| Forma principal do BVRP previsto no retorno | **f(variáveis) − IV**: a floresta prevê `rv_30d_fut` = RV(t+1 a t+30) = alvo + `iv_30d` (= `vh_30d` de t+30) e subtrai a `iv_30d` de t. **Forma direta** (a floresta prevê o BVRP, como no T5) **como robustez**. Ridge nas duas formas como robustez (seção 9.2). Fixado antes de ver qualquer resultado do T11 |
| Previsão da RV | mesmas 33 origens, embargo s ≤ t − 30, validação cruzada embargada em cada reestimação, grades estendidas (floresta 75, Ridge 17); janela expansiva (principal) e móvel de 730; métricas do T5. Referência adicional **"média da RV − IV"** (média histórica da RV menos a IV de t): mede quanto do desempenho da forma f(variáveis) − IV vem só da IV |
| Placebo da forma f(variáveis) − IV | **no espaço da RV** (R² da RV prevista contra a média histórica da RV): no espaço do BVRP, até uma previsão sem informação (constante − IV) carrega a IV, e o R² placebo poderia ser positivo sem vazamento. Critério: mediana ≤ 0,01 (10 permutações na floresta, 200 no Ridge; ver a mudança de critério abaixo) |
| Robustez (seção 10) | floresta f(variáveis) − IV com `regime_alta_fixo` como variável adicional (expansiva) |
| Retorno | 987 datas; h = 1, 5, 10, 20, 30, 60; inferência do T10 (EP HAC h+1; bootstrap em dois níveis com hiperparâmetros fixos por origem, bloco de 60; escala coerente; contraprova só do 2º nível; Bonferroni 0,0083). **Mesmas reamostragens em todos os horizontes e nos quatro regressores** (semente do T10; o Ridge direto reproduz as réplicas do T10) |
| B | 999 por regressor (bloco de 60); blocos de 30 e 90 com B = 199 só no principal. **B = 499 no principal se a validação cruzada escolher `max_features` = 1 em ≥ 11 das 33 origens** (cada ajuste fica ~4× mais caro) |
| **Teste F (principal): F2**, H0: β_h = 0 para todo h | Wald com a covariância dos β* na escala coerente: W = β̂′(Λ⁻¹Σ*Λ⁻¹)⁻¹β̂ = β̄*′Σ*⁻¹β̄* ~ χ²(6), F = W/6 (análogo multivariado do z = β̄*/EP_boot do T10). Complementos: **max-\|t\|** do bootstrap (p ajustado por horizonte, leva em conta a correlação entre os h; responde às comparações múltiplas) e **versão HAC** empilhada (escores conjuntos, Newey–West com 61 defasagens; ignora o 1º estágio). Reportar o **número de condição** e a **matriz de correlação** dos β* (horizontes sobrepostos: Σ* perto de singular, e o Wald pode rejeitar por contrastes entre horizontes) |
| Teste F (complemento): F1 | anotação da p. 82, por h: ret ~ BVRP previsto + D + D × BVRP previsto, Wald dos 3 coeficientes (e dos 2 de regime), HAC h+1, corte fixo e expansivo (sem look-ahead) |
| F2 no T10 | aplicado às réplicas gravadas do T10 (sem recalcular), saída em `outputs/T11/` |
| Quebras (C7.1) | importância por impureza (média das 33 origens) e por permutação fora da amostra; dependência parcial do BVRP previsto em `vh_30d` e `iv_30d`, padrão e **coerente** (recalcula `vrp_30d` = `vh_30d` − `iv_30d`, identidade exata nos dados; `iv_menos_ma*` não são recalculadas — ressalva); limiares escolhidos pelas árvores em `vh_30d`, `vh_60d`, `vh_90d` e `vrp_30d`, ponderados pela redução de impureza, contra o corte do T9 (37,3) e a faixa do corte expansivo |

**Mudança do critério do placebo (fixada em 01/10/2026, antes da rodada completa).** Em todos os
placebos do T11 (leitura A, no espaço da RV, e leitura B), o critério passa de "mediana do R² placebo
≤ 0" para **"mediana ≤ 0,01"** (`TOL_PLACEBO` em `retorno_bvrp_T11.py`).
- **Motivo:** com o alvo embaralhado, a floresta colapsa na média histórica, e o R² placebo fica
  centrado em zero; o critério "≤ 0" passa a falhar por acaso. No teste de fumaça (2 permutações),
  a leitura B teve mediana positiva em 4 dos 6 horizontes (+0,001 a +0,006). **Diagnóstico com
  20 permutações** (h = 1, 20, 60): medianas de 0,0000, −0,0010 e +0,0006, com ~50% das permutações
  positivas (0,50; 0,40; 0,55) e magnitude abaixo de 0,001 — distribuição centrada em zero, sem
  sinal de vazamento. Um vazamento real daria R² placebo claramente positivo (os modelos do T5
  tiveram R² de 0,10 a 0,22); a tolerância de 1% da variância separa os dois casos.
- **O placebo serve só para detectar vazamento.** Na leitura B, a evidência de previsibilidade vem
  do **Clark–West** (HAC h+1) e do **R² fora da amostra contra a média histórica**, não do placebo.
  O embargo da leitura B é verificado à parte (`test_T11.py`, testes 3 e 7).
- **O T5 passou no critério antigo, mais rígido** (medianas entre −0,016 e 0,000), e **não é refeito**.

### 12.1 Resultados do T11

Rodada completa de 01/10/2026 (10:16 a 16:06; B = 999 por regressor e B = 199 nos blocos de 30 e
90; 987 datas, 21/06/2023 a 03/03/2026). `test_T11.py`: todos os 28 testes passaram
(`outputs/T11/log_test_T11.txt`); o Ridge direto reproduz exatamente as 999 réplicas e o resumo do
T10. A validação cruzada escolheu `max_features` = 1 em 10 das 33 origens (abaixo do limite de 11):
B = 999 no principal. β em pontos percentuais de retorno por p.p. de BVRP previsto (β × 100 dos CSVs).

> **Conclusão para o Cap. 7: a versão não linear não prevê o retorno.** Na forma principal
> (floresta, f(variáveis) − IV), β ≈ 0 em todos os horizontes (−0,006 a +0,070; p HAC de 0,55 a
> 0,94; R² ≈ 0) e o teste conjunto nos 6 horizontes não rejeita (**F2: χ²(6) = 1,45, p = 0,96**;
> blocos de 30 e 90: 0,95 e 0,96; max-|t|: 0,79; HAC empilhado: 0,93). Na leitura B (floresta
> direto no retorno), **nenhum horizonte é significante no Clark–West** (p de 0,11 a 0,94).

**Previsão do BVRP: forma direta × f(variáveis) − IV** (janela expansiva; móvel entre parênteses)

| Modelo | R² vs. média | CW p | % negativas | Média nos dias com BVRP > 0 |
|---|---|---|---|---|
| Ridge direto (T5) | 0,223 (0,095) | 0,001 | 93% | −5,1 |
| Floresta direta (T5) | 0,143 (0,073) | 0,001 | 100% | −6,8 |
| **Floresta f(variáveis) − IV (principal)** | **0,009** (−0,069) | < 0,001 | 67% | −0,8 |
| Floresta f(variáveis) − IV com `regime_alta_fixo` | −0,041 | 0,001 | 68% | −1,2 |
| Ridge f(variáveis) − IV | −0,073 (−0,047) | 0,001 | 63% | +0,7 |
| Média da RV − IV (só a IV) | −0,588 (−0,268) | < 0,001 | 25% | +9,8 |

- **A forma f(variáveis) − IV prevê o BVRP pior que a direta porque impõe coeficiente −1 à IV.** A
  referência "média da RV − IV" (só a IV, com peso 1) tem R² de −0,59: a IV com coeficiente
  unitário é uma previsão ruim da RV futura; na forma direta, o modelo aprende um efeito da IV
  encolhido. Diebold–Mariano direta × f(variáveis) − IV: floresta p = 0,28; Ridge p = 0,063.
- **A forma principal foi fixada antes dos resultados** (seção 12) e **a conclusão sobre os retornos
  não muda com nenhuma das formas** nem dos modelos (tabela abaixo e F2).
- A dummy de regime como variável adicional piora a previsão (DM p = 0,005 contra a floresta sem
  ela). A floresta para a RV escolhe árvores mais profundas que no T5 (profundidade 6 em 13 origens,
  1 em 7). Placebo (espaço da RV): medianas de 0,001 (floresta) e 0,000 (Ridge).

**Retorno sobre o BVRP previsto** (p HAC, h+1)

| h | Principal: β (p) | Floresta direta: β (p) | Ridge f(variáveis) − IV: p | Ridge direto (T10): β (p) |
|---|---|---|---|---|
| 1 | −0,006 (0,55) | −0,057 (0,15) | 0,12 | −0,045 (0,011) |
| 5 | −0,006 (0,87) | −0,113 (0,34) | 0,40 | −0,144 (0,025) |
| 10 | +0,005 (0,94) | −0,142 (0,49) | 0,58 | −0,250 (0,029) |
| 20 | +0,030 (0,83) | −0,294 (0,49) | 0,88 | −0,459 (0,046) |
| 30 | +0,070 (0,74) | −0,357 (0,61) | 0,90 | −0,586 (0,091) |
| 60 | −0,076 (0,84) | −1,235 (0,33) | 0,90 | −1,433 (0,028) |

- **Instabilidade de λ com β̂ ≈ 0.** Na forma principal, λ = β̄*/β̂ fica instável e chega a ser
  negativo (ex.: −0,35 em h = 10, bloco de 60); o EP corrigido EP_boot/λ fica indefinido (NaN no
  CSV) em parte dos horizontes. **Nesses casos, reportar o EP HAC e o F2**, que, escrito como
  β̄*′Σ*⁻¹β̄*, continua definido. A correção de escala do T10 pressupõe β̂ longe de zero. (C7, C6)

**Teste conjunto nos 6 horizontes (F2)**

| Regressor | Wald, dois níveis (p) | max-\|t\| (p global) | HAC empilhado (p) | Só 2º nível (p) |
|---|---|---|---|---|
| **Floresta f(variáveis) − IV** | **0,96** | 0,79 | 0,93 | 0,95 |
| Floresta direta | 0,98 | 0,71 | 0,23 | 0,51 |
| Ridge f(variáveis) − IV | 0,62 | 0,38 | 0,23 | 0,39 |
| **Ridge direto (= T10)** | **0,37** | 0,115 | **0,019** | **0,032** |

- **Para o Cap. 6 (T10):** o F2 do T10 **só rejeita ignorando o 1º estágio** (HAC empilhado 0,019;
  só 2º nível 0,032) e **não rejeita com ele** (0,37; blocos de 30 e 90: 0,21 e 0,27; α
  reescolhido: 0,67) — ilustração direta do problema do regressor gerado (Pagan, 1984). Por
  horizonte, o max-|t| dá p ajustado de 0,115 em h = 1 (Bonferroni: 0,26; sem correção: 0,043):
  menos conservador que Bonferroni, mas nenhum horizonte abaixo de 0,05. Saídas:
  `outputs/T11/teste_conjunto_T10.csv` e `max_t_por_horizonte_T10.csv`. (C6)
- **Condicionamento:** correlação entre os β* de horizontes vizinhos de 0,8 a 0,97; número de
  condição de 245 a 840; menor autovalor ≈ 0,01 (horizontes sobrepostos). Matrizes em
  `correlacao_betas_T11.csv` e `correlacao_betas_T10.csv`.

**F1 (anotação da p. 82):** Wald de 3 coeficientes (BVRP previsto, D, D × BVRP previsto) por h. No
principal, p de 0,36 a 0,89 (corte fixo) e de 0,42 a 0,73 (expansivo). Nos quatro regressores,
**3 de 48 células abaixo de 0,0083, sem padrão** (Ridge direto em h = 60: 0,004 fixo e 0,006
expansivo; floresta direta, expansivo, h = 20: 0,006, mas 0,06 com o corte fixo). **A diferença por
regime é sugestiva e frágil**, como no T10 (seção 11.4). (C7.4)

**Como a árvore escolhe as quebras (C7.1)**

- **Forma direta:** `vh_30d` é a variável mais importante (ΔEQM por permutação 4,5), depois
  `vh_90d`; a floresta corta `vh_30d` entre 55 e 58 (mediana ponderada 56,1), com 0% da redução de
  impureza entre 35 e 40. Dependência parcial plana abaixo de ~54, caindo 1 a 3 p.p. acima de 55
  (BVRP previsto mais negativo na alta volatilidade).
- **Forma f(variáveis) − IV:** a previsão é dominada pela IV (inclinação −1 por construção) e por
  `vrp_30d`; dentro do modelo da RV, a redução de impureza se concentra em `vh_90d` (cortes em
  61–63) e `vh_60d` (~63). `vh_30d` responde por 6% da redução de impureza (usada em 23 das 33
  origens), cortada em 45–49 (mediana ponderada 48,5; 1% entre 35 e 40); o maior salto da
  dependência parcial varia de 26 a 50 entre as origens (mediana 45).
- **Para o texto do C7.1:** **as árvores cortam a volatilidade entre 45 e 58, não em 37,3**, na ponta
  de cima ou acima da faixa do corte expansivo (27 a 51), e **preferem `vh_60d` e `vh_90d`** à de 30
  dias — "a árvore vai escolher qual é a volatilidade que importa" (reunião, 1:13:02). Figuras:
  `fig_T11_cortes.png`, `fig_T11_dependencia_parcial.png`, `fig_T11_dependencia_parcial_2d.png`,
  `fig_T11_importancia.png`.

**Leitura B (floresta direto no retorno)**

| h | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| R² vs. média | 0,001 | −0,001 | 0,000 | 0,021 | −0,075 | −0,358 |
| Clark–West p | 0,13 | 0,19 | 0,20 | 0,11 | 0,20 | 0,94 |
| R² dentro da amostra (média) | 0,02 | 0,03 | 0,04 | 0,07 | 0,15 | 0,39 |

Nenhum horizonte sobrevive a Bonferroni; em h = 60 a floresta erra mais que a média (DM p = 0,014).
O R² dentro da amostra cresce com h (sobreajuste aos alvos sobrepostos), e fora da amostra vira
negativo. Placebo: medianas de −0,006 a 0,002.

## 13. A1 — estratégias de negociação do apêndice (antigo Cap. 7)

Script: `code/estrategias_A1.py`; testes: `code/test_A1.py`; saídas em `outputs/A1/`. Os scripts
antigos (`scripts/regenerate_cap7.py`, `scripts/regenerate_cap7_figs.py`, `scripts/cap7/`) e as
pastas `figs/` e `tables/` não foram alterados.

**Desenho (fixado antes dos resultados):**

| Item | Decisão |
|---|---|
| Sinal principal | BVRP previsto pelo Ridge do T5, janela expansiva (987 datas, 21/06/2023 a 03/03/2026), por coerência com o Cap. 6 |
| Robustez | proxy `vrp_30d` (conhecida em t) na amostra completa — é a regra publicada |
| Ponte | proxy avaliada nas datas do Ridge (limiar com o histórico desde 2021): separa o efeito do sinal do efeito do período |
| Limiar | q(t) = quantil q de {sinal_s : s ≤ t} (t entra, pois é conhecido no fim de t); q = 60, 70, 80 e 90% (80% de referência) |
| Burn-in | 252 observações do sinal (principal); 126 como sensibilidade no Ridge. **Com 252, a negociação efetiva com o Ridge dura 735 dias (27/02/2024 a 02/03/2026)**; com 126, 861 dias |
| Alinhamento | a posição de t rende close(t+1)/close(t) − 1; a posição de 03/03/2026 é descartada (retorno fora da amostra) |
| Direções | **as duas são reportadas**: comprado/neutro com sinal alto (regra publicada) e vendido/neutro com sinal alto, ambas com 0, 10 e 30 bps |
| Regimes | `regime_alta_fixo` do T9 (corte de 37,3, vigente em t), só a partir de 21/06/2023; a estratégia fica neutra no outro regime |
| Métricas | Sharpe **aritmético** (principal); o geométrico do publicado só nas decomposições. Drawdown a partir do valor inicial 1. Compra e manutenção sem custo; fora do regime, a estratégia fica neutra |
| Sortino (**mudança no A1**) | nas tabelas finais (principal, quantis, robustez, regimes e compra e manutenção), a definição usual: média × 365 / (√média(min(r, 0)²) × √365), com o desvio abaixo de zero calculado sobre **todos** os dias. O publicado usava o desvio-padrão só dos retornos negativos (em torno da média deles), que ignora os dias sem perda e mede a dispersão das perdas, não o tamanho delas. Nas decomposições, a fórmula antiga vale até a penúltima etapa, e a troca é a última etapa (q80: etapa 7; compra e manutenção: etapa 4) |
| Comparação | decomposição em etapas, cada uma mudando uma coisa: publicado → réplica com dados pré-T0 (`git show 11a364f^`, reproduz os 100 números das Tabs. 7.1–7.4) → dados pós-T0 → retorno simples no lugar do log-retorno (achado abaixo) → avaliação sem os dias de burn-in → ponte → Ridge → Sortino usual. A Tab. 7.1 (compra e manutenção desde 2017) também é refeita, com a decomposição pré/pós-T0 |

**Ligação com o T10 (para o texto do apêndice, A3):** o β do retorno futuro sobre o BVRP previsto é
negativo em todos os horizontes (seção 11.4; h = 1: −0,045 p.p. por p.p., p boot 0,043, sem
sobreviver a Bonferroni). BVRP previsto alto → retorno futuro menor: é isso que produz Sharpes
negativos na regra comprada/neutra, e a vendida/neutra é a sua espelho (Sharpe bruto com o sinal
trocado). **As duas direções são reportadas para não escolher a direção depois dos resultados**; a
fragilidade do β no T10 vale igualmente para qualquer ganho da vendida.

**Achado do mapeamento:** as Tabs. 7.1–7.4 publicadas são de antes do T0 (N = 1.775, com o buraco de
mar/2023) e incluíam os 252 dias de burn-in (posição zero) nas métricas da estratégia, mas não nas do
compra e manutenção "casado" (N = 1.523). As duas coisas entram como etapas da decomposição.

**Achado do passo 3 (log-retorno composto como simples):** a coluna `ret` de `vrp_with_regimes.csv`
é o **log-retorno** (`code/build_vrp_dataset.py:53`), e `scripts/regenerate_cap7.py` (e
`regenerate_cap7_figs.py`) a compunha como retorno simples — `(1 + r).prod()`, `(1 + r).cumprod()` e
custos subtraídos de r. Isso puxa para baixo o retorno anualizado e aprofunda o drawdown de todas as
séries das Tabs. 7.2–7.4 e das Figs. 7.2–7.5. A Tab. 7.1 não é afetada (usa `pct_change` de
`btc_prices.csv`). Só os scripts do Cap. 7 fazem isso (busca por `(1 + ret...` em `*.py`). O A1 usa
close(t+1)/close(t) − 1, e a troca entra como etapa 3 da decomposição (`test_A1.py` confirma que,
com expm1(ret), o código antigo fora do burn-in reproduz os 1.554 retornos novos da proxy).

### 13.1 Resultados do A1

Rodada de 01/10/2026; `test_A1.py`: 29 verificações, todas passaram. Valores sem custo, salvo
indicação; Sharpe aritmético; Sortino usual.

**Principal — Ridge, q80, 27/02/2024 a 02/03/2026 (735 dias):**

| Regra | Ret. anual | Vol. | Sharpe 0 / 10 / 30 bps | Sortino | DD máx. | Giro (op./ano) | Tempo posic. |
|---|---|---|---|---|---|---|---|
| Comprado/neutro | −9,3% | 25,5% | −0,25 / −0,35 / −0,55 | −0,36 | −28,2% | 24,8 (50 op.) | 41,0% |
| Vendido/neutro | +3,3% | 25,5% | +0,25 / +0,16 / −0,04 | +0,36 | −39,9% | 24,8 (50 op.) | 41,0% |
| Compra e manutenção | +9,4% | 49,2% | 0,43 | 0,64 | −49,5% | — | 100% |

- **A3 / T10:** a comprada tem Sharpe negativo em todos os quantis e custos; a vendida, positivo
  sem custo em todos os quantis (q60 0,14; q70 0,34; q80 0,25; q90 0,39), coerente com o β < 0 do
  T10. Mas **nenhuma das duas supera o compra e manutenção** no mesmo período (Sharpe 0,43), e a
  vendida q80 zera com 30 bps (−0,04); só q70 e q90 ficam positivas com 30 bps (0,09 e 0,05). O
  único ganho é de risco: drawdown de −28% (comprada) contra −50%, com 41% do tempo posicionado.
- **Burn-in 126 (861 dias, desde 24/10/2023):** mesmo quadro — comprada q80 −0,24; vendida +0,24
  (−0,03 com 30 bps); compra e manutenção 0,85 (o período extra é de alta).

**Robustez — proxy `vrp_30d`, q80:**

| Variante | Dias | Comprado 0 / 10 / 30 | Vendido 0 / 10 / 30 | C&M Sharpe | DD comprado / C&M |
|---|---|---|---|---|---|
| Amostra completa (30/11/2021 a 02/03/2026) | 1.554 | 0,13 / 0,03 / −0,17 | −0,13 / −0,23 / −0,44 | 0,34 | −36,4% / −72,4% |
| Ponte (datas do Ridge) | 735 | 0,02 / −0,08 / −0,26 | −0,02 / −0,11 / −0,30 | 0,43 | −27,3% / −49,5% |

- Com a proxy, a direção se inverte em relação ao Ridge (comprada levemente positiva na amostra
  completa), mas o Sharpe fica abaixo do compra e manutenção e some com custos. **Nas mesmas datas,
  a diferença Ridge × proxy (−0,25 × +0,02 na comprada) é do sinal, não do período.**

**Regimes do T9 (q80, desde 21/06/2023):**

- Ridge: alta volatilidade, 80 dias posicionados, comprada 0,20 (−0,36 com 30 bps); baixa
  volatilidade, 221 dias, comprada −0,45 e vendida +0,45 (+0,31 com 30 bps). O compra e manutenção
  no regime de baixa tem Sharpe −0,45: a vendida no regime de baixa só reflete que o BTC caiu
  nesses dias, e o sinal fica alto em quase todos eles (221 de 227).
- Proxy: **nenhum dia** com sinal alto no regime de baixa (proxy = RV − IV muito negativa quando a
  RV é baixa, mecanicamente abaixo do q80); no regime de alta, 238 dias, comprada 0,23.
- Na Tab. 7.4 antiga, só o look-ahead do `qcut` não muda o quadro (tercis pós-T0 com retorno
  simples: Sharpe geom. −0,56 / 0,24 / −0,02); a troca é de desenho (3 tercis → 2 regimes do T9).

**Decomposição (comprado/neutro, q80; Sharpe geométrico | aritmético; C&M ao lado):**

| Etapa | Sharpe geom. | Sharpe | DD máx. | C&M Sharpe geom. | C&M DD |
|---|---|---|---|---|---|
| 0. Publicado | −0,195 | — | −38,3% | −0,174 | −78,0% |
| 1. Réplica, dados pré-T0 | −0,195 | −0,081 | −38,3% | −0,174 | −78,0% |
| 2. Dados pós-T0 | −0,106 | 0,007 | −39,7% | −0,173 | −78,0% |
| 3. Retorno simples no lugar do log-retorno | 0,007 | 0,121 | −36,4% | 0,084 | −72,4% |
| 4. Sem os dias de burn-in, DD desde 1 | 0,007 | 0,131 | −36,4% | 0,084 | −72,4% |
| 5. Ponte (datas do Ridge) | −0,096 | 0,016 | −27,3% | 0,191 | −49,5% |
| 6. Ridge (principal) | −0,364 | −0,254 | −28,2% | 0,191 | −49,5% |
| 7. Sortino usual | (Sortino −0,238 → −0,363) | | | | |

- **O que mudou pelo T0:** etapa 2 (Sharpe geom. −0,195 → −0,106; 99 → 107 operações).
- **O que mudou pelo log-retorno:** etapa 3 — a maior mudança da proxy. O compra e manutenção
  "casado" do publicado (Sharpe −0,17, DD −78%) era artefato da composição do log-retorno; com
  retorno simples, Sharpe geom. 0,08 (aritmético 0,34) e DD −72%. **A frase do texto antigo de que
  a estratégia tem Sharpe "ligeiramente inferior" a um compra e manutenção negativo deixa de valer:
  o compra e manutenção é positivo e melhor que a estratégia em Sharpe.** A redução de drawdown
  (−36% × −72%) sobrevive.
- **Look-ahead:** a regra do q80 não tinha (etapas 2 → 4 só mudam a avaliação); o `qcut` da Tab. 7.4
  sim, e saiu com os regimes do T9.
- **Nova definição:** o BVRP prospectivo não pode ser sinal; o sinal passa a ser o Ridge (etapas 5 →
  6): Sharpe 0,02 → −0,25 na comprada, nas mesmas datas.

**Compra e manutenção, 17/08/2017 a 03/03/2026 (Tab. 7.1):** o T0 quase não mexe. Pré-T0 → pós-T0:
ret. anual 38,7% → 38,3%; vol. 68,7% → 68,3%; Sharpe geom. 0,563 → 0,560 (aritmético 0,82); DD
−83,2% nos dois; Sortino antigo 1,11 → 1,10, usual 1,20. O maior retorno diário era o de 32 dias
(23,0% em 01/04/2023); depois do T0, é 22,5% (07/12/2017). Num histórico de 3.120 dias, um retorno
de 32 dias pesa pouco; o efeito do T0 aparece nas séries curtas do BVRP, não aqui.

**Para o texto do apêndice (A2–A4):** análise exploratória, sem alfa: nenhuma regra supera o compra e
manutenção em Sharpe em nenhum período testado; a direção que "funciona" depende do sinal (Ridge:
vendida; proxy: comprada, fraca) e desaparece com 30 bps na q80; o que se mantém é a menor exposição
(drawdown menor por ficar fora 60–75% do tempo), não informação sobre a direção dos retornos.

### 13.2 Varredura: log-retorno composto como simples fora do Cap. 7 (01/10/2026)

Busca em `code/`, `scripts/` (`*.py`) e nos notebooks versionados por composição de retornos
(`cumprod`, `.prod(`, `(1 + ...ret`) e por todo leitor da coluna `ret` (log-retorno de
`build_vrp_dataset.py:53`). **Só o Cap. 7 antigo compõe `ret` como retorno simples.**

| Onde | Retorno usado | Composição? | Situação |
|---|---|---|---|
| `scripts/regenerate_cap7.py`, `scripts/regenerate_cap7_figs.py`, `chapters/Cap7_Estratégia.ipynb` | `ret` (log) | sim, `(1 + r)` | **erro**; substituídos pelo A1 (scripts antigos intactos) |
| `code/retorno_bvrp_T10.py`, `code/retorno_bvrp_T11.py` (e `regenerate_cap5.py`, `plot_vrp_vs_return_T12.py`, `analyze_magnitude_bvrp.py`) | `ret_fut_h` = close(t+h)/close(t) − 1, **simples** (`build_targets.py:28-33`) | não; regressando de MQO/árvores | correto |
| `code/estrategias_A1.py` | close(t+1)/close(t) − 1, simples | sim, `(1 + r)` | correto (a réplica usa `ret` só para reproduzir o publicado e para a etapa 3) |
| `code/build_ml_dataset_T4.py` | log de `close` | `ret_acum_5d/30d` = soma de log-retornos | correto (log-retorno acumulado) |
| `code/build_ml_dataset.py` (legado) | `ret` (log) | `ret_lag_5/20` = soma | correto |
| `code/analyze_bvrp_by_return_sign.py` | `ret` (log) | não; só o sinal | correto (o sinal não muda) |
| `code/build_desc_stats_T1.py:204` | `ret` (log) | não; estatísticas descritivas | correto, mas **a Tab. 3.1 descreve o log-retorno**: o texto deve dizer "log-retorno diário" (C3) |

## 14. Convenção de notação: VH e RV (Fase 2, C2.4/C2.5, 06/10/2026)

Decisão do usuário na reescrita do Cap. 2. **Vale para todos os capítulos, tabelas, figuras,
legendas e notas** (inclusive os rótulos gravados dentro dos PNGs).

| Símbolo | Significado | Onde aparece |
|---|---|---|
| **VH** (volatilidade histórica) | medida empírica: raiz de (365/h) × soma dos log-retornos diários ao quadrado numa janela de h dias, sem subtrair a média | todo uso empírico, nas duas janelas: retrospectiva, `VH_{t−29:t}` (`vh_30d` em t), e prospectiva, `VH_{t+1:t+30}` (`vh_30d` em t+30) |
| **RV** (volatilidade realizada) | só a grandeza teórica da definição do prêmio, VRP_t = E(RV_{t+1:t+h} \| I_t) − E*(RV_{t+1:t+h} \| I_t) | Cap. 2, Seção "Prêmio de risco de volatilidade", e onde a definição for retomada (C4.5) |
| **IV** | volatilidade implícita, IV_t = E*(RV_{t+1:t+h} \| I_t), medida pelo DVOL | todo o texto |
| **BVRP** | alvo prospectivo, `VH_{t+1:t+30} − IV_t` (coluna `bvrp_30d_fut`) | todo o texto |
| **BVRP^proxy** | proxy retrospectiva, `VH_{t−29:t} − IV_t` (coluna `vrp_30d`), sob passeio aleatório sem deriva; aproxima a expectativa, não a realização | todo o texto |

- O texto diz, uma vez (Cap. 2, Seção "Volatilidade histórica"), que a VH é o estimador da RV com
  dados diários.
- "Volatilidade realizada" fica reservada à RV teórica e à literatura de alta frequência
  (Andersen et al., 2003). Na varredura F7, toda "volatilidade realizada" que descreve a medida
  empírica vira "volatilidade histórica", e todo `RV`/`RV_{30,t}` empírico vira `VH`.
- Os nomes de colunas no código (`vh_30d`, `rv_30d_fut`, `vrp_30d`) não mudam; `rv_30d_fut`
  é a VH prospectiva (ver seção 12).

## 15. Referências pendentes para os próximos capítulos (Fase 2, 06/10/2026)

Registradas na reescrita do Cap. 2. Os metadados das entradas novas foram conferidos pelo usuário.

| Item | Ação | Tarefa |
|---|---|---|
| `bollerslev2014international` (Bollerslev, Marrone, Xu e Zhou, 2014, *JFQA* 49(3)) | Entrou no .bib e no Cap. 2 (réplica internacional da previsibilidade). **Citar também no Cap. 6** (atual Cap. 5): trata de vieses de amostra finita em regressões com retornos sobrepostos | **C6** |
| `bollerslev2011VRP` | Entrada corrigida para Bollerslev, Gibson e Zhou (2011), *J. Econometrics* 160(1), 235–245 (chave mantida). No Cap. 2, a previsibilidade no S&P 500 passou a citar `bollerslev2009expected` | feito (Cap. 2) |
| `cap8_regimes.tex:246`, efeito alavancagem | A frase cita `bollerslev2011VRP` e `bali2009volatilitypremium`, que não tratam do efeito. **Precisa de outra referência** (Black, 1976; Christie, 1982), **fora do .bib**: perguntar antes de acrescentar | **C7** |

## 16. Cap. 3 (Fase 2, 06/10/2026)

| Item | Registro | Tarefa |
|---|---|---|
| **C3.4 pode ser duplicata do C3.2** | O plano cita "p. 45, R 56:46" para o C3.4 ("gráfico que compara grandezas diferentes, *misleading*"), mas nesse trecho a reunião trata da tabela do H1 (*"essa tabela aí não serve para nada"*). O único *misleading* da reunião (~45:40–45:43) é o boxplot de 2026, ano incompleto, que já é o C3.2. **Aplicado como C3.4**, por decisão do usuário: a legenda do painel (a) da série temporal diz que só a VH prospectiva cobre a mesma janela da IV de t e que a retrospectiva é a da proxy. Confirmar com o Prof. Marcelo | **C3.4** → reunião |
| Data de 2026 nas legendas | O plano diz "dados até julho/agosto"; a amostra termina em **03/03/2026**. Legendas e eixo usam a data real (1º/01 a 03/03/2026, N = 62). O primeiro ano também é incompleto (24/03 a 31/12/2021, N = 283) e ganhou a mesma marcação | **C3.2** |
| Geradores do Cap. 3 | `build_desc_stats_T1.py`, `plot_cap3_T1.py`, `plot_vrp_vs_return_T12.py` e `regimes_T9.py` ganharam `--publicar-cap3` (cópia para `tables/cap3/` e `figs/cap3/` com os nomes do .tex) e rótulos na notação da seção 14. Os cálculos não mudaram | **C3** |
| Citações do H1 | `newey1987simple` (convenção de pesos de Bartlett, h+1 defasagens) e `newey1994automatic` (regra automática do statsmodels, L = 7), conforme a anotação da p. 45 (*"fazer uma referência mais precisa à regra"*). Usar as mesmas chaves no C4.4 | **C3.3**, C4.4 |
