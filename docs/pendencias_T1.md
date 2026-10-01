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
| 2 | `scripts/regenerate_cap7.py:306` (e `regenerate_cap7_figs.py:72`) | `pd.qcut` na amostra inteira para o regime de RV da estratégia: look-ahead mesmo na definição antiga | E4 | A1: sinal por quantis em janela expansiva, sem look-ahead | **A1** |
| 3 | `scripts/regenerate_cap6_scatter.py:31` | `close` (I(1)) como feature | E3: eliminar `close` | T4: **transformar** variáveis não estacionárias (ex.: diferenças) em vez de eliminá-las | **T4** |
| 4 | `scripts/regenerate_cap7.py:180` | custos de 0/5/10 bps | E6: 10/30 bps | A1: "manter os custos de transação" | **A1** (confirmar se os cenários mudam) |
| 5 | `scripts/regenerate_cap7.py:100`, `scripts/regenerate_cap7_figs.py:64`, `code/plot_vrp_timeseries.py:23` | data final da amostra fixa em `"2026-03-03"` (continua correta depois do T0, mas é frágil) | Decisão 2 | — | sem tarefa própria; tratar quando o script for tocado (A1, T1) |
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
