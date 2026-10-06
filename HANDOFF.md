> Registro histórico. Os N e definições aqui foram superados na Fase 1 da revisão (out/2026): ver docs/pendencias_T1.md e a memória do projeto (project_bvrp_prospectivo.md).

# Estado da revisão — retomada

_Última atualização: 01/07/2026, sessão de revisão Cap. 3/4/5/6 — os quatro capítulos revisados, corrigidos e sincronizados._

## Estado atual

**Os quatro capítulos da leva (3, 4, 5, 6) estão revisados, corrigidos e sincronizados.**
Cap. 3 e Cap. 6 têm verificação visual completa no PDF compilado; Cap. 4 e Cap. 5
tiveram correções pontuais de texto (baixo risco, verificação visual ainda pendente,
mas mudança é texto puro, não lógica/dado). `local`, `overleaf/master` e `origin/main`
sincronizados em `24a5cc3`.

**O Cap. 5 está FECHADO nesta leva.** Duas partes:
- **Parte B (verificação prévia):** as três tabelas do capítulo (`tab_ols_basico_multihoriz.tex`,
  `tab_ols_rv_iv_multihoriz.tex`, `tab_bootstrap_ci.tex`) foram confirmadas **frescas**,
  não fósseis — reexecutei `regenerate_cap5.py` e `git diff` contra as tabelas commitadas
  veio vazio. Diferente do que se encontrou no Cap. 3, aqui não havia descompasso
  script↔tabela.
- **Quick win do bootstrap (commit `658c3f0`):** três referências citadas em prosa solta
  (Künsch 1989, Politis \& Romano 1994, Stambaugh 1999) nunca tinham entrada no
  `references.bib` nem `\citet`/`\citep` de verdade — adicionadas as três entradas e
  vinculado o texto com `natbib` (`\citet{kunsch1989}`, `\citet{politisromano1994}`,
  `\citep{stambaugh1999}`).
- **Correções de redação (commit `24a5cc3`):** Seção 5.3 tinha uma imprecisão --- dizia
  que $h\in\{5,30\}$ tinham "o mesmo sinal negativo" quando na verdade os sinais são
  opostos nesses horizontes (só a magnitude deixa de cancelar, diferente de
  $h\in\{10,20\}$ onde os sinais são de fato iguais); Seção 5.5 alinhada à formulação
  do prêmio de seguro padronizada no Cap. 3 (nomeando vendedores e compradores de
  volatilidade, não só "pago pelos compradores").

**O Cap. 4 recebeu uma correção pontual (commit `75d5177`):** a Seção "Sobreposição
temporal" afirmava `maxlags = h-1` para o HAC/Newey-West, contradizendo o resto do
capítulo (que já dizia `h`), o Cap. 5 inteiro (3 menções, todas `h`) e o script real
(`regenerate_cap5.py`, `regenerate_cap9.py` — ambos usam `maxlags=h`). Corrigido para
`h`, com referência cruzada à seção de inferência do capítulo.

**O Cap. 3 está FECHADO.** Revisado, corrigido e verificado visualmente no PDF
compilado. Era o capítulo mais traiçoeiro da leva: quatro figuras eram fósseis no
sinal antigo (IV−RV, de antes da "Decisão 1 — Fase 0" que corrigiu o cálculo para
RV−IV), um script (`plot_vrp_vs_return.py`) salvava em `figuras/` em vez de
`figs/cap3/` (a figura nunca chegava ao PDF mesmo regenerada), e a amostra do
histograma/boxplot (`vrp_30d_dataset.csv`, N=1.835) divergia da amostra canônica
das tabelas (`vrp_with_targets.csv`, N=1.775) por 60 dias — cuja remoção **inverteu**
uma conclusão do texto: 2026 deixou de ser "o ano mais comprimido" e passou a ser
**o de maior dispersão de toda a série** (IQR=40,11), porque os 60 dias removidos
(abr–mai/2026) diluíam um pico de estresse agudo do BVRP em fev–mar/2026 (visível
na Fig. 3.1, dentro da amostra canônica). Corrigido: sinal RV−IV nas 4 figuras,
bug de path no script, amostra alinhada a N=1.775 em tudo, texto das Seções 3.3/3.4
reescrito com os números reais (assimetria +0,61, curtose +0,91, modo ≈−9, 2026
como outlier de dispersão com nota de período parcial N=62), e a Seção 3.5/Fig. 3.4
corrigida de "correlação +0,20 motivando H1" para "+0,013, estatisticamente nula,
antecipação do null do Cap. 5". Commits: `8510b2b`, `bbdfa12`, `c277583`, `1e5ab4a`,
`a03e65a`. Sincronizado: `local`/`overleaf/master`/`origin/main` todos em `a03e65a`.

**Nota conceitual não-bloqueante:** o pico de estresse de fev–mar/2026 que aparece
no Cap. 3 está dentro da amostra usada nas regressões do Cap. 5 e no período de
teste OOS do Cap. 6 (dez/2024–mar/2026 termina exatamente nesse pico). Não é erro —
é característica real da amostra —, mas vale ter consciência de que o fim da janela
amostral contém um evento extremo, caso a banca pergunte sobre sensibilidade dos
resultados ao fim da amostra.

**Pendência de dados não-bloqueante:** `vrp_30d_dataset.csv` está 60 dias à frente
de `vrp_with_targets.csv` (a fonte canônica). Não é bug — é atualização assíncrona
entre os dois pipelines —, mas as duas fontes deveriam ser reconciliadas em algum
momento para não reabrir essa divergência em uma próxima regeneração.

**O Cap. 6 está FECHADO.** Merge com `overleaf/master` (Plano v3.1) feito (commit
`8327188`), correção residual de N=458→457 na legenda da Fig. 6.1 feita (commit
`8a14aa6`), e **compilação confirmada no Overleaf: 0 erros**, apenas 1 warning
inofensivo de `\showhyphens`. Verificação visual (não só log) confirmou: as três
figuras do Cap. 6 renderizam como imagens reais (não caíram na caixa `\IfFileExists`),
e os números batem entre texto/tabela/figura/script — N=1.523/457, persistência
R²=0,9365, lineares≈0,39, árvores 0,42–0,44, regime≈12,4 p.p.

`local` (main, commit `8a14aa6`), `overleaf/master` e `origin/main` (GitHub) estão
todos sincronizados no mesmo commit. Três lugares físicos, nenhuma divergência.

O `cap6_ml_unificado.tex` final tem a costura das duas dimensões: a reescrita
forecasting completa (script + tabelas + figuras + prosa) como base, com a
terminologia e a fusão de subseções do v3.1 (Plano v3.1) transplantadas por cima.
No caminho, dois bugs foram corrigidos: (1) o merge automático introduziu em
silêncio — sem gerar conflito — a redundância "floresta aleatória (floresta
aleatória)", resultado de um find-replace cego do commit `c7fd24d` do v3.1;
corrigido. (2) A legenda da Fig. 6.1 ficou com N=458 residual depois da correção
do `shift(-1)` para forecasting genuíno (que reduziu o teste de 458 para 457
observações); corrigido no commit `8a14aa6`.

**Verificação visual cobriu só o Cap. 6.** Os outros 10 arquivos que vieram do v3.1
via merge "auto-resolvido" (`Nota_metodologica.tex`, `apendiceA_variaveis.tex`,
`cap2_referencial.tex`, `cap3_dados.tex`, `cap4_metodologia.tex`, `cap8_regimes.tex`,
`tables/cap3/adf_kpss_table.tex`, `tables/cap3/desc_stats_cap3.tex`,
`tables/cap7/tab7_1_perf_buy_hold.tex`, `tables/tab7/tab7_2_perf_vrp_quantile.tex`)
**compilaram sem erro** (fazem parte do mesmo 0-erros do PDF), mas **não foram
conferidos visualmente um a um** — só o Cap. 6 recebeu esse escrutínio. Em especial,
a tabela ADF/KPSS nova do Cap. 3 (`adf_kpss_table.tex`) nunca foi vista renderizada;
"compilou sem erro" não é o mesmo padrão de verificação que os quatro pontos do Cap. 6
receberam (essa distinção é a lição central desta sessão — "auto-merged sem
CONFLICT" ou "compila sem erro" não são garantia de conteúdo correto, só de sintaxe
válida).

Backups de segurança, congelados antes do merge (permanecem válidos):
- `backup-main-1782881323` / `backup-main-1782881189` (estado do `main` local pré-merge)
- `backup-overleaf-1782881323` / `backup-overleaf-1782881189` (estado do `overleaf/master`
  pré-merge)

## Pendente — tudo não-bloqueante, nesta ordem de relevância

1. **CORREÇÃO a uma entrada anterior deste arquivo: o MBB da magnitude não está
   pendente — já foi feito.** Uma versão anterior deste HANDOFF listava "MBB ainda
   pendente de rodar" como item de maior prioridade; isso estava desatualizado. O
   teste de magnitude (`|ret_fut_30d| ~ BVRP`) mora no **Cap. 8** (`cap8_regimes.tex`,
   seção "Apêndice do capítulo: robustez do teste de magnitude",
   `sec:cap8-apendice-magnitude`), não no Cap. 5, e a tabela já publicada ali reporta
   **quatro** métodos, não só HAC:
   ```
   Newey--West (HAC)        p=0,0027   Sobreposta (N=1.775)
   Hansen--Hodrick           p=0,0054   Sobreposta (N=1.775)
   Moving Block Bootstrap    p=0,0120   Sobreposta (N=1.775)
   Não sobreposta (mediana)  p=0,1048   30 fases, N≈59 cada
   ```
   Script: `code/analyze_magnitude_bvrp.py` (roda HAC + Breusch-Pagan + MBB +
   decomposição RV/IV; a decomposição mostra `beta_RV=-0,0016` (p=0,0071) e
   `beta_IV=+0,0018` (p=0,0065), simetria=6,0% — BVRP capta a combinação, não RV
   ou IV isolados). Reexecutado nesta sessão para confirmar: números idênticos aos
   já publicados no texto, nada mudou. O texto já é honesto sobre a fragilidade
   ("significância estatística é marginal sob a correção mais conservadora...
   evidência sugestiva, não achado confirmatório"). **Não há código pendente aqui.**
   Se sobra algo, é só revisitar se o posicionamento (apêndice de capítulo, não
   corpo principal) continua sendo o desejado — não é bloqueante.

2. **Nota pré-defesa:** conferir volume/número de páginas das três referências do
   bootstrap (Künsch 1989, Politis \& Romano 1994, Stambaugh 1999) contra o DOI/fonte
   original — foram digitadas de memória/conhecimento geral, não copiadas de uma fonte
   primária conferida.

3. **Reconciliação de datasets:** `vrp_30d_dataset.csv` está 60 dias à frente de
   `vrp_with_targets.csv` (a fonte canônica). Não é bug agora (já tratado no Cap. 3
   truncando para a amostra comum), mas as duas fontes deveriam convergir num único
   pipeline de atualização para não reabrir a divergência numa próxima regeneração.

4. **Verificação visual ainda não feita (baixo risco, mudança foi só texto):**
   - Cap. 4 e Cap. 5: as correções desta leva (`75d5177`, `658c3f0`, `24a5cc3`) ainda
     não foram vistas renderizadas no PDF — mudança é texto puro (sem tocar em dado,
     script ou tabela), risco mínimo, mas confirma antes de considerar 100% fechado.
   - Tabela ADF/KPSS nova do Cap. 3 (`tables/cap3/adf_kpss_table.tex`, trazida pelo
     merge v3.1) — nunca foi vista renderizada; "compilou sem erro" não é o mesmo
     padrão de verificação que os outros itens já receberam.

5. **Pós-defesa (não bloqueante):** dívida de nomenclatura `cap5_`/`cap8`/`tab8` dentro
   do arquivo do Cap. 6 — strings funcionam, mas confunde quem procurar por `cap6_`
   intuitivamente.

## O que este ciclo consertou (contexto, não é mais problema)

Cap. 6 entrou nesta sessão com: `.tex` alegando forecasting enquanto o código fazia
nowcasting (alvo não deslocado), benchmark de persistência citado no texto mas
inexistente em qualquer script, tabela de benchmarks fabricada à mão dentro do
capítulo, N e split hardcoded descrevendo um experimento diferente do que o código
rodava, três datasets/capítulos fósseis (`ml_dataset.csv` duplicado em duas pastas,
`cap6_fusao.tex` órfão) e regime de volatilidade com risco de look-ahead (já
corrigido antes desta sessão via `.expanding()`, confirmado). Sai com pipeline
forecasting genuíno, todo número gerado por script versionado, e nenhuma tabela
digitada à mão.
