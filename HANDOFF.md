# Estado da revisão — retomada

_Última atualização: 01/07/2026, sessão de revisão Cap. 3/5/6 — Cap. 3 e Cap. 6 fechados e verificados visualmente._

## Estado atual

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

## Pendente, nesta ordem

1. **Cap. 4 — verificar "maxlags = h−1" na Seção 4.4.2 (linha ~174-175).** Confere
   contra o resto do capítulo e contra o script (que usa `maxlags=h`, não `h-1`) —
   último ponto aberto da revisão dos quatro capítulos desta leva (3/4/5/6).

2. **Conferir visualmente a tabela ADF/KPSS do Cap. 3** (`tables/cap3/adf_kpss_table.tex`,
   trazida pelo merge v3.1) — único arquivo do merge ainda não visto renderizado no PDF,
   mesmo padrão de escrutínio que o Cap. 6 e o Cap. 3 já receberam (não só "compilou
   sem erro").

3. **Cap. 5**, pendências que apareceram no caminho:
   - Três entradas de bibliografia soltas em texto puro no capítulo de bootstrap
     (Künsch 1989, Politis & Romano 1994, Stambaugh 1999) — precisam virar `\cite`
     de verdade no `references.bib`, mesmo tratamento que já foi dado a
     Ferson-Sarkissian-Simin (2003).
   - MBB (Moving Block Bootstrap) ainda pendente de rodar formalmente na regressão de
     magnitude (`|ret_fut_30d| ~ BVRP`) para fechar a robustez daquele achado.

4. **Pós-defesa (não bloqueante):** dívida de nomenclatura `cap5_`/`cap8`/`tab8` dentro
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
