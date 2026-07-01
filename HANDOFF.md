# Estado da revisão — retomada

_Última atualização: 01/07/2026, sessão de revisão Cap. 5/Cap. 6 — Cap. 6 fechado e verificado visualmente._

## Estado atual

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

1. **Conferir visualmente a tabela ADF/KPSS do Cap. 3** (`tables/cap3/adf_kpss_table.tex`,
   trazida pelo merge v3.1) — único arquivo do merge ainda não visto renderizado no PDF,
   mesmo padrão de escrutínio que o Cap. 6 recebeu (não só "compilou sem erro").

2. **Cap. 5**, pendências que apareceram no caminho:
   - Três entradas de bibliografia soltas em texto puro no capítulo de bootstrap
     (Künsch 1989, Politis & Romano 1994, Stambaugh 1999) — precisam virar `\cite`
     de verdade no `references.bib`, mesmo tratamento que já foi dado a
     Ferson-Sarkissian-Simin (2003).
   - MBB (Moving Block Bootstrap) ainda pendente de rodar formalmente na regressão de
     magnitude (`|ret_fut_30d| ~ BVRP`) para fechar a robustez daquele achado.

3. **Pós-defesa (não bloqueante):** dívida de nomenclatura `cap5_`/`cap8`/`tab8` dentro
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
