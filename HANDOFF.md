# Estado da revisão — retomada

_Última atualização: 01/07/2026, sessão de revisão Cap. 5/Cap. 6._

## Estado atual

Commit `935c871` no `main` local tem o Cap. 6 com pipeline de **forecasting completo**
(script + tabelas + figuras + prosa, coerentes entre si) — **mas nunca foi compilado
em lugar nenhum.**

Backups de segurança, congelados antes de qualquer merge:
- `backup-main-1782881323` (estado do `main` local)
- `backup-overleaf-1782881323` (estado do `overleaf/master` no momento do backup)

## Pendente, nesta ordem

1. **Merge com `overleaf/master`.** `git rev-list --left-right --count main...overleaf/master`
   dá `4 15` — 4 commits só no `main` local, 15 só no `overleaf/master`. São ortogonais
   em conteúdo:
   - Local (`main`): números e narrativa do Cap. 5 (bootstrap, órfãos) e Cap. 6
     (forecasting, benchmark de persistência, prosa reescrita).
   - Overleaf (`overleaf/master`, "Plano v3.1"): estrutura e nomenclatura — fusão de
     subseções do Cap. 6 (`Mu1`), correção "Random Forest→floresta aleatória",
     correções pontuais em Cap. 2/3/4/7, apêndice A, nota metodológica.

   `cap6_ml_unificado.tex` foi tocado pelos dois lados e vai conflitar linha a linha,
   mas as duas versões são combináveis (nenhuma invalida a outra). A resolução não é
   escolher um lado — é costurar a prosa numérica nova dentro da estrutura de seções
   fundida pelo v3.1. Atenção: a fusão de subseções pode deslocar onde cada bloco de
   prosa cai; não é um merge puramente mecânico.

2. **Compilar no Overleaf** (só depois do merge resolvido).

3. **Conferir no PDF, com o próprio olho, não só no log:**
   - Três refs resolvem sem `??`: `eq:cap5_persist`, `tab:cap8_oos_performance`,
     `tab:cap8_coeficientes`.
   - Três figuras renderizam de verdade (o `\IfFileExists` falha silencioso).
   - Números batem o `\input`: N=1.523/457, persistência R²=0,9365, lineares≈0,39,
     árvores 0,42–0,44, regime≈12,4 p.p. (LASSO e Ridge).
   - Sem `Overfull \hbox` grave na `tab8_coeficientes` (ganhou coluna mais longa).

   Só depois desses quatro pontos baterem o Cap. 6 fecha de verdade.

## Mais pra frente (não bloqueante)

- Dívida de nomenclatura `cap5_`/`cap8`/`tab8` dentro do arquivo do Cap. 6 — strings
  funcionam, mas confunde quem procurar por `cap6_` intuitivamente. Limpeza pós-defesa.
- **Cap. 5**, pendências que apareceram no caminho:
  - Três entradas de bibliografia soltas em texto puro no capítulo de bootstrap
    (Künsch 1989, Politis & Romano 1994, Stambaugh 1999) — precisam virar `\cite`
    de verdade no `references.bib`, mesmo tratamento que já foi dado a
    Ferson-Sarkissian-Simin (2003).
  - MBB (Moving Block Bootstrap) ainda pendente de rodar formalmente na regressão de
    magnitude (`|ret_fut_30d| ~ BVRP`) para fechar a robustez daquele achado.

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
