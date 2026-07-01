# Estado da revisão — retomada

_Última atualização: 01/07/2026, sessão de revisão Cap. 5/Cap. 6 — pós-merge com overleaf/master._

## Estado atual

**O merge com `overleaf/master` (Plano v3.1) está feito e fechado.** Commit `8327188`
no `main` local é o merge commit (dois pais); `git rev-list --left-right --count
main...overleaf/master` dá `7 0` — o `0` do lado direito confirma que todo o v3.1 foi
absorvido, nada dele ficou pra trás. O `7` do lado esquerdo é esperado (os 6 commits
locais anteriores + o próprio commit de merge, que só existe em `main` por definição).

O `cap6_ml_unificado.tex` resultante tem a costura das duas dimensões: a reescrita
forecasting completa (script + tabelas + figuras + prosa, N=1.523/457, persistência
R²=0,9365, regime≈12,4 p.p.) como base, com a terminologia e a fusão de subseções do
v3.1 transplantadas por cima. No caminho, corrigiu-se um bug que o merge automático
introduziu silenciosamente (sem gerar conflito): o commit `c7fd24d` do v3.1 fez um
find-replace cego de "Random Forest"→"floresta aleatória" que duplicou o termo dentro
de um parêntese que já o traduzia — "floresta aleatória (floresta aleatória)". Corrigido.

**Ainda não compilado em lugar nenhum.** Esse é o próximo passo, e o único que falta.

Backups de segurança, congelados antes do merge (permanecem válidos, o merge não os
tocou):
- `backup-main-1782881323` / `backup-main-1782881189` (estado do `main` local pré-merge)
- `backup-overleaf-1782881323` / `backup-overleaf-1782881189` (estado do `overleaf/master`
  pré-merge)

## Pendente, nesta ordem

1. **Compilar no Overleaf.** Duas camadas de verificação, não uma:

   - **Camada 1 — Cap. 6 (o que foi reescrito):**
     - Três refs resolvem sem `??`: `eq:cap5_persist`, `tab:cap8_oos_performance`,
       `tab:cap8_coeficientes`.
     - Três figuras renderizam de verdade (o `\IfFileExists` falha silencioso).
     - Números batem o `\input`: N=1.523/457, persistência R²=0,9365, lineares≈0,39,
       árvores 0,42–0,44, regime≈12,4 p.p. (LASSO e Ridge).
     - Sem `Overfull \hbox` grave na `tab8_coeficientes` (ganhou coluna mais longa).

   - **Camada 2 — os 10 arquivos que vieram do v3.1 via merge "auto-resolvido":**
     `Nota_metodologica.tex`, `apendiceA_variaveis.tex`, `cap2_referencial.tex`,
     `cap3_dados.tex`, `cap4_metodologia.tex`, `cap8_regimes.tex`,
     `tables/cap3/adf_kpss_table.tex` (novo), `tables/cap3/desc_stats_cap3.tex`,
     `tables/cap7/tab7_1_perf_buy_hold.tex`, `tables/tab7/tab7_2_perf_vrp_quantile.tex`.
     "Auto-merged sem CONFLICT" não é garantia de conteúdo correto — foi assim que o
     bug "floresta aleatória (floresta aleatória)" passou batido no Cap. 6. Esses 10
     arquivos entram no PDF pela primeira vez desde o merge; olho neles também, não só
     nos três da Camada 1.

2. **Só depois de compilar limpo, os dois pushes, nesta ordem:**
   - `git push overleaf main:master` primeiro (leva a costura pro Overleaf; deve ser
     fast-forward do lado do Overleaf, já que o merge absorveu o v3.1 inteiro — se o
     Git reclamar de non-fast-forward ou pedir `--force`, **parar**, algo mudou no
     Overleaf depois do último fetch, investigar antes, nunca `--force` sem confirmar
     o que seria apagado).
   - `git push origin main` depois (GitHub, segundo lugar físico do trabalho).
   - Se a compilação revelar problema: consertar local, recommitar, só então pushar.
     O remote nunca deve ver estado quebrado.

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
