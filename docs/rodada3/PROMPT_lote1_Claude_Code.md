Você vai executar o LOTE 1 da Rodada 3 de revisão da dissertação (BVRP, FGV EESP). Comunique-se sempre em português do Brasil.

## Contexto
O Prof. Marcelo Fernandes anotou o PDF v3_3_2 e mandou um e-mail. Todos os pontos foram extraídos e classificados em blocos. Leia ANTES de qualquer coisa, nesta ordem:
1. `docs/rodada3/Plano_Rodada3_Anotacoes_Marcelo_v3_3_2.md` (plano, decisões pendentes e inventário por bloco)
2. `docs/rodada3/Anotacoes_Marcelo_v3_3_2_completo.csv` (161 pontos + EM1–EM4; colunas: id, pág. do PDF, trecho marcado, comentário do Marcelo, ação proposta, bloco)
3. `docs/pendencias_T1.md` (histórico das rodadas anteriores; só consulta)
O ramo atual deve ser `rodada2-mcs`. Há alterações já feitas e ainda sem commit em `frontmatter/folha_aprovacao.tex` e `frontmatter/folha_rosto.tex` (troca da banca; os `.bak` ficam). Não mexa nelas e não as descarte.

## Regras obrigatórias
1. NUNCA altere um arquivo sem antes me mostrar exatamente o que vai mudar (trecho antes × depois, com arquivo e linha) e esperar minha confirmação.
2. Faça backup `.bak` de cada arquivo antes de modificá-lo.
3. Uma tarefa por vez, na ordem abaixo. Trabalhe por capítulo, em levas de no máximo 8 itens por confirmação.
4. Não faça commit, não crie ramo, não faça push sem eu pedir.
5. Terminologia: "aprendizado de máquina", "floresta aleatória", "MQO", "fora da amostra", "formadores de mercado", "modelo de referência", "neutra ao risco". Tratamento no texto: ver o plano (primeira pessoa fica para o bloco B8, não faça agora).
6. Se algo for ambíguo, PERGUNTE antes de decidir.
7. Os números da dissertação não podem mudar neste lote. Se qualquer edição mexer em número, tabela gerada por script ou figura, pare e me avise.

## Escopo do Lote 1 (mecânico, sem decisão pendente)
Os números abaixo são os IDs (#) do CSV, na numeração da v3_3_2.

**B6, LaTeX (4 itens):** #14 (`\noindent`), #37 (`\mathbb{E}^*`), #40 (`\usepackage{amssymb}` no preâmbulo), #53 (`\mathbb{E}`). Procure todas as ocorrências equivalentes em todo o texto, não só a marcada, e me liste antes.

**B7, referências (4 itens):**
- #68: ordenar a citação em ordem cronológica e, no mesmo ano, alfabética: (Pan, 2002; Carr e Wu, 2009; Bollerslev et al., 2009, 2011). ANTES, verifique no `.bib` se os autores de Bollerslev et al. 2011 e 2019 são os mesmos; se não forem, me avise e não aplique.
- #82: incluir no `.bib` Cobb, Koppstein e Chen (1983), "Estimation and moment recursion relations for multimodal distributions of the exponential family", Journal of the American Statistical Association 78, 124–130. Só a entrada; a frase sobre moda × antimoda eu decido depois.
- #160 e #161: atualizar Almeida, Grith, Miftachov e Wang para "Option-Implied Risk Premia and Cryptocurrency Market Regimes", Journal of Business and Economic Statistics, forthcoming (2026+). Confira os metadados (título, autores, periódico) em fonte oficial antes de me propor a entrada; se não conseguir confirmar, diga.

**B5, cortes e trocas curtas (35 itens):**
#4, #5, #6, #7, #8, #11, #35, #36, #38, #41, #42, #43, #44, #45, #55, #56, #63, #74, #81, #83, #86, #94, #96, #100, #107, #108, #119, #120, #123, #127, #143, #144, #152, #153, #157.
Para cada um: localize o trecho no `.tex` buscando um pedaço curto e distintivo do campo "trecho" do CSV (o texto extraído do PDF tem hifenização e quebras de linha estranhas, então não busque a frase inteira). Mostre o PARÁGRAFO inteiro antes × depois, não só o fragmento: vários cortes são de pedaços de frase (por exemplo, "(Implied Volatility," e "o,") e podem deixar parênteses ou pontuação soltos. Cortes de referência cruzada: confirme que nenhum `\ref` ou `\label` fica órfão.
Se um item estiver ligado a texto que o Marcelo também pediu para reescrever (decisões D1 a D5), pule e me diga.

## FORA do Lote 1 (não toque)
- Decisões pendentes: D1 (definição e notação do BVRP, VH × RV, RBVRP), D2 (HAR), D3 (regressor gerado), D4 (fusão de capítulos), D5 (previsão média e profundidade das árvores).
- O bootstrap da atenuação (#145–#150, EM3) está EM ESPERA, aguardando resposta do Marcelo. Não altere `code/retorno_bvrp_T10.py` nem o Cap. 6.
- Blocos B1 a B4, B8 e B9, e qualquer script em `code/`.
- O Fase0 (`Revisões da Dissertação/Dissertação Completa/Fase0_Decisoes_Para_Claude_Desktop.md`) é de 12/05/2026 e está superado em pontos como a definição retrospectiva do BVRP, a amostra de 1.686 observações e os 7 capítulos. Não aplique essas regras neste lote, porque não vamos mexer em definições nem em scripts. Se achar conflito, me avise.

## Como começar
1. Confirme em uma frase que leu os três arquivos e diga em qual ramo estamos e quais arquivos têm alterações sem commit.
2. Faça uma tabela de mapeamento dos 43 itens: ID, arquivo `.tex` e linha onde está o trecho, ação. Marque os que não achou ou deixou de fora. Mostre a tabela e espere minha confirmação.
3. Depois, execute por leva (capítulo a capítulo, até 8 itens), sempre com antes × depois e confirmação.
4. Ao fim, rode `scripts/verificar_latex.py` e me diga o resultado. Se o pacote `brazilian` do babel estiver disponível na sua máquina, compile e confira as páginas alteradas.
5. No fim, me entregue um resumo: itens aplicados, pulados (com o motivo) e arquivos com `.bak`.
