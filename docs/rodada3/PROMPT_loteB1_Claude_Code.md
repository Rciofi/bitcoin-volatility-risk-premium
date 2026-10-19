Você vai executar o LOTE B1 da Rodada 3 de revisão da dissertação (BVRP, FGV EESP): definições e notação. Comunique-se sempre em português do Brasil.

## Contexto e decisões já tomadas (19/10)
O Prof. Marcelo respondeu às perguntas sobre o bootstrap, e eu confirmei duas decisões:
- **D1 (definição e notação):** vale a definição do TEXTO ATUAL, prospectiva: a medida é a volatilidade (variância) realizada nos 30 dias seguintes menos a IV (DVOL). A distinção VH × RV deixa de existir: a variância realizada nos próximos h dias é RV(t+1:t+h) = (365/h) · Σ r², com retornos logarítmicos diários. Notação: **RBVRP** para a medida (R = realizado no futuro); **HBVRP** para a proxy retrospectiva (H = histórico, realizado no passado); **BVRP** fica reservado ao prêmio teórico (E[RV | informação em t] − IV).
- **D3 (regressor gerado):** a análise principal dos Caps. 6 e 7 passará a usar os previsores diretamente na regressão de retornos, e o bootstrap da atenuação vira robustez/apêndice. **Isso NÃO é executado neste lote** (é o bloco B2, que muda números). Neste lote, apenas não escreva nada que contradiga essa direção e me avise se algum item B1 depender dela.
- O Fase0 (12/05/2026) e as regras 7 e 8 do Projeto (definição retrospectiva do BVRP; amostra de 1.686 observações) estão SUPERADOS por D1. Não os aplique e não sinalize esse conflito de novo.

## Leia antes, nesta ordem
1. `docs/rodada3/Plano_Rodada3_Anotacoes_Marcelo_v3_3_2.md` (seção 0 e o inventário do bloco B1; se estiver desatualizado em relação ao que descrevo aqui, vale o que está neste prompt).
2. `docs/rodada3/Anotacoes_Marcelo_v3_3_2_completo.csv` (colunas: id, pág., trecho, comentário do Marcelo, ação, bloco).
3. `docs/rodada3/pontos_de_atencao.md`.
O ramo é `rodada2-mcs`. Não faça commit, não crie ramo, não faça push sem eu pedir. `chapters/Cap6_ML.ipynb` (não rastreado) fica como está.

## Regras obrigatórias
1. NUNCA altere um arquivo sem antes me mostrar exatamente o que vai mudar (parágrafo inteiro antes × depois, com arquivo e linha) e esperar minha confirmação.
2. Faça backup `.bak` (nome novo se já existir `.bak`, sem sobrescrever) antes de modificar cada arquivo.
3. Uma tarefa por vez; levas de no máximo 8 itens por confirmação, capítulo a capítulo.
4. Terminologia: "aprendizado de máquina", "floresta aleatória", "MQO", "fora da amostra", "formadores de mercado", "modelo de referência", "neutra ao risco". Primeira pessoa fica para o bloco B8; não faça agora.
5. Se algo for ambíguo, PERGUNTE antes de decidir.
6. **Nenhum número, tabela ou figura pode mudar.** Se uma edição tocar tabela ou figura gerada por script (`code/`), rótulo dentro de imagem, ou qualquer número, PARE e me avise. Não edite `code/` e não edite arquivos `.tex` que são saída de script; liste-os para mim.
7. Cortes de referência cruzada: confirme que nenhum `\ref`, `\label` ou `\cite` fica órfão (rode `scripts/verificar_latex.py` ao fim de cada leva).

## Passo 0 (antes de editar qualquer coisa): inventário da notação
Faça uma busca em TODOS os `.tex` (capítulos, frontmatter, apêndices, legendas de tabelas e figuras) e me entregue uma tabela com:
a) todas as ocorrências de "BVRP" (e `\mathrm{BVRP}`, `\text{BVRP}` e variantes), classificadas em: (i) o prêmio teórico; (ii) a medida realizada (futura, 30 dias); (iii) a proxy retrospectiva; (iv) ambígua;
b) todas as ocorrências de "proxy", "VH", "RV", "variância histórica", "volatilidade histórica", "VH(", "RV(";
c) as equações (4.1) e (5.1), seus `\label` e onde são referenciadas;
d) quais ocorrências estão em tabelas/figuras geradas por script (não editar; só listar).
Mostre a tabela e espere minha confirmação da regra de renomeação antes de aplicar qualquer troca. Proponha a regra (por exemplo: onde o texto fala da medida usada nas regressões, vira RBVRP; onde fala da proxy retrospectiva, vira HBVRP; onde fala do prêmio teórico, continua BVRP) e aponte os casos (iv) para eu decidir.

## Escopo (22 itens do B1 + 4 adiados do Lote 1 = 26 itens; IDs na numeração da v3_3_2)
**Notação e definição (Cap. 1 e 2):** #10, #13, #15, #17, #18, #9, #16 (ver nota abaixo), #20 se ainda não estiver feito.
**IV ao quadrado (Cap. 2):** #39: definir IV² no texto e eliminar a nota de rodapé da desigualdade de Jensen. Cuide de não alterar nenhuma equação usada nos cálculos; se a definição de IV² exigir mudar equações ou rótulos de tabela, me avise antes.
**VH × RV (Cap. 2):** #46 (eliminar a distinção; RV nos próximos h dias = soma dos retornos² × 365/h; citar French, Schwert e Stambaugh, 1987. Atenção: a anotação do Marcelo diz "Journal of Financial Econometrics", mas o artigo conhecido é "Expected stock returns and volatility", Journal of Financial Economics, 19(1). Confirme título, periódico e páginas em fonte oficial, me proponha a entrada do `references.bib` e só insira depois da minha confirmação; se não conseguir confirmar, diga); #47, #48, #49, #50, #51, #52 (cortes ligados a VH). **Adiados do Lote 1 que dependem de D1:** #43 ("(realized volatility,"), #44 (referência cruzada "(Seção 2.3);"), #45 ("o,").
**Persistência (Cap. 2):** #65, #58 (parágrafo "muito confuso").
**Log-retornos e ADF/KPSS (Cap. 4):** #98 e #99 (escrever no texto a justificativa dele: log-retornos não são convenção; são cruciais para a agregação temporal, pois o retorno em baixa frequência é exatamente a soma dos retornos em alta frequência), #101 (argumento ADF/KPSS sobre níveis: preço e médias móveis da IV nunca entram sozinhos, só como desvio; reescrever).
**Cap. 5:** #121 (eliminar a equação (5.1) e a notação extra, e usar a (4.1); atenção ao `\label` e às referências), #120 (adiado do Lote 1: "nos próximos 30 dias" no lugar de "da janela de 30 dias que começa no dia seguinte à previsão"), #126 (cortar o ADF/KPSS repetido; relacionado a #101).

**Notas:**
- #16 e #58 falam do papel da proxy como regressor/variável explicativa. Esse papel mudará no B2 (D3). Escreva o texto de forma que continue verdadeiro nos dois cenários e me diga o que ficou dependente do B2.
- #10 e #15 falam de "proxy" e "BVRP" como rótulos de regime; reescreva conforme a observação do Marcelo (ambas as medidas são estimativas; a proxy é uma observação duplamente ruidosa: junta o erro do passeio aleatório com a diferença entre a vol esperada e a vol realizada entre t−29 e t).
- Se qualquer item colidir com uma decisão ainda aberta (D2: HAR; D4: fusão Cap. 2 e 4; D5: previsão média/árvores), pule e me diga.

## FORA deste lote
D2, D4, D5, o bloco B2 inteiro (bootstrap, teste F, EQM, HAR, regressão direta), B3, B4, B8, B9, `code/` inteiro, e o restante do B5. Itens ainda adiados: #94 (D4), #143, #144 (bootstrap, vão com a reescrita do apêndice), #157 (tabela de script).

## Como começar
1. Em uma frase: confirme que leu os três arquivos, o ramo e o estado do `git status`.
2. Faça o Passo 0 (inventário) e espere minha confirmação da regra de renomeação.
3. Faça a tabela de mapeamento dos 26 itens (ID, arquivo e linha, ação; marque os que não achou ou deixou de fora) e espere minha confirmação.
4. Execute por leva, com antes × depois e confirmação.
5. Ao fim: rode `scripts/verificar_latex.py`, compile se o `babel brazilian` estiver disponível (se não estiver, diga) e me entregue o resumo: itens aplicados, pulados (motivo), dependências do B2, arquivos com `.bak`. Sem commit até eu pedir.
