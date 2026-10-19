# Plano da 3ª rodada — anotações do Prof. Marcelo (v3_3_2 + e-mail de 18/10)

Origem: PDF `Dissertação_v3_3_2` (97 páginas) anotado pelo Prof. Marcelo (autores `marce` e `marcelo.fernandes`, a mesma pessoa) e os pontos do e-mail dele. Extração direta das anotações do PDF: **161 pontos**, mais **4 itens do e-mail** (EM1–EM4). A v3_3_2 repete as 142 anotações da v3_2 e acrescenta **19 novos** (marcados "novo" no CSV). **A numeração mudou** em relação ao plano anterior: #1 continua #1, e do antigo #2 em diante soma-se 1 (antigo #45 = novo #46, por exemplo). Este arquivo substitui a versão anterior.

Páginas citadas são as do PDF (a página 15 é o início do Capítulo 1).

## 0. Status em 18/10 (23h55)

| Item | Situação |
|---|---|
| Banca e remoção do Osvaldo | **Feito** (folha de aprovação e folha de rosto) |
| **Lote 1 (43 itens previstos)** | **Aplicado, sem commit.** 34 feitos + #57 (extra, texto do Marcelo), 1 sem ação (#40) e 8 adiados. Falta conferir o PDF no Overleaf |
| Bootstrap da atenuação (D7) | **Em espera** pela resposta do Marcelo; e-mail com 6 perguntas redigido |
| D1 (definição/notação do BVRP) | Aguardando você: confirmar que as regras 7 e 8 e o Fase0 estão superados |
| D2, D3, D4, D5 | Aguardando você |
| Blocos B1 a B4, B8, B9 e o restante do B5 | Não iniciados |

**Lote 1, adiados:** #43, #44, #45, #120 (D1); #94 (D4); #143, #144 (D7); #157 (tabela gerada por script). Os cortes de #107 e #108 e a nova citação de Almeida et al. (2026, no prelo) já estão no texto. Pontos de atenção em `docs/rodada3/pontos_de_atencao.md` (#74 e #100).

Próximo passo sugerido: decidir D1 e D3, que destravam B1, B2 e o Cap. 7.

## 1. Panorama

| Onde | Pág. PDF | Pontos |
|---|---|---|
| Pré-textual | 3–7 | 5 |
| Cap. 1 | 15–18 | 30 |
| Cap. 2 | 19–29 | 38 |
| Cap. 3 | 30–38 | 18 |
| Cap. 4 | 39–44 | 27 |
| Cap. 5 | 45–53 | 23 |
| Cap. 6 | 54–60 | 17 |
| Cap. 7 | 61–71 | 1 |
| Cap. 8/apêndices | 72–87 | 0 |
| Referências | 95 | 2 |

- Os 19 novos concentram-se no Cap. 6 (pág. 54–59): causalidade, **bootstrap da atenuação**, teste F e Tabela 6.5.
- Cap. 7 e Cap. 8 têm só uma marca (#159, primeira pessoa). Os comentários sobre eles vieram no e-mail: o Cap. 8 pode ser mais conciso e o Cap. 7 "ficou meio perdido". Ele disse que não leu a conclusão.
- Cerca de metade dos pontos são cortes e redação. Os que pesam: bloco B1 (22), bloco B2 (15) e os dois capítulos finais.

## 2. Relação com a Rodada 2 (comentários de 07/10)

| Item da Rodada 2 | Situação |
|---|---|
| T1 título em português | Feito: a capa traz "…Previsão com Aprendizado de Máquina" |
| M1–M5 MCS | O *model confidence set* já aparece no resumo, no abstract, no Cap. 4 (pág. 44–45) e no Cap. 5 (pág. 49). Falta conferir se bate com o plano M1–M5 |
| E1 assimetria (Wang 2001, Alexander 2006) | Nenhuma anotação trata disso; continua em espera |
| E2 gráfico do C3.4 | Possível ligação com #79 (tirar a proxy retrospectiva do gráfico, pág. 33). A confirmar |
| E3 proxy × prêmio prospectivo | #15, #32 e #58 são a leitura dele; viram D1 e D2 |
| **Bootstrap (Cap. 6): "tranquilo, faz sentido"** | **Reaberto.** No e-mail e em #145–#150 ele agora tem "sérias dúvidas" sobre a correção do viés de atenuação (D7) |

## 3. Decisões que dependem de você antes de executar

**D1 — Definição e notação do BVRP (#46, #47, #15, #17, #18, #121).** Ele quer acabar com a distinção entre VH e RV (RV nos próximos h dias = soma dos retornos² × 365/h) e chamar a sua medida de BVRP com chapéu ou RBVRP. **Atenção:** li o Fase0 (versão 1.0, de 12/05/2026) e as regras do projeto. Eles fixam BVRP_t = RV_t − IV_t com RV retrospectiva (regra 7), amostra de 1.686 observações (regra 8) e 7 capítulos. O texto atual já é diferente: alvo prospectivo (VH dos 30 dias seguintes − DVOL), 987 previsões fora da amostra, 8 capítulos e MCS. As regras 7 e 8 e o Fase0 parecem superados pela reunião de 29/09. Preciso que você confirme isso, para eu parar de sinalizar a definição atual como "antiga".

**D2 — HAR no Cap. 5 (#32).** Trocar a proxy retrospectiva por VH(t:t), VH(t−4:t), VH(t−9:t), VH(t−29:t), seguindo Corsi (2009). Muda a especificação e os números; previsões e MCS têm de ser refeitos.

**D3 — Regressor gerado (#21, #115).** Incluir direto, na regressão de retornos, as variáveis que prevêem o BVRP, em vez do BVRP previsto, e olhar o ganho de EQM. É a mudança de maior alcance nos Caps. 6 e 7. **Se você adotar (a), o problema do bootstrap (D7) desaparece**, porque deixa de existir regressor gerado. Por isso sugiro decidir D3 antes de D7.

**D4 — Estrutura (#61, #62, #92).** Preferência dele: eliminar do Cap. 2 o parágrafo que rediscute Alexander e Imeraj, Almeida et al. e Han, levando o que precisar à introdução. Fundir os Caps. 2 e 4: "talvez não valha a pena mudar agora". Sugiro só a primeira parte.

**D5 — Previsão média e árvores (#135, #136).** Testar a média das previsões dos métodos; refazer a grade das árvores com profundidade mínima 3 (ou justificar a profundidade 1).

**D6 — Banca (#1, #2, EM4). FEITO em 18/10.** Banca: Marcelo Fernandes (orientador), Matheus Carrijo e Fernando Chague, todos professores doutores da FGV EESP. O Osvaldo saiu (não pode participar da banca). Alterações no ramo `rodada2-mcs`, com backup `.bak`: `frontmatter/folha_aprovacao.tex` (Osvaldo → Matheus Carrijo; campo livre → Fernando Chague) e `frontmatter/folha_rosto.tex` (linha de co-orientador removida). Não restam outras menções ao Osvaldo nos `.tex`, `.md` e `.bib`. A compilação completa não rodou no ambiente (falta o pacote `brazilian` do babel), mas as páginas de rosto saem com os nomes certos. Falta só conferir no Overleaf.

**D7 — Bootstrap do viés de atenuação (#145–#150, EM3, Cap. 6, pág. 55). EM ESPERA até a resposta do Marcelo.** Perguntas já redigidas (objetivo do bootstrap, geração dos erros, retornos fixos ou simulados, fim da reamostragem dos pares, alternativa com os previsores direto e teste F/EQM, papel no texto). Hoje o texto faz o seguinte: em cada reamostragem, reestima o modelo de previsão em cada origem com treino reamostrado em blocos e reestima β com os pares (previsão, retorno) das 987 datas, também reamostrados em blocos. Define λ = média dos β* / β̂ (entre 0,59 e 0,70) e usa o erro-padrão do bootstrap dividido por λ, afirmando que β̂ "não é atenuado" e é uma medida conservadora. As objeções dele: (i) reamostrar os pares previsão-retorno não faz muito sentido; (ii) **o viés de atenuação está exatamente no β̂**, então a frase "β̂ não atenuado" está errada; (iii) não está claro como a atenuação afeta o β* nem se dá para corrigir β̂ com o bootstrap; (iv) Künsch (1989) é referência geral e não trata de atenuação. Ele propõe um bootstrap paramétrico: simular erros de previsão do BVRP, gerar séries alternativas condicionais aos previsores (uma por modelo), reestimar cada modelo, tratar a previsão original como "verdade", regredir o retorno na original e em cada alternativa e comparar a média dos β* com o β̂ original. Opções: (a) reimplementar como ele descreve (muda λ, erros-padrão, valores-p e as Tabelas 6.1–6.2); (b) manter o método e reescrever a justificativa, sem "não atenuado" nem "conservador"; (c) eliminar o problema pela D3(a). Ele mesmo disse que pode não ter entendido bem, então vale perguntar a ele depois de decidirmos.

## 4. Blocos de trabalho, na ordem proposta

Regras do projeto mantidas: mostrar cada alteração antes de aplicar, backup `.bak`, uma tarefa por vez, números antigos × novos ao fim de cada tarefa que mexer em resultado.

| Bloco | Conteúdo | Pontos | Nº | Natureza |
|---|---|---|---|---|
| **B1** | Conceitos, definições e notação (VH × RV, proxy, IV², log-retornos). Define o vocabulário do resto do texto; fazer primeiro. | #9–10, #13, #15–18, #39, #46–52, #58, #65, #98–99, #101, #121, #126 | 22 | Texto, sem números novos |
| **B2** | Especificação e resultados (HAR, regressor gerado, bootstrap da atenuação, teste F, EQM, árvores, previsão média, autocorrelação). Únicos que podem mudar números. Exigem rodar o pipeline e comparar antigo × novo. | #21, #32, #76, #78–79, #115, #129, #135–136, #145–147, #150–151, EM3 | 15 | Números |
| **B3** | Estrutura (fusão Cap. 2 + Cap. 4, mover seções, cortar títulos). Antes de mexer em frases, para não editar texto que vai se mover ou sair. | #61–62, #92, #95, #97, #110, #112 | 7 | Estrutura |
| **B4** | Introdução e posicionamento (Cap. 1). Redistribuir os contrastes no fim do parágrafo de cada artigo; quebras de parágrafo. | #12, #25–31, #33–34, #88 | 11 | Texto |
| **B5** | Clareza e redação (frases, cortes, definições pontuais). Maior bloco; correções locais e independentes. | #3–8, #11, #20, #22–24, #35–36, #38, #41–45, #54–57, #59–60, #63–64, #66–67, #70, #73–75, #77, #80–81, #83–87, #89–90, #94, #96, #100, #102, #106–109, #111, #114, #116–117, #119–120, #123–125, #127–128, #130, #132–134, #137–138, #140–144, #149, #152–155, #157–158 | 80 | Texto |
| **B6** | Notação e LaTeX. amssymb, \mathbb{E}, \noindent, símbolos do embargo. | #14, #37, #40, #53, #103–105, #122 | 8 | LaTeX |
| **B7** | Referências e banca. Bibliografia, ordem das citações, folha de aprovação. | #1–2, #68, #82, #148, #160–161, EM4 | 8 | Referências |
| **B8** | Varredura final: primeira pessoa. Último passo: cerca de 68 ocorrências de "a/esta/desta dissertação" do resumo ao Cap. 7. | #19, #69, #71–72, #93, #113, #118, #131, #159 | 9 | Texto |
| **B9** | Capítulos finais (Cap. 7 "perdido" e Cap. 8 mais conciso). Depois do B2, porque a mensagem final depende dos resultados. | EM1, EM2 | 2 | Texto e estrutura |
| — | Sem ação | #91, #139, #156 | 3 | — |

Por que essa ordem: B1 e B2 definem definições e números; B3 move e corta; depois vale polir frases (B4–B5), LaTeX e referências (B6–B7). O B9 vem depois dos resultados e a varredura de primeira pessoa (B8) fica por último.

Dependências: D3 antes de D7; B2 depende de D2, D3, D5, D7. #65 (persistência), #76 e #78 tratam do mesmo assunto e devem ser feitos juntos. #101 e #126 são o mesmo comentário (ADF/KPSS).

Banca (B7: #1, #2, EM4): concluída. Bootstrap (#145–#150, EM3): em espera. Os demais itens do B7 (#68, #82, #148, #160, #161) seguem abertos.

## 5. Inventário completo por bloco

### B1 — Conceitos, definições e notação (VH × RV, proxy, IV², log-retornos) (22)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 9 | 15 | Cap. 1 | Destaque | previsão neutra | o que seria uma previsão neutra? | Definir ou trocar "previsão neutra" |
| 10 | 15 | Cap. 1 | Tachado | que se realiza depois | volatilidade realizada | Trocar por "volatilidade realizada" (segue da decisão D1) |
| 13 | 16 | Cap. 1 | Destaque | regride | mas isso corresponde a calcular uma média. melhor não falar ainda em regressão para evitar dúvidas sobre má especificação. | Trocar "regride" (é uma média; evitar falar de regressão aqui) |
| 15 | 16 | Cap. 1 | Destaque | a proxy registra um prêmio maior nesse regime, e o BVRP, nenhum | não sei se gosto desses rótulos. BVRP = E(VH / info em t) - VIX  portanto, ambas as medidas são estimativas. uma usa a variância realizada entre t+1 e t+30 (ou seja, uma observação ruidosa da volatilidade esperada) e a outra calcula a variância histórica nos últimos 30 dias (sob a premissa de passeio aleatório, uma observação duplamente ruidosa da vol esperada).  duplamente porque junta o erro do passeio aleatório com o erro que vem da diferença entre a vol esperada e a vol realizada entre t e t-29. | Reescrever os rótulos proxy/BVRP: ambas são estimativas (BVRP = E(VH / informação em t) − IV); proxy = observação duplamente ruidosa |
| 16 | 16 | Cap. 1 | Destaque | não pode ser usado como variável explicativa | claro que pode, assim é em qualquer regressão preditiva | Corrigir o argumento: a proxy pode, sim, ser regressor (vale em qualquer regressão preditiva) |
| 17 | 16 | Cap. 1 | Destaque | BVRP | chamaria tua medida de $\widehat{\mbox{BVRP}$ ou de RBVRP para deixar claro que se trata de uma estimativa (R de realizado). assim fica mais fácil identificar quando você fala do prêmio de risco de volatilidade mesmo. | Renomear a medida: BVRP com chapéu (\widehat{BVRP}) ou RBVRP, para distinguir da definição teórica |
| 18 | 16 | Cap. 1 | Nota | — | R = realizado no futuro H = realizado no passado, portanto histórico | Notação: R = realizado no futuro; H = realizado no passado (histórico) |
| 39 | 19 | Cap. 2 | Tachado | 1A rigor, o DVOL, como o VIX, é a raiz da variância esperada no mundo neutro ao risco, IVt = √ t E∗( (RV2 t… | mais fácil definir IV ao quadrado no texto e evitar essa situação | Definir IV ao quadrado no texto e eliminar a nota de rodapé sobre a desigualdade de Jensen |
| 46 | 21 | Cap. 2 | Destaque | A VH é, assim, o estimador com dados diários da RV que entra na definição teórica do prêmio | não entendi. VH é uma média, RV é uma soma.  Como você está agora olhando entre t+1 e t+h, você pode chamar de volatilidade realizada nos próximos h dias mesmo. Você soma os retornos diários ao quadrado e depois só anualiza multiplicando por 365/h.  perdão, bati tanto nesta tecla, mas agora que definiu BVRP dessa forma, a distinção entre VH e RV não faz mais sentido.  RV mensal usando dados diários, como em French, Schwert & Stambaugh (Journal of Financial Econometrics, 1987) | Eliminar a distinção VH × RV: RV nos próximos h dias = soma dos retornos² × 365/h (French, Schwert e Stambaugh, 1987) (decisão D1) |
| 47 | 21 | Cap. 2 | Destaque | A escolha de muda as propriedades da medida. Com grande, a janela troca h h apenas uma de observações de um… | essa discussão faz sentido para vol histórica, mas não para vol realizada | Discussão sobre o tamanho da janela vale para volatilidade histórica, não realizada; ajustar ou cortar |
| 48 | 21 | Cap. 2 | Tachado | Alternativamente, pode-se definir a variância histórica como: (︂ )︂ 365 1 h−1 ∑ 2 VH(h t h) = r2 t− , H t t… | — | Cortar a definição alternativa de VH |
| 49 | 21 | Cap. 2 | Tachado | Embora a volatilidade histórica seja usualmente construída a partir de retornos logarítmicos simples, pode-… | Não faz o menor sentido aqui, já que você quer olhar a variação quadrática do Bitcoin | Cortar o trecho de retornos em excesso (não faz sentido: o objeto é a variação quadrática do Bitcoin) |
| 50 | 21 | Cap. 2 | Tachado | No entanto, para frequências diárias e, em particular, no contexto de criptoativos, a diferença empírica en… | — | Cortar (continuação do trecho de retornos em excesso) |
| 51 | 22 | Cap. 2 | Tachado | seguida neste trabalho. | — | Cortar (fim do trecho) |
| 52 | 22 | Cap. 2 | Tachado | A volatilidade realizada propriamente dita exigiria dados intradiários, o que au- mentaria a precisão (high… | — | Cortar o parágrafo sobre dados intradiários e a escolha por volatilidade histórica |
| 58 | 23 | Cap. 2 | Destaque | Nesta dissertação, a proxy aproxima a expectativa do prêmio, e não sua realização: ela entra como variável … | muito confuso | Reescrever o parágrafo da proxy ("muito confuso"); ligado ao E3 de proxy × prêmio prospectivo |
| 65 | 23 | Cap. 2 | Sublinhado | tende a cancelar boa parte dele; o Capítulo 3 mede esse padrão na amostra. No BVRP há, além disso, uma font… | ficou contraditório. que tal dizer Por um lado, espera-se menos persistência em BVRP do que observamos nas medidas de volatilidade porque boa parte dessa persistência é comum. Por outro lado, como estamos medindo diariamente variâncias realizadas de 30 dias, espera-se um nível maior de persistência do que aquela esperada na frequência mensal por conta da sobreposição de 29 dias. | Reescrever a persistência sem contradição: "Por um lado, espera-se menos persistência em BVRP… Por outro, como estimamos diariamente variâncias realizadas de 30 dias, espera-se mais persistência do que na frequência mensal, por conta da sobreposição de 29 dias" |
| 98 | 40 | Cap. 4 | Destaque | O retorno simples corresponde ao retorno efetivo de uma posição no ativo; a volatilidade histórica, por sua… | então por que não usar log retornos aqui também? (não é importante, mas dá espaço para avaliador chato reclamar. tenha uma resposta para isso.) | Log-retornos: preparar resposta ao avaliador (ver #99) |
| 99 | 40 | Cap. 4 | Nota | — | o uso de log retornos não é uma convenção. é crucial para a agregação temporal, pois garante que o retorno na frequência mais baixa seja exatamente igual à soma dos retornos na frequência mais alta. | Resposta dele: log-retorno não é convenção; é crucial para a agregação temporal (retorno de baixa frequência = soma dos de alta). Usar no texto |
| 101 | 41 | Cap. 4 | Destaque | o preço e as médias móveis da IV, em que o ADF não rejeita a raiz unitária e o KPSS rejeita a estacionaried… | opa, não faz muito sentido já que não entram nunca sozinhas em nível, aparecendo sempre como desvio entre uma e outra. | Argumento ADF/KPSS sobre níveis não faz sentido (preço e médias móveis da IV nunca entram sozinhos, só como desvio); reescrever |
| 121 | 46 | Cap. 5 | Destaque | A equação (5.1) é a equação (4.1) com a janela explícita; o restante do texto usa BVRPt. Pt | meio desnecessário introduzir mais uma notação e mais uma equação para a mesma coisa. | Eliminar a equação (5.1) e a notação extra (é a (4.1) com a janela explícita) (ligado a D1) |
| 126 | 46 | Cap. 5 | Destaque | O preço e as médias móveis da IV, em que o ADF não rejeita a raiz unitária e o KPSS rejeita a estacionaried… | já comentei sobre isso | Cortar: o argumento ADF/KPSS já foi comentado (ver #101) |

### B2 — Especificação e resultados (HAR, regressor gerado, bootstrap da atenuação, teste F, EQM, árvores, previsão média, autocorrelação) (15)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 21 | 16 | Cap. 1 | Destaque | BVRP previsto é um regressor gerado, | uma solução mais simples, como falei anteriormente, é colocar todos as variáveis que você usou para prever o BVRP diretamente (em vez de usar o previsto) | Regressor gerado: avaliar incluir diretamente as variáveis que prevêem o BVRP, em vez do BVRP previsto (decisão D3) |
| 32 | 18 | Cap. 1 | Destaque | proxy retrospectiva entra como variável explicativa | não sei se faz muito sentido. do lado esquerdo, você vai ter VH(t+1:t+30)-IV(t). do lado direito, vai ter VH(t-29:t)-IV(t). Repare que IV é mensurável em t, portanto é apenas uma normalização de certa forma. o que você realmente quer prever é VH(t+1:t+30). Faz mais sentido usar VH(t:t), VH(t-4:t), VH(t-9:t) e VH(t-29:t), considerando as evidências a favor do modelo HAR (Corsi, 2009). | Proxy retrospectiva como regressor não faz sentido (IV é só normalização); usar VH(t:t), VH(t−4:t), VH(t−9:t), VH(t−29:t), à la HAR (Corsi, 2009) (decisão D2) |
| 76 | 32 | Cap. 3 | Destaque | Os insumos do prêmio são bem mais persistentes: na mesma defasagem, a autocorrelação é de 0,38 e 0,39 nas v… | é o contrário. o fato de você suavizar BVRP diário sobreponto 29 das 30 observações aumenta a persistência. | Correção de conteúdo: é o contrário; suavizar o BVRP diário com 29 de 30 observações sobrepostas AUMENTA a persistência. Revisar a frase e o resultado associado (ligado a #65, #78) |
| 78 | 33 | Cap. 3 | Destaque | numa amostra sem sobreposição, | legal, mas talvez valha a pena você usar essa série mensal para calcular a autocorrelação do BVRP sem sobreposição. (mas aí precisa comparar com a autocorrelação das volatilidades mensais) | Calcular a autocorrelação do BVRP sem sobreposição (série mensal) e comparar com a das volatilidades mensais |
| 79 | 33 | Cap. 3 | Destaque | retrospectiva | sinceramente, não vejo muito ganho de incluí-la no gráfico. polui muito até porque reflete a outra proxy com defasagem. | Retirar a proxy retrospectiva do gráfico (polui e reflete a outra proxy defasada); verificar se é o gráfico do C3.4 (E2) |
| 115 | 43 | Cap. 4 | Destaque | Por isso, a inferência principal usa um bootstrap em blocos em dois níveis (Künsch, 1989). | mas se o interesse é em previsão, você pode olhar diretamente para os ganhos de MSE (EQM, em português). afinal, o coeficiente pode ser significante, mas não necessariamente implicar ganho de previsão. é o que vc faz no MCS | Se o interesse é previsão, olhar diretamente os ganhos de EQM (MSE), não só o coeficiente (como no MCS); avaliar o que isso muda no Cap. 6 (decisão D3) |
| 129 | 47 | Cap. 5 | Destaque | ∆5I IVt − | não faz diferença já que você controla por IV | Δ5 IV não faz diferença, já que se controla por IV; avaliar remover a variável (muda números) |
| 135 | 47 | Cap. 5 | Destaque | profundidade 1 | não faz o menor sentido usar profundidades menor que 3, em geral. os resultados indicam realmente falta de previsibilidade. bacana, pois é como se esses modelos estivessem escolhendo os momentos em que há previsibilidade e aqueles sem previsibilidade. | Profundidade menor que 3 não faz sentido em geral; refazer a grade ou explicar a leitura (os resultados indicam falta de previsibilidade; as árvores escolhem os momentos com e sem previsibilidade) |
| 136 | 49 | Cap. 5 | Destaque | MQO Ridge 0,097 0,223 −0,037 0,095 <0,001 0,004 <0,001 0,002 0,707 1,000† 0,647 1,000† 0,291 0,211 < | parece o melhor. mas seria legal tentar a previsão média dos diversos métodos. | Testar a previsão média (ensemble) dos diversos métodos (decisão D5) |
| 145 **(novo)** | 55 | Cap. 6 | Destaque | reestima o modelo de previsão em cada origem, com um treino reamostrado em blocos dentro da própria janela … | não ficou muito claro como você faz isso. logo em seguida você fala de reamostrar o par previsão-retorno, que não faz muito sentido. | Bootstrap: não ficou claro como reestima o modelo com treino reamostrado; reamostrar o par (previsão, retorno) não faz muito sentido (decisão D7) |
| 146 **(novo)** | 55 | Cap. 6 | Destaque | ˆβh comparar , que não é atenuado ˆβh | como assim? o viés de atenuação é exatamente no beta estimado. | Bootstrap: "β̂ não atenuado" está errado; o viés de atenuação está exatamente no β̂ estimado (D7) |
| 147 **(novo)** | 55 | Cap. 6 | Destaque | em ˆβh O ˆβh estimado com o BVRP previsto é, portanto, uma medida conservadora da relação entre o prêmio e … | mas como o viés de atenuação afeta o beta estrela? dá para corrigir o viés de beta chapéu usando as estimativas de bootstrap? | Bootstrap: como a atenuação afeta o β*? Dá para corrigir o viés de β̂ com as estimativas do bootstrap? A frase "medida conservadora" depende disso (D7) |
| 150 **(novo)** | 55 | Cap. 6 | Nota | — | se eu tentasse corrigir o viés de atenuação com bootstrap, criaria amostras artificiais do erro de previsão de BVRP e usaria os parâmetros estimados para criar novas séries de BVRP condicional aos previsores (cada modelo daria uma série diferente naturalmente). em seguida, reestimaria cada modelo para obter previsões alternativas de BVRP em cada uma dessas amostras de bootstrap. no metaverso de bootstrap, a previsão original é a verdade, pois estamos usando os parâmetros estimados nessa amostra original. finalmente, rodaríamos a regressão preditiva dos retornos na previsão original do BVRP e para cada previsão alternativa. Compararia a média dos betas de bootstrap com a estimativa original para calcular o fator de atenuação. | Algoritmo proposto por ele para corrigir a atenuação: simular erros de previsão do BVRP, gerar séries alternativas condicionais aos previsores (uma por modelo), reestimar cada modelo, tomar a previsão original como "verdade" no mundo do bootstrap, regredir retorno na original e em cada alternativa e comparar a média dos β* com o β̂ original → fator de atenuação (D7) |
| 151 **(novo)** | 58 | Cap. 6 | Destaque | Nenhum coeficiente individual é significante depois da correção de Bonferroni | sim, mas aqui você deveria testar beta1=beta2=0 usando uma estatística-F | Cap. 6: testar β1 = β2 = 0 com estatística F, em vez de coeficientes individuais com Bonferroni |
| EM3 | 54–56 | Cap. 6 | E-mail | Bootstrap do viés de atenuação | Tenho sérias dúvidas sobre o bootstrap que você usou para corrigir o viés de atenuação, mas talvez não tenha entendido bem. | EM ESPERA. Reabre o que estava fechado na Rodada 2. Detalhes em #145–#150 e na decisão D7 |

### B3 — Estrutura (fusão Cap. 2 + Cap. 4, mover seções, cortar títulos) (7)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 61 | 23 | Cap. 2 | Destaque | Os dois trabalhos que estudam o prêmio de volatilidade do Bitcoin com dados da Deribit (Alexander e Imeraj,… | já que você inclui um capítulo de referencial teórico, você precisa discutí-los novamente. | Cap. 2 repete a discussão de Alexander e Imeraj, Almeida et al. e Han: eliminar daqui e levar o que for preciso à introdução (preferência dele) (decisão D4) |
| 62 | 23 | Cap. 2 | Nota | — | ou você elimina esse parágrafo daqui, colocando o que precisar na introdução. (minha preferência) | Mesma decisão: eliminar o parágrafo do Cap. 2 |
| 92 | 39 | Cap. 4 | Destaque | Metodologia | talvez não valha a pena mudar agora, mas a estrutura poderia estar bem mais enxuta juntando o capítulo 2 de referencial teórico com esse capítulo. afinal meio que versam sobre a mesma coisa. | Possível fusão do Cap. 2 (referencial teórico) com o Cap. 4 (metodologia); ele mesmo diz que talvez não valha mudar agora (decisão D4) |
| 95 | 39 | Cap. 4 | Destaque | (Seção 4.5). | mas se o capítulo 3 já respondeu, por que aparece aqui e , em particular, na seção 4.5? | Se o Cap. 3 já respondeu, por que reaparece no Cap. 4 e na Seção 4.5? Justificar ou cortar |
| 97 | 40 | Cap. 4 | Destaque | Definição das variáveis e horizontes de previsão | isso deveria estar na seção de dados. por isso que as seções de dados e metodologia costumam ficar no mesmo capítulo | Mover "Definição das variáveis e horizontes de previsão" para o Cap. 3 (dados e metodologia ficam juntos) |
| 110 | 42 | Cap. 4 | Tachado | 4.4.2 Retornos e BVRP previsto: modelo linear | — | Cortar o título 4.4.2 (juntar com a subseção anterior) |
| 112 | 43 | Cap. 4 | Tachado | 4.4.3 Retornos e BVRP previsto: modelo em árvore | — | Cortar o título 4.4.3 |

### B4 — Introdução e posicionamento (Cap. 1) (11)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 12 | 16 | Cap. 1 | Destaque | A única hipótese formal da dissertação (H1) é que o prêmio é negativo em média. | aqui você trocou de assunto, portanto talvez fosse o caso de quebrar parágrafo ou fazer ao menos uma transição. | Quebrar o parágrafo ou criar transição antes da hipótese H1 |
| 25 | 17 | Cap. 1 | Destaque | ouro. Almeida | quebra de parágrafo | Quebrar parágrafo depois de "ouro. Almeida" |
| 26 | 17 | Cap. 1 | Destaque | Nenhum dos dois prevê o prêmio nem regride o retorno do Bitcoin nele. | eu colocaria essa frase no final do parágrafo acima (considerando a quebra que sugeri) e novamente no final deste. | Mover a frase "Nenhum dos dois prevê…" para o fim do parágrafo anterior e repetir no fim deste |
| 27 | 17 | Cap. 1 | Tachado | Em trabalho relacionado, | quebraria parágrafo também | Quebrar parágrafo antes de "Em trabalho relacionado" |
| 28 | 17 | Cap. 1 | Destaque | série, agregado por semana: as inovações | ficou esquisita essa frase. basta ir direto ao ponto. Han (2026) mostra que as inovações do DVOL semanal formam um fator de risco de volatilidade apreçado no corte transversal de criptoativos. | Reescrever: "Han (2026) mostra que as inovações do DVOL semanal formam um fator de risco de volatilidade apreçado no corte transversal de criptoativos." |
| 29 | 18 | Cap. 1 | Destaque | Ele não mede o prêmio de volatilidade. | tenta sempre contrastar com o que você fez para ficar clara sua contribuição. | Contrastar com o que foi feito aqui, para deixar clara a contribuição |
| 30 | 18 | Cap. 1 | Destaque | Primeiro, a amostra: cinco anos de dados diários, de março de 2021 a março de 2026, mais longa que a de Ale… | é uma diferença relevante, mas não é que você acrescente um elemento a mais. | Amostra: é diferença relevante, mas não "acrescenta um elemento"; reformular |
| 31 | 18 | Cap. 1 | Destaque | A volatilidade implícita é o mesmo índice usado por Han (2026), o DVOL, e não um índice próprio | essa constatação deveria estar lá na frase final do parágrafo do Han, quando você se compara. | Mover "DVOL é o mesmo índice de Han (2026)" para o fim do parágrafo do Han |
| 33 | 18 | Cap. 1 | Destaque | Quarto, os regimes: como Almeida et al. (2024), a dissertação usa dois regimes de volatilidade, mas definid… | legal, mas deveria estar no final do parágrafo sobre Almeida et al, quando você se compara. | Mover o item dos regimes para o fim do parágrafo de Almeida et al. |
| 34 | 18 | Cap. 1 | Nota | — | ou seja, acho que distribuiria cada uma dessas frases nos parágrafos acima. sempre no final do parágrafo de cada paper, comparando o seu ao deles. | Regra geral: distribuir cada contraste no fim do parágrafo do respectivo artigo |
| 88 | 36 | Cap. 3 | Destaque | obtidos de outra forma | talvez valha a pena descrever como fazem (até mesmo para você diferenciar sua contribuição da deles na introdução) | Descrever como Almeida et al. obtêm o prêmio (e usar na introdução para diferenciar a contribuição) |

### B5 — Clareza e redação (frases, cortes, definições pontuais) (80)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 3 | 6 | Pré-textual | Nota | — | está super concisa, o que é ótimo, mas dá para melhorar o flow. | Resumo: já está concisa; melhorar o fluxo entre as frases |
| 4 | 7 | Pré-textual | Tachado | volatility subsequently realized | subsequent realized volatility | Abstract: trocar por "subsequent realized volatility" — **FEITO (Lote 1)** |
| 5 | 7 | Pré-textual | Destaque | exceeds | typically exceeds (para não repetir on average) | Abstract: "typically exceeds" (evita repetir "on average") — **FEITO (Lote 1)** |
| 6 | 15 | Cap. 1 | Tachado | na amostra desta dissertação, de março de 2021 a março de 2026, foi de 61,9% a.a. | — | Cortar a média da amostra (61,9% a.a.) da frase — **FEITO (Lote 1)** |
| 7 | 15 | Cap. 1 | Destaque | ; em | ponto e nova frase | Ponto final e nova frase em vez de ";" — **FEITO (Lote 1)** |
| 8 | 15 | Cap. 1 | Tachado | (Seção 2.4.1). | — | Cortar a referência cruzada — **FEITO (Lote 1)** |
| 11 | 15 | Cap. 1 | Destaque | prevê | ajuda prever | "ajuda prever" em vez de "prevê" — **FEITO (Lote 1)** |
| 20 | 16 | Cap. 1 | Destaque | a regressão de Bollerslev et al. (2009) | não existe um objeto chamdo regressão de Bollerslev et al. Você pode dizer algo como: Como em Bollerslev et al. (2009), estimamos uma regressão preditva para entender se o prêmio de risco de volatilidade ajuda prever o retorno do Bitcoin até 60 dias à frente. | Reescrever: "Como em Bollerslev et al. (2009), estimamos uma regressão preditiva…" (não existe "a regressão de Bollerslev") |
| 22 | 17 | Cap. 1 | Tachado | são os primeiros: | — | Cortar "são os primeiros:" |
| 23 | 17 | Cap. 1 | Tachado | com o mesmo sinal da adotada aqui. | — | Cortar "com o mesmo sinal da adotada aqui" |
| 24 | 17 | Cap. 1 | Tachado | com o sinal oposto ao desta dissertação | sei que você está falando sobre VH-IV contra IV-VH, mas ficou ambíguo. | Trecho sobre o sinal oposto ficou ambíguo (VH−IV × IV−VH); cortar ou reescrever |
| 35 | 18 | Cap. 1 | Tachado | (Seção 3.5). | — | Cortar a referência cruzada (Seção 3.5) — **FEITO (Lote 1)** |
| 36 | 19 | Cap. 2 | Tachado | (Implied Volatility, | — | Cortar "(Implied Volatility," — **FEITO (Lote 1)** |
| 38 | 19 | Cap. 2 | Tachado | (Seção 2.4.1). | — | Cortar a referência cruzada — **FEITO (Lote 1)** |
| 41 | 20 | Cap. 2 | Tachado | fatores como os seguintes | os seguintes fatores podem influenciar | "Os seguintes fatores podem influenciar" — **FEITO (Lote 1)** |
| 42 | 21 | Cap. 2 | Destaque | por | de | "por" → "de" — **FEITO (Lote 1)** |
| 43 | 21 | Cap. 2 | Tachado | (realized volatility, | — | Cortar "(realized volatility," — **ADIADO: depende da D1** |
| 44 | 21 | Cap. 2 | Tachado | (Seção 2.3); | — | Cortar a referência cruzada (Seção 2.3) — **ADIADO: depende da D1** |
| 45 | 21 | Cap. 2 | Tachado | o, | — | Cortar "o," — **ADIADO: depende da D1** |
| 54 | 22 | Cap. 2 | Destaque | RV é a volatilidade realizada entre t + 1 e t + h, é a informação disponível It Vt+1:t+h h em t e o asteris… | acho que tudo isso já foi definido antes | Cortar: já foi definido antes |
| 55 | 23 | Cap. 2 | Tachado | A hipótese é uma simplificação, e | — | Cortar "A hipótese é uma simplificação, e" — **FEITO (Lote 1)** |
| 56 | 23 | Cap. 2 | Tachado | su | — | Cortar "su" — **FEITO (Lote 1)** |
| 57 | 23 | Cap. 2 | Tachado | limitações, estimando a expectativa física com modelos de previsão da variância | limitações dessa simplicação, considerando modelos mais gerais de previsão da variância. | Reescrever: "limitações dessa simplificação, considerando modelos mais gerais de previsão da variância" — **FEITO (Lote 1)** |
| 59 | 23 | Cap. 2 | Destaque | esse fato estilizado | qual? | Qual fato estilizado? Explicitar |
| 60 | 23 | Cap. 2 | Destaque | s — achado replicado em diversos índices acionários internacionais (Bollerslev et al., 2014) | outra frase. | Separar em outra frase (Bollerslev et al., 2014) |
| 63 | 23 | Cap. 2 | Tachado | característica que tem implicações relevantes para a inferência estatística em modelos preditivos | — | Cortar "característica que tem implicações relevantes…" — **FEITO (Lote 1)** |
| 64 | 23 | Cap. 2 | Destaque | alvos | ???? | "alvos" (????): esclarecer o termo |
| 66 | 24 | Cap. 2 | Destaque | janelas expansivas para qualquer cálculo de limiares estatísticos. | não guarda nenhuma relação | "Janelas expansivas para qualquer cálculo de limiares" não tem relação com o contexto; mover ou cortar |
| 67 | 24 | Cap. 2 | Destaque | quando o regressor de uma regressão preditiva é persistente e suas inovações se correlacionam com as do ret… | inverteria a ordem. falaria primeiro que Stambaugh (1999) demonstra que há um viés e depois falaria como vc resolve por reamostragem. Fica mais natural apresentar o problema antes da solução. | Inverter a ordem: primeiro Stambaugh (1999) mostra o viés; depois a solução por reamostragem |
| 70 | 25 | Cap. 2 | Destaque | no modelo em árvore do Capítulo 7, o próprio modelo escolhe os cortes. | ficou perdido aqui | Trecho sobre o modelo em árvore do Cap. 7 "ficou perdido"; reescrever |
| 73 | 29 | Cap. 2 | Destaque | A floresta aleatória é também o modelo único em árvore | essa frase ficou esquisita | Frase sobre a floresta aleatória "ficou esquisita"; reescrever |
| 74 | 30 | Cap. 3 | Tachado | Com exceção do teste de H1, | — | Cortar "Com exceção do teste de H1," — **FEITO (Lote 1)** |
| 75 | 31 | Cap. 3 | Destaque | D.P. | a nota da tabela precisa explicar o que significa cada uma dessas entradas. | Nota da tabela deve explicar cada entrada (D.P. etc.) |
| 77 | 33 | Cap. 3 | Destaque | a 90. | que exagero! | "a 90": exagero; suavizar |
| 80 | 35 | Cap. 3 | Tachado | essa decomposição exigiria dados intradiários, indisponíveis nesta dissertação. | — | Cortar a frase sobre decomposição com dados intradiários |
| 81 | 35 | Cap. 3 | Destaque | bimodal, e | quebra de frase | Quebrar a frase depois de "bimodal," — **FEITO (Lote 1)** |
| 83 | 36 | Cap. 3 | Destaque | (Capítulo 4). | no Capítulo 4 | "no Capítulo 4" — **FEITO (Lote 1)** |
| 84 | 36 | Cap. 3 | Destaque | a mistura troca de solução | não entendi o que isso quer dizer | "a mistura troca de solução": não ficou claro; reescrever |
| 85 | 36 | Cap. 3 | Tachado | Como BVRP = VH IV, prêmio maior corresponde a BVRP mais negativo; Almeida et al. − (2024), que medem o prêm… | repetitivo | Cortar o trecho repetitivo sobre o sinal oposto de Almeida et al. |
| 86 | 36 | Cap. 3 | Destaque | maior | em magnitude | "maior em magnitude" — **FEITO (Lote 1)** |
| 87 | 36 | Cap. 3 | Destaque | nas 1.412 | nos 1412 dias | Ajustar para "nos 1412 dias" |
| 89 | 36 | Cap. 3 | Destaque | diferença se inverte: | isso indica reversão à média da variância realizada | Interpretar: a inversão indica reversão à média da variância realizada |
| 90 | 37 | Cap. 3 | Tachado | Nos anos completos | Até 2025 | Trocar "Nos anos completos" por "Até 2025" |
| 94 | 39 | Cap. 4 | Destaque | um capítulo | uma seção | "uma seção" em vez de "um capítulo" — **ADIADO: depende da D4** |
| 96 | 40 | Cap. 4 | Tachado | em apêndice ( | no | "no apêndice" — **FEITO (Lote 1)** |
| 100 | 40 | Cap. 4 | Tachado | são 17, em | formam | "formam" — **FEITO (Lote 1)** |
| 102 | 41 | Cap. 4 | Destaque | o último dia com retorno de 60 dias à frente. O alvo do Capítulo 5 só precisa de 30 dias à frente, mas uma … | não entendi (e o leitor não deve precisar nunca voltar atrás no texto para entender) | Trecho do último dia com retorno de 60 dias à frente ficou incompreensível; reescrever sem exigir que o leitor volte |
| 106 | 42 | Cap. 4 | Tachado | Um modelo é uma especificação com suas premissas; o método de mínimos quadrados ordinários (MQO) é um estim… | — | Cortar o parágrafo "modelo × método" (MQO é estimador) e a condição de ortogonalidade; ligado a #132 |
| 107 | 42 | Cap. 4 | Tachado | (Capítulo 2) | — | Cortar a referência cruzada (Capítulo 2) — **FEITO (Lote 1)** |
| 108 | 42 | Cap. 4 | Tachado | (Seção 4.6). | — | Cortar a referência cruzada (Seção 4.6) — **FEITO (Lote 1)** |
| 109 | 42 | Cap. 4 | Destaque | referências | para o quê? | "referências" para o quê? Esclarecer |
| 111 | 42 | Cap. 4 | Destaque | A previsão principal | vc está falando da previsão do BVRP e do retorno? | "A previsão principal": é do BVRP ou do retorno? Esclarecer |
| 114 | 43 | Cap. 4 | Tachado | Não se aplica correção de amostra pequena, e o valor- -p vem da distribuição normal. | — | Cortar a nota sobre correção de amostra pequena e valor-p normal |
| 116 | 43 | Cap. 4 | Destaque | o embargo | o que vc quer dizer com isso? | "o embargo": o que quer dizer? Definir |
| 117 | 44 | Cap. 4 | Destaque | O teste de Diebold e Mariano (1995), com perda quadrática, alternativa bilateral e o mesmo erro-padrão HAC,… | desnecessário, mas tudo bem | Teste de Diebold–Mariano para pares: "desnecessário, mas tudo bem"; opcional cortar |
| 119 | 45 | Cap. 5 | Tachado | (Seção 4.2): | — | Cortar a referência cruzada (Seção 4.2) — **FEITO (Lote 1)** |
| 120 | 45 | Cap. 5 | Destaque | da janela de 30 dias que começa no dia seguinte à previsão | nos próximos 30 dias | "nos próximos 30 dias" — **ADIADO: depende da D1 (eq. 5.1)** |
| 123 | 46 | Cap. 5 | Tachado | (Seção 4.3). | — | Cortar a referência cruzada (Seção 4.3) — **FEITO (Lote 1)** |
| 124 | 46 | Cap. 5 | Destaque | Variáveis explicativas | repetição | "Variáveis explicativas": repetição; cortar |
| 125 | 46 | Cap. 5 | Destaque | razão | por que razão em vez de diferença? | Por que "razão" em vez de diferença? Justificar ou trocar |
| 127 | 46 | Cap. 5 | Destaque | mecânica | faz mais sentido modificar o verbo: mecanicamente | "mecânica" → "mecanicamente" — **FEITO (Lote 1)** |
| 128 | 46 | Cap. 5 | Tachado | As variáveis já trazem não linearidade em relação aos retornos diários. A volatilidade histórica é a raiz d… | — | Cortar o parágrafo sobre não linearidade das variáveis construídas |
| 130 | 47 | Cap. 5 | Destaque | razão com a média móvel | ah, em logs! | "Razão com a média móvel… em logs!": explicitar que é em logs |
| 132 | 47 | Cap. 5 | Destaque | MQO; o Ridge (Hoerl e Kennard, 1970) e o LASSO | não são modelos e, sim, métodos | Terminologia: Ridge e LASSO são métodos, não modelos |
| 133 | 47 | Cap. 5 | Destaque | embargada | o que isso quer dizer? | "embargada": o que quer dizer? Definir |
| 134 | 47 | Cap. 5 | Destaque | dobras expansivas | o que seria uma dobra expansiva? | "dobras expansivas": o que é? Definir |
| 137 | 51 | Cap. 5 | Destaque | O placebo não serve como teste de significância: as permutações destroem a autocorrelação dos alvos sobrepo… | aqui eu acho que vc confundiu o leitor | Reescrever o parágrafo do placebo (confundiu o leitor) |
| 138 | 51 | Cap. 5 | Destaque | Ridge supera o MQO | é tipicamente o caso. ridge dá mais estabilidade. | Informativo: Ridge supera MQO é o caso típico (mais estabilidade); sem ação, talvez usar na redação |
| 140 | 51 | Cap. 5 | Destaque | O LASSO vai além | talvez seja aquém. LASSO não funciona bem quando as variáveis explicativas são muito correlacionadas. tende a zerar tudo menos uma. | "O LASSO vai além" → talvez "aquém": LASSO funciona mal com regressores muito correlacionados (zera tudo menos um); corrigir o argumento |
| 141 | 52 | Cap. 5 | Destaque | As árvores oferecem um teste mais flexível, porque não precisam de um corte fixado de antemão: | depende. a arvore prevê a média considerando apenas a partição em que estamos. quando você estima modelos lineares com dois regimes, não é só a média que muda com a partição, mas também os coeficientes de regressão. | Árvore prevê a média condicional à partição; em dois regimes lineares mudam também os coeficientes; reescrever a comparação |
| 142 **(novo)** | 54 | Cap. 6 | Destaque | e não o efeito causal do prêmio sobre o preço | causalidade é uma parada complicada. minha sugestão é simplesmente não falar de causalidade, apenas de previsibilidade ou capacidade/habilidade preditiva. | Cap. 6: não falar de causalidade, só de previsibilidade/capacidade preditiva; buscar "causal" em todo o texto |
| 143 **(novo)** | 55 | Cap. 6 | Tachado | da Seção 4.5 | — | Cortar "da Seção 4.5" — **ADIADO: bootstrap em espera (D7)** |
| 144 **(novo)** | 55 | Cap. 6 | Tachado | um procedimento não paramétrico | — | Cortar "um procedimento não paramétrico" — **ADIADO: bootstrap em espera (D7)** |
| 149 **(novo)** | 55 | Cap. 6 | Destaque | vez | — | Destaque sem comentário em "vez" (Cap. 6, pág. 55); conferir o contexto |
| 152 **(novo)** | 59 | Cap. 6 | Tachado | (Seção 4.7) | — | Cortar a referência cruzada (Seção 4.7) — **FEITO (Lote 1)** |
| 153 **(novo)** | 59 | Cap. 6 | Destaque | (Tabela 6.5). | ver Tabela 6.5 | "ver Tabela 6.5" — **FEITO (Lote 1)** |
| 154 **(novo)** | 59 | Cap. 6 | Destaque | = 30 h | mas aqui é um exemplo | h = 30 é só um exemplo; deixar claro no texto |
| 155 **(novo)** | 59 | Cap. 6 | Destaque | Alta | não entendi essa tabela. deveria ter alfa, beta, delta e gama. | Tabela 6.5 não ficou clara; deveria mostrar α, β, δ e γ |
| 157 **(novo)** | 59 | Cap. 6 | Tachado | Dt = 1 no  por p.p. | — | Cortar "Dt = 1 no … por p.p." — **ADIADO: tabela gerada por script; ver #155** |
| 158 **(novo)** | 59 | Cap. 6 | Destaque | regime d  Valores- -p | precisa dizer ao que se referem esses p-valores, ambos estão como p. o segundo p é de gama ou de beta+gama? | Tabela 6.5: dizer a que se referem os dois valores-p (o segundo é de γ ou de β+γ?) |

### B6 — Notação e LaTeX (8)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 14 | 16 | Cap. 1 | Destaque | Existência do prêmio | tenta colocar um \noindent antes do texto para que não haja espaçamento inicial à direita | Colocar \noindent antes do texto — **FEITO (Lote 1)** |
| 37 | 19 | Cap. 2 | Destaque | E∗ | use \mathbb{E}^* para destacar o operador de esperança. | Usar \mathbb{E}^* para o operador de esperança — **FEITO (Lote 1)** |
| 40 | 19 | Cap. 2 | Nota | — | talvez precise adicionar o pacote amssymb no preâmbulo: \usepackage{amssymb} | Adicionar \usepackage{amssymb} no preâmbulo — **SEM AÇÃO: amssymb já estava no preâmbulo (Lote 1)** |
| 53 | 22 | Cap. 2 | Destaque | E( | \mathbb{E}, talvez precise adicionar o pacote amssymb no preâmbulo | \mathbb{E} (amssymb) — **FEITO (Lote 1)** |
| 103 | 41 | Cap. 4 | Destaque | — s ≤ t − h o embargo: o alvo de olha dias à frente e só é conhecido em s + — s h s h | ficou incompreensível. quem é s? quem é t? quem é h? | Definição do embargo ilegível: quem é s, t, h? |
| 104 | 41 | Cap. 4 | Nota | — | no final, deu para entender, mas fazendo uma ginástica | Notação do embargo ("entendi, mas com ginástica") |
| 105 | 41 | Cap. 4 | Destaque | v | quem é? | Definir "v" |
| 122 | 46 | Cap. 5 | Destaque | s | quem é s? | Quem é s? |

### B7 — Referências e banca (8)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 1 | 3 | Pré-textual | Destaque | Prof. Dr. Instituição | Podemos chamar Fernando Chague, que toca junto com Jefferson o centro de Digital Finance da FGV e, de repente, João Marco Braga de Cunha, que era o portfolio manager da Hasdex e agora está no Itau. | FEITO: banca definida por e-mail (Matheus Carrijo e Fernando Chague; João Marco sai). Folha de aprovação atualizada (D6) |
| 2 **(novo)** | 3 | Pré-textual | Nota | — | Ou Matheus Carrijo que terminou faz pouco seu doutorado com Fernando Chague. Um dos capítulos era sobre bitcoin. | FEITO: Matheus Carrijo (doutorado recente com Chague; um capítulo da tese é sobre Bitcoin) incluído na folha de aprovação |
| 68 | 24 | Cap. 2 | Destaque | (Bollerslev et al., 2011; Pan, 2002; Carr e Wu, 2009; Bollerslev et al., 2009). | em geral, ordena-se cronologicamente as referências. se publicadas no mesmo ano, siga a ordem alfabética.  no caso, precisa ainda averiguar se os autores de Bollerslev et al em 2011 e 2019 são iguais. Se sim, o certo seria escrever (Pan, 2002; Carr e Wu, 2009; Bollerslev et al., 2009, 2011) | Citações em ordem cronológica (mesmo ano: alfabética); verificar se os autores de Bollerslev et al. 2011 e 2019 coincidem. Proposta: (Pan, 2002; Carr e Wu, 2009; Bollerslev et al., 2009, 2011) — **FEITO (Lote 1)** |
| 82 | 35 | Cap. 3 | Destaque | as modas associam-se às médias dos regimes, e a antimoda separa os dois. | em um modelo estocástico de equilíbrio, as modas correspondem aos equilíbrios estáveis e as antimodas aos equilíbrios instáveis.  Cobb, L., Koppstein, P., Chen, N.H., 1983. Estimation and moment recursion relations for multimodal distributions of the exponential family. Journal of the American Statistical Association 78, 124–130. | Acrescentar nota: moda = equilíbrio estável, antimoda = instável (modelo estocástico de equilíbrio); incluir Cobb, Koppstein e Chen (1983, JASA 78:124–130) nas referências — **FEITO (Lote 1)** |
| 148 **(novo)** | 55 | Cap. 6 | Destaque | (Künsch, 1989), | mas essa é uma referência geral de bootstrap estacionário. não trata do problema de viés de atenuação. | Künsch (1989) é referência geral de bootstrap estacionário; não trata de viés de atenuação. Ajustar a citação (D7) |
| 160 | 95 | Referências | Destaque | Almeida, C., Grith, M., Miftachov, R., e Wang, Z. (2024). Risk premia in the Bitcoin market. arXiv:2410.151… | Option-Implied Risk Premia and Cryptocurrency Market Regimes  (with M. Grith, R. Miftachov, and Z. Wang) Journal of Business and Economic Statistics, 2026+ | Atualizar a referência de Almeida et al.: Journal of Business and Economic Statistics, forthcoming (2026+), título "Option-Implied Risk Premia and Cryptocurrency Market Regimes" — **FEITO (Lote 1)** |
| 161 | 95 | Referências | Nota | — | forthcoming | Marcar como "forthcoming" — **FEITO (Lote 1)** |
| EM4 | 3 | Pré-textual | E-mail | Banca | Nomes para banca e incluir na dissertação: Matheus Carrijo e Fernando Chague. | FEITO: folha de aprovação e folha de rosto atualizadas (D6) |

### B8 — Varredura final: primeira pessoa (9)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 19 | 16 | Cap. 1 | Destaque | a dissertação | use a primeira pessoa (eu ou nós). fica mais direto e toma a responsabilidade mais para si | Primeira pessoa (eu ou nós) |
| 69 | 24 | Cap. 2 | Destaque | A dissertação | use a primeira pessoa (eu ou nós) | Primeira pessoa |
| 71 | 26 | Cap. 2 | Tachado | desta dissertação | — | Cortar "desta dissertação" |
| 72 | 26 | Cap. 2 | Tachado | Na leitura desta dissertação | — | Cortar "Na leitura desta dissertação" |
| 93 | 39 | Cap. 4 | Destaque | a dissertação | primeira pessoa | Primeira pessoa |
| 113 | 43 | Cap. 4 | Destaque | A dissertação | — | "A dissertação" → primeira pessoa |
| 118 | 44 | Cap. 4 | Destaque | a dissertação | — | "a dissertação" → primeira pessoa |
| 131 | 47 | Cap. 5 | Destaque | A dissertação | — | "A dissertação" → primeira pessoa |
| 159 **(novo)** | 61 | Cap. 7 | Destaque | A dissertação | — | Primeira pessoa ("A dissertação", início do Cap. 7) |

### B9 — Capítulos finais (Cap. 7 "perdido" e Cap. 8 mais conciso) (2)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| EM1 | 72–75 | Cap. 8 | E-mail | Conclusão | Não li a conclusão, mas pode ser mais concisa. | Enxugar o Cap. 8 (hoje cerca de 1.260 palavras em 4 páginas, com Limitações e nota de reprodutibilidade); fazer depois do B2, porque a mensagem final depende dos resultados |
| EM2 | 61–71 | Cap. 7 | E-mail | Penúltimo capítulo | O penúltimo capítulo ficou meio perdido. | Diagnosticar: o capítulo tem cerca de 3.460 palavras em 11 páginas e nenhuma anotação além de #159. Pedir a ele um exemplo do que está perdido (fio condutor? papel do capítulo?) e propor roteiro; depende de D3 |

### — — Sem ação (3)

| # | Pág. | Onde | Marca | Trecho marcado | Comentário do Marcelo | Ação proposta |
|---|---|---|---|---|---|---|
| 91 | 38 | Cap. 3 | Nota | — | excelente | Comentário positivo ("excelente"); sem ação |
| 139 | 51 | Cap. 5 | Destaque | Com sinal fraco, essa troca compensa | perfeito | Comentário positivo ("perfeito"); sem ação |
| 156 **(novo)** | 59 | Cap. 6 | Destaque | R ˆβh Dt+γhDt  BV ; no de alta: Pt+εt+ + ˆγh; ˆβh ˆβh ˆγh | acho que entendi. | "Acho que entendi" (equação do regime de alta); sem ação |
