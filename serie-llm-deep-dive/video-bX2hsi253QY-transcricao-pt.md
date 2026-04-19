# Transcrição (PT-BR): TurboQuant, KV cache e LLMs locais

**Vídeo:** [`bX2hsi253QY`](https://www.youtube.com/watch?v=bX2hsi253QY)  
**Formato:** Markdown · Quebras de linha do ASR fundidas em parágrafos · Ajustes óbvios de legenda: VRAM, RAM, LM Studio, Qwen, KV cache, embedding, etc.

---

## Abertura

Na última semana, o Google lançou aí um modelo de compressão, um algoritmo de compressão chamado **TurboQuant**. E esse algoritmo vem para revolucionar as LLMs — pelo menos é isso que eles dizem. Nesse vídeo aqui a gente vai entender o que que é o TurboQuant, para que que ele serve e um pouquinho das tretas que estão rolando também. Mas antes da gente entender o que que é o Turbo, vamos pensar um pouco no problema que ele resolve.

---

## Memória da GPU (VRAM) e latência

Normalmente pra gente rodar uma LLM localmente, a gente usa a **VRAM**, tá? A memória da placa de vídeo. Isso vai permitir que os núcleos de processamento da placa de vídeo consigam acessar essa memória de forma muito rápida. Se esse modelo ele não couber 100% nessa memória, a gente vai criar uma latência muito grande, porque ele vai ter que ficar dividindo um pedaço na VRAM, um pedaço na **RAM** e aí vai ter que ficar fazendo trocas e essas trocas são lentas. Então a forma mais performática é fazendo com que o seu modelo caiba 100% na VRAM. Outra coisa que vai ser importante também é a taxa de transferência dessa memória, mas nesse momento aqui pra gente o que vai ser mais importante entender é que o modelo precisa caber 100% na VRAM. Mas isso não é o suficiente.

---

## Demonstração: LM Studio, Qwen e o “tamanho real” na memória

Vamos lá pra tela, para eu te mostrar um exemplo na prática. Eu tô aqui com o meu **LM Studio** aberto e se eu olhar os modelos que eu tenho aqui, um deles vai ser esse **Qwen 3** de 27 bilhões de parâmetros, que ele tá com uma quantização **Q8**. Normalmente o modelo sem quantização, ele vai ter os seus pesos representados em **16 bits**. E aí, se a gente quiser diminuir um pouco o espaço ocupado na memória, a gente quantiza ele — a gente tira um pouco de precisão desses pesos. Então, o Q8, por exemplo, seria um modelo onde a precisão dos pesos foi diminuída de 16 para 8 bits e aí ele ocupa menos espaço. Isso vai ser importante pra gente fazer o modelo caber na memória.

Então aqui, por exemplo, o nosso Qwen 3, 27 bilhões de parâmetros com 8 bits, ele vai ocupar ali mais ou menos **27,5 GB**. Isso quer dizer que ele tecnicamente cabe numa placa de **32 GB** de memória. Só que quando a gente tenta carregar esse modelo aqui, a gente se depara com um outro problema. Primeiro que a gente tem aqui 27,5, mas se eu clico aqui, aqui em cima já aparece para mim que a **memória estimada** vai ser de **31,83 GB**.

Então, pro nosso modelo rodar, não basta que ele caiba na memória. O **contexto** também precisa caber na memória.

---

## O que é contexto

O que que é o contexto? O contexto são todas as mensagens que a gente troca com o modelo — todas as entradas, todas as respostas. Esse conjuntão é o contexto. Então, quanto mais mensagens a gente troca, maior esse contexto, mais espaço ele ocupa na memória. E o tamanho desse espaço que ele ocupa vai depender também do tamanho do modelo, tá? Mas a gente consegue simular aqui pra gente ter uma ideia. Então, vamos pegar como exemplo esse Qwen de 27 bilhões de parâmetros e vamos aumentar o contexto dele aqui pra gente ver o que que acontece aqui na memória, tá? Vou aumentando, ó, e a gente vê que ele vai aumentando junto. E quando eu chego no máximo do contexto, que são **262.000 tokens**, a memória que ele vai ocupar de forma estimada aqui seria **75 GB**.

Então, tecnicamente, sim: o meu modelo lá com 27,5 GB, ele cabe na minha placa de 32 GB, mas com contexto muito pequeno, né? Eu consigo trocar meia dúzia de mensagens aqui e a partir daí ele já não vai mais caber na minha placa de vídeo.

E aí quando você atinge esse limite você tem algumas técnicas também para lidar com isso. Uma seria **resumir** tudo que foi dito ali naquela conversa e aí o modelo passa a ter uma ideia do que que foi falado, mas aí não tem mais o histórico e aí ele começa a alucinar e se perder. Ou a gente vai **movendo essa janela** de contexto e perdendo uma parte do início ou uma parte do final. Bem, existem algumas técnicas para lidar com isso, só que o ideal era a gente ter **mais contexto**, né? Então, uma forma de fazer isso é quantizando um negócio chamado **KV cache**.

---

## Por que o KV cache importa — e como a LLM gera palavra a palavra

O **KV cache** é o grande responsável pra gente precisar de mais memória quando a gente aumenta o contexto. Mas pra gente entender isso, a gente precisa entender um pouquinho de como uma LLM funciona. Se você já usou uma LLM e se você tá vendo esse vídeo, eu acredito que você já tenha usado, né?

A gente já viu que a LLM ela não responde tudo de uma vez pra gente, ela vai botando pecinha por pecinha — o que a gente chama de **tokens**. E pra gente simplificar aqui, a gente vai pensar nos tokens como se fossem **palavras**, tá?

Então, por exemplo, se a LLM vai falar pra gente: *“Eu adoro comer manga com leite”*, ela pode, né? Ela é uma máquina — você não pode fazer isso em casa, né, meu querido? Senão vai passar mal. Aí de onde que tiraram essa ideia, né? Que manga com leite não pode. Ai, meu Deus.

Bem, quando a LLM começa a escrever essa frase, ela não sabe que ela vai escrever essa frase. Ela escreve a primeira palavra, *eu*, né? E aí, baseado nessa primeira palavra, ela calcula ali estatisticamente quais as possibilidades da próxima palavra, como se fosse um **autocomplete** de celular que tenta completar a frase para você. No fundo, a LLM ela funciona mais ou menos desse jeito e aí ela escreve a próxima palavra. Agora, a partir dessas duas palavras, ela decide qual vai ser a próxima e ela vai fazendo isso sucessivamente, tá?

Só que é bem importante entender que a próxima palavra ela vai ser baseada em **todo o conjunto das palavras que vieram antes**. É assim que ela consegue diferenciar. *Eu adoro comer manga com leite* de *minha camisa está com a manga rasgada*. Nos dois casos a gente tem a palavra **manga**, mas nessas frases elas têm significados completamente diferentes. E pra LLM entender que manga aqui em *eu adoro comer manga com leite* é uma fruta, e manga em *minha camisa está com a manga rasgada* não é uma fruta, ela vai levar em consideração **todas as palavras anteriores**. E essa consideração das palavras anteriores é a chamada **atenção**. E foi aí que tudo começou. Foi aí que LLMs começaram quando o Google lançou o paper *Attention Is All You Need*.

**Attention** é exatamente essa técnica de cada palavra, ela meio que “atende” as outras palavras que já vieram antes.

---

## Do texto aos embeddings; Q, K e V

Digamos que agora o contrário: você mandou pra LLM que você adora comer manga com leite — você é doido, sei lá. E aí quando ela olha aquele *eu* ali, ela simplesmente vai transformar isso num **token**, depois num **número**, porque computadores não entendem palavras, eles entendem números. Esse número vai se tornar um vetor de **embedding** e a partir daí ela vai modificar esse vetor de embedding para conseguir entender o significado de cada palavra. E daria para se aprofundar muito aqui, tá? Falar de tokenizer, de embedding de fato, mas vamos tentar simplificar da seguinte forma.

Então, quando ela escreve o *adoro*, ela começa com o número e depois ela modifica esse número baseado no *eu*. E aí agora ela tem o número do *adoro*. Quando ela escreve *comer*, ela vai modificar esse número baseado no *adoro* e baseado no *eu* também que vieram antes. Quando ela escreve *manga*, *comer*, *adoro* e *eu* vão influenciar o cálculo por trás do embedding do *manga*.

*(Referência: canal **3Blue1Brown**, excelente sobre attention.)*

Então, se você tem aqui, ó, a palavra *tower*, ela vai ter uma representação numérica que vai posicionar ela em algum lugar num espaço imaginário. Só que se essa palavra não fosse *tower*, fosse *ei tower* em inglês, seria *torre* e *eel* — ela muda de lugar, ela se ajusta. Pra gente também funciona dessa forma mais ou menos: a gente muda o significado conforme a gente vai combinando as palavras de uma frase.

E essa daqui é a **fórmula da atenção**, tá? Eu não vou te explicar muito bem essa fórmula não. O que eu quero que a gente veja aqui é que tem ali um **Q**, um **K** e um **V**. Esse Q, K e V eles são **query**, **key** e **value**, tá? A **query** ela vai ser relativa à próxima palavra que a gente tá tentando achar — que a LLM tá tentando achar. E a partir daí ela vai calcular a **key** e o **value**. Se a gente olhar aqui o exemplo que eu fiz, essas setinhas, elas sempre vão funcionar dessa forma: a próxima palavra vai ser influenciada por todas as palavras que já passaram, mas ela **não** é influenciada pelas palavras futuras.

---

## Por que existe o KV cache (performance)

Então, uma vez que você calculou aqui a sexta palavra, que é *leite* nessa frase, se você quiser calcular a próxima palavra, a sétima palavra, tecnicamente você teria que calcular novamente a primeira palavra, depois calcular novamente a segunda palavra, terceira palavra e assim por diante. Então isso seria performaticamente **terrível**, né? Se toda vez você tiver que calcular todas as palavras de uma frase, quer dizer que a primeira palavra vai ser calculada relativamente rápido, a segunda um pouco devagar. Agora, a 10ª palavra já vai ser extremamente lenta. 11ª mais lenta ainda e assim por diante.

Então, pra gente resolver o problema de performance, em vez de fazer essa conta aqui toda vez para cada token que já passou, a gente armazena todo esse cálculo que já foi feito no **cache**. Então, quando a gente **prevê** o próximo token, a gente, em vez de calcular tudo, a gente olha o **KV cache**. E é por isso que quando a gente prevê aqui muitos tokens na nossa conversa, a gente **explode** a quantidade de memória que a gente precisa para armazenar esse KV cache.

E o valor de **K** e valor de **V** vai ser **proporcional ao tamanho do modelo**, tá? Então, modelos maiores vão precisar de mais memória para KV; modelos menores, menos memória, tá?

---

## Quantizar o KV no LM Studio (antes do TurboQuant)

Qual que é a forma da gente resolver esse problema? **Quantizando o KV.** Então, se a gente olhar aqui no próprio LM Studio, a gente tem uma opção aqui embaixo, ó, de **KV cache quantization** — **K** e **V**. Se a gente clicar aqui, ó, a gente vê que o padrão é que ele tenha **16 bits**, tanto pro K quanto pro V. Se a gente diminuir isso aqui, ó, olha o que que vai acontecer aqui com a memória, tá? Quando eu diminuir, ó, para **Q8** já diminuiu uns 10 GB ali. Se eu diminuo agora o V para Q8 também, uh, já diminuiu mais 10 GB. Então, a gente passou de 73 GB e aí eu precisaria de uma placa de 80 GB ou 96 GB, que seria o normal que a gente tem no mercado aí, para caber agora em **64 GB**, né? Então, economizou bastante memória. Se a gente quantizar mais aqui, ó, pra **4**, pô, agora já cabe em **48 GB**. Então, olha só: antes a gente precisava de uma placa de 80 GB para rodar. Agora a gente consegue rodar com 48. Placas de 48 ou no caso de Macs que tenham 48 GB de memória unificada são muito mais baratas do que de 80, de 96.

Mas até aí a gente não tá falando ainda do **TurboQuant**, tá? A gente já vai falar dele.

---

## Quantização = menos precisão (analogia dos 558 ml)

Por quê? Você pode pensar: pô, então vou quantizar mais aqui? E como eu expliquei rapidamente, **quantizar** é você pegar um número que você precisava de 16 bits para representar ele e representar com **menos bits** — ou seja, você vai perder precisão. É como se você **arredondasse** um número, né? Mas arredondar números geram problemas.

Então, imagina que você tem um líquido que você tem ali **558 ml**, mas quando você precisa representar ele, você não consegue representar como mililitro. Você tem que representar como **litro**. Aí você consegue representar ele como 0,558 L, né? Mas também não dá: você só tem **duas casas decimais** para representar. Então ele pode virar **0,56 L** ou **0,55 L**, dependendo da forma como você arredonda. Aí você olha ali, tá? Então eu tenho **0,55 L**, eu pego um copo que cabe exatamente 0,55. O que que vai acontecer quando você botar esse líquido nesse copo? Vai **transbordar**, porque na verdade a tua representação era 0,55, mas tinha **558** ali. Então 8 ml transbordaram, né? Então esse é o problema da gente quantizar: a gente perde precisão, a gente começa a errar e quanto mais a gente quantiza, mais erros gera, tá?

Tanto que se a gente olhar aqui, não tem muito menos do que **Q4** aqui, porque Q4 é o limiar de quando a gente começa a errar demais e a coisa se torna basicamente inútil. A gente também vê quantização quando a gente tá baixando o modelo, como eu falei lá. Esse daqui, por exemplo, é o **Qwen 3** de 27 bilhões de parâmetros quantizado **Q8**, né? Com 8 bits de precisão. Se a gente tivesse ele **Q3**, ele seria um modelo muito mais burro do que o Q8. Então, quando a gente quantiza, a gente perde precisão.

---

## O que o vídeo explica sobre o TurboQuant (ângulo, módulo, “polar” + 1 bit)

E aí que entra o **TurboQuant**. Como eu falei quando eu tava usando esse exemplo aqui, cada token ou cada KV, cada conjunto **key–value**, né, ele vai ser um **ponto** no espaço, mais ou menos. Então, num espaço tridimensional como esse aqui, a gente teria três dimensões, tá? Na LLM são muito mais dimensões, mas a gente pode pensar ele como num ponto aqui. E quando a gente quantiza, ele se mexe um pouco, né? e a gente perde a precisão.

O que o TurboQuant faz é: em vez dele ter todos esses números que precisariam ali para representar o KV, ele vai basicamente transformar isso num **ângulo** e num **módulo**. Então, em vez dele ter as coordenadas, ele vai ter um ângulo ali e um tamanho que vai indicar exatamente onde tá aquele ponto ali. E o que eles alegam é que representar o dado dessa forma faz com que você consiga preservar a qualidade do dado mesmo numa quantização mais forte.

Seria como se essa representação fizesse com que o **posicionamento** daquele valor ali fosse **menos afetado** pela quantização, pela falta de precisão. E para ele conseguir isso, ele usa essas duas partes aqui do algoritmo. O primeiro é o **PolarQuant** — é exatamente isso que eu falei de usar o ângulo, né, e a distância para poder chegar no ponto. E o segundo é um **ajuste** que precisa ser feito ali, mas esse ajuste ele tem um custo muito baixo de **só um bit**, mas é o suficiente para corrigir qualquer erro que possa acontecer durante a transformação no PolarQuant ali.

> **Nota:** No paper *TurboQuant* (Google / arXiv) o método é descrito em termos de rotação aleatória, quantização por coordenada e **QJL** no resíduo — o roteiro do vídeo usa “PolarQuant” e ângulo de forma informal; vale contrastar com o artigo para o detalhe matemático.

O que que isso significaria na prática? Significa que se a gente puder armazenar todo o **KV cache** usando **menos bits** sem perder precisão, pô, aí a gente não vai precisar ocupar tanto espaço na memória. A gente poderia ter KV usando **3,5 bits**, por exemplo, sem perder qualidade. Como a gente viu na LM Studio, já dá pra gente usar **Q4**, só que a gente perde **muita** qualidade quando usa. Se a gente conseguisse 4 bits ou 3 bits **sem perder**, seria excelente. Isso significaria na prática a gente poder rodar **modelos maiores** em **hardware mais barato**.

---

## Paper, código e discussões (marketing vs. implementações)

Mas já tem umas **tretas** rolando aí, tá? Nesse momento o TurboQuant ele é um **paper**, ele ainda não foi implementado, ou pelo menos não foi liberado um código da implementação dele oficialmente. Ele é um material de pesquisa e dentro do mundo dos pesquisadores já tá rolando aqui umas tretas — uma galera acusando os outros aí, ó. Então, já tem um pesquisador aqui falando que tem algumas questões técnicas incorretas e comparações que levam a entender algo que não é exatamente como é.

Então, inclusive, ó, aqui no post do próprio **Google Research** é falado num aumento de velocidade de **seis a oito vezes**, sem perda de memória. Só que o que a galera tem visto é, na verdade, o **contrário**. Por conta das transformações que precisam ser feitas, acaba se criando até uma **sobrecarga** ali e fica um pouco mais **lento**, pelo menos pro **pré-processamento**, que é quando a LLM ela precisa calcular todos os KVs do **primeiro prompt**, né? Imagina que você entra com todo o teu código ali, ela precisa transformar aquilo em tokens, depois transformar em KV. E se ela vai armazenar isso no **cache** de uma forma diferente, ela vai precisar transformar nessa forma diferente e isso vai gerar mais computação — ou seja, vai ficar mais lento.

Agora depois sim, na hora do **decode**, na hora de calcular os próximos tokens, que ela vai precisar ler todo esse KV cache — se ele ocupar **menos espaço**, é menos coisa para ler, né? Então aí sim pode ficar de fato um pouco mais rápido. Talvez esse seja o ponto onde eles estejam acusando de **misleading**, né? Porque vai ficar muito mais lento pra depois ficar mais rápido. Mas pode **valer** a pena, né? Às vezes você não conseguia rodar uma LLM na tua máquina e aí com essa técnica você consegue rodar — dá para esperar um pouquinho mais ali o **pré-fill**, né?

E mesmo que a gente não tivesse nenhum ganho de velocidade no decode na parte geração de tokens, ainda assim seria interessante.

Esse daqui é, por exemplo, uma das primeiras implementações feitas pela comunidade aqui no próprio **dia 25**, um dia depois de ter sido anunciada. E aqui ele mostra, ó, que o processamento de prompt, por exemplo, foi **221 tokens por segundo**, enquanto a **turbo 4**, que seria a técnica do TurboQuant com 4 bits, caiu para **sete tokens por segundo**. Então, caiu muito. E a **geração** ela caiu também, ela não aumentou. Agora, a **compressão** do KV realmente foi o esperado. Pode ser que isso aqui seja um problema de **implementação**, pode ser sim, mas o que eu tenho visto aqui é que ninguém teve de fato os ganhos de **seis a oito vezes** de velocidade.

Um desenvolvedor da comunidade ligado ao **MLX** também fez uma implementação, ele fez até antes, fez no dia **24** e ele conseguiu testes interessantes aqui, ó: uma diminuição muito grande no armazenamento aqui do KV cache, né, da quantidade. Então, **4,9 vezes menor** a um TurboQuant **2,5**; **5** e **3,8 vezes menor** do que o 16 bits numa quantização **3,5**. E o teste dele foi pegar um contexto, né, e procurar algo dentro desse contexto — porque normalmente quando você quantiza demais esse KV cache você não consegue encontrar porque começa a perder precisão, você não acha. E aqui ele achou **seis de seis**. Esse aqui não é um teste que, pô, esgotou todas as possibilidades — é um teste pequeno até, mas demonstra o **potencial** da tecnologia.

O problema aqui é o seguinte. Se a gente olhar aqui na implementação dele do TurboQuant no **MLX**, que é o framework da Apple para rodar LLMs, a gente olha aqui embaixo e tem uma **nota**, ó: primeiro dizendo que essa é uma implementação **inicial**, né, que precisa ser melhorada e que ele não conseguiu chegar perto ainda da **velocidade**, né, do resultado de melhoria de velocidade que o pessoal do Google diz que teria com essa técnica. E ele fala aqui, ó, que ele não conseguiu ver nem o **prefill**, que seria o processamento do prompt, nem o **decode**, que seria a inferência mesmo, a geração dos tokens, chegando nem perto ali da melhoria em **oito vezes**, né, em questão de velocidade.

Outro caso interessante é o de um canal *(nome no áudio ilegível na transcrição)* que implementou no **MLX** e passou a ver uma **queda de acurácia** em relação ao modelo **full**, né, com 16 bits, muito grande. Então quando ele quantiza aqui para **Q9**, não com a técnica do TurboQuant, mas só a quantização normal, ele tem uma acurácia de **99,6%** — ou seja, em 500 casos, o modelo Q9 escolheu 498 vezes o mesmo token que o modelo full escolheria. Quando ele cai para **Q4**, aí a acurácia já desce para **97%**. Só que daqui para baixo ele tá usando o TurboQuant e aí a gente vê que a acurácia ela cai mais ainda. Então comparado com o Q4 normal **não houve um ganho** por ter usado o TurboQuant e se eu não me engano ele fez esses testes no **Minimax**. Aqui ele só usou o **PolarQuant**, né? E aqui ele usou o PolarQuant com aquele ajuste lá do **one bit** e **piorou** em vez de melhorar. Deveria melhorar, mas piorou. Com **3 bits** foi pior ainda, chegando a uma acurácia só de **45%**. E pode ser que talvez a **interpretação** dele da implementação não tenha sido correta ou precisa ajustar alguma coisa, porque isso aqui estaria muito longe, né, do que era o esperado.

---

## Fechamento

Então, o que que a gente pode esperar? Bem, nas próximas semanas vão sair mais implementações do TurboQuant, mais testes também. Então, o que a gente pode fazer é ficar de olho e ver se realmente a gente vai conseguir esses resultados de quantização **sem perda** e de **velocidade** também. Seria incrível, mas eu vou te falar que se a gente tivesse só a **quantização sem perda**, com a mesma velocidade, sem cair muito, já seria top demais.

Eu vi algumas pessoas que conseguiram implementar para modelos com **atenção simples**, sem *multi-layer attention* complexo, por exemplo, e tiveram alguns resultados interessantes. Mas é isso: até ficar tudo redondinho para funcionar com **qualquer modelo** que a gente tem, eu acho que ainda vai demorar um pouquinho, tá? Mas a gente pode ficar de olho e se vocês quiserem eu posso trazer aqui, testar algumas das versões que a comunidade já liberou.

Mas eu confesso que eu tô **esperançoso**, **empolgado**, mas também não quero ainda falar: pô, vai mudar tudo, mudou tudo — se der certo vai mudar tudo, tá? Mas não sei, não sei — tô um pouco assim agora, né, meio assim.

Mas de qualquer forma, a gente tá vivendo uma época interessantíssima, principalmente por conta de modelos como **Qwen** que liberou vários modelos aí. Você pode ver até, eu fiz um vídeo aqui sobre o Qwen **0,8** bilhões de parâmetros e **2** bilhões de parâmetros, que, pô, cabem basicamente qualquer aparelho — até aparelho **celular** eles cabem também. E modelos um pouco melhores, na verdade bem incríveis já, que cabem até **16 GB**, como Qwen de **4** bilhões de parâmetros e de **9** bilhões de parâmetros. Cabe nessa aqui, ó, **RTX 5060 Ti**. Em breve vou falar sobre ela aqui no canal por enquanto.

Assiste esses vídeos aqui que você vai ver que a gente tá numa fase onde a gente já tem modelos realmente bons rodando em **hardware acessível**. E deixa aqui nos comentários o que que você acha sobre o TurboQuant. Eu vou ficando por aqui, até o próximo vídeo.

---

## Glossário rápido

| Termo | Observação |
|--------|------------|
| VRAM | Memória da GPU |
| KV cache | Cache das chaves e valores já computados na atenção |
| Quantização | Menos bits para representar números → menos memória, menos precisão |
| Prefill / prompt processing | Fase em que o prompt inicial é processado e o KV é preenchido |
| Decode | Geração token a token após o prefill |
| TurboQuant | Algoritmo em paper; implementações comunitárias em evolução |

---

*Markdown: parágrafos unidos a partir da transcrição automática; nomes próprios e números conforme o áudio (confira no vídeo se precisar de precisão exata).*
