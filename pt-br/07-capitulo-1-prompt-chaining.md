# Capítulo 1: Encadeamento de Prompts

# Visão Geral do Padrão de Encadeamento de Prompts

O encadeamento de prompts, às vezes referido como padrão Pipeline, representa um paradigma poderoso para lidar com tarefas intrincadas ao aproveitar grandes modelos de linguagem (LLMs). Em vez de esperar que um LLM resolva um problema complexo em uma única etapa monolítica, o encadeamento de prompts defende uma estratégia de dividir e conquistar. A ideia central é quebrar o problema original e assustador em uma sequência de sub-problemas menores e mais gerenciáveis. Cada sub-problema é abordado individualmente através de um prompt especificamente projetado, e a saída gerada de um prompt é estrategicamente alimentada como entrada no prompt subsequente na cadeia.

Esta técnica de processamento sequencial introduz inerentemente modularidade e clareza na interação com LLMs. Ao decompor uma tarefa complexa, torna-se mais fácil entender e debugar cada etapa individual, tornando o processo geral mais robusto e interpretável. Cada etapa na cadeia pode ser meticulosamente elaborada e otimizada para focar em um aspecto específico do problema maior, levando a saídas mais precisas e focadas.

A saída de uma etapa atuando como entrada para a próxima é crucial. Esta passagem de informação estabelece uma cadeia de dependência, daí o nome, onde o contexto e resultados de operações anteriores guiam o processamento subsequente. Isso permite que o LLM construa sobre seu trabalho anterior, refine sua compreensão, e progressivamente se aproxime da solução desejada.

Além disso, o encadeamento de prompts não é apenas sobre quebrar problemas; ele também permite a integração de conhecimento e ferramentas externas. A cada etapa, o LLM pode ser instruído a interagir com sistemas externos, APIs ou bancos de dados, enriquecendo seu conhecimento e habilidades além de seus dados de treinamento internos. Esta capacidade expande dramaticamente o potencial dos LLMs, permitindo que funcionem não apenas como modelos isolados, mas como componentes integrais de sistemas mais amplos e inteligentes.

A significância do encadeamento de prompts se estende além da simples resolução de problemas. Ele serve como uma técnica fundamental para construir agentes de IA sofisticados. Esses agentes podem utilizar cadeias de prompts para planejar, raciocinar e agir autonomamente em ambientes dinâmicos. Ao estruturar estrategicamente a sequência de prompts, um agente pode se envolver em tarefas que requerem raciocínio multi-etapas, planejamento e tomada de decisões. Tais workflows de agentes podem imitar processos de pensamento humano mais de perto, permitindo interações mais naturais e eficazes com domínios e sistemas complexos.

**Limitações de prompts únicos:** Para tarefas multifacetadas, usar um único prompt complexo para um LLM pode ser ineficiente, causando o modelo a lutar com restrições e instruções, potencialmente levando a negligência de instrução onde partes do prompt são negligenciadas, deriva contextual onde o modelo perde o rastro do contexto inicial, propagação de erro onde erros iniciais se amplificam, prompts que requerem uma janela de contexto mais longa onde o modelo recebe informação insuficiente para responder e alucinação onde a carga cognitiva aumenta a chance de informação incorreta. Por exemplo, uma consulta pedindo para analisar um relatório de pesquisa de mercado, resumir achados, identificar tendências com pontos de dados, e esboçar um email corre o risco de falha pois o modelo pode resumir bem mas falhar em extrair dados ou esboçar um email adequadamente.

**Confiabilidade Aprimorada Através de Decomposição Sequencial:** O encadeamento de prompts aborda esses desafios quebrando a tarefa complexa em um workflow sequencial focado, o que melhora significativamente a confiabilidade e controle. Dado o exemplo acima, uma abordagem de pipeline ou encadeada pode ser descrita da seguinte forma:

1. Prompt Inicial (Resumo): "Resuma os achados-chave do seguinte relatório de pesquisa de mercado: [texto]." O foco único do modelo é resumo, aumentando a precisão desta etapa inicial.
2. Segundo Prompt (Identificação de Tendências): "Usando o resumo, identifique as três principais tendências emergentes e extraia os pontos de dados específicos que suportam cada tendência: [saída da etapa 1]." Este prompt agora é mais restrito e constrói diretamente sobre uma saída validada.
3. Terceiro Prompt (Composição de Email): "Esboce um email conciso para a equipe de marketing que delineia as seguintes tendências e seus dados de suporte: [saída da etapa 2]."

Esta decomposição permite controle mais granular sobre o processo. Cada etapa é mais simples e menos ambígua, o que reduz a carga cognitiva no modelo e leva a uma saída final mais precisa e confiável. Esta modularidade é análoga a um pipeline computacional onde cada função executa uma operação específica antes de passar seu resultado para a próxima. Para garantir uma resposta precisa para cada tarefa específica, o modelo pode ser atribuído um papel distinto em cada estágio. Por exemplo, no cenário dado, o prompt inicial poderia ser designado como "Analista de Mercado", o prompt subsequente como "Analista de Comércio", e o terceiro prompt como "Escritor de Documentação Especialista", e assim por diante.

**O Papel da Saída Estruturada:** A confiabilidade de uma cadeia de prompts é altamente dependente da integridade dos dados passados entre etapas. Se a saída de um prompt é ambígua ou mal formatada, o prompt subsequente pode falhar devido a entrada defeituosa. Para mitigar isso, especificar um formato de saída estruturado, como JSON ou XML, é crucial.

Por exemplo, a saída da etapa de identificação de tendências poderia ser formatada como um objeto JSON:

| `{  "trends": [    {      "trend_name": "Personalização Impulsionada por IA",      "supporting_data": "73% dos consumidores preferem fazer negócios com marcas que usam informação pessoal para tornar suas experiências de compra mais relevantes."    },    {      "trend_name": "Marcas Sustentáveis e Éticas",      "supporting_data": "Vendas de produtos com reivindicações relacionadas a ESG cresceram 28% nos últimos cinco anos, comparado a 20% para produtos sem."    }  ] }` |
| :---- |

Este formato estruturado garante que os dados sejam legíveis por máquina e podem ser precisamente analisados e inseridos no próximo prompt sem ambiguidade. Esta prática minimiza erros que podem surgir da interpretação de linguagem natural e é um componente-chave na construção de sistemas robustos baseados em LLM multi-etapas.

# Aplicações Práticas e Casos de Uso

O encadeamento de prompts é um padrão versátil aplicável em uma ampla gama de cenários ao construir sistemas agênticos. Sua utilidade central reside em quebrar problemas complexos em etapas sequenciais e gerenciáveis. Aqui estão várias aplicações práticas e casos de uso:

**1. Workflows de Processamento de Informação:** Muitas tarefas envolvem processar informação bruta através de múltiplas transformações. Por exemplo, resumir um documento, extrair entidades-chave, e então usar essas entidades para consultar um banco de dados ou gerar um relatório. Uma cadeia de prompts poderia ser:

* Prompt 1: Extrair conteúdo de texto de uma URL ou documento dado.
* Prompt 2: Resumir o texto limpo.
* Prompt 3: Extrair entidades específicas (ex., nomes, datas, localizações) do resumo ou texto original.
* Prompt 4: Usar as entidades para buscar uma base de conhecimento interna.
* Prompt 5: Gerar um relatório final incorporando o resumo, entidades, e resultados da busca.

Esta metodologia é aplicada em domínios como análise automatizada de conteúdo, desenvolvimento de assistentes de pesquisa impulsionados por IA, e geração complexa de relatórios.

**2. Resposta a Consultas Complexas:** Responder perguntas complexas que requerem múltiplas etapas de raciocínio ou recuperação de informação é um caso de uso principal. Por exemplo, "Quais foram as principais causas da queda da bolsa de valores em 1929, e como a política governamental respondeu?"

* Prompt 1: Identificar as sub-perguntas centrais na consulta do usuário (causas da queda, resposta governamental).
* Prompt 2: Pesquisar ou recuperar informação especificamente sobre as causas da queda de 1929.
* Prompt 3: Pesquisar ou recuperar informação especificamente sobre a resposta de política governamental à queda da bolsa de valores de 1929.
* Prompt 4: Sintetizar a informação das etapas 2 e 3 em uma resposta coerente à consulta original.

Esta metodologia de processamento sequencial é integral ao desenvolvimento de sistemas de IA capazes de inferência multi-etapas e síntese de informação. Tais sistemas são necessários quando uma consulta não pode ser respondida de um único ponto de dados, mas em vez disso necessita uma série de etapas lógicas ou a integração de informação de fontes diversas.

Por exemplo, um agente de pesquisa automatizado projetado para gerar um relatório abrangente sobre um tópico específico executa um workflow computacional híbrido. Inicialmente, o sistema recupera numerosos artigos relevantes. A tarefa subsequente de extrair informação-chave de cada artigo pode ser executada concorrentemente para cada fonte. Este estágio é bem adequado para processamento paralelo, onde sub-tarefas independentes são executadas simultaneamente para maximizar eficiência.

No entanto, uma vez que as extrações individuais estão completas, o processo torna-se inerentemente sequencial. O sistema deve primeiro colacionar os dados extraídos, então sintetizá-los em um rascunho coerente, e finalmente revisar e refinar este rascunho para produzir um relatório final. Cada um desses estágios posteriores é logicamente dependente da conclusão bem-sucedida do precedente. É aqui que o encadeamento de prompts é aplicado: os dados colacionados servem como entrada para o prompt de síntese, e o texto sintetizado resultante torna-se a entrada para o prompt de revisão final. Portanto, operações complexas frequentemente combinam processamento paralelo para coleta independente de dados com encadeamento de prompts para as etapas dependentes de síntese e refinamento.

**3. Extração e Transformação de Dados:** A conversão de texto não estruturado em um formato estruturado é tipicamente alcançada através de um processo iterativo, requerendo modificações sequenciais para melhorar a precisão e completude da saída.

* Prompt 1: Tentar extrair campos específicos (ex., nome, endereço, valor) de um documento de fatura.
* Processamento: Verificar se todos os campos requeridos foram extraídos e se atendem aos requisitos de formato.
* Prompt 2 (Condicional): Se campos estão faltando ou malformados, elaborar um novo prompt pedindo ao modelo para especificamente encontrar a informação faltante/malformada, talvez fornecendo contexto da tentativa falhada.
* Processamento: Validar os resultados novamente. Repetir se necessário.
* Saída: Fornecer os dados estruturados extraídos e validados.

Esta metodologia de processamento sequencial é particularmente aplicável à extração e análise de dados de fontes não estruturadas como formulários, faturas ou emails. Por exemplo, resolver problemas complexos de Reconhecimento Óptico de Caracteres (OCR), como processar um formulário PDF, é mais efetivamente tratado através de uma abordagem decomposta e multi-etapas.

Inicialmente, um grande modelo de linguagem é empregado para executar a extração primária de texto da imagem do documento. Seguindo isso, o modelo processa a saída bruta para normalizar os dados, uma etapa onde pode converter texto numérico, como "mil e cinquenta", em seu equivalente numérico, 1050. Um desafio significativo para LLMs é executar cálculos matemáticos precisos. Portanto, em uma etapa subsequente, o sistema pode delegar qualquer operação aritmética requerida para uma ferramenta de calculadora externa. O LLM identifica o cálculo necessário, alimenta os números normalizados para a ferramenta, e então incorpora o resultado preciso. Esta sequência encadeada de extração de texto, normalização de dados, e uso de ferramenta externa alcança um resultado final e preciso que é frequentemente difícil de obter confiavelmente de uma única consulta LLM.

**4. Workflows de Geração de Conteúdo:** A composição de conteúdo complexo é uma tarefa procedural que é tipicamente decomposta em fases distintas, incluindo ideação inicial, delineamento estrutural, rascunho, e revisão subsequente.

* Prompt 1: Gerar 5 ideias de tópicos baseadas no interesse geral do usuário.
* Processamento: Permitir ao usuário selecionar uma ideia ou automaticamente escolher a melhor.
* Prompt 2: Baseado no tópico selecionado, gerar um delineamento detalhado.
* Prompt 3: Escrever uma seção de rascunho baseada no primeiro ponto no delineamento.
* Prompt 4: Escrever uma seção de rascunho baseada no segundo ponto no delineamento, fornecendo a seção anterior para contexto. Continuar isso para todos os pontos do delineamento.
* Prompt 5: Revisar e refinar o rascunho completo para coerência, tom e gramática.

Esta metodologia é empregada para uma gama de tarefas de geração de linguagem natural, incluindo a composição automatizada de narrativas criativas, documentação técnica, e outras formas de conteúdo textual estruturado.

**5. Agentes Conversacionais com Estado:** Embora arquiteturas abrangentes de gerenciamento de estado empreguem métodos mais complexos que vinculação sequencial, o encadeamento de prompts fornece um mecanismo fundamental para preservar continuidade conversacional. Esta técnica mantém contexto construindo cada turno conversacional como um novo prompt que sistematicamente incorpora informação ou entidades extraídas de interações precedentes na sequência de diálogo.

* Prompt 1: Processar Enunciação do Usuário 1, identificar intenção e entidades-chave.
* Processamento: Atualizar estado da conversa com intenção e entidades.
* Prompt 2: Baseado no estado atual, gerar uma resposta e/ou identificar a próxima peça de informação requerida.
* Repetir para turnos subsequentes, com cada nova enunciação do usuário iniciando uma cadeia que aproveita a história conversacional acumulada (estado).

Este princípio é fundamental para o desenvolvimento de agentes conversacionais, permitindo-lhes manter contexto e coerência através de diálogos multi-turnos estendidos. Ao preservar a história conversacional, o sistema pode entender e responder adequadamente a entradas do usuário que dependem de informação previamente trocada.

**6. Geração e Refinamento de Código:** A geração de código funcional é tipicamente um processo multi-estágios, requerendo que um problema seja decomposto em uma sequência de operações lógicas discretas que são executadas progressivamente.

* Prompt 1: Entender a solicitação do usuário para uma função de código. Gerar pseudocódigo ou um delineamento.
* Prompt 2: Escrever o rascunho inicial de código baseado no delineamento.
* Prompt 3: Identificar erros potenciais ou áreas para melhoria no código (talvez usando uma ferramenta de análise estática ou outra chamada LLM).
* Prompt 4: Reescrever ou refinar o código baseado nos problemas identificados.
* Prompt 5: Adicionar documentação ou casos de teste.

Em aplicações como desenvolvimento de software assistido por IA, a utilidade do encadeamento de prompts deriva de sua capacidade de decompor tarefas complexas de codificação em uma série de sub-problemas gerenciáveis. Esta estrutura modular reduz a complexidade operacional para o grande modelo de linguagem a cada etapa. Criticamente, esta abordagem também permite a inserção de lógica determinística entre chamadas de modelo, permitindo processamento intermediário de dados, validação de saída, e ramificação condicional dentro do workflow. Por este método, uma única solicitação multifacetada que poderia de outra forma levar a resultados não confiáveis ou incompletos é convertida em uma sequência estruturada de operações gerenciadas por um framework de execução subjacente.

**7. Raciocínio multimodal e multi-etapas:** Analisar datasets com modalidades diversas necessita quebrar o problema em tarefas menores baseadas em prompts. Por exemplo, interpretar uma imagem que contém uma foto com texto incorporado, rótulos destacando segmentos específicos de texto, e dados tabulares explicando cada rótulo, requer tal abordagem.

* Prompt 1: Extrair e compreender o texto da solicitação de imagem do usuário.
* Prompt 2: Vincular o texto extraído da imagem com seus rótulos correspondentes.
* Prompt 3: Interpretar a informação coletada usando uma tabela para determinar a saída requerida.

# Exemplo de Código Hands-On

Implementar encadeamento de prompts varia de chamadas de função sequenciais diretas dentro de um script até a utilização de frameworks especializados projetados para gerenciar fluxo de controle, estado e integração de componentes. Frameworks como LangChain, LangGraph, Crew AI, e o Google Agent Development Kit (ADK) oferecem ambientes estruturados para construir e executar esses processos multi-etapas, o que é particularmente vantajoso para arquiteturas complexas.

Para o propósito de demonstração, LangChain e LangGraph são escolhas adequadas pois suas APIs centrais são explicitamente projetadas para compor cadeias e grafos de operações. LangChain fornece abstrações fundamentais para sequências lineares, enquanto LangGraph estende essas capacidades para suportar computações stateful e cíclicas, que são necessárias para implementar comportamentos agênticos mais sofisticados. Este exemplo focará em uma sequência linear fundamental.

O seguinte código implementa uma cadeia de prompts de duas etapas que funciona como um pipeline de processamento de dados. O estágio inicial é projetado para analisar texto não estruturado e extrair informação específica. O estágio subsequente então recebe esta saída extraída e a transforma em um formato de dados estruturado.

Para replicar este procedimento, as bibliotecas requeridas devem primeiro ser instaladas. Isso pode ser realizado usando o seguinte comando:

| `pip install langchain langchain-community langchain-openai langgraph` |
| :---- |

Note que langchain-openai pode ser substituído pelo pacote apropriado para um provedor de modelo diferente. Posteriormente, o ambiente de execução deve ser configurado com as credenciais de API necessárias para o provedor de modelo de linguagem selecionado, como OpenAI, Google Gemini, ou Anthropic.

| `import os from langchain_openai import ChatOpenAI from langchain_core.prompts import ChatPromptTemplate from langchain_core.output_parsers import StrOutputParser # Para melhor segurança, carregue variáveis de ambiente de um arquivo .env # from dotenv import load_dotenv # load_dotenv() # Certifique-se de que sua OPENAI_API_KEY está definida no arquivo .env # Inicializar o Modelo de Linguagem (usar ChatOpenAI é recomendado) llm = ChatOpenAI(temperature=0) # --- Prompt 1: Extrair Informação --- prompt_extract = ChatPromptTemplate.from_template(    "Extraia as especificações técnicas do seguinte texto:\n\n{text_input}" ) # --- Prompt 2: Transformar para JSON --- prompt_transform = ChatPromptTemplate.from_template(    "Transforme as seguintes especificações em um objeto JSON com 'cpu', 'memory', e 'storage' como chaves:\n\n{specifications}" ) # --- Construir a Cadeia usando LCEL --- # O StrOutputParser() converte a saída de mensagem do LLM para uma string simples. extraction_chain = prompt_extract | llm | StrOutputParser() # A cadeia completa passa a saída da cadeia de extração para a variável 'specifications' # para o prompt de transformação. full_chain = (    {"specifications": extraction_chain}    | prompt_transform    | llm    | StrOutputParser() ) # --- Executar a Cadeia --- input_text = "O novo modelo de laptop apresenta um processador octa-core de 3,5 GHz, 16GB de RAM e um SSD NVMe de 1TB." # Executar a cadeia com o dicionário de texto de entrada. final_result = full_chain.invoke({"text_input": input_text}) print("\n--- Saída JSON Final ---") print(final_result)` |
| :---- |

Este código Python demonstra como usar a biblioteca LangChain para processar texto. Ele utiliza dois prompts separados: um para extrair especificações técnicas de uma string de entrada e outro para formatar essas especificações em um objeto JSON. O modelo ChatOpenAI é empregado para interações de modelo de linguagem, e o StrOutputParser garante que a saída esteja em um formato de string utilizável. A LangChain Expression Language (LCEL) é usada para encadear elegantemente esses prompts e o modelo de linguagem. A primeira cadeia, extraction_chain, extrai as especificações. A full_chain então pega a saída da extração e a usa como entrada para o prompt de transformação. Um texto de entrada de exemplo descrevendo um laptop é fornecido. A full_chain é invocada com este texto, processando-o através de ambas as etapas. O resultado final, uma string JSON contendo as especificações extraídas e formatadas, é então impresso.

# Engenharia de Contexto e Engenharia de Prompt

A Engenharia de Contexto (veja Fig.1) é a disciplina sistemática de projetar, construir e entregar um ambiente informacional completo para um modelo de IA antes da geração de tokens. Esta metodologia afirma que a qualidade da saída de um modelo é menos dependente da arquitetura do modelo em si e mais da riqueza do contexto fornecido.

![][image1]

Fig.1: A Engenharia de Contexto é a disciplina de construir um ambiente informacional rico e abrangente para uma IA, pois a qualidade deste contexto é um fator principal para permitir desempenho Agêntico avançado.

Ela representa uma evolução significativa da engenharia de prompt tradicional, que se concentra principalmente em otimizar a formulação da consulta imediata do usuário. A Engenharia de Contexto expande este escopo para incluir várias camadas de informação, como o **prompt do sistema**, que é um conjunto fundamental de instruções definindo os parâmetros operacionais da IA—por exemplo, *"Você é um escritor técnico; seu tom deve ser formal e preciso."* O contexto é ainda mais enriquecido com dados externos. Isso inclui documentos recuperados, onde a IA busca ativamente informação de uma base de conhecimento para informar sua resposta, como puxar especificações técnicas para um projeto. Também incorpora saídas de ferramentas, que são os resultados da IA usando uma API externa para obter dados em tempo real, como consultar um calendário para determinar a disponibilidade do usuário. Estes dados explícitos são combinados com dados implícitos críticos, como identidade do usuário, histórico de interação e estado ambiental. O princípio central é que mesmo modelos avançados têm desempenho inferior quando fornecidos com uma visão limitada ou mal construída do ambiente operacional.

Esta prática, portanto, reformula a tarefa de meramente responder uma pergunta para construir um quadro operacional abrangente para o agente. Por exemplo, um agente com engenharia de contexto não apenas responderia a uma consulta, mas primeiro integraria a disponibilidade do calendário do usuário (uma saída de ferramenta), o relacionamento profissional com o destinatário de um email (dados implícitos), e notas de reuniões anteriores (documentos recuperados). Isso permite que o modelo gere saídas que são altamente relevantes, personalizadas e pragmaticamente úteis. O componente de "engenharia" envolve criar pipelines robustos para buscar e transformar estes dados em tempo de execução e estabelecer loops de feedback para melhorar continuamente a qualidade do contexto.

Para implementar isso, sistemas de tuning especializados podem ser usados para automatizar o processo de melhoria em escala. Por exemplo, ferramentas como o otimizador de prompt do Google Vertex AI podem aprimorar o desempenho do modelo avaliando sistematicamente respostas contra um conjunto de entradas de exemplo e métricas de avaliação predefinidas. Esta abordagem é eficaz para adaptar prompts e instruções do sistema através de diferentes modelos sem requerer reescrita manual extensiva. Ao fornecer tal otimizador com prompts de exemplo, instruções do sistema e um template, ele pode refinar programaticamente as entradas contextuais, oferecendo um método estruturado para implementar os loops de feedback requeridos para Engenharia de Contexto sofisticada.

Esta abordagem estruturada é o que diferencia uma ferramenta de IA rudimentar de um sistema mais sofisticado e contextualmente consciente. Ela trata o contexto em si como um componente primário, colocando importância crítica no que o agente sabe, quando ele sabe, e como usa essa informação. A prática garante que o modelo tenha uma compreensão bem-arredondada da intenção, histórico e ambiente atual do usuário. Em última análise, a Engenharia de Contexto é uma metodologia crucial para avançar chatbots sem estado para sistemas altamente capazes e conscientes da situação.

# Em Resumo

**O quê:** Tarefas complexas frequentemente sobrecarregam LLMs quando tratadas dentro de um único prompt, levando a problemas significativos de desempenho. A carga cognitiva no modelo aumenta a probabilidade de erros como negligenciar instruções, perder contexto e gerar informação incorreta. Um prompt monolítico luta para gerenciar múltiplas restrições e etapas de raciocínio sequencial efetivamente. Isso resulta em saídas não confiáveis e imprecisas, pois o LLM falha em abordar todas as facetas da solicitação multifacetada.

**Por quê:** O encadeamento de prompts fornece uma solução padronizada quebrando um problema complexo em uma sequência de sub-tarefas menores e interconectadas. Cada etapa na cadeia usa um prompt focado para executar uma operação específica, melhorando significativamente a confiabilidade e controle. A saída de um prompt é passada como entrada para o próximo, criando um workflow lógico que progressivamente constrói em direção à solução final. Esta estratégia modular de dividir e conquistar torna o processo mais gerenciável, mais fácil de debugar, e permite a integração de ferramentas externas ou formatos de dados estruturados entre etapas. Este padrão é fundamental para desenvolver sistemas Agênticos sofisticados e multi-etapas que podem planejar, raciocinar e executar workflows complexos.

**Regra prática:** Use este padrão quando uma tarefa é muito complexa para um único prompt, envolve múltiplos estágios de processamento distintos, requer interação com ferramentas externas entre etapas, ou ao construir sistemas Agênticos que precisam executar raciocínio multi-etapas e manter estado.

**Resumo visual**

![][image2]

Fig. 2: Padrão de Encadeamento de Prompts: Agentes recebem uma série de prompts do usuário, com a saída de cada agente servindo como entrada para o próximo na cadeia.

# Principais Takeaways

Aqui estão alguns principais takeaways:

* O Encadeamento de Prompts quebra tarefas complexas em uma sequência de etapas menores e focadas. Isso é ocasionalmente conhecido como padrão Pipeline.
* Cada etapa em uma cadeia envolve uma chamada LLM ou lógica de processamento, usando a saída da etapa anterior como entrada.
* Este padrão melhora a confiabilidade e gerenciabilidade de interações complexas com modelos de linguagem.
* Frameworks como LangChain/LangGraph, e Google ADK fornecem ferramentas robustas para definir, gerenciar e executar essas sequências multi-etapas.

# Conclusão

Ao desconstruir problemas complexos em uma sequência de sub-tarefas mais simples e gerenciáveis, o encadeamento de prompts fornece um framework robusto para guiar grandes modelos de linguagem. Esta estratégia de "dividir e conquistar" melhora significativamente a confiabilidade e controle da saída focando o modelo em uma operação específica por vez. Como um padrão fundamental, ele permite o desenvolvimento de agentes de IA sofisticados capazes de raciocínio multi-etapas, integração de ferramentas e gerenciamento de estado. Em última análise, dominar o encadeamento de prompts é crucial para construir sistemas robustos e conscientes do contexto que podem executar workflows intrincados muito além das capacidades de um único prompt.

# Referências

1. Documentação LangChain sobre LCEL: [https://python.langchain.com/v0.2/docs/core\_modules/expression\_language/](https://python.langchain.com/v0.2/docs/core_modules/expression_language/)
2. Documentação LangGraph: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
3. Guia de Engenharia de Prompt - Encadeamento de Prompts: [https://www.promptingguide.ai/techniques/chaining](https://www.promptingguide.ai/techniques/chaining)
4. Documentação da API OpenAI (Conceitos Gerais de Prompting): [https://platform.openai.com/docs/guides/gpt/prompting](https://platform.openai.com/docs/guides/gpt/prompting)
5. Documentação Crew AI (Tarefas e Processos): [https://docs.crewai.com/](https://docs.crewai.com/)
6. Google AI for Developers (Guias de Prompting): [https://cloud.google.com/discover/what-is-prompt-engineering?hl=en](https://cloud.google.com/discover/what-is-prompt-engineering?hl=en)
7. Vertex Prompt Optimizer [https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer)
