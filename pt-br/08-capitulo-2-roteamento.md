# Capítulo 2: Roteamento

# Visão Geral do Padrão de Roteamento

Embora o processamento sequencial via encadeamento de prompts seja uma técnica fundamental para executar workflows lineares determinísticos com modelos de linguagem, sua aplicabilidade é limitada em cenários que requerem respostas adaptativas. Sistemas agênticos do mundo real frequentemente devem arbitrar entre múltiplas ações potenciais baseadas em fatores contingentes, como o estado do ambiente, entrada do usuário, ou o resultado de uma operação precedente. Esta capacidade para tomada de decisão dinâmica, que governa o fluxo de controle para diferentes funções especializadas, ferramentas ou sub-processos, é alcançada através de um mecanismo conhecido como roteamento.

O roteamento introduz lógica condicional no framework operacional de um agente, permitindo uma mudança de um caminho de execução fixo para um modelo onde o agente avalia dinamicamente critérios específicos para selecionar de um conjunto de ações subsequentes possíveis. Isso permite um comportamento de sistema mais flexível e consciente do contexto.

Por exemplo, um agente projetado para consultas de clientes, quando equipado com uma função de roteamento, pode primeiro classificar uma consulta recebida para determinar a intenção do usuário. Baseado nesta classificação, pode então direcionar a consulta para um agente especializado para resposta direta de perguntas, uma ferramenta de recuperação de banco de dados para informação de conta, ou um procedimento de escalação para questões complexas, em vez de usar um único caminho de resposta predeterminado. Portanto, um agente mais sofisticado usando roteamento poderia:

1. Analisar a consulta do usuário.
2. **Roteirizar** a consulta baseada em sua *intenção*:
   * Se a intenção é "verificar status do pedido", roteirizar para um sub-agente ou cadeia de ferramentas que interage com o banco de dados de pedidos.
   * Se a intenção é "informação do produto", roteirizar para um sub-agente ou cadeia que busca no catálogo de produtos.
   * Se a intenção é "suporte técnico", roteirizar para uma cadeia diferente que acessa guias de solução de problemas ou escala para um humano.
   * Se a intenção não é clara, roteirizar para um sub-agente de esclarecimento ou cadeia de prompts.

O componente central do padrão de Roteamento é um mecanismo que executa a avaliação e direciona o fluxo. Este mecanismo pode ser implementado de várias maneiras:

* **Roteamento Baseado em LLM:** O próprio modelo de linguagem pode ser promptado para analisar a entrada e produzir um identificador específico ou instrução que indica o próximo passo ou destino. Por exemplo, um prompt pode pedir ao LLM para "Analisar a seguinte consulta do usuário e produzir apenas a categoria: 'Status do Pedido', 'Info do Produto', 'Suporte Técnico', ou 'Outro'." O sistema agêntico então lê esta saída e direciona o workflow correspondentemente.
* **Roteamento Baseado em Embeddings:** A consulta de entrada pode ser convertida em um embedding vetorial (veja RAG, Capítulo 14). Este embedding é então comparado com embeddings representando diferentes rotas ou capacidades. A consulta é roteirizada para a rota cujo embedding é mais similar. Isso é útil para roteamento semântico, onde a decisão é baseada no significado da entrada em vez de apenas palavras-chave.
* **Roteamento Baseado em Regras:** Isso envolve usar regras predefinidas ou lógica (ex., declarações if-else, casos switch) baseadas em palavras-chave, padrões ou dados estruturados extraídos da entrada. Isso pode ser mais rápido e determinístico que roteamento baseado em LLM, mas é menos flexível para lidar com entradas matizadas ou novas.
* **Roteamento Baseado em Modelo de Machine Learning**: emprega um modelo discriminativo, como um classificador, que foi especificamente treinado em um pequeno corpus de dados rotulados para executar uma tarefa de roteamento. Embora compartilhe similaridades conceituais com métodos baseados em embeddings, sua característica-chave é o processo de fine-tuning supervisionado, que ajusta os parâmetros do modelo para criar uma função de roteamento especializada. Esta técnica é distinta do roteamento baseado em LLM porque o componente de tomada de decisão não é um modelo generativo executando um prompt no tempo de inferência. Em vez disso, a lógica de roteamento é codificada dentro dos pesos aprendidos do modelo fine-tuned. Embora LLMs possam ser usados em uma etapa de pré-processamento para gerar dados sintéticos para aumentar o conjunto de treinamento, eles não estão envolvidos na decisão de roteamento em tempo real.

Mecanismos de roteamento podem ser implementados em múltiplos pontos dentro do ciclo operacional de um agente. Podem ser aplicados no início para classificar uma tarefa primária, em pontos intermediários dentro de uma cadeia de processamento para determinar uma ação subsequente, ou durante uma sub-rotina para selecionar a ferramenta mais apropriada de um conjunto dado.

Frameworks computacionais como LangChain, LangGraph, e o Google's Agent Developer Kit (ADK) fornecem construtos explícitos para definir e gerenciar tal lógica condicional. Com sua arquitetura de grafo baseada em estado, LangGraph é particularmente bem adequado para cenários de roteamento complexos onde decisões são contingentes sobre o estado acumulado de todo o sistema. Similarmente, o ADK do Google fornece componentes fundamentais para estruturar as capacidades e modelos de interação de um agente, que servem como base para implementar lógica de roteamento. Dentro dos ambientes de execução fornecidos por estes frameworks, desenvolvedores definem os possíveis caminhos operacionais e as funções ou avaliações baseadas em modelo que ditam as transições entre nós no grafo computacional.

A implementação de roteamento permite que um sistema se mova além do processamento sequencial determinístico. Facilita o desenvolvimento de fluxos de execução mais adaptativos que podem responder dinamicamente e apropriadamente a uma gama mais ampla de entradas e mudanças de estado.

# Aplicações Práticas e Casos de Uso

O padrão de roteamento é um mecanismo de controle crítico no design de sistemas agênticos adaptativos, permitindo-lhes alterar dinamicamente seu caminho de execução em resposta a entradas variáveis e estados internos. Sua utilidade abrange múltiplos domínios fornecendo uma camada necessária de lógica condicional.

Na interação humano-computador, como com assistentes virtuais ou tutores dirigidos por IA, o roteamento é empregado para interpretar a intenção do usuário. Uma análise inicial de uma consulta em linguagem natural determina a ação subsequente mais apropriada, seja invocando uma ferramenta específica de recuperação de informação, escalando para um operador humano, ou selecionando o próximo módulo em um currículo baseado no desempenho do usuário. Isso permite que o sistema se mova além de fluxos de diálogo lineares e responda contextualmente.

Dentro de pipelines automatizados de processamento de dados e documentos, o roteamento serve como uma função de classificação e distribuição. Dados recebidos, como emails, tickets de suporte ou payloads de API, são analisados baseados em conteúdo, metadados ou formato. O sistema então direciona cada item para um workflow correspondente, como um processo de ingestão de leads de vendas, uma função específica de transformação de dados para formatos JSON ou CSV, ou um caminho de escalação de questões urgentes.

Em sistemas complexos envolvendo múltiplas ferramentas ou agentes especializados, o roteamento age como um despachante de alto nível. Um sistema de pesquisa composto de agentes distintos para busca, sumarização e análise de informação usaria um roteador para atribuir tarefas ao agente mais adequado baseado no objetivo atual. Similarmente, um assistente de código de IA usa roteamento para identificar a linguagem de programação e a intenção do usuário—para debugar, explicar ou traduzir—antes de passar um snippet de código para a ferramenta especializada correta.

Em última análise, o roteamento fornece a capacidade para arbitragem lógica que é essencial para criar sistemas funcionalmente diversos e conscientes do contexto. Transforma um agente de um executor estático de sequências pré-definidas em um sistema dinâmico que pode tomar decisões sobre o método mais eficaz para realizar uma tarefa sob condições em mudança.

# Exemplo de Código Hands-On (LangChain)

Implementar roteamento em código envolve definir os caminhos possíveis e a lógica que decide qual caminho tomar. Frameworks como LangChain e LangGraph fornecem componentes e estruturas específicas para isso. A estrutura de grafo baseada em estado do LangGraph é particularmente intuitiva para visualizar e implementar lógica de roteamento.

Este código demonstra um sistema simples tipo agente usando LangChain e a IA Generativa do Google. Ele configura um "coordenador" que roteia solicitações do usuário para diferentes manipuladores simulados de "sub-agentes" baseados na intenção da solicitação (reserva, informação, ou não claro). O sistema usa um modelo de linguagem para classificar a solicitação e então a delega para a função manipuladora apropriada, simulando um padrão básico de delegação frequentemente visto em arquiteturas multi-agente.

Primeiro, certifique-se de ter as bibliotecas necessárias instaladas:

```python
pip install langchain langgraph google-cloud-aiplatform langchain-google-genai google-adk deprecated pydantic
```

Você também precisará configurar seu ambiente com sua chave de API para o modelo de linguagem que escolher (ex., OpenAI, Google Gemini, Anthropic).

# Exemplo de Código Hands-On (Google ADK)

O Agent Development Kit (ADK) é um framework para engenharia de sistemas agênticos, fornecendo um ambiente estruturado para definir as capacidades e comportamentos de um agente. Em contraste com arquiteturas baseadas em grafos computacionais explícitos, o roteamento dentro do paradigma ADK é tipicamente implementado definindo um conjunto discreto de "ferramentas" que representam as funções do agente. A seleção da ferramenta apropriada em resposta a uma consulta do usuário é gerenciada pela lógica interna do framework, que aproveita um modelo subjacente para combinar a intenção do usuário com o manipulador funcional correto.

# Em Resumo

**O que**: Sistemas agênticos frequentemente devem responder a uma ampla variedade de entradas e situações que não podem ser tratadas por um único processo linear. Um workflow sequencial simples carece da capacidade de tomar decisões baseadas em contexto. Sem um mecanismo para escolher a ferramenta ou sub-processo correto para uma tarefa específica, o sistema permanece rígido e não adaptativo. Esta limitação torna difícil construir aplicações sofisticadas que podem gerenciar a complexidade e variabilidade de solicitações do usuário do mundo real.

**Por quê**: O padrão de Roteamento fornece uma solução padronizada introduzindo lógica condicional no framework operacional de um agente. Permite que o sistema primeiro analise uma consulta recebida para determinar sua intenção ou natureza. Baseado nesta análise, o agente direciona dinamicamente o fluxo de controle para a ferramenta especializada, função ou sub-agente mais apropriada. Esta decisão pode ser dirigida por vários métodos, incluindo promptar LLMs, aplicar regras predefinidas, ou usar similaridade semântica baseada em embeddings. Em última análise, o roteamento transforma um caminho de execução estático e predeterminado em um workflow flexível e consciente do contexto capaz de selecionar a melhor ação possível.

**Regra de Ouro**: Use o padrão de Roteamento quando um agente deve decidir entre múltiplos workflows distintos, ferramentas ou sub-agentes baseados na entrada do usuário ou no estado atual. É essencial para aplicações que precisam triar ou classificar solicitações recebidas para lidar com diferentes tipos de tarefas, como um bot de suporte ao cliente distinguindo entre consultas de vendas, suporte técnico e questões de gerenciamento de conta.

#### **Resumo Visual:**

![][image1]  
Fig.1: Padrão Router, usando um LLM como Router

# Principais Takeaways

* O roteamento permite que agentes tomem decisões dinâmicas sobre o próximo passo em um workflow baseado em condições.
* Permite que agentes lidem com entradas diversas e adaptem seu comportamento, movendo-se além da execução linear.
* A lógica de roteamento pode ser implementada usando LLMs, sistemas baseados em regras ou similaridade de embeddings.
* Frameworks como LangGraph e Google ADK fornecem maneiras estruturadas de definir e gerenciar roteamento dentro de workflows de agentes, embora com abordagens arquiteturais diferentes.

# Conclusão

O padrão de Roteamento é um passo crítico na construção de sistemas agênticos verdadeiramente dinâmicos e responsivos. Implementando roteamento, nos movemos além de fluxos de execução simples e lineares e capacitamos nossos agentes a tomar decisões inteligentes sobre como processar informação, responder à entrada do usuário e utilizar ferramentas ou sub-agentes disponíveis.

Vimos como o roteamento pode ser aplicado em vários domínios, desde chatbots de atendimento ao cliente até pipelines complexos de processamento de dados. A capacidade de analisar entrada e direcionar condicionalmente o workflow é fundamental para criar agentes que podem lidar com a variabilidade inerente de tarefas do mundo real.

Os exemplos de código usando LangChain e Google ADK demonstram duas abordagens diferentes, mas eficazes, para implementar roteamento. A estrutura baseada em grafo do LangGraph fornece uma maneira visual e explícita de definir estados e transições, tornando-o ideal para workflows complexos de múltiplas etapas com lógica de roteamento intrincada. O Google ADK, por outro lado, frequentemente se concentra em definir capacidades distintas (Ferramentas) e depende da capacidade do framework de rotear solicitações do usuário para o manipulador de ferramenta apropriado, o que pode ser mais simples para agentes com um conjunto bem definido de ações discretas.

Dominar o padrão de Roteamento é essencial para construir agentes que podem navegar inteligentemente diferentes cenários e fornecer respostas ou ações personalizadas baseadas no contexto. É um componente chave na criação de aplicações agênticas versáteis e robustas.

# Referências

1. Documentação LangGraph: [https://www.langchain.com/](https://www.langchain.com/)    
2. Documentação Google Agent Developer Kit: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
