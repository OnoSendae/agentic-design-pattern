# Apêndice C - Visão Geral Rápida de Frameworks Agentivos

# LangChain

LangChain é um framework para desenvolver aplicações alimentadas por LLMs. Sua força central está em sua LangChain Expression Language (LCEL), que permite "conectar" componentes em uma cadeia. Isso cria uma sequência linear clara onde a saída de uma etapa se torna a entrada para a próxima. É construído para workflows que são Grafos Acíclicos Direcionados (DAGs), significando que o processo flui em uma direção sem loops.

Use para:

* RAG Simples: Recuperar um documento, criar um prompt, obter uma resposta de um LLM.
* Resumo: Pegar texto do usuário, alimentá-lo a um prompt de resumo, e retornar a saída.
* Extração: Extrair dados estruturados (como JSON) de um bloco de texto.

Python

```text
# Uma cadeia LCEL simples conceitualmente # (Este não é código executável, apenas ilustra o fluxo) chain = prompt | model | output_parse
```

### LangGraph

LangGraph é uma biblioteca construída sobre o LangChain para lidar com sistemas agentivos mais avançados. Permite definir seu workflow como um grafo com nós (funções ou cadeias LCEL) e arestas (lógica condicional). Sua principal vantagem é a capacidade de criar ciclos, permitindo que a aplicação faça loop, tente novamente, ou chame ferramentas em uma ordem flexível até que uma tarefa seja completa. Ele gerencia explicitamente o estado da aplicação, que é passado entre nós e atualizado ao longo do processo.

Use para:

* Sistemas Multi-agente: Um agente supervisor roteia tarefas para agentes trabalhadores especializados, potencialmente fazendo loop até que o objetivo seja atingido.
* Agentes Planejar-e-Executar: Um agente cria um plano, executa uma etapa, e então faz loop de volta para atualizar o plano baseado no resultado.
* Human-in-the-Loop: O grafo pode esperar entrada humana antes de decidir qual nó ir a seguir.

| Feature | LangChain | LangGraph |
| :---- | :---- | :---- |
| Core Abstraction | Chain (usando LCEL) | Graph of Nodes |
| Workflow Type | Linear (Directed Acyclic Graph) | Cyclical (Graphs with loops) |
| State Management | Generally stateless per run | Explicit and persistent state object |
| Primary Use | Simple, predictable sequences | Complex, dynamic, stateful agents |

### Qual Você Deve Usar?

* Escolha LangChain quando sua aplicação tem um fluxo claro, previsível e linear de etapas. Se você pode definir o processo de A para B para C sem precisar fazer loop de volta, LangChain com LCEL é a ferramenta perfeita.
* Escolha LangGraph quando você precisa que sua aplicação raciocine, planeje ou opere em um loop. Se seu agente precisa usar ferramentas, refletir sobre os resultados, e potencialmente tentar novamente com uma abordagem diferente, você precisa da natureza cíclica e com estado do LangGraph.

Python

```python
# Classe de estado do grafo
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

# Nós
def call_llm_1(state: State):
    """Primeira chamada LLM para gerar piada inicial"""
    msg = llm.invoke(f"Escreva uma piada sobre {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    """Segunda chamada LLM para gerar história"""
    msg = llm.invoke(f"Escreva uma história sobre {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    """Terceira chamada LLM para gerar poema"""
    msg = llm.invoke(f"Escreva um poema sobre {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """Combine a piada e história em uma única saída"""
    combined = f"Aqui está uma história, piada e poema sobre {state['topic']}!\n\n"
    combined += f"HISTÓRIA:\n{state['story']}\n\n"
    combined += f"PIADA:\n{state['joke']}\n\n"
    combined += f"POEMA:\n{state['poem']}"
    return {"combined_output": combined}

# Construir workflow
parallel_builder = StateGraph(State)

# Adicionar nós
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Adicionar arestas para conectar nós
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()

# Mostrar workflow
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# Invocar
state = parallel_workflow.invoke({"topic": "gatos"})
) print(state["combined_output"])
```

Este código define e executa um workflow LangGraph que opera em paralelo. Seu propósito principal é simultaneamente gerar uma piada, uma história e um poema sobre um tópico dado e então combiná-los em uma única saída de texto formatada.

# ADK do Google

O Agent Development Kit do Google, ou ADK, fornece um framework estruturado de alto nível para construir e implantar aplicações compostas de múltiplos agentes de IA interagindo. Ele contrasta com LangChain e LangGraph ao oferecer um sistema mais opinativo e orientado à produção para orquestrar colaboração de agentes, ao invés de fornecer os blocos de construção fundamentais para a lógica interna de um agente.

LangChain opera no nível mais fundamental, oferecendo os componentes e interfaces padronizadas para criar sequências de operações, como chamar um modelo e analisar sua saída. LangGraph estende isso introduzindo um fluxo de controle mais flexível e poderoso; ele trata o workflow de um agente como um grafo com estado. Usando LangGraph, um desenvolvedor define explicitamente nós, que são funções ou ferramentas, e arestas, que ditam o caminho de execução. Esta estrutura de grafo permite raciocínio cíclico complexo onde o sistema pode fazer loop, tentar tarefas novamente, e tomar decisões baseadas em um objeto de estado explicitamente gerenciado que é passado entre nós. Ele dá ao desenvolvedor controle fino sobre o processo de pensamento de um único agente ou a capacidade de construir um sistema multi-agente a partir de primeiros princípios.

O ADK do Google abstrai muito desta construção de grafo de baixo nível. Em vez de pedir ao desenvolvedor para definir cada nó e aresta, ele fornece padrões arquiteturais pré-construídos para interação multi-agente. Por exemplo, ADK tem tipos de agente integrados como SequentialAgent ou ParallelAgent, que gerenciam o fluxo de controle entre diferentes agentes automaticamente. É arquitetado ao redor do conceito de uma "equipe" de agentes, frequentemente com um agente primário delegando tarefas para sub-agentes especializados. Estado e gerenciamento de sessão são tratados mais implicitamente pelo framework, fornecendo uma abordagem mais coesa mas menos granular que a passagem de estado explícita do LangGraph. Portanto, enquanto LangGraph dá as ferramentas detalhadas para projetar a fiação intrincada de um único robô ou uma equipe, o ADK do Google dá uma linha de montagem de fábrica projetada para construir e gerenciar uma frota de robôs que já sabem como trabalhar juntos.

Python

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

dice_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    name="question_answer_agent",
    description="Um agente assistente útil que pode responder perguntas.",
    instruction="""Responda à consulta usando busca do google""",
    tools=[google_search],
)
```

Este código cria um agente aumentado por busca. Quando este agente recebe uma pergunta, ele não apenas confiará em seu conhecimento pré-existente. Em vez disso, seguindo suas instruções, ele usará a ferramenta Google Search para encontrar informação relevante e em tempo real da web e então usar essa informação para construir sua resposta.

# Crew.AI

CrewAI oferece um framework de orquestração para construir sistemas multi-agente focando em papéis colaborativos e processos estruturados. Ele opera em um nível mais alto de abstração que toolkits fundamentais, fornecendo um modelo conceitual que espelha uma equipe humana. Em vez de definir o fluxo granular de lógica como um grafo, o desenvolvedor define os atores e suas atribuições, e CrewAI gerencia sua interação.

Os componentes centrais deste framework são Agentes, Tarefas e a Crew. Um Agente é definido não apenas por sua função mas por uma persona, incluindo um papel específico, um objetivo e uma história de fundo, que guia seu comportamento e estilo de comunicação. Uma Tarefa é uma unidade discreta de trabalho com uma descrição clara e saída esperada, atribuída a um Agente específico. A Crew é a unidade coesa que contém os Agentes e a lista de Tarefas, e executa um Processo predefinido. Este processo dita o workflow, que é tipicamente sequencial, onde a saída de uma tarefa se torna a entrada para a próxima na linha, ou hierárquico, onde um agente tipo gerente delega tarefas e coordena o workflow entre outros agentes.

Quando comparado a outros frameworks, CrewAI ocupa uma posição distinta. Ele se afasta do gerenciamento de estado e fluxo de controle explícito de baixo nível do LangGraph, onde um desenvolvedor conecta cada nó e aresta condicional. Em vez de construir uma máquina de estado, o desenvolvedor projeta uma carta de equipe. Enquanto o ADK do Google fornece uma plataforma abrangente e orientada à produção para todo o ciclo de vida do agente, CrewAI se concentra especificamente na lógica de colaboração de agentes e para simular uma equipe de especialistas.

Python

```python
@crew
def crew(self) -> Crew:
    """Cria a equipe de pesquisa"""
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        verbose=True,
    )
```

Este código configura um workflow sequencial para uma equipe de agentes de IA, onde eles abordam uma lista de tarefas em uma ordem específica, com logging detalhado habilitado para monitorar seu progresso.

# Outros frameworks de desenvolvimento de agentes

**Microsoft AutoGen**: AutoGen é um framework centrado em orquestrar múltiplos agentes que resolvem tarefas através de conversação. Sua arquitetura permite que agentes com capacidades distintas interajam, permitindo decomposição complexa de problemas e resolução colaborativa. A principal vantagem do AutoGen é sua abordagem flexível e dirigida por conversação que suporta interações multi-agente dinâmicas e complexas. No entanto, este paradigma conversacional pode levar a caminhos de execução menos previsíveis e pode requerer engenharia de prompt sofisticada para garantir que tarefas convirjam eficientemente.

**LlamaIndex**: LlamaIndex é fundamentalmente um framework de dados projetado para conectar modelos de linguagem grandes com fontes de dados externas e privadas. Ele se destaca em criar pipelines sofisticados de ingestão e recuperação de dados, que são essenciais para construir agentes conhecedores que podem executar RAG. Enquanto suas capacidades de indexação e consulta de dados são excepcionalmente poderosas para criar agentes conscientes do contexto, suas ferramentas nativas para fluxo de controle agentivo complexo e orquestração multi-agente são menos desenvolvidas comparadas a frameworks agentivos-first. LlamaIndex é ótimo quando o desafio técnico central é recuperação e síntese de dados.

**Haystack**: Haystack é um framework open-source projetado para construir sistemas de busca escaláveis e prontos para produção alimentados por modelos de linguagem. Sua arquitetura é composta de nós modulares e interoperáveis que formam pipelines para recuperação de documentos, resposta a perguntas e resumo. A principal força do Haystack é seu foco em performance e escalabilidade para tarefas de recuperação de informação em larga escala, tornando-o adequado para aplicações de nível empresarial. Uma troca potencial é que seu design, otimizado para pipelines de busca, pode ser mais rígido para implementar comportamentos agentivos altamente dinâmicos e criativos.

**MetaGPT**: MetaGPT implementa um sistema multi-agente atribuindo papéis e tarefas baseados em um conjunto predefinido de Procedimentos Operacionais Padrão (SOPs). Este framework estrutura colaboração de agentes para imitar uma empresa de desenvolvimento de software, com agentes assumindo papéis como gerentes de produto ou engenheiros para completar tarefas complexas. Esta abordagem dirigida por SOP resulta em saídas altamente estruturadas e coerentes, que é uma vantagem significativa para domínios especializados como geração de código. A principal limitação do framework é seu alto grau de especialização, tornando-o menos adaptável para tarefas agentivas de propósito geral fora de seu design central.

**SuperAGI**: SuperAGI é um framework open-source projetado para fornecer um sistema completo de gerenciamento de ciclo de vida para agentes autônomos. Ele inclui recursos para provisionamento de agentes, monitoramento e uma interface gráfica, visando aprimorar a confiabilidade da execução de agentes. O benefício-chave é seu foco em prontidão para produção, com mecanismos integrados para lidar com modos de falha comuns como looping e para fornecer observabilidade no desempenho de agentes. Uma desvantagem potencial é que sua abordagem de plataforma abrangente pode introduzir mais complexidade e overhead que um framework mais leve baseado em biblioteca.

**Semantic Kernel**: Desenvolvido pela Microsoft, Semantic Kernel é um SDK que integra modelos de linguagem grandes com código de programação convencional através de um sistema de "plugins" e "planners." Ele permite que um LLM invoque funções nativas e orquestre workflows, efetivamente tratando o modelo como um motor de raciocínio dentro de uma aplicação de software maior. Sua principal força é sua integração perfeita com codebases empresariais existentes, particularmente em ambientes .NET e Python. O overhead conceitual de sua arquitetura de plugin e planner pode apresentar uma curva de aprendizado mais íngreme comparada a frameworks de agentes mais diretos.

**Strands Agents**: Um SDK leve e flexível da AWS que usa uma abordagem dirigida por modelo para construir e executar agentes de IA. É projetado para ser simples e escalável, suportando tudo desde assistentes conversacionais básicos até sistemas autônomos multi-agente complexos. O framework é agnóstico a modelos, oferecendo suporte amplo para vários provedores de LLM, e inclui integração nativa com o MCP para fácil acesso a ferramentas externas. Sua principal vantagem é sua simplicidade e flexibilidade, com um loop de agente customizável que é fácil de começar. Uma troca potencial é que seu design leve significa que desenvolvedores podem precisar construir mais da infraestrutura operacional circundante, como sistemas avançados de monitoramento ou gerenciamento de ciclo de vida, que frameworks mais abrangentes podem fornecer prontos para uso.

# Conclusão

A paisagem de frameworks agentivos oferece um espectro diverso de ferramentas, desde bibliotecas de baixo nível para definir lógica de agentes até plataformas de alto nível para orquestrar colaboração multi-agente. No nível fundamental, LangChain habilita workflows lineares simples, enquanto LangGraph introduz grafos cíclicos com estado para raciocínio mais complexo. Frameworks de nível mais alto como CrewAI e ADK do Google mudam o foco para orquestrar equipes de agentes com papéis predefinidos, enquanto outros como LlamaIndex se especializam em aplicações intensivas em dados. Esta variedade apresenta aos desenvolvedores uma troca central entre o controle granular de sistemas baseados em grafo e o desenvolvimento simplificado de plataformas mais opinativas. Consequentemente, selecionar o framework certo depende de se a aplicação requer uma sequência simples, um loop de raciocínio dinâmico, ou uma equipe gerenciada de especialistas. Em última análise, este ecossistema em evolução capacita desenvolvedores a construir sistemas de IA cada vez mais sofisticados escolhendo o nível preciso de abstração que seu projeto demanda.

# Referências

1. LangChain, [https://www.langchain.com/](https://www.langchain.com/)
2. LangGraph, [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)
3. Google ADK, [https://developers.google.com/adk](https://developers.google.com/adk)
4. CrewAI, [https://www.crewai.com/](https://www.crewai.com/)
5. Microsoft AutoGen, [https://microsoft.github.io/autogen/](https://microsoft.github.io/autogen/)
6. LlamaIndex, [https://www.llamaindex.ai/](https://www.llamaindex.ai/)
7. Haystack, [https://haystack.deepset.ai/](https://haystack.deepset.ai/)
8. MetaGPT, [https://github.com/geekan/MetaGPT](https://github.com/geekan/MetaGPT)
9. SuperAGI, [https://superagi.com/](https://superagi.com/)
10. Semantic Kernel, [https://github.com/microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel)
11. Strands Agents, [https://github.com/aws/strands-agents](https://github.com/aws/strands-agents)
