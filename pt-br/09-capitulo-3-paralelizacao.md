# Capítulo 3: Paralelização

# Visão Geral do Padrão de Paralelização

Nos capítulos anteriores, exploramos o Encadeamento de Prompts para workflows sequenciais e o Roteamento para tomada de decisão dinâmica e transições entre diferentes caminhos. Embora esses padrões sejam essenciais, muitas tarefas agênticas complexas envolvem múltiplas sub-tarefas que podem ser executadas *simultaneamente* em vez de uma após a outra. É aqui que o padrão de **Paralelização** se torna crucial.

A paralelização envolve executar múltiplos componentes, como chamadas LLM, usos de ferramentas, ou até mesmo sub-agentes inteiros, concorrentemente (veja Fig.1). Em vez de esperar uma etapa completar antes de iniciar a próxima, a execução paralela permite que tarefas independentes sejam executadas ao mesmo tempo, reduzindo significativamente o tempo total de execução para tarefas que podem ser divididas em partes independentes.

Considere um agente projetado para pesquisar um tópico e resumir suas descobertas. Uma abordagem sequencial poderia:

1. Buscar pela Fonte A.
2. Resumir a Fonte A.
3. Buscar pela Fonte B.
4. Resumir a Fonte B.
5. Sintetizar uma resposta final dos resumos A e B.

Uma abordagem paralela poderia em vez disso:

1. Buscar pela Fonte A *e* Buscar pela Fonte B simultaneamente.
2. Uma vez que ambas as buscas estejam completas, Resumir a Fonte A *e* Resumir a Fonte B simultaneamente.
3. Sintetizar uma resposta final dos resumos A e B (esta etapa é tipicamente sequencial, aguardando que as etapas paralelas terminem).

A ideia central é identificar partes do workflow que não dependem da saída de outras partes e executá-las em paralelo. Isso é particularmente eficaz quando lidando com serviços externos (como APIs ou bancos de dados) que têm latência, pois você pode emitir múltiplas solicitações concorrentemente.

Implementar paralelização frequentemente requer frameworks que suportam execução assíncrona ou multi-threading/multi-processamento. Frameworks agênticos modernos são projetados com operações assíncronas em mente, permitindo que você defina facilmente etapas que podem ser executadas em paralelo.

![][image1]

Fig.1. Exemplo de paralelização com sub-agentes

# Aplicações Práticas e Casos de Uso

O padrão de paralelização é fundamental para otimizar sistemas agênticos que precisam processar múltiplas tarefas independentes simultaneamente. Sua aplicação abrange diversos cenários:

**1. Coleta de Dados e Pesquisa:** Quando um agente precisa reunir informações de múltiplas fontes independentes.

* **Pesquisa na Web:** Buscar informações sobre um tópico em diferentes sites simultaneamente.
* **Análise de Documentos:** Processar múltiplos documentos em paralelo para extrair insights.
* **Validação de Dados:** Verificar informações contra diferentes bancos de dados simultaneamente.

**2. Análise de Dados e Relatórios:** Sistemas que precisam processar grandes volumes de dados frequentemente se beneficiam da paralelização para acelerar o tempo de processamento.

* **Processamento de Documentos:** Analisar múltiplos documentos simultaneamente para extrair informações específicas, como sentimentos, entidades nomeadas, ou classificações.
* **Geração de Relatórios:** Criar diferentes seções de um relatório em paralelo (resumo executivo, análise de dados, recomendações) e depois combiná-las.
* **Validação de Dados:** Verificar múltiplos campos ou registros simultaneamente para identificar inconsistências ou erros.

**3. Sistemas de Recomendação:** Plataformas que fornecem recomendações personalizadas podem usar paralelização para processar diferentes tipos de dados simultaneamente.

* **Análise de Perfil:** Analisar histórico de compras, preferências, e comportamento do usuário em paralelo.
* **Processamento de Conteúdo:** Analisar descrições de produtos, reviews, e categorias simultaneamente.
* **Cálculo de Similaridade:** Computar similaridades entre usuários e itens em paralelo para acelerar o processo de recomendação.

**4. Sistemas de Monitoramento e Alertas:** Sistemas que monitoram múltiplas fontes de dados podem usar paralelização para processar diferentes streams simultaneamente.

* **Monitoramento de Logs:** Analisar logs de diferentes serviços ou aplicações simultaneamente para detectar anomalias.
* **Monitoramento de Performance:** Coletar métricas de diferentes componentes do sistema em paralelo.
* **Detecção de Eventos:** Processar múltiplos tipos de eventos (segurança, performance, negócio) simultaneamente para gerar alertas apropriados.

**5. Geração de Conteúdo com Múltiplos Componentes:** Gerar diferentes partes de uma peça complexa de conteúdo em paralelo.

* **Caso de Uso:** Um agente criando um email de marketing.
* **Tarefas Paralelas:** Gerar uma linha de assunto, rascunhar o corpo do email, encontrar uma imagem relevante, e criar o texto do botão de call-to-action simultaneamente.
* **Benefício:** Monta o email final de forma mais eficiente.

**6. Validação e Verificação:** Realizar múltiplas verificações ou validações independentes concorrentemente.

* **Caso de Uso:** Um agente verificando entrada do usuário.
* **Tarefas Paralelas:** Verificar formato do email, validar número de telefone, verificar endereço contra um banco de dados, e verificar profanidade simultaneamente.
* **Benefício:** Fornece feedback mais rápido sobre a validade da entrada.

**7. Processamento Multi-Modal:** Processar diferentes modalidades (texto, imagem, áudio) da mesma entrada concorrentemente.

* **Caso de Uso:** Um agente analisando uma postagem de mídia social com texto e uma imagem.
* **Tarefas Paralelas:** Analisar o texto para sentimento e palavras-chave *e* analisar a imagem para objetos e descrição da cena simultaneamente.
* **Benefício:** Integra insights de diferentes modalidades mais rapidamente.

**8. Teste A/B ou Geração de Múltiplas Opções:** Gerar múltiplas variações de uma resposta ou saída em paralelo para selecionar a melhor.

* **Caso de Uso:** Um agente gerando diferentes opções de texto criativo.
* **Tarefas Paralelas:** Gerar três manchetes diferentes para um artigo simultaneamente usando prompts ou modelos ligeiramente diferentes.
* **Benefício:** Permite comparação rápida e seleção da melhor opção.

A paralelização é uma técnica de otimização fundamental no design agêntico, permitindo aos desenvolvedores construir aplicações mais performáticas e responsivas aproveitando a execução concorrente para tarefas independentes.

# Exemplo de Código Hands-On (LangChain)

A execução paralela dentro do framework LangChain é facilitada pela LangChain Expression Language (LCEL). O método principal envolve estruturar múltiplos componentes executáveis dentro de um constructo de dicionário ou lista. Quando esta coleção é passada como entrada para um componente subsequente na cadeia, o runtime LCEL executa os executáveis contidos concorrentemente.

No contexto do LangGraph, este princípio é aplicado à topologia do grafo. Workflows paralelos são definidos arquitetando o grafo de tal forma que múltiplos nós, sem dependências sequenciais diretas, podem ser iniciados a partir de um único nó comum. Esses caminhos paralelos executam independentemente antes que seus resultados possam ser agregados em um ponto de convergência subsequente no grafo.

A seguinte implementação demonstra um workflow de processamento paralelo construído com o framework LangChain. Este workflow é projetado para executar duas operações independentes concorrentemente em resposta a uma única consulta do usuário. Esses processos paralelos são instanciados como cadeias ou funções distintas, e suas respectivas saídas são subsequentemente agregadas em um resultado unificado.

Os pré-requisitos para esta implementação incluem a instalação dos pacotes Python necessários, como langchain, langchain-community, e uma biblioteca de provedor de modelo como langchain-openai. Além disso, uma chave API válida para o modelo de linguagem escolhido deve ser configurada no ambiente local para autenticação.

```python
import os
import asyncio
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- Configuração ---
# Certifique-se de que sua variável de ambiente de chave API está definida (ex., OPENAI_API_KEY)
try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
except Exception as e:
    print(f"Erro ao inicializar modelo de linguagem: {e}")
    llm = None

# --- Definir Cadeias Independentes ---
# Essas três cadeias representam tarefas distintas que podem ser executadas em paralelo.
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Resuma o seguinte tópico de forma concisa:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Gere três perguntas interessantes sobre o seguinte tópico:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identifique 5-10 termos-chave do seguinte tópico, separados por vírgulas:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

# --- Construir a Cadeia Paralela + Síntese ---
# 1. Definir o bloco de tarefas para executar em paralelo. Os resultados destas,
# junto com o tópico original, serão alimentados no próximo passo.
map_chain = RunnableParallel({
    "summary": summarize_chain,
    "questions": questions_chain,
    "key_terms": terms_chain,
    "topic": RunnablePassthrough(),  # Passar o tópico original através
})

# 2. Definir o prompt de síntese final que combinará os resultados paralelos.
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Com base nas seguintes informações:
Resumo: {summary}
Perguntas Relacionadas: {questions}
Termos-Chave: {key_terms}

Sintetize uma resposta abrangente."""),
    ("user", "Tópico original: {topic}")
])

# 3. Construir a cadeia completa canalizando os resultados paralelos diretamente
# no prompt de síntese, seguido pelo LLM e parser de saída.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- Executar a Cadeia ---
async def run_parallel_example(topic: str) -> None:
    """
    Invoca assincronamente a cadeia de processamento paralelo com um tópico específico
    e imprime o resultado sintetizado.
    
    Args:
        topic: O tópico de entrada a ser processado pelas cadeias LangChain.
    """
    if not llm:
        print("LLM não inicializado. Não é possível executar o exemplo.")
        return
    
    print(f"\n--- Executando Exemplo Paralelo LangChain para Tópico: '{topic}' ---")
    
    try:
        # A entrada para `ainvoke` é a única string 'topic',
        # então passada para cada executável no `map_chain`.
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- Resposta Final ---")
        print(response)
    except Exception as e:
        print(f"\nOcorreu um erro durante a execução da cadeia: {e}")

if __name__ == "__main__":
    test_topic = "A história da exploração espacial"
    # Em Python 3.7+, asyncio.run é a maneira padrão de executar uma função async.
    asyncio.run(run_parallel_example(test_topic))
```

O código Python fornecido implementa uma aplicação LangChain projetada para processar um tópico dado eficientemente aproveitando a execução paralela. Note que asyncio fornece concorrência, não paralelismo. Ele consegue isso em uma única thread usando um loop de eventos que muda inteligentemente entre tarefas quando uma está inativa (ex., aguardando uma solicitação de rede). Isso cria o efeito de múltiplas tarefas progredindo de uma vez, mas o código em si ainda está sendo executado por apenas uma thread, limitado pelo Global Interpreter Lock (GIL) do Python.

O código começa importando módulos essenciais do langchain_openai e langchain_core, incluindo componentes para modelos de linguagem, prompts, parsing de saída, e estruturas executáveis. O código tenta inicializar uma instância ChatOpenAI, especificamente usando o modelo "gpt-4o-mini", com uma temperatura especificada para controlar a criatividade. Um bloco try-except é usado para robustez durante a inicialização do modelo de linguagem. Três "cadeias" LangChain independentes são então definidas, cada uma projetada para realizar uma tarefa distinta no tópico de entrada. A primeira cadeia é para resumir o tópico de forma concisa, usando uma mensagem de sistema e uma mensagem de usuário contendo o placeholder do tópico. A segunda cadeia é configurada para gerar três perguntas interessantes relacionadas ao tópico. A terceira cadeia é configurada para identificar entre 5 e 10 termos-chave do tópico de entrada, solicitando que sejam separados por vírgulas. Cada uma dessas cadeias independentes consiste em um ChatPromptTemplate adaptado à sua tarefa específica, seguido pelo modelo de linguagem inicializado e um StrOutputParser para formatar a saída como string.

Um bloco RunnableParallel é então construído para agrupar essas três cadeias, permitindo que executem simultaneamente. Este executável paralelo também inclui um RunnablePassthrough para garantir que o tópico de entrada original esteja disponível para etapas subsequentes. Um ChatPromptTemplate separado é definido para a etapa de síntese final, tomando o resumo, perguntas, termos-chave, e o tópico original como entrada para gerar uma resposta abrangente. A cadeia completa de processamento ponta a ponta, nomeada full_parallel_chain, é criada sequenciando o map_chain (o bloco paralelo) no prompt de síntese, seguido pelo modelo de linguagem e o parser de saída. Uma função assíncrona run_parallel_example é fornecida para demonstrar como invocar este full_parallel_chain. Esta função toma o tópico como entrada e usa invoke para executar a cadeia assíncrona. Finalmente, o bloco padrão Python if __name__ == "__main__": mostra como executar o run_parallel_example com um tópico de exemplo, neste caso, "A história da exploração espacial", usando asyncio.run para gerenciar a execução assíncrona.

Em essência, este código configura um workflow onde múltiplas chamadas LLM (para resumir, perguntas, e termos) acontecem ao mesmo tempo para um tópico dado, e seus resultados são então combinados por uma chamada LLM final. Isso mostra a ideia central da paralelização em um workflow agêntico usando LangChain.

# Exemplo de Código Hands-On (Google ADK)

Ok, vamos agora voltar nossa atenção para um exemplo concreto ilustrando esses conceitos dentro do framework Google ADK. Vamos examinar como as primitivas ADK, como ParallelAgent e SequentialAgent, podem ser aplicadas para construir um fluxo de agente que aproveita a execução concorrente para eficiência melhorada.

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

GEMINI_MODEL = "gemini-2.0-flash"

# --- 1. Definir Sub-Agentes Pesquisadores (para executar em paralelo) ---

# Pesquisador 1: Energia Renovável
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""Você é um Assistente de Pesquisa em IA especializado em energia.
Pesquise os últimos avanços em 'fontes de energia renovável'.
Use a ferramenta Google Search fornecida.
Resuma seus principais achados de forma concisa (1-2 frases).
Saída *apenas* o resumo.""",
    description="Pesquisa fontes de energia renovável.",
    tools=[google_search],
    # Armazenar resultado no estado para o agente de fusão
    output_key="renewable_energy_result"
)

# Pesquisador 2: Veículos Elétricos
researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction="""Você é um Assistente de Pesquisa em IA especializado em transporte.
Pesquise os últimos desenvolvimentos em 'tecnologia de veículos elétricos'.
Use a ferramenta Google Search fornecida.
Resuma seus principais achados de forma concisa (1-2 frases).
Saída *apenas* o resumo.""",
    description="Pesquisa tecnologia de veículos elétricos.",
    tools=[google_search],
    # Armazenar resultado no estado para o agente de fusão
    output_key="ev_technology_result"
)

# Pesquisador 3: Captura de Carbono
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction="""Você é um Assistente de Pesquisa em IA especializado em soluções climáticas.
Pesquise o estado atual de 'métodos de captura de carbono'.
Use a ferramenta Google Search fornecida.
Resuma seus principais achados de forma concisa (1-2 frases).
Saída *apenas* o resumo.""",
    description="Pesquisa métodos de captura de carbono.",
    tools=[google_search],
    # Armazenar resultado no estado para o agente de fusão
    output_key="carbon_capture_result"
)

# --- 2. Criar o ParallelAgent (Executa pesquisadores concorrentemente) ---
# Este agente orquestra a execução concorrente dos pesquisadores.
# Ele termina uma vez que todos os pesquisadores tenham completado e armazenado seus resultados no estado.
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Executa múltiplos agentes de pesquisa em paralelo para reunir informações."
)

# --- 3. Definir o Agente de Fusão (Executa *após* os agentes paralelos) ---
# Este agente toma os resultados armazenados no estado da sessão pelos agentes paralelos
# e os sintetiza em uma única resposta estruturada com atribuições.
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,  # Ou potencialmente um modelo mais poderoso se necessário para síntese
    instruction="""Você é um Assistente de IA responsável por combinar achados de pesquisa em um relatório estruturado.
Sua tarefa principal é sintetizar os seguintes resumos de pesquisa, atribuindo claramente os achados às suas áreas de origem.
Estruture sua resposta usando cabeçalhos para cada tópico.
Certifique-se de que o relatório é coerente e integra os pontos-chave suavemente.

**Crucialmente: Sua resposta inteira DEVE ser fundamentada *exclusivamente* nas informações fornecidas nos 'Resumos de Entrada' abaixo. NÃO adicione nenhum conhecimento externo, fatos, ou detalhes não presentes nesses resumos específicos.**

**Resumos de Entrada:**
* **Energia Renovável:** {renewable_energy_result}
* **Veículos Elétricos:** {ev_technology_result}
* **Captura de Carbono:** {carbon_capture_result}

**Formato de Saída:**
## Resumo dos Recentes Avanços em Tecnologia Sustentável

### Achados de Energia Renovável (Baseado nos achados do RenewableEnergyResearcher)
[Sintetize e elabore *apenas* no resumo de entrada de energia renovável fornecido acima.]

### Achados de Veículos Elétricos (Baseado nos achados do EVResearcher)
[Sintetize e elabore *apenas* no resumo de entrada de VE fornecido acima.]

### Achados de Captura de Carbono (Baseado nos achados do CarbonCaptureResearcher)
[Sintetize e elabore *apenas* no resumo de entrada de captura de carbono fornecido acima.]

### Conclusão Geral
[Forneça uma declaração conclusiva breve (1-2 frases) que conecte *apenas* os achados apresentados acima.]

Saída *apenas* o relatório estruturado seguindo este formato. Não inclua frases introdutórias ou conclusivas fora desta estrutura, e adira estritamente ao uso apenas do conteúdo de resumo de entrada fornecido.""",
    description="Combina achados de pesquisa de agentes paralelos em um relatório estruturado e citado, estritamente fundamentado em entradas fornecidas.",
    # Nenhuma ferramenta necessária para fusão
    # Nenhuma output_key necessária aqui, pois sua resposta direta é a saída final da sequência
)

# --- 4. Criar o SequentialAgent (Orquestra o fluxo geral) ---
# Este é o agente principal que será executado. Ele primeiro executa o ParallelAgent
# para popular o estado, e então executa o MergerAgent para produzir a saída final.
sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    # Executar pesquisa paralela primeiro, depois mesclar
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordena pesquisa paralela e sintetiza os resultados."
)

root_agent = sequential_pipeline_agent
```

Este código define um sistema multi-agente usado para pesquisar e sintetizar informações sobre avanços em tecnologia sustentável. Ele configura três instâncias LlmAgent para atuar como pesquisadores especializados. ResearcherAgent_1 foca em fontes de energia renovável, ResearcherAgent_2 pesquisa tecnologia de veículos elétricos, e ResearcherAgent_3 investiga métodos de captura de carbono. Cada agente pesquisador é configurado para usar um GEMINI_MODEL e a ferramenta google_search. Eles são instruídos a resumir seus achados de forma concisa (1-2 frases) e armazenar esses resumos no estado da sessão usando output_key.

Um ParallelAgent nomeado ParallelWebResearchAgent é então criado para executar esses três agentes pesquisadores concorrentemente. Isso permite que a pesquisa seja conduzida em paralelo, potencialmente economizando tempo. O ParallelAgent completa sua execução uma vez que todos os seus sub-agentes (os pesquisadores) tenham terminado e popularo o estado.

Em seguida, um MergerAgent (também um LlmAgent) é definido para sintetizar os resultados da pesquisa. Este agente toma os resumos armazenados no estado da sessão pelos pesquisadores paralelos como entrada. Sua instrução enfatiza que a saída deve ser estritamente baseada apenas nos resumos de entrada fornecidos, proibindo a adição de conhecimento externo. O MergerAgent é projetado para estruturar os achados combinados em um relatório com cabeçalhos para cada tópico e uma breve conclusão geral.

Finalmente, um SequentialAgent nomeado ResearchAndSynthesisPipeline é criado para orquestrar todo o workflow. Como controlador principal, este agente principal primeiro executa o ParallelAgent para realizar a pesquisa. Uma vez que o ParallelAgent esteja completo, o SequentialAgent então executa o MergerAgent para sintetizar as informações coletadas. O sequential_pipeline_agent é definido como o root_agent, representando o ponto de entrada para executar este sistema multi-agente. O processo geral é projetado para reunir eficientemente informações de múltiplas fontes em paralelo e depois combiná-las em um único relatório estruturado.

# Em Resumo

**O que:** Muitos workflows agênticos envolvem múltiplas sub-tarefas que devem ser completadas para atingir um objetivo final. Uma execução puramente sequencial, onde cada tarefa aguarda a anterior terminar, é frequentemente ineficiente e lenta. Esta latência se torna um gargalo significativo quando as tarefas dependem de operações de I/O externas, como chamar diferentes APIs ou consultar múltiplos bancos de dados. Sem um mecanismo para execução concorrente, o tempo total de processamento é a soma de todas as durações de tarefas individuais, prejudicando o desempenho geral do sistema e responsividade.

**Por quê:** O padrão de Paralelização fornece uma solução padronizada permitindo a execução simultânea de tarefas independentes. Ele funciona identificando componentes de um workflow, como usos de ferramentas ou chamadas LLM, que não dependem das saídas imediatas uns dos outros. Frameworks agênticos como LangChain e o Google ADK fornecem construções integradas para definir e gerenciar essas operações concorrentes. Por exemplo, um processo principal pode invocar várias sub-tarefas que executam em paralelo e aguardar todas elas completarem antes de prosseguir para a próxima etapa. Ao executar essas tarefas independentes ao mesmo tempo em vez de uma após a outra, este padrão reduz drasticamente o tempo total de execução.

**Regra prática:** Use este padrão quando um workflow contém múltiplas operações independentes que podem executar simultaneamente, como buscar dados de várias APIs, processar diferentes pedaços de dados, ou gerar múltiplas peças de conteúdo para síntese posterior.

**Resumo visual**

**![][image2]**

Fig.2: Padrão de design de paralelização

# Principais Takeaways

Aqui estão os principais takeaways:

* A paralelização é um padrão para executar tarefas independentes concorrentemente para melhorar a eficiência.
* É particularmente útil quando as tarefas envolvem aguardar recursos externos, como chamadas de API.
* A adoção de uma arquitetura concorrente ou paralela introduz complexidade e custo substanciais, impactando fases-chave de desenvolvimento como design, depuração e logging do sistema.
* Frameworks como LangChain e Google ADK fornecem suporte integrado para definir e gerenciar execução paralela.
* Na LangChain Expression Language (LCEL), RunnableParallel é uma construção-chave para executar múltiplos executáveis lado a lado.
* O Google ADK pode facilitar execução paralela através de Delegação Dirigida por LLM, onde o LLM de um agente Coordenador identifica sub-tarefas independentes e dispara seu tratamento concorrente por sub-agentes especializados.
* A paralelização ajuda a reduzir a latência geral e torna sistemas agênticos mais responsivos para tarefas complexas.

# Conclusão

O padrão de paralelização é um método para otimizar workflows computacionais executando sub-tarefas independentes concorrentemente. Esta abordagem reduz a latência geral, particularmente em operações complexas que envolvem múltiplas inferências de modelo ou chamadas para serviços externos.

Frameworks fornecem mecanismos distintos para implementar este padrão. No LangChain, construções como RunnableParallel são usadas para definir explicitamente e executar múltiplas cadeias de processamento simultaneamente. Em contraste, frameworks como o Google Agent Developer Kit (ADK) podem alcançar paralelização através de delegação multi-agente, onde um modelo coordenador primário atribui diferentes sub-tarefas a agentes especializados que podem operar concorrentemente.

Ao integrar processamento paralelo com fluxos de controle sequenciais (encadeamento) e condicionais (roteamento), torna-se possível construir sistemas computacionais sofisticados e de alta performance capazes de gerenciar eficientemente tarefas diversas e complexas.

# Referências

Aqui estão alguns recursos para leitura adicional sobre o padrão de Paralelização e conceitos relacionados:

1. Documentação da LangChain Expression Language (LCEL) (Paralelismo): [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)   
2. Documentação do Google Agent Developer Kit (ADK) (Sistemas Multi-Agente): [https://google.github.io/adk-docs/agents/multi-agents/](https://google.github.io/adk-docs/agents/multi-agents/)  
3. Documentação do Python asyncio: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)

[image1]: ../assets/08-chapter-3-image-1-line-168.png

[image2]: ../assets/08-chapter-3-image-2-line-170.png
