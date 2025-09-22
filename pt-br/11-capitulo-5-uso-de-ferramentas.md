# Capítulo 5: Uso de Ferramentas (Function Calling)

# Visão Geral do Padrão de Uso de Ferramentas

Até agora, discutimos padrões agênticos que envolvem principalmente orquestrar interações entre modelos de linguagem e gerenciar o fluxo de informação dentro do workflow interno do agente (Encadeamento, Roteamento, Paralelização, Reflexão). No entanto, para agentes serem verdadeiramente úteis e interagir com o mundo real ou sistemas externos, eles precisam da capacidade de usar Ferramentas.

O padrão de Uso de Ferramentas, frequentemente implementado através de um mecanismo chamado Function Calling, permite que um agente interaja com APIs externas, bancos de dados, serviços, ou até mesmo execute código. Ele permite que o LLM no núcleo do agente decida quando e como usar uma função externa específica baseado na solicitação do usuário ou no estado atual da tarefa.

O processo tipicamente envolve:

1. **Definição de Ferramenta:** Funções ou capacidades externas são definidas e descritas ao LLM. Esta descrição inclui o propósito da função, seu nome, e os parâmetros que aceita, junto com seus tipos e descrições.  
2. **Decisão do LLM:** O LLM recebe a solicitação do usuário e as definições de ferramentas disponíveis. Baseado em seu entendimento da solicitação e das ferramentas, o LLM decide se chamar uma ou mais ferramentas é necessário para cumprir a solicitação.  
3. **Geração de Chamada de Função:** Se o LLM decide usar uma ferramenta, ele gera uma saída estruturada (frequentemente um objeto JSON) que especifica o nome da ferramenta a chamar e os argumentos (parâmetros) para passar a ela, extraídos da solicitação do usuário.  
4. **Execução de Ferramenta:** O framework agêntico ou camada de orquestração intercepta esta saída estruturada. Ele identifica a ferramenta solicitada e executa a função externa real com os argumentos fornecidos.  
5. **Observação/Resultado:** A saída ou resultado da execução da ferramenta é retornado ao agente.  
6. **Processamento do LLM (Opcional mas comum):** O LLM recebe a saída da ferramenta como contexto e a usa para formular uma resposta final ao usuário ou decidir sobre o próximo passo no workflow (que pode envolver chamar outra ferramenta, refletir, ou fornecer uma resposta final).

Este padrão é fundamental porque quebra as limitações dos dados de treinamento do LLM e permite que ele acesse informações atualizadas, execute cálculos que não pode fazer internamente, interaja com dados específicos do usuário, ou acione ações do mundo real. Function calling é o mecanismo técnico que faz a ponte entre as capacidades de raciocínio do LLM e a vasta gama de funcionalidades externas disponíveis.

Embora "function calling" descreva adequadamente invocar funções de código específicas e pré-definidas, é útil considerar o conceito mais expansivo de "tool calling". Este termo mais amplo reconhece que as capacidades de um agente podem se estender muito além da execução simples de funções. Uma "ferramenta" pode ser uma função tradicional, mas também pode ser um endpoint de API complexo, uma solicitação a um banco de dados, ou até mesmo uma instrução direcionada a outro agente especializado. Esta perspectiva nos permite visualizar sistemas mais sofisticados onde, por exemplo, um agente principal pode delegar uma tarefa complexa de análise de dados a um "agente analista" dedicado ou consultar uma base de conhecimento externa através de sua API. Pensar em termos de "tool calling" captura melhor o potencial completo dos agentes para atuar como orquestradores através de um ecossistema diverso de recursos digitais e outras entidades inteligentes.

Frameworks como LangChain, LangGraph, e Google Agent Developer Kit (ADK) fornecem suporte robusto para definir ferramentas e integrá-las em workflows de agentes, frequentemente aproveitando as capacidades nativas de function calling de LLMs modernos como aqueles na série Gemini ou OpenAI. No "canvas" destes frameworks, você define as ferramentas e então configura agentes (tipicamente LLM Agents) para estar ciente e ser capaz de usar estas ferramentas.

O Uso de Ferramentas é um padrão fundamental para construir agentes poderosos, interativos e conscientes do exterior.

# Aplicações Práticas e Casos de Uso

O padrão de Uso de Ferramentas é aplicável em virtualmente qualquer cenário onde um agente precisa ir além de gerar texto para executar uma ação ou recuperar informação específica e dinâmica:

1\. Recuperação de Informação de Fontes Externas:  
Agentes podem buscar informações atualizadas, dados de mercado, previsões meteorológicas, ou qualquer informação que não esteja em seus dados de treinamento.

* **Caso de Uso:** Um assistente de viagem que precisa verificar preços atuais de voos e disponibilidade de hotéis.  
  * **Ferramentas:** APIs de companhias aéreas, APIs de reserva de hotéis, APIs de previsão meteorológica.  
  * **Benefício:** Fornece informações precisas e atualizadas que são críticas para decisões de viagem.

2\. Execução de Cálculos e Análises:  
Agentes podem realizar cálculos complexos, análises estatísticas, ou processar dados usando ferramentas especializadas.

* **Caso de Uso:** Um assistente financeiro que analisa portfólios de investimento.  
  * **Ferramentas:** Bibliotecas de cálculo financeiro, APIs de dados de mercado, ferramentas de análise estatística.  
  * **Benefício:** Fornece análises precisas e insights baseados em dados reais.

3\. Interação com Sistemas de Usuário:  
Agentes podem acessar e manipular dados específicos do usuário, como arquivos, calendários, ou configurações.

* **Caso de Uso:** Um assistente pessoal que gerencia a agenda do usuário.  
  * **Ferramentas:** APIs de calendário (Google Calendar, Outlook), sistemas de arquivos, aplicações de produtividade.  
  * **Benefício:** Fornece funcionalidade prática que melhora a produtividade do usuário.

4\. Automação de Tarefas:  
Agentes podem automatizar tarefas repetitivas ou complexas através de integrações com sistemas externos.

* **Caso de Uso:** Um agente de marketing que gerencia campanhas publicitárias.  
  * **Ferramentas:** APIs de plataformas de mídia social, ferramentas de email marketing, sistemas de CRM.  
  * **Benefício:** Automatiza processos complexos e melhora a eficiência operacional.

5\. Monitoramento e Alertas:  
Agentes podem monitorar sistemas, métricas, ou eventos e acionar alertas ou ações quando necessário.

* **Caso de Uso:** Um agente de monitoramento de infraestrutura.  
  * **Ferramentas:** APIs de monitoramento de sistemas, ferramentas de logging, sistemas de notificação.  
  * **Benefício:** Proporciona monitoramento proativo e resposta rápida a problemas.

6\. Integração com Serviços de Terceiros:  
Agentes podem integrar com uma ampla gama de serviços externos para expandir suas capacidades.

* **Caso de Uso:** Um agente de e-commerce que gerencia vendas online.  
  * **Ferramentas:** APIs de processamento de pagamento, sistemas de inventário, APIs de envio.  
  * **Benefício:** Fornece uma experiência completa e integrada para o usuário.

O padrão de Uso de Ferramentas é essencial para criar agentes que podem funcionar efetivamente no mundo real, fornecendo valor prático além de simples geração de texto.

1\. Recuperação de Informação de Fontes Externas:  
Acessando dados em tempo real ou informação que não está presente nos dados de treinamento do LLM.

* **Caso de Uso:** Um agente meteorológico.  
  * **Ferramenta:** Uma API meteorológica que recebe uma localização e retorna as condições meteorológicas atuais.  
  * **Fluxo do Agente:** Usuário pergunta, "Como está o tempo em Londres?", LLM identifica a necessidade da ferramenta meteorológica, chama a ferramenta com "Londres", ferramenta retorna dados, LLM formata os dados em uma resposta amigável ao usuário.

2\. Interação com Bancos de Dados e APIs:  
Executando consultas, atualizações, ou outras operações em dados estruturados.

* **Caso de Uso:** Um agente de e-commerce.  
  * **Ferramentas:** Chamadas de API para verificar inventário de produtos, obter status de pedidos, ou processar pagamentos.  
  * **Fluxo do Agente:** Usuário pergunta "O produto X está em estoque?", LLM chama a API de inventário, ferramenta retorna contagem de estoque, LLM informa ao usuário o status do estoque.

3\. Executando Cálculos e Análise de Dados:  
Usando calculadoras externas, bibliotecas de análise de dados, ou ferramentas estatísticas.

* **Caso de Uso:** Um agente financeiro.  
  * **Ferramentas:** Uma função calculadora, uma API de dados do mercado de ações, uma ferramenta de planilha.  
  * **Fluxo do Agente:** Usuário pergunta "Qual é o preço atual da AAPL e calcule o lucro potencial se eu comprasse 100 ações a $150?", LLM chama API de ações, obtém preço atual, então chama ferramenta calculadora, obtém resultado, formata resposta.

4\. Enviando Comunicações:  
Enviando emails, mensagens, ou fazendo chamadas de API para serviços de comunicação externos.

* **Caso de Uso:** Um agente assistente pessoal.  
  * **Ferramenta:** Uma API de envio de email.  
  * **Fluxo do Agente:** Usuário diz, "Envie um email para João sobre a reunião de amanhã.", LLM chama uma ferramenta de email com o destinatário, assunto e corpo extraídos da solicitação.

5\. Executando Código:  
Executando snippets de código em um ambiente seguro para executar tarefas específicas.

* **Caso de Uso:** Um agente assistente de codificação.  
  * **Ferramenta:** Um interpretador de código.  
  * **Fluxo do Agente:** Usuário fornece um snippet Python e pergunta, "O que este código faz?", LLM usa a ferramenta interpretador para executar o código e analisar sua saída.

6\. Controlando Outros Sistemas ou Dispositivos:  
Interagindo com dispositivos de casa inteligente, plataformas IoT, ou outros sistemas conectados.

* **Caso de Uso:** Um agente de casa inteligente.  
  * **Ferramenta:** Uma API para controlar luzes inteligentes.  
  * **Fluxo do Agente:** Usuário diz, "Desligue as luzes da sala de estar." LLM chama a ferramenta de casa inteligente com o comando e dispositivo alvo.

O Uso de Ferramentas é o que transforma um modelo de linguagem de um gerador de texto em um agente capaz de perceber, raciocinar e agir no mundo digital ou físico (veja Fig. 1\)

![][image1]

Fig.1: Alguns exemplos de um Agente usando Ferramentas

# Exemplo de Código Hands-On (LangChain)

A implementação do uso de ferramentas dentro do framework LangChain é um processo de dois estágios. Inicialmente, uma ou mais ferramentas são definidas, tipicamente encapsulando funções Python existentes ou outros componentes executáveis. Subsequentemente, estas ferramentas são vinculadas a um modelo de linguagem, concedendo assim ao modelo a capacidade de gerar uma solicitação estruturada de uso de ferramenta quando determina que uma chamada de função externa é necessária para cumprir uma consulta do usuário.

A seguinte implementação demonstrará este princípio primeiro definindo uma função simples para simular uma ferramenta de recuperação de informação. Seguindo isso, um agente será construído e configurado para aproveitar esta ferramenta em resposta à entrada do usuário. A execução deste exemplo requer a instalação das bibliotecas principais do LangChain e um pacote de provedor específico do modelo. Além disso, autenticação adequada com o serviço de modelo de linguagem selecionado, tipicamente via uma chave API configurada no ambiente local, é um pré-requisito necessário.

| ``import os, getpass import asyncio import nest_asyncio from typing import List from dotenv import load_dotenv import logging from langchain_google_genai import ChatGoogleGenerativeAI from langchain_core.prompts import ChatPromptTemplate from langchain_core.tools import tool as langchain_tool from langchain.agents import create_tool_calling_agent, AgentExecutor # UNCOMMENT # Prompt the user securely and set API keys as an environment variables os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ") os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ") try:   # A model with function/tool calling capabilities is required.   llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)   print(f"✅ Language model initialized: {llm.model}") except Exception as e:   print(f"🛑 Error initializing language model: {e}")   llm = None # --- Define a Tool --- @langchain_tool def search_information(query: str) -> str:   """   Provides factual information on a given topic. Use this tool to find answers to phrases   like 'capital of France' or 'weather in London?'.   """   print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")   # Simulate a search tool with a dictionary of predefined results.   simulated_results = {       "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",       "capital of france": "The capital of France is Paris.",       "population of earth": "The estimated population of Earth is around 8 billion people.",       "tallest mountain": "Mount Everest is the tallest mountain above sea level.",       "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."   }   result = simulated_results.get(query.lower(), simulated_results["default"])   print(f"--- TOOL RESULT: {result} ---")   return result tools = [search_information] # --- Create a Tool-Calling Agent --- if llm:   # This prompt template requires an `agent_scratchpad` placeholder for the agent's internal steps.   agent_prompt = ChatPromptTemplate.from_messages([       ("system", "You are a helpful assistant."),       ("human", "{input}"),       ("placeholder", "{agent_scratchpad}"),   ])   # Create the agent, binding the LLM, tools, and prompt together.   agent = create_tool_calling_agent(llm, tools, agent_prompt)   # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.   # The 'tools' argument is not needed here as they are already bound to the agent.   agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools) async def run_agent_with_tool(query: str):   """Invokes the agent executor with a query and prints the final response."""   print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")   try:       response = await agent_executor.ainvoke({"input": query})       print("\n--- ✅ Final Agent Response ---")       print(response["output"])   except Exception as e:       print(f"\n🛑 An error occurred during agent execution: {e}") async def main():   """Runs all agent queries concurrently."""   tasks = [       run_agent_with_tool("What is the capital of France?"),       run_agent_with_tool("What's the weather like in London?"),       run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response   ]   await asyncio.gather(*tasks) nest_asyncio.apply() asyncio.run(main())`` |
| :---- |

O código configura um agente que chama ferramentas usando a biblioteca LangChain e o modelo Google Gemini. Ele define uma ferramenta search_information que simula fornecer respostas factuais para consultas específicas. A ferramenta tem respostas pré-definidas para "weather in london," "capital of france," e "population of earth," e uma resposta padrão para outras consultas. Um modelo ChatGoogleGenerativeAI é inicializado, garantindo que tenha capacidades de chamada de ferramenta. Um ChatPromptTemplate é criado para guiar a interação do agente. A função create_tool_calling_agent é usada para combinar o modelo de linguagem, ferramentas e prompt em um agente. Um AgentExecutor é então configurado para gerenciar a execução do agente e invocação de ferramentas. A função assíncrona run_agent_with_tool é definida para invocar o agente com uma consulta dada e imprimir o resultado. A função assíncrona main prepara múltiplas consultas para serem executadas concorrentemente. Estas consultas são projetadas para testar tanto as respostas específicas quanto padrão da ferramenta search_information. Finalmente, a chamada asyncio.run(main()) executa todas as tarefas do agente. O código inclui verificações para inicialização bem-sucedida do LLM antes de prosseguir com configuração e execução do agente.

# Exemplo de Código Hands-On (CrewAI)

Este código fornece um exemplo prático de como implementar function calling (Ferramentas) dentro do framework CrewAI. Ele configura um cenário simples onde um agente é equipado com uma ferramenta para buscar informação. O exemplo demonstra especificamente buscar um preço de ação simulado usando este agente e ferramenta.

| `# pip install crewai langchain-openai import os from crewai import Agent, Task, Crew from crewai.tools import tool import logging # --- Melhor Prática: Configurar Logging --- # Uma configuração básica de logging ajuda na depuração e rastreamento da execução do crew. logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # --- Configurar sua Chave API --- # Para produção, é recomendado usar um método mais seguro para gerenciamento de chaves # como variáveis de ambiente carregadas em tempo de execução ou um gerenciador de segredos. # # Definir a variável de ambiente para seu provedor LLM escolhido (ex., OPENAI_API_KEY) # os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" # os.environ["OPENAI_MODEL_NAME"] = "gpt-4o" # --- 1. Ferramenta Refatorada: Retorna Dados Limpos --- # A ferramenta agora retorna dados brutos (um float) ou levanta um erro Python padrão. # Isso a torna mais reutilizável e força o agente a lidar com resultados adequadamente. @tool("Stock Price Lookup Tool") def get_stock_price(ticker: str) -> float:    """    Busca o último preço simulado de ação para um símbolo de ticker de ação dado.    Retorna o preço como um float. Levanta um ValueError se o ticker não for encontrado.    """    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")    simulated_prices = {        "AAPL": 178.15,        "GOOGL": 1750.30,        "MSFT": 425.50,    }    price = simulated_prices.get(ticker.upper())    if price is not None:        return price    else:        # Levantar um erro específico é melhor que retornar uma string.        # O agente está equipado para lidar com exceções e pode decidir sobre a próxima ação.        raise ValueError(f"Preço simulado para ticker '{ticker.upper()}' não encontrado.") # --- 2. Definir o Agente --- # A definição do agente permanece a mesma, mas agora aproveitará a ferramenta melhorada. financial_analyst_agent = Agent(  role='Senior Financial Analyst',  goal='Analisar dados de ações usando ferramentas fornecidas e reportar preços chave.',  backstory="Você é um analista financeiro experiente adepto a usar fontes de dados para encontrar informação de ações. Você fornece respostas claras e diretas.",  verbose=True,  tools=[get_stock_price],  # Permitir delegação pode ser útil, mas não é necessário para esta tarefa simples.  allow_delegation=False, ) # --- 3. Tarefa Refinada: Instruções Mais Claras e Tratamento de Erro --- # A descrição da tarefa é mais específica e guia o agente sobre como reagir # tanto a recuperação bem-sucedida de dados quanto a erros potenciais. analyze_aapl_task = Task(  description=(      "Qual é o preço atual simulado de ação da Apple (ticker: AAPL)? "      "Use a 'Stock Price Lookup Tool' para encontrá-lo. "      "Se o ticker não for encontrado, você deve reportar que foi incapaz de recuperar o preço."  ),  expected_output=(      "Uma única frase clara declarando o preço simulado de ação para AAPL. "      "Por exemplo: 'O preço simulado de ação para AAPL é $178.15.' "      "Se o preço não puder ser encontrado, declare isso claramente."  ),  agent=financial_analyst_agent, ) # --- 4. Formular o Crew --- # O crew orquestra como o agente e tarefa trabalham juntos. financial_crew = Crew(  agents=[financial_analyst_agent],  tasks=[analyze_aapl_task],  verbose=True # Definir como False para logs menos detalhados em produção ) # --- 5. Executar o Crew dentro de um Bloco de Execução Principal --- # Usar um bloco __name__ == "__main__": é uma melhor prática Python padrão. def main():    """Função principal para executar o crew."""    # Verificar chave API antes de começar para evitar erros de runtime.    if not os.environ.get("OPENAI_API_KEY"):        print("ERRO: A variável de ambiente OPENAI_API_KEY não está definida.")        print("Por favor, defina-a antes de executar o script.")        return    print("\n## Iniciando o Financial Crew...")    print("---------------------------------")       # O método kickoff inicia a execução.    result = financial_crew.kickoff()    print("\n---------------------------------")    print("## Execução do crew finalizada.")    print("\nResultado Final:\n", result) if __name__ == "__main__":    main()` |
| :---- |

Este código demonstra uma aplicação simples usando a biblioteca Crew.ai para simular uma tarefa de análise financeira. Ele define uma ferramenta customizada, get_stock_price, que simula buscar preços de ações para tickers pré-definidos. A ferramenta é projetada para retornar um número de ponto flutuante para tickers válidos ou levantar um ValueError para inválidos. Um Agente Crew.ai nomeado financial_analyst_agent é criado com o papel de Senior Financial Analyst. Este agente recebe a ferramenta get_stock_price para interagir. Uma Tarefa é definida, analyze_aapl_task, instruindo especificamente o agente a encontrar o preço simulado de ação para AAPL usando a ferramenta. A descrição da tarefa inclui instruções claras sobre como lidar com casos de sucesso e falha ao usar a ferramenta. Um Crew é montado, compreendendo o financial_analyst_agent e o analyze_aapl_task. A configuração verbose está habilitada tanto para o agente quanto para o crew para fornecer logging detalhado durante a execução. A parte principal do script executa a tarefa do crew usando o método kickoff() dentro de um bloco padrão if __name__ == "__main__":. Antes de iniciar o crew, ele verifica se a variável de ambiente OPENAI_API_KEY está definida, que é necessária para o agente funcionar. O resultado da execução do crew, que é a saída da tarefa, é então impresso no console. O código também inclui configuração básica de logging para melhor rastreamento das ações do crew e chamadas de ferramentas. Ele usa variáveis de ambiente para gerenciamento de chave API, embora observe que métodos mais seguros são recomendados para ambientes de produção. Em resumo, a lógica central mostra como definir ferramentas, agentes e tarefas para criar um workflow colaborativo no Crew.ai.

# Código Hands-on (Google ADK)

# O Google Agent Developer Kit (ADK) inclui uma biblioteca de ferramentas nativamente integradas que podem ser incorporadas diretamente às capacidades de um agente.

# **Pesquisa Google:** Um exemplo primário de tal componente é a ferramenta Google Search. Esta ferramenta serve como uma interface direta para o motor de busca Google, equipando o agente com a funcionalidade para realizar buscas na web e recuperar informação externa.

| `from google.adk.agents import Agent from google.adk.runners import Runner from google.adk.sessions import InMemorySessionService from google.adk.tools import google_search from google.genai import types import nest_asyncio import asyncio # Definir variáveis necessárias para configuração de Sessão e execução de Agente APP_NAME="Google Search_agent" USER_ID="user1234" SESSION_ID="1234" # Definir Agente com acesso à ferramenta de busca root_agent = ADKAgent(   name="basic_search_agent",   model="gemini-2.0-flash-exp",   description="Agente para responder perguntas usando Google Search.",   instruction="Posso responder suas perguntas pesquisando na internet. Apenas me pergunte qualquer coisa!",   tools=[google_search] # Google Search é uma ferramenta pré-construída para realizar buscas Google. ) # Interação do Agente async def call_agent(query):   """   Função auxiliar para chamar o agente com uma consulta.   """   # Sessão e Runner   session_service = InMemorySessionService()   session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)   runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)   content = types.Content(role='user', parts=[types.Part(text=query)])   events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)   for event in events:       if event.is_final_response():           final_response = event.content.parts[0].text           print("Resposta do Agente: ", final_response) nest_asyncio.apply() asyncio.run(call_agent("qual é a última notícia de IA?"))` |
| :---- |

Este código demonstra como criar e usar um agente básico alimentado pelo Google ADK para Python. O agente é projetado para responder perguntas utilizando Google Search como ferramenta. Primeiro, bibliotecas necessárias do IPython, google.adk, e google.genai são importadas. Constantes para o nome da aplicação, ID do usuário, e ID da sessão são definidas. Uma instância Agent nomeada "basic_search_agent" é criada com uma descrição e instruções indicando seu propósito. Ela está configurada para usar a ferramenta Google Search, que é uma ferramenta pré-construída fornecida pelo ADK. Um InMemorySessionService (ver Capítulo 8) é inicializado para gerenciar sessões para o agente. Uma nova sessão é criada para os IDs de aplicação, usuário e sessão especificados. Um Runner é instanciado, ligando o agente criado com o serviço de sessão. Este runner é responsável por executar as interações do agente dentro de uma sessão. Uma função auxiliar call_agent é definida para simplificar o processo de enviar uma consulta ao agente e processar a resposta. Dentro de call_agent, a consulta do usuário é formatada como um objeto types.Content com o papel 'user'. O método runner.run é chamado com o ID do usuário, ID da sessão, e o conteúdo da nova mensagem. O método runner.run retorna uma lista de eventos representando as ações e respostas do agente. O código itera através destes eventos para encontrar a resposta final. Se um evento é identificado como a resposta final, o conteúdo de texto dessa resposta é extraído. A resposta extraída do agente é então impressa no console. Finalmente, a função call_agent é chamada com a consulta "qual é a última notícia de IA?" para demonstrar o agente em ação.

**Execução de código:** O Google ADK apresenta componentes integrados para tarefas especializadas, incluindo um ambiente para execução dinâmica de código. A ferramenta built_in_code_execution fornece ao agente um intérprete Python em sandbox. Isso permite ao modelo escrever e executar código para realizar tarefas computacionais, manipular estruturas de dados, e executar scripts procedurais. Tal funcionalidade é crítica para abordar problemas que requerem lógica determinística e cálculos precisos, que estão fora do escopo da geração de linguagem probabilística sozinha.

**Pesquisa empresarial:** Este código define uma aplicação Google ADK usando a biblioteca google.adk em Python. Ela usa especificamente um VSearchAgent, que é projetado para responder perguntas pesquisando um datastore Vertex AI Search especificado. O código inicializa um VSearchAgent nomeado "q2_strategy_vsearch_agent", fornecendo uma descrição, o modelo a usar ("gemini-2.0-flash-exp"), e o ID do datastore Vertex AI Search. O DATASTORE_ID é esperado para ser definido como uma variável de ambiente. Ele então configura um Runner para o agente, usando um InMemorySessionService para gerenciar histórico de conversa. Uma função assíncrona call_vsearch_agent_async é definida para interagir com o agente. Esta função recebe uma consulta, constrói um objeto de conteúdo de mensagem, e chama o método run_async do runner para enviar a consulta ao agente. A função então transmite a resposta do agente de volta ao console conforme ela chega. Ela também imprime informação sobre a resposta final, incluindo qualquer atribuição de fonte do datastore. Tratamento de erro está incluído para capturar exceções durante a execução do agente, fornecendo mensagens informativas sobre problemas potenciais como um ID de datastore incorreto ou permissões faltantes. Outra função assíncrona run_vsearch_example é fornecida para demonstrar como chamar o agente com consultas de exemplo. O bloco de execução principal verifica se o DATASTORE_ID está definido e então executa o exemplo usando asyncio.run. Ele inclui uma verificação para lidar com casos onde o código é executado em um ambiente que já tem um loop de eventos rodando, como um notebook Jupyter.

**Vertex Extensions:** Uma extensão Vertex AI é um wrapper de API estruturado que permite a um modelo conectar com APIs externas para processamento de dados em tempo real e execução de ações. Extensões oferecem segurança de nível empresarial, privacidade de dados, e garantias de performance. Elas podem ser usadas para tarefas como gerar e executar código, consultar websites, e analisar informação de datastores privados. O Google fornece extensões pré-construídas para casos de uso comuns como Code Interpreter e Vertex AI Search, com a opção de criar customizadas. O benefício principal das extensões inclui controles empresariais fortes e integração perfeita com outros produtos Google. A diferença chave entre extensões e function calling está em sua execução: Vertex AI executa extensões automaticamente, enquanto function calls requerem execução manual pelo usuário ou cliente.

# Em Resumo

**O que:** LLMs são geradores de texto poderosos, mas eles são fundamentalmente desconectados do mundo exterior. Seu conhecimento é estático, limitado aos dados em que foram treinados, e eles carecem da capacidade de executar ações ou recuperar informação em tempo real. Esta limitação inerente os impede de completar tarefas que requerem interação com APIs externas, bancos de dados, ou serviços. Sem uma ponte para estes sistemas externos, sua utilidade para resolver problemas do mundo real é severamente limitada.

**Por que:** O padrão de Uso de Ferramentas, frequentemente implementado via function calling, fornece uma solução padronizada para este problema. Ele funciona descrevendo funções externas disponíveis, ou "ferramentas," ao LLM de uma forma que ele pode entender. Baseado na solicitação de um usuário, o LLM agêntico pode então decidir se uma ferramenta é necessária e gerar um objeto de dados estruturado (como um JSON) especificando qual função chamar e com quais argumentos. Uma camada de orquestração executa esta chamada de função, recupera o resultado, e o alimenta de volta ao LLM. Isso permite ao LLM incorporar informação externa atualizada ou o resultado de uma ação em sua resposta final, efetivamente dando-lhe a capacidade de agir.

**Regra geral:** Use o padrão de Uso de Ferramentas sempre que um agente precisa sair do conhecimento interno do LLM e interagir com o mundo exterior. Isso é essencial para tarefas que requerem dados em tempo real (ex., verificar tempo, preços de ações), acessar informação privada ou proprietária (ex., consultar o banco de dados de uma empresa), executar cálculos precisos, executar código, ou acionar ações em outros sistemas (ex., enviar um email, controlar dispositivos inteligentes).

**Resumo visual:**

**![][image2]**

Fig.2: Padrão de design de uso de ferramentas

# Principais Takeaways

* O padrão de Uso de Ferramentas é fundamental para conectar LLMs ao mundo exterior, permitindo que eles executem ações reais e acessem informação atualizada além de seus dados de treinamento.  
* Ele funciona através de um processo estruturado: definição de ferramentas, decisão do LLM, geração de chamada de função, execução de ferramenta, e processamento do resultado.  
* Function calling é o mecanismo técnico que faz a ponte entre capacidades de raciocínio do LLM e funcionalidades externas disponíveis.  
* O padrão é aplicável em virtualmente qualquer cenário onde um agente precisa ir além de gerar texto para executar uma ação ou recuperar informação específica e dinâmica.  
* Frameworks como LangChain, CrewAI, e Google ADK fornecem suporte robusto para definir ferramentas e integrá-las em workflows de agentes.  
* O uso efetivo de ferramentas transforma um modelo de linguagem de um gerador de texto em um agente capaz de perceber, raciocinar e agir no mundo digital ou físico.  
* Este padrão é essencial para criar agentes que podem funcionar efetivamente no mundo real, fornecendo valor prático além de simples geração de texto.

# Conclusão

O padrão de Uso de Ferramentas representa uma transformação fundamental na capacidade dos agentes de IA para interagir com o mundo real. Através de function calling e tool calling, agentes podem transcender as limitações de seus dados de treinamento e executar ações práticas que fornecem valor real aos usuários. Este padrão é essencial para qualquer sistema agêntico que aspira a ser verdadeiramente útil e funcional no mundo real, estabelecendo a base para a próxima geração de aplicações de IA interativas e práticas.
