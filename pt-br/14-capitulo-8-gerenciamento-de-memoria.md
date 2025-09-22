# Capítulo 8: Gerenciamento de Memória

O gerenciamento eficaz de memória é crucial para agentes inteligentes reterem informações. Agentes requerem diferentes tipos de memória, muito parecido com humanos, para operar eficientemente. Este capítulo aprofunda o gerenciamento de memória, especificamente abordando os requisitos de memória imediata (curto prazo) e persistente (longo prazo) dos agentes.

Em sistemas de agentes, memória refere-se à capacidade de um agente de reter e utilizar informações de interações passadas, observações e experiências de aprendizado. Esta capacidade permite que agentes tomem decisões informadas, mantenham contexto conversacional e melhorem ao longo do tempo. A memória do agente é geralmente categorizada em dois tipos principais:

* **Memória de Curto Prazo (Memória Contextual):** Similar à memória de trabalho, isso mantém informações atualmente sendo processadas ou acessadas recentemente. Para agentes usando modelos de linguagem grandes (LLMs), a memória de curto prazo existe principalmente dentro da janela de contexto. Esta janela contém mensagens recentes, respostas do agente, resultados do uso de ferramentas e reflexões do agente da interação atual, todos os quais informam as respostas e ações subsequentes do LLM. A janela de contexto tem uma capacidade limitada, restringindo a quantidade de informações recentes que um agente pode acessar diretamente. O gerenciamento eficiente de memória de curto prazo envolve manter as informações mais relevantes dentro deste espaço limitado, possivelmente através de técnicas como sumarizar segmentos de conversa mais antigos ou enfatizar detalhes-chave. O advento de modelos com janelas de 'contexto longo' simplesmente expande o tamanho desta memória de curto prazo, permitindo que mais informações sejam mantidas dentro de uma única interação. No entanto, este contexto ainda é efêmero e é perdido uma vez que a sessão conclui, e pode ser caro e ineficiente processar toda vez. Consequentemente, agentes requerem tipos de memória separados para alcançar verdadeira persistência, recordar informações de interações passadas e construir uma base de conhecimento duradoura.  
* **Memória de Longo Prazo (Memória Persistente):** Isso atua como um repositório para informações que agentes precisam reter através de várias interações, tarefas ou períodos estendidos, similar a bases de conhecimento de longo prazo. Os dados são tipicamente armazenados fora do ambiente de processamento imediato do agente, frequentemente em bancos de dados, grafos de conhecimento ou bancos de dados vetoriais. Em bancos de dados vetoriais, as informações são convertidas em vetores numéricos e armazenadas, permitindo que agentes recuperem dados baseados em similaridade semântica ao invés de correspondências exatas de palavras-chave, um processo conhecido como busca semântica. Quando um agente precisa de informações da memória de longo prazo, ele consulta o armazenamento externo, recupera dados relevantes e os integra no contexto de curto prazo para uso imediato, assim combinando conhecimento prévio com a interação atual.

# Aplicações Práticas e Casos de Uso

O gerenciamento de memória é vital para agentes rastrearem informações e executarem inteligentemente ao longo do tempo. Isso é essencial para agentes superarem capacidades básicas de resposta a perguntas. As aplicações incluem:

* **Chatbots e IA Conversacional:** Manter o fluxo de conversa depende da memória de curto prazo. Chatbots requerem lembrar entradas anteriores do usuário para fornecer respostas coerentes. A memória de longo prazo permite que chatbots recordem preferências do usuário, problemas passados ou discussões anteriores, oferecendo interações personalizadas e contínuas.  
* **Agentes Orientados a Tarefas:** Agentes gerenciando tarefas de múltiplos passos precisam de memória de curto prazo para rastrear passos anteriores, progresso atual e objetivos gerais. Esta informação pode residir no contexto da tarefa ou armazenamento temporário. A memória de longo prazo é crucial para acessar dados específicos relacionados ao usuário não no contexto imediato.  
* **Experiências Personalizadas:** Agentes oferecendo interações personalizadas utilizam memória de longo prazo para armazenar e recuperar preferências do usuário, comportamentos passados e informações pessoais. Isso permite que agentes adaptem suas respostas e sugestões.  
* **Aprendizado e Melhoria:** Agentes podem refinar sua performance aprendendo de interações passadas. Estratégias bem-sucedidas, erros e novas informações são armazenados na memória de longo prazo, facilitando adaptações futuras. Agentes de aprendizado por reforço armazenam estratégias aprendidas ou conhecimento desta forma.  
* **Recuperação de Informação (RAG):** Agentes projetados para responder perguntas acessam uma base de conhecimento, sua memória de longo prazo, frequentemente implementada dentro de Geração Aumentada por Recuperação (RAG). O agente recupera documentos ou dados relevantes para informar suas respostas.  
* **Sistemas Autônomos:** Robôs ou carros autônomos requerem memória para mapas, rotas, localizações de objetos e comportamentos aprendidos. Isso envolve memória de curto prazo para arredores imediatos e memória de longo prazo para conhecimento ambiental geral.

A memória permite que agentes mantenham histórico, aprendam, personalizem interações e gerenciem problemas complexos dependentes do tempo.

# Código Hands-On: Gerenciamento de Memória no Google Agent Developer Kit (ADK)

O Google Agent Developer Kit (ADK) oferece um método estruturado para gerenciar contexto e memória, incluindo componentes para aplicação prática. Uma compreensão sólida de Session, State e Memory do ADK é vital para construir agentes que precisam reter informações.

Assim como em interações humanas, agentes requerem a capacidade de recordar trocas anteriores para conduzir conversas coerentes e naturais. O ADK simplifica o gerenciamento de contexto através de três conceitos principais e seus serviços associados.

Toda interação com um agente pode ser considerada um thread de conversa único. Agentes podem precisar acessar dados de interações anteriores. O ADK estrutura isso da seguinte forma:

* **Session:** Um thread de chat individual que registra mensagens e ações (Events) para essa interação específica, também armazenando dados temporários (State) relevantes para essa conversa.  
* **State (session.state):** Dados armazenados dentro de uma Session, contendo informações relevantes apenas para o thread de chat ativo atual.  
* **Memory:** Um repositório pesquisável de informações provenientes de vários chats passados ou fontes externas, servindo como um recurso para recuperação de dados além da conversa imediata.

O ADK fornece serviços dedicados para gerenciar componentes críticos essenciais para construir agentes complexos, com estado e conscientes do contexto. O SessionService gerencia threads de chat (objetos Session) lidando com sua iniciação, registro e término, enquanto o MemoryService supervisiona o armazenamento e recuperação de conhecimento de longo prazo (Memory).

Tanto o SessionService quanto o MemoryService oferecem várias opções de configuração, permitindo que usuários escolham métodos de armazenamento baseados nas necessidades da aplicação. Opções em memória estão disponíveis para propósitos de teste, embora os dados não persistam através de reinicializações. Para armazenamento persistente e escalabilidade, o ADK também suporta serviços baseados em banco de dados e nuvem.

## Session: Acompanhando Cada Chat

Um objeto Session no ADK é projetado para rastrear e gerenciar threads de chat individuais. Na iniciação de uma conversa com um agente, o SessionService gera um objeto Session, representado como `google.adk.sessions.Session`. Este objeto encapsula todos os dados relevantes para um thread de conversa específico, incluindo identificadores únicos (id, app_name, user_id), um registro cronológico de eventos como objetos Event, uma área de armazenamento para dados temporários específicos da sessão conhecidos como state, e um timestamp indicando a última atualização (last_update_time). Desenvolvedores tipicamente interagem com objetos Session indiretamente através do SessionService. O SessionService é responsável por gerenciar o ciclo de vida de sessões de conversa, que inclui iniciar novas sessões, retomar sessões anteriores, registrar atividade da sessão (incluindo atualizações de estado), identificar sessões ativas e gerenciar a remoção de dados da sessão. O ADK fornece várias implementações de SessionService com mecanismos de armazenamento variados para histórico da sessão e dados temporários, como o InMemorySessionService, que é adequado para teste mas não fornece persistência de dados através de reinicializações da aplicação.

| `# Exemplo: Usando InMemorySessionService # Isso é adequado para desenvolvimento local e teste onde persistência # de dados através de reinicializações da aplicação não é requerida. from google.adk.sessions import InMemorySessionService session_service = InMemorySessionService()` |
| :---- |

Então há o DatabaseSessionService se você quiser salvamento confiável em um banco de dados que você gerencia.

| `# Exemplo: Usando DatabaseSessionService # Isso é adequado para produção ou desenvolvimento requerendo armazenamento persistente. # Você precisa configurar uma URL do banco de dados (ex., para SQLite, PostgreSQL, etc.). # Requer: pip install google-adk[sqlalchemy] e um driver de banco de dados (ex., psycopg2 para PostgreSQL) from google.adk.sessions import DatabaseSessionService # Exemplo usando um arquivo SQLite local: db_url = "sqlite:///./my_agent_data.db" session_service = DatabaseSessionService(db_url=db_url)` |
| :---- |

Além disso, há o VertexAiSessionService que usa infraestrutura Vertex AI para produção escalável no Google Cloud.

| `# Exemplo: Usando VertexAiSessionService # Isso é adequado para produção escalável no Google Cloud Platform, aproveitando # infraestrutura Vertex AI para gerenciamento de sessão. # Requer: pip install google-adk[vertexai] e configuração/autenticação GCP from google.adk.sessions import VertexAiSessionService PROJECT_ID = "your-gcp-project-id" # Substitua pelo seu ID do projeto GCP LOCATION = "us-central1" # Substitua pela sua localização GCP desejada # O app_name usado com este serviço deve corresponder ao ID do Reasoning Engine ou nome REASONING_ENGINE_APP_NAME = "projects/your-gcp-project-id/locations/us-central1/reasoningEngines/your-engine-id" # Substitua pelo nome do recurso Reasoning Engine session_service = VertexAiSessionService(project=PROJECT_ID, location=LOCATION) # Ao usar este serviço, passe REASONING_ENGINE_APP_NAME para métodos do serviço: # session_service.create_session(app_name=REASONING_ENGINE_APP_NAME, ...) # session_service.get_session(app_name=REASONING_ENGINE_APP_NAME, ...) # session_service.append_event(session, event, app_name=REASONING_ENGINE_APP_NAME) # session_service.delete_session(app_name=REASONING_ENGINE_APP_NAME, ...)` |
| :---- |

Escolher um SessionService apropriado é crucial pois determina como o histórico de interação do agente e dados temporários são armazenados e sua persistência.

Cada troca de mensagem envolve um processo cíclico: Uma mensagem é recebida, o Runner recupera ou estabelece uma Session usando o SessionService, o agente processa a mensagem usando o contexto da Session (estado e interações históricas), o agente gera uma resposta e pode atualizar o estado, o Runner encapsula isso como um Event, e o método session_service.append_event registra o novo evento e atualiza o estado no armazenamento. A Session então aguarda a próxima mensagem. Idealmente, o método delete_session é empregado para terminar a sessão quando a interação conclui. Este processo ilustra como o SessionService mantém continuidade gerenciando o histórico específico da Session e dados temporários.

## State: O Bloco de Rascunho da Session

No ADK, cada Session, representando um thread de chat, inclui um componente de estado similar à memória de trabalho temporária de um agente durante a duração dessa conversa específica. Enquanto session.events registra todo o histórico do chat, session.state armazena e atualiza pontos de dados dinâmicos relevantes para o chat ativo.

Fundamentalmente, session.state opera como um dicionário, armazenando dados como pares chave-valor. Sua função principal é permitir que o agente retenha e gerencie detalhes essenciais para diálogo coerente, como preferências do usuário, progresso da tarefa, coleta incremental de dados ou flags condicionais influenciando ações subsequentes do agente.

A estrutura do estado compreende chaves de string emparelhadas com valores de tipos Python serializáveis, incluindo strings, números, booleanos, listas e dicionários contendo estes tipos básicos. O estado é dinâmico, evoluindo ao longo da conversa. A permanência destas mudanças depende do SessionService configurado.

A organização do estado pode ser alcançada usando prefixos de chave para definir escopo e persistência de dados. Chaves sem prefixos são específicas da sessão.

* O prefixo user: associa dados com um ID de usuário através de todas as sessões.   
* O prefixo app: designa dados compartilhados entre todos os usuários da aplicação.   
* O prefixo temp: indica dados válidos apenas para o turno de processamento atual e não são persistentemente armazenados.

O agente acessa todos os dados de estado através de um único dicionário session.state. O SessionService lida com recuperação, mesclagem e persistência de dados. O estado deve ser atualizado ao adicionar um Event ao histórico da sessão via session_service.append_event(). Isso garante rastreamento preciso, salvamento adequado em serviços persistentes e manipulação segura de mudanças de estado.

1. **A Maneira Simples: Usando output_key (para Respostas de Texto do Agente):** Esta é a maneira mais fácil se você apenas quiser salvar a resposta final de texto do seu agente diretamente no estado. Quando você configura seu LlmAgent, apenas diga a ele o output_key que você quer usar. O Runner vê isso e automaticamente cria as ações necessárias para salvar a resposta no estado quando ele anexa o evento. Vamos ver um exemplo de código demonstrando atualização de estado via output_key.

| `# Importar classes necessárias do Google Agent Developer Kit (ADK) from google.adk.agents import LlmAgent from google.adk.sessions import InMemorySessionService, Session from google.adk.runners import Runner from google.genai.types import Content, Part # Definir um LlmAgent com um output_key. greeting_agent = LlmAgent(    name="Greeter",    model="gemini-2.0-flash",    instruction="Gere uma saudação curta e amigável.",    output_key="last_greeting" ) # --- Configurar Runner e Session --- app_name, user_id, session_id = "state_app", "user1", "session1" session_service = InMemorySessionService() runner = Runner(    agent=greeting_agent,    app_name=app_name,    session_service=session_service ) session = session_service.create_session(    app_name=app_name,    user_id=user_id,    session_id=session_id ) print(f"Estado inicial: {session.state}") # --- Executar o Agente --- user_message = Content(parts=[Part(text="Hello")]) print("\n--- Executando o agente ---") for event in runner.run(    user_id=user_id,    session_id=session_id,    new_message=user_message ):    if event.is_final_response():      print("Agente respondeu.") # --- Verificar Estado Atualizado --- # Verificar corretamente o estado *após* o runner ter terminado de processar todos os eventos. updated_session = session_service.get_session(app_name, user_id, session_id) print(f"\nEstado após execução do agente: {updated_session.state}")` |
| :---- |

Nos bastidores, o Runner vê seu output_key e automaticamente cria as ações necessárias com um state_delta quando ele chama append_event.

2. **A Maneira Padrão: Usando EventActions.state_delta (para Atualizações Mais Complicadas):** Para momentos quando você precisa fazer coisas mais complexas – como atualizar várias chaves de uma vez, salvar coisas que não são apenas texto, direcionar escopos específicos como user: ou app:, ou fazer atualizações que não estão vinculadas à resposta final de texto do agente – você construirá manualmente um dicionário de suas mudanças de estado (o state_delta) e o incluirá dentro das EventActions do Event que você está anexando. Vamos ver um exemplo:

| `import time from google.adk.tools.tool_context import ToolContext from google.adk.sessions import InMemorySessionService # --- Definir a Abordagem Recomendada Baseada em Ferramentas --- def log_user_login(tool_context: ToolContext) -> dict:    """    Atualiza o estado da sessão em um evento de login do usuário.    Esta ferramenta encapsula todas as mudanças de estado relacionadas a um login do usuário.    Args:        tool_context: Fornecido automaticamente pelo ADK, dá acesso ao estado da sessão.    Returns:        Um dicionário confirmando que a ação foi bem-sucedida.    """    # Acessar o estado diretamente através do contexto fornecido.    state = tool_context.state       # Obter valores atuais ou padrões, então atualizar o estado.    # Isso é muito mais limpo e co-localiza a lógica.    login_count = state.get("user:login_count", 0) + 1    state["user:login_count"] = login_count    state["task_status"] = "active"    state["user:last_login_ts"] = time.time()    state["temp:validation_needed"] = True       print("Estado atualizado de dentro da ferramenta `log_user_login`.")       return {        "status": "success",        "message": f"Login do usuário rastreado. Total de logins: {login_count}."    } # --- Demonstração de Uso --- # Em uma aplicação real, um Agente LLM decidiria chamar esta ferramenta. # Aqui, simulamos uma chamada direta para fins de demonstração. # 1. Configuração session_service = InMemorySessionService() app_name, user_id, session_id = "state_app_tool", "user3", "session3" session = session_service.create_session(    app_name=app_name,    user_id=user_id,    session_id=session_id,    state={"user:login_count": 0, "task_status": "idle"} ) print(f"Estado inicial: {session.state}") # 2. Simular uma chamada de ferramenta (em uma app real, o ADK Runner faz isso) # Criamos um ToolContext manualmente apenas para este exemplo independente. from google.adk.tools.tool_context import InvocationContext mock_context = ToolContext(    invocation_context=InvocationContext(        app_name=app_name, user_id=user_id, session_id=session_id,        session=session, session_service=session_service    ) ) # 3. Executar a ferramenta log_user_login(mock_context) # 4. Verificar o estado atualizado updated_session = session_service.get_session(app_name, user_id, session_id) print(f"Estado após execução da ferramenta: {updated_session.state}") # A saída esperada mostrará a mesma mudança de estado que o # caso "Antes", # mas a organização do código é significativamente mais limpa # e mais robusta.` |
| :---- |

Este código demonstra uma abordagem baseada em ferramentas para gerenciar estado de sessão do usuário em uma aplicação. Ele define uma função *log_user_login*, que atua como uma ferramenta. Esta ferramenta é responsável por atualizar o estado da sessão quando um usuário faz login.  
A função toma um objeto ToolContext, fornecido pelo ADK, para acessar e modificar o dicionário de estado da sessão. Dentro da ferramenta, ela incrementa um *user:login_count*, define o *task_status* como "active", registra o *user:last_login_ts (timestamp)*, e adiciona uma flag temporária temp:validation_needed. 

A parte de demonstração do código simula como esta ferramenta seria usada. Ela configura um serviço de sessão em memória e cria uma sessão inicial com algum estado pré-definido. Um ToolContext é então criado manualmente para imitar o ambiente no qual o ADK Runner executaria a ferramenta. A função log_user_login é chamada com este contexto simulado. Finalmente, o código recupera a sessão novamente para mostrar que o estado foi atualizado pela execução da ferramenta. O objetivo é mostrar como encapsular mudanças de estado dentro de ferramentas torna o código mais limpo e organizado comparado a manipular estado diretamente fora de ferramentas.

Note que a modificação direta do dicionário `session.state` após recuperar uma sessão é fortemente desencorajada pois contorna o mecanismo padrão de processamento de eventos. Tais mudanças diretas não serão registradas no histórico de eventos da sessão, podem não ser persistidas pelo `SessionService` selecionado, podem levar a problemas de concorrência, e não atualizarão metadados essenciais como timestamps. Os métodos recomendados para atualizar o estado da sessão são usar o parâmetro `output_key` em um `LlmAgent` (especificamente para respostas finais de texto do agente) ou incluir mudanças de estado dentro de `EventActions.state_delta` ao anexar um evento via `session_service.append_event()`. O `session.state` deve ser usado principalmente para ler dados existentes.

Para recapitular, ao projetar seu estado, mantenha simples, use tipos de dados básicos, dê nomes claros às suas chaves e use prefixos corretamente, evite aninhamento profundo, e sempre atualize estado usando o processo append_event.

## Memory: Conhecimento de Longo Prazo com MemoryService

Em sistemas de agentes, o componente Session mantém um registro do histórico de chat atual (eventos) e dados temporários (estado) específicos para uma única conversa. No entanto, para agentes reterem informações através de múltiplas interações ou acessarem dados externos, gerenciamento de conhecimento de longo prazo é necessário. Isso é facilitado pelo MemoryService.

| `# Exemplo: Usando InMemoryMemoryService # Isso é adequado para desenvolvimento local e teste onde persistência # de dados através de reinicializações da aplicação não é requerida.  # Conteúdo de memória é perdido quando o app para. from google.adk.memory import InMemoryMemoryService memory_service = InMemoryMemoryService()` |
| :---- |

Session e State podem ser conceitualizados como memória de curto prazo para uma única sessão de chat, enquanto o Conhecimento de Longo Prazo gerenciado pelo MemoryService funciona como um repositório persistente e pesquisável. Este repositório pode conter informações de múltiplas interações passadas ou fontes externas. O MemoryService, conforme definido pela interface BaseMemoryService, estabelece um padrão para gerenciar este conhecimento pesquisável de longo prazo. Suas funções principais incluem adicionar informações, que envolve extrair conteúdo de uma sessão e armazená-lo usando o método add_session_to_memory, e recuperar informações, que permite que um agente consulte o armazenamento e receba dados relevantes usando o método search_memory.

O ADK oferece várias implementações para criar este armazenamento de conhecimento de longo prazo. O InMemoryMemoryService fornece uma solução de armazenamento temporário adequada para propósitos de teste, mas dados não são preservados através de reinicializações da aplicação. Para ambientes de produção, o VertexAiRagMemoryService é tipicamente utilizado. Este serviço aproveita o serviço de Geração Aumentada por Recuperação (RAG) do Google Cloud, permitindo capacidades de busca semântica escalável e persistente (Também, consulte o capítulo 14 sobre RAG).

| `# Exemplo: Usando VertexAiRagMemoryService # Isso é adequado para produção escalável no GCP, aproveitando # Vertex AI RAG (Retrieval Augmented Generation) para memória persistente e  # pesquisável. # Requer: pip install google-adk[vertexai], configuração # GCP/autenticação, e um Corpus RAG Vertex AI. from google.adk.memory import VertexAiRagMemoryService # O nome do recurso do seu Corpus RAG Vertex AI RAG_CORPUS_RESOURCE_NAME = "projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id" # Substitua pelo nome do recurso do seu Corpus # Configuração opcional para comportamento de recuperação SIMILARITY_TOP_K = 5 # Número de resultados principais para recuperar VECTOR_DISTANCE_THRESHOLD = 0.7 # Limiar para similaridade vetorial memory_service = VertexAiRagMemoryService(    rag_corpus=RAG_CORPUS_RESOURCE_NAME,    similarity_top_k=SIMILARITY_TOP_K,    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD ) # Ao usar este serviço, métodos como add_session_to_memory  # e search_memory interagirão com o Corpus RAG Vertex AI especificado.` |
| :---- |

# Código hands-on: Gerenciamento de Memória no LangChain e LangGraph

No LangChain e LangGraph, Memória é um componente crítico para criar aplicações conversacionais inteligentes e com sensação natural. Isso permite que um agente de IA lembre informações de interações passadas, aprenda com feedback e se adapte às preferências do usuário. A funcionalidade de memória do LangChain fornece a base para isso referenciando um histórico armazenado para enriquecer prompts atuais e então registrando a última troca para uso futuro. Conforme agentes lidam com tarefas mais complexas, esta capacidade se torna essencial tanto para eficiência quanto satisfação do usuário.

**Memória de Curto Prazo:** Isso é escopo de thread, significando que rastreia a conversa em andamento dentro de uma única sessão ou thread. Isso fornece contexto imediato, mas um histórico completo pode desafiar a janela de contexto de um LLM, potencialmente levando a erros ou performance ruim. O LangGraph gerencia memória de curto prazo como parte do estado do agente, que é persistido via um checkpointer, permitindo que um thread seja retomado a qualquer momento.

**Memória de Longo Prazo:** Isso armazena dados específicos do usuário ou nível de aplicação através de sessões e é compartilhado entre threads conversacionais. É salvo em "namespaces" customizados e pode ser recordado a qualquer momento em qualquer thread. O LangGraph fornece armazenamentos para salvar e recordar memórias de longo prazo, permitindo que agentes retenham conhecimento indefinidamente.

O LangChain fornece várias ferramentas para gerenciar histórico de conversa, variando de controle manual a integração automatizada dentro de chains.

**ChatMessageHistory: Gerenciamento Manual de Memória.** Para controle direto e simples sobre o histórico de uma conversa fora de uma chain formal, a classe ChatMessageHistory é ideal. Isso permite o rastreamento manual de trocas de diálogo.

| `from langchain.memory import ChatMessageHistory # Inicializar o objeto de histórico history = ChatMessageHistory() # Adicionar mensagens do usuário e IA history.add_user_message("Estou indo para Nova York na próxima semana.") history.add_ai_message("Ótimo! É uma cidade fantástica.") # Acessar a lista de mensagens print(history.messages)` |
| :---- |

**ConversationBufferMemory: Memória Automatizada para Chains**. Para integrar memória diretamente em chains, ConversationBufferMemory é uma escolha comum. Isso mantém um buffer da conversa e o disponibiliza para seu prompt. Seu comportamento pode ser customizado com dois parâmetros principais:

* memory_key: Uma string que especifica o nome da variável em seu prompt que manterá o histórico do chat. Padrão para "history".  
* return_messages: Um booleano que dita o formato do histórico.  
  * Se False (o padrão), retorna uma única string formatada, que é ideal para LLMs padrão.  
  * Se True, retorna uma lista de objetos de mensagem, que é o formato recomendado para Chat Models.

| `from langchain.memory import ConversationBufferMemory # Inicializar memória memory = ConversationBufferMemory() # Salvar um turno de conversa memory.save_context({"input": "Como está o clima?"}, {"output": "Está ensolarado hoje."}) # Carregar a memória como string print(memory.load_memory_variables({}))` |
| :---- |

Integrar esta memória em um LLMChain permite que o modelo acesse o histórico da conversa e forneça respostas contextualmente relevantes

| `from langchain_openai import OpenAI from langchain.chains import LLMChain from langchain.prompts import PromptTemplate from langchain.memory import ConversationBufferMemory # 1. Definir LLM e Prompt llm = OpenAI(temperature=0) template = """Você é um agente de viagem útil. Conversa anterior: {history} Nova pergunta: {question} Resposta:""" prompt = PromptTemplate.from_template(template) # 2. Configurar Memória # A memory_key "history" corresponde à variável no prompt memory = ConversationBufferMemory(memory_key="history") # 3. Construir a Chain conversation = LLMChain(llm=llm, prompt=prompt, memory=memory) # 4. Executar a Conversa response = conversation.predict(question="Quero reservar um voo.") print(response) response = conversation.predict(question="Meu nome é Sam, a propósito.") print(response) response = conversation.predict(question="Qual era meu nome mesmo?") print(response)` |
| :---- |

Para eficácia melhorada com modelos de chat, é recomendado usar uma lista estruturada de objetos de mensagem definindo `return_messages=True`.

| `from langchain_openai import ChatOpenAI from langchain.chains import LLMChain from langchain.memory import ConversationBufferMemory from langchain_core.prompts import (    ChatPromptTemplate,    MessagesPlaceholder,    SystemMessagePromptTemplate,    HumanMessagePromptTemplate, ) # 1. Definir Chat Model e Prompt llm = ChatOpenAI() prompt = ChatPromptTemplate(    messages=[        SystemMessagePromptTemplate.from_template("Você é um assistente amigável."),        MessagesPlaceholder(variable_name="chat_history"),        HumanMessagePromptTemplate.from_template("{question}")    ] ) # 2. Configurar Memória # return_messages=True é essencial para modelos de chat memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # 3. Construir a Chain conversation = LLMChain(llm=llm, prompt=prompt, memory=memory) # 4. Executar a Conversa response = conversation.predict(question="Oi, eu sou Jane.") print(response) response = conversation.predict(question="Você lembra do meu nome?") print(response)` |
| :---- |

**Tipos de Memória de Longo Prazo**: A memória de longo prazo permite que sistemas retenham informações através de diferentes conversas, fornecendo um nível mais profundo de contexto e personalização. Isso pode ser dividido em três tipos análogos à memória humana:

* **Memória Semântica: Lembrando Fatos:** Isso envolve reter fatos específicos e conceitos, como preferências do usuário ou conhecimento de domínio. É usado para fundamentar as respostas de um agente, levando a interações mais personalizadas e relevantes. Esta informação pode ser gerenciada como um "perfil" do usuário continuamente atualizado (um documento JSON) ou como uma "coleção" de documentos factuais individuais.  
* **Memória Episódica: Lembrando Experiências:** Isso envolve recordar eventos ou ações passadas. Para agentes de IA, a memória episódica é frequentemente usada para lembrar como realizar uma tarefa. Na prática, é frequentemente implementada através de prompting few-shot, onde um agente aprende de sequências de interação bem-sucedidas passadas para executar tarefas corretamente.  
* **Memória Procedural: Lembrando Regras:** Esta é a memória de como executar tarefas—as instruções principais e comportamentos do agente, frequentemente contidas em seu prompt do sistema. É comum para agentes modificarem seus próprios prompts para se adaptar e melhorar. Uma técnica eficaz é "Reflexão," onde um agente é promptado com suas instruções atuais e interações recentes, então solicitado a refinar suas próprias instruções.

# Principais Conclusões

Para recapitular rapidamente os pontos principais sobre gerenciamento de memória:

* A memória é super importante para agentes manterem controle das coisas, aprenderem e personalizarem interações.  
* IA Conversacional depende tanto da memória de curto prazo para contexto imediato dentro de um único chat quanto da memória de longo prazo para conhecimento persistente através de múltiplas sessões.  
* Memória de curto prazo (as coisas imediatas) é temporária, frequentemente limitada pela janela de contexto do LLM ou como o framework passa contexto.  
* Memória de longo prazo (as coisas que ficam por aí) salva informações através de chats diferentes usando armazenamento externo como bancos de dados vetoriais e é acessada por busca.  
* Frameworks como ADK têm partes específicas como Session (o thread do chat), State (dados temporários do chat), e MemoryService (o conhecimento pesquisável de longo prazo) para gerenciar memória.  
* O SessionService do ADK lida com toda a vida de uma sessão de chat, incluindo seu histórico (eventos) e dados temporários (estado).  
* O session.state do ADK é um dicionário para dados temporários do chat. Prefixos (user:, app:, temp:) te dizem onde os dados pertencem e se eles ficam por aí.  
* No ADK, você deve atualizar estado usando EventActions.state_delta ou output_key ao adicionar eventos, não mudando o dicionário de estado diretamente.  
* O MemoryService do ADK é para colocar informações em armazenamento de longo prazo e deixar agentes pesquisarem, frequentemente usando ferramentas.  
* O LangChain oferece ferramentas práticas como ConversationBufferMemory para automaticamente injetar o histórico de uma única conversa em um prompt, permitindo que um agente recorde contexto imediato.  
* O LangGraph permite memória avançada de longo prazo usando um armazenamento para salvar e recuperar fatos semânticos, experiências episódicas, ou até mesmo regras procedurais atualizáveis através de diferentes sessões de usuário.  
* Memory Bank é um serviço gerenciado que fornece agentes com memória persistente de longo prazo automaticamente extraindo, armazenando e recordando informações específicas do usuário para permitir conversas personalizadas e contínuas através de frameworks como Google ADK, LangGraph, e CrewAI.

# Conclusão

Este capítulo mergulhou no trabalho realmente importante de gerenciamento de memória para sistemas de agentes, mostrando a diferença entre o contexto de curta duração e o conhecimento que fica por muito tempo. Falamos sobre como estes tipos de memória são configurados e onde você os vê sendo usados na construção de agentes mais inteligentes que podem lembrar das coisas. Damos uma olhada detalhada em como o Google ADK te dá peças específicas como Session, State, e MemoryService para lidar com isso. Agora que cobrimos como agentes podem lembrar das coisas, tanto de curto quanto longo prazo, podemos passar para como eles podem aprender e se adaptar. O próximo padrão "Aprendizado e Adaptação" é sobre um agente mudando como pensa, age, ou o que sabe, tudo baseado em novas experiências ou dados.

# Referências

1. ADK Memory, [https://google.github.io/adk-docs/sessions/memory/](https://google.github.io/adk-docs/sessions/memory/)   
2. LangGraph Memory, [https://langchain-ai.github.io/langgraph/concepts/memory/](https://langchain-ai.github.io/langgraph/concepts/memory/)   
3. Vertex AI Agent Engine Memory Bank, [https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-memory-bank-in-public-preview](https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-memory-bank-in-public-preview)
