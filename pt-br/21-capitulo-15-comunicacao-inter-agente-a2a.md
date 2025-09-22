# Capítulo 15: Comunicação Inter-Agente (A2A)

Agentes de IA individuais frequentemente enfrentam limitações ao lidar com problemas complexos e multifacetados, mesmo com capacidades avançadas. Para superar isto, a Comunicação Inter-Agente (A2A) permite que diversos agentes de IA, potencialmente construídos com diferentes frameworks, colaborem efetivamente. Esta colaboração envolve coordenação perfeita, delegação de tarefas e troca de informação.

O protocolo A2A do Google é um padrão aberto projetado para facilitar esta comunicação universal. Este capítulo explorará A2A, suas aplicações práticas e sua implementação dentro do Google ADK.

# Visão Geral do Padrão de Comunicação Inter-Agente

O protocolo Agent2Agent (A2A) é um padrão aberto projetado para permitir comunicação e colaboração entre diferentes frameworks de agentes de IA. Ele garante interoperabilidade, permitindo que agentes de IA desenvolvidos com tecnologias como LangGraph, CrewAI ou Google ADK trabalhem juntos independentemente de sua origem ou diferenças de framework.

A2A é apoiado por uma gama de empresas de tecnologia e provedores de serviços, incluindo Atlassian, Box, LangChain, MongoDB, Salesforce, SAP e ServiceNow. A Microsoft planeja integrar A2A ao Azure AI Foundry e Copilot Studio, demonstrando seu compromisso com protocolos abertos. Adicionalmente, Auth0 e SAP estão integrando suporte A2A em suas plataformas e agentes.

Como um protocolo open-source, A2A acolhe contribuições da comunidade para facilitar sua evolução e adoção generalizada.

## Conceitos Centrais do A2A

O protocolo A2A fornece uma abordagem estruturada para interações de agentes, construída sobre vários conceitos centrais. Uma compreensão completa destes conceitos é crucial para qualquer pessoa desenvolvendo ou integrando com sistemas compatíveis com A2A. Os pilares fundamentais do A2A incluem Atores Centrais, Cartão de Agente, Descoberta de Agente, Comunicação e Tarefas, Mecanismos de Interação e Segurança, todos os quais serão revisados em detalhes.

**Atores Centrais:** A2A envolve três entidades principais:

* Usuário: Inicia solicitações para assistência de agente.  
* Cliente A2A (Agente Cliente): Uma aplicação ou agente de IA que atua em nome do usuário para solicitar ações ou informação.  
* Servidor A2A (Agente Remoto): Um agente de IA ou sistema que fornece um endpoint HTTP para processar solicitações de cliente e retornar resultados. O agente remoto opera como um sistema "opaco", significando que o cliente não precisa entender seus detalhes operacionais internos.

**Cartão de Agente:** A identidade digital de um agente é definida por seu Cartão de Agente, geralmente um arquivo JSON. Este arquivo contém informação chave para interação de cliente e descoberta automática, incluindo a identidade do agente, URL do endpoint e versão. Também detalha capacidades suportadas como streaming ou notificações push, habilidades específicas, modos de entrada/saída padrão e requisitos de autenticação. Abaixo está um exemplo de um Cartão de Agente para um WeatherBot.

```json
{
"name": "WeatherBot", "description": "Fornece previsões meteorológicas precisas e dados históricos.", "url": "http://weather-service.example.com/a2a", "version": "1.0.0", "capabilities":  {
"streaming": true, "pushNotifications": false, "stateTransitionHistory": true }
, "authentication":  {
"schemes": [ "apiKey" ] }
, "defaultInputModes": [ "text" ], "defaultOutputModes": [ "text" ], "skills": [  {
"id": "get_current_weather", "name": "Obter Clima Atual", "description": "Recuperar clima em tempo real para qualquer localização.", "inputModes": [ "text" ], "outputModes": [ "text" ], "examples": [ "Qual é o clima em Paris?", "Condições atuais em Tóquio" ], "tags": [ "clima", "atual", "tempo-real" ] }
,  {
"id": "get_forecast", "name": "Obter Previsão", "description": "Obter previsões meteorológicas de 5 dias.", "inputModes": [ "text" ], "outputModes": [ "text" ], "examples": [ "Previsão de 5 dias para Nova York", "Vai chover em Londres neste fim de semana?" ], "tags": [ "clima", "previsão", "predição" ] }
] }
```

**Descoberta de agente:** permite que clientes encontrem Cartões de Agente, que descrevem as capacidades de Servidores A2A disponíveis. Várias estratégias existem para este processo:

* URI Bem-Conhecido: Agentes hospedam seu Cartão de Agente em um caminho padronizado (ex., /.well-known/agent.json). Esta abordagem oferece acessibilidade ampla, frequentemente automatizada, para uso público ou específico de domínio.  
* Registros Curados: Estes fornecem um catálogo centralizado onde Cartões de Agente são publicados e podem ser consultados baseados em critérios específicos. Isto é bem adequado para ambientes empresariais precisando de gerenciamento centralizado e controle de acesso.  
* Configuração Direta: Informação do Cartão de Agente é embarcada ou compartilhada privadamente. Este método é apropriado para sistemas fortemente acoplados ou privados onde descoberta dinâmica não é crucial.

Independentemente do método escolhido, é importante proteger endpoints de Cartão de Agente. Isto pode ser alcançado através de controle de acesso, TLS mútuo (mTLS) ou restrições de rede, especialmente se o cartão contém informação sensível (embora não secreta).

**Comunicações e Tarefas:** No framework A2A, comunicação é estruturada em torno de tarefas assíncronas, que representam as unidades fundamentais de trabalho para processos de longa duração. Cada tarefa é atribuída um identificador único e move através de uma série de estados—como submetido, trabalhando ou concluído—um design que suporta processamento paralelo em operações complexas. Comunicação entre agentes ocorre através de uma Mensagem.

Esta comunicação contém atributos, que são metadados chave-valor descrevendo a mensagem (como sua prioridade ou tempo de criação), e uma ou mais partes, que carregam o conteúdo atual sendo entregue, como texto simples, arquivos ou dados JSON estruturados. As saídas tangíveis geradas por um agente durante uma tarefa são chamadas artefatos. Como mensagens, artefatos também são compostos de uma ou mais partes e podem ser transmitidos incrementalmente conforme resultados se tornam disponíveis. Toda comunicação dentro do framework A2A é conduzida sobre HTTP(S) usando o protocolo JSON-RPC 2.0 para payloads. Para manter continuidade através de múltiplas interações, um contextId gerado pelo servidor é usado para agrupar tarefas relacionadas e preservar contexto.

**Mecanismos de Interação**: Solicitação/Resposta (Polling) Server-Sent Events (SSE). A2A fornece múltiplos métodos de interação para atender várias necessidades de aplicações de IA, cada um com um mecanismo distinto:

* Solicitação/Resposta Síncrona: Para operações rápidas e imediatas. Neste modelo, o cliente envia uma solicitação e aguarda ativamente o servidor processá-la e retornar uma resposta completa em uma única troca síncrona.  
* Polling Assíncrono: Adequado para tarefas que levam mais tempo para processar. O cliente envia uma solicitação, e o servidor imediatamente a reconhece com um status "trabalhando" e um ID de tarefa. O cliente é então livre para realizar outras ações e pode periodicamente fazer polling do servidor enviando novas solicitações para verificar o status da tarefa até que seja marcada como "concluída" ou "falhou."  
* Atualizações de Streaming (Server-Sent Events - SSE): Ideal para receber resultados incrementais em tempo real. Este método estabelece uma conexão persistente e unidirecional do servidor para o cliente. Permite que o agente remoto empurre continuamente atualizações, como mudanças de status ou resultados parciais, sem o cliente precisar fazer múltiplas solicitações.  
* Notificações Push (Webhooks): Projetado para tarefas muito longas ou intensivas em recursos onde manter uma conexão constante ou polling frequente é ineficiente. O cliente pode registrar uma URL de webhook, e o servidor enviará uma notificação assíncrona (um "push") para aquela URL quando o status da tarefa muda significativamente (ex., na conclusão).

O Cartão de Agente especifica se um agente suporta capacidades de streaming ou notificação push. Além disso, A2A é agnóstico de modalidade, significando que pode facilitar estes padrões de interação não apenas para texto, mas também para outros tipos de dados como áudio e vídeo, permitindo aplicações de IA ricas e multimodais. Tanto capacidades de streaming quanto notificação push são especificadas dentro do Cartão de Agente.

```text
#Exemplo de Solicitação Síncrona  {
"jsonrpc": "2.0", "id": "1", "method": "sendTask", "params":  {
"id": "task-001", "sessionId": "session-001", "message":  {
"role": "user", "parts": [  {
"type": "text", "text": "Qual é a taxa de câmbio de USD para EUR?" }
] }
, "acceptedOutputModes": ["text/plain"], "historyLength": 5 }
}
```

A solicitação síncrona usa o método sendTask, onde o cliente pede e espera uma única resposta completa para sua consulta. Em contraste, a solicitação de streaming usa o método sendTaskSubscribe para estabelecer uma conexão persistente, permitindo que o agente envie de volta múltiplas atualizações incrementais ou resultados parciais ao longo do tempo.

```text
# Exemplo de Solicitação de Streaming  {
"jsonrpc": "2.0", "id": "2", "method": "sendTaskSubscribe", "params":  {
"id": "task-002", "sessionId": "session-001", "message":  {
"role": "user", "parts": [  {
"type": "text", "text": "Qual é a taxa de câmbio para JPY para GBP hoje?" }
] }
, "acceptedOutputModes": ["text/plain"], "historyLength": 5 }
}
```

**Segurança:** Comunicação Inter-Agente (A2A): Comunicação Inter-Agente (A2A) é um componente vital da arquitetura de sistema, permitindo troca de dados segura e perfeita entre agentes. Ela garante robustez e integridade através de vários mecanismos integrados.

Segurança de Camada de Transporte Mútua (TLS): Conexões criptografadas e autenticadas são estabelecidas para prevenir acesso não autorizado e interceptação de dados, garantindo comunicação segura.

Logs de Auditoria Abrangentes: Toda comunicação inter-agente é meticulosamente registrada, detalhando fluxo de informação, agentes envolvidos e ações. Esta trilha de auditoria é crucial para responsabilidade, solução de problemas e análise de segurança.

Declaração de Cartão de Agente: Requisitos de autenticação são explicitamente declarados no Cartão de Agente, um artefato de configuração delineando a identidade, capacidades e políticas de segurança do agente. Isto centraliza e simplifica gerenciamento de autenticação.

Manipulação de Credenciais: Agentes tipicamente se autenticam usando credenciais seguras como tokens OAuth 2.0 ou chaves de API, passadas via headers HTTP. Este método previne exposição de credenciais em URLs ou corpos de mensagem, aprimorando a segurança geral.

## A2A vs. MCP

A2A é um protocolo que complementa o Protocolo de Contexto de Modelo (MCP) da Anthropic (veja Fig. 1). Enquanto MCP foca em estruturar contexto para agentes e sua interação com dados e ferramentas externas, A2A facilita coordenação e comunicação entre agentes, permitindo delegação de tarefas e colaboração.

![][image1]

Fig.1: Comparação dos Protocolos A2A e MCP

O objetivo do A2A é aprimorar eficiência, reduzir custos de integração e fomentar inovação e interoperabilidade no desenvolvimento de sistemas de IA multi-agente complexos. Portanto, uma compreensão completa dos componentes centrais e métodos operacionais do A2A é essencial para seu design, implementação e aplicação efetivos na construção de sistemas de agentes de IA colaborativos e interoperáveis.

# Aplicações Práticas e Casos de Uso

Comunicação Inter-Agente é indispensável para construir soluções de IA sofisticadas através de domínios diversos, permitindo modularidade, escalabilidade e inteligência aprimorada.

* **Colaboração Multi-Framework:** O caso de uso primário do A2A é permitir que agentes de IA independentes, independentemente de seus frameworks subjacentes (ex., ADK, LangChain, CrewAI), comuniquem e colaborem. Isto é fundamental para construir sistemas multi-agente complexos onde diferentes agentes se especializam em diferentes aspectos de um problema.  
* **Orquestração de Fluxo de Trabalho Automatizado:** Em configurações empresariais, A2A pode facilitar fluxos de trabalho complexos permitindo que agentes deleguem e coordenem tarefas. Por exemplo, um agente pode lidar com coleta inicial de dados, então delegar para outro agente para análise, e finalmente para um terceiro para geração de relatório, todos comunicando via protocolo A2A.  
* **Recuperação Dinâmica de Informação:** Agentes podem comunicar para recuperar e trocar informação em tempo real. Um agente primário pode solicitar dados de mercado ao vivo de um "agente de busca de dados" especializado, que então usa APIs externas para reunir a informação e enviá-la de volta.

# Exemplo de Código Hands-On

Vamos examinar as aplicações práticas do protocolo A2A. O repositório em [https://github.com/google-a2a/a2a-samples/tree/main/samples](https://github.com/google-a2a/a2a-samples/tree/main/samples) fornece exemplos em Java, Go e Python que ilustram como vários frameworks de agentes, como LangGraph, CrewAI, Azure AI Foundry e AG2, podem comunicar usando A2A. Todo código neste repositório é liberado sob licença Apache 2.0. Para ilustrar ainda mais os conceitos centrais do A2A, revisaremos trechos de código focando na configuração de um Servidor A2A usando um agente baseado em ADK com ferramentas autenticadas pelo Google. Olhando em [https://github.com/google-a2a/a2a-samples/blob/main/samples/python/agents/birthday_planner_adk/calendar_agent/adk_agent.py](https://github.com/google-a2a/a2a-samples/blob/main/samples/python/agents/birthday_planner_adk/calendar_agent/adk_agent.py)

```python
import datetime
from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
from google.adk.tools.google_api_tool import CalendarToolset  # type: ignore[import-untyped]

async def create_agent(client_id, client_secret) -> LlmAgent:
    """Constrói o agente ADK."""
    toolset = CalendarToolset(client_id=client_id, client_secret=client_secret)
    
    return LlmAgent(
        model='gemini-2.0-flash-001',
        name='calendar_agent',
        description="Um agente que pode ajudar a gerenciar o calendário de um usuário",
        instruction=f"""Você é um agente que pode ajudar a gerenciar o calendário de um usuário.
Usuários solicitarão informação sobre o estado de seu calendário ou para fazer mudanças em seu calendário.
Use as ferramentas fornecidas para interagir com a API do calendário.
Se não especificado, assuma que o calendário que o usuário quer é o calendário 'primário'.
Ao usar as ferramentas da API do Calendário, use timestamps RFC3339 bem formados.
Hoje é {datetime.datetime.now()}.""",
        tools=await toolset.get_tools(),
    )
```

Este código Python define uma função assíncrona `create_agent` que constrói um LlmAgent ADK. Ele começa inicializando um `CalendarToolset` usando as credenciais de cliente fornecidas para acessar a API do Google Calendar. Subsequentemente, uma instância `LlmAgent` é criada, configurada com um modelo Gemini especificado, um nome descritivo e instruções para gerenciar o calendário de um usuário. O agente é equipado com ferramentas de calendário do `CalendarToolset`, permitindo que interaja com a API do Calendário e responda a consultas de usuário sobre estados de calendário ou modificações. As instruções do agente incorporam dinamicamente a data atual para contexto temporal. Para ilustrar como um agente é construído, vamos examinar uma seção chave do calendar_agent encontrado nas amostras A2A no GitHub.

O código abaixo mostra como o agente é definido com suas instruções específicas e ferramentas. Por favor, note que apenas o código necessário para explicar esta funcionalidade é mostrado; você pode acessar o arquivo completo aqui: [https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/birthday_planner_adk/calendar_agent/__main__.py](https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/birthday_planner_adk/calendar_agent/__main__.py)

```python
def main(host: str, port: int): # Verificar se uma chave de API está definida. # Não necessário se usando APIs Vertex AI. if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv( 'GOOGLE_API_KEY' ): raise ValueError( 'Variável de ambiente GOOGLE_API_KEY não definida e ' 'GOOGLE_GENAI_USE_VERTEXAI não é TRUE.' ) skill = AgentSkill( id='check_availability', name='Verificar Disponibilidade', description="Verifica a disponibilidade de um usuário para um tempo usando seu Google Calendar", tags=['calendar'], examples=['Estou livre das 10h às 11h amanhã?'], ) agent_card = AgentCard( name='Agente de Calendário', description="Um agente que pode gerenciar o calendário de um usuário", url=f'http:// {
host}
: {
port}
/', version='1.0.0', defaultInputModes=['text'], defaultOutputModes=['text'], capabilities=AgentCapabilities(streaming=True), skills=[skill], ) adk_agent = asyncio.run(create_agent( client_id=os.getenv('GOOGLE_CLIENT_ID'), client_secret=os.getenv('GOOGLE_CLIENT_SECRET'), )) runner = Runner( app_name=agent_card.name, agent=adk_agent, artifact_service=InMemoryArtifactService(), session_service=InMemorySessionService(), memory_service=InMemoryMemoryService(), ) agent_executor = ADKAgentExecutor(runner, agent_card) async def handle_auth(request: Request) -> PlainTextResponse: await agent_executor.on_auth_callback( str(request.query_params.get('state')), str(request.url) ) return PlainTextResponse('Autenticação bem-sucedida.') request_handler = DefaultRequestHandler( agent_executor=agent_executor, task_store=InMemoryTaskStore() ) a2a_app = A2AStarletteApplication( agent_card=agent_card, http_handler=request_handler ) routes = a2a_app.routes() routes.append( Route( path='/authenticate', methods=['GET'], endpoint=handle_auth, ) ) app = Starlette(routes=routes) uvicorn.run(app, host=host, port=port) if __name__ == '__main__': main()
```

Este código demonstra a configuração completa de um servidor A2A usando um agente ADK. Ele começa verificando as credenciais necessárias e então define um `AgentSkill` específico para verificação de disponibilidade de calendário. O `AgentCard` é criado com informações de identidade e capacidades do agente. O agente ADK é construído usando a função `create_agent` assíncrona, e um `Runner` é configurado com serviços de memória, sessão e artefatos. O `ADKAgentExecutor` conecta o runner ao cartão de agente, e um handler de solicitação é configurado para gerenciar comunicação A2A. A aplicação Starlette é criada com rotas A2A e um endpoint de autenticação, permitindo que o servidor seja executado na porta especificada.


**Diagrama de Comunicação**

![][image2]

# Principais Conclusões

Principais Conclusões:

* O protocolo Google A2A é um padrão aberto baseado em HTTP que facilita comunicação e colaboração entre agentes de IA construídos com diferentes frameworks.  
* Um AgentCard serve como um identificador digital para um agente, permitindo descoberta automática e compreensão de suas capacidades por outros agentes.  
* A2A oferece tanto interações de solicitação-resposta síncronas (usando `tasks/send`) quanto atualizações de streaming (usando `tasks/sendSubscribe`) para acomodar necessidades variadas de comunicação.  
* O protocolo suporta conversas multi-turno, incluindo um estado `input-required`, que permite que agentes solicitem informação adicional e mantenham contexto durante interações.  
* A2A encoraja uma arquitetura modular onde agentes especializados podem operar independentemente em portas diferentes, permitindo escalabilidade e distribuição de sistema.  
* Ferramentas como Trickle AI auxiliam na visualização e rastreamento de comunicações A2A, o que ajuda desenvolvedores monitorar, debugar e otimizar sistemas multi-agente.  
* Enquanto A2A é um protocolo de alto nível para gerenciar tarefas e fluxos de trabalho entre diferentes agentes, o Protocolo de Contexto de Modelo (MCP) fornece uma interface padronizada para LLMs se conectarem com recursos externos

# Conclusões

O protocolo de Comunicação Inter-Agente (A2A) estabelece um padrão aberto vital para superar o isolamento inerente de agentes de IA individuais. Ao fornecer um framework comum baseado em HTTP, garante colaboração perfeita e interoperabilidade entre agentes construídos em diferentes plataformas, como Google ADK, LangGraph ou CrewAI. Um componente central é o Cartão de Agente, que serve como uma identidade digital, definindo claramente as capacidades de um agente e permitindo descoberta dinâmica por outros agentes. A flexibilidade do protocolo suporta vários padrões de interação, incluindo solicitações síncronas, polling assíncrono e streaming em tempo real, atendendo a uma ampla gama de necessidades de aplicação.

Isto permite a criação de arquiteturas modulares e escaláveis onde agentes especializados podem ser combinados para orquestrar fluxos de trabalho automatizados complexos. Segurança é um aspecto fundamental, com mecanismos integrados como mTLS e requisitos explícitos de autenticação para proteger comunicações. Enquanto complementa outros padrões como MCP, o foco único do A2A é na coordenação de alto nível e delegação de tarefas entre agentes. O forte apoio de grandes empresas de tecnologia e a disponibilidade de implementações práticas destacam sua crescente importância. Este protocolo pavimenta o caminho para desenvolvedores construírem sistemas multi-agente mais sofisticados, distribuídos e inteligentes. Em última análise, A2A é um pilar fundamental para fomentar um ecossistema inovador e interoperável de IA colaborativa.

# Referências

1. Chen, B. (2025, April 22). *How to Build Your First Google A2A Project: A Step-by-Step Tutorial*. Trickle.so Blog. [https://www.trickle.so/blog/how-to-build-google-a2a-project](https://www.trickle.so/blog/how-to-build-google-a2a-project)   
2. Google A2A GitHub Repository. [https://github.com/google-a2a/A2A](https://github.com/google-a2a/A2A)   
3. Google Agent Development Kit (ADK) [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)   
4. Getting Started with Agent-to-Agent (A2A) Protocol: [https://codelabs.developers.google.com/intro-a2a-purchasing-concierge\#0](https://codelabs.developers.google.com/intro-a2a-purchasing-concierge#0)   
5. Google AgentDiscovery \- [https://a2a-protocol.org/latest/](https://a2a-protocol.org/latest/)    
6. Communication between different AI frameworks such as LangGraph, CrewAI, and Google ADK [https://www.trickle.so/blog/how-to-build-google-a2a-project](https://www.trickle.so/blog/how-to-build-google-a2a-project#setting-up-your-a2a-development-environment)   
7. Designing Collaborative Multi-Agent Systems with the A2A Protocol [https://www.oreilly.com/radar/designing-collaborative-multi-agent-systems-with-the-a2a-protocol/](https://www.oreilly.com/radar/designing-collaborative-multi-agent-systems-with-the-a2a-protocol/)

[image1]: ../assets/20-chapter-15-image-1-line-149.png

[image2]: ../assets/20-chapter-15-image-2-line-151.png
