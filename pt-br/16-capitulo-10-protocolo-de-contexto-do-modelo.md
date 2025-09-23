# Capítulo 10: Protocolo de Contexto do Modelo

Para permitir que LLMs funcionem efetivamente como agentes, suas capacidades devem se estender além da geração multimodal. Interação com o ambiente externo é necessária, incluindo acesso a dados atuais, utilização de software externo e execução de tarefas operacionais específicas. O Protocolo de Contexto do Modelo (MCP) aborda esta necessidade fornecendo uma interface padronizada para LLMs interagirem com recursos externos. Este protocolo serve como um mecanismo chave para facilitar integração consistente e previsível.

# Visão Geral do Padrão MCP

Imagine um adaptador universal que permite que qualquer LLM se conecte a qualquer sistema externo, banco de dados ou ferramenta sem uma integração customizada para cada um. Isso é essencialmente o que o Protocolo de Contexto do Modelo (MCP) é. É um padrão aberto projetado para padronizar como LLMs como Gemini, modelos GPT da OpenAI, Mixtral e Claude se comunicam com aplicações externas, fontes de dados e ferramentas. Pense nisso como um mecanismo de conexão universal que simplifica como LLMs obtêm contexto, executam ações e interagem com vários sistemas.

O MCP opera em uma arquitetura cliente-servidor. Ele define como diferentes elementos—dados (referidos como recursos), templates interativos (que são essencialmente prompts), e funções acionáveis (conhecidas como ferramentas)—são expostos por um servidor MCP. Estes são então consumidos por um cliente MCP, que pode ser uma aplicação host LLM ou um agente de IA em si. Esta abordagem padronizada reduz dramaticamente a complexidade de integrar LLMs em ambientes operacionais diversos.

No entanto, o MCP é um contrato para uma "interface agêntica," e sua efetividade depende fortemente do design das APIs subjacentes que ele expõe. Existe um risco de que desenvolvedores simplesmente envolvam APIs pré-existentes e legadas sem modificação, o que pode ser subótimo para um agente. Por exemplo, se a API de um sistema de tickets apenas permite recuperar detalhes completos de tickets um por um, um agente solicitado a sumarizar tickets de alta prioridade será lento e impreciso em altos volumes. Para ser verdadeiramente efetivo, a API subjacente deve ser melhorada com características determinísticas como filtragem e ordenação para ajudar o agente não-determinístico trabalhar eficientemente. Isso destaca que agentes não substituem magicamente fluxos de trabalho determinísticos; eles frequentemente requerem suporte determinístico mais forte para ter sucesso.

Além disso, o MCP pode envolver uma API cuja entrada ou saída ainda não é inerentemente compreensível pelo agente. Uma API só é útil se seu formato de dados é amigável ao agente, uma garantia que o MCP em si não impõe. Por exemplo, criar um servidor MCP para um armazenamento de documentos que retorna arquivos como PDFs é principalmente inútil se o agente consumidor não pode fazer parse do conteúdo PDF. A melhor abordagem seria primeiro criar uma API que retorna uma versão textual do documento, como Markdown, que o agente pode realmente ler e processar. Isso demonstra que desenvolvedores devem considerar não apenas a conexão, mas a natureza dos dados sendo trocados para garantir verdadeira compatibilidade.

# MCP vs. Chamada de Função de Ferramenta

O Protocolo de Contexto do Modelo (MCP) e chamada de função de ferramenta são mecanismos distintos que permitem que LLMs interajam com capacidades externas (incluindo ferramentas) e executem ações. Embora ambos sirvam para estender capacidades LLM além da geração de texto, eles diferem em sua abordagem e nível de abstração.

A chamada de função de ferramenta pode ser pensada como uma solicitação direta de um LLM para uma ferramenta ou função específica e pré-definida. Note que neste contexto usamos as palavras "ferramenta" e "função" de forma intercambiável. Esta interação é caracterizada por um modelo de comunicação um-para-um, onde o LLM formata uma solicitação baseada em seu entendimento da intenção do usuário requerendo ação externa. O código da aplicação então executa esta solicitação e retorna o resultado ao LLM. Este processo é frequentemente proprietário e varia entre diferentes provedores LLM.

Em contraste, o Protocolo de Contexto do Modelo (MCP) opera como uma interface padronizada para LLMs descobrirem, se comunicarem com e utilizarem capacidades externas. Ele funciona como um protocolo aberto que facilita interação com uma ampla gama de ferramentas e sistemas, visando estabelecer um ecossistema onde qualquer ferramenta compatível pode ser acessada por qualquer LLM compatível. Isso fomenta interoperabilidade, composabilidade e reutilização através de diferentes sistemas e implementações. Ao adotar um modelo federado, melhoramos significativamente a interoperabilidade e desbloqueamos o valor de ativos existentes. Esta estratégia nos permite trazer serviços díspares e legados para um ecossistema moderno simplesmente envolvendo-os em uma interface compatível com MCP. Estes serviços continuam operando independentemente, mas agora podem ser compostos em novas aplicações e fluxos de trabalho, com sua colaboração orquestrada por LLMs. Isso fomenta agilidade e reutilização sem requerer reescritas custosas de sistemas fundamentais.

Aqui está uma análise das distinções fundamentais entre MCP e chamada de função de ferramenta:

| Característica | Chamada de Função de Ferramenta | Protocolo de Contexto do Modelo (MCP) |
| ----- | ----- | ----- |
| **Padronização** | Proprietário e específico do fornecedor. O formato e implementação diferem entre provedores LLM. | Um protocolo aberto e padronizado, promovendo interoperabilidade entre diferentes LLMs e ferramentas. |
| **Escopo** | Um mecanismo direto para um LLM solicitar a execução de uma função específica e pré-definida. | Um framework mais amplo para como LLMs e ferramentas externas se descobrem e se comunicam. |
| **Arquitetura** | Uma interação um-para-um entre o LLM e a lógica de manipulação de ferramentas da aplicação. | Uma arquitetura cliente-servidor onde aplicações alimentadas por LLM (clientes) podem se conectar a e utilizar vários servidores MCP (ferramentas). |
| **Descoberta** | O LLM é explicitamente informado quais ferramentas estão disponíveis no contexto de uma conversa específica. | Permite descoberta dinâmica de ferramentas disponíveis. Um cliente MCP pode consultar um servidor para ver que capacidades ele oferece. |
| **Reutilização** | Integrações de ferramentas são frequentemente fortemente acopladas com a aplicação específica e LLM sendo usados. | Promove o desenvolvimento de "servidores MCP" reutilizáveis e independentes que podem ser acessados por qualquer aplicação compatível. |

Pense na chamada de função de ferramenta como dar a uma IA um conjunto específico de ferramentas customizadas, como uma chave inglesa e chave de fenda particulares. Isso é eficiente para uma oficina com um conjunto fixo de tarefas. O MCP (Protocolo de Contexto do Modelo), por outro lado, é como criar um sistema universal e padronizado de tomadas de energia. Ele não fornece as ferramentas em si, mas permite que qualquer ferramenta compatível de qualquer fabricante se conecte e funcione, permitindo uma oficina dinâmica e sempre expansível.

Em resumo, chamada de função fornece acesso direto a algumas funções específicas, enquanto MCP é o framework de comunicação padronizado que permite que LLMs descubram e usem uma vasta gama de recursos externos. Para aplicações simples, ferramentas específicas são suficientes; para sistemas de IA complexos e interconectados que precisam se adaptar, um padrão universal como MCP é essencial.

# Considerações Adicionais para MCP

Embora o MCP apresente um framework poderoso, uma avaliação completa requer considerar vários aspectos cruciais que influenciam sua adequação para um caso de uso dado. Vamos ver alguns aspectos em mais detalhes:

* **Ferramenta vs. Recurso vs. Prompt**: É importante entender os papéis específicos destes componentes. Um recurso são dados estáticos (ex., um arquivo PDF, um registro de banco de dados). Uma ferramenta é uma função executável que executa uma ação (ex., enviando um email, consultando uma API). Um prompt é um template que guia o LLM em como interagir com um recurso ou ferramenta, garantindo que a interação seja estruturada e efetiva.  
* **Descobertabilidade**: Uma vantagem chave do MCP é que um cliente MCP pode dinamicamente consultar um servidor para aprender que ferramentas e recursos ele oferece. Este mecanismo de descoberta "just-in-time" é poderoso para agentes que precisam se adaptar a novas capacidades sem serem reimplantados.  
* **Segurança**: Expor ferramentas e dados via qualquer protocolo requer medidas de segurança robustas. Uma implementação MCP deve incluir autenticação e autorização para controlar quais clientes podem acessar quais servidores e que ações específicas eles são permitidos a executar.  
* **Implementação**: Embora o MCP seja um padrão aberto, sua implementação pode ser complexa. No entanto, provedores estão começando a simplificar este processo. Por exemplo, alguns provedores de modelo como Anthropic ou FastMCP oferecem SDKs que abstraem muito do código boilerplate, tornando mais fácil para desenvolvedores criarem e conectarem clientes e servidores MCP.  
* **Tratamento de Erros**: Uma estratégia abrangente de tratamento de erros é crítica. O protocolo deve definir como erros (ex., falha de execução de ferramenta, servidor indisponível, solicitação inválida) são comunicados de volta ao LLM para que ele possa entender a falha e potencialmente tentar uma abordagem alternativa.  
* **Servidor Local vs. Remoto**: Servidores MCP podem ser implantados localmente na mesma máquina que o agente ou remotamente em um servidor diferente. Um servidor local pode ser escolhido por velocidade e segurança com dados sensíveis, enquanto uma arquitetura de servidor remoto permite acesso compartilhado e escalável a ferramentas comuns através de uma organização.  
* **Sob Demanda vs. Lote**: O MCP pode suportar tanto sessões interativas sob demanda quanto processamento de lote em maior escala. A escolha depende da aplicação, de um agente conversacional em tempo real precisando acesso imediato a ferramentas a um pipeline de análise de dados que processa registros em lotes.  
* **Mecanismo de Transporte**: O protocolo também define as camadas de transporte subjacentes para comunicação. Para interações locais, ele usa JSON-RPC sobre STDIO (entrada/saída padrão) para comunicação inter-processo eficiente. Para conexões remotas, ele aproveita protocolos amigáveis à web como HTTP Streamable e Server-Sent Events (SSE) para permitir comunicação cliente-servidor persistente e eficiente.

O Protocolo de Contexto do Modelo usa um modelo cliente-servidor para padronizar fluxo de informação. Entender interação de componentes é chave para o comportamento agêntico avançado do MCP:

1. **Modelo de Linguagem Grande (LLM)**: A inteligência central. Ele processa solicitações do usuário, formula planos e decide quando precisa acessar informação externa ou executar uma ação.  
2. **Cliente MCP**: Esta é uma aplicação ou wrapper ao redor do LLM. Ele atua como o intermediário, traduzindo a intenção do LLM em uma solicitação formal que conforma ao padrão MCP. É responsável por descobrir, conectar e se comunicar com Servidores MCP.  
3. **Servidor MCP**: Esta é a porta de entrada para o mundo externo. Ele expõe um conjunto de ferramentas, recursos e prompts para qualquer Cliente MCP autorizado. Cada servidor é tipicamente responsável por um domínio específico, como uma conexão com o banco de dados interno de uma empresa, um serviço de email ou uma API pública.  
4. ​​**Serviço de Terceiros (3P) Opcional:** Isto representa a ferramenta externa real, aplicação ou fonte de dados que o Servidor MCP gerencia e expõe. É o endpoint final que executa a ação solicitada, como consultar um banco de dados proprietário, interagir com uma plataforma SaaS ou chamar uma API pública de clima.

A interação flui da seguinte forma:

1. **Descoberta**: O Cliente MCP, em nome do LLM, consulta um Servidor MCP para perguntar que capacidades ele oferece. O servidor responde com um manifesto listando suas ferramentas disponíveis (ex., send_email), recursos (ex., customer_database) e prompts.  
2. **Formulação de Solicitação**: O LLM determina que precisa usar uma das ferramentas descobertas. Por exemplo, ele decide enviar um email. Ele formula uma solicitação, especificando a ferramenta a usar (send_email) e os parâmetros necessários (destinatário, assunto, corpo).  
3. **Comunicação do Cliente**: O Cliente MCP toma a solicitação formulada do LLM e a envia como uma chamada padronizada para o Servidor MCP apropriado.  
4. **Execução do Servidor**: O Servidor MCP recebe a solicitação. Ele autentica o cliente, valida a solicitação e então executa a ação especificada interfaciando com o software subjacente (ex., chamando a função send() de uma API de email).  
5. **Resposta e Atualização de Contexto**: Após execução, o Servidor MCP envia uma resposta padronizada de volta ao Cliente MCP. Esta resposta indica se a ação foi bem-sucedida e inclui qualquer saída relevante (ex., um ID de confirmação para o email enviado). O cliente então passa este resultado de volta ao LLM, atualizando seu contexto e permitindo que ele prossiga com o próximo passo de sua tarefa.

# Aplicações Práticas e Casos de Uso

O MCP amplia significativamente as capacidades de IA/LLM, tornando-as mais versáteis e poderosas. Aqui estão nove casos de uso chave:

* **Integração de Banco de Dados:** O MCP permite que LLMs e agentes acessem e interajam perfeitamente com dados estruturados em bancos de dados. Por exemplo, usando o MCP Toolbox para Bancos de Dados, um agente pode consultar datasets do Google BigQuery para recuperar informação em tempo real, gerar relatórios ou atualizar registros, tudo impulsionado por comandos de linguagem natural.  
* **Orquestração de Mídia Generativa:** O MCP permite que agentes se integrem com serviços avançados de mídia generativa. Através de Ferramentas MCP para Serviços Genmedia, um agente pode orquestrar fluxos de trabalho envolvendo Imagen do Google para geração de imagem, Veo do Google para criação de vídeo, Chirp 3 HD do Google para vozes realistas, ou Lyria do Google para composição musical, permitindo criação de conteúdo dinâmico dentro de aplicações de IA.  
* **Interação com API Externa:** O MCP fornece uma maneira padronizada para LLMs chamarem e receberem respostas de qualquer API externa. Isso significa que um agente pode buscar dados de clima ao vivo, puxar preços de ações, enviar emails ou interagir com sistemas CRM, estendendo suas capacidades muito além de seu modelo de linguagem central.  
* **Extração de Informação Baseada em Raciocínio:** Aproveitando as fortes habilidades de raciocínio de um LLM, o MCP facilita extração efetiva de informação dependente de consulta que supera sistemas convencionais de busca e recuperação. Ao invés de uma ferramenta de busca tradicional retornando um documento inteiro, um agente pode analisar o texto e extrair a cláusula, figura ou declaração precisa que responde diretamente a uma pergunta complexa do usuário.  
* **Desenvolvimento de Ferramenta Customizada:** Desenvolvedores podem construir ferramentas customizadas e expô-las via um servidor MCP (ex., usando FastMCP). Isso permite que funções internas especializadas ou sistemas proprietários sejam disponibilizados para LLMs e outros agentes em um formato padronizado e facilmente consumível, sem precisar modificar o LLM diretamente.  
* **Comunicação Padronizada LLM-para-Aplicação:** O MCP garante uma camada de comunicação consistente entre LLMs e as aplicações com as quais eles interagem. Isso reduz sobrecarga de integração, promove interoperabilidade entre diferentes provedores LLM e aplicações host, e simplifica o desenvolvimento de sistemas agênticos complexos.  
* **Orquestração de Fluxo de Trabalho Complexo:** Ao combinar várias ferramentas e fontes de dados expostas por MCP, agentes podem orquestrar fluxos de trabalho altamente complexos e multi-etapas. Um agente poderia, por exemplo, recuperar dados de cliente de um banco de dados, gerar uma imagem de marketing personalizada, redigir um email personalizado, e então enviá-lo, tudo interagindo com diferentes serviços MCP.  
* **Controle de Dispositivo IoT:** O MCP pode facilitar interação LLM com dispositivos Internet of Things (IoT). Um agente poderia usar MCP para enviar comandos a eletrodomésticos inteligentes, sensores industriais ou robótica, permitindo controle e automação de sistemas físicos em linguagem natural.  
* **Automação de Serviços Financeiros:** Em serviços financeiros, o MCP poderia permitir que LLMs interajam com várias fontes de dados financeiros, plataformas de trading ou sistemas de conformidade. Um agente pode analisar dados de mercado, executar trades, gerar conselhos financeiros personalizados ou automatizar relatórios regulatórios, tudo mantendo comunicação segura e padronizada.

Em resumo, o Protocolo de Contexto do Modelo (MCP) permite que agentes acessem informação em tempo real de bancos de dados, APIs e recursos web. Ele também permite que agentes executem ações como enviar emails, atualizar registros, controlar dispositivos e executar tarefas complexas integrando e processando dados de várias fontes. Adicionalmente, o MCP suporta ferramentas de geração de mídia para aplicações de IA.

# Exemplo de Código Hands-On com ADK

Esta seção descreve como conectar a um servidor MCP local que fornece operações de sistema de arquivos, permitindo que um agente ADK interaja com o sistema de arquivos local.

## Configuração do Agente com MCPToolset

Para configurar um agente para interação com sistema de arquivos, um arquivo `agent.py` deve ser criado (ex., em `./adk_agent_samples/mcp_agent/agent.py`). O `MCPToolset` é instanciado dentro da lista `tools` do objeto `LlmAgent`. É crucial substituir `"/path/to/your/folder"` na lista `args` pelo caminho absoluto para um diretório no sistema local que o servidor MCP pode acessar. Este diretório será a raiz para as operações de sistema de arquivos executadas pelo agente.

```python
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Criar um caminho absoluto confiável para uma pasta nomeada 'mcp_managed_files'
# dentro do mesmo diretório deste script de agente.
# Isso garante que o agente funcione out-of-the-box para demonstração.
# Para produção, você apontaria isso para um local mais persistente e seguro.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_managed_files")

# Garantir que o diretório alvo existe antes do agente precisar dele.
os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='filesystem_assistant_agent',
    instruction=(
        'Ajude o usuário a gerenciar seus arquivos. Você pode listar arquivos, ler arquivos e escrever arquivos. '
        f'Você está operando no seguinte diretório: {TARGET_FOLDER_PATH}'
    ),
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y",  # Argumento para npx auto-confirmar instalação
                    "@modelcontextprotocol/server-filesystem",
                    # Isto DEVE ser um caminho absoluto para uma pasta.
                    TARGET_FOLDER_PATH,
                ],
            ),
            # Opcional: Você pode filtrar quais ferramentas do servidor MCP são expostas.
            # Por exemplo, para apenas permitir leitura:
            # tool_filter=['list_directory', 'read_file']
        )
    ],
)
```

`npx` (Node Package Execute), empacotado com versões npm (Node Package Manager) 5.2.0 e posteriores, é uma utilidade que permite execução direta de pacotes Node.js do registro npm. Isso elimina a necessidade de instalação global. Em essência, `npx` serve como um executor de pacote npm, e é comumente usado para executar muitos servidores MCP da comunidade, que são distribuídos como pacotes Node.js.

Criar um arquivo `__init__.py` é necessário para garantir que o arquivo agent.py seja reconhecido como parte de um pacote Python descobrível para o Agent Development Kit (ADK). Este arquivo deve residir no mesmo diretório que agent.py.

```python
# ./adk_agent_samples/mcp_agent/__init__.py
from . import agent
```

Certamente, outros comandos suportados estão disponíveis para uso. Por exemplo, conectar ao python3 pode ser alcançado da seguinte forma:

```python
connection_params = StdioConnectionParams(
    server_params={
        "command": "python3",
        "args": ["./agent/mcp_server.py"],
        "env": {
            "SERVICE_ACCOUNT_PATH": SERVICE_ACCOUNT_PATH,
            "DRIVE_FOLDER_ID": DRIVE_FOLDER_ID
        }
    }
)
```

UVX, no contexto de Python, refere-se a uma ferramenta de linha de comando que utiliza uv para executar comandos em um ambiente Python temporário e isolado. Essencialmente, permite que você execute ferramentas e pacotes Python sem precisar instalá-los globalmente ou dentro do ambiente do seu projeto. Você pode executá-lo via servidor MCP.

```python
connection_params = StdioConnectionParams(
    server_params={
        "command": "uvx",
        "args": ["mcp-google-sheets@latest"],
        "env": {
            "SERVICE_ACCOUNT_PATH": SERVICE_ACCOUNT_PATH,
            "DRIVE_FOLDER_ID": DRIVE_FOLDER_ID
        }
    }
)
```

Uma vez que o Servidor MCP é criado, o próximo passo é conectar a ele.

## Conectando o Servidor MCP com ADK Web

Para começar, execute 'adk web'. Navegue até o diretório pai de mcp_agent (ex., adk_agent_samples) no seu terminal e execute:

```bash
cd ./adk_agent_samples # Ou seu diretório pai equivalente adk web
```

Uma vez que a UI ADK Web carregou no seu navegador, selecione o `filesystem_assistant_agent` do menu de agentes. Em seguida, experimente com prompts como:

* "Mostre-me o conteúdo desta pasta."  
* "Leia o arquivo `sample.txt`." (Isso assume que `sample.txt` está localizado em `TARGET_FOLDER_PATH`.)  
* "O que há em `another_file.md`?"

## Criando um Servidor MCP com FastMCP

FastMCP é um framework Python de alto nível projetado para agilizar o desenvolvimento de servidores MCP. Ele fornece uma camada de abstração que simplifica complexidades de protocolo, permitindo que desenvolvedores se concentrem na lógica principal.

A biblioteca permite definição rápida de ferramentas, recursos e prompts usando decoradores Python simples. Uma vantagem significativa é sua geração automática de esquema, que interpreta inteligentemente assinaturas de função Python, dicas de tipo e strings de documentação para construir especificações de interface de modelo de IA necessárias. Esta automação minimiza configuração manual e reduz erro humano.

Além da criação básica de ferramentas, FastMCP facilita padrões arquiteturais avançados como composição de servidor e proxy. Isso permite desenvolvimento modular de sistemas complexos e multi-componentes e integração perfeita de serviços existentes em um framework acessível por IA. Adicionalmente, FastMCP inclui otimizações para aplicações eficientes, distribuídas e escaláveis impulsionadas por IA.

## Configuração do servidor com FastMCP

## Para ilustrar, considere uma ferramenta básica "greet" fornecida pelo servidor. Agentes ADK e outros clientes MCP podem interagir com esta ferramenta usando HTTP uma vez que ela esteja ativa.

```python
# fastmcp_server.py
# Este script demonstra como criar um servidor MCP simples usando FastMCP.
# Ele expõe uma única ferramenta que gera uma saudação.

# 1. Certifique-se de ter o FastMCP instalado:
# pip install fastmcp
from fastmcp import FastMCP, Client

# Inicializar o servidor FastMCP.
mcp_server = FastMCP()

# Definir uma função de ferramenta simples.
# O decorador `@mcp_server.tool` registra esta função Python como uma ferramenta MCP.
# A docstring torna-se a descrição da ferramenta para o LLM.
@mcp_server.tool
def greet(name: str) -> str:
    """
    Gera uma saudação personalizada.
    
    Args:
        name: O nome da pessoa a cumprimentar.
    
    Returns:
        Uma string de saudação.
    """
    return f"Olá, {name}! Prazer em conhecê-lo."

# Ou se você quiser executá-lo a partir do script:
if __name__ == "__main__":
    mcp_server.run(
        transport="http",
        host="127.0.0.1",
        port=8000
    )
```

Este script Python define uma única função chamada greet, que toma o nome de uma pessoa e retorna uma saudação personalizada. O decorador @tool() acima desta função automaticamente a registra como uma ferramenta que uma IA ou outro programa pode usar. A string de documentação da função e dicas de tipo são usadas pelo FastMCP para dizer ao Agente como a ferramenta funciona, que entradas ela precisa e o que retornará.

Quando o script é executado, ele inicia o servidor FastMCP, que escuta por solicitações em localhost:8000. Isso torna a função greet disponível como um serviço de rede. Um agente poderia então ser configurado para conectar a este servidor e usar a ferramenta greet para gerar saudações como parte de uma tarefa maior. O servidor executa continuamente até ser parado manualmente.

## Consumindo o Servidor FastMCP com um Agente ADK

Um agente ADK pode ser configurado como um cliente MCP para usar um servidor FastMCP em execução. Isso requer configurar HttpServerParameters com o endereço de rede do servidor FastMCP, que é geralmente http://localhost:8000.

Um parâmetro tool_filter pode ser incluído para restringir o uso de ferramentas do agente a ferramentas específicas oferecidas pelo servidor, como 'greet'. Quando solicitado com uma solicitação como "Cumprimente John Doe," o LLM incorporado do agente identifica a ferramenta 'greet' disponível via MCP, a invoca com o argumento "John Doe," e retorna a resposta do servidor. Este processo demonstra a integração de ferramentas definidas pelo usuário expostas através do MCP com um agente ADK.

Para estabelecer esta configuração, um arquivo de agente (ex., agent.py localizado em ./adk_agent_samples/fastmcp_client_agent/) é requerido. Este arquivo instanciará um agente ADK e usará HttpServerParameters para estabelecer uma conexão com o servidor FastMCP operacional.

```python
# ./adk_agent_samples/fastmcp_client_agent/agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, HttpServerParameters

# Definir o endereço do servidor FastMCP.
# Certifique-se de que seu fastmcp_server.py (definido anteriormente) esteja executando nesta porta.
FASTMCP_SERVER_URL = "http://localhost:8000"

root_agent = LlmAgent(
    model='gemini-2.0-flash',  # Ou seu modelo preferido
    name='fastmcp_greeter_agent',
    instruction='Você é um assistente amigável que pode cumprimentar pessoas pelo nome. Use a ferramenta "greet".',
    tools=[
        MCPToolset(
            connection_params=HttpServerParameters(
                url=FASTMCP_SERVER_URL,
            ),
            # Opcional: Filtrar quais ferramentas do servidor MCP são expostas
            # Para este exemplo, esperamos apenas 'greet'
            tool_filter=['greet']
        )
    ],
)
```

O script define um Agente nomeado fastmcp_greeter_agent que usa um modelo de linguagem Gemini. Ele recebe uma instrução específica para atuar como um assistente amigável cujo propósito é cumprimentar pessoas. Crucialmente, o código equipa este agente com uma ferramenta para executar sua tarefa. Ele configura um MCPToolset para conectar a um servidor separado executando em localhost:8000, que espera ser o servidor FastMCP do exemplo anterior. O agente é especificamente concedido acesso à ferramenta greet hospedada naquele servidor. Em essência, este código configura o lado cliente do sistema, criando um agente inteligente que entende que seu objetivo é cumprimentar pessoas e sabe exatamente qual ferramenta externa usar para realizá-lo.

Criar um arquivo `__init__.py` dentro do diretório fastmcp_client_agent é necessário. Isso garante que o agente seja reconhecido como um pacote Python descobrível para o ADK.

Para começar, abra um novo terminal e execute `python fastmcp_server.py` para iniciar o servidor FastMCP. Em seguida, vá para o diretório pai de `fastmcp_client_agent` (por exemplo, `adk_agent_samples`) no seu terminal e execute `adk web`. Uma vez que a UI ADK Web carregue no seu navegador, selecione o `fastmcp_greeter_agent` do menu de agentes. Você pode então testá-lo inserindo um prompt como "Cumprimente John Doe." O agente usará a ferramenta `greet` no seu servidor FastMCP para criar uma resposta.

# Em Resumo

**O que:** Para funcionar como agentes efetivos, LLMs devem se mover além da geração simples de texto. Eles requerem a capacidade de interagir com o ambiente externo para acessar dados atuais e utilizar software externo. Sem um método de comunicação padronizado, cada integração entre um LLM e uma ferramenta ou fonte de dados externa torna-se um esforço customizado, complexo e não reutilizável. Esta abordagem ad-hoc dificulta escalabilidade e torna difícil e ineficiente construir sistemas de IA complexos e interconectados.

**Por quê:** O Protocolo de Contexto do Modelo (MCP) oferece uma solução padronizada atuando como uma interface universal entre LLMs e sistemas externos. Ele estabelece um protocolo aberto e padronizado que define como capacidades externas são descobertas e usadas. Operando em um modelo cliente-servidor, o MCP permite que servidores exponham ferramentas, recursos de dados e prompts interativos para qualquer cliente compatível. Aplicações alimentadas por LLM atuam como estes clientes, descobrindo dinamicamente e interagindo com recursos disponíveis de uma maneira previsível. Esta abordagem padronizada fomenta um ecossistema de componentes interoperáveis e reutilizáveis, simplificando dramaticamente o desenvolvimento de fluxos de trabalho agênticos complexos.

**Regra de ouro:** Use o Protocolo de Contexto do Modelo (MCP) ao construir sistemas agênticos complexos, escaláveis ou de nível empresarial que precisam interagir com um conjunto diverso e evolutivo de ferramentas externas, fontes de dados e APIs. É ideal quando interoperabilidade entre diferentes LLMs e ferramentas é uma prioridade, e quando agentes requerem a capacidade de descobrir dinamicamente novas capacidades sem serem reimplantados. Para aplicações mais simples com um número fixo e limitado de funções pré-definidas, chamada direta de função de ferramenta pode ser suficiente.

**Resumo visual![][image1]**

Fig.1: Protocolo de Contexto do Modelo

# Principais Conclusões

Estas são as principais conclusões:

* O Protocolo de Contexto do Modelo (MCP) é um padrão aberto facilitando comunicação padronizada entre LLMs e aplicações externas, fontes de dados e ferramentas.  
* Ele emprega uma arquitetura cliente-servidor, definindo os métodos para expor e consumir recursos, prompts e ferramentas.  
* O Agent Development Kit (ADK) suporta tanto utilizar servidores MCP existentes quanto expor ferramentas ADK via um servidor MCP.  
* FastMCP simplifica o desenvolvimento e gerenciamento de servidores MCP, particularmente para expor ferramentas implementadas em Python.  
* Ferramentas MCP para Serviços Genmedia permite que agentes se integrem com capacidades de mídia generativa do Google Cloud (Imagen, Veo, Chirp 3 HD, Lyria).  
* MCP permite que LLMs e agentes interajam com sistemas do mundo real, acessem informação dinâmica e executem ações além da geração de texto.

# Conclusão

O Protocolo de Contexto do Modelo (MCP) é um padrão aberto que facilita comunicação entre Modelos de Linguagem Grandes (LLMs) e sistemas externos. Ele emprega uma arquitetura cliente-servidor, permitindo que LLMs acessem recursos, utilizem prompts e executem ações através de ferramentas padronizadas. O MCP permite que LLMs interajam com bancos de dados, gerenciem fluxos de trabalho de mídia generativa, controlem dispositivos IoT e automatizem serviços financeiros. Exemplos práticos demonstram configurar agentes para se comunicar com servidores MCP, incluindo servidores de sistema de arquivos e servidores construídos com FastMCP, ilustrando sua integração com o Agent Development Kit (ADK). O MCP é um componente chave para desenvolver agentes de IA interativos que se estendem além de capacidades básicas de linguagem.

# Referências

1. Model Context Protocol (MCP) Documentation. (Latest). *Model Context Protocol (MCP)*. [https://google.github.io/adk-docs/mcp/](https://google.github.io/adk-docs/mcp/)  
2. FastMCP Documentation. FastMCP. [https://github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)  
3. MCP Tools for Genmedia Services. *MCP Tools for Genmedia Services*. [https://google.github.io/adk-docs/mcp/#mcp-servers-for-google-cloud-genmedia](https://google.github.io/adk-docs/mcp/#mcp-servers-for-google-cloud-genmedia)  
4. MCP Toolbox for Databases Documentation. (Latest). *MCP Toolbox for Databases*. [https://google.github.io/adk-docs/mcp/databases/](https://google.github.io/adk-docs/mcp/databases/)

[image1]: ../assets/15-chapter-10-image-1-line-194.png
