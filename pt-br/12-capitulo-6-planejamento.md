# Capítulo 6: Planejamento

Comportamento inteligente frequentemente envolve mais do que simplesmente reagir à entrada imediata. Requer previsão, quebrar tarefas complexas em passos menores e gerenciáveis, e estratégias de como alcançar um resultado desejado. É aqui que o padrão de Planejamento entra em jogo. Em sua essência, planejamento é a capacidade para um agente ou um sistema de agentes formular uma sequência de ações para mover de um estado inicial em direção a um estado objetivo.

# Visão Geral do Padrão de Planejamento

No contexto da IA, é útil pensar em um agente de planejamento como um especialista ao qual você delega um objetivo complexo. Quando você pede para ele "organizar um offsite da equipe", você está definindo o que—o objetivo e suas restrições—mas não o como. A tarefa central do agente é traçar autonomamente um curso para esse objetivo. Ele deve primeiro entender o estado inicial (ex., orçamento, número de participantes, datas desejadas) e o estado objetivo (um offsite com sucesso reservado), e então descobrir a sequência ótima de ações para conectá-los. O plano não é conhecido antecipadamente; ele é criado em resposta à solicitação.

Uma característica marcante deste processo é a adaptabilidade. Um plano inicial é meramente um ponto de partida, não um script rígido. O verdadeiro poder do agente é sua capacidade de incorporar nova informação e direcionar o projeto ao redor de obstáculos. Por exemplo, se o local preferido se torna indisponível ou um fornecedor de catering escolhido está completamente reservado, um agente capaz não simplesmente falha. Ele se adapta. Ele registra a nova restrição, re-avalia suas opções, e formula um novo plano, talvez sugerindo locais alternativos ou datas.

No entanto, é crucial reconhecer o trade-off entre flexibilidade e previsibilidade. Planejamento dinâmico é uma ferramenta específica, não uma solução universal. Quando a solução de um problema já é bem compreendida e repetível, restringir o agente a um workflow predeterminado e fixo é mais eficaz. Esta abordagem limita a autonomia do agente para reduzir incerteza e o risco de comportamento imprevisível, garantindo um resultado confiável e consistente. Portanto, a decisão de usar um agente de planejamento versus um agente de execução de tarefas simples depende de uma única pergunta: o "como" precisa ser descoberto, ou já é conhecido?

# Aplicações Práticas e Casos de Uso

O padrão de Planejamento é um processo computacional central em sistemas autônomos, permitindo que um agente sintetize uma sequência de ações para alcançar um objetivo especificado, particularmente dentro de ambientes dinâmicos ou complexos. Este processo transforma um objetivo de alto nível em um plano estruturado composto de passos discretos e executáveis.

Em domínios como automação de tarefas procedurais, planejamento é usado para orquestrar workflows complexos. Por exemplo, um processo de negócio como integração de um novo funcionário pode ser decomposto em uma sequência direcionada de sub-tarefas, como criar contas de sistema, atribuir módulos de treinamento, e coordenar com diferentes departamentos. O agente gera um plano para executar estes passos em uma ordem lógica, invocando ferramentas necessárias ou interagindo com vários sistemas para gerenciar dependências.

Dentro de robótica e navegação autônoma, planejamento é fundamental para travessia de espaço de estados. Um sistema, seja um robô físico ou uma entidade virtual, deve gerar um caminho ou sequência de ações para transicionar de um estado inicial para um estado objetivo. Isso envolve otimizar para métricas como tempo ou consumo de energia enquanto adere a restrições ambientais, como evitar obstáculos ou seguir regulamentos de tráfego.

Este padrão também é crítico para síntese de informação estruturada. Quando encarregado de gerar uma saída complexa como um relatório de pesquisa, um agente pode formular um plano que inclui fases distintas para coleta de informação, sumarização de dados, estruturação de conteúdo, e refinamento iterativo. Similarmente, em cenários de suporte ao cliente envolvendo resolução de problemas multi-passo, um agente pode criar e seguir um plano sistemático para diagnóstico, implementação de solução, e escalação.

Em essência, o padrão de Planejamento permite que um agente se mova além de ações simples e reativas para comportamento orientado a objetivos. Ele fornece a estrutura lógica necessária para resolver problemas que requerem uma sequência coerente de operações interdependentes.

# Código Hands-on (Crew AI)

A seguinte seção demonstrará uma implementação do padrão Planner usando o framework Crew AI. Este padrão envolve um agente que primeiro formula um plano multi-passo para abordar uma consulta complexa e então executa esse plano sequencialmente.

```python
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Carregar variáveis de ambiente do arquivo .env para segurança
load_dotenv()

# 1. Definir explicitamente o modelo de linguagem para clareza
llm = ChatOpenAI(model="gpt-4-turbo")

# 2. Definir um agente claro e focado
planner_writer_agent = Agent(
    role='Article Planner and Writer',
    goal='Planejar e então escrever um resumo conciso e envolvente sobre um tópico especificado.',
    backstory=(
        'Você é um escritor técnico experiente e estrategista de conteúdo. '
        'Sua força está em criar um plano claro e acionável antes de escrever, '
        'garantindo que o resumo final seja tanto informativo quanto fácil de digerir.'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm  # Atribuir o LLM específico ao agente
)

# 3. Definir uma tarefa com uma saída esperada mais estruturada e específica
topic = "A importância do Reinforcement Learning na IA"
high_level_task = Task(
    description=(
        f"1. Criar um plano de pontos para um resumo sobre o tópico: '{topic}'.\n"
        f"2. Escrever o resumo baseado em seu plano, mantendo-o em torno de 200 palavras."
    ),
    expected_output=(
        "Um relatório final contendo duas seções distintas:\n\n"
        "### Plano\n"
        "- Uma lista com marcadores delineando os pontos principais do resumo.\n\n"
        "### Resumo\n"
        "- Um resumo conciso e bem estruturado do tópico."
    ),
    agent=planner_writer_agent,
)

# Criar o crew com um processo claro
crew = Crew(
    agents=[planner_writer_agent],
    tasks=[high_level_task],
    process=Process.sequential,
)

# Executar a tarefa
print("## Executando a tarefa de planejamento e escrita ##")
result = crew.kickoff()
print("\n\n---\n## Resultado da Tarefa ##\n---")
print(result)
```

Este código usa a biblioteca CrewAI para criar um agente de IA que planeja e escreve um resumo sobre um tópico dado. Ele começa importando bibliotecas necessárias, incluindo Crew.ai e langchain_openai, e carregando variáveis de ambiente de um arquivo .env. Um modelo de linguagem ChatOpenAI é explicitamente definido para uso com o agente. Um Agente nomeado planner_writer_agent é criado com um papel e objetivo específicos: planejar e então escrever um resumo conciso. A história do agente enfatiza sua expertise em planejamento e escrita técnica. Uma Tarefa é definida com uma descrição clara para primeiro criar um plano e então escrever um resumo sobre o tópico "A importância do Reinforcement Learning na IA", com um formato específico para a saída esperada. Um Crew é montado com o agente e tarefa, configurado para processá-los sequencialmente. Finalmente, o método crew.kickoff() é chamado para executar a tarefa definida e o resultado é impresso.

# Google DeepResearch

Google Gemini DeepResearch (veja Fig.1) é um sistema baseado em agentes projetado para recuperação e síntese autônoma de informação. Ele funciona através de um pipeline agêntico multi-passo que consulta dinâmica e iterativamente o Google Search para explorar sistematicamente tópicos complexos. O sistema é projetado para processar um grande corpus de fontes baseadas na web, avaliar os dados coletados para relevância e lacunas de conhecimento, e realizar buscas subsequentes para abordá-las. A saída final consolida a informação verificada em um resumo estruturado multi-página com citações para as fontes originais.

Expandindo isso, a operação do sistema não é um único evento de consulta-resposta, mas um processo gerenciado e de longa duração. Ele começa desconstruindo o prompt do usuário em um plano de pesquisa multi-ponto (veja Fig. 1), que é então apresentado ao usuário para revisão e modificação. Isso permite uma modelagem colaborativa da trajetória de pesquisa antes da execução. Uma vez que o plano é aprovado, o pipeline agêntico inicia seu loop de busca-e-análise iterativo. Isso envolve mais do que apenas executar uma série de buscas pré-definidas; o agente formula e refina dinamicamente suas consultas baseado na informação que coleta, identificando ativamente lacunas de conhecimento, corroborando pontos de dados, e resolvendo discrepâncias.

![][image1]  
Fig. 1: Agente Google Deep Research gerando um plano de execução para usar Google Search como ferramenta.

Um componente arquitetural chave é a capacidade do sistema de gerenciar este processo de forma assíncrona. Este design garante que a investigação, que pode envolver analisar centenas de fontes, seja resiliente a falhas de ponto único e permite que o usuário se desengaje e seja notificado na conclusão. O sistema também pode integrar documentos fornecidos pelo usuário, combinando informação de fontes privadas com sua pesquisa baseada na web. A saída final não é meramente uma lista concatenada de achados, mas um relatório estruturado multi-página. Durante a fase de síntese, o modelo executa uma avaliação crítica da informação coletada, identificando temas principais e organizando o conteúdo em uma narrativa coerente com seções lógicas. O relatório é projetado para ser interativo, frequentemente incluindo recursos como uma visão geral de áudio, gráficos, e links para as fontes citadas originais, permitindo verificação e exploração adicional pelo usuário. Além dos resultados sintetizados, o modelo retorna explicitamente a lista completa de fontes que pesquisou e consultou (veja Fig.2). Estas são apresentadas como citações, fornecendo transparência completa e acesso direto à informação primária. Este processo inteiro transforma uma consulta simples em um corpo abrangente e sintetizado de conhecimento.

![][image2]  
Fig. 2: Um exemplo de plano de Deep Research sendo executado, resultando em Google Search sendo usado como ferramenta para pesquisar várias fontes web.

Ao mitigar o substancial investimento de tempo e recursos requerido para aquisição manual de dados e síntese, Gemini DeepResearch fornece um método mais estruturado e exaustivo para descoberta de informação. O valor do sistema é particularmente evidente em tarefas de pesquisa complexas e multi-facetadas através de vários domínios.

Por exemplo, em análise competitiva, o agente pode ser direcionado a reunir e colacionar sistematicamente dados sobre tendências de mercado, especificações de produtos de concorrentes, sentimento público de diversas fontes online, e estratégias de marketing. Este processo automatizado substitui a tarefa laboriosa de rastrear manualmente múltiplos concorrentes, permitindo que analistas se foquem em interpretação estratégica de ordem superior ao invés de coleta de dados (veja Fig. 3).

![][image3]  
Fig. 3: Saída final gerada pelo agente Google Deep Research, analisando em nosso nome fontes obtidas usando Google Search como ferramenta.

Similarmente, em exploração acadêmica, o sistema serve como uma ferramenta poderosa para conduzir extensas revisões de literatura. Ele pode identificar e resumir artigos fundamentais, traçar o desenvolvimento de conceitos através de numerosas publicações, e mapear frentes de pesquisa emergentes dentro de um campo específico, acelerando assim a fase inicial e mais demorada da investigação acadêmica.

A eficiência desta abordagem deriva da automação do ciclo de busca-e-filtro iterativo, que é um gargalo central na pesquisa manual. Completude é alcançada pela capacidade do sistema de processar um volume maior e variedade de fontes de informação do que é tipicamente viável para um pesquisador humano dentro de um timeframe comparável. Este escopo mais amplo de análise ajuda a reduzir o potencial para viés de seleção e aumenta a probabilidade de descobrir informação menos óbvia mas potencialmente crítica, levando a um entendimento mais robusto e bem suportado do assunto.

# OpenAI Deep Research API

A OpenAI Deep Research API é uma ferramenta especializada projetada para automatizar tarefas de pesquisa complexas. Ela utiliza um modelo agêntico avançado que pode raciocinar, planejar e sintetizar independentemente informação de fontes do mundo real. Diferente de um modelo simples de Q&A, ela recebe uma consulta de alto nível e quebra autonomamente em sub-perguntas, executa buscas na web usando suas ferramentas integradas, e entrega um relatório final estruturado e rico em citações. A API fornece acesso programático direto a todo este processo, usando no momento da escrita modelos como o3-deep-research-2025-06-26 para síntese de alta qualidade e o mais rápido o4-mini-deep-research-2025-06-26 para aplicações sensíveis à latência

A Deep Research API é útil porque automatiza o que seria de outra forma horas de pesquisa manual, entregando relatórios profissionais e baseados em dados adequados para informar estratégia de negócios, decisões de investimento, ou recomendações de política. Seus benefícios chave incluem:

* **Saída Estruturada e Citada:** Ela produz relatórios bem organizados com citações inline ligadas a metadados de fonte, garantindo que afirmações sejam verificáveis e baseadas em dados.  
* **Transparência:** Diferente do processo abstrato no ChatGPT, a API expõe todos os passos intermediários, incluindo o raciocínio do agente, as consultas específicas de busca web que executou, e qualquer código que executou. Isso permite depuração detalhada, análise, e um entendimento mais profundo de como a resposta final foi construída.  
* **Extensibilidade:** Ela suporta o Model Context Protocol (MCP), permitindo que desenvolvedores conectem o agente a bases de conhecimento privadas e fontes de dados internas, misturando pesquisa web pública com informação proprietária.

Para usar a API, você envia uma solicitação ao endpoint client.responses.create, especificando um modelo, um prompt de entrada, e as ferramentas que o agente pode usar. A entrada tipicamente inclui uma system_message que define a persona do agente e formato de saída desejado, junto com a user_query. Você deve também incluir a ferramenta web_search_preview e pode opcionalmente adicionar outras como code_interpreter ou ferramentas MCP customizadas (ver Capítulo 10) para dados internos.

```python
from openai import OpenAI

# Inicializar o cliente com sua chave API
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Definir o papel do agente e a pergunta de pesquisa do usuário
system_message = """Você é um pesquisador profissional preparando um relatório estruturado e baseado em dados.
Foque em insights ricos em dados, use fontes confiáveis, e inclua citações inline."""

user_query = "Pesquise o impacto econômico da semaglutida nos sistemas globais de saúde."

# Criar a chamada da Deep Research API
response = client.responses.create(
    model="o3-deep-research-2025-06-26",
    input=[
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_query
                }
            ]
        }
    ],
    reasoning={
        "summary": "auto"
    },
    tools=[
        {
            "type": "web_search_preview"
        }
    ]
)

# Acessar e imprimir o relatório final da resposta
final_report = response.output[-1].content[0].text
print(final_report)

# --- ACESSAR CITAÇÕES INLINE E METADADOS ---
print("--- CITAÇÕES ---")
annotations = response.output[-1].content[0].annotations

if not annotations:
    print("Nenhuma anotação encontrada no relatório.")
else:
    for i, citation in enumerate(annotations):
        # O span de texto ao qual a citação se refere
        cited_text = final_report[citation.start_index:citation.end_index]
        print(f"Citação {i+1}:")
        print(f" Texto Citado: {cited_text}")
        print(f" Título: {citation.title}")
        print(f" URL: {citation.url}")
        print(f" Localização: chars {citation.start_index}–{citation.end_index}")
        print("\n" + "="*50 + "\n")

# --- INSPECIONAR PASSOS INTERMEDIÁRIOS ---
print("--- PASSOS INTERMEDIÁRIOS ---")

# 1. Passos de Raciocínio: Planos internos e sumários gerados pelo modelo.
try:
    reasoning_step = next(item for item in response.output if item.type == "reasoning")
    print("\n[Encontrado um Passo de Raciocínio]")
    for summary_part in reasoning_step.summary:
        print(f" - {summary_part.text}")
except StopIteration:
    print("\nNenhum passo de raciocínio encontrado.")

# 2. Chamadas de Busca Web: As consultas exatas que o agente executou.
try:
    search_step = next(item for item in response.output if item.type == "web_search_call")
    print("\n[Encontrada uma Chamada de Busca Web]")
    print(f" Consulta Executada: '{search_step.action['query']}'")
    print(f" Status: {search_step.status}")
except StopIteration:
    print("\nNenhum passo de busca web encontrado.")

# 3. Execução de Código: Qualquer código executado pelo agente usando o interpretador de código.
try:
    code_step = next(item for item in response.output if item.type == "code_interpreter_call")
    print("\n[Encontrado um Passo de Execução de Código]")
    print(" Entrada do Código:")
    print(f" ```python\n{code_step.input}\n ```")
    print(" Saída do Código:")
    print(f" {code_step.output}")
except StopIteration:
    print("\nNenhum passo de execução de código encontrado.")
```

Este snippet de código utiliza a API OpenAI para executar uma tarefa de "Deep Research". Ele começa inicializando o cliente OpenAI com sua chave API, que é crucial para autenticação. Então, ele define o papel do agente de IA como um pesquisador profissional e define a pergunta de pesquisa do usuário sobre o impacto econômico da semaglutida. O código constrói uma chamada de API para o modelo o3-deep-research-2025-06-26, fornecendo a mensagem de sistema definida e consulta do usuário como entrada. Ele também solicita um sumário automático do raciocínio e habilita capacidades de busca web. Após fazer a chamada da API, ele extrai e imprime o relatório final gerado. 

Subsequentemente, ele tenta acessar e exibir citações inline e metadados das anotações do relatório, incluindo o texto citado, título, URL, e localização dentro do relatório. Finalmente, ele inspeciona e imprime detalhes sobre os passos intermediários que o modelo tomou, como passos de raciocínio, chamadas de busca web (incluindo a consulta executada), e quaisquer passos de execução de código se um interpretador de código foi usado.

# Em Resumo

**O que:** Problemas complexos frequentemente não podem ser resolvidos com uma única ação e requerem previsão para alcançar um resultado desejado. Sem uma abordagem estruturada, um sistema agêntico luta para lidar com solicitações multifacetadas que envolvem múltiplos passos e dependências. Isso torna difícil quebrar objetivos de alto nível em uma série gerenciável de tarefas menores e executáveis. Consequentemente, o sistema falha em estrategizar efetivamente, levando a resultados incompletos ou incorretos quando confrontado com objetivos intrincados.

**Por que:** O padrão de Planejamento oferece uma solução padronizada tendo um sistema agêntico primeiro criar um plano coerente para abordar um objetivo. Ele envolve decompor um objetivo de alto nível em uma sequência de passos menores e acionáveis ou sub-objetivos. Isso permite ao sistema gerenciar workflows complexos, orquestrar várias ferramentas, e lidar com dependências em uma ordem lógica. LLMs são particularmente bem adequados para isso, pois podem gerar planos plausíveis e efetivos baseados em seus vastos dados de treinamento. Esta abordagem estruturada transforma um agente reativo simples em um executor estratégico que pode trabalhar proativamente em direção a um objetivo complexo e até mesmo adaptar seu plano se necessário.

**Regra geral:** Use este padrão quando a solicitação de um usuário é muito complexa para ser lidada por uma única ação ou ferramenta. É ideal para automatizar processos multi-passo, como gerar um relatório de pesquisa detalhado, integrar um novo funcionário, ou executar uma análise competitiva. Aplique o padrão de Planejamento sempre que uma tarefa requer uma sequência de operações interdependentes para alcançar um resultado final e sintetizado.

**Resumo visual**  
**![][image4]**  
Fig.4; Padrão de design de planejamento

# Principais Takeaways

* Planejamento permite que agentes quebrem objetivos complexos em passos acionáveis e sequenciais.  
* É essencial para lidar com tarefas multi-passo, automação de workflow, e navegar ambientes complexos.  
* LLMs podem executar planejamento gerando abordagens passo-a-passo baseadas em descrições de tarefas.  
* Prompts ou design de tarefas explícitos para requerer passos de planejamento encoraja este comportamento em frameworks de agentes.  
* Google Deep Research é um agente analisando em nosso nome fontes obtidas usando Google Search como ferramenta. Ele reflete, planeja e executa

# Conclusão

O padrão de Planejamento representa uma evolução fundamental na capacidade dos agentes de IA para lidar com problemas complexos que requerem estratégia e coordenação. Através da decomposição de objetivos de alto nível em sequências estruturadas de ações, agentes podem transcender reações simples e desenvolver comportamentos orientados a objetivos que são tanto adaptativos quanto eficazes. Este padrão é essencial para qualquer sistema agêntico que aspira a lidar com desafios do mundo real que não podem ser resolvidos através de abordagens monolíticas ou reativas, estabelecendo a base para a próxima geração de aplicações de IA estratégicas e proativas.

[image1]: ../assets/11-chapter-6-image-1-line-108.png

[image2]: ../assets/11-chapter-6-image-2-line-110.png

[image3]: ../assets/11-chapter-6-image-3-line-112.png

[image4]: ../assets/11-chapter-6-image-4-line-114.png
