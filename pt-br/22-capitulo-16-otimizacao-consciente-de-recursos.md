# Capítulo 16: Otimização Consciente de Recursos

A Otimização Consciente de Recursos permite que agentes inteligentes monitorem e gerenciem dinamicamente recursos computacionais, temporais e financeiros durante a operação. Isto difere do planejamento simples, que foca principalmente no sequenciamento de ações. A Otimização Consciente de Recursos requer que agentes tomem decisões sobre execução de ações para atingir objetivos dentro de orçamentos de recursos especificados ou para otimizar eficiência. Isto envolve escolher entre modelos mais precisos mas caros e modelos mais rápidos e de menor custo, ou decidir se alocar computação adicional para uma resposta mais refinada versus retornar uma resposta mais rápida e menos detalhada.

Por exemplo, considere um agente encarregado de analisar um grande conjunto de dados para um analista financeiro. Se o analista precisa de um relatório preliminar imediatamente, o agente pode usar um modelo mais rápido e acessível para resumir rapidamente tendências chave. No entanto, se o analista requer uma previsão altamente precisa para uma decisão de investimento crítica e tem um orçamento maior e mais tempo, o agente alocaria mais recursos para utilizar um modelo preditivo poderoso, mais lento, mas mais preciso. Uma estratégia chave nesta categoria é o mecanismo de fallback, que atua como uma salvaguarda quando um modelo preferido está indisponível devido a estar sobrecarregado ou limitado. Para garantir degradação graciosa, o sistema automaticamente muda para um modelo padrão ou mais acessível, mantendo continuidade de serviço ao invés de falhar completamente.

# Aplicações Práticas e Casos de Uso

Casos de uso práticos incluem:

* **Uso de LLM Otimizado por Custo:** Um agente decidindo se usar um LLM grande e caro para tarefas complexas ou um menor e mais acessível para consultas mais simples, baseado em uma restrição de orçamento.  
* **Operações Sensíveis à Latência:** Em sistemas em tempo real, um agente escolhe um caminho de raciocínio mais rápido mas potencialmente menos abrangente para garantir uma resposta oportuna.  
* **Eficiência Energética:** Para agentes implantados em dispositivos de borda ou com energia limitada, otimizando seu processamento para conservar vida da bateria.  
* **Fallback para confiabilidade de serviço:** Um agente automaticamente muda para um modelo de backup quando a escolha primária está indisponível, garantindo continuidade de serviço e degradação graciosa.  
* **Gerenciamento de Uso de Dados:** Um agente optando por recuperação de dados resumida ao invés de downloads de conjunto de dados completos para economizar largura de banda ou armazenamento.  
* **Alocação de Tarefas Adaptativa:** Em sistemas multi-agente, agentes se atribuem tarefas baseados em sua carga computacional atual ou tempo disponível.

# Exemplo de Código Hands-On

Um sistema inteligente para responder perguntas de usuários pode avaliar a dificuldade de cada pergunta. Para consultas simples, utiliza um modelo de linguagem custo-efetivo como Gemini Flash. Para consultas complexas, um modelo de linguagem mais poderoso, mas caro (como Gemini Pro) é considerado. A decisão de usar o modelo mais poderoso também depende da disponibilidade de recursos, especificamente restrições de orçamento e tempo. Este sistema seleciona dinamicamente modelos apropriados.

Por exemplo, considere um planejador de viagem construído com um agente hierárquico. O planejamento de alto nível, que envolve entender a solicitação complexa de um usuário, dividindo-a em um itinerário multi-passo e fazendo decisões lógicas, seria gerenciado por um LLM sofisticado e mais poderoso como Gemini Pro. Este é o agente "planejador" que requer uma compreensão profunda de contexto e a capacidade de raciocinar.

No entanto, uma vez que o plano é estabelecido, as tarefas individuais dentro daquele plano, como procurar preços de voos, verificar disponibilidade de hotéis ou encontrar avaliações de restaurantes, são essencialmente consultas web simples e repetitivas. Estas "chamadas de função de ferramenta" podem ser executadas por um modelo mais rápido e acessível como Gemini Flash. É mais fácil visualizar por que o modelo acessível pode ser usado para estas buscas web diretas, enquanto a fase de planejamento intrincado requer a maior inteligência do modelo mais avançado para garantir um plano de viagem coerente e lógico.

O ADK do Google suporta esta abordagem através de sua arquitetura multi-agente, que permite aplicações modulares e escaláveis. Diferentes agentes podem lidar com tarefas especializadas. A flexibilidade de modelo permite o uso direto de vários modelos Gemini, incluindo tanto Gemini Pro quanto Gemini Flash, ou integração de outros modelos através de LiteLLM. As capacidades de orquestração do ADK suportam roteamento dinâmico e dirigido por LLM para comportamento adaptativo. Recursos de avaliação integrados permitem avaliação sistemática do desempenho do agente, que pode ser usado para refinamento do sistema (veja o Capítulo sobre Avaliação e Monitoramento).

Em seguida, dois agentes com configuração idêntica mas utilizando modelos e custos diferentes serão definidos.

```python
# Estrutura conceitual tipo Python, não código executável
from google.adk.agents import Agent
# from google.adk.models.lite_llm import LiteLlm  # Se usando modelos não suportados diretamente pelo Agent padrão do ADK

# Agente usando o Gemini Pro 2.5 mais caro
gemini_pro_agent = Agent(
    name="GeminiProAgent",
    model="gemini-2.5-pro",  # Placeholder para nome real do modelo se diferente
    description="Um agente altamente capaz para consultas complexas.",
    instruction="Você é um assistente especialista para resolução de problemas complexos."
)

# Agente usando o Gemini Flash 2.5 menos caro
gemini_flash_agent = Agent(
    name="GeminiFlashAgent",
    model="gemini-2.5-flash",  # Placeholder para nome real do modelo se diferente
    description="Um agente rápido e eficiente para consultas simples.",
    instruction="Você é um assistente rápido para questões diretas."
)
```

Um Agente Roteador pode direcionar consultas baseadas em métricas simples como comprimento da consulta, onde consultas mais curtas vão para modelos menos caros e consultas mais longas para modelos mais capazes. No entanto, um Agente Roteador mais sofisticado pode utilizar modelos LLM ou ML para analisar nuances e complexidade da consulta. Este roteador LLM pode determinar qual modelo de linguagem downstream é mais adequado. Por exemplo, uma consulta solicitando recall factual é roteada para um modelo flash, enquanto uma consulta complexa requerendo análise profunda é roteada para um modelo pro.

Técnicas de otimização podem aprimorar ainda mais a efetividade do roteador LLM. Ajuste de prompt envolve criar prompts para guiar o roteador LLM para melhores decisões de roteamento. O fine-tuning do roteador LLM em um conjunto de dados de consultas e suas escolhas de modelo ótimas melhora sua precisão e eficiência. Esta capacidade de roteamento dinâmico equilibra qualidade de resposta com custo-efetividade.

```python
# Estrutura conceitual tipo Python, não código executável
from google.adk.agents import Agent, BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
import asyncio

class QueryRouterAgent(BaseAgent):
    name: str = "QueryRouter"
    description: str = "Roteia consultas de usuários para o agente LLM apropriado baseado em complexidade."
    
    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        user_query = context.current_message.text  # Assumindo entrada de texto
        query_length = len(user_query.split())  # Métrica simples: número de palavras
        
        if query_length < 20:  # Exemplo de threshold para simplicidade vs. complexidade
            print(f"Roteando para Agente Gemini Flash para consulta curta (comprimento: {query_length})")
            # Em uma configuração ADK real, você 'transferiria_para_agente' ou invocaria diretamente
            # Para demonstração, simularemos uma chamada e retornaremos sua resposta
            response = await gemini_flash_agent.run_async(context.current_message)
            yield Event(author=self.name, content=f"Agente Flash processou: {response}")
        else:
            print(f"Roteando para Agente Gemini Pro para consulta longa (comprimento: {query_length})")
            response = await gemini_pro_agent.run_async(context.current_message)
            yield Event(author=self.name, content=f"Agente Pro processou: {response}")
```

O Agente Crítico avalia respostas de modelos de linguagem, fornecendo feedback que serve várias funções. Para auto-correção, identifica erros ou inconsistências, solicitando que o agente respondente refine sua saída para qualidade aprimorada. Também avalia sistematicamente respostas para monitoramento de desempenho, rastreando métricas como precisão e relevância, que são usadas para otimização.

Além disso, seu feedback pode sinalizar aprendizado por reforço ou fine-tuning; identificação consistente de respostas inadequadas do modelo Flash, por exemplo, pode refinar a lógica do agente roteador. Embora não gerencie diretamente o orçamento, o Agente Crítico contribui para gerenciamento indireto de orçamento identificando escolhas de roteamento subótimas, como direcionar consultas simples para um modelo Pro ou consultas complexas para um modelo Flash, o que leva a resultados pobres. Isto informa ajustes que melhoram alocação de recursos e economia de custos.

O Agente Crítico pode ser configurado para revisar apenas o texto gerado do agente respondente ou tanto a consulta original quanto o texto gerado, permitindo uma avaliação abrangente do alinhamento da resposta com a pergunta inicial.

```text
CRITIC_SYSTEM_PROMPT = """
Você é o **Agente Crítico**, servindo como o braço de garantia de qualidade de nosso sistema 
assistente de pesquisa colaborativo. 

Sua função primária é **revisar e desafiar meticulosamente** informação do Agente Pesquisador, 
garantindo **precisão, completude e apresentação imparcial**.

Seus deveres abrangem:

* **Avaliar descobertas de pesquisa** para correção factual, minuciosidade e possíveis tendências.

* **Identificar qualquer dado ausente** ou inconsistências no raciocínio.

* **Levantar questões críticas** que poderiam refinar ou expandir o entendimento atual.

* **Oferecer sugestões construtivas** para aprimoramento ou explorar diferentes ângulos.

* **Validar que a saída final é abrangente** e equilibrada.

Toda crítica deve ser construtiva. Seu objetivo é fortificar a pesquisa, não invalidá-la. 
Estruture seu feedback claramente, chamando atenção para pontos específicos para revisão. 

Seu objetivo geral é garantir que o produto de pesquisa final atenda aos mais altos padrões 
de qualidade possíveis.
"""
```

O Agente Crítico opera baseado em um prompt de sistema pré-definido que delineia seu papel, responsabilidades e abordagem de feedback. Um prompt bem projetado para este agente deve estabelecer claramente sua função como avaliador. Deve especificar as áreas para foco crítico e enfatizar fornecer feedback construtivo ao invés de mera rejeição. O prompt também deve encorajar a identificação de tanto pontos fortes quanto fracos, e deve guiar o agente sobre como estruturar e apresentar seu feedback.

# Código Hands-On com OpenAI

Este sistema usa uma estratégia de otimização consciente de recursos para lidar com consultas de usuários eficientemente. Primeiro classifica cada consulta em uma de três categorias para determinar o caminho de processamento mais apropriado e custo-efetivo. Esta abordagem evita desperdiçar recursos computacionais em solicitações simples enquanto garante que consultas complexas recebam a atenção necessária. As três categorias são:

* simple: Para questões diretas que podem ser respondidas diretamente sem raciocínio complexo ou dados externos.  
* reasoning: Para consultas que requerem dedução lógica ou processos de pensamento multi-passo, que são roteadas para modelos mais poderosos.  
* internet_search: Para questões precisando de informação atual, que automaticamente aciona uma Busca Google para fornecer uma resposta atualizada.

O código está sob licença MIT e disponível no Github: ([https://github.com/mahtabsyed/21-Agentic-Patterns/blob/main/16_Resource_Aware_Opt_LLM_Reflection_v2.ipynb](https://github.com/mahtabsyed/21-Agentic-Patterns/blob/main/16_Resource_Aware_Opt_LLM_Reflection_v2.ipynb))

```python
# Licença MIT
# Copyright (c) 2025 Mahtab Syed
# https://www.linkedin.com/in/mahtabsyed/

import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI

# Carregar variáveis de ambiente
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not OPENAI_API_KEY or not GOOGLE_CUSTOM_SEARCH_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError(
        "Por favor, defina OPENAI_API_KEY, GOOGLE_CUSTOM_SEARCH_API_KEY e GOOGLE_CSE_ID em seu arquivo .env."
    )

client = OpenAI(api_key=OPENAI_API_KEY) # --- Passo 1: Classificar o Prompt ---
def classify_prompt(prompt: str) -> dict:
    system_message = {
        "role": "system", 
        "content": (
            "Você é um classificador que analisa prompts de usuários e retorna uma de três categorias APENAS:\n\n"
            "- simple\n"
            "- reasoning\n"
            "- internet_search\n\n"
            "Regras:\n"
            "- Use 'simple' para questões factuais diretas que não precisam de raciocínio ou eventos atuais.\n"
            "- Use 'reasoning' para questões de lógica, matemática ou inferência multi-passo.\n"
            "- Use 'internet_search' se o prompt se refere a eventos atuais, dados recentes ou coisas não em seus dados de treinamento.\n\n"
            "Responda APENAS com JSON como:\n"
            '{"classification": "simple"}'
        ),
    }
    
    user_message = {
        "role": "user", 
        "content": prompt
    }
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_message, user_message],
        temperature=1
    )
    
    reply = response.choices[0].message.content
    return json.loads(reply)

# --- Passo 2: Busca Google ---
def google_search(query: str, num_results=1) -> list:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CUSTOM_SEARCH_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        if "items" in results and results["items"]:
            return [
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": item.get("link"),
                }
                for item in results["items"]
            ]
        else:
            return []
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# --- Passo 3: Gerar Resposta ---
def generate_response(prompt: str, classification: str, search_results=None) -> str:
    if classification == "simple":
        model = "gpt-4o-mini"
        full_prompt = prompt
    elif classification == "reasoning":
        model = "o4-mini"
        full_prompt = prompt
    elif classification == "internet_search":
        model = "gpt-4o"
        
        # Converter cada resultado de busca dict para uma string legível
        if search_results:
            search_context = "\n".join(
                [
                    f"Título: {item.get('title')}\n"
                    f"Snippet: {item.get('snippet')}\n"
                    f"Link: {item.get('link')}\n"
                    for item in search_results
                ]
            )
        else:
            search_context = "Nenhum resultado de busca encontrado."
        
        full_prompt = f"""Use os seguintes resultados da web para responder à consulta do usuário:
{search_context}

Consulta: {prompt}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        temperature=1,
    )
    
    return response.choices[0].message.content, model

# --- Passo 4: Roteador Combinado ---
def handle_prompt(prompt: str) -> dict:
    classification_result = classify_prompt(prompt)
    # Remover ou comentar a próxima linha para evitar impressão duplicada
    # print("\n🔍 Resultado da Classificação:", classification_result)
    
    classification = classification_result["classification"]
    search_results = None
    
    if classification == "internet_search":
        search_results = google_search(prompt)
        # print("\n🔍 Resultados da Busca:", search_results)
    
    answer, model = generate_response(prompt, classification, search_results)
    
    return {
        "classification": classification,
        "response": answer,
        "model": model
    }

# Teste do sistema
test_prompt = "Qual é a capital da Austrália?"
# test_prompt = "Explique o impacto da computação quântica na criptografia."
# test_prompt = "Quando começa o Australian Open 2026, me dê a data completa?"

result = handle_prompt(test_prompt)
print("🔍 Classificação:", result["classification"])
print("🧠 Modelo Usado:", result["model"])
print("🧠 Resposta:\n", result["response"])
```

Este código Python implementa um sistema de roteamento de prompt para responder perguntas de usuários. Começa carregando chaves de API necessárias de um arquivo .env para OpenAI e Google Custom Search. A funcionalidade central está em classificar o prompt do usuário em três categorias: simple, reasoning ou internet search. Uma função dedicada utiliza um modelo OpenAI para este passo de classificação. Se o prompt requer informação atual, uma busca Google é realizada usando a API Google Custom Search. Outra função então gera a resposta final, selecionando um modelo OpenAI apropriado baseado na classificação. Para consultas de internet search, os resultados da busca são fornecidos como contexto ao modelo. A função principal handle_prompt orquestra este fluxo de trabalho, chamando as funções de classificação e busca (se necessário) antes de gerar a resposta. Retorna a classificação, o modelo usado e a resposta gerada. Este sistema direciona eficientemente diferentes tipos de consultas para métodos otimizados para uma melhor resposta.

# Exemplo de Código Hands-On (OpenRouter)

OpenRouter oferece uma interface unificada para centenas de modelos de IA via um único endpoint de API. Fornece failover automatizado e otimização de custos, com integração fácil através de seu SDK ou framework preferido.

```python
import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer <OPENROUTER_API_KEY>",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Opcional. URL do site para rankings no openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>",      # Opcional. Título do site para rankings no openrouter.ai.
    },
    data=json.dumps({
        "model": "openai/gpt-4o",  # Opcional
        "messages": [
            {
                "role": "user",
                "content": "Qual é o significado da vida?"
            }
        ]
    })
)
```

Este trecho de código usa a biblioteca requests para interagir com a API OpenRouter. Envia uma requisição POST para o endpoint de conclusão de chat com uma mensagem de usuário. A requisição inclui headers de autorização com uma chave de API e informação opcional do site. O objetivo é obter uma resposta de um modelo de linguagem especificado, neste caso, "openai/gpt-4o".

Openrouter oferece duas metodologias distintas para roteamento e determinação do modelo computacional usado para processar uma requisição dada.

* **Seleção de Modelo Automatizada:** Esta função roteia uma requisição para um modelo otimizado escolhido de um conjunto curado de modelos disponíveis. A seleção é baseada no conteúdo específico do prompt do usuário. O identificador do modelo que finalmente processa a requisição é retornado nos metadados da resposta.

```json
{
"model": "openrouter/auto", ... // Outros params }
```

* **Fallback de Modelo Sequencial:** Este mecanismo fornece redundância operacional permitindo que usuários especifiquem uma lista hierárquica de modelos. O sistema primeiro tentará processar a requisição com o modelo primário designado na sequência. Se este modelo primário falhar em responder devido a qualquer número de condições de erro—como indisponibilidade de serviço, limitação de taxa ou filtragem de conteúdo—o sistema automaticamente re-roteará a requisição para o próximo modelo especificado na sequência. Este processo continua até que um modelo na lista execute com sucesso a requisição ou a lista seja esgotada. O custo final da operação e o identificador do modelo retornado na resposta corresponderão ao modelo que completou com sucesso a computação.

```json
{
"models": ["anthropic/claude-3.5-sonnet", "gryphe/mythomax-l2-13b"], ... // Outros params }
```

OpenRouter oferece um leaderboard detalhado ([https://openrouter.ai/rankings](https://openrouter.ai/rankings)) que classifica modelos de IA disponíveis baseados em sua produção cumulativa de tokens. Também oferece modelos mais recentes de diferentes provedores (ChatGPT, Gemini, Claude) (veja Fig. 1)

![][image1]  
Fig. 1: Site OpenRouter ([https://openrouter.ai/](https://openrouter.ai/))

# Além da Mudança Dinâmica de Modelo: Um Espectro de Otimizações de Recursos de Agente

Otimização consciente de recursos é primordial no desenvolvimento de sistemas de agentes inteligentes que operam eficientemente e efetivamente dentro de restrições do mundo real. Vamos ver várias técnicas adicionais:

**Mudança Dinâmica de Modelo** é uma técnica crítica envolvendo a seleção estratégica de grandes modelos de linguagem baseada nas intricacidades da tarefa em questão e os recursos computacionais disponíveis. Quando confrontado com consultas simples, um LLM leve e custo-efetivo pode ser implantado, enquanto problemas complexos e multifacetados necessitam a utilização de modelos mais sofisticados e intensivos em recursos.

**Uso e Seleção de Ferramentas Adaptativa** garante que agentes possam escolher inteligentemente de uma suíte de ferramentas, selecionando a mais apropriada e eficiente para cada sub-tarefa específica, com consideração cuidadosa dada a fatores como custos de uso de API, latência e tempo de execução. Esta seleção dinâmica de ferramentas aprimora a eficiência geral do sistema otimizando o uso de APIs e serviços externos.

**Poda e Resumo Contextual** desempenha um papel vital no gerenciamento da quantidade de informação processada por agentes, minimizando estrategicamente a contagem de tokens de prompt e reduzindo custos de inferência resumindo inteligentemente e retendo seletivamente apenas a informação mais relevante do histórico de interação, prevenindo overhead computacional desnecessário.

**Predição Proativa de Recursos** envolve agentes antecipando necessidades futuras de recursos baseadas em padrões históricos e demandas atuais, permitindo alocação preventiva e otimização de recursos antes que restrições se tornem limitantes.

**Exploração Sensível a Custo em Sistemas Multi-Agente** refere-se a agentes tomando decisões sobre exploração de novos caminhos ou estratégias baseadas em custos computacionais esperados versus benefícios potenciais, especialmente em ambientes onde múltiplos agentes competem por recursos limitados.

**Implantação Eficiente em Energia** é crucial para agentes operando em dispositivos com recursos limitados, onde otimizações focam em minimizar consumo de energia através de técnicas como quantização de modelo, pruning de rede neural e execução de inferência otimizada.

**Consciência de Paralelização e Computação Distribuída** permite que agentes identifiquem oportunidades para dividir tarefas em componentes paralelos ou distribuídos, maximizando utilização de recursos disponíveis e reduzindo tempo total de execução.

**Políticas de Alocação de Recursos Aprendidas** envolvem agentes desenvolvendo estratégias de otimização de recursos através de aprendizado por reforço ou análise de padrões históricos, melhorando continuamente suas decisões de alocação de recursos ao longo do tempo.

**Degradação Graciosa e Mecanismos de Fallback** garantem que agentes possam manter funcionalidade básica mesmo quando recursos se tornam limitados, automaticamente mudando para modos de operação menos intensivos em recursos quando necessário.

**Priorização de Tarefas Críticas** permite que agentes identifiquem e priorizem tarefas essenciais quando recursos são limitados, garantindo que operações críticas recebam recursos adequados mesmo em condições de restrição.


**Diagrama de Otimização**

# Principais Conclusões

* Otimização Consciente de Recursos é Essencial: Agentes inteligentes podem gerenciar recursos computacionais, temporais e financeiros dinamicamente. Decisões sobre uso de modelo e caminhos de execução são feitas baseadas em restrições e objetivos em tempo real.  
* Arquitetura Multi-Agente para Escalabilidade: O ADK do Google fornece um framework multi-agente, permitindo design modular. Diferentes agentes (respondente, roteador, crítico) lidam com tarefas específicas.  
* Roteamento Dinâmico Dirigido por LLM: Um Agente Roteador direciona consultas para modelos de linguagem (Gemini Flash para simples, Gemini Pro para complexas) baseado em complexidade de consulta e orçamento. Isto otimiza custo e desempenho.  
* Funcionalidade do Agente Crítico: Um Agente Crítico dedicado fornece feedback para auto-correção, monitoramento de desempenho e refinamento de lógica de roteamento, aprimorando efetividade do sistema.  
* Otimização Através de Feedback e Flexibilidade: Capacidades de avaliação para crítica e flexibilidade de integração de modelo contribuem para comportamento adaptativo e auto-aprimorador do sistema.  
* Otimizações Adicionais Conscientes de Recursos: Outros métodos incluem Uso e Seleção de Ferramentas Adaptativa, Poda e Resumo Contextual, Predição Proativa de Recursos, Exploração Sensível a Custo em Sistemas Multi-Agente, Implantação Eficiente em Energia, Consciência de Paralelização e Computação Distribuída, Políticas de Alocação de Recursos Aprendidas, Degradação Graciosa e Mecanismos de Fallback, e Priorização de Tarefas Críticas.

# Conclusões

Otimização consciente de recursos é essencial para o desenvolvimento de agentes inteligentes, permitindo operação eficiente dentro de restrições do mundo real. Ao gerenciar recursos computacionais, temporais e financeiros, agentes podem atingir desempenho ótimo e custo-efetividade. Técnicas como mudança dinâmica de modelo, uso adaptativo de ferramentas e poda contextual são cruciais para atingir estas eficiências. Estratégias avançadas, incluindo políticas de alocação de recursos aprendidas e degradação graciosa, aprimoram a adaptabilidade e resiliência de um agente sob condições variáveis. Integrar estes princípios de otimização no design de agentes é fundamental para construir sistemas de IA escaláveis, robustos e sustentáveis.

# Referências

1. Google's Agent Development Kit (ADK): [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)   
2. Gemini Flash 2.5 & Gemini 2.5 Pro:  [https://aistudio.google.com/](https://aistudio.google.com/)   
3. OpenRouter: [https://openrouter.ai/docs/quickstart](https://openrouter.ai/docs/quickstart)

[image1]: ../assets/21-chapter-16-image-1-line-147.png
