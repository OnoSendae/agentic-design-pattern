# Capítulo 14: Recuperação de Conhecimento (RAG)

LLMs exibem capacidades substanciais na geração de texto similar ao humano. No entanto, sua base de conhecimento é tipicamente confinada aos dados nos quais foram treinados, limitando seu acesso a informação em tempo real, dados específicos da empresa ou detalhes altamente especializados. A Recuperação de Conhecimento (RAG, ou Geração Aumentada por Recuperação), aborda esta limitação. RAG permite que LLMs acessem e integrem informação externa, atual e específica de contexto, aprimorando assim a precisão, relevância e base factual de suas saídas.

Para agentes de IA, isto é crucial pois permite que eles fundamentem suas ações e respostas em dados verificáveis em tempo real além de seu treinamento estático. Esta capacidade permite que executem tarefas complexas com precisão, como acessar as políticas mais recentes da empresa para responder a uma pergunta específica ou verificar o inventário atual antes de fazer um pedido. Ao integrar conhecimento externo, RAG transforma agentes de simples conversadores em ferramentas efetivas e baseadas em dados capazes de executar trabalho significativo.

# Visão Geral do Padrão de Recuperação de Conhecimento (RAG)

O padrão de Recuperação de Conhecimento (RAG) aprimora significativamente as capacidades de LLMs concedendo-lhes acesso a bases de conhecimento externas antes de gerar uma resposta. Ao invés de confiar exclusivamente em seu conhecimento interno pré-treinado, RAG permite que LLMs "consultem" informação, muito como um humano pode consultar um livro ou pesquisar na internet. Este processo capacita LLMs a fornecer respostas mais precisas, atualizadas e verificáveis.

Quando um usuário faz uma pergunta ou dá um prompt a um sistema de IA usando RAG, a consulta não é enviada diretamente ao LLM. Ao invés disso, o sistema primeiro vasculha uma vasta base de conhecimento externa—uma biblioteca altamente organizada de documentos, bancos de dados ou páginas web—para informação relevante. Esta busca não é uma simples correspondência de palavras-chave; é uma "busca semântica" que entende a intenção do usuário e o significado por trás de suas palavras. Esta busca inicial extrai os snippets ou "pedaços" mais pertinentes de informação. Estas peças extraídas são então "aumentadas," ou adicionadas, ao prompt original, criando uma consulta mais rica e informada. Finalmente, este prompt aprimorado é enviado ao LLM. Com este contexto adicional, o LLM pode gerar uma resposta que não é apenas fluente e natural, mas também fundamentada factualmente nos dados recuperados.

O framework RAG fornece vários benefícios significativos. Permite que LLMs acessem informação atualizada, superando assim as limitações de seus dados de treinamento estáticos. Esta abordagem também reduz o risco de "alucinação"—a geração de informação falsa—ao fundamentar respostas em dados verificáveis. Além disso, LLMs podem utilizar conhecimento especializado encontrado em documentos internos da empresa ou wikis. Uma vantagem vital deste processo é a capacidade de oferecer "citações," que identificam a fonte exata da informação, aprimorando assim a confiabilidade e verificabilidade das respostas da IA.

Para apreciar completamente como RAG funciona, é essencial entender alguns conceitos centrais (veja Fig.1):

**Embeddings**: No contexto de LLMs, embeddings são representações numéricas de texto, como palavras, frases ou documentos inteiros. Estas representações estão na forma de um vetor, que é uma lista de números. A ideia chave é capturar o significado semântico e os relacionamentos entre diferentes peças de texto em um espaço matemático. Palavras ou frases com significados similares terão embeddings que estão mais próximos uns dos outros neste espaço vetorial. Por exemplo, imagine um gráfico 2D simples. A palavra "gato" pode ser representada pelas coordenadas (2, 3), enquanto "gatinho" estaria muito próximo em (2.1, 3.1). Em contraste, a palavra "carro" teria uma coordenada distante como (8, 1), refletindo seu significado diferente. Na realidade, estes embeddings estão em um espaço muito mais alto-dimensional com centenas ou até milhares de dimensões, permitindo uma compreensão muito sutil da linguagem.

**Similaridade de Texto:** Similaridade de texto refere-se à medida de quão similares duas peças de texto são. Isto pode ser em um nível superficial, olhando para a sobreposição de palavras (similaridade lexical), ou em um nível mais profundo baseado em significado. No contexto de RAG, similaridade de texto é crucial para encontrar a informação mais relevante na base de conhecimento que corresponde à consulta do usuário. Por exemplo, considere as frases: "Qual é a capital da França?" e "Qual cidade é a capital da França?". Embora a redação seja diferente, elas estão fazendo a mesma pergunta. Um bom modelo de similaridade de texto reconheceria isto e atribuiria uma pontuação de similaridade alta a estas duas frases, mesmo que compartilhem apenas algumas palavras. Isto é frequentemente calculado usando os embeddings dos textos.

**Similaridade Semântica e Distância:** Similaridade semântica é uma forma mais avançada de similaridade de texto que foca puramente no significado e contexto do texto, ao invés de apenas nas palavras usadas. Ela visa entender se duas peças de texto transmitem o mesmo conceito ou ideia. Distância semântica é o inverso disto; uma alta similaridade semântica implica uma baixa distância semântica, e vice-versa. Em RAG, busca semântica depende de encontrar documentos com a menor distância semântica à consulta do usuário. Por exemplo, as frases "um companheiro felino peludo" e "um gato doméstico" não têm palavras em comum além de "um". No entanto, um modelo que entende similaridade semântica reconheceria que se referem à mesma coisa e as consideraria altamente similares. Isto é porque seus embeddings estariam muito próximos no espaço vetorial, indicando uma pequena distância semântica. Esta é a "busca inteligente" que permite que RAG encontre informação relevante mesmo quando a redação do usuário não corresponde exatamente ao texto na base de conhecimento.

![][image1]

Fig.1: Conceitos Centrais RAG: Chunking, Embeddings e Banco de Dados Vetorial

**Chunking de Documentos:** Chunking é o processo de dividir documentos grandes em peças menores e mais gerenciáveis, ou "pedaços." Para um sistema RAG funcionar eficientemente, ele não pode alimentar documentos grandes inteiros no LLM. Ao invés disso, processa estes pedaços menores. A maneira como documentos são divididos é importante para preservar o contexto e significado da informação. Por exemplo, ao invés de tratar um manual de usuário de 50 páginas como um único bloco de texto, uma estratégia de chunking pode dividi-lo em seções, parágrafos ou até frases. Por exemplo, uma seção sobre "Solução de Problemas" seria um pedaço separado do "Guia de Instalação." Quando um usuário faz uma pergunta sobre um problema específico, o sistema RAG pode então recuperar o pedaço de solução de problemas mais relevante, ao invés do manual inteiro. Isto torna o processo de recuperação mais rápido e a informação fornecida ao LLM mais focada e relevante para a necessidade imediata do usuário. Uma vez que documentos são divididos, o sistema RAG deve empregar uma técnica de recuperação para encontrar as peças mais relevantes para uma consulta dada. O método primário é busca vetorial, que usa embeddings e distância semântica para encontrar pedaços que são conceitualmente similares à pergunta do usuário. Uma técnica mais antiga, mas ainda valiosa, é BM25, um algoritmo baseado em palavras-chave que classifica pedaços baseado em frequência de termos sem entender significado semântico. Para obter o melhor dos dois mundos, abordagens de busca híbrida são frequentemente usadas, combinando a precisão de palavras-chave do BM25 com a compreensão contextual da busca semântica. Esta fusão permite recuperação mais robusta e precisa, capturando tanto correspondências literais quanto relevância conceitual.

**Bancos de dados vetoriais:** Um banco de dados vetorial é um tipo especializado de banco de dados projetado para armazenar e consultar embeddings eficientemente. Após documentos serem divididos e convertidos em embeddings, estes vetores de alta dimensão são armazenados em um banco de dados vetorial. Técnicas de recuperação tradicionais, como busca baseada em palavras-chave, são excelentes para encontrar documentos contendo palavras exatas de uma consulta mas carecem de compreensão profunda da linguagem. Elas não reconheceriam que "companheiro felino peludo" significa "gato." É aqui que bancos de dados vetoriais se destacam. Eles são construídos especificamente para busca semântica. Ao armazenar texto como vetores numéricos, podem encontrar resultados baseados em significado conceitual, não apenas sobreposição de palavras-chave. Quando a consulta de um usuário também é convertida em um vetor, o banco de dados usa algoritmos altamente otimizados (como HNSW - Hierarchical Navigable Small World) para buscar rapidamente através de milhões de vetores e encontrar os que estão "mais próximos" em significado. Esta abordagem é muito superior para RAG porque descobre contexto relevante mesmo se a redação do usuário é completamente diferente dos documentos fonte. Em essência, enquanto outras técnicas buscam por palavras, bancos de dados vetoriais buscam por significado. Esta tecnologia é implementada em várias formas, desde bancos de dados gerenciados como Pinecone e Weaviate até soluções open-source como Chroma DB, Milvus e Qdrant. Mesmo bancos de dados existentes podem ser aumentados com capacidades de busca vetorial, como visto com Redis, Elasticsearch e Postgres (usando a extensão pgvector). Os mecanismos centrais de recuperação são frequentemente alimentados por bibliotecas como FAISS da Meta AI ou ScaNN do Google Research, que são fundamentais para a eficiência destes sistemas.

**Desafios do RAG:** Apesar de seu poder, o padrão RAG não está sem seus desafios. Uma questão primária surge quando a informação necessária para responder a uma consulta não está confinada a um único pedaço mas está espalhada através de múltiplas partes de um documento ou até vários documentos. Em tais casos, o recuperador pode falhar em reunir todo o contexto necessário, levando a uma resposta incompleta ou imprecisa. A efetividade do sistema também é altamente dependente da qualidade do processo de chunking e recuperação; se pedaços irrelevantes são recuperados, pode introduzir ruído e confundir o LLM. Além disso, sintetizar efetivamente informação de fontes potencialmente contraditórias permanece um obstáculo significativo para estes sistemas. Além disso, outro desafio é que RAG requer que toda a base de conhecimento seja pré-processada e armazenada em bancos de dados especializados, como bancos de dados vetoriais ou de grafos, o que é uma tarefa considerável. Consequentemente, este conhecimento requer reconciliação periódica para permanecer atualizado, uma tarefa crucial ao lidar com fontes em evolução como wikis de empresa. Todo este processo pode ter um impacto notável na performance, aumentando latência, custos operacionais e o número de tokens usados no prompt final.

Em resumo, o padrão de Geração Aumentada por Recuperação (RAG) representa um salto significativo para frente em tornar IA mais conhecedora e confiável. Ao integrar perfeitamente um passo de recuperação de conhecimento externo no processo de geração, RAG aborda algumas das limitações centrais de LLMs autônomos. Os conceitos fundamentais de embeddings e similaridade semântica, combinados com técnicas de recuperação como busca de palavras-chave e híbrida, permitem que o sistema encontre inteligentemente informação relevante, que é tornada gerenciável através de chunking estratégico. Todo este processo de recuperação é alimentado por bancos de dados vetoriais especializados projetados para armazenar e consultar eficientemente milhões de embeddings em escala. Embora desafios em recuperar informação fragmentada ou contraditória persistam, RAG capacita LLMs a produzir respostas que não são apenas contextualmente apropriadas mas também ancoradas em fatos verificáveis, fomentando maior confiança e utilidade na IA.

**Graph RAG:** GraphRAG é uma forma avançada de Geração Aumentada por Recuperação que utiliza um grafo de conhecimento ao invés de um simples banco de dados vetorial para recuperação de informação. Ele responde consultas complexas navegando os relacionamentos explícitos (arestas) entre entidades de dados (nós) dentro desta base de conhecimento estruturada. Uma vantagem chave é sua capacidade de sintetizar respostas de informação fragmentada através de múltiplos documentos, uma falha comum do RAG tradicional. Ao entender estas conexões, GraphRAG fornece respostas mais contextualmente precisas e sutis.

Casos de uso incluem análise financeira complexa, conectando empresas a eventos de mercado, e pesquisa científica para descobrir relacionamentos entre genes e doenças. O principal inconveniente, no entanto, é a complexidade, custo e expertise significativos necessários para construir e manter um grafo de conhecimento de alta qualidade. Esta configuração também é menos flexível e pode introduzir latência mais alta comparada a sistemas de busca vetorial mais simples. A efetividade do sistema é inteiramente dependente da qualidade e completude da estrutura de grafo subjacente. Consequentemente, GraphRAG oferece raciocínio contextual superior para questões intrincadas mas a um custo de implementação e manutenção muito mais alto. Em resumo, ele se destaca onde insights profundos e interconectados são mais críticos que a velocidade e simplicidade do RAG padrão.

**RAG Agêntico:** Uma evolução deste padrão, conhecida como **RAG Agêntico** (veja Fig.2), introduz uma camada de raciocínio e tomada de decisão para aprimorar significativamente a confiabilidade da extração de informação. Ao invés de apenas recuperar e aumentar, um "agente"—um componente de IA especializado—atua como um guardião crítico e refinador de conhecimento. Ao invés de aceitar passivamente os dados inicialmente recuperados, este agente interroga ativamente sua qualidade, relevância e completude, como ilustrado pelos seguintes cenários.

Primeiro, um agente se destaca em reflexão e validação de fonte. Se um usuário pergunta, "Qual é a política da nossa empresa sobre trabalho remoto?" um RAG padrão pode puxar um post de blog de 2020 junto com o documento de política oficial de 2025. O agente, no entanto, analisaria os metadados dos documentos, reconheceria a política de 2025 como a fonte mais atual e autoritativa, e descartaria o post de blog desatualizado antes de enviar o contexto correto ao LLM para uma resposta precisa.

![][image2]

Fig.2: RAG Agêntico introduz um agente de raciocínio que avalia ativamente, reconcilia e refina informação recuperada para garantir uma resposta final mais precisa e confiável.

Segundo, um agente é adepto em reconciliar conflitos de conhecimento. Imagine um analista financeiro pergunta, "Qual foi o orçamento Q1 do Projeto Alpha?" O sistema recupera dois documentos: uma proposta inicial declarando um orçamento de €50.000 e um relatório financeiro finalizado listando-o como €65.000. Um RAG Agêntico identificaria esta contradição, priorizaria o relatório financeiro como a fonte mais confiável, e forneceria ao LLM a figura verificada, garantindo que a resposta final seja baseada nos dados mais precisos.

Terceiro, um agente pode realizar raciocínio multi-passo para sintetizar respostas complexas. Se um usuário pergunta, "Como as funcionalidades e preços do nosso produto se comparam aos do Competidor X?" o agente decomporia isto em sub-consultas separadas. Ele iniciaria buscas distintas para as funcionalidades de seu próprio produto, seus preços, funcionalidades do Competidor X, e preços do Competidor X. Após reunir estas peças individuais de informação, o agente as sintetizaria em um contexto comparativo estruturado antes de alimentá-lo ao LLM, permitindo uma resposta abrangente que uma recuperação simples não poderia ter produzido.

Quarto, um agente pode identificar lacunas de conhecimento e usar ferramentas externas. Suponha que um usuário pergunta, "Qual foi a reação imediata do mercado ao nosso novo produto lançado ontem?" O agente busca a base de conhecimento interna, que é atualizada semanalmente, e não encontra informação relevante. Reconhecendo esta lacuna, ele pode então ativar uma ferramenta—como uma API de busca web ao vivo—para encontrar artigos de notícias recentes e sentimento de mídia social. O agente então usa esta informação externa recém-coletada para fornecer uma resposta até o minuto, superando as limitações de seu banco de dados interno estático.

**Desafios do RAG Agêntico:** Embora poderoso, a camada agêntica introduz seu próprio conjunto de desafios. O principal inconveniente é um aumento significativo em complexidade e custo. Projetar, implementar e manter a lógica de tomada de decisão do agente e integrações de ferramentas requer esforço de engenharia substancial e adiciona às despesas computacionais. Esta complexidade também pode levar a latência aumentada, pois os ciclos de reflexão, uso de ferramentas e raciocínio multi-passo do agente levam mais tempo que um processo de recuperação direto padrão. Além disso, o próprio agente pode se tornar uma nova fonte de erro; um processo de raciocínio falho poderia causar que ele ficasse preso em loops inúteis, interpretasse mal uma tarefa, ou descartasse impropriamente informação relevante, degradando em última análise a qualidade da resposta final.

### **Em resumo:** RAG Agêntico representa uma evolução sofisticada do padrão de recuperação padrão, transformando-o de um pipeline de dados passivo em um framework ativo de resolução de problemas. Ao embarcar uma camada de raciocínio que pode avaliar fontes, reconciliar conflitos, decompor questões complexas e usar ferramentas externas, agentes melhoram dramaticamente a confiabilidade e profundidade das respostas geradas. Este avanço torna a IA mais confiável e capaz, embora venha com trade-offs importantes em complexidade do sistema, latência e custo que devem ser cuidadosamente gerenciados.

# Aplicações Práticas e Casos de Uso

Recuperação de Conhecimento (RAG) está mudando como Grandes Modelos de Linguagem (LLMs) são utilizados através de várias indústrias, aprimorando sua capacidade de fornecer respostas mais precisas e contextualmente relevantes.

Aplicações incluem:

* **Busca Empresarial e Q&A:** Organizações podem desenvolver chatbots internos que respondem a consultas de funcionários usando documentação interna como políticas de RH, manuais técnicos e especificações de produtos. O sistema RAG extrai seções relevantes destes documentos para informar a resposta do LLM.  
* **Suporte ao Cliente e Helpdesks:** Sistemas baseados em RAG podem oferecer respostas precisas e consistentes a consultas de clientes acessando informação de manuais de produtos, perguntas frequentes (FAQs) e tickets de suporte. Isto pode reduzir a necessidade de intervenção humana direta para questões rotineiras.  
* **Recomendação de Conteúdo Personalizado:** Ao invés de correspondência básica de palavras-chave, RAG pode identificar e recuperar conteúdo (artigos, produtos) que está semanticamente relacionado às preferências do usuário ou interações anteriores, levando a recomendações mais relevantes.  
* **Resumo de Notícias e Eventos Atuais:** LLMs podem ser integrados com feeds de notícias em tempo real. Quando solicitado sobre um evento atual, o sistema RAG recupera artigos recentes, permitindo que o LLM produza um resumo atualizado.

Ao incorporar conhecimento externo, RAG estende as capacidades de LLMs além de simples comunicação para funcionar como sistemas de processamento de conhecimento.

# Exemplo de Código Hands-On (ADK)

Para ilustrar o padrão de Recuperação de Conhecimento (RAG), vamos ver três exemplos.

Primeiro, é como usar Google Search para fazer RAG e fundamentar LLMs em resultados de busca. Como RAG envolve acessar informação externa, a ferramenta Google Search é um exemplo direto de um mecanismo de recuperação integrado que pode aumentar o conhecimento de um LLM.

```python
from google.adk.tools import google_search
from google.adk.agents import Agent

search_agent = Agent(
    name="research_assistant",
    model="gemini-2.0-flash-exp",
    instruction="Você ajuda usuários a pesquisar tópicos. Quando solicitado, use a ferramenta Google Search",
    tools=[google_search]
)
```

Segundo, esta seção explica como utilizar capacidades Vertex AI RAG dentro do Google ADK. O código fornecido demonstra a inicialização de VertexAiRagMemoryService do ADK. Isto permite estabelecer uma conexão com um Google Cloud Vertex AI RAG Corpus. O serviço é configurado especificando o nome do recurso do corpus e parâmetros opcionais como SIMILARITY_TOP_K e VECTOR_DISTANCE_THRESHOLD. Estes parâmetros influenciam o processo de recuperação. SIMILARITY_TOP_K define o número de resultados similares principais a serem recuperados. VECTOR_DISTANCE_THRESHOLD define um limite na distância semântica para os resultados recuperados. Esta configuração permite que agentes realizem recuperação de conhecimento semântico escalável e persistente do RAG Corpus designado. O processo efetivamente integra funcionalidades RAG do Google Cloud em um agente ADK, apoiando assim o desenvolvimento de respostas fundamentadas em dados factuais.

```python
# Importar a classe necessária VertexAiRagMemoryService do módulo google.adk.memory.
from google.adk.memory import VertexAiRagMemoryService

RAG_CORPUS_RESOURCE_NAME = "projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id"

# Definir um parâmetro opcional para o número de resultados similares principais a recuperar.
# Isto controla quantos pedaços de documentos relevantes o serviço RAG retornará.
SIMILARITY_TOP_K = 5

# Definir um parâmetro opcional para o threshold de distância vetorial.
# Este threshold determina a distância semântica máxima permitida para resultados recuperados;
# resultados com uma distância maior que este valor podem ser filtrados.
VECTOR_DISTANCE_THRESHOLD = 0.7

# Inicializar uma instância de VertexAiRagMemoryService.
# Isto configura a conexão com seu Vertex AI RAG Corpus.
# - rag_corpus: Especifica o identificador único para seu RAG Corpus.
# - similarity_top_k: Define o número máximo de resultados similares a buscar.
# - vector_distance_threshold: Define o threshold de similaridade para filtrar resultados.
memory_service = VertexAiRagMemoryService(
    rag_corpus=RAG_CORPUS_RESOURCE_NAME,
    similarity_top_k=SIMILARITY_TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD
)
```

# Exemplo de Código Hands-On (LangChain)

Terceiro, vamos percorrer um exemplo completo usando LangChain.

```python
import os import requests from typing import List, Dict, Any, TypedDict from langchain_community.document_loaders import TextLoader from langchain_core.documents import Document from langchain_core.prompts import ChatPromptTemplate from langchain_core.output_parsers import StrOutputParser from langchain_community.embeddings import OpenAIEmbeddings from langchain_community.vectorstores import Weaviate from langchain_openai import ChatOpenAI from langchain.text_splitter import CharacterTextSplitter from langchain.schema.runnable import RunnablePassthrough from langgraph.graph import StateGraph, END import weaviate from weaviate.embedded import EmbeddedOptions import dotenv # Carregar variáveis de ambiente (ex., OPENAI_API_KEY) dotenv.load_dotenv() # Definir sua chave de API OpenAI (garanta que está carregada do .env ou definida aqui) # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # --- 1. Preparação de Dados (Pré-processamento) --- # Carregar dados url = "https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/state_of_the_union.txt" res = requests.get(url) with open("state_of_the_union.txt", "w") as f: f.write(res.text) loader = TextLoader('./state_of_the_union.txt') documents = loader.load() # Dividir documentos text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) chunks = text_splitter.split_documents(documents) # Embed e armazenar pedaços em Weaviate client = weaviate.Client( embedded_options = EmbeddedOptions() ) vectorstore = Weaviate.from_documents( client = client, documents = chunks, embedding = OpenAIEmbeddings(), by_text = False ) # Definir o recuperador retriever = vectorstore.as_retriever() # Inicializar LLM llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # --- 2. Definir o Estado para LangGraph --- class RAGGraphState(TypedDict): question: str documents: List[Document] generation: str # --- 3. Definir os Nós (Funções) --- def retrieve_documents_node(state: RAGGraphState) -> RAGGraphState: """Recupera documentos baseado na pergunta do usuário.""" question = state["question"] documents = retriever.invoke(question) return  {
"documents": documents, "question": question, "generation": ""}
def generate_response_node(state: RAGGraphState) -> RAGGraphState: """Gera uma resposta usando o LLM baseado nos documentos recuperados.""" question = state["question"] documents = state["documents"] # Template de prompt do PDF template = """Você é um assistente para tarefas de pergunta e resposta. Use as seguintes peças de contexto recuperado para responder à pergunta. Se você não souber a resposta, apenas diga que não sabe. Use no máximo três frases e mantenha a resposta concisa. Pergunta:  {
question}
Contexto:  {
context}
Resposta: """ prompt = ChatPromptTemplate.from_template(template) # Formatar o contexto dos documentos context = "\n\n".join([doc.page_content for doc in documents]) # Criar a cadeia RAG rag_chain = prompt | llm | StrOutputParser() # Invocar a cadeia generation = rag_chain.invoke( {
"context": context, "question": question}
) return  {
"question": question, "documents": documents, "generation": generation}
# --- 4. Construir o Grafo LangGraph --- workflow = StateGraph(RAGGraphState) # Adicionar nós workflow.add_node("retrieve", retrieve_documents_node) workflow.add_node("generate", generate_response_node) # Definir o ponto de entrada workflow.set_entry_point("retrieve") # Adicionar arestas (transições) workflow.add_edge("retrieve", "generate") workflow.add_edge("generate", END) # Compilar o grafo app = workflow.compile() # --- 5. Executar a Aplicação RAG --- if __name__ == "__main__": print("\n--- Executando Consulta RAG ---") query = "O que o presidente disse sobre Justice Breyer" inputs =  {
"question": query}
for s in app.stream(inputs): print(s) print("\n--- Executando outra Consulta RAG ---") query_2 = "O que o presidente disse sobre a economia?" inputs_2 =  {
"question": query_2}
for s in app.stream(inputs_2): print(s)
```

Este código Python ilustra um pipeline de Geração Aumentada por Recuperação (RAG) implementado com LangChain e LangGraph. O processo começa com a criação de uma base de conhecimento derivada de um documento de texto, que é segmentado em pedaços e transformado em embeddings. Estes embeddings são então armazenados em um armazém vetorial Weaviate, facilitando recuperação eficiente de informação. Um StateGraph no LangGraph é utilizado para gerenciar o fluxo de trabalho entre duas funções chave: `retrieve_documents_node` e `generate_response_node`. A função `retrieve_documents_node` consulta o armazém vetorial para identificar pedaços de documentos relevantes baseados na entrada do usuário. Subsequentemente, a função `generate_response_node` utiliza a informação recuperada e um template de prompt pré-definido para produzir uma resposta usando um Grande Modelo de Linguagem OpenAI (LLM). O método `app.stream` permite a execução de consultas através do pipeline RAG, demonstrando a capacidade do sistema de gerar saídas contextualmente relevantes.

# Em Resumo

**O que**: LLMs possuem habilidades impressionantes de geração de texto mas são fundamentalmente limitados por seus dados de treinamento. Este conhecimento é estático, significando que não inclui informação em tempo real ou dados privados específicos de domínio. Consequentemente, suas respostas podem estar desatualizadas, imprecisas ou carecer do contexto específico necessário para tarefas especializadas. Esta lacuna restringe sua confiabilidade para aplicações exigindo respostas atuais e factuais.

**Por quê**: O padrão de Geração Aumentada por Recuperação (RAG) fornece uma solução padronizada conectando LLMs a fontes de conhecimento externas. Quando uma consulta é recebida, o sistema primeiro recupera snippets relevantes de informação de uma base de conhecimento especificada. Estes snippets são então anexados ao prompt original, enriquecendo-o com contexto oportuno e específico. Este prompt aumentado é então enviado ao LLM, permitindo que gere uma resposta que é precisa, verificável e fundamentada em dados externos. Este processo efetivamente transforma o LLM de um raciocinador de livro fechado em um de livro aberto, aprimorando significativamente sua utilidade e confiabilidade.

**Regra de ouro**: Use este padrão quando precisar que um LLM responda perguntas ou gere conteúdo baseado em informação específica, atualizada ou proprietária que não fazia parte de seus dados de treinamento originais. É ideal para construir sistemas Q&A sobre documentos internos, bots de suporte ao cliente e aplicações exigindo respostas verificáveis e baseadas em fatos com citações.

**Resumo visual**

**![][image3]**

Padrão de Recuperação de Conhecimento: um agente de IA para consultar e recuperar informação de bancos de dados estruturados

**![][image4]**

Fig. 3: Padrão de Recuperação de Conhecimento: um agente de IA para encontrar e sintetizar informação da internet pública em resposta a consultas do usuário.

# Principais Conclusões

* Recuperação de Conhecimento (RAG) aprimora LLMs permitindo que acessem informação externa, atualizada e específica.  
* O processo envolve Recuperação (buscando uma base de conhecimento por snippets relevantes) e Aumento (adicionando estes snippets ao prompt do LLM).  
* RAG ajuda LLMs superar limitações como dados de treinamento desatualizados, reduz "alucinações," e permite integração de conhecimento específico de domínio.  
* RAG permite respostas atribuíveis, pois a resposta do LLM é fundamentada em fontes recuperadas.  
* GraphRAG aproveita um grafo de conhecimento para entender os relacionamentos entre diferentes peças de informação, permitindo que responda questões complexas que requerem sintetizar dados de múltiplas fontes.  
* RAG Agêntico move além de simples recuperação de informação usando um agente inteligente para raciocinar ativamente sobre, validar e refinar conhecimento externo, garantindo uma resposta mais precisa e confiável.  
* Aplicações práticas abrangem busca empresarial, suporte ao cliente, pesquisa legal e recomendações personalizadas.

# Conclusão

Em conclusão, Geração Aumentada por Recuperação (RAG) aborda a limitação central do conhecimento estático de um Grande Modelo de Linguagem conectando-o a fontes de dados externas e atualizadas. O processo funciona primeiro recuperando snippets relevantes de informação e então aumentando o prompt do usuário, permitindo que o LLM gere respostas mais precisas e contextualmente conscientes. Isto é tornado possível por tecnologias fundamentais como embeddings, busca semântica e bancos de dados vetoriais, que encontram informação baseada em significado ao invés de apenas palavras-chave. Ao fundamentar saídas em dados verificáveis, RAG reduz significativamente erros factuais e permite o uso de informação proprietária, aprimorando confiança através de citações.

Uma evolução avançada, RAG Agêntico, introduz uma camada de raciocínio que valida ativamente, reconcilia e sintetiza conhecimento recuperado para ainda maior confiabilidade. Similarmente, abordagens especializadas como GraphRAG aproveitam grafos de conhecimento para navegar relacionamentos explícitos de dados, permitindo que o sistema sintetize respostas a consultas altamente complexas e interconectadas. Este agente pode resolver informação conflitante, realizar consultas multi-passo e usar ferramentas externas para encontrar dados faltantes. Embora estes métodos avançados adicionem complexidade e latência, eles melhoram drasticamente a profundidade e confiabilidade da resposta final. Aplicações práticas para estes padrões já estão transformando indústrias, desde busca empresarial e suporte ao cliente até entrega de conteúdo personalizada. Apesar dos desafios, RAG é um padrão crucial para tornar IA mais conhecedora, confiável e útil. Em última análise, transforma LLMs de conversadores de livro fechado em ferramentas poderosas de raciocínio de livro aberto.

# Referências

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)   
2. Google AI for Developers Documentation.  *Retrieval Augmented Generation \- [https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview)*   
3. Retrieval-Augmented Generation with Graphs (GraphRAG), [https://arxiv.org/abs/2501.00309](https://arxiv.org/abs/2501.00309)   
4. LangChain and LangGraph: Leonie Monigatti, "Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation,"  [*https://medium.com/data-science/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2*](https://medium.com/data-science/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)   
5.  Google Cloud Vertex AI RAG Corpus [*https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/manage-your-rag-corpus\#corpus-management*](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/manage-your-rag-corpus#corpus-management)

[image1]: ../assets/19-chapter-14-image-1-line-135.png

[image2]: ../assets/19-chapter-14-image-2-line-137.png

[image3]: ../assets/19-chapter-14-image-3-line-139.png

[image4]: ../assets/19-chapter-14-image-4-line-141.png
