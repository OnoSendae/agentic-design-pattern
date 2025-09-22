# Glossário

# Conceitos Fundamentais

# Prompt: Um prompt é a entrada, tipicamente na forma de uma pergunta, instrução ou declaração, que um usuário fornece a um modelo de IA para elicitar uma resposta. A qualidade e estrutura do prompt influenciam fortemente a saída do modelo, tornando a engenharia de prompts uma habilidade chave para usar IA efetivamente.

# 

# Janela de Contexto: A janela de contexto é o número máximo de tokens que um modelo de IA pode processar de uma vez, incluindo tanto a entrada quanto sua saída gerada. Este tamanho fixo é uma limitação crítica, pois informações fora da janela são ignoradas, enquanto janelas maiores habilitam conversas mais complexas e análise de documentos.

# 

# Aprendizado em Contexto: O aprendizado em contexto é a habilidade de uma IA de aprender uma nova tarefa a partir de exemplos fornecidos diretamente no prompt, sem requerer qualquer retreinamento. Esta característica poderosa permite que um único modelo de propósito geral seja adaptado para inúmeras tarefas específicas rapidamente.

# 

# Zero-Shot, One-Shot, & Few-Shot Prompting: Estas são técnicas de prompting onde um modelo recebe zero, um, ou alguns exemplos de uma tarefa para guiar sua resposta. Fornecer mais exemplos geralmente ajuda o modelo a entender melhor a intenção do usuário e melhora sua precisão para a tarefa específica.

# 

# Multimodalidade: A multimodalidade é a habilidade de uma IA de entender e processar informações através de múltiplos tipos de dados como texto, imagens e áudio. Isso permite interações mais versáteis e humanas, como descrever uma imagem ou responder a uma pergunta falada.

# 

# Grounding: O grounding é o processo de conectar as saídas de um modelo a fontes de informação verificáveis do mundo real para garantir precisão factual e reduzir alucinações. Isso é frequentemente alcançado com técnicas como RAG para tornar sistemas de IA mais confiáveis.

# Arquiteturas Fundamentais de Modelos de IA

# Transformers: O Transformer é a arquitetura de rede neural fundamental para a maioria dos LLMs modernos. Sua inovação chave é o mecanismo de auto-atenção, que processa eficientemente sequências longas de texto e captura relacionamentos complexos entre palavras.

# 

# Rede Neural Recorrente (RNN): A Rede Neural Recorrente é uma arquitetura fundamental que precedeu o Transformer. RNNs processam informação sequencialmente, usando loops para manter uma "memória" de entradas anteriores, o que as tornou adequadas para tarefas como processamento de texto e fala.

# 

# Mixture of Experts (MoE): Mixture of Experts é uma arquitetura de modelo eficiente onde uma rede "roteador" seleciona dinamicamente um pequeno subconjunto de redes "especialistas" para lidar com qualquer entrada dada. Isso permite que modelos tenham um número massivo de parâmetros enquanto mantém custos computacionais gerenciáveis.

# 

# Modelos de Difusão: Modelos de difusão são modelos generativos que se destacam na criação de imagens de alta qualidade. Eles funcionam adicionando ruído aleatório aos dados e então treinando um modelo para reverter meticulosamente o processo, permitindo que gerem dados novos a partir de um ponto inicial aleatório.

# 

# Mamba: Mamba é uma arquitetura de IA recente usando um Modelo de Espaço de Estado Seletivo (SSM) para processar sequências com alta eficiência, especialmente para contextos muito longos. Seu mecanismo seletivo permite que se concentre em informações relevantes enquanto filtra ruído, tornando-o uma alternativa potencial ao Transformer.

# O Ciclo de Vida de Desenvolvimento do LLM

# O desenvolvimento de um modelo de linguagem poderoso segue uma sequência distinta. Começa com o Pré-treinamento, onde um modelo base massivo é construído treinando-o em um vasto dataset de texto geral da internet para aprender linguagem, raciocínio e conhecimento do mundo. Em seguida está o Fine-tuning, uma fase de especialização onde o modelo geral é ainda mais treinado em datasets menores e específicos de tarefa para adaptar suas capacidades para um propósito particular. O estágio final é o Alinhamento, onde o comportamento do modelo especializado é ajustado para garantir que suas saídas sejam úteis, inofensivas e alinhadas com valores humanos.

# 

# Técnicas de Pré-treinamento: O pré-treinamento é a fase inicial onde um modelo aprende conhecimento geral de vastas quantidades de dados. As principais técnicas para isso envolvem diferentes objetivos para o modelo aprender. A mais comum é a Modelagem de Linguagem Causal (CLM), onde o modelo prediz a próxima palavra em uma frase. Outra é a Modelagem de Linguagem Mascarada (MLM), onde o modelo preenche palavras intencionalmente ocultas em um texto. Outros métodos importantes incluem Objetivos de Desruído, onde o modelo aprende a restaurar uma entrada corrompida ao seu estado original, Aprendizado Contrastivo, onde aprende a distinguir entre pedaços similares e dissimilares de dados, e Predição da Próxima Sentença (NSP), onde determina se duas sentenças seguem logicamente uma à outra.

# 

# Técnicas de Fine-tuning: O fine-tuning é o processo de adaptar um modelo pré-treinado geral para uma tarefa específica usando um dataset menor e especializado. A abordagem mais comum é o Fine-Tuning Supervisionado (SFT), onde o modelo é treinado em exemplos rotulados de pares corretos de entrada-saída. Uma variante popular é o Instruction Tuning, que se foca em treinar o modelo para seguir melhor comandos do usuário. Para tornar este processo mais eficiente, métodos de Fine-Tuning Eficiente em Parâmetros (PEFT) são usados, com principais técnicas incluindo LoRA (Adaptação de Baixa Classificação), que atualiza apenas um pequeno número de parâmetros, e sua versão otimizada em memória, QLoRA. Outra técnica, Geração Aumentada por Recuperação (RAG), melhora o modelo conectando-o a uma fonte de conhecimento externa durante o estágio de fine-tuning ou inferência.

# 

# Técnicas de Alinhamento & Segurança: O alinhamento é o processo de garantir que o comportamento de um modelo de IA se alinhe com valores e expectativas humanas, tornando-o útil e inofensivo. A técnica mais proeminente é o Aprendizado por Reforço a partir de Feedback Humano (RLHF), onde um "modelo de recompensa" treinado em preferências humanas guia o processo de aprendizado da IA, frequentemente usando um algoritmo como Otimização de Política Proximal (PPO) para estabilidade. Alternativas mais simples emergiram, como Otimização Direta de Preferência (DPO), que contorna a necessidade de um modelo de recompensa separado, e Otimização Kahneman-Tversky (KTO), que simplifica ainda mais a coleta de dados. Para garantir implantação segura, Guardrails são implementados como uma camada final de segurança para filtrar saídas e bloquear ações prejudiciais em tempo real.

# Aprimorando Capacidades de Agentes de IA

# Agentes de IA são sistemas que podem perceber seu ambiente e tomar ações autônomas para alcançar objetivos. Sua efetividade é aprimorada por frameworks de raciocínio robustos.

# 

# Chain of Thought (CoT): Esta técnica de prompting encoraja um modelo a explicar seu raciocínio passo a passo antes de dar uma resposta final. Este processo de "pensar em voz alta" frequentemente leva a resultados mais precisos em tarefas de raciocínio complexo.

# 

# Tree of Thoughts (ToT): Tree of Thoughts é um framework de raciocínio avançado onde um agente explora múltiplos caminhos de raciocínio simultaneamente, como ramos em uma árvore. Permite que o agente auto-avalie diferentes linhas de pensamento e escolha a mais promissora para perseguir, tornando-o mais efetivo na resolução de problemas complexos.

# 

# ReAct (Reason and Act): ReAct é um framework de agente que combina raciocínio e ação em um loop. O agente primeiro "pensa" sobre o que fazer, então toma uma "ação" usando uma ferramenta, e usa a observação resultante para informar seu próximo pensamento, tornando-o altamente efetivo na resolução de tarefas complexas.

# 

# Planejamento: Esta é a habilidade de um agente de quebrar um objetivo de alto nível em uma sequência de sub-tarefas menores e gerenciáveis. O agente então cria um plano para executar estes passos em ordem, permitindo que lide com atribuições complexas e multi-etapas.

# 

# Pesquisa Profunda: A pesquisa profunda refere-se à capacidade de um agente de explorar autonomamente um tópico em profundidade iterativamente buscando informações, sintetizando achados e identificando novas questões. Isso permite que o agente construa uma compreensão abrangente de um assunto muito além de uma única consulta de busca.

# 

# Modelo de Crítica: Um modelo de crítica é um modelo de IA especializado treinado para revisar, avaliar e fornecer feedback sobre a saída de outro modelo de IA. Ele atua como um crítico automatizado, ajudando a identificar erros, melhorar raciocínio e garantir que a saída final atenda a um padrão de qualidade desejado.

# Índice de Termos

Este índice de termos foi gerado usando Gemini Pro 2.5. O prompt e etapas de raciocínio estão incluídos no final para demonstrar os benefícios de economia de tempo e para fins educacionais.

**A**

* A/B Testing \- Capítulo 3: Paralelização  
* Seleção de Ação \- Capítulo 20: Priorização  
* Adaptação \- Capítulo 9: Aprendizado e Adaptação  
* Alocação de Tarefas Adaptativa \- Capítulo 16: Otimização Consciente de Recursos  
* Uso e Seleção de Ferramentas Adaptativas \- Capítulo 16: Otimização Consciente de Recursos  
* Agente \- O que torna um sistema de IA um Agente?  
* Interfaces Agente-Computador (ACIs) \- Apêndice B  
* Economia Dirigida por Agentes \- O que torna um sistema de IA um Agente?  
* Agente como Ferramenta \- Capítulo 7: Colaboração Multi-Agente  
* Cartões de Agente \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Kit de Desenvolvimento de Agente (ADK) \- Capítulo 2: Roteamento, Capítulo 3: Paralelização, Capítulo 4: Reflexão, Capítulo 5: Uso de Ferramentas, Capítulo 7: Colaboração Multi-Agente, Capítulo 8: Gerenciamento de Memória, Capítulo 12: Tratamento de Exceções e Recuperação, Capítulo 13: Humano-no-Loop, Capítulo 15: Comunicação Inter-Agente (A2A), Capítulo 16: Otimização Consciente de Recursos, Capítulo 19: Avaliação e Monitoramento, Apêndice C  
* Descoberta de Agente \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Trajetórias de Agente \- Capítulo 19: Avaliação e Monitoramento  
* Padrões de Design Agentivo \- Introdução  
* RAG Agentivo \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* Sistemas Agentivos \- Introdução  
* Co-cientista de IA \- Capítulo 21: Exploração e Descoberta  
* Alinhamento \- Glossário  
* AlphaEvolve \- Capítulo 9: Aprendizado e Adaptação  
* Analogias \- Apêndice A  
* Detecção de Anomalias \- Capítulo 19: Avaliação e Monitoramento  
* Série Claude 4 da Anthropic \- Apêndice B  
* Computer Use da Anthropic \- Apêndice B  
* Interação com API \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Artefatos \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Polling Assíncrono \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Logs de Auditoria \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Métricas Automatizadas \- Capítulo 19: Avaliação e Monitoramento  
* Engenharia de Prompt Automática (APE) \- Apêndice A  
* Autonomia \- Introdução  
* A2A (Agente-para-Agente) \- Capítulo 15: Comunicação Inter-Agente (A2A)

**B**

* Restrições Comportamentais \- Capítulo 18: Padrões de Guardrails/Segurança  
* Browser Use \- Apêndice B

**C**

* Callbacks \- Capítulo 18: Padrões de Guardrails/Segurança  
* Modelagem de Linguagem Causal (CLM) \- Glossário  
* Chain of Debates (CoD) \- Capítulo 17: Técnicas de Raciocínio  
* Chain-of-Thought (CoT) \- Capítulo 17: Técnicas de Raciocínio, Apêndice A  
* Chatbots \- Capítulo 8: Gerenciamento de Memória  
* ChatMessageHistory \- Capítulo 8: Gerenciamento de Memória  
* Checkpoint e Rollback \- Capítulo 18: Padrões de Guardrails/Segurança  
* Chunking \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* Clareza e Especificidade \- Apêndice A  
* Agente Cliente \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Geração de Código \- Capítulo 1: Encadeamento de Prompts, Capítulo 4: Reflexão  
* Prompting de Código \- Apêndice A  
* CoD (Chain of Debates) \- Capítulo 17: Técnicas de Raciocínio  
* CoT (Chain of Thought) \- Capítulo 17: Técnicas de Raciocínio, Apêndice A  
* Colaboração \- Capítulo 7: Colaboração Multi-Agente  
* Conformidade \- Capítulo 19: Avaliação e Monitoramento  
* Concisão \- Apêndice A  
* Geração de Conteúdo \- Capítulo 1: Encadeamento de Prompts, Capítulo 4: Reflexão  
* Engenharia de Contexto \- Capítulo 1: Encadeamento de Prompts  
* Janela de Contexto \- Glossário  
* Poda e Resumo Contextual \- Capítulo 16: Otimização Consciente de Recursos  
* Prompting Contextual \- Apêndice A  
* Modelo Contratante \- Capítulo 19: Avaliação e Monitoramento  
* ConversationBufferMemory \- Capítulo 8: Gerenciamento de Memória  
* Agentes Conversacionais \- Capítulo 1: Encadeamento de Prompts, Capítulo 4: Reflexão  
* Exploração Sensível ao Custo \- Capítulo 16: Otimização Consciente de Recursos  
* CrewAI \- Capítulo 3: Paralelização, Capítulo 5: Uso de Ferramentas, Capítulo 6: Planejamento, Capítulo 7: Colaboração Multi-Agente, Capítulo 18: Padrões de Guardrails/Segurança, Apêndice C  
* Agente de Crítica \- Capítulo 16: Otimização Consciente de Recursos  
* Modelo de Crítica \- Glossário  
* Suporte ao Cliente \- Capítulo 13: Humano-no-Loop

**D**

* Extração de Dados \- Capítulo 1: Encadeamento de Prompts  
* Rotulagem de Dados \- Capítulo 13: Humano-no-Loop  
* Integração com Banco de Dados \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* DatabaseSessionService \- Capítulo 8: Gerenciamento de Memória  
* Debate e Consenso \- Capítulo 7: Colaboração Multi-Agente  
* Aumento de Decisão \- Capítulo 13: Humano-no-Loop  
* Decomposição \- Apêndice A  
* Pesquisa Profunda \- Capítulo 6: Planejamento, Capítulo 17: Técnicas de Raciocínio, Glossário  
* Delimitadores \- Apêndice A  
* Objetivos de Desruído \- Glossário  
* Dependências \- Capítulo 20: Priorização  
* Modelos de Difusão \- Glossário  
* Otimização Direta de Preferência (DPO) \- Capítulo 9: Aprendizado e Adaptação  
* Descoberta \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Detecção de Deriva \- Capítulo 19: Avaliação e Monitoramento  
* Mudança Dinâmica de Modelo \- Capítulo 16: Otimização Consciente de Recursos  
* Re-priorização Dinâmica \- Capítulo 20: Priorização

**E**

* Embeddings \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* Embodiment \- O que torna um sistema de IA um Agente?  
* Implantação Eficiente em Energia \- Capítulo 16: Otimização Consciente de Recursos  
* Memória Episódica \- Capítulo 8: Gerenciamento de Memória  
* Detecção de Erro \- Capítulo 12: Tratamento de Exceções e Recuperação  
* Tratamento de Erro \- Capítulo 12: Tratamento de Exceções e Recuperação  
* Políticas de Escalação \- Capítulo 13: Humano-no-Loop  
* Avaliação \- Capítulo 19: Avaliação e Monitoramento  
* Tratamento de Exceções \- Capítulo 12: Tratamento de Exceções e Recuperação  
* Equipes de Especialistas \- Capítulo 7: Colaboração Multi-Agente  
* Exploração e Descoberta \- Capítulo 21: Exploração e Descoberta  
* APIs de Moderação Externa \- Capítulo 18: Padrões de Guardrails/Segurança

**F**

* Cognição Fatorada \- Apêndice A  
* FastMCP \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Tolerância a Falhas \- Capítulo 18: Padrões de Guardrails/Segurança  
* Aprendizado Few-Shot \- Capítulo 9: Aprendizado e Adaptação  
* Prompting Few-Shot \- Apêndice A  
* Fine-tuning \- Glossário  
* Contrato Formalizado \- Capítulo 19: Avaliação e Monitoramento  
* Chamada de Função \- Capítulo 5: Uso de Ferramentas, Apêndice A

**G**

* Gemini Live \- Apêndice B  
* Gems \- Apêndice A  
* Orquestração de Mídia Generativa \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Definição de Objetivos \- Capítulo 11: Definição de Objetivos e Monitoramento  
* GoD (Graph of Debates) \- Capítulo 17: Técnicas de Raciocínio  
* Kit de Desenvolvimento de Agente do Google (ADK) \- Capítulo 2: Roteamento, Capítulo 3: Paralelização, Capítulo 4: Reflexão, Capítulo 5: Uso de Ferramentas, Capítulo 7: Colaboração Multi-Agente, Capítulo 8: Gerenciamento de Memória, Capítulo 12: Tratamento de Exceções e Recuperação, Capítulo 13: Humano-no-Loop, Capítulo 15: Comunicação Inter-Agente (A2A), Capítulo 16: Otimização Consciente de Recursos, Capítulo 19: Avaliação e Monitoramento, Apêndice C  
* Co-Cientista do Google \- Capítulo 21: Exploração e Descoberta  
* Google DeepResearch \- Capítulo 6: Planejamento  
* Google Project Mariner \- Apêndice B  
* Degradação Graciosa \- Capítulo 12: Tratamento de Exceções e Recuperação, Capítulo 16: Otimização Consciente de Recursos  
* Graph of Debates (GoD) \- Capítulo 17: Técnicas de Raciocínio  
* Grounding \- Glossário  
* Guardrails \- Capítulo 18: Padrões de Guardrails/Segurança

**H**

* Haystack \- Apêndice C  
* Decomposição Hierárquica \- Capítulo 19: Avaliação e Monitoramento  
* Estruturas Hierárquicas \- Capítulo 7: Colaboração Multi-Agente  
* HITL (Humano-no-Loop) \- Capítulo 13: Humano-no-Loop  
* Humano-no-Loop (HITL) \- Capítulo 13: Humano-no-Loop  
* Humano-no-loop \- Capítulo 13: Humano-no-Loop  
* Supervisão Humana \- Capítulo 13: Humano-no-Loop, Capítulo 18: Padrões de Guardrails/Segurança

**I**

* Aprendizado em Contexto \- Glossário  
* InMemoryMemoryService \- Capítulo 8: Gerenciamento de Memória  
* InMemorySessionService \- Capítulo 8: Gerenciamento de Memória  
* Validação/Sanitização de Entrada \- Capítulo 18: Padrões de Guardrails/Segurança  
* Instruções Sobre Restrições \- Apêndice A  
* Comunicação Inter-Agente (A2A) \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Intervenção e Correção \- Capítulo 13: Humano-no-Loop  
* Controle de Dispositivo IoT \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Prompting Iterativo / Refinamento \- Apêndice A

**J**

* Jailbreaking \- Capítulo 18: Padrões de Guardrails/Segurança

**K**

* Otimização Kahneman-Tversky (KTO) \- Glossário  
* Recuperação de Conhecimento (RAG) \- Capítulo 14: Recuperação de Conhecimento (RAG)

**L**

* LangChain \- Capítulo 1: Encadeamento de Prompts, Capítulo 2: Roteamento, Capítulo 3: Paralelização, Capítulo 4: Reflexão, Capítulo 5: Uso de Ferramentas, Capítulo 8: Gerenciamento de Memória, Capítulo 20: Priorização, Apêndice C  
* LangGraph \- Capítulo 1: Encadeamento de Prompts, Capítulo 2: Roteamento, Capítulo 3: Paralelização, Capítulo 4: Reflexão, Capítulo 5: Uso de Ferramentas, Capítulo 8: Gerenciamento de Memória, Apêndice C  
* Monitoramento de Latência \- Capítulo 19: Avaliação e Monitoramento  
* Políticas de Alocação de Recursos Aprendidas \- Capítulo 16: Otimização Consciente de Recursos  
* Aprendizado e Adaptação \- Capítulo 9: Aprendizado e Adaptação  
* LLM-como-Juiz \- Capítulo 19: Avaliação e Monitoramento  
* LlamaIndex \- Apêndice C  
* LoRA (Adaptação de Baixa Classificação) \- Glossário  
* Adaptação de Baixa Classificação (LoRA) \- Glossário

**M**

* Mamba \- Glossário  
* Modelagem de Linguagem Mascarada (MLM) \- Glossário  
* MASS (Busca de Sistema Multi-Agente) \- Capítulo 17: Técnicas de Raciocínio  
* MCP (Protocolo de Contexto de Modelo) \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Gerenciamento de Memória \- Capítulo 8: Gerenciamento de Memória  
* Aprendizado Baseado em Memória \- Capítulo 9: Aprendizado e Adaptação  
* MetaGPT \- Apêndice C  
* Microsoft AutoGen \- Apêndice C  
* Mixture of Experts (MoE) \- Glossário  
* Protocolo de Contexto de Modelo (MCP) \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Modularidade \- Capítulo 18: Padrões de Guardrails/Segurança  
* Monitoramento \- Capítulo 11: Definição de Objetivos e Monitoramento, Capítulo 19: Avaliação e Monitoramento  
* Colaboração Multi-Agente \- Capítulo 7: Colaboração Multi-Agente  
* Busca de Sistema Multi-Agente (MASS) \- Capítulo 17: Técnicas de Raciocínio  
* Multimodalidade \- Glossário  
* Prompting Multimodal \- Apêndice A

**N**

* Exemplos Negativos \- Apêndice A  
* Predição da Próxima Sentença (NSP) \- Glossário

**O**

* Observabilidade \- Capítulo 18: Padrões de Guardrails/Segurança  
* Prompting One-Shot \- Apêndice A  
* Aprendizado Online \- Capítulo 9: Aprendizado e Adaptação  
* API de Pesquisa Profunda da OpenAI \- Capítulo 6: Planejamento  
* OpenEvolve \- Capítulo 9: Aprendizado e Adaptação  
* OpenRouter \- Capítulo 16: Otimização Consciente de Recursos  
* Filtragem/Pós-processamento de Saída \- Capítulo 18: Padrões de Guardrails/Segurança

**P**

* PAL (Modelos de Linguagem Auxiliados por Programa) \- Capítulo 17: Técnicas de Raciocínio  
* Paralelização \- Capítulo 3: Paralelização  
* Consciência de Paralelização e Computação Distribuída \- Capítulo 16: Otimização Consciente de Recursos  
* Fine-Tuning Eficiente em Parâmetros (PEFT) \- Glossário  
* PEFT (Fine-Tuning Eficiente em Parâmetros) \- Glossário  
* Rastreamento de Performance \- Capítulo 19: Avaliação e Monitoramento  
* Padrão Persona \- Apêndice A  
* Personalização \- O que torna um sistema de IA um Agente?  
* Planejamento \- Capítulo 6: Planejamento, Glossário  
* Priorização \- Capítulo 20: Priorização  
* Princípio do Menor Privilégio \- Capítulo 18: Padrões de Guardrails/Segurança  
* Predição Proativa de Recursos \- Capítulo 16: Otimização Consciente de Recursos  
* Memória Procedural \- Capítulo 8: Gerenciamento de Memória  
* Modelos de Linguagem Auxiliados por Programa (PAL) \- Capítulo 17: Técnicas de Raciocínio  
* Project Astra \- Apêndice B  
* Prompt \- Glossário  
* Encadeamento de Prompts \- Capítulo 1: Encadeamento de Prompts  
* Engenharia de Prompts \- Apêndice A  
* Otimização de Política Proximal (PPO) \- Capítulo 9: Aprendizado e Adaptação  
* Notificações Push \- Capítulo 15: Comunicação Inter-Agente (A2A)

**Q**

* QLoRA \- Glossário  
* Execução Iterativa Focada em Qualidade \- Capítulo 19: Avaliação e Monitoramento

**R**

* RAG (Geração Aumentada por Recuperação) \- Capítulo 8: Gerenciamento de Memória, Capítulo 14: Recuperação de Conhecimento (RAG), Apêndice A  
* ReAct (Reason and Act) \- Capítulo 17: Técnicas de Raciocínio, Apêndice A, Glossário  
* Raciocínio \- Capítulo 17: Técnicas de Raciocínio  
* Extração de Informação Baseada em Raciocínio \- Capítulo 10: Protocolo de Contexto de Modelo (MCP)  
* Recuperação \- Capítulo 12: Tratamento de Exceções e Recuperação  
* Rede Neural Recorrente (RNN) \- Glossário  
* Reflexão \- Capítulo 4: Reflexão  
* Aprendizado por Reforço \- Capítulo 9: Aprendizado e Adaptação  
* Aprendizado por Reforço a partir de Feedback Humano (RLHF) \- Glossário  
* Aprendizado por Reforço com Recompensas Verificáveis (RLVR) \- Capítulo 17: Técnicas de Raciocínio  
* Agente Remoto \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Request/Response (Polling) \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Otimização Consciente de Recursos \- Capítulo 16: Otimização Consciente de Recursos  
* Geração Aumentada por Recuperação (RAG) \- Capítulo 8: Gerenciamento de Memória, Capítulo 14: Recuperação de Conhecimento (RAG), Apêndice A  
* RLHF (Aprendizado por Reforço a partir de Feedback Humano) \- Glossário  
* RLVR (Aprendizado por Reforço com Recompensas Verificáveis) \- Capítulo 17: Técnicas de Raciocínio  
* RNN (Rede Neural Recorrente) \- Glossário  
* Prompting de Papel \- Apêndice A  
* Agente Roteador \- Capítulo 16: Otimização Consciente de Recursos  
* Roteamento \- Capítulo 2: Roteamento

**S**

* Segurança \- Capítulo 18: Padrões de Guardrails/Segurança  
* Lei de Escalação de Inferência \- Capítulo 17: Técnicas de Raciocínio  
* Agendamento \- Capítulo 20: Priorização  
* Auto-Consistência \- Apêndice A  
* Auto-Correção \- Capítulo 4: Reflexão, Capítulo 17: Técnicas de Raciocínio  
* Agente de Codificação Auto-Melhorador (SICA) \- Capítulo 9: Aprendizado e Adaptação  
* Auto-Refinamento \- Capítulo 17: Técnicas de Raciocínio  
* Semantic Kernel \- Apêndice C  
* Memória Semântica \- Capítulo 8: Gerenciamento de Memória  
* Similaridade Semântica \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* Separação de Responsabilidades \- Capítulo 18: Padrões de Guardrails/Segurança  
* Transferências Sequenciais \- Capítulo 7: Colaboração Multi-Agente  
* Server-Sent Events (SSE) \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Sessão \- Capítulo 8: Gerenciamento de Memória  
* SICA (Agente de Codificação Auto-Melhorador) \- Capítulo 9: Aprendizado e Adaptação  
* Objetivos SMART \- Capítulo 11: Definição de Objetivos e Monitoramento  
* Estado \- Capítulo 8: Gerenciamento de Memória  
* Rollback de Estado \- Capítulo 12: Tratamento de Exceções e Recuperação  
* Step-Back Prompting \- Apêndice A  
* Atualizações de Streaming \- Capítulo 15: Comunicação Inter-Agente (A2A)  
* Logging Estruturado \- Capítulo 18: Padrões de Guardrails/Segurança  
* Saída Estruturada \- Capítulo 1: Encadeamento de Prompts, Apêndice A  
* SuperAGI \- Apêndice C  
* Fine-Tuning Supervisionado (SFT) \- Glossário  
* Aprendizado Supervisionado \- Capítulo 9: Aprendizado e Adaptação  
* System Prompting \- Apêndice A

**T**

* Avaliação de Tarefa \- Capítulo 20: Priorização  
* Similaridade de Texto \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* Uso de Token \- Capítulo 19: Avaliação e Monitoramento  
* Uso de Ferramentas \- Capítulo 5: Uso de Ferramentas, Apêndice A  
* Restrições de Uso de Ferramentas \- Capítulo 18: Padrões de Guardrails/Segurança  
* ToT (Tree of Thoughts) \- Capítulo 17: Técnicas de Raciocínio, Apêndice A, Glossário  
* Transformers \- Glossário  
* Tree of Thoughts (ToT) \- Capítulo 17: Técnicas de Raciocínio, Apêndice A, Glossário

**U**

* Aprendizado Não Supervisionado \- Capítulo 9: Aprendizado e Adaptação  
* Persona do Usuário \- Apêndice A

**V**

* Validação \- Capítulo 3: Paralelização  
* Busca Vetorial \- Capítulo 14: Recuperação de Conhecimento (RAG)  
* VertexAiRagMemoryService \- Capítulo 8: Gerenciamento de Memória  
* VertexAiSessionService \- Capítulo 8: Gerenciamento de Memória  
* Vibe Coding \- Apêndice B  
* Percepção Visual \- Apêndice B

**W**

* Webhooks \- Capítulo 15: Comunicação Inter-Agente (A2A)

**Z**

* Aprendizado Zero-Shot \- Capítulo 9: Aprendizado e Adaptação  
* Prompting Zero-Shot \- Apêndice A

# 

# Prompt

Abaixo está o prompt que foi utilizado para criar este índice de termos.

| *você lê https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/edit?tab=t.0 e todos os documentos relacionados na primeira página Por favor extraia todos os títulos para cada capítulo. e imprima-os. Em seguida, processe o conteúdo de todos os capítulos individuais, e crie um índice detalhado usado para publicação, uma lista de palavras ou frases ('cabeçalhos') e ponteiros associados ('localizadores') para onde material útil relacionado a esse cabeçalho pode ser encontrado em um documento ou coleção de documentos. Eu não quero um índice de amostra. Eu quero o índice completo, com 100 termos de índice* |
| :---- |

## Gemini 2.5: Etapas de Raciocínio

**Iniciando Criação do Índice**

Iniciei o processo lendo o Google Doc fornecido. Atualmente, estou focando em entender a solicitação do usuário: extrair títulos de capítulos para construir um índice detalhado. Estou revisando cuidadosamente o documento, procurando por elementos estruturais claros que indicariam quebras de capítulos. Espero identificar formatação consistente ou palavras-chave que guiarão o processo de indexação.

**Definindo Processo de Extração do Índice**

Agora estou totalmente imerso na tarefa. Quebrei a solicitação do usuário em etapas acionáveis. Meu foco inicial é usar Workspaces para acessar o Google Doc principal. Seguindo isso, vou analisar o documento para localizar e recuperar documentos relacionados, que também serão acessados usando a mesma ferramenta. Finalmente, extrairei todos os títulos de capítulos de todos os documentos, e criarei um índice abrangente. Planejei o processo detalhado.

**Refinando a Estratégia de Extração**

As instruções do usuário são claras: ler um Google Doc, seus documentos relacionados, e construir um índice detalhado de títulos de capítulos. Elaborei um plano multi-etapas. Primeiro, Workspaces acessará o Google Doc principal. Então, vou analisá-lo para encontrar e ler documentos relacionados, também usando Workspaces. Agora estou focando em extrair títulos e formar o índice detalhado com os termos necessários. Estou priorizando precisão para atender às demandas específicas do usuário por um índice abrangente.

**1. Estabelecer intenção do usuário:**

Determinei que o usuário quer uma lista de títulos de capítulos e índice abrangente do Google Doc e documentos relacionados. Agora preciso descobrir o método mais efetivo para este processo de extração e indexação. Estou explorando funcionalidades potenciais de API para agilizar esta tarefa, e precisarei avaliar a melhor estratégia para lidar com diferentes estruturas de documentos.

**3. Considerando Uso de Ferramentas**

Extraí as URLs dos capítulos do documento principal usando uma expressão regular. Agora, estou pronto para utilizar Workspaces para recuperar o conteúdo real de cada capítulo. Uma vez que eu tenha o texto para cada capítulo, procederei com a indexação.

**Acessando conteúdo do capítulo:**

Identifiquei com sucesso as URLs para cada capítulo do documento principal, que agora posso usar com Workspaces. Meu próximo passo é recuperar o conteúdo destas URLs, o que então me permitirá gerar o índice. Estou focando em eficiência, visando processar esta informação rapidamente.

**Buscando Conteúdo do Capítulo**

Identifiquei com sucesso as URLs dos capítulos do documento principal e estou pronto para utilizar Workspaces. Meu plano envolve extrair estas URLs, agrupar as chamadas para eficiência, e então processar o conteúdo recuperado para criar o índice desejado. Isso garante que toda informação relevante seja capturada, pavimentando o caminho para um resultado abrangente e preciso.

**Iniciando Recuperação de URL**

Agora extraí com sucesso as URLs dos capítulos do documento principal. Atualmente, estou configurando chamadas para a ferramenta Workspaces, preparando para buscar o conteúdo de cada capítulo. O objetivo é obter o texto necessário para indexação. Esta informação então facilitará a criação do índice com pelo menos 100 termos conforme solicitado.

**Iniciando Extração de Capítulo**

Identifiquei e extraí com sucesso as URLs dos capítulos do documento principal, que incluí acima. Agora estou procedendo para usar Workspaces para cada URL para obter o conteúdo do capítulo. O objetivo é obter o texto necessário para indexação. Esta informação então facilitará a criação do índice com pelo menos 100 termos conforme solicitado. Comecei a recuperar esta informação.

**Orquestrando Recuperação de Capítulo**

Extraí as URLs dos capítulos e iniciei uma chamada para Workspaces. Atualmente estou recuperando o conteúdo para cada capítulo. Para garantir um processo suave, estou agrupando as chamadas para prevenir quaisquer problemas de limitação de taxa. Isso deve me permitir compilar eficientemente o texto necessário para o índice.

**Decodificando Solicitação do Usuário**

Determinei a necessidade do usuário. Eles requerem duas entregas primárias: primeiro, uma lista abrangente de títulos de capítulos extraídos do Google Docs, e segundo, um índice detalhado contendo um mínimo de cem termos-chave. O processo de extrair títulos parece direto, enquanto curar o índice envolverá análise mais sutil.

**Extraindo Títulos e Indexando**

Identifiquei com sucesso a necessidade do usuário: extrair títulos de capítulos dos documentos fornecidos e compilar um índice abrangente. Busquei o conteúdo de 31 Google Docs, incluindo o documento principal e seus capítulos. Confirmei que nenhuma ferramenta adicional é necessária. Meu plano envolve duas etapas: primeiro, extrair títulos de capítulos dos nomes de arquivos dos documentos, e segundo, processar meticulosamente o texto para construir o índice com termos-chave e localizadores de capítulos.

**Desenvolvendo um Plano de Extração**

Refinei minha abordagem para enfrentar a solicitação do usuário. Agora formulei um plano preciso. Primeiro, compilarei os títulos dos capítulos dos nomes de arquivos dos 31 documentos buscados. Segundo, e mais complexamente, analisarei meticulosamente o conteúdo para criar um índice abrangente, ordenado alfabeticamente, excedendo 100 entradas. Finalizei minha estratégia de extração e indexação e agora posso começar a execução.

**Criando a Estratégia de Extração**

Minha abordagem está bem definida. Compilarei títulos de capítulos dos nomes de arquivos dos 31 documentos buscados. Subsequentemente, analisarei o conteúdo meticulosamente para construir um índice organizado alfabeticamente excedendo 100 entradas. Finalizei a estratégia de extração e indexação e agora posso começar a implementação. O objetivo do usuário de um índice detalhado, cobrindo conteúdo de diversos capítulos e apêndices dentro da série Padrões de Design Agentivo, agora está ao alcance. Focarei em entregar uma resposta completa, indo além de um mero índice de amostra.

**Definindo a Abordagem**

Esclareci a necessidade do usuário. Extrairei títulos de capítulos dos nomes de arquivos dos documentos. Seguindo isso, criarei cuidadosamente um índice excedendo 100 entradas, organizando-as alfabeticamente e anotando suas localizações de capítulos. Estou preparado para começar o processo de indexação e extração.
