# Glossário

# Conceitos Fundamentais

**Prompt:** Um prompt é a entrada, tipicamente na forma de uma pergunta, instrução ou declaração, que um usuário fornece a um modelo de IA para elicitar uma resposta. A qualidade e estrutura do prompt influenciam fortemente a saída do modelo, tornando a engenharia de prompts uma habilidade chave para usar IA efetivamente.

**Janela de Contexto:** A janela de contexto é o número máximo de tokens que um modelo de IA pode processar de uma vez, incluindo tanto a entrada quanto sua saída gerada. Este tamanho fixo é uma limitação crítica, pois informações fora da janela são ignoradas, enquanto janelas maiores habilitam conversas mais complexas e análise de documentos.

**Aprendizado em Contexto:** O aprendizado em contexto é a habilidade de uma IA de aprender uma nova tarefa a partir de exemplos fornecidos diretamente no prompt, sem requerer qualquer retreinamento. Esta característica poderosa permite que um único modelo de propósito geral seja adaptado para inúmeras tarefas específicas rapidamente.

**Prompting Zero-Shot, One-Shot, & Few-Shot:** Estas são técnicas de prompting onde um modelo recebe zero, um, ou alguns exemplos de uma tarefa para guiar sua resposta. Fornecer mais exemplos geralmente ajuda o modelo a entender melhor a intenção do usuário e melhora sua precisão para a tarefa específica.

**Multimodalidade:** A multimodalidade é a habilidade de uma IA de entender e processar informações através de múltiplos tipos de dados como texto, imagens e áudio. Isso permite interações mais versáteis e humanas, como descrever uma imagem ou responder a uma pergunta falada.

**Grounding:** O grounding é o processo de conectar as saídas de um modelo a fontes de informação verificáveis do mundo real para garantir precisão factual e reduzir alucinações. Isso é frequentemente alcançado com técnicas como RAG para tornar sistemas de IA mais confiáveis.

# Arquiteturas Fundamentais de Modelos de IA

**Transformers:** O Transformer é a arquitetura de rede neural fundamental para a maioria dos LLMs modernos. Sua inovação chave é o mecanismo de auto-atenção, que processa eficientemente sequências longas de texto e captura relacionamentos complexos entre palavras.

**Rede Neural Recorrente (RNN):** A Rede Neural Recorrente é uma arquitetura fundamental que precedeu o Transformer. RNNs processam informação sequencialmente, usando loops para manter uma "memória" de entradas anteriores, o que as tornou adequadas para tarefas como processamento de texto e fala.

**Mixture of Experts (MoE):** Mixture of Experts é uma arquitetura de modelo eficiente onde uma rede "roteador" seleciona dinamicamente um pequeno subconjunto de redes "especialistas" para lidar com qualquer entrada dada. Isso permite que modelos tenham um número massivo de parâmetros enquanto mantém custos computacionais gerenciáveis.

**Modelos de Difusão:** Modelos de difusão são modelos generativos que se destacam na criação de imagens de alta qualidade. Eles funcionam adicionando ruído aleatório aos dados e então treinando um modelo para reverter meticulosamente o processo, permitindo que gerem dados novos a partir de um ponto inicial aleatório.

**Mamba:** Mamba é uma arquitetura de IA recente usando um Modelo de Espaço de Estado Seletivo (SSM) para processar sequências com alta eficiência, especialmente para contextos muito longos. Seu mecanismo seletivo permite que se concentre em informações relevantes enquanto filtra ruído, tornando-o uma alternativa potencial ao Transformer.

# O Ciclo de Vida de Desenvolvimento do LLM

O desenvolvimento de um modelo de linguagem poderoso segue uma sequência distinta. Começa com o Pré-treinamento, onde um modelo base massivo é construído treinando-o em um vasto dataset de texto geral da internet para aprender linguagem, raciocínio e conhecimento do mundo. Em seguida está o Fine-tuning, uma fase de especialização onde o modelo geral é ainda mais treinado em datasets menores e específicos de tarefa para adaptar suas capacidades para um propósito particular. O estágio final é o Alinhamento, onde o comportamento do modelo especializado é ajustado para garantir que suas saídas sejam úteis, inofensivas e alinhadas com valores humanos.

**Técnicas de Pré-treinamento:** O pré-treinamento é a fase inicial onde um modelo aprende conhecimento geral de vastas quantidades de dados. As principais técnicas para isso envolvem diferentes objetivos para o modelo aprender. A mais comum é a Modelagem de Linguagem Causal (CLM), onde o modelo prediz a próxima palavra em uma frase. Outra é a Modelagem de Linguagem Mascarada (MLM), onde o modelo preenche palavras intencionalmente ocultas em um texto. Outros métodos importantes incluem Objetivos de Desruído, onde o modelo aprende a restaurar uma entrada corrompida ao seu estado original, Aprendizado Contrastivo, onde aprende a distinguir entre pedaços similares e dissimilares de dados, e Predição da Próxima Sentença (NSP), onde determina se duas sentenças seguem logicamente uma à outra.

**Técnicas de Fine-tuning:** O fine-tuning é o processo de adaptar um modelo pré-treinado geral para uma tarefa específica usando um dataset menor e especializado. A abordagem mais comum é o Fine-Tuning Supervisionado (SFT), onde o modelo é treinado em exemplos rotulados de pares corretos de entrada-saída. Uma variante popular é o Instruction Tuning, que se foca em treinar o modelo para seguir melhor comandos do usuário. Para tornar este processo mais eficiente, métodos de Fine-Tuning Eficiente em Parâmetros (PEFT) são usados, com principais técnicas incluindo LoRA (Adaptação de Baixa Classificação), que atualiza apenas um pequeno número de parâmetros, e sua versão otimizada em memória, QLoRA. Outra técnica, Geração Aumentada por Recuperação (RAG), melhora o modelo conectando-o a uma fonte de conhecimento externa durante o estágio de fine-tuning ou inferência.

**Técnicas de Alinhamento & Segurança:** O alinhamento é o processo de garantir que o comportamento de um modelo de IA se alinhe com valores e expectativas humanas, tornando-o útil e inofensivo. A técnica mais proeminente é o Aprendizado por Reforço a partir de Feedback Humano (RLHF), onde um "modelo de recompensa" treinado em preferências humanas guia o processo de aprendizado da IA, frequentemente usando um algoritmo como Otimização de Política Proximal (PPO) para estabilidade. Alternativas mais simples emergiram, como Otimização Direta de Preferência (DPO), que contorna a necessidade de um modelo de recompensa separado, e Otimização Kahneman-Tversky (KTO), que simplifica ainda mais a coleta de dados. Para garantir implantação segura, Guardrails são implementados como uma camada final de segurança para filtrar saídas e bloquear ações prejudiciais em tempo real.

# Aprimorando Capacidades de Agentes de IA

Agentes de IA são sistemas que podem perceber seu ambiente e tomar ações autônomas para alcançar objetivos. Sua efetividade é aprimorada por frameworks de raciocínio robustos.

**Chain of Thought (CoT):** Esta técnica de prompting encoraja um modelo a explicar seu raciocínio passo a passo antes de dar uma resposta final. Este processo de "pensar em voz alta" frequentemente leva a resultados mais precisos em tarefas de raciocínio complexo.

**Tree of Thoughts (ToT):** Tree of Thoughts é um framework de raciocínio avançado onde um agente explora múltiplos caminhos de raciocínio simultaneamente, como ramos em uma árvore. Permite que o agente auto-avalie diferentes linhas de pensamento e escolha a mais promissora para perseguir, tornando-o mais efetivo na resolução de problemas complexos.

**ReAct (Reason and Act):** ReAct é um framework de agente que combina raciocínio e ação em um loop. O agente primeiro "pensa" sobre o que fazer, então toma uma "ação" usando uma ferramenta, e usa a observação resultante para informar seu próximo pensamento, tornando-o altamente efetivo na resolução de tarefas complexas.

**Planejamento:** Esta é a habilidade de um agente de quebrar um objetivo de alto nível em uma sequência de sub-tarefas menores e gerenciáveis. O agente então cria um plano para executar estes passos em ordem, permitindo que lide com atribuições complexas e multi-etapas.

**Pesquisa Profunda:** A pesquisa profunda refere-se à capacidade de um agente de explorar autonomamente um tópico em profundidade iterativamente buscando informações, sintetizando achados e identificando novas questões. Isso permite que o agente construa uma compreensão abrangente de um assunto muito além de uma única consulta de busca.

**Modelo de Crítica:** Um modelo de crítica é um modelo de IA especializado treinado para revisar, avaliar e fornecer feedback sobre a saída de outro modelo de IA. Ele atua como um crítico automatizado, ajudando a identificar erros, melhorar raciocínio e garantir que a saída final atenda a um padrão de qualidade desejado.
