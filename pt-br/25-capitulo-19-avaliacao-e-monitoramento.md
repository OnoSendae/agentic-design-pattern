# Capítulo 19: Avaliação e Monitoramento

Este capítulo examina metodologias que permitem que agentes inteligentes avaliem sistematicamente seu desempenho, monitorem progresso em direção aos objetivos e detectem anomalias operacionais. Enquanto o Capítulo 11 delineia definição de objetivos e monitoramento, e o Capítulo 17 aborda mecanismos de Raciocínio, este capítulo foca na medição contínua, frequentemente externa, da efetividade, eficiência e conformidade de um agente com requisitos. Isto inclui definir métricas, estabelecer loops de feedback e implementar sistemas de relatório para garantir que o desempenho do agente se alinhe com expectativas em ambientes operacionais (veja Fig.1)

![][image1]

Fig:1. Melhores práticas para avaliação e monitoramento

# Aplicações Práticas e Casos de Uso

Aplicações e Casos de Uso Mais Comuns:

* **Rastreamento de Desempenho em Sistemas Vivos:** Monitorar continuamente a precisão, latência e consumo de recursos de um agente implantado em um ambiente de produção (ex., taxa de resolução de um chatbot de atendimento ao cliente, tempo de resposta).  
* **Teste A/B para Melhorias de Agente:** Comparar sistematicamente o desempenho de diferentes versões de agente ou estratégias em paralelo para identificar abordagens ótimas (ex., tentar dois algoritmos de planejamento diferentes para um agente de logística).  
* **Auditorias de Conformidade e Segurança:** Gerar relatórios de auditoria automatizados que rastreiam a conformidade de um agente com diretrizes éticas, requisitos regulatórios e protocolos de segurança ao longo do tempo. Estes relatórios podem ser verificados por um human-in-the-loop ou outro agente, e podem gerar KPIs ou acionar alertas ao identificar problemas.  
* **Sistemas empresariais:** Para governar IA Agêntica em sistemas corporativos, um novo instrumento de controle, o "Contrato" de IA, é necessário. Este acordo dinâmico codifica os objetivos, regras e controles para tarefas delegadas por IA.  
* **Detecção de Deriva:** Monitorar a relevância ou precisão das saídas de um agente ao longo do tempo, detectando quando seu desempenho degrada devido a mudanças na distribuição de dados de entrada (deriva de conceito) ou mudanças ambientais.  
* **Detecção de Anomalia no Comportamento do Agente:** Identificar ações incomuns ou inesperadas tomadas por um agente que podem indicar um erro, um ataque malicioso ou um comportamento emergente indesejado.  
* **Avaliação do Progresso de Aprendizado:** Para agentes projetados para aprender, rastrear sua curva de aprendizado, melhoria em habilidades específicas ou capacidades de generalização em diferentes tarefas ou conjuntos de dados.

# Exemplo de Código Hands-On

Desenvolver um framework abrangente de avaliação para agentes de IA é um empreendimento desafiador, comparável a uma disciplina acadêmica ou uma publicação substancial em sua complexidade. Esta dificuldade decorre da multiplicidade de fatores a considerar, como desempenho do modelo, interação do usuário, implicações éticas e impacto social mais amplo. No entanto, para implementação prática, o foco pode ser estreitado para casos de uso críticos essenciais para o funcionamento eficiente e efetivo de agentes de IA.

**Avaliação de Resposta do Agente:** Este processo central é essencial para avaliar a qualidade e precisão das saídas de um agente. Envolve determinar se o agente entrega informação pertinente, correta, lógica, imparcial e precisa em resposta a entradas dadas. Métricas de avaliação podem incluir correção factual, fluência, precisão gramatical e aderência ao propósito pretendido do usuário.

O problema reside em seu método de comparação. A função executa uma comparação estrita, caractere por caractere, das duas strings. No exemplo fornecido:

* agent_response: "The capital of France is Paris."  
* ground_truth: "Paris is the capital of France."

Mesmo após remover espaços em branco e converter para minúsculas, estas duas strings não são idênticas. Como resultado, a função retornará incorretamente uma pontuação de precisão de `0.0`, mesmo que ambas as frases transmitam o mesmo significado.

Uma comparação direta falha em avaliar similaridade semântica, tendo sucesso apenas se a resposta de um agente corresponder exatamente à saída esperada. Uma avaliação mais efetiva necessita técnicas avançadas de Processamento de Linguagem Natural (NLP) para discernir o significado entre frases. Para avaliação abrangente de agentes de IA em cenários do mundo real, métricas mais sofisticadas são frequentemente indispensáveis. Estas métricas podem abranger Medidas de Similaridade de String como distância de Levenshtein e similaridade de Jaccard, Análise de Palavra-chave para presença ou ausência de palavras-chave específicas, Similaridade Semântica usando similaridade de cosseno com modelos de embedding, Avaliações LLM-como-Juiz (discutidas mais tarde para avaliar correção e utilidade nuances), e Métricas Específicas de RAG como fidelidade e relevância.

**Monitoramento de Latência:** Monitoramento de Latência para Ações de Agente é crucial em aplicações onde a velocidade da resposta ou ação de um agente de IA é um fator crítico. Este processo mede a duração necessária para um agente processar solicitações e gerar saídas. Latência elevada pode afetar adversamente a experiência do usuário e a efetividade geral do agente, particularmente em ambientes em tempo real ou interativos. Em aplicações práticas, simplesmente imprimir dados de latência no console é insuficiente. Registrar esta informação em um sistema de armazenamento persistente é recomendado. Opções incluem arquivos de log estruturados (ex., JSON), bancos de dados de série temporal (ex., InfluxDB, Prometheus), data warehouses (ex., Snowflake, BigQuery, PostgreSQL), ou plataformas de observabilidade (ex., Datadog, Splunk, Grafana Cloud).

**Rastreamento de Uso de Token para Interações LLM:** Para agentes alimentados por LLM, rastrear uso de token é crucial para gerenciar custos e otimizar alocação de recursos. Cobrança por interações LLM frequentemente depende do número de tokens processados (entrada e saída). Portanto, uso eficiente de token reduz diretamente despesas operacionais. Além disso, monitorar contagens de token ajuda a identificar áreas potenciais para melhoria em processos de engenharia de prompt ou geração de resposta.

**Métrica Personalizada para "Utilidade" usando LLM-como-Juiz:** Avaliar qualidades subjetivas como a "utilidade" de um agente de IA apresenta desafios além de métricas objetivas padrão. Um framework potencial envolve usar um LLM como avaliador. Esta abordagem LLM-como-Juiz avalia a saída de outro agente de IA baseada em critérios pré-definidos para "utilidade." Aproveitando as capacidades linguísticas avançadas dos LLMs, este método oferece avaliações nuances, similares às humanas, de qualidades subjetivas, superando correspondência simples de palavras-chave ou avaliações baseadas em regras. Embora em desenvolvimento, esta técnica mostra promessa para automatizar e escalar avaliações qualitativas.

O código Python define uma classe LLMJudgeForLegalSurvey projetada para avaliar a qualidade de questões de pesquisa legal usando um modelo de IA generativa. Ele utiliza a biblioteca google.generativeai para interagir com modelos Gemini. 

A funcionalidade central envolve enviar uma questão de pesquisa ao modelo junto com uma rubrica detalhada para avaliação. A rubrica especifica cinco critérios para julgar questões de pesquisa: Clareza e Precisão, Neutralidade e Viés, Relevância e Foco, Completude e Apropriação para Audiência. Para cada critério, uma pontuação de 1 a 5 é atribuída, e uma justificativa detalhada e feedback são necessários na saída. O código constrói um prompt que inclui a rubrica e a questão de pesquisa a ser avaliada. 

O método judge_survey_question envia este prompt ao modelo Gemini configurado, solicitando uma resposta JSON formatada de acordo com a estrutura definida. A saída JSON esperada inclui uma pontuação geral, uma justificativa resumida, feedback detalhado para cada critério, uma lista de preocupações e uma ação recomendada. A classe lida com erros potenciais durante a interação do modelo de IA, como problemas de decodificação JSON ou respostas vazias. O script demonstra sua operação avaliando exemplos de questões de pesquisa legal, ilustrando como a IA avalia qualidade baseada nos critérios pré-definidos.

Antes de concluirmos, vamos examinar vários métodos de avaliação, considerando seus pontos fortes e fracos.

| Método de Avaliação | Pontos Fortes | Pontos Fracos |
| :---- | :---- | :---- |
| Avaliação Humana  | Captura comportamento sutil | Difícil de escalar, caro e demorado, pois considera fatores humanos subjetivos. |
| LLM-como-Juiz | Consistente, eficiente e escalável.  | Etapas intermediárias podem ser negligenciadas. Limitado pelas capacidades do LLM. |
| Métricas Automatizadas  | Escalável, eficiente e objetivo | Limitação potencial em capturar capacidades completas. |

# Trajetórias de Agentes

Avaliar trajetórias de agentes é essencial, pois testes de software tradicionais são insuficientes. Código padrão produz resultados previsíveis de aprovação/reprovação, enquanto agentes operam probabilisticamente, necessitando avaliação qualitativa tanto da saída final quanto da trajetória do agente—a sequência de etapas tomadas para alcançar uma solução. Avaliar sistemas multi-agente é desafiador porque estão constantemente em fluxo. Isto requer desenvolver métricas sofisticadas que vão além do desempenho individual para medir a efetividade da comunicação e trabalho em equipe. Além disso, os próprios ambientes não são estáticos, demandando que métodos de avaliação, incluindo casos de teste, se adaptem ao longo do tempo.

Isto envolve examinar a qualidade das decisões, o processo de raciocínio e o resultado geral. Implementar avaliações automatizadas é valioso, particularmente para desenvolvimento além do estágio de protótipo. Analisar trajetória e uso de ferramentas inclui avaliar as etapas que um agente emprega para alcançar um objetivo, como seleção de ferramenta, estratégias e eficiência de tarefa. Por exemplo, um agente abordando uma consulta de produto de um cliente pode idealmente seguir uma trajetória envolvendo determinação de intenção, uso de ferramenta de busca de banco de dados, revisão de resultado e geração de relatório. As ações reais do agente são comparadas a esta trajetória esperada, ou verdade fundamental, para identificar erros e ineficiências. Métodos de comparação incluem correspondência exata (exigindo uma correspondência perfeita à sequência ideal), correspondência em ordem (ações corretas em ordem, permitindo etapas extras), correspondência em qualquer ordem (ações corretas em qualquer ordem, permitindo etapas extras), precisão (medindo a relevância de ações previstas), recall (medindo quantas ações essenciais são capturadas) e uso de ferramenta única (verificando uma ação específica). A seleção de métrica depende de requisitos específicos do agente, com cenários de alto risco potencialmente exigindo correspondência exata, enquanto situações mais flexíveis podem usar correspondência em ordem ou qualquer ordem.

Avaliação de agentes de IA envolve duas abordagens primárias: usar arquivos de teste e usar arquivos evalset. Arquivos de teste, em formato JSON, representam interações ou sessões simples de agente-modelo únicas e são ideais para teste unitário durante desenvolvimento ativo, focando em execução rápida e complexidade de sessão simples. Cada arquivo de teste contém uma única sessão com múltiplas voltas, onde uma volta é uma interação usuário-agente incluindo a consulta do usuário, trajetória esperada de uso de ferramenta, respostas intermediárias do agente e resposta final. Por exemplo, um arquivo de teste pode detalhar uma solicitação do usuário para "Desligar device_2 no Quarto," especificando o uso do agente de uma ferramenta set_device_info com parâmetros como localização: Quarto, device_id: device_2 e status: OFF, e uma resposta final esperada de "Defini o status do device_2 como off." Arquivos de teste podem ser organizados em pastas e podem incluir um arquivo test_config.json para definir critérios de avaliação. Arquivos evalset utilizam um dataset chamado "evalset" para avaliar interações, contendo múltiplas sessões potencialmente longas adequadas para simular conversas complexas de múltiplas voltas e testes de integração. Um arquivo evalset compreende múltiplos "evals," cada um representando uma sessão distinta com uma ou mais "voltas" que incluem consultas do usuário, uso esperado de ferramenta, respostas intermediárias e uma resposta final de referência. Um exemplo evalset pode incluir uma sessão onde o usuário primeiro pergunta "O que você pode fazer?" e então diz "Role um dado de 10 lados duas vezes e então verifique se 9 é primo ou não," definindo chamadas esperadas de ferramenta roll_die e uma chamada de ferramenta check_prime, junto com a resposta final resumindo as rolagens de dado e a verificação de primo.

**Multi-agentes**: Avaliar um sistema de IA complexo com múltiplos agentes é muito como avaliar um projeto de equipe. Porque há muitas etapas e transferências, sua complexidade é uma vantagem, permitindo verificar a qualidade do trabalho em cada estágio. Você pode examinar quão bem cada "agente" individual executa seu trabalho específico, mas também deve avaliar como todo o sistema está se desempenhando como um todo.

Para fazer isto, você faz perguntas-chave sobre a dinâmica da equipe, apoiadas por exemplos concretos:

* Os agentes estão cooperando efetivamente? Por exemplo, após um 'Agente de Reserva de Voo' garantir um voo, ele passa com sucesso as datas e destino corretos para o 'Agente de Reserva de Hotel'? Uma falha na cooperação pode levar a um hotel sendo reservado para a semana errada.  
* Eles criaram um bom plano e o seguiram? Imagine que o plano é primeiro reservar um voo, depois um hotel. Se o 'Agente de Hotel' tenta reservar um quarto antes do voo ser confirmado, ele se desviou do plano. Você também verifica se um agente fica preso, por exemplo, procurando infinitamente por um carro de aluguel "perfeito" e nunca passando para a próxima etapa.  
* O agente certo está sendo escolhido para a tarefa certa? Se um usuário pergunta sobre o clima para sua viagem, o sistema deve usar um 'Agente de Clima' especializado que fornece dados ao vivo. Se ao invés disso usar um 'Agente de Conhecimento Geral' que dá uma resposta genérica como "geralmente é quente no verão," ele escolheu a ferramenta errada para o trabalho.  
* Finalmente, adicionar mais agentes melhora o desempenho? Se você adicionar um novo 'Agente de Reserva de Restaurante' à equipe, isso torna o planejamento geral da viagem melhor e mais eficiente? Ou cria conflitos e desacelera o sistema, indicando um problema com escalabilidade?

# De Agentes a Contratantes Avançados

Recentemente, foi proposto (Agent Companion, gulli et al.) uma evolução de agentes de IA simples para "contratantes" avançados, movendo-se de sistemas probabilísticos, frequentemente não confiáveis para sistemas mais determinísticos e responsáveis projetados para ambientes complexos de alto risco (veja Fig.2). 

Os agentes de IA comuns de hoje operam em instruções breves e subespecificadas, o que os torna adequados para demonstrações simples mas frágeis em produção, onde ambiguidade leva à falha. O modelo "contratante" aborda isto estabelecendo uma relação rigorosa e formalizada entre o usuário e a IA, construída sobre uma base de termos claramente definidos e mutuamente acordados, muito como um acordo de serviço legal no mundo humano. Esta transformação é apoiada por quatro pilares-chave que coletivamente garantem clareza, confiabilidade e execução robusta de tarefas que estavam anteriormente além do escopo de sistemas autônomos.

Primeiro é o pilar do Contrato Formalizado, uma especificação detalhada que serve como a única fonte de verdade para uma tarefa. Vai muito além de um prompt simples. Por exemplo, um contrato para uma tarefa de análise financeira não diria apenas "analise as vendas do último trimestre"; exigiria "um relatório PDF de 20 páginas analisando vendas do mercado europeu do Q1 2025, incluindo cinco visualizações de dados específicas, uma análise comparativa contra Q1 2024, e uma avaliação de risco baseada no dataset incluído de interrupções da cadeia de suprimentos." Este contrato define explicitamente os entregáveis necessários, suas especificações precisas, as fontes de dados aceitáveis, o escopo do trabalho, e até o custo computacional esperado e tempo de conclusão, tornando o resultado objetivamente verificável.

Segundo é o pilar de um Ciclo de Vida Dinâmico de Negociação e Feedback. O contrato não é um comando estático mas o início de um diálogo. O agente contratante pode analisar os termos iniciais e negociar. Por exemplo, se um contrato exige o uso de uma fonte de dados proprietária específica que o agente não pode acessar, ele pode retornar feedback declarando, "O banco de dados XYZ especificado é inacessível. Por favor, forneça credenciais ou aprove o uso de um banco de dados público alternativo, que pode alterar ligeiramente a granularidade dos dados." Esta fase de negociação, que também permite ao agente sinalizar ambiguidades ou riscos potenciais, resolve mal-entendidos antes que a execução comece, prevenindo falhas custosas e garantindo que a saída final se alinhe perfeitamente com a intenção real do usuário.

![][image2]

Fig. 2: Exemplo de execução de contrato entre agentes

O terceiro pilar é Execução Iterativa Focada em Qualidade. Ao contrário de agentes projetados para respostas de baixa latência, um contratante prioriza correção e qualidade. Opera em um princípio de auto-validação e correção. Para um contrato de geração de código, por exemplo, o agente não apenas escreveria o código; geraria múltiplas abordagens algorítmicas, compilaria e executaria contra uma suíte de testes unitários definidos dentro do contrato, pontuaria cada solução em métricas como desempenho, segurança e legibilidade, e apenas submeteria a versão que passa todos os critérios de validação. Este loop interno de gerar, revisar e melhorar seu próprio trabalho até que as especificações do contrato sejam atendidas é crucial para construir confiança em suas saídas.

# Em Resumo

**O que:** Avaliar agentes inteligentes vai além de testes tradicionais para medir continuamente sua efetividade, eficiência e aderência a requisitos em ambientes do mundo real. Aplicações práticas de avaliação de agente incluem rastreamento de desempenho em sistemas vivos, teste A/B para melhorias, auditorias de conformidade e detectar deriva ou anomalias no comportamento. Avaliação básica de agente envolve avaliar precisão de resposta, enquanto cenários do mundo real demandam métricas mais sofisticadas como monitoramento de latência e rastreamento de uso de token para agentes alimentados por LLM. Trajetórias de agente, a sequência de etapas que um agente toma, são cruciais para avaliação, comparando ações reais contra um caminho ideal de verdade fundamental para identificar erros e ineficiências. O ADK fornece métodos estruturados de avaliação através de arquivos de teste individuais para teste unitário e arquivos evalset abrangentes para teste de integração, ambos definindo comportamento esperado do agente. Avaliações de agente podem ser executadas via uma UI baseada na web para teste interativo, programaticamente com pytest para integração CI/CD, ou através de uma interface de linha de comando para fluxos de trabalho automatizados. Para tornar IA confiável para tarefas complexas de alto risco, devemos mover de prompts simples para "contratos" formais que definem precisamente entregáveis verificáveis e escopo. Este acordo estruturado permite que os Agentes negociem, esclareçam ambiguidades e validem iterativamente seu próprio trabalho, transformando-o de uma ferramenta imprevisível em um sistema responsável e confiável.

**Por quê:** Avaliar efetivamente agentes de IA requer mover além de verificações simples de precisão para uma avaliação contínua e multi-facetada de seu desempenho em ambientes dinâmicos. Isto envolve monitoramento prático de métricas como latência e consumo de recursos, bem como análise sofisticada do processo de tomada de decisão de um agente através de sua trajetória. Para qualidades nuances como utilidade, métodos inovadores como o LLM-como-Juiz estão se tornando essenciais, enquanto frameworks como o ADK do Google fornecem ferramentas estruturadas para teste unitário e de integração. O desafio se intensifica com sistemas multi-agente, onde o foco muda para avaliar sucesso colaborativo e cooperação efetiva.

**Regra de ouro:** Use avaliação e monitoramento contínuos para qualquer agente de IA implantado em ambientes de produção onde desempenho, confiabilidade e conformidade são críticos. Implemente métricas de latência e uso de recursos para agentes alimentados por LLM, utilize LLM-como-Juiz para avaliações qualitativas, e considere contratos formais para tarefas de alto risco que requerem verificabilidade e responsabilidade.

**Resumo visual**

Fig. 1: Padrão de design de avaliação e monitoramento


**Métricas e Monitoramento**

# Principais Conclusões

* Avaliar agentes inteligentes vai além de testes tradicionais para medir continuamente sua efetividade, eficiência e aderência a requisitos em ambientes do mundo real.  
* Aplicações práticas de avaliação de agente incluem rastreamento de desempenho em sistemas vivos, teste A/B para melhorias, auditorias de conformidade e detectar deriva ou anomalias no comportamento.  
* Avaliação básica de agente envolve avaliar precisão de resposta, enquanto cenários do mundo real demandam métricas mais sofisticadas como monitoramento de latência e rastreamento de uso de token para agentes alimentados por LLM.  
* Trajetórias de agente, a sequência de etapas que um agente toma, são cruciais para avaliação, comparando ações reais contra um caminho ideal de verdade fundamental para identificar erros e ineficiências.  
* O ADK fornece métodos estruturados de avaliação através de arquivos de teste individuais para teste unitário e arquivos evalset abrangentes para teste de integração, ambos definindo comportamento esperado do agente.  
* Avaliações de agente podem ser executadas via uma UI baseada na web para teste interativo, programaticamente com pytest para integração CI/CD, ou através de uma interface de linha de comando para fluxos de trabalho automatizados.  
* Para tornar IA confiável para tarefas complexas de alto risco, devemos mover de prompts simples para "contratos" formais que definem precisamente entregáveis verificáveis e escopo. Este acordo estruturado permite que os Agentes negociem, esclareçam ambiguidades e validem iterativamente seu próprio trabalho, transformando-os de ferramentas imprevisíveis em sistemas responsáveis e confiáveis.

# Conclusões

Em conclusão, avaliar efetivamente agentes de IA requer mover além de verificações simples de precisão para uma avaliação contínua e multi-facetada de seu desempenho em ambientes dinâmicos. Isto envolve monitoramento prático de métricas como latência e consumo de recursos, bem como análise sofisticada do processo de tomada de decisão de um agente através de sua trajetória. Para qualidades nuances como utilidade, métodos inovadores como o LLM-como-Juiz estão se tornando essenciais, enquanto frameworks como o ADK do Google fornecem ferramentas estruturadas para teste unitário e de integração. O desafio se intensifica com sistemas multi-agente, onde o foco muda para avaliar sucesso colaborativo e cooperação efetiva.

Para garantir confiabilidade em aplicações críticas, o paradigma está mudando de agentes simples dirigidos por prompt para "contratantes" avançados vinculados por acordos formais. Estes agentes contratantes operam em termos explícitos e verificáveis, permitindo-lhes negociar, decompor tarefas e auto-validar seu trabalho para atender padrões rigorosos de qualidade. Esta abordagem estruturada transforma agentes de ferramentas imprevisíveis em sistemas responsáveis capazes de lidar com tarefas complexas de alto risco. Em última análise, esta evolução é crucial para construir a confiança necessária para implantar IA agêntica sofisticada em domínios críticos para a missão.

# Referências

Pesquisa relevante inclui:

1. ADK Web: [https://github.com/google/adk-web](https://github.com/google/adk-web)   
2. ADK Evaluate: [https://google.github.io/adk-docs/evaluate/](https://google.github.io/adk-docs/evaluate/)  
3. Survey on Evaluation of LLM-based Agents, [https://arxiv.org/abs/2503.16416](https://arxiv.org/abs/2503.16416)   
4. Agent-as-a-Judge: Evaluate Agents with Agents, [https://arxiv.org/abs/2410.10934](https://arxiv.org/abs/2410.10934)   
5. Agent Companion, gulli et al: [https://www.kaggle.com/whitepaper-agent-companion](https://www.kaggle.com/whitepaper-agent-companion)

[image1]: ../assets/24-chapter-19-image-1-line-157.png

[image2]: ../assets/24-chapter-19-image-2-line-159.png
