# Capítulo 21: Exploração e Descoberta

Este capítulo explora padrões que permitem que agentes inteligentes busquem ativamente informações novas, descubram novas possibilidades e identifiquem desconhecidos desconhecidos dentro de seu ambiente operacional. Exploração e descoberta diferem de comportamentos reativos ou otimização dentro de um espaço de solução pré-definido. Ao invés disso, focam em agentes proativamente aventurando-se em territórios não familiares, experimentando com novas abordagens e gerando novo conhecimento ou compreensão. Este padrão é crucial para agentes operando em domínios abertos, complexos ou em rápida evolução onde conhecimento estático ou soluções pré-programadas são insuficientes. Enfatiza a capacidade do agente de expandir sua compreensão e capacidades.

# Aplicações Práticas e Casos de Uso

Agentes de IA possuem a capacidade de priorizar e explorar inteligentemente, o que leva a aplicações através de vários domínios. Ao avaliar e ordenar autonomamente ações potenciais, estes agentes podem navegar ambientes complexos, descobrir insights ocultos e impulsionar inovação. Esta capacidade de exploração priorizada permite que otimizem processos, descubram novo conhecimento e gerem conteúdo.

Exemplos:

* **Automação de Pesquisa Científica:** Um agente projeta e executa experimentos, analisa resultados e formula novas hipóteses para descobrir materiais novos, candidatos a drogas ou princípios científicos.  
* **Jogos e Geração de Estratégia:** Agentes exploram estados de jogo, descobrindo estratégias emergentes ou identificando vulnerabilidades em ambientes de jogo (ex., AlphaGo).  
* **Pesquisa de Mercado e Detecção de Tendências:** Agentes escaneiam dados não estruturados (mídia social, notícias, relatórios) para identificar tendências, comportamentos de consumidor ou oportunidades de mercado.  
* **Descoberta de Vulnerabilidades de Segurança:** Agentes sondam sistemas ou bases de código para encontrar falhas de segurança ou vetores de ataque.  
* **Geração de Conteúdo Criativo:** Agentes exploram combinações de estilos, temas ou dados para gerar peças artísticas, composições musicais ou obras literárias.  
* **Educação e Treinamento Personalizado:** Tutores de IA priorizam caminhos de aprendizado e entrega de conteúdo baseado no progresso, estilo de aprendizado e áreas que precisam de melhoria de um estudante.

Google Co-Scientist

Um co-cientista de IA é um sistema de IA desenvolvido pelo Google Research projetado como um colaborador científico computacional. Ele assiste cientistas humanos em aspectos de pesquisa como geração de hipóteses, refinamento de propostas e design experimental. Este sistema opera no LLM Gemini.

O desenvolvimento do co-cientista de IA aborda desafios na pesquisa científica. Estes incluem processar grandes volumes de informação, gerar hipóteses testáveis e gerenciar planejamento experimental. O co-cientista de IA apoia pesquisadores executando tarefas que envolvem processamento e síntese de informação em larga escala, potencialmente revelando relacionamentos dentro dos dados. Seu propósito é aumentar processos cognitivos humanos lidando com aspectos computacionalmente exigentes de pesquisa em estágio inicial.

**Arquitetura de Sistema e Metodologia:** A arquitetura do co-cientista de IA é baseada em um framework multi-agente, estruturado para emular processos colaborativos e iterativos. Este design integra agentes de IA especializados, cada um com um papel específico em contribuir para um objetivo de pesquisa. Um agente supervisor gerencia e coordena as atividades destes agentes individuais dentro de um framework de execução de tarefa assíncrona que permite escalonamento flexível de recursos computacionais.

Os agentes centrais e suas funções incluem (veja Fig. 1):

* **Agente de geração**: Inicia o processo produzindo hipóteses iniciais através de exploração de literatura e debates científicos simulados.  
* **Agente de reflexão**: Atua como revisor par, avaliando criticamente a correção, novidade e qualidade das hipóteses geradas.  
* **Agente de classificação**: Empregua um torneio baseado em Elo para comparar, classificar e priorizar hipóteses através de debates científicos simulados.  
* **Agente de evolução**: Refina continuamente hipóteses de alta classificação simplificando conceitos, sintetizando ideias e explorando raciocínio não convencional.  
* **Agente de proximidade**: Computa um grafo de proximidade para agrupar ideias similares e assistir na exploração da paisagem de hipóteses.  
* **Agente de meta-revisão**: Sintetiza insights de todas as revisões e debates para identificar padrões comuns e fornecer feedback, permitindo que o sistema melhore continuamente.

A fundação operacional do sistema depende do Gemini, que fornece compreensão de linguagem, raciocínio e habilidades generativas. O sistema incorpora "escalonamento de computação em tempo de teste," um mecanismo que aloca recursos computacionais aumentados para raciocinar iterativamente e aprimorar saídas. O sistema processa e sintetiza informação de fontes diversas, incluindo literatura acadêmica, dados baseados na web e bancos de dados.

![][image1]Fig. 1: (Cortesia dos Autores) Co-Cientista de IA: Ideação para Validação

O sistema segue uma abordagem iterativa "gerar, debater e evoluir" espelhando o método científico. Seguindo a entrada de um problema científico de um cientista humano, o sistema se engaja em um ciclo auto-melhorador de geração de hipóteses, avaliação e refinamento. Hipóteses passam por avaliação sistemática, incluindo avaliações internas entre agentes e um mecanismo de classificação baseado em torneio.

**Validação e Resultados:** A utilidade do co-cientista de IA foi demonstrada em vários estudos de validação, particularmente em biomedicina, avaliando seu desempenho através de benchmarks automatizados, revisões de especialistas e experimentos end-to-end de laboratório úmido.

**Avaliação Automatizada e de Especialista:** No benchmark desafiador GPQA, a classificação interna Elo do sistema foi mostrada ser concordante com a precisão de seus resultados, alcançando uma precisão top-1 de 78,4% no conjunto "diamante" difícil. Análise através de mais de 200 objetivos de pesquisa demonstrou que escalonar computação em tempo de teste melhora consistentemente a qualidade das hipóteses, conforme medido pela classificação Elo. Em um conjunto curado de 15 problemas desafiadores, o co-cientista de IA superou outros modelos de IA state-of-the-art e as soluções de "melhor palpite" fornecidas por especialistas humanos. Em uma avaliação de pequena escala, especialistas biomédicos classificaram as saídas do co-cientista como mais novas e impactantes comparadas a outros modelos baseline. As propostas do sistema para reutilização de drogas, formatadas como páginas Specific Aims do NIH, também foram julgadas de alta qualidade por um painel de seis oncologistas especialistas.

**Validação Experimental End-to-End:**

Reutilização de Droga: Para leucemia mieloide aguda (AML), o sistema propôs candidatos a drogas novos. Alguns destes, como KIRA6, foram sugestões completamente novas sem evidência pré-clínica prévia para uso em AML. Experimentos in vitro subsequentes confirmaram que KIRA6 e outras drogas sugeridas inibiram viabilidade de células tumorais em concentrações clinicamente relevantes em múltiplas linhas de células AML.

Descoberta de Alvo Novo: O sistema identificou alvos epigenéticos novos para fibrose hepática. Experimentos de laboratório usando organoides hepáticos humanos validaram estes achados, mostrando que drogas direcionando os modificadores epigenéticos sugeridos tiveram atividade anti-fibrótica significativa. Uma das drogas identificadas já é aprovada pela FDA para outra condição, abrindo uma oportunidade para reutilização.

Resistência Antimicrobiana: O co-cientista de IA recapitulou independentemente achados experimentais não publicados. Foi incumbido de explicar por que certos elementos genéticos móveis (cf-PICIs) são encontrados através de muitas espécies bacterianas. Em dois dias, a hipótese de maior classificação do sistema foi que cf-PICIs interagem com caudas de fago diversas para expandir seu alcance de hospedeiro. Isto espelhou a descoberta nova, experimentalmente validada que um grupo de pesquisa independente havia alcançado após mais de uma década de pesquisa.

**Aumento e Limitações:** A filosofia de design por trás do co-cientista de IA enfatiza aumento ao invés de automação completa da pesquisa humana. Pesquisadores interagem com e guiam o sistema através de linguagem natural, fornecendo feedback, contribuindo com suas próprias ideias e direcionando os processos exploratórios da IA em um paradigma colaborativo "cientista-no-loop". No entanto, o sistema tem algumas limitações. Seu conhecimento é restringido por sua dependência de literatura de acesso aberto, potencialmente perdendo trabalho prévio crítico atrás de paywalls. Também tem acesso limitado a resultados experimentais negativos, que raramente são publicados mas cruciais para cientistas experientes. Além disso, o sistema herda limitações dos LLMs subjacentes, incluindo potencial para imprecisões factuais ou "alucinações".

**Segurança:** Segurança é uma consideração crítica, e o sistema incorpora múltiplas salvaguardas. Todos os objetivos de pesquisa são revisados para segurança na entrada, e hipóteses geradas também são verificadas para prevenir que o sistema seja usado para pesquisa insegura ou antiética. Uma avaliação preliminar de segurança usando 1.200 objetivos de pesquisa adversarial encontrou que o sistema poderia rejeitar robustamente entradas perigosas. Para garantir desenvolvimento responsável, o sistema está sendo disponibilizado para mais cientistas através de um Programa de Testador Confiável para reunir feedback do mundo real.

# Exemplo de Código Hands-On

Vamos olhar um exemplo concreto de IA agêntica para Exploração e Descoberta em ação: Agent Laboratory, um projeto desenvolvido por Samuel Schmidgall sob a Licença MIT.

"Agent Laboratory" é um framework de fluxo de trabalho de pesquisa autônomo projetado para aumentar empreendimentos científicos humanos ao invés de substituí-los. Este sistema aproveita LLMs especializados para automatizar várias etapas do processo de pesquisa científica, permitindo assim que pesquisadores humanos dediquem mais recursos cognitivos à conceituação e análise crítica.

O framework integra "AgentRxiv," um repositório descentralizado para agentes de pesquisa autônomos. AgentRxiv facilita a deposição, recuperação e desenvolvimento de saídas de pesquisa.

Agent Laboratory guia o processo de pesquisa através de fases distintas:

1. **Revisão de Literatura:** Durante esta fase inicial, agentes especializados dirigidos por LLM são incumbidos da coleta autônoma e análise crítica de literatura acadêmica pertinente. Isto envolve aproveitar bancos de dados externos como arXiv para identificar, sintetizar e categorizar pesquisa relevante, estabelecendo efetivamente uma base de conhecimento abrangente para as etapas subsequentes.  
2. **Experimentação:** Esta fase abrange a formulação colaborativa de designs experimentais, preparação de dados, execução de experimentos e análise de resultados. Agentes utilizam ferramentas integradas como Python para geração e execução de código, e Hugging Face para acesso a modelos, para conduzir experimentação automatizada. O sistema é projetado para refinamento iterativo, onde agentes podem adaptar e otimizar procedimentos experimentais baseados em resultados em tempo real.  
3. **Escrita de Relatório:** Na fase final, o sistema automatiza a geração de relatórios de pesquisa abrangentes. Isto envolve sintetizar achados da fase de experimentação com insights da revisão de literatura, estruturar o documento de acordo com convenções acadêmicas e integrar ferramentas externas como LaTeX para formatação profissional e geração de figuras.  
4. **Compartilhamento de Conhecimento**: AgentRxiv é uma plataforma permitindo que agentes de pesquisa autônomos compartilhem, acessem e avancem colaborativamente descobertas científicas. Permite que agentes construam sobre achados prévios, fomentando progresso de pesquisa cumulativo.

A arquitetura modular do Agent Laboratory garante flexibilidade computacional. O objetivo é aprimorar produtividade de pesquisa automatizando tarefas enquanto mantém o pesquisador humano.

**Análise de código:** Embora uma análise abrangente de código esteja além do escopo deste livro, quero fornecer-lhe alguns insights-chave e encorajá-lo a se aprofundar no código por conta própria.

**Julgamento:** Para emular processos avaliativos humanos, o sistema emprega um mecanismo de julgamento agêntico tripartite para avaliar saídas. Isto envolve o deploy de três agentes autônomos distintos, cada um configurado para avaliar a produção de uma perspectiva específica, imitando assim coletivamente a natureza nuances e multi-facetada do julgamento humano. Esta abordagem permite uma avaliação mais robusta e abrangente, movendo além de métricas singulares para capturar uma avaliação qualitativa mais rica.

Os agentes de julgamento são projetados com um prompt específico que emula de perto o framework cognitivo e critérios de avaliação tipicamente empregados por revisores humanos. Este prompt guia os agentes a analisar saídas através de uma lente similar a como um especialista humano faria, considerando fatores como relevância, coerência, precisão factual e qualidade geral. Ao criar estes prompts para espelhar protocolos de revisão humana, o sistema visa alcançar um nível de sofisticação avaliativa que se aproxima do discernimento similar ao humano.

Neste sistema multi-agente, o processo de pesquisa é estruturado em torno de papéis especializados, espelhando uma hierarquia acadêmica típica para otimizar fluxo de trabalho e otimizar saída.

**Agente Professor:** O Agente Professor funciona como o diretor de pesquisa primário, responsável por estabelecer a agenda de pesquisa, definir questões de pesquisa e delegar tarefas a outros agentes. Este agente define a direção estratégica e garante alinhamento com objetivos do projeto.

**Agente PostDoc:** O papel do Agente PostDoc é executar a pesquisa. Isto inclui conduzir revisões de literatura, projetar e implementar experimentos, e gerar saídas de pesquisa como artigos. Importante, o Agente PostDoc tem a capacidade de escrever e executar código, permitindo a implementação prática de protocolos experimentais e análise de dados. Este agente é o produtor primário de artefatos de pesquisa.

**Agentes Revisores:** Agentes revisores executam avaliações críticas de saídas de pesquisa do Agente PostDoc, avaliando a qualidade, validade e rigor científico de artigos e resultados experimentais. Esta fase de avaliação emula o processo de revisão por pares em ambientes acadêmicos para garantir um alto padrão de saída de pesquisa antes da finalização.

**Agentes de Engenharia ML**: Os Agentes de Engenharia de Machine Learning servem como engenheiros de machine learning, engajando-se em colaboração dialógica com um estudante de PhD para desenvolver código. Sua função central é gerar código descomplicado para pré-processamento de dados, integrando insights derivados da revisão de literatura fornecida e protocolo experimental. Isto garante que os dados sejam apropriadamente formatados e preparados para o experimento designado.

# Em Resumo

**O que:** Agentes de IA operando em ambientes complexos enfrentam uma multiplicidade de ações potenciais, objetivos conflitantes e recursos finitos. Sem um método claro para determinar seu próximo movimento, estes agentes correm o risco de se tornar ineficientes e inefetivos. Isto pode levar a atrasos operacionais significativos ou uma falha completa em realizar objetivos primários. O desafio central é gerenciar este número esmagador de escolhas para garantir que o agente aja propositalmente e logicamente.

**Por quê:** O padrão de Exploração e Descoberta fornece uma solução padronizada para este problema permitindo que agentes busquem ativamente informações novas e possibilidades. Isto é alcançado através de sistemas multi-agente que podem gerar hipóteses, projetar experimentos e descobrir conhecimento novo de forma autônoma. Esta capacidade Agêntica permite que o sistema se aventure proativamente em territórios não familiares, experimente com novas abordagens e gere novo conhecimento ou compreensão. Ao fazer isto, o agente demonstra comportamento verdadeiramente agêntico, movendo-se além de seguir instruções passivamente para explorar proativamente seu ambiente e expandir suas capacidades.

**Regra de ouro:** Use o padrão de Exploração e Descoberta quando um sistema Agêntico deve operar em domínios abertos, complexos ou em rápida evolução onde conhecimento estático ou soluções pré-programadas são insuficientes, e quando o objetivo é descobrir novo conhecimento, gerar hipóteses ou explorar possibilidades não mapeadas.

**Resumo visual**

![][image1]

Fig. 1: Padrão de design de exploração e descoberta

# Principais Conclusões

* Exploração e Descoberta em IA permitem que agentes busquem ativamente novas informações e possibilidades, que é essencial para navegar ambientes complexos e em evolução.  
* Sistemas como o Co-Cientista do Google demonstram como Agentes podem gerar hipóteses e projetar experimentos autonomamente, suplementando pesquisa científica humana.  
* O framework multi-agente, exemplificado pelos papéis especializados do Agent Laboratory, melhora pesquisa através da automação de revisão de literatura, experimentação e escrita de relatório.  
* Em última análise, estes Agentes visam aprimorar criatividade e resolução de problemas humanos gerenciando tarefas computacionalmente intensivas, acelerando assim inovação e descoberta.

# Conclusão

Em conclusão, o padrão de Exploração e Descoberta é a própria essência de um sistema verdadeiramente agêntico, definindo sua capacidade de mover-se além de seguir instruções passivamente para explorar proativamente seu ambiente. Este impulso agêntico inato é o que capacita uma IA a operar autonomamente em domínios complexos, não meramente executando tarefas mas independentemente definindo sub-objetivos para descobrir informações novas. Este comportamento agêntico avançado é mais poderosamente realizado através de frameworks multi-agente onde cada agente incorpora um papel específico e proativo em um processo colaborativo maior. Por exemplo, o sistema altamente agêntico do Co-cientista do Google apresenta agentes que geram, debatem e evoluem hipóteses científicas autonomamente.

Frameworks como Agent Laboratory estruturam isto ainda mais criando uma hierarquia agêntica que imita equipes de pesquisa humana, permitindo que o sistema auto-gerencie todo o ciclo de vida de descoberta. O núcleo deste padrão reside em orquestrar comportamentos agênticos emergentes, permitindo que o sistema persiga objetivos de longo prazo e abertos com intervenção humana mínima. Isto eleva a parceria humano-IA, posicionando a IA como um colaborador agêntico genuíno que lida com a execução autônoma de tarefas exploratórias. Ao delegar este trabalho proativo de descoberta a um sistema agêntico, o intelecto humano é significativamente aumentado, acelerando inovação. O desenvolvimento de tais capacidades agênticas poderosas também necessita um forte compromisso com segurança e supervisão ética. Em última análise, este padrão fornece o projeto para criar IA verdadeiramente agêntica, transformando ferramentas computacionais em parceiros independentes e orientados por objetivos na busca do conhecimento.

# Referências

1. Exploration-Exploitation Dilemma**:** Um problema fundamental em aprendizado por reforço e tomada de decisão sob incerteza. [https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)   
2. Google Co-Scientist: [https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)   
3. Agent Laboratory: Using LLM Agents as Research Assistants [https://github.com/SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)   
4. AgentRxiv: Towards Collaborative Autonomous Research: [https://agentrxiv.github.io/](https://agentrxiv.github.io/)

[image1]: ../assets/26-chapter-21-image-1-line-146.png
