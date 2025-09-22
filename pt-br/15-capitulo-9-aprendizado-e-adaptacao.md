# Capítulo 9: Aprendizado e Adaptação

O aprendizado e a adaptação são fundamentais para aprimorar as capacidades dos agentes de inteligência artificial. Estes processos permitem que agentes evoluam além de parâmetros predefinidos, permitindo que melhorem autonomamente através de experiência e interação ambiental. Ao aprender e se adaptar, agentes podem gerenciar efetivamente situações novas e otimizar sua performance sem intervenção manual constante. Este capítulo explora os princípios e mecanismos que sustentam o aprendizado e adaptação de agentes em detalhes.

# O panorama geral

Agentes aprendem e se adaptam mudando seu pensamento, ações ou conhecimento baseado em novas experiências e dados. Isso permite que agentes evoluam de simplesmente seguir instruções para se tornarem mais inteligentes ao longo do tempo.

* **Aprendizado por Reforço:** Agentes tentam ações e recebem recompensas por resultados positivos e penalidades por negativos, aprendendo comportamentos ótimos em situações em mudança. Útil para agentes controlando robôs ou jogando jogos.  
* **Aprendizado Supervisionado:** Agentes aprendem de exemplos rotulados, conectando entradas a saídas desejadas, permitindo tarefas como tomada de decisão e reconhecimento de padrões. Ideal para agentes classificando emails ou prevendo tendências.  
* **Aprendizado Não Supervisionado:** Agentes descobrem conexões e padrões ocultos em dados não rotulados, auxiliando em insights, organização e criando um mapa mental de seu ambiente. Útil para agentes explorando dados sem orientação específica.  
* **Aprendizado Few-Shot/Zero-Shot com Agentes Baseados em LLM:** Agentes aproveitando LLMs podem rapidamente se adaptar a novas tarefas com exemplos mínimos ou instruções claras, permitindo respostas rápidas a novos comandos ou situações.  
* **Aprendizado Online:** Agentes continuamente atualizam conhecimento com novos dados, essencial para reações em tempo real e adaptação contínua em ambientes dinâmicos. Crítico para agentes processando fluxos de dados contínuos.  
* **Aprendizado Baseado em Memória:** Agentes recordam experiências passadas para ajustar ações atuais em situações similares, aprimorando consciência contextual e tomada de decisão. Eficaz para agentes com capacidades de recordação de memória.

Agentes se adaptam mudando estratégia, compreensão ou objetivos baseado no aprendizado. Isso é vital para agentes em ambientes imprevisíveis, em mudança ou novos.

**Otimização de Política Proximal (PPO)** é um algoritmo de aprendizado por reforço usado para treinar agentes em ambientes com uma gama contínua de ações, como controlar as articulações de um robô ou um personagem em um jogo. Seu objetivo principal é melhorar de forma confiável e estável a estratégia de tomada de decisão de um agente, conhecida como sua política.

A ideia central por trás do PPO é fazer atualizações pequenas e cuidadosas na política do agente. Ele evita mudanças drásticas que poderiam causar colapso de performance. Aqui está como funciona:

1. Coletar Dados: O agente interage com seu ambiente (ex., joga um jogo) usando sua política atual e coleta um lote de experiências (estado, ação, recompensa).  
2. Avaliar um Objetivo "Substituto": O PPO calcula como uma atualização de política potencial mudaria a recompensa esperada. No entanto, ao invés de apenas maximizar esta recompensa, ele usa uma função objetivo especial "recortada".  
3. O Mecanismo de "Recorte": Esta é a chave para a estabilidade do PPO. Ele cria uma "região de confiança" ou uma zona segura ao redor da política atual. O algoritmo é impedido de fazer uma atualização que seja muito diferente da estratégia atual. Este recorte age como um freio de segurança, garantindo que o agente não dê um passo enorme e arriscado que desfaça seu aprendizado.

Em resumo, o PPO equilibra melhorar performance com permanecer próximo a uma estratégia conhecida e funcionando, o que previne falhas catastróficas durante o treinamento e leva a aprendizado mais estável.

**Otimização Direta de Preferências (DPO)** é um método mais recente projetado especificamente para alinhar Modelos de Linguagem Grandes (LLMs) com preferências humanas. Ele oferece uma alternativa mais simples e direta ao usar PPO para esta tarefa.

Para entender o DPO, ajuda primeiro entender o método tradicional de alinhamento baseado em PPO:

* A Abordagem PPO (Processo de Dois Passos):  
  1. Treinar um Modelo de Recompensa: Primeiro, você coleta dados de feedback humano onde pessoas avaliam ou comparam diferentes respostas de LLM (ex., "Resposta A é melhor que Resposta B"). Estes dados são usados para treinar um modelo de IA separado, chamado modelo de recompensa, cujo trabalho é prever qual pontuação um humano daria a qualquer nova resposta.  
  2. Fine-Tuning com PPO: Em seguida, o LLM é fine-tuned usando PPO. O objetivo do LLM é gerar respostas que obtenham a pontuação mais alta possível do modelo de recompensa. O modelo de recompensa atua como o "juiz" no jogo de treinamento.

Este processo de dois passos pode ser complexo e instável. Por exemplo, o LLM pode encontrar uma brecha e aprender a "hackear" o modelo de recompensa para obter pontuações altas para respostas ruins.

* A Abordagem DPO (Processo Direto): O DPO pula o modelo de recompensa inteiramente. Ao invés de traduzir preferências humanas em uma pontuação de recompensa e então otimizar para essa pontuação, o DPO usa os dados de preferência diretamente para atualizar a política do LLM.  
* Funciona usando uma relação matemática que liga diretamente dados de preferência à política ótima. Ele essencialmente ensina ao modelo: "Aumente a probabilidade de gerar respostas como a *preferida* e diminua a probabilidade de gerar respostas como a *desfavorecida*."

Em essência, o DPO simplifica o alinhamento otimizando diretamente o modelo de linguagem em dados de preferência humana. Isso evita a complexidade e potencial instabilidade de treinar e usar um modelo de recompensa separado, tornando o processo de alinhamento mais eficiente e robusto.

# Aplicações Práticas e Casos de Uso

Agentes adaptativos exibem performance aprimorada em ambientes variáveis através de atualizações iterativas impulsionadas por dados experienciais.

* **Agentes assistentes personalizados** refinam protocolos de interação através de análise longitudinal de comportamentos individuais do usuário, garantindo geração de resposta altamente otimizada.  
* **Agentes bots de trading** otimizam algoritmos de tomada de decisão ajustando dinamicamente parâmetros do modelo baseado em dados de mercado em tempo real de alta resolução, maximizando assim retornos financeiros e mitigando fatores de risco.  
* **Agentes de aplicação** otimizam interface do usuário e funcionalidade através de modificação dinâmica baseada em comportamento observado do usuário, resultando em maior engajamento do usuário e intuitividade do sistema.  
* **Agentes robóticos e de veículos autônomos** aprimoram capacidades de navegação e resposta integrando dados de sensores e análise de ações históricas, permitindo operação segura e eficiente através de diversas condições ambientais.  
* **Agentes de detecção de fraude** melhoram detecção de anomalias refinando modelos preditivos com padrões fraudulentos recém-identificados, aprimorando segurança do sistema e minimizando perdas financeiras.  
* **Agentes de recomendação** melhoram precisão de seleção de conteúdo empregando algoritmos de aprendizado de preferência do usuário, fornecendo recomendações altamente individualizadas e contextualmente relevantes.  
* **Agentes de IA de jogos** aprimoram engajamento do jogador adaptando dinamicamente algoritmos estratégicos, aumentando assim complexidade e desafio do jogo.  
* **Agentes de Aprendizado de Base de Conhecimento**: Agentes podem aproveitar Geração Aumentada por Recuperação (RAG) para manter uma base de conhecimento dinâmica de descrições de problemas e soluções comprovadas (consulte o Capítulo 14). Ao armazenar estratégias bem-sucedidas e desafios encontrados, o agente pode referenciar estes dados durante tomada de decisão, permitindo que se adapte a novas situações mais efetivamente aplicando padrões previamente bem-sucedidos ou evitando armadilhas conhecidas.

# Estudo de Caso: O Agente de Codificação Auto-Melhorador (SICA)

O Agente de Codificação Auto-Melhorador (SICA), desenvolvido por Maxime Robeyns, Laurence Aitchison, e Martin Szummer, representa um avanço no aprendizado baseado em agentes, demonstrando a capacidade de um agente modificar seu próprio código fonte. Isso contrasta com abordagens tradicionais onde um agente pode treinar outro; o SICA atua tanto como o modificador quanto a entidade modificada, refinando iterativamente sua base de código para melhorar performance através de vários desafios de codificação.

A auto-melhoria do SICA opera através de um ciclo iterativo (consulte Fig.1). Inicialmente, o SICA revisa um arquivo de suas versões passadas e sua performance em testes de benchmark. Ele seleciona a versão com a pontuação de performance mais alta, calculada baseada em uma fórmula ponderada considerando sucesso, tempo e custo computacional. Esta versão selecionada então empreende a próxima rodada de auto-modificação. Ela analisa o arquivo para identificar melhorias potenciais e então altera diretamente sua base de código. O agente modificado é subsequentemente testado contra benchmarks, com os resultados registrados no arquivo. Este processo se repete, facilitando aprendizado diretamente de performance passada. Este mecanismo de auto-melhoria permite que o SICA evolua suas capacidades sem requerer paradigmas de treinamento tradicionais.

![][image1]

Fig.1: Auto-melhoria do SICA, aprendendo e se adaptando baseado em suas versões passadas

O SICA passou por significativa auto-melhoria, levando a avanços em edição e navegação de código. Inicialmente, o SICA utilizava uma abordagem básica de sobrescrita de arquivo para mudanças de código. Ele subsequentemente desenvolveu um "Editor Inteligente" capaz de edições mais inteligentes e contextuais. Isso evoluiu para um "Editor Inteligente Aprimorado com Diff," incorporando diffs para modificações direcionadas e edição baseada em padrões, e uma "Ferramenta de Sobrescrita Rápida" para reduzir demandas de processamento. 

O SICA implementou ainda "Otimização de Saída de Diff Mínimo" e "Minimização de Diff Sensível ao Contexto," usando parsing de Árvore de Sintaxe Abstrata (AST) para eficiência. Adicionalmente, um "Normalizador de Entrada SmartEditor" foi adicionado. Em termos de navegação, o SICA criou independentemente um "Localizador de Símbolo AST," usando o mapa estrutural do código (AST) para identificar definições dentro da base de código. Posteriormente, um "Localizador de Símbolo Híbrido" foi desenvolvido, combinando uma busca rápida com verificação AST. Isso foi ainda mais otimizado via "Parsing AST Otimizado no Localizador de Símbolo Híbrido" para focar em seções relevantes de código, melhorando velocidade de busca.(consulte Fig. 2)

![][image2]

Fig.2: Performance através de iterações. Melhorias principais são anotadas com suas modificações correspondentes de ferramenta ou agente. (cortesia de Maxime Robeyns, Martin Szummer, Laurence Aitchison)

A arquitetura do SICA compreende um kit de ferramentas fundamental para operações básicas de arquivo, execução de comando e cálculos aritméticos. Inclui mecanismos para submissão de resultados e invocação de sub-agentes especializados (codificação, resolução de problemas e raciocínio). Estes sub-agentes decompõem tarefas complexas e gerenciam o comprimento de contexto do LLM, especialmente durante ciclos de melhoria estendidos. 

Um supervisor assíncrono, outro LLM, monitora o comportamento do SICA, identificando problemas potenciais como loops ou estagnação. Ele se comunica com o SICA e pode intervir para parar execução se necessário. O supervisor recebe um relatório detalhado das ações do SICA, incluindo um grafo de chamadas e um log de mensagens e ações de ferramenta, para identificar padrões e ineficiências.

O LLM do SICA organiza informações dentro de sua janela de contexto, sua memória de curto prazo, de uma maneira estruturada crucial para sua operação. Esta estrutura inclui um Prompt do Sistema definindo objetivos do agente, documentação de ferramentas e sub-agentes, e instruções do sistema. Um Prompt Principal contém a declaração do problema ou instrução, conteúdo de arquivos abertos, e um mapa de diretório. Mensagens do Assistente registram o raciocínio passo-a-passo do agente, registros e resultados de chamadas de ferramentas e sub-agentes, e comunicações do supervisor. Esta organização facilita fluxo eficiente de informação, aprimorando operação do LLM e reduzindo tempo e custos de processamento. Inicialmente, mudanças de arquivo eram registradas como diffs, mostrando apenas modificações e periodicamente consolidadas.

**SICA: Uma Olhada no Código:** Mergulhar mais profundamente na implementação do SICA revela várias escolhas de design chave que sustentam suas capacidades. Como discutido, o sistema é construído com uma arquitetura modular, incorporando vários sub-agentes, como um agente de codificação, um agente resolvedor de problemas, e um agente de raciocínio. Estes sub-agentes são invocados pelo agente principal, muito como chamadas de ferramenta, servindo para decompor tarefas complexas e gerenciar eficientemente comprimento de contexto, especialmente durante aquelas iterações de meta-melhoria estendidas. 

O projeto é ativamente desenvolvido e visa fornecer um framework robusto para aqueles interessados em pós-treinar LLMs em uso de ferramentas e outras tarefas agênticas, com o código completo disponível para exploração e contribuição adicional no repositório GitHub [https://github.com/MaximeRobeyns/self_improving_coding_agent/](https://github.com/MaximeRobeyns/self_improving_coding_agent/).

Para segurança, o projeto enfatiza fortemente containerização Docker, significando que o agente executa dentro de um container Docker dedicado. Esta é uma medida crucial, pois fornece isolamento da máquina host, mitigando riscos como manipulação inadvertida do sistema de arquivos dada a capacidade do agente de executar comandos shell. 

Para garantir transparência e controle, o sistema apresenta observabilidade robusta através de uma página web interativa que visualiza eventos no barramento de eventos e o grafo de chamadas do agente. Isso oferece insights abrangentes nas ações do agente, permitindo que usuários inspecionem eventos individuais, leiam mensagens do supervisor, e colapsem traços de sub-agentes para compreensão mais clara. 

Em termos de sua inteligência principal, o framework do agente suporta integração LLM de vários provedores, permitindo experimentação com diferentes modelos para encontrar o melhor ajuste para tarefas específicas. Finalmente, um componente crítico é o supervisor assíncrono, um LLM que executa concorrentemente com o agente principal. Este supervisor avalia periodicamente o comportamento do agente para desvios patológicos ou estagnação e pode intervir enviando notificações ou mesmo cancelando a execução do agente se necessário. Ele recebe uma representação textual detalhada do estado do sistema, incluindo um grafo de chamadas e um fluxo de eventos de mensagens LLM, chamadas de ferramenta e respostas, o que permite detectar padrões ineficientes ou trabalho repetido. 

Um desafio notável na implementação inicial do SICA foi promptar o agente baseado em LLM para independentemente propor modificações novas, inovadoras, viáveis e envolventes durante cada iteração de meta-melhoria. Esta limitação, particularmente em fomentar aprendizado aberto e criatividade autêntica em agentes LLM, permanece uma área chave de investigação na pesquisa atual.

# AlphaEvolve e OpenEvolve

**AlphaEvolve** é um agente de IA desenvolvido pelo Google projetado para descobrir e otimizar algoritmos. Ele utiliza uma combinação de LLMs, especificamente modelos Gemini (Flash e Pro), sistemas de avaliação automatizada, e um framework de algoritmo evolutivo. Este sistema visa avançar tanto matemática teórica quanto aplicações práticas de computação.

O AlphaEvolve emprega um conjunto de modelos Gemini. O Flash é usado para gerar uma ampla gama de propostas iniciais de algoritmo, enquanto o Pro fornece análise mais aprofundada e refinamento. Algoritmos propostos são então automaticamente avaliados e pontuados baseado em critérios predefinidos. Esta avaliação fornece feedback que é usado para melhorar iterativamente as soluções, levando a algoritmos otimizados e novos.

Em computação prática, o AlphaEvolve foi implantado dentro da infraestrutura do Google. Ele demonstrou melhorias no agendamento de data center, resultando em uma redução de 0,7% no uso global de recursos computacionais. Ele também contribuiu para design de hardware sugerindo otimizações para código Verilog em Tensor Processing Units (TPUs) futuras. Além disso, o AlphaEvolve acelerou performance de IA, incluindo uma melhoria de velocidade de 23% em um kernel principal da arquitetura Gemini e até 32,5% de otimização de instruções de baixo nível de GPU para FlashAttention.

No reino de pesquisa fundamental, o AlphaEvolve contribuiu para a descoberta de novos algoritmos para multiplicação de matrizes, incluindo um método para matrizes complexas 4x4 que usa 48 multiplicações escalares, superando soluções previamente conhecidas. Em pesquisa matemática mais ampla, ele redescobriu soluções existentes state-of-the-art para mais de 50 problemas abertos em 75% dos casos e melhorou soluções existentes em 20% dos casos, com exemplos incluindo avanços no problema do número de beijo.

**OpenEvolve** é um agente de codificação evolutivo que aproveita LLMs (consulte Fig.3) para otimizar iterativamente código. Ele orquestra um pipeline de geração de código, avaliação e seleção impulsionados por LLM para continuamente aprimorar programas para uma ampla gama de tarefas. Um aspecto chave do OpenEvolve é sua capacidade de evoluir arquivos de código inteiros, ao invés de ser limitado a funções únicas. O agente é projetado para versatilidade, oferecendo suporte para múltiplas linguagens de programação e compatibilidade com APIs compatíveis com OpenAI para qualquer LLM. Além disso, incorpora otimização multi-objetivo, permite engenharia de prompt flexível, e é capaz de avaliação distribuída para lidar eficientemente com desafios complexos de codificação. 

![][image3]

Fig. 3: A arquitetura interna do OpenEvolve é gerenciada por um controlador. Este controlador orquestra vários componentes chave: o amostrador de programa, Banco de Dados de Programa, Pool de Avaliadores, e Conjuntos de LLM. Sua função principal é facilitar seus processos de aprendizado e adaptação para aprimorar qualidade de código.

Este trecho de código usa a biblioteca OpenEvolve para realizar otimização evolutiva em um programa. Ele inicializa o sistema OpenEvolve com caminhos para um programa inicial, um arquivo de avaliação, e um arquivo de configuração. A linha evolve.run(iterations=1000) inicia o processo evolutivo, executando por 1000 iterações para encontrar uma versão melhorada do programa. Finalmente, ele imprime as métricas do melhor programa encontrado durante a evolução, formatadas para quatro casas decimais. 

```python
from openevolve import OpenEvolve

# Inicializar o sistema
evolve = OpenEvolve(
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)

# Executar a evolução
best_program = await evolve.run(iterations=1000)

print(f"Métricas do melhor programa:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")
```

# Em Resumo

**O que:** Agentes de IA frequentemente operam em ambientes dinâmicos e imprevisíveis onde lógica pré-programada é insuficiente. Sua performance pode degradar quando enfrentam situações novas não antecipadas durante seu design inicial. Sem a capacidade de aprender da experiência, agentes não podem otimizar suas estratégias ou personalizar suas interações ao longo do tempo. Esta rigidez limita sua efetividade e os impede de alcançar verdadeira autonomia em cenários complexos do mundo real.

**Por quê:** A solução padronizada é integrar mecanismos de aprendizado e adaptação, transformando agentes estáticos em sistemas dinâmicos e evolutivos. Isso permite que um agente refine autonomamente seu conhecimento e comportamentos baseado em novos dados e interações. Sistemas agênticos podem usar vários métodos, de aprendizado por reforço a técnicas mais avançadas como auto-modificação, como visto no Agente de Codificação Auto-Melhorador (SICA). Sistemas avançados como o AlphaEvolve do Google aproveitam LLMs e algoritmos evolutivos para descobrir soluções inteiramente novas e mais eficientes para problemas complexos. Ao aprender continuamente, agentes podem dominar novas tarefas, aprimorar sua performance e se adaptar a condições em mudança sem requerer reprogramação manual constante.

**Regra de ouro:** Use este padrão ao construir agentes que devem operar em ambientes dinâmicos, incertos ou evolutivos. É essencial para aplicações requerendo personalização, melhoria contínua de performance e a capacidade de lidar com situações novas autonomamente.

**Resumo visual**

**![][image4]**

Fig.4: Padrão de aprendizado e adaptação

# Principais Conclusões

* Aprendizado e Adaptação são sobre agentes ficando melhores no que fazem e lidando com novas situações usando suas experiências.  
* "Adaptação" é a mudança visível no comportamento ou conhecimento de um agente que vem do aprendizado.  
* SICA, o Agente de Codificação Auto-Melhorador, se auto-melhora modificando seu código baseado em performance passada. Isso levou a ferramentas como o Editor Inteligente e Localizador de Símbolo AST.  
* Ter "sub-agentes" especializados e um "supervisor" ajuda estes sistemas auto-melhoradores gerenciarem tarefas grandes e permanecerem no caminho certo.  
* A maneira como a "janela de contexto" de um LLM é configurada (com prompts do sistema, prompts principais, e mensagens do assistente) é super importante para quão eficientemente agentes trabalham.  
* Este padrão é vital para agentes que precisam operar em ambientes que estão sempre mudando, incertos, ou requerem um toque pessoal.  
* Construir agentes que aprendem frequentemente significa conectá-los com ferramentas de machine learning e gerenciar como os dados fluem.  
* Um sistema de agente, equipado com ferramentas básicas de codificação, pode autonomamente se editar, e assim melhorar sua performance em tarefas de benchmark  
* AlphaEvolve é o agente de IA do Google que aproveita LLMs e um framework evolutivo para autonomamente descobrir e otimizar algoritmos, aprimorando significativamente tanto pesquisa fundamental quanto aplicações práticas de computação.

# Conclusão

Este capítulo examina os papéis cruciais de aprendizado e adaptação na Inteligência Artificial. Agentes de IA aprimoram sua performance através de aquisição contínua de dados e experiência. O Agente de Codificação Auto-Melhorador (SICA) exemplifica isso melhorando autonomamente suas capacidades através de modificações de código.

Revisamos os componentes fundamentais da IA agêntica, incluindo arquitetura, aplicações, planejamento, colaboração multi-agente, gerenciamento de memória, e aprendizado e adaptação. Princípios de aprendizado são particularmente vitais para melhoria coordenada em sistemas multi-agente. Para alcançar isso, dados de tuning devem refletir com precisão a trajetória completa de interação, capturando as entradas e saídas individuais de cada agente participante.

Estes elementos contribuem para avanços significativos, como o AlphaEvolve do Google. Este sistema de IA independentemente descobre e refina algoritmos por LLMs, avaliação automatizada e uma abordagem evolutiva, impulsionando progresso em pesquisa científica e técnicas computacionais. Tais padrões podem ser combinados para construir sistemas de IA sofisticados. Desenvolvimentos como AlphaEvolve demonstram que descoberta e otimização algorítmica autônoma por agentes de IA são alcançáveis.

# Referências

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.   
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.   
3. Self-Improving Coding Agent (SICA), [https://github.com/MaximeRobeyns/self_improving_coding_agent/](https://github.com/MaximeRobeyns/self_improving_coding_agent/)   
4. AlphaEvolve: Autonomous Discovery and Optimization of Algorithms, [https://arxiv.org/abs/2406.18588](https://arxiv.org/abs/2406.18588)   
5. OpenEvolve: Evolutionary Coding Agent, [https://github.com/openevolve/openevolve](https://github.com/openevolve/openevolve)

[image1]: ../assets/14-chapter-9-image-1-line-157.png

[image2]: ../assets/14-chapter-9-image-2-line-159.png

[image3]: ../assets/14-chapter-9-image-3-line-161.png

[image4]: ../assets/14-chapter-9-image-4-line-163.png
