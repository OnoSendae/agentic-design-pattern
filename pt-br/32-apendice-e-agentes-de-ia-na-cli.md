# Apêndice E - Agentes de IA na CLI

# Introdução

A linha de comando do desenvolvedor, há muito tempo um bastião de comandos precisos e imperativos, está passando por uma transformação profunda. Está evoluindo de um shell simples para um workspace inteligente e colaborativo alimentado por uma nova classe de ferramentas: Interfaces de Linha de Comando de Agentes de IA (CLIs). Esses agentes vão além de meramente executar comandos; eles entendem linguagem natural, mantêm contexto sobre todo o seu codebase, e podem executar tarefas complexas e multi-etapas que automatizam partes significativas do ciclo de vida de desenvolvimento.

Este guia fornece uma visão aprofundada de quatro players líderes neste campo emergente, explorando suas forças únicas, casos de uso ideais e filosofias distintas para ajudá-lo a determinar qual ferramenta melhor se adapta ao seu workflow. É importante notar que muitos dos casos de uso de exemplo fornecidos para uma ferramenta específica frequentemente podem ser realizados pelos outros agentes também. O diferenciador-chave entre essas ferramentas frequentemente está na qualidade, eficiência e nuance dos resultados que são capazes de alcançar para uma tarefa dada. Existem benchmarks específicos projetados para medir essas capacidades, que serão discutidos nas seções seguintes.

# Claude CLI (Claude Code)

O Claude CLI da Anthropic é projetado como um agente de codificação de alto nível com um entendimento profundo e holístico da arquitetura de um projeto. Sua força central é sua natureza "agentiva", permitindo-lhe criar um modelo mental do seu repositório para tarefas complexas e multi-etapas. A interação é altamente conversacional, assemelhando-se a uma sessão de programação em par onde ele explica seus planos antes de executar. Isso o torna ideal para desenvolvedores profissionais trabalhando em projetos de larga escala envolvendo refatoração significativa ou implementando features com impactos arquiteturais amplos.

**Casos de Uso de Exemplo:**

1. **Refatoração de Larga Escala:** Você pode instruí-lo: "Nossa autenticação de usuário atual depende de cookies de sessão. Refatore todo o codebase para usar JWTs stateless, atualizando os endpoints de login/logout, middleware e manipulação de tokens do frontend." Claude então lerá todos os arquivos relevantes e realizará as mudanças coordenadas.
2. **Integração de API:** Após ser fornecido com uma especificação OpenAPI para um novo serviço de clima, você poderia dizer: "Integre esta nova API de clima. Crie um módulo de serviço para lidar com as chamadas da API, adicione um novo componente para exibir o clima, e atualize o dashboard principal para incluí-lo."
3. **Geração de Documentação**: Apontando para um módulo complexo com código mal documentado, você pode pedir: "Analise o arquivo ./src/utils/data_processing.js. Gere comentários TSDoc abrangentes para cada função, explicando seu propósito, parâmetros e valor de retorno."

Claude CLI funciona como um assistente de codificação especializado, com ferramentas inerentes para tarefas centrais de desenvolvimento, incluindo ingestão de arquivos, análise de estrutura de código e geração de edições. Sua integração profunda com Git facilita gerenciamento direto de branches e commits. A extensibilidade do agente é mediada pelo Multi-tool Control Protocol (MCP), habilitando usuários a definir e integrar ferramentas customizadas. Isso permite interações com APIs privadas, consultas de banco de dados e execução de scripts específicos do projeto. Esta arquitetura posiciona o desenvolvedor como o árbitro do escopo funcional do agente, efetivamente caracterizando Claude como um motor de raciocínio aumentado por ferramentas definidas pelo usuário.

# Gemini CLI

O Gemini CLI do Google é um agente de IA versátil e open-source projetado para poder e acessibilidade. Ele se destaca com o modelo avançado Gemini 2.5 Pro, uma janela de contexto massiva e capacidades multimodais (processando imagens e texto). Sua natureza open-source, tier gratuito generoso e loop "Reason and Act" o tornam um excelente generalista transparente, controlável e acessível para uma audiência ampla, de hobbyistas a desenvolvedores empresariais, especialmente aqueles dentro do ecossistema Google Cloud.

**Casos de Uso de Exemplo:**

1. **Desenvolvimento Multimodal:** Você fornece uma captura de tela de um componente web de um arquivo de design (gemini describe component.png) e o instrui: "Escreva o código HTML e CSS para construir um componente React que pareça exatamente com isso. Certifique-se de que seja responsivo."
2. **Gerenciamento de Recursos Cloud:** Usando sua integração integrada Google Cloud, você pode comandar: "Encontre todos os clusters GKE no projeto de produção que estão executando versões mais antigas que 1.28 e gere um comando gcloud para atualizá-los um por um."
3. **Integração de Ferramentas Empresariais (via MCP):** Um desenvolvedor fornece ao Gemini uma ferramenta customizada chamada get-employee-details que se conecta à API interna de RH da empresa. O prompt é: "Rascunhe um documento de boas-vindas para nosso novo contratado. Primeiro, use a ferramenta get-employee-details --id=E90210 para buscar o nome e equipe deles, e então preencha o welcome_template.md com essa informação."
4. **Refatoração de Larga Escala**: Um desenvolvedor precisa refatorar um codebase Java grande para substituir uma biblioteca de logging depreciada por um novo framework de logging estruturado. Eles podem usar Gemini com um prompt como: Leia todos os arquivos *.java no diretório 'src/main/java'. Para cada arquivo, substitua todas as instâncias do import 'org.apache.log4j' e sua classe 'Logger' por 'org.slf4j.Logger' e 'LoggerFactory'. Reescreva a instanciação do logger e todas as chamadas .info(), .debug() e .error() para usar o novo formato estruturado com pares chave-valor.

Gemini CLI está equipado com um conjunto de ferramentas integradas que permitem interagir com seu ambiente. Isso inclui ferramentas para operações de sistema de arquivos (como leitura e escrita), uma ferramenta shell para executar comandos, e ferramentas para acessar a internet via busca web e busca. Para contexto mais amplo, usa ferramentas especializadas para ler múltiplos arquivos de uma vez e uma ferramenta de memória para salvar informação para sessões posteriores. Esta funcionalidade é construída em uma base segura: sandboxing isola as ações do modelo para prevenir risco, enquanto servidores MCP atuam como uma ponte, habilitando Gemini a se conectar com segurança ao seu ambiente local ou outras APIs.

# Aider

Aider é um assistente de codificação de IA open-source que atua como um verdadeiro programador em par trabalhando diretamente em seus arquivos e fazendo commits de mudanças para o Git. Sua característica definidora é sua direção; ele aplica edições, executa testes para validá-las, e automaticamente faz commit de toda mudança bem-sucedida. Sendo agnóstico a modelos, dá aos usuários controle completo sobre custo e capacidades. Seu workflow centrado em git o torna perfeito para desenvolvedores que valorizam eficiência, controle e um rastro transparente e auditável de todas as modificações de código.

**Casos de Uso de Exemplo:**

1. **Desenvolvimento Orientado a Testes (TDD):** Um desenvolvedor pode dizer: "Crie um teste que falha para uma função que calcula o fatorial de um número." Após Aider escrever o teste e ele falhar, o próximo prompt é: "Agora, escreva o código para fazer o teste passar." Aider implementa a função e executa o teste novamente para confirmar.
2. **Eliminação Precisa de Bugs:** Dado um relatório de bug, você pode instruir Aider: "A função calculate_total em billing.py falha em anos bissextos. Adicione o arquivo ao contexto, corrija o bug, e verifique sua correção contra o conjunto de testes existente."
3. **Atualizações de Dependência:** Você poderia instruí-lo: "Nosso projeto usa uma versão desatualizada da biblioteca 'requests'. Por favor, vá através de todos os arquivos Python, atualize as declarações de import e quaisquer chamadas de função depreciadas para serem compatíveis com a versão mais recente, e então atualize requirements.txt."

# GitHub Copilot CLI

GitHub Copilot CLI estende o popular programador em par de IA para o terminal, com sua principal vantagem sendo sua integração nativa e profunda com o ecossistema GitHub. Ele entende o contexto de um projeto *dentro do GitHub*. Suas capacidades de agente permitem que seja atribuído a uma issue do GitHub, trabalhe em uma correção, e submeta um pull request para revisão humana.

**Casos de Uso de Exemplo:**

1. **Resolução Automatizada de Issues:** Um gerente atribui um ticket de bug (ex., "Issue #123: Corrija erro off-by-one na paginação") ao agente Copilot. O agente então faz checkout de uma nova branch, escreve o código, e submete um pull request referenciando a issue, tudo sem intervenção manual do desenvolvedor.
2. **Q&A Consciente do Repositório:** Um novo desenvolvedor na equipe pode perguntar: "Onde neste repositório está definida a lógica de conexão do banco de dados, e quais variáveis de ambiente ela requer?" Copilot CLI usa sua consciência de todo o repo para fornecer uma resposta precisa com caminhos de arquivo.
3. **Ajudante de Comando Shell:** Quando incerto sobre um comando shell complexo, um usuário pode perguntar: gh? encontre todos os arquivos maiores que 50MB, comprima-os, e coloque-os em uma pasta de arquivo. Copilot gerará o comando shell exato necessário para executar a tarefa.

# Terminal-Bench: Um Benchmark para Agentes de IA em Interfaces de Linha de Comando

Terminal-Bench é um framework de avaliação inovador projetado para avaliar a proficiência de agentes de IA em executar tarefas complexas dentro de uma interface de linha de comando. O terminal é identificado como um ambiente ótimo para operação de agentes de IA devido à sua natureza baseada em texto e sandboxed. O lançamento inicial, Terminal-Bench-Core-v0, compreende 80 tarefas curadas manualmente abrangendo domínios como workflows científicos e análise de dados. Para garantir comparações equitativas, Terminus, um agente minimalista, foi desenvolvido para servir como um testbed padronizado para vários modelos de linguagem. O framework é projetado para extensibilidade, permitindo a integração de diversos agentes através de containerização ou conexões diretas. Desenvolvimentos futuros incluem habilitar avaliações massivamente paralelas e incorporar benchmarks estabelecidos. O projeto encoraja contribuições open-source para expansão de tarefas e aprimoramento colaborativo do framework.

# Conclusão

O surgimento desses poderosos agentes de linha de comando de IA marca uma mudança fundamental no desenvolvimento de software, transformando o terminal em um ambiente dinâmico e colaborativo. Como vimos, não há uma única ferramenta "melhor"; em vez disso, um ecossistema vibrante está se formando onde cada agente oferece uma força especializada. A escolha ideal depende inteiramente das necessidades do desenvolvedor: Claude para tarefas arquiteturais complexas, Gemini para resolução de problemas versátil e multimodal, Aider para edição de código centrada em git e direta, e GitHub Copilot para integração perfeita no workflow do GitHub. À medida que essas ferramentas continuam a evoluir, proficiência em aproveitá-las se tornará uma habilidade essencial, mudando fundamentalmente como desenvolvedores constroem, debuggam e gerenciam software.

# Referências

1. Anthropic. *Claude*. [https://docs.anthropic.com/en/docs/claude-code/cli-reference](https://docs.anthropic.com/en/docs/claude-code/cli-reference)   
2. Google Gemini Cli [https://github.com/google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)   
3. Aider. [https://aider.chat/](https://aider.chat/)  
4. GitHub *Copilot CLI* [https://docs.github.com/en/copilot/github-copilot-enterprise/copilot-cli](https://docs.github.com/en/copilot/github-copilot-enterprise/copilot-cli)  
5. Terminal Bench: [https://www.tbench.ai/](https://www.tbench.ai/)
