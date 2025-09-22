# Apêndice D - Construindo um Agente com AgentSpace

# Visão Geral

AgentSpace é uma plataforma projetada para facilitar uma "empresa dirigida por agentes" integrando inteligência artificial em workflows diários. Em seu núcleo, fornece uma capacidade de busca unificada através de toda a pegada digital de uma organização, incluindo documentos, emails e bancos de dados. Este sistema utiliza modelos de IA avançados, como o Gemini do Google, para compreender e sintetizar informação dessas fontes variadas.

A plataforma habilita a criação e implantação de "agentes" de IA especializados que podem executar tarefas complexas e automatizar processos. Esses agentes não são meramente chatbots; eles podem raciocinar, planejar e executar ações multi-etapas autonomamente. Por exemplo, um agente poderia pesquisar um tópico, compilar um relatório com citações, e até gerar um resumo em áudio.

Para alcançar isso, AgentSpace constrói um grafo de conhecimento empresarial, mapeando as relações entre pessoas, documentos e dados. Isso permite que a IA entenda contexto e entregue resultados mais relevantes e personalizados. A plataforma também inclui uma interface no-code chamada Agent Designer para criar agentes customizados sem requerer expertise técnica profunda.

Além disso, AgentSpace suporta um sistema multi-agente onde diferentes agentes de IA podem se comunicar e colaborar através de um protocolo aberto conhecido como o Protocolo Agent2Agent (A2A). Esta interoperabilidade permite workflows mais complexos e orquestrados. Segurança é um componente fundamental, com recursos como controles de acesso baseados em papéis e criptografia de dados para proteger informação empresarial sensível. Em última análise, AgentSpace visa aprimorar produtividade e tomada de decisão ao incorporar sistemas inteligentes e autônomos diretamente no tecido operacional de uma organização.

# Como construir um Agente com a UI do AgentSpace

A Figura 1 ilustra como acessar AgentSpace selecionando AI Applications do Google Cloud Console.

![][image1]  
Fig. 1: Como usar o Google Cloud Console para acessar AgentSpace

Seu agente pode ser conectado a vários serviços, incluindo Calendar, Google Mail, Workaday, Jira, Outlook e Service Now (veja Fig. 2).

![][image2]  
Fig. 2: Integrar com diversos serviços, incluindo plataformas Google e de terceiros.

O Agente pode então utilizar seu próprio prompt, escolhido de uma galeria de prompts pré-fabricados fornecidos pelo Google, como ilustrado na Fig. 3.

![][image3]  
Fig.3: Galeria de Prompts Pré-montados do Google

Alternativamente você pode criar seu próprio prompt como na Fig.4, que será então usado por seu agente  
![][image4]  
Fig.4: Customizando o Prompt do Agente    
   
AgentSpace oferece vários recursos avançados como integração com datastores para armazenar seus próprios dados, integração com Google Knowledge Graph ou com seu Knowledge Graph privado, Interface Web para expor seu agente à Web, e Analytics para monitorar uso, e mais (veja Fig. 5)   
![][image5]  
Fig. 5: Capacidades avançadas do AgentSpace 

Após a conclusão, a interface de chat do AgentSpace (Fig. 6) estará acessível.

![][image6]  
Fig. 6: A Interface do Usuário do AgentSpace para iniciar um chat com seu Agente.

# Conclusão

Em conclusão, AgentSpace fornece um framework funcional para desenvolver e implantar agentes de IA dentro da infraestrutura digital existente de uma organização. A arquitetura do sistema conecta processos de backend complexos, como raciocínio autônomo e mapeamento de grafo de conhecimento empresarial, a uma interface gráfica do usuário para construção de agentes. Através desta interface, usuários podem configurar agentes integrando vários serviços de dados e definindo seus parâmetros operacionais via prompts, resultando em sistemas automatizados customizados e conscientes do contexto.

Esta abordagem abstrai a complexidade técnica subjacente, habilitando a construção de sistemas multi-agente especializados sem requerer expertise de programação profunda. O objetivo primário é incorporar capacidades analíticas e operacionais automatizadas diretamente em workflows, aumentando assim a eficiência do processo e aprimorando análise dirigida por dados. Para instrução prática, módulos de aprendizado hands-on estão disponíveis, como o laboratório "Build a Gen AI Agent with Agentspace" no Google Cloud Skills Boost, que fornece um ambiente estruturado para aquisição de habilidades.

# Referências

1. Create a no-code agent with Agent Designer, [https://cloud.google.com/agentspace/agentspace-enterprise/docs/agent-designer](https://cloud.google.com/agentspace/agentspace-enterprise/docs/agent-designer)   
2. Google Cloud Skills Boost, [https://www.cloudskillsboost.google/](https://www.cloudskillsboost.google/)

[image1]: ../assets/30-appendix-d-image-1-line-54.png

[image2]: ../assets/30-appendix-d-image-2-line-56.png

[image3]: ../assets/30-appendix-d-image-3-line-58.png

[image4]: ../assets/30-appendix-d-image-4-line-60.png

[image5]: ../assets/30-appendix-d-image-5-line-62.png

[image6]: ../assets/30-appendix-d-image-6-line-64.png
