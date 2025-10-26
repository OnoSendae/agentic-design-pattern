#!/usr/bin/env python3
"""
Script para gerar metadados completos dos podcasts para distribuição no Spotify
Baseado no livro "Agentic Design Patterns" de Antonio Gulli
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configurações base
BASE_CONFIG = {
    "author": "Antonio Gulli",
    "author_links": {
        "linkedin": "https://www.linkedin.com/in/searchguy/",
        "twitter": "https://x.com/antoniogulli",
        "original_book": "https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/preview?tab=t.0",
        "code_repository": "https://drive.google.com/drive/u/0/folders/1Y3U3IrYCiJ3E45Z8okR5eCg7OPnWQtPV"
    },
    "copyright_notice": "Este episódio é baseado no livro \"Agentic Design Patterns: A Practical Guide to Building Intelligent Systems\" de Antonio Gulli. Todos os direitos reservados ao autor original.",
    "material_link": "Acesse o livro completo e códigos em: https://drive.google.com/drive/u/0/folders/1Y3U3IrYCiJ3E45Z8okR5eCg7OPnWQtPV"
}

# Mapeamento dos episódios principais (01-21)
MAIN_EPISODES = [
    {
        "number": "01",
        "title": "Prompt Chaining - Como Encadear Prompts para IA",
        "audio_file": "01-Prompt_Chaining__Como_o_Encadeamento_de_Prompts_e_a_Engenharia_.mp3",
        "chapter_file": "07-capitulo-1-prompt-chaining.md",
        "duration": "00:20:00",
        "description": "Descubra o poder do Prompt Chaining, um dos padrões mais fundamentais para construir sistemas de IA inteligentes. Explore como quebrar tarefas complexas em sequências de prompts menores e mais gerenciáveis.",
        "key_concepts": ["Prompt Chaining", "Sequential Processing", "Workflow Decomposition", "Structured Output"],
        "frameworks": ["LangChain", "LangGraph", "Google ADK"],
        "tags": ["IA", "Agentes", "LLM", "Prompt Engineering", "LangChain", "Workflows"]
    },
    {
        "number": "02",
        "title": "Roteamento em IA - O Segredo dos Agentes Inteligentes",
        "audio_file": "02-Roteamento_em_IA__O_Segredo_para_Agentes_Inteligentes_que_Decid.mp3",
        "chapter_file": "08-capitulo-2-roteamento.md",
        "duration": "00:19:00",
        "description": "Explore o padrão de Roteamento, o mecanismo que permite aos agentes de IA tomar decisões dinâmicas e adaptar seu comportamento baseado no contexto.",
        "key_concepts": ["Conditional Logic", "Intent Classification", "Dynamic Routing", "Multi-Agent Coordination"],
        "frameworks": ["LangGraph", "Google ADK", "Embedding Models"],
        "tags": ["IA", "Agentes", "Roteamento", "Classificação", "LangGraph", "Decision Making"]
    },
    {
        "number": "03",
        "title": "Paralelização em Agentes de IA - Sistemas Mais Rápidos",
        "audio_file": "03-Paralelização_em_Agentes_de_IA__O_Segredo_para_Sistemas_Mais_Rá.mp3",
        "chapter_file": "09-capitulo-3-paralelizacao.md",
        "duration": "00:16:00",
        "description": "Aprenda como implementar paralelização em agentes de IA para criar sistemas mais rápidos e eficientes, executando múltiplas tarefas simultaneamente.",
        "key_concepts": ["Parallel Processing", "Concurrent Execution", "Performance Optimization", "Async Operations"],
        "frameworks": ["LangChain", "AsyncIO", "Concurrent Processing"],
        "tags": ["IA", "Agentes", "Paralelização", "Performance", "Async", "Concurrency"]
    },
    {
        "number": "04",
        "title": "Reflexão em Agentes de IA - Autoavaliação e Melhoria",
        "audio_file": "04-Reflexão_em_Agentes_de_IA__Autoavaliação_e_o_Poder_do_Padrão_Pr.mp3",
        "chapter_file": "10-capitulo-4-reflexao.md",
        "duration": "00:19:00",
        "description": "Descubra como implementar reflexão em agentes de IA, permitindo que eles se autoavaliem e melhorem continuamente seu desempenho.",
        "key_concepts": ["Self-Reflection", "Self-Evaluation", "Continuous Improvement", "Error Correction"],
        "frameworks": ["LangChain", "Self-Correction Patterns"],
        "tags": ["IA", "Agentes", "Reflexão", "Autoavaliação", "Melhoria Contínua", "Self-Correction"]
    },
    {
        "number": "05",
        "title": "Uso de Ferramentas - Como IA Conecta LLMs ao Mundo Real",
        "audio_file": "05-Uso_de_Ferramentas_(Function_Calling)__Como_a_IA_Conecta_LLMs_a.mp3",
        "chapter_file": "11-capitulo-5-uso-de-ferramentas.md",
        "duration": "00:17:00",
        "description": "Explore como agentes de IA podem usar ferramentas externas e APIs para interagir com o mundo real, expandindo suas capacidades além do texto.",
        "key_concepts": ["Function Calling", "Tool Integration", "API Integration", "External Services"],
        "frameworks": ["LangChain Tools", "OpenAI Functions", "Google ADK"],
        "tags": ["IA", "Agentes", "Function Calling", "APIs", "Ferramentas", "Integração"]
    },
    {
        "number": "06",
        "title": "Planejamento na IA - De Reativo a Estrategista Autônomo",
        "audio_file": "06-Planejamento_na_IA__De_Ferramenta_Reativa_a_Estrategista_Autôno.mp3",
        "chapter_file": "12-capitulo-6-planejamento.md",
        "duration": "00:20:00",
        "description": "Aprenda como implementar planejamento em agentes de IA, transformando-os de ferramentas reativas em estrategistas autônomos capazes de decompor tarefas complexas.",
        "key_concepts": ["Task Planning", "Goal Decomposition", "Strategic Thinking", "Autonomous Planning"],
        "frameworks": ["LangGraph", "Planning Algorithms", "Goal-Oriented Systems"],
        "tags": ["IA", "Agentes", "Planejamento", "Estratégia", "Autonomia", "Goal Decomposition"]
    },
    {
        "number": "07",
        "title": "Colaboração Multiagente - Equipes de IA Trabalhando Juntas",
        "audio_file": "07-Colaboração_Multiagente_em_IA__Por_Que_e_Como_Juntar_Equipes_de.mp3",
        "chapter_file": "13-capitulo-7-colaboracao-multiagente.md",
        "duration": "00:19:00",
        "description": "Descubra como criar sistemas multiagente onde diferentes agentes de IA colaboram para resolver problemas complexos de forma coordenada.",
        "key_concepts": ["Multi-Agent Systems", "Agent Coordination", "Collaborative AI", "Distributed Intelligence"],
        "frameworks": ["CrewAI", "LangGraph", "Multi-Agent Frameworks"],
        "tags": ["IA", "Agentes", "Multiagente", "Colaboração", "Coordenação", "CrewAI"]
    },
    {
        "number": "08",
        "title": "Memória de Agentes de IA - Do Contexto Volátil à Profundidade",
        "audio_file": "08-Memória_de_Agentes_de_IA__Do_Contexto_Volátil_à_Profundidade_do.mp3",
        "chapter_file": "14-capitulo-8-gerenciamento-de-memoria.md",
        "duration": "00:20:00",
        "description": "Explore técnicas avançadas de gerenciamento de memória em agentes de IA, desde contexto volátil até sistemas de memória persistente e estruturada.",
        "key_concepts": ["Memory Management", "Context Persistence", "Long-term Memory", "Memory Retrieval"],
        "frameworks": ["LangChain Memory", "Vector Databases", "Memory Systems"],
        "tags": ["IA", "Agentes", "Memória", "Contexto", "Persistência", "Vector DB"]
    },
    {
        "number": "09",
        "title": "Aprendizado e Auto-Evolução - PPO, DPO e Adaptação Contínua",
        "audio_file": "09-Agentes_de_IA__Do_Aprendizado_à_Auto-Evolução_–_Como_PPO,_DPO_e.mp3",
        "chapter_file": "15-capitulo-9-aprendizado-e-adaptacao.md",
        "duration": "00:23:00",
        "description": "Aprenda sobre técnicas de aprendizado e adaptação em agentes de IA, incluindo PPO, DPO e outros métodos de auto-evolução.",
        "key_concepts": ["Reinforcement Learning", "PPO", "DPO", "Continuous Learning", "Self-Improvement"],
        "frameworks": ["RL Frameworks", "Learning Algorithms", "Adaptation Systems"],
        "tags": ["IA", "Agentes", "Aprendizado", "PPO", "DPO", "Reinforcement Learning"]
    },
    {
        "number": "10",
        "title": "MCP - O Adaptador Universal que Transforma LLMs em Agentes",
        "audio_file": "10-MCP__O_Adaptador_Universal_que_Transforma_LLMs_em_Agentes_de_IA.mp3",
        "chapter_file": "16-capitulo-10-protocolo-de-contexto-do-modelo.md",
        "duration": "00:19:00",
        "description": "Descubra o Model Context Protocol (MCP), o protocolo que permite comunicação padronizada entre LLMs e ferramentas externas.",
        "key_concepts": ["Model Context Protocol", "Standardized Communication", "Tool Integration", "Protocol Design"],
        "frameworks": ["MCP", "Protocol Standards", "Communication Frameworks"],
        "tags": ["IA", "Agentes", "MCP", "Protocolo", "Comunicação", "Padronização"]
    },
    {
        "number": "11",
        "title": "IA Proativa vs. Reativa - Transformando Comportamento",
        "audio_file": "11-IA_Proativa_vs.mp3",
        "chapter_file": "17-capitulo-11-definicao-de-objetivos-e-monitoramento.md",
        "duration": "00:23:00",
        "description": "Explore a diferença entre IA proativa e reativa, e como implementar sistemas que antecipam necessidades e tomam iniciativas.",
        "key_concepts": ["Proactive AI", "Reactive AI", "Initiative Taking", "Anticipatory Systems"],
        "frameworks": ["Goal-Oriented Systems", "Proactive Frameworks"],
        "tags": ["IA", "Agentes", "Proativo", "Reativo", "Iniciativa", "Antecipação"]
    },
    {
        "number": "12",
        "title": "Tratamento de Exceções - Agentes Imbatíveis",
        "audio_file": "12-IA_Imbatível__Como_o_Tratamento_de_Exceções_Transforma_Agentes_.mp3",
        "chapter_file": "18-capitulo-12-tratamento-de-excecoes-e-recuperacao.md",
        "duration": "00:15:00",
        "description": "Aprenda como implementar tratamento robusto de exceções em agentes de IA, criando sistemas resilientes e confiáveis.",
        "key_concepts": ["Exception Handling", "Error Recovery", "Resilient Systems", "Fault Tolerance"],
        "frameworks": ["Error Handling Patterns", "Recovery Systems"],
        "tags": ["IA", "Agentes", "Exceções", "Recuperação", "Resiliência", "Error Handling"]
    },
    {
        "number": "13",
        "title": "Human-in-the-Loop - O Segredo da IA Responsável",
        "audio_file": "13-Human-in-the-Loop_(HITL)__O_Segredo_da_IA_Responsável_–_Estraté.mp3",
        "chapter_file": "19-capitulo-13-human-in-the-loop.md",
        "duration": "00:18:00",
        "description": "Descubra como implementar Human-in-the-Loop (HITL) em agentes de IA para garantir supervisão humana e responsabilidade.",
        "key_concepts": ["Human-in-the-Loop", "Human Oversight", "Responsible AI", "Supervision Systems"],
        "frameworks": ["HITL Frameworks", "Supervision Systems"],
        "tags": ["IA", "Agentes", "HITL", "Supervisão", "Responsabilidade", "Human Oversight"]
    },
    {
        "number": "14",
        "title": "RAG Desvendado - Conectando LLMs ao Conhecimento Atual",
        "audio_file": "14-RAG_Desvendado__Como_Conectar_LLMs_ao_Conhecimento_Atual_e_Acab.mp3",
        "chapter_file": "20-capitulo-14-recuperacao-de-conhecimento-rag.md",
        "duration": "00:22:00",
        "description": "Explore Retrieval-Augmented Generation (RAG), a técnica que permite aos LLMs acessar conhecimento atual e específico.",
        "key_concepts": ["RAG", "Retrieval-Augmented Generation", "Knowledge Integration", "Vector Search"],
        "frameworks": ["LangChain RAG", "Vector Databases", "Embedding Models"],
        "tags": ["IA", "Agentes", "RAG", "Retrieval", "Conhecimento", "Vector Search"]
    },
    {
        "number": "15",
        "title": "A2A do Google - Protocolo que Une Agentes de IA",
        "audio_file": "15-A2A_do_Google__O_Protocolo_Que_Une_Agentes_de_IA_de_Diferentes_.mp3",
        "chapter_file": "21-capitulo-15-comunicacao-inter-agente-a2a.md",
        "duration": "00:26:00",
        "description": "Descubra o Agent-to-Agent (A2A) protocol do Google, que permite comunicação padronizada entre diferentes agentes de IA.",
        "key_concepts": ["Agent-to-Agent", "A2A Protocol", "Inter-Agent Communication", "Protocol Standards"],
        "frameworks": ["Google A2A", "Communication Protocols"],
        "tags": ["IA", "Agentes", "A2A", "Protocolo", "Comunicação", "Google"]
    },
    {
        "number": "16",
        "title": "Otimização Consciente de Recursos - O Malabarismo Dinâmico",
        "audio_file": "16-Otimização_Consciente_de_Recursos_na_IA__O_Malabarismo_Dinâmico.mp3",
        "chapter_file": "22-capitulo-16-otimizacao-consciente-de-recursos.md",
        "duration": "00:20:00",
        "description": "Aprenda sobre otimização consciente de recursos em agentes de IA, balanceando eficiência, custo e performance.",
        "key_concepts": ["Resource Optimization", "Cost Management", "Performance Balancing", "Efficiency"],
        "frameworks": ["Resource Management", "Optimization Algorithms"],
        "tags": ["IA", "Agentes", "Otimização", "Recursos", "Eficiência", "Performance"]
    },
    {
        "number": "17",
        "title": "Agentes Autônomos - Chain-of-Thought, ReAct e Lei de Escala",
        "audio_file": "17-Agentes_Autônomos__Chain-of-Thought,_ReAct_e_a_Lei_de_Escala_da.mp3",
        "chapter_file": "23-capitulo-17-tecnicas-de-racionio.md",
        "duration": "00:19:00",
        "description": "Explore técnicas avançadas de raciocínio em agentes autônomos, incluindo Chain-of-Thought e ReAct.",
        "key_concepts": ["Chain-of-Thought", "ReAct", "Autonomous Reasoning", "Scaling Laws"],
        "frameworks": ["Reasoning Frameworks", "Autonomous Systems"],
        "tags": ["IA", "Agentes", "Chain-of-Thought", "ReAct", "Raciocínio", "Autonomia"]
    },
    {
        "number": "18",
        "title": "Guardrails em IA - Engenharia de Confiança e Defesa em Camadas",
        "audio_file": "18-Guardrails_em_IA__Engenharia_de_Confiança,_Defesa_em_Camadas_e_.mp3",
        "chapter_file": "24-capitulo-18-padroes-de-seguranca-guardrails.md",
        "duration": "00:16:00",
        "description": "Aprenda sobre guardrails em IA, implementando múltiplas camadas de segurança e controle para sistemas confiáveis.",
        "key_concepts": ["Guardrails", "AI Safety", "Defense in Depth", "Trust Engineering"],
        "frameworks": ["Safety Frameworks", "Guardrail Systems"],
        "tags": ["IA", "Agentes", "Guardrails", "Segurança", "Confiança", "AI Safety"]
    },
    {
        "number": "19",
        "title": "Avaliação e Monitoramento - Do Teste AB ao LLM-as-a-Judge",
        "audio_file": "19-Avaliação_e_Monitoramento_de_Agentes_de_IA__Do_Teste_AB_ao_LLM-.mp3",
        "chapter_file": "25-capitulo-19-avaliacao-e-monitoramento.md",
        "duration": "00:25:00",
        "description": "Explore técnicas de avaliação e monitoramento de agentes de IA, desde testes A/B tradicionais até LLM-as-a-Judge.",
        "key_concepts": ["Evaluation Methods", "A/B Testing", "LLM-as-a-Judge", "Performance Monitoring"],
        "frameworks": ["Evaluation Frameworks", "Monitoring Systems"],
        "tags": ["IA", "Agentes", "Avaliação", "Monitoramento", "A/B Testing", "LLM Judge"]
    },
    {
        "number": "20",
        "title": "Priorização em IA - A Bússola Interna da Automação",
        "audio_file": "20-Priorização_em_IA__A_Bússola_Interna_que_Transforma_Automação_e.mp3",
        "chapter_file": "26-capitulo-20-priorizacao.md",
        "duration": "00:13:00",
        "description": "Descubra como implementar sistemas de priorização em agentes de IA para otimizar a tomada de decisões.",
        "key_concepts": ["Priority Systems", "Decision Optimization", "Resource Allocation", "Task Prioritization"],
        "frameworks": ["Priority Frameworks", "Decision Systems"],
        "tags": ["IA", "Agentes", "Priorização", "Decisões", "Otimização", "Alocação"]
    },
    {
        "number": "21",
        "title": "Exploração e Descoberta - Caça aos Desconhecidos Desconhecidos",
        "audio_file": "21-IA_Colaboradora_e_a_Caça_aos__Desconhecidos_Desconhecidos__Como.mp3",
        "chapter_file": "27-capitulo-21-exploracao-e-descoberta.md",
        "duration": "00:17:00",
        "description": "Explore como agentes de IA podem descobrir conhecimento novo e lidar com situações não previstas.",
        "key_concepts": ["Exploration", "Discovery", "Unknown Unknowns", "Novelty Detection"],
        "frameworks": ["Exploration Frameworks", "Discovery Systems"],
        "tags": ["IA", "Agentes", "Exploração", "Descoberta", "Novidade", "Unknown Unknowns"]
    }
]

# Episódios especiais (a-g)
SPECIAL_EPISODES = [
    {
        "number": "A",
        "title": "Engenharia de Prompt - O Segredo de Conversar com IA",
        "audio_file": "a-Engenharia_de_Prompt__O_Segredo_de_Como_Conversar_com_a_IA_para.mp3",
        "chapter_file": "28-apendice-a-tecnicas-avancadas-de-prompting.md",
        "duration": "00:19:00",
        "description": "Mergulhe nas técnicas avançadas de engenharia de prompts que formam a base de toda interação efetiva com modelos de linguagem.",
        "key_concepts": ["Prompt Engineering", "Chain-of-Thought", "Few-Shot Learning", "Zero-Shot Prompting"],
        "frameworks": ["LangChain", "OpenAI", "Google Gemini"],
        "tags": ["IA", "Prompt Engineering", "LLM", "Chain-of-Thought", "Few-Shot Learning"]
    },
    {
        "number": "B",
        "title": "Agentes de IA - Do Clique Humano à Ação no Mundo Real",
        "audio_file": "b-Agentes_de_IA__Do_Clique_Humano_(GUI)_à_Ação_no_Mundo_Real_e_o_.mp3",
        "chapter_file": "29-apendice-b-interacoes-agentivas-de-ia.md",
        "duration": "00:26:00",
        "description": "Explore como agentes de IA podem interagir com interfaces gráficas e executar ações no mundo real.",
        "key_concepts": ["GUI Interaction", "Real-world Actions", "Human-Computer Interaction", "Automation"],
        "frameworks": ["GUI Automation", "Real-world Integration"],
        "tags": ["IA", "Agentes", "GUI", "Automação", "Mundo Real", "Interação"]
    },
    {
        "number": "C",
        "title": "LangChain, LangGraph e CrewAI - Frameworks Essenciais",
        "audio_file": "c-LangChain,_LangGraph_e_CrewAI__Decifrando_os_Frameworks_Essenci.mp3",
        "chapter_file": "30-apendice-c-visao-geral-rapida-de-frameworks-agentivos.md",
        "duration": "00:21:00",
        "description": "Compare e analise os principais frameworks para desenvolvimento de agentes de IA: LangChain, LangGraph e CrewAI.",
        "key_concepts": ["LangChain", "LangGraph", "CrewAI", "Framework Comparison", "Development Tools"],
        "frameworks": ["LangChain", "LangGraph", "CrewAI"],
        "tags": ["IA", "Agentes", "LangChain", "LangGraph", "CrewAI", "Frameworks"]
    },
    {
        "number": "D",
        "title": "Google AgentSpace - Plataforma que Transforma Empresas",
        "audio_file": "d-Google_AgentSpace__Desvendando_a_Plataforma_que_Transforma_Empr.mp3",
        "chapter_file": "31-apendice-d-construindo-um-agente-com-agentspace.md",
        "duration": "00:13:00",
        "description": "Descubra o Google AgentSpace, a plataforma empresarial para construção e deploy de agentes de IA em escala.",
        "key_concepts": ["Google AgentSpace", "Enterprise AI", "Scalable Deployment", "Business Integration"],
        "frameworks": ["Google AgentSpace", "Enterprise Platforms"],
        "tags": ["IA", "Agentes", "Google AgentSpace", "Empresarial", "Escala", "Deploy"]
    },
    {
        "number": "E",
        "title": "Agentes de IA na CLI - Claude, Gemini, Aider e Copilot",
        "audio_file": "e-Agentes_de_IA_na_Linha_de_Comando__Claude,_Gemini,_Aider_e_Copi.mp3",
        "chapter_file": "32-apendice-e-agentes-de-ia-na-cli.md",
        "duration": "00:20:00",
        "description": "Explore como usar agentes de IA diretamente na linha de comando para desenvolvimento e automação.",
        "key_concepts": ["CLI Integration", "Command Line AI", "Development Tools", "Automation"],
        "frameworks": ["Claude CLI", "Gemini CLI", "Aider", "Copilot CLI"],
        "tags": ["IA", "Agentes", "CLI", "Linha de Comando", "Desenvolvimento", "Automação"]
    },
    {
        "number": "F",
        "title": "Anatomia da Razão - Como LLMs Explicam seu Raciocínio",
        "audio_file": "f-Anatomia_da_Razão__Como_Gemini,_ChatGPT_e_Outros_LLMs_Explicam_.mp3",
        "chapter_file": "33-apendice-f-sob-o-capo-uma-visao-interna-dos-motores-de-racionio-dos-agentes.md",
        "duration": "00:18:00",
        "description": "Mergulhe na anatomia interna dos motores de raciocínio dos agentes de IA e como eles explicam seu processo de pensamento.",
        "key_concepts": ["Reasoning Engines", "Explanation Generation", "Internal Processes", "Transparency"],
        "frameworks": ["Reasoning Systems", "Explanation Frameworks"],
        "tags": ["IA", "Agentes", "Raciocínio", "Explicação", "Transparência", "Motores"]
    },
    {
        "number": "G",
        "title": "Agentes de Codificação - Do Clique à Ação no Mundo Real",
        "audio_file": "g-Agentes_de_IA__Do_Clique_Humano_(GUI)_à_Ação_no_Mundo_Real_e_o_.mp3",
        "chapter_file": "34-apendice-g-agentes-de-codificacao.md",
        "duration": "00:26:00",
        "description": "Explore agentes especializados em codificação e como eles podem auxiliar no desenvolvimento de software.",
        "key_concepts": ["Coding Agents", "Software Development", "Code Generation", "Programming Assistance"],
        "frameworks": ["Coding Frameworks", "Development Tools"],
        "tags": ["IA", "Agentes", "Codificação", "Desenvolvimento", "Programação", "Assistência"]
    }
]

def generate_episode_description(episode):
    """Gera descrição completa do episódio"""
    return f"""🎯 **Sobre este episódio:**

{episode['description']}

🔍 **O que você vai aprender:**
• Conceitos fundamentais de {', '.join(episode['key_concepts'][:3])}
• Implementação prática com frameworks modernos
• Casos de uso reais e aplicações empresariais
• Técnicas avançadas de otimização

💻 **Conteúdo prático:**
• Exemplos de código executáveis
• Demonstrações hands-on
• Integração com {', '.join(episode['frameworks'][:2])}
• Melhores práticas de implementação

📚 **Recursos mencionados:**
• {', '.join(episode['frameworks'])}
• Documentação técnica detalhada
• Exemplos de código no repositório

⚖️ **Direitos Autorais:**
{BASE_CONFIG['copyright_notice']}

📖 **Material Completo:**
{BASE_CONFIG['material_link']}

👤 **Autor Original:**
{BASE_CONFIG['author']} - LinkedIn: {BASE_CONFIG['author_links']['linkedin']}"""

def generate_show_metadata():
    """Gera metadados da série completa"""
    return {
        "title": "Padrões de Design Agênticos: Guia Completo em Português",
        "description": """🤖 **Sobre a Série:**

Esta série de podcasts é uma tradução completa e adaptação do livro "Agentic Design Patterns: A Practical Guide to Building Intelligent Systems" de Antonio Gulli, traduzida para português brasileiro.

🧠 **O que são Agentes de IA?**
Um agente de IA é um sistema projetado para perceber seu ambiente e tomar ações para alcançar um objetivo específico. É uma evolução de um Grande Modelo de Linguagem (LLM) padrão, aprimorado com as habilidades de planejar, usar ferramentas e interagir com seus arredores.

📚 **Conteúdo da Série:**
• 21 episódios cobrindo padrões fundamentais de design agêntico
• 7 episódios especiais sobre frameworks e aplicações práticas
• Exemplos de código executáveis
• Casos de uso do mundo real
• Técnicas avançadas de implementação

🎯 **Para quem é esta série:**
• Desenvolvedores de software interessados em IA
• Engenheiros de Machine Learning
• Arquitetos de sistemas inteligentes
• Estudantes e profissionais de tecnologia
• Qualquer pessoa curiosa sobre o futuro da IA

⚖️ **Direitos Autorais:**
Esta série é baseada no livro "Agentic Design Patterns: A Practical Guide to Building Intelligent Systems" de Antonio Gulli. Todos os direitos reservados ao autor original. Esta é uma tradução e adaptação para fins educacionais.

📖 **Material Completo:**
Acesse o livro completo, códigos e recursos em: https://drive.google.com/drive/u/0/folders/1Y3U3IrYCiJ3E45Z8okR5eCg7OPnWQtPV

👤 **Autor Original:**
Antonio Gulli
• LinkedIn: https://www.linkedin.com/in/searchguy/
• Twitter: https://x.com/antoniogulli
• Livro Original: https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/preview?tab=t.0

🎧 **Episódios:**
Esta série contém 28 episódios completos cobrindo todos os aspectos dos padrões de design agênticos, desde conceitos fundamentais até implementações avançadas.""",
        "category": "Technology",
        "subcategory": "Education",
        "language": "Portuguese (Brazil)",
        "keywords": ["Inteligência Artificial", "Agentes de IA", "Machine Learning", "Desenvolvimento de Software", "LLM", "Padrões de Design", "Programação", "Tecnologia"],
        "author": BASE_CONFIG["author"],
        "author_links": BASE_CONFIG["author_links"]
    }

def generate_all_episodes():
    """Gera metadados para todos os episódios"""
    all_episodes = []
    start_date = datetime(2024, 1, 15)
    
    # Episódios principais (01-21)
    for i, episode in enumerate(MAIN_EPISODES):
        episode_metadata = {
            "episode_number": episode["number"],
            "episode_type": "main",
            "title": f"[Episódio {episode['number']}] {episode['title']}",
            "description": generate_episode_description(episode),
            "audio_file": episode["audio_file"],
            "chapter_file": episode["chapter_file"],
            "duration": episode["duration"],
            "tags": episode["tags"],
            "publish_date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "show_notes": {
                "key_concepts": episode["key_concepts"],
                "practical_examples": ["Code examples", "Use cases", "Implementation guides"],
                "frameworks_mentioned": episode["frameworks"],
                "external_links": ["Drive link", "Author LinkedIn", "Original Book"]
            }
        }
        all_episodes.append(episode_metadata)
    
    # Episódios especiais (A-G)
    for i, episode in enumerate(SPECIAL_EPISODES):
        episode_metadata = {
            "episode_number": episode["number"],
            "episode_type": "special",
            "title": f"[Episódio {episode['number']}] {episode['title']}",
            "description": generate_episode_description(episode),
            "audio_file": episode["audio_file"],
            "chapter_file": episode["chapter_file"],
            "duration": episode["duration"],
            "tags": episode["tags"],
            "publish_date": (start_date + timedelta(days=21+i)).strftime("%Y-%m-%d"),
            "show_notes": {
                "key_concepts": episode["key_concepts"],
                "practical_examples": ["Code examples", "Use cases", "Implementation guides"],
                "frameworks_mentioned": episode["frameworks"],
                "external_links": ["Drive link", "Author LinkedIn", "Original Book"]
            }
        }
        all_episodes.append(episode_metadata)
    
    return all_episodes

def main():
    """Função principal para gerar todos os metadados"""
    print("🎧 Gerando metadados completos para Spotify...")
    
    # Gerar metadados da série
    show_metadata = generate_show_metadata()
    
    # Gerar metadados dos episódios
    episodes = generate_all_episodes()
    
    # Criar estrutura final
    complete_metadata = {
        "show_metadata": show_metadata,
        "episodes": episodes,
        "generation_info": {
            "generated_at": datetime.now().isoformat(),
            "total_episodes": len(episodes),
            "main_episodes": len(MAIN_EPISODES),
            "special_episodes": len(SPECIAL_EPISODES),
            "based_on": "Agentic Design Patterns: A Practical Guide to Building Intelligent Systems by Antonio Gulli"
        }
    }
    
    # Salvar arquivo JSON
    output_file = "spotify-complete-metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Metadados gerados com sucesso!")
    print(f"📁 Arquivo salvo: {output_file}")
    print(f"📊 Total de episódios: {len(episodes)}")
    print(f"🎯 Episódios principais: {len(MAIN_EPISODES)}")
    print(f"⭐ Episódios especiais: {len(SPECIAL_EPISODES)}")
    
    # Gerar também arquivo CSV para upload
    generate_csv_export(episodes)
    
    return complete_metadata

def generate_csv_export(episodes):
    """Gera arquivo CSV para upload no Spotify"""
    import csv
    
    csv_file = "spotify-episodes-upload.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Cabeçalho
        writer.writerow([
            'Episode Number',
            'Title',
            'Description',
            'Audio File',
            'Duration',
            'Publish Date',
            'Tags',
            'Type'
        ])
        
        # Dados dos episódios
        for episode in episodes:
            writer.writerow([
                episode['episode_number'],
                episode['title'],
                episode['description'].replace('\n', ' '),  # Remove quebras de linha para CSV
                episode['audio_file'],
                episode['duration'],
                episode['publish_date'],
                ', '.join(episode['tags']),
                episode['episode_type']
            ])
    
    print(f"📊 Arquivo CSV gerado: {csv_file}")

if __name__ == "__main__":
    main()

