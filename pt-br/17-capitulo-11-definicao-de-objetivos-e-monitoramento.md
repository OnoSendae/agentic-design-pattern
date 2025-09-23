# Capítulo 11: Definição de Objetivos e Monitoramento

Para agentes de IA serem verdadeiramente efetivos e propositais, eles precisam de mais do que apenas a capacidade de processar informação ou usar ferramentas; eles precisam de um senso claro de direção e uma maneira de saber se estão realmente tendo sucesso. É aqui que o padrão de Definição de Objetivos e Monitoramento entra em jogo. Trata-se de dar aos agentes objetivos específicos para trabalhar e equipá-los com os meios para rastrear seu progresso e determinar se esses objetivos foram alcançados.

# Visão Geral do Padrão de Definição de Objetivos e Monitoramento

Pense em planejar uma viagem. Você não simplesmente aparece espontaneamente no seu destino. Você decide para onde quer ir (o estado objetivo), descobre de onde está começando (o estado inicial), considera opções disponíveis (transporte, rotas, orçamento), e então mapeia uma sequência de passos: reservar passagens, fazer malas, viajar para o aeroporto/estação, embarcar no transporte, chegar, encontrar acomodação, etc. Este processo passo-a-passo, frequentemente considerando dependências e restrições, é fundamentalmente o que queremos dizer com planejamento em sistemas agênticos.

No contexto de agentes de IA, planejamento tipicamente envolve um agente tomando um objetivo de alto nível e gerando autonomamente, ou semi-autonomamente, uma série de passos intermediários ou sub-objetivos. Estes passos podem então ser executados sequencialmente ou em um fluxo mais complexo, potencialmente envolvendo outros padrões como uso de ferramentas, roteamento ou colaboração multi-agente. O mecanismo de planejamento pode envolver algoritmos de busca sofisticados, raciocínio lógico, ou cada vez mais, aproveitando as capacidades de modelos de linguagem grandes (LLMs) para gerar planos plausíveis e efetivos baseados em seus dados de treinamento e entendimento de tarefas.

Uma boa capacidade de planejamento permite que agentes enfrentem problemas que não são consultas simples de passo único. Permite que lidem com solicitações multifacetadas, se adaptem a circunstâncias em mudança replanejando, e orquestrem fluxos de trabalho complexos. É um padrão fundamental que sustenta muitos comportamentos agênticos avançados, transformando um sistema reativo simples em um que pode trabalhar proativamente em direção a um objetivo definido.

# Aplicações Práticas e Casos de Uso

O padrão de Definição de Objetivos e Monitoramento é essencial para construir agentes que podem operar autonomamente e confiavelmente em cenários complexos do mundo real. Aqui estão algumas aplicações práticas:

* **Automação de Suporte ao Cliente:** O objetivo de um agente pode ser "resolver consulta de cobrança do cliente." Ele monitora a conversa, verifica entradas do banco de dados e usa ferramentas para ajustar a cobrança. O sucesso é monitorado confirmando a mudança de cobrança e recebendo feedback positivo do cliente. Se o problema não for resolvido, ele escala.  
* **Sistemas de Aprendizado Personalizados:** Um agente de aprendizado pode ter o objetivo de "melhorar a compreensão de álgebra dos estudantes." Ele monitora o progresso do estudante em exercícios, adapta materiais de ensino e rastreia métricas de performance como precisão e tempo de conclusão, ajustando sua abordagem se o estudante tiver dificuldades.  
* **Assistentes de Gerenciamento de Projeto:** Um agente pode ser encarregado de "garantir que marco do projeto X seja completado até a data Y." Ele monitora status de tarefas, comunicações da equipe e disponibilidade de recursos, sinalizando atrasos e sugerindo ações corretivas se o objetivo estiver em risco.  
* **Bots de Trading Automatizados:** O objetivo de um agente de trading pode ser "maximizar ganhos do portfólio enquanto permanece dentro da tolerância de risco." Ele monitora continuamente dados de mercado, o valor atual do seu portfólio e indicadores de risco, executando trades quando condições se alinham com seus objetivos e ajustando estratégia se limites de risco forem violados.  
* **Robótica e Veículos Autônomos:** O objetivo primário de um veículo autônomo é "transportar passageiros com segurança de A para B." Ele monitora constantemente seu ambiente (outros veículos, pedestres, sinais de trânsito), seu próprio estado (velocidade, combustível) e seu progresso ao longo da rota planejada, adaptando seu comportamento de direção para alcançar o objetivo com segurança e eficiência.  
* **Moderação de Conteúdo:** O objetivo de um agente pode ser "identificar e remover conteúdo prejudicial da plataforma X." Ele monitora conteúdo recebido, aplica modelos de classificação e rastreia métricas como falsos positivos/negativos, ajustando seus critérios de filtragem ou escalando casos ambíguos para revisores humanos.

Este padrão é fundamental para agentes que precisam operar confiavelmente, alcançar resultados específicos e se adaptar a condições dinâmicas, fornecendo o framework necessário para autogerenciamento inteligente.

# Exemplo de Código Hands-On

Para ilustrar o padrão de Definição de Objetivos e Monitoramento, temos um exemplo usando APIs LangChain e OpenAI. Este script Python descreve um agente de IA autônomo projetado para gerar e refinar código Python. Sua função central é produzir soluções para problemas especificados, garantindo aderência a benchmarks de qualidade definidos pelo usuário.

Ele emprega um padrão de "definição de objetivos e monitoramento" onde não apenas gera código uma vez, mas entra em um ciclo iterativo de criação, auto-avaliação e melhoria. O sucesso do agente é medido por seu próprio julgamento impulsionado por IA sobre se o código gerado atende com sucesso aos objetivos iniciais. A saída final é um arquivo Python polido, comentado e pronto para uso que representa a culminação deste processo de refinamento.

 **Dependências**: 

```python
pip install langchain_openai openai python-dotenv .env file with key in OPENAI_API_KEY
```

Você pode entender melhor este script imaginando-o como um programador de IA autônomo designado para um projeto (consulte Fig. 1). O processo começa quando você entrega à IA um brief detalhado do projeto, que é o problema específico de codificação que ela precisa resolver.

```python
# MIT License
# Copyright (c) 2025 Mahtab Syed
# https://www.linkedin.com/in/mahtabsyed/

"""
Exemplo de Código Hands-On - Iteração 2
Para ilustrar o padrão de Definição de Objetivos e Monitoramento, temos um exemplo usando APIs LangChain e OpenAI:

Objetivo: Construir um Agente de IA que pode escrever código para um caso de uso especificado baseado em objetivos especificados:
- Aceita um problema de codificação (caso de uso) em código ou pode ser como entrada.
- Aceita uma lista de objetivos (ex., "simples", "testado", "lida com casos extremos") em código ou pode ser entrada.
- Usa um LLM (como GPT-4o) para gerar e refinar código Python até que os objetivos sejam atingidos.
  (Estou usando máx 5 iterações, isso poderia ser baseado em um objetivo definido também)
- Para verificar se atingimos nossos objetivos estou pedindo ao LLM para julgar isso e responder apenas True ou False que torna mais fácil parar as iterações.
- Salva o código final em um arquivo .py com um nome de arquivo limpo e um comentário de cabeçalho.
"""

import os
import random
import re
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# 🔐 Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("❌ Por favor, defina a variável de ambiente OPENAI_API_KEY.")

# ✅ Inicializar modelo OpenAI
print("📡 Inicializando LLM OpenAI (gpt-4o)...")
llm = ChatOpenAI(
    model="gpt-4o",  # Se você não tem acesso ao gpt-4o use outros LLMs OpenAI
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
)

# --- Funções Utilitárias ---
def generate_prompt(use_case: str, goals: list[str], previous_code: str = "", feedback: str = "") -> str:
    print("📝 Construindo prompt para geração de código...")
    base_prompt = f"""
Você é um agente de codificação de IA. Seu trabalho é escrever código Python baseado no seguinte caso de uso:

Caso de Uso: {use_case}

Seus objetivos são:
{chr(10).join(f"- {g.strip()}" for g in goals)}
"""
    if previous_code:
        print("🔄 Adicionando código anterior ao prompt para refinamento.")
        base_prompt += f"\nCódigo previamente gerado:\n{previous_code}"
    
    if feedback:
        print("📋 Incluindo feedback para revisão.")
        base_prompt += f"\nFeedback na versão anterior:\n{feedback}\n"
    
    base_prompt += "\nPor favor, retorne apenas o código Python revisado. Não inclua comentários ou explicações fora do código."
    return base_prompt

def get_code_feedback(code: str, goals: list[str]) -> str:
    print("🔍 Avaliando código contra os objetivos...")
    feedback_prompt = f"""
Você é um revisor de código Python. Um snippet de código é mostrado abaixo. 
Baseado nos seguintes objetivos:

{chr(10).join(f"- {g.strip()}" for g in goals)}

Por favor, critique este código e identifique se os objetivos são atendidos. 
Mencione se melhorias são necessárias para clareza, simplicidade, correção, 
tratamento de casos extremos, ou cobertura de testes.

Código:
{code}
"""
    return llm.invoke(feedback_prompt)

def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """
    Usa o LLM para avaliar se os objetivos foram atingidos baseado no texto de feedback.
    Retorna True ou False (parseado da saída do LLM).
    """
    review_prompt = f"""
Você é um revisor de IA. Aqui estão os objetivos:
{
chr(10).join(f"- {g.strip()}" for g in goals)}

Aqui está o feedback no código:
\"\"\"
{feedback_text}
\"\"\"

Baseado no feedback acima, os objetivos foram atingidos? 
Responda com apenas uma palavra: True ou False.
"""
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true"

def clean_code_block(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def add_comment_header(code: str, use_case: str) -> str:
    comment = f"# Este programa Python implementa o seguinte caso de uso:\n# {use_case.strip()}\n"
    return comment + "\n" + code

def to_snake_case(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return re.sub(r"\s+", "_", text.strip().lower())

def save_code_to_file(code: str, use_case: str) -> str:
    print("💾 Salvando código final no arquivo...")
    summary_prompt = (
        f"Resuma o seguinte caso de uso em uma única palavra ou frase em minúsculas, "
        f"não mais que 10 caracteres, adequada para um nome de arquivo Python:\n\n{use_case}"
    )
    raw_summary = llm.invoke(summary_prompt).content.strip()
    short_name = re.sub(r"[^a-zA-Z0-9_]", "", raw_summary.replace(" ", "_").lower())[:10]
    random_suffix = str(random.randint(1000, 9999))
    filename = f"{short_name}
_{random_suffix}.py"
    filepath = Path.cwd() / filename
    with open(filepath, "w") as f:
        f.write(code)
    print(f"✅ Código salvo em: {filepath}")
    return str(filepath)

# --- Função Principal do Agente ---
def run_code_agent(use_case: str, goals_input: str, max_iterations: int = 5) -> str:
    goals = [g.strip() for g in goals_input.split(",")]
    print(f"\n🎯 Caso de Uso: {use_case}")
    print("🎯 Objetivos:")
    for g in goals:
        print(f" - {g}")
    
    previous_code = ""
    feedback = ""
    
    for i in range(max_iterations):
        print(f"\n=== 🔁 Iteração {i + 1} de {max_iterations} ===")
        
        prompt = generate_prompt(use_case, goals, previous_code, 
                                feedback if isinstance(feedback, str) else feedback.content)
        print("🚧 Gerando código...")
        code_response = llm.invoke(prompt)
        raw_code = code_response.content.strip()
        code = clean_code_block(raw_code)
        
        print("\n🧾 Código Gerado:\n" + "-" * 50 + f"\n{code}\n" + "-" * 50)
        
        print("\n📤 Submetendo código para revisão de feedback...")
        feedback = get_code_feedback(code, goals)
        feedback_text = feedback.content.strip()
        
        print("\n📥 Feedback Recebido:\n" + "-" * 50 + f"\n{feedback_text}\n" + "-" * 50)
        
        if goals_met(feedback_text, goals):
            print("✅ LLM confirma que objetivos são atingidos. Parando iteração.")
            break
        
        print("🛠️ Objetivos não totalmente atingidos. Preparando para próxima iteração...")
        previous_code = code
    
    final_code = add_comment_header(code, use_case)
    return save_code_to_file(final_code, use_case)

# --- Execução de Teste CLI ---
if __name__ == "__main__":
    print("\n🧠 Bem-vindo ao Agente de Geração de Código IA")
    
    # Exemplo 1
    use_case_input = "Escreva código para encontrar BinaryGap de um inteiro positivo dado"
    goals_input = ("Código simples de entender, Funcionalmente correto, "
                   "Lida com casos extremos abrangentes, Aceita apenas entrada de inteiro positivo, "
                   "imprime os resultados com alguns exemplos")
    run_code_agent(use_case_input, goals_input)
    
    # Exemplo 2
    # use_case_input = ("Escreva código para contar o número de arquivos no diretório atual "
    #                   "e todos os seus sub diretórios aninhados, e imprimir a contagem total")
    # goals_input = ("Código simples de entender, Funcionalmente correto, "
    #                "Lida com casos extremos abrangentes, Ignorar recomendações de performance, "
    #                "Ignorar recomendações para usar suite de testes como unittest ou pytest")
    # run_code_agent(use_case_input, goals_input)
    
    # Exemplo 3
    # use_case_input = ("Escreva código que toma uma entrada de linha de comando de um arquivo "
    #                   "word doc ou docx e o abre e conta o número de palavras, e caracteres "
    #                   "nele e imprime tudo")
    # goals_input = "Código simples de entender, Funcionalmente correto, Lida com casos extremos"
    # run_code_agent(use_case_input, goals_input)
```

Junto com este brief, você fornece uma lista de verificação de qualidade rigorosa, que representa os objetivos que o código final deve atender—critérios como "a solução deve ser simples," "deve ser funcionalmente correta," ou "precisa lidar com casos extremos inesperados."

![][image1]

Fig.1: Exemplo de Definição de Objetivos e Monitoramento

Com esta atribuição em mãos, o programador de IA começa a trabalhar e produz seu primeiro rascunho do código. No entanto, ao invés de submeter imediatamente esta versão inicial, ele pausa para executar um passo crucial: uma auto-revisão rigorosa. Ele compara meticulosamente sua própria criação contra cada item na lista de verificação de qualidade que você forneceu, atuando como seu próprio inspetor de garantia de qualidade. Após esta inspeção, ele renderiza um veredito simples e imparcial sobre seu próprio progresso: "True" se o trabalho atende todos os padrões, ou "False" se fica aquém.

Se o veredito for "False," a IA não desiste. Ela entra em uma fase de revisão reflexiva, usando os insights de sua auto-crítica para identificar as fraquezas e reescrever inteligentemente o código. Este ciclo de rascunho, auto-revisão e refinamento continua, com cada iteração visando chegar mais perto dos objetivos. Este processo se repete até que a IA finalmente alcance um status "True" satisfazendo todos os requisitos, ou até atingir um limite pré-definido de tentativas, muito como um desenvolvedor trabalhando contra um prazo. Uma vez que o código passa nesta inspeção final, o script empacota a solução polida, adicionando comentários úteis e salvando-a em um arquivo Python novo e limpo, pronto para uso.

**Caveats e Considerações:** É importante notar que esta é uma ilustração exemplar e não código pronto para produção. Para aplicações do mundo real, vários fatores devem ser considerados. Um LLM pode não compreender completamente o significado pretendido de um objetivo e pode incorretamente avaliar sua performance como bem-sucedida. Mesmo se o objetivo for bem compreendido, o modelo pode alucinar. Quando o mesmo LLM é responsável tanto por escrever o código quanto julgar sua qualidade, pode ter mais dificuldade descobrindo que está indo na direção errada.

Em última análise, LLMs não produzem código impecável por magia; você ainda precisa executar e testar o código produzido. Além disso, o "monitoramento" no exemplo simples é básico e cria um risco potencial do processo executar para sempre. 

```text
Atue como um revisor de código especialista com um compromisso profundo de produzir código limpo, 
correto e simples. 

Sua missão principal é eliminar "alucinações" de código garantindo que toda sugestão seja 
fundamentada na realidade e melhores práticas. 

Quando eu fornecer um snippet de código, quero que você:

-- **Identifique e Corrija Erros:** 
   Aponte qualquer falha lógica, bugs ou erros de runtime potenciais.

-- **Simplifique e Refatore:** 
   Sugira mudanças que tornem o código mais legível, eficiente e manutenível sem sacrificar correção.

-- **Forneça Explicações Claras:** 
   Para cada mudança sugerida, explique por que é uma melhoria, referenciando princípios de 
   código limpo, performance ou segurança.

-- **Ofereça Código Corrigido:** 
   Mostre o "antes" e "depois" de suas mudanças sugeridas para que a melhoria seja clara.

Seu feedback deve ser direto, construtivo e sempre visando melhorar a qualidade do código.
```

Uma abordagem mais robusta envolve separar estas preocupações dando papéis específicos a uma equipe de agentes. Por exemplo, construí uma equipe pessoal de agentes de IA usando Gemini onde cada um tem um papel específico:

* O Programador Parceiro: Ajuda a escrever e debater código.  
* O Revisor de Código: Detecta erros e sugere melhorias.  
* O Documentador: Gera documentação clara e concisa.  
* O Escritor de Testes: Cria testes unitários abrangentes.  
* O Refinador de Prompt: Otimiza interações com a IA.

Neste sistema multi-agente, o Revisor de Código, atuando como uma entidade separada do agente programador, tem um prompt similar ao juiz no exemplo, que melhora significativamente a avaliação objetiva. Esta estrutura naturalmente leva a melhores práticas, pois o agente Escritor de Testes pode cumprir a necessidade de escrever testes unitários para o código produzido pelo Programador Parceiro.

Deixo ao leitor interessado a tarefa de adicionar estes controles mais sofisticados e tornar o código mais próximo de pronto para produção.

# Em Resumo

**O que**: Agentes de IA frequentemente carecem de uma direção clara, impedindo-os de agir com propósito além de tarefas simples e reativas. Sem objetivos definidos, eles não podem independentemente enfrentar problemas complexos e multi-etapas ou orquestrar fluxos de trabalho sofisticados. Além disso, não há mecanismo inerente para eles determinarem se suas ações estão levando a um resultado bem-sucedido. Isso limita sua autonomia e os impede de serem verdadeiramente efetivos em cenários dinâmicos do mundo real onde mera execução de tarefas é insuficiente.

**Por quê**: O padrão de Definição de Objetivos e Monitoramento fornece uma solução padronizada incorporando um senso de propósito e auto-avaliação em sistemas agênticos. Envolve definir explicitamente objetivos claros e mensuráveis para o agente alcançar. Concomitantemente, estabelece um mecanismo de monitoramento que rastreia continuamente o progresso do agente e o estado de seu ambiente contra estes objetivos. Isso cria um loop de feedback crucial, permitindo que o agente avalie sua performance, corrija seu curso e adapte seu plano se desviar do caminho para o sucesso. Ao implementar este padrão, desenvolvedores podem transformar agentes reativos simples em sistemas proativos e orientados a objetivos capazes de operação autônoma e confiável.

**Regra de ouro**: Use este padrão quando um agente de IA deve executar autonomamente uma tarefa multi-etapas, adaptar-se a condições dinâmicas e alcançar confiavelmente um objetivo específico de alto nível sem intervenção humana constante.

**Resumo visual**:

![][image2]

Fig.2: Padrões de design de objetivos

# Principais conclusões

Principais conclusões incluem:

* Definição de Objetivos e Monitoramento equipa agentes com propósito e mecanismos para rastrear progresso.  
* Objetivos devem ser específicos, mensuráveis, alcançáveis, relevantes e limitados no tempo (SMART).  
* Definir claramente métricas e critérios de sucesso é essencial para monitoramento efetivo.  
* Monitoramento envolve observar ações do agente, estados ambientais e saídas de ferramentas.  
* Loops de feedback do monitoramento permitem que agentes se adaptem, revisem planos ou escalem problemas.  
* No ADK do Google, objetivos são frequentemente transmitidos através de instruções do agente, com monitoramento realizado através de gerenciamento de estado e interações de ferramentas.

# Conclusão

Este capítulo focou no paradigma crucial de Definição de Objetivos e Monitoramento. Destacei como este conceito transforma agentes de IA de meros sistemas reativos em entidades proativas e orientadas a objetivos. O texto enfatizou a importância de definir objetivos claros e mensuráveis e estabelecer procedimentos rigorosos de monitoramento para rastrear progresso. Aplicações práticas demonstraram como este paradigma suporta operação autônoma confiável através de vários domínios, incluindo atendimento ao cliente e robótica. Um exemplo conceitual de codificação ilustra a implementação destes princípios dentro de um framework estruturado, usando diretrizes de agente e gerenciamento de estado para guiar e avaliar o alcance dos objetivos especificados de um agente. Em última análise, equipar agentes com a capacidade de formular e supervisionar objetivos é um passo fundamental para construir sistemas de IA verdadeiramente inteligentes e responsáveis.

# Referências

1. Mahtab Syed, [https://www.linkedin.com/in/mahtabsyed/](https://www.linkedin.com/in/mahtabsyed/)

[image1]: ../assets/16-chapter-11-image-1-line-104.png

[image2]: ../assets/16-chapter-11-image-2-line-106.png
