# 🎧 Template de Metadados para Spotify - Padrões de Design Agênticos

## 📋 **Estrutura de Metadados por Episódio**

### 🎯 **Campos Obrigatórios do Spotify:**

#### **1. Título do Episódio**
- Formato: `[Episódio XX] Título Principal - Subtítulo Descritivo`
- Máximo: 60 caracteres (recomendado)
- Exemplo: `[Episódio 01] Prompt Chaining - Como Encadear Prompts para IA`

#### **2. Descrição do Episódio**
- Máximo: 4000 caracteres
- Estrutura:
  ```
  🎯 **Sobre este episódio:**
  [Descrição do padrão/conceito em 2-3 parágrafos]

  🔍 **O que você vai aprender:**
  • [Ponto 1]
  • [Ponto 2]
  • [Ponto 3]

  💻 **Conteúdo prático:**
  [Exemplos de código e implementação]

  📚 **Recursos mencionados:**
  • [Link 1]
  • [Link 2]

  ⚖️ **Direitos Autorais:**
  Este episódio é baseado no livro "Agentic Design Patterns: A Practical Guide to Building Intelligent Systems" de Antonio Gulli. Todos os direitos reservados ao autor original.
  
  📖 **Material Completo:**
  Acesse o livro completo e códigos em: [LINK DO DRIVE]
  
  👤 **Autor Original:**
  Antonio Gulli - LinkedIn: https://www.linkedin.com/in/searchguy/
  ```

#### **3. Tags/Categorias**
- Categoria principal: `Technology`
- Subcategorias: `Artificial Intelligence`, `Programming`, `Software Development`
- Tags: `IA`, `Agentes`, `LLM`, `Machine Learning`, `Desenvolvimento de Software`

#### **4. Duração**
- Extrair do arquivo MP3 (já disponível nos metadados)

#### **5. Data de Publicação**
- Sequencial baseado na ordem dos episódios

---

## 🎙️ **Metadados da Série (Show Metadata)**

### **Título da Série:**
`Padrões de Design Agênticos: Guia Completo em Português`

### **Descrição da Série:**
```
🤖 **Sobre a Série:**

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
Acesse o livro completo, códigos e recursos em: [LINK DO DRIVE]

👤 **Autor Original:**
Antonio Gulli
• LinkedIn: https://www.linkedin.com/in/searchguy/
• Twitter: https://x.com/antoniogulli
• Livro Original: https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/preview?tab=t.0

🎧 **Episódios:**
[Será preenchido com lista completa dos 28 episódios]
```

### **Categoria da Série:**
- Principal: `Technology`
- Secundária: `Education`

### **Idioma:**
`Portuguese (Brazil)`

### **Palavras-chave:**
`Inteligência Artificial`, `Agentes de IA`, `Machine Learning`, `Desenvolvimento de Software`, `LLM`, `Padrões de Design`, `Programação`, `Tecnologia`

---

## 📊 **Estrutura de Dados para Automação**

### **JSON Template para Episódio:**
```json
{
  "episode_number": "01",
  "episode_type": "main|special",
  "title": "[Episódio 01] Prompt Chaining - Como Encadear Prompts para IA",
  "description": "...",
  "duration": "00:20:00",
  "chapter_file": "07-capitulo-1-prompt-chaining.md",
  "audio_file": "01-Prompt_Chaining__Como_o_Encadeamento_de_Prompts_e_a_Engenharia_.mp3",
  "tags": ["IA", "Agentes", "LLM", "Prompt Engineering"],
  "publish_date": "2024-01-15",
  "show_notes": {
    "key_concepts": ["Prompt Chaining", "Sequential Processing"],
    "practical_examples": ["Code examples", "Use cases"],
    "frameworks_mentioned": ["LangChain", "LangGraph"],
    "external_links": ["Drive link", "Author links"]
  }
}
```

---

## 🔄 **Processo de Geração de Metadados**

### **Fase 1: Preparação**
1. ✅ Mapear todos os 28 episódios
2. ✅ Identificar arquivos de áudio correspondentes
3. ✅ Mapear capítulos do livro relacionados
4. ✅ Extrair duração dos arquivos MP3

### **Fase 2: Geração de Conteúdo**
1. 🔄 Ler conteúdo de cada capítulo
2. 🔄 Gerar descrições personalizadas
3. 🔄 Identificar conceitos-chave
4. 🔄 Listar frameworks e ferramentas mencionadas

### **Fase 3: Validação**
1. ⏳ Verificar conformidade com limites do Spotify
2. ⏳ Validar links e referências
3. ⏳ Confirmar direitos autorais
4. ⏳ Revisar qualidade do conteúdo

### **Fase 4: Exportação**
1. ⏳ Gerar arquivos JSON estruturados
2. ⏳ Criar planilha CSV para upload
3. ⏳ Preparar arquivos de metadados para cada episódio
4. ⏳ Documentar processo completo

---

## 📋 **Checklist de Qualidade**

### **Para Cada Episódio:**
- [ ] Título dentro do limite de caracteres
- [ ] Descrição informativa e envolvente
- [ ] Links funcionais incluídos
- [ ] Direitos autorais claramente indicados
- [ ] Referência ao material completo
- [ ] Tags relevantes aplicadas
- [ ] Duração correta especificada

### **Para a Série:**
- [ ] Descrição completa e atrativa
- [ ] Categoria apropriada selecionada
- [ ] Idioma correto especificado
- [ ] Palavras-chave relevantes
- [ ] Informações do autor completas
- [ ] Links para recursos externos
- [ ] Aviso legal adequado

---

## 🎯 **Próximos Passos**

1. **Gerar metadados para todos os 28 episódios**
2. **Criar arquivos estruturados (JSON/CSV)**
3. **Validar conformidade legal**
4. **Preparar para upload no Spotify**
5. **Documentar processo para futuras atualizações**

