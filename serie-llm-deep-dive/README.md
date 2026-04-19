# 🧠 Série: LLMs em Profundidade — Da Atenção ao TurboQuant e Além

[![📚 19 Posts Principais](https://img.shields.io/badge/📚-19_Posts_Principais-blue.svg)](#-mapa-da-série-onda-base--posts-horizontais)
[![🔬 6 Apêndices DEEP](https://img.shields.io/badge/🔬-6_Apêndices_DEEP-purple.svg)](#-apêndices-deep-onda-2--aprofundamentos-verticais)
[![🛠️ 4 Sub-séries](https://img.shields.io/badge/🛠️-4_Sub--séries-orange.svg)](#-sub-séries-especializadas-onda-4--verticais-hands-on)
[![📐 Série Acadêmica](https://img.shields.io/badge/📐-TurboQuant_arXiv:2504.19874-red.svg)](./turboquant/)
[![🌍 Português BR](https://img.shields.io/badge/🌍-Português_BR-yellow.svg)](#)

> **Série didática e progressiva** de posts técnicos no estilo *blog técnico / nota de divulgação científica*, partindo da **arquitetura de uma LLM** até **TurboQuant** e **além** (contexto longo, hardware, frameworks, sparsity, MoE, agentes, RAG, multimodal, reasoning e coding agents).

Cada post combina **fundamento técnico**, **analogias com o mundo real**, **diagramas Mermaid**, **tabelas comparativas** e **referências oficiais** (papers, documentação, repositórios).

---

## 🎯 Sobre a série

Esta série complementa o livro **Padrões de Design Agênticos** trazendo o **substrato técnico** que faz os agentes funcionarem: como uma LLM é construída, como ela é servida em produção, como cabe no seu laptop e como evolui (reasoning, multimodal, agentic).

A jornada parte da **arquitetura básica** de uma LLM, passa pelos **gargalos de inferência** (atenção, KV cache, memória), aprofunda em **quantização** (pesos e KV), chega ao **TurboQuant** com rigor matemático, e segue **além**: contexto longo, hardware, frameworks de inferência, treino e alinhamento, embeddings, RAG, agentes, segurança, multimodalidade, reasoning models e o **loop agêntico de codificação** (Cursor, Claude Code, Aider, Antigravity, Codex, etc.).

### 📊 Estatísticas

| 📚 Conteúdo | 🔢 Quantidade | 📝 Descrição |
|------------|--------------|--------------|
| **Posts Horizontais** | 19 | Da arquitetura Transformer ao loop agêntico de codificação |
| **Apêndices DEEP** | 6 | Aprofundamentos verticais (provas, tutoriais, walkthroughs) |
| **Sub-séries especializadas** | 4 | Hands-on: modelos open 2026, inferência local, math, llama |
| **Série acadêmica TurboQuant** | 1 (10 docs) | Tratamento formal do paper arXiv:2504.19874 |
| **Documentos de apoio** | 8 | Glossário, cheatsheet, FAQ, timeline, bibliografia, roadmaps |
| **Fontes-base** | 2 | Transcrição + síntese inicial do vídeo de origem |
| **Idioma** | PT-BR | Termos técnicos preservados em inglês quando padrão da área |

---

## 🧭 Comece por aqui

| Se você é... | Comece por |
|---|---|
| 🆕 **Novo na série** | [`00-INDEX.md`](./00-INDEX.md) — visão geral completa da série |
| 🚀 **Quer 1 página de referência** | [`CHEATSHEET.md`](./CHEATSHEET.md) — fórmulas e comandos |
| 🌳 **Não sabe que técnica usar** | [`DECISION-TREE.md`](./DECISION-TREE.md) — árvores Mermaid: gargalo → solução |
| 📚 **Quer um glossário** | [`GLOSSARY.md`](./GLOSSARY.md) — termos com link para o post |
| ❓ **Tem dúvidas pontuais** | [`FAQ.md`](./FAQ.md) — perguntas frequentes |
| 📜 **Quer o histórico** | [`TIMELINE.md`](./TIMELINE.md) — 2017–2026, marcos da série |
| 📖 **Quer ler papers** | [`BIBLIOGRAPHY.md`](./BIBLIOGRAPHY.md) — papers e docs em clusters temáticos |

---

## 🗺️ Mapa da série (Onda base — Posts horizontais)

| # | Post | Tema central | Pré-requisito |
|---|------|--------------|---------------|
| 01 | [🏗️ Arquitetura Transformer & LLMs decoder-only](./01-arquitetura-transformer-decoder-llm.md) | Tokens, embeddings, blocos decoder, geração autoregressiva | — |
| 02 | [👁️ Atenção em profundidade: MHA, MQA, GQA, MLA, FlashAttention](./02-attention-mha-mqa-gqa-mla-flashattention.md) | Variantes de atenção, custos quadráticos, FlashAttention 1/2/3 | 01 |
| 03 | [💾 KV cache: anatomia, custos e PagedAttention/vLLM](./03-kv-cache-anatomia-pagedattention-vllm.md) | Por que existe, fórmula de tamanho, fragmentação, vLLM | 01, 02 |
| 04 | [⚖️ Quantização de pesos: GPTQ, AWQ, GGUF, bitsandbytes](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) | INT8, INT4, NF4, formatos GGUF/EXL2, calibração | 01 |
| 05 | [🗜️ Quantização de KV cache: KIVI, KVQuant, CacheGen](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) | Por que KV é difícil; outliers; per-channel/per-token | 03, 04 |
| 06 | [📐 TurboQuant em profundidade: polar, JL e Lloyd–Max](./06-turboquant-deep-dive-polar-jl-lloydmax.md) | Quantização não-enviesada, MSE/IP, two-stage IP, cota $4^{-b}$ | 04, 05 |
| 07 | [📏 Contexto longo: RoPE, YaRN, Ring/StreamingLLM, Mamba](./07-contexto-longo-rope-yarn-ring-streaming.md) | Estender janela, escalar atenção, alternativas ao Transformer | 02, 03 |
| 08 | [🚀 Além da quantização: sparsity, speculative decoding, MoE, distillation](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) | Outras alavancas para acelerar/reduzir custo | 01–06 |
| 09 | [🎓 Treinamento de LLMs: pretraining, SFT, DPO, GRPO, RLHF](./09-treinamento-pretraining-sft-dpo-grpo-rlhf.md) | Pipeline completo de treino: do next-token-prediction ao alinhamento por RL | 01, 04 |
| 10 | [🖥️ Hardware: H100, H200, B100/B200, MI300X, TPU, Apple, Groq](./10-hardware-h100-h200-b100-b200-mi300x-tpu-apple-groq.md) | Chips, HBM, NVLink, datacenters, edge | 01–03 |
| 11 | [⚙️ Frameworks de inferência: vLLM, SGLang, TRT-LLM, TGI, llama.cpp, MLX, Ollama, KTransformers](./11-frameworks-vllm-sglang-trtllm-tgi-llamacpp-mlx-ollama.md) | Qual usar e como configurar para produção | 03, 04, 05 |
| 12 | [🧬 Embeddings: contrastive, MTEB, Matryoshka, multilingue, multimodal](./12-embeddings-contrastive-mteb-matryoshka-multimodal.md) | Anatomia, treinamento e taxonomia dos modelos de embedding | 01 |
| 13 | [🔍 RAG em profundidade: chunking, retrieval híbrido, reranking, GraphRAG, agentic, eval](./13-rag-chunking-retrieval-rerank-graph-agentic-eval.md) | Arquitetura completa de RAG produção 2026 | 06, 12 |
| 14 | [🤖 Agentes, tool use, MCP, ReAct e multi-agent](./14-agentes-tool-use-mcp-react-multiagent.md) | Function calling, MCP, padrões de loop, multi-agent | 11, 13 |
| 15 | [📊 Avaliação: MMLU, GPQA, ARC-AGI, LLM-as-judge, Arena, contaminação](./15-eval-mmlu-gpqa-arc-agi-llm-judge-arena-contamination.md) | Benchmarks, judges, arenas e eval custom de produção | — |
| 16 | [🛡️ Segurança: jailbreaks, prompt injection, alinhamento, red-team, governança](./16-seguranca-jailbreaks-prompt-injection-alinhamento-redteam.md) | Superfície de ataque, defesas, RSP/EU AI Act/PL 2338 | 09 |
| 17 | [🎨 Multimodalidade: VLM, áudio, vídeo, omni — CLIP, LLaVA, Qwen-VL, Gemini](./17-multimodalidade-vlm-audio-video-omni-clip-llava-qwen-vl-gemini.md) | Arquitetura, treino e capabilities de modelos multimodais | 01–03 |
| 18 | [🧩 Reasoning models: o1, o3, R1, QwQ, GRPO e test-time compute](./18-reasoning-models-o1-o3-r1-qwq-grpo-test-time-compute.md) | CoT escalado por RL e tempo de inferência | 09 |
| 19 | [👨‍💻 Loop agêntico de codificação: Cursor, Antigravity, Claude Code, Aider, Cline, OpenCode, Codex CLI, Devin](./19-loop-agentico-coding-cursor-claude-code-aider-cline-opencode-antigravity-codex.md) | Coding agents IDE/CLI/background, edit formats, MCP em código, vibe coding | 14 |

---

## 🔬 Apêndices DEEP (Onda 2 — Aprofundamentos verticais)

| Apêndice | Foco | Ligado a |
|----------|------|----------|
| [02-DEEP](./02-DEEP-online-softmax-flashattention.md) | Online softmax (prova por indução) + FlashAttention 1/2/3/4 + Triton kernel + backward | Post 02 |
| [04-DEEP](./04-DEEP-gptq-qlora-handson.md) | GPTQ algoritmo passo a passo (OBQ → Cholesky) + QLoRA receita completa em código | Post 04 |
| [05-DEEP](./05-DEEP-outliers-kv-quant-tutorial.md) | Análise empírica de outliers em K (notebook) + tutorial vLLM/llama.cpp/MLX/TRT-LLM | Post 05 |
| [06-DEEP](./06-DEEP-mlx-turboquant-walkthrough.md) | Walkthrough TurboQuant em MLX + provas dos Teoremas 1/2 + censo de implementações | Post 06 |
| [07-DEEP](./07-DEEP-yarn-ssm-math.md) | RoPE → PI → NTK → YaRN → LongRoPE matemático + SSM/Mamba/Mamba-2/Jamba completos | Post 07 |
| [08-DEEP](./08-DEEP-speculative-math-eagle.md) | Prova de equivalência distribucional do speculative + EAGLE-1/2/3 + tree decoding + MTP | Post 08 |

---

## 🛠️ Sub-séries especializadas (Onda 4 — Verticais hands-on)

### 🌐 [`serie-modelos-open-2026/`](./serie-modelos-open-2026/) — Hands-on dos modelos open-weights chineses e Gemma

| # | Post | Foco |
|---|------|------|
| 01 | [Qwen 3 hands-on: dense, MoE, Coder, VL, Omni](./serie-modelos-open-2026/01-qwen3-hands-on-dense-moe-coder-vl-omni.md) | Família Alibaba completa: variantes, quantização, serving, fine-tune, casos PT-BR |
| 02 | [Kimi K2+ MoE 1T agentic hands-on](./serie-modelos-open-2026/02-kimi-k2-moe-1t-agentic-handson.md) | MoE 1T da Moonshot AI: KTransformers em consumer, agentic com SWE-bench/τ-bench |
| 03 | [Gemma 3 e 4 hands-on: multimodal & on-device](./serie-modelos-open-2026/03-gemma-3-4-handson-multimodal-ondevice.md) | Família Google: 1B/4B/12B/27B, vision nativa, edge Android/iOS, ShieldGemma |
| 04 | [DeepSeek V3, R1, R2 hands-on: MLA, MTP, FP8 training](./serie-modelos-open-2026/04-deepseek-v3-r1-r2-handson-mla-mtp.md) | 671B MoE eficiente: SGLang+FlashMLA, KTransformers, R1-Distill, R2 reasoning |

### 🏠 [`serie-inferencia-local/`](./serie-inferencia-local/) — Rodar LLMs em casa, no laptop ou no servidor caseiro

| # | Post | Foco |
|---|------|------|
| 01 | [llama.cpp deep workflow: imatrix, quantize, serve](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md) | Pipeline ponta-a-ponta: build → HF → GGUF → imatrix → quantize → bench → llama-server |
| 02 | [MLX no Mac Apple Silicon: mlx-lm, mlx-vlm, fine-tune, distributed](./serie-inferencia-local/02-mlx-mac-silicon-mlx-lm-mlx-vlm-fine-tune.md) | UMA, mlx-lm/mlx-vlm/mlx-audio, LoRA local, cluster Thunderbolt 5 |
| 03 | [Ollama, LM Studio, Open WebUI, Jan, Msty, AnythingLLM](./serie-inferencia-local/03-ollama-lmstudio-openwebui-jan-msty.md) | UX-first: Modelfile, GUI, RAG built-in, MCP, multi-user, voice |
| 04 | [Hardware builds: Mac, PC DDR5, RTX 3090/4090/5090, Pro 6000, MI300X](./serie-inferencia-local/04-hardware-builds-mac-pc-ddr5-rtx-3090-4090-mi300x-amador.md) | 10 builds R\$ 5k → R\$ 400k, ROI vs API hosted, energia/ruído BR |

### 📐 [`serie-llm-math/`](./serie-llm-math/) — Matemática essencial para entender LLMs

| # | Post | Foco |
|---|------|------|
| 01 | [Álgebra linear essencial: vetores, matrizes, SVD, normas](./serie-llm-math/01-algebra-linear-essencial-vetores-matrizes-svd-norms.md) | Da norma L2 ao SVD, com NumPy e aplicação direta a Transformer/LoRA/MLA |
| 02 | [Cálculo, gradientes, autograd e backpropagation](./serie-llm-math/02-calculo-derivadas-gradientes-autograd-backprop.md) | Regra da cadeia matricial, micrograd-style, Adam/AdamW/Lion/Muon, mixed precision |
| 03 | [Probabilidade e info theory: entropia, KL, perplexidade, rate-distortion](./serie-llm-math/03-probabilidade-info-theory-entropia-kl-perplexidade-rate-distortion.md) | Cross-entropy loss, KL no RLHF/DPO/GRPO, sampling (top-k/p/min-p), R(D) e TurboQuant |

### 🦙 [`serie-hands-on-llama/`](./serie-hands-on-llama/) — Hands-on com a família Llama *(em construção)*

---

## 📘 Série acadêmica: [`turboquant/`](./turboquant/)

Tratamento **formal** do paper [TurboQuant — arXiv:2504.19874](https://arxiv.org/abs/2504.19874), capítulo a capítulo:

| # | Capítulo | Foco |
|---|----------|------|
| — | [INDEX.md](./turboquant/INDEX.md) | Mapa da série acadêmica |
| 01 | [Fundamentos e definição formal](./turboquant/01-fundamentos-e-definicao-formal.md) | O problema, notação, MSE vs IP |
| 02 | [Trabalhos relacionados](./turboquant/02-trabalhos-relacionados.md) | PQ, OPQ, RQ, JL e quantização vetorial |
| 03 | [Preliminares: β-esfera e concentração](./turboquant/03-preliminares-beta-esfera-e-concentracao.md) | Geometria de alta dimensão |
| 04 | [Shannon Lower Bound](./turboquant/04-shannon-lower-bound.md) | Limite teórico de rate-distortion |
| 05 | [QJL — Quantized Johnson–Lindenstrauss](./turboquant/05-qjl-quantized-johnson-lindenstrauss.md) | Projeção quantizada |
| 06 | [TurboQuant: MSE e produto interno](./turboquant/06-turboquant-mse-e-produto-interno.md) | Algoritmo principal |
| 07 | [Limites inferiores e experimentos](./turboquant/07-limites-inferiores-e-experimentos.md) | Validação empírica |
| — | [paper.pt-BR.md](./turboquant/paper.pt-BR.md) | Tradução comentada do paper |
| — | [2504.19874v1.pdf](./turboquant/2504.19874v1.pdf) | PDF original |

---

## 🛤️ Trilhas de leitura sugeridas

Escolha sua trilha de acordo com o seu objetivo:

| Perfil | Trilha sugerida |
|--------|-----------------|
| 🆕 **Iniciante em LLMs** | 01 → 02 → 03 → 04 → 05 |
| 📐 **Curioso por TurboQuant** | 03 → 05 → 06 → 06-DEEP → série [`turboquant/`](./turboquant/) |
| ⚙️ **Engenheiro de inferência** (vLLM/llama.cpp/MLX) | 03 → 04 → 07 → 08 → 10 → 11 |
| 🎓 **Quem treina/alinha modelos** | 01 → 04 → 09 → 10 → 16 → 18 |
| 🔍 **Quem constrói RAG** | 01 → 03 → 12 → 13 → 11 |
| 🤖 **Quem constrói agentes** | 11 → 13 → 14 → 18 → 16 |
| 👨‍💻 **Quem usa coding agents** (Cursor, Antigravity, Aider) | 19 → 14 → 16 |
| 🎨 **Quem trabalha com multimodal/voz/vídeo** | 01 → 12 → 17 → 11 |
| 📊 **Avaliação em produção** | 15 → 13 → 14 → 18 |
| 🛡️ **Segurança e compliance** | 16 → 14 → 19 → 09 |
| 🏠 **Rodar Qwen / Kimi K2 / Gemma / DeepSeek em casa** | 04 → 05 → [`serie-modelos-open-2026/`](./serie-modelos-open-2026/) → [`serie-inferencia-local/`](./serie-inferencia-local/) |
| 🦙 **Dominar `llama.cpp`** | 04 → [`serie-inferencia-local/01`](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md) → 05-DEEP → 11 |
| 🍎 **Dev Mac (Apple Silicon)** | [`serie-inferencia-local/02`](./serie-inferencia-local/02-mlx-mac-silicon-mlx-lm-mlx-vlm-fine-tune.md) → [`serie-inferencia-local/03`](./serie-inferencia-local/03-ollama-lmstudio-openwebui-jan-msty.md) → 19 |
| 💰 **Hardware caseiro / pequeno time** | [`serie-inferencia-local/04`](./serie-inferencia-local/04-hardware-builds-mac-pc-ddr5-rtx-3090-4090-mi300x-amador.md) → 10 → [`serie-inferencia-local/01`](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md) |
| 🧮 **Fundamentos matemáticos antes dos DEEPs** | [`serie-llm-math/01–03`](./serie-llm-math/) em ordem antes de 02-DEEP/06-DEEP/09 |
| 🔬 **Pesquisador / acadêmico** | Posts 01–08 em ordem + série [`turboquant/`](./turboquant/) |

---

## 📚 Documentos de apoio (Onda 1 — referência rápida)

| Documento | Conteúdo |
|-----------|----------|
| [`00-INDEX.md`](./00-INDEX.md) | **Índice mestre** da série, com mapa das ondas e como ler |
| [`GLOSSARY.md`](./GLOSSARY.md) | Termos e siglas com link para o post correspondente |
| [`CHEATSHEET.md`](./CHEATSHEET.md) | Fórmulas (KV), prefill vs decode, comandos de exemplo |
| [`DECISION-TREE.md`](./DECISION-TREE.md) | Árvores Mermaid: gargalo → técnica sugerida |
| [`BIBLIOGRAPHY.md`](./BIBLIOGRAPHY.md) | Papers e docs anotados em clusters temáticos |
| [`FAQ.md`](./FAQ.md) | Perguntas frequentes com ponte para os posts |
| [`TIMELINE.md`](./TIMELINE.md) | Linha do tempo 2017–2026 (marcos da série) |
| [`EXPANSION-ROADMAP.md`](./EXPANSION-ROADMAP.md) | Mapa original de expansões (Ondas 1–4) — **todas executadas** |
| [`ROADMAP-FUTURO.md`](./ROADMAP-FUTURO.md) | Mapa pós-Onda 4: Ondas 5+ (Otimização extrema, DEEP residuais, Labs, Case Studies, séries 2026: Voice/Video/On-device/Diffusion/Synthetic data/Compliance) |

---

## 🎬 Fontes-base

A série foi construída a partir de um vídeo central sobre **TurboQuant** e expandida em ondas:

| Documento | Conteúdo |
|-----------|----------|
| [`video-bX2hsi253QY-transcricao-pt.md`](./video-bX2hsi253QY-transcricao-pt.md) | Transcrição em PT-BR do vídeo de origem |
| [`video-bX2hsi253QY-turboquant-artigo-completo.md`](./video-bX2hsi253QY-turboquant-artigo-completo.md) | Síntese inicial em formato de artigo |
| [`turboquant/`](./turboquant/) | Tratamento formal do paper arXiv:2504.19874 |

---

## 🧩 Ondas de construção da série

A série foi construída em **ondas iterativas**, cada uma adicionando uma camada de profundidade:

```mermaid
flowchart LR
  O0[Onda 0<br/>Vídeo + Paper TurboQuant] --> O1[Onda 1<br/>Posts 01–08 + Apoio]
  O1 --> O2[Onda 2<br/>DEEPs 02/04/05/06/07/08]
  O2 --> O3[Onda 3<br/>Posts 09–19 horizontais]
  O3 --> O4[Onda 4<br/>Sub-séries hands-on]
  O4 --> O5[Onda 5+<br/>ROADMAP-FUTURO]
```

| Onda | Escopo | Status |
|------|--------|--------|
| **0** | Vídeo + paper TurboQuant + síntese inicial | ✅ |
| **1** | Posts 01–08 + documentos de apoio | ✅ |
| **2** | Apêndices DEEP (provas, tutoriais, walkthroughs) | ✅ |
| **3** | Posts 09–19 (treino, hardware, frameworks, RAG, agentes, segurança, multimodal, reasoning, coding) | ✅ |
| **4** | Sub-séries verticais hands-on (modelos open 2026, inferência local, math) | ✅ |
| **5+** | DEEPs residuais, Labs, Case Studies, séries 2026 | 📋 [planejado](./ROADMAP-FUTURO.md) |

---

## 🎨 Convenções editoriais

- **Tom**: técnico primeiro, **analogia** logo em seguida.
- **Diagramas**: Mermaid (`flowchart`, `sequenceDiagram`, `graph`, `timeline`).
- **Comparações**: tabelas com colunas claras (técnica, custo, quando usar, *trade-off*).
- **Referências**: cada post tem seção final com links oficiais (arXiv, docs, repositórios, blogs canônicos).
- **Idioma**: Português (BR), com termos técnicos preservados em inglês quando padrão da área.
- **Matemática**: LaTeX com delimitadores **compatíveis com GitHub MathJax** — `$...$` para inline e `$$...$$` para display (em parágrafo próprio, com linhas em branco antes/depois). Provas formais nos apêndices DEEP e na série [`turboquant/`](./turboquant/).
- **Código**: blocos com linguagem, comandos reproduzíveis e *flags* explicadas; preferência por exemplos `bash`/`python`/`C++`/`Swift`/`Mojo` conforme contexto.

> ℹ️ **Nota sobre fórmulas no GitHub**: o renderer Markdown do GitHub usa **MathJax apenas com cifrões** (`$...$` e `$$...$$`), **não** com os delimitadores LaTeX puros (`\(...\)` / `\[...\]`) que costumam vir do Pandoc/Quarto. Além disso, qualquer `$` solto (ex.: `R$ 5k`, `$60 M`) precisa ser **escapado como `\$`**, senão o GitHub interpreta pares de cifrões como fórmula e quebra o texto. Há dois scripts utilitários em `scripts/`:
>
> - [`fix-math-delimiters.py`](./scripts/fix-math-delimiters.py) — converte `\(...\)` → `$...$` e `\[...\]` → `$$...$$` (preservando blocos de código fenced).
> - [`escape-currency.py`](./scripts/escape-currency.py) — escapa cifrões de moeda (`R$`, `US$`, `AU$`, etc.); com `--loose-dollars` também escapa USD soltos (`$60 M`), protegendo automaticamente pares de math válidos.
>
> Uso típico (idempotente, seguro re-rodar):
>
> ```bash
> python3 scripts/fix-math-delimiters.py . --dry-run   # inspecionar
> python3 scripts/fix-math-delimiters.py .             # aplicar
> python3 scripts/escape-currency.py . --loose-dollars --dry-run
> python3 scripts/escape-currency.py . --loose-dollars
> ```

---

## 🔗 Ligação com o livro principal

Esta série é **complementar** ao livro *[Padrões de Design Agênticos](../readme.md)* e dialoga diretamente com:

| Capítulo do livro | Posts desta série |
|-------------------|-------------------|
| **MCP** (cap. 10) | [14](./14-agentes-tool-use-mcp-react-multiagent.md) (MCP em agentes), [19](./19-loop-agentico-coding-cursor-claude-code-aider-cline-opencode-antigravity-codex.md) (MCP em coding) |
| **Tool Use** (cap. 5) | [14](./14-agentes-tool-use-mcp-react-multiagent.md) (function calling, ReAct) |
| **Multi-Agent** (cap. 7) | [14](./14-agentes-tool-use-mcp-react-multiagent.md) (padrões multi-agent) |
| **RAG** (cap. 14) | [12](./12-embeddings-contrastive-mteb-matryoshka-multimodal.md), [13](./13-rag-chunking-retrieval-rerank-graph-agentic-eval.md) |
| **Reasoning** (cap. 17) | [18](./18-reasoning-models-o1-o3-r1-qwq-grpo-test-time-compute.md) (reasoning models) |
| **Safety** (cap. 18) | [16](./16-seguranca-jailbreaks-prompt-injection-alinhamento-redteam.md) (jailbreaks, red-team, governança) |
| **Evaluation** (cap. 19) | [15](./15-eval-mmlu-gpqa-arc-agi-llm-judge-arena-contamination.md) (benchmarks, judges, eval custom) |
| **Apêndice G — Coding Agents** | [19](./19-loop-agentico-coding-cursor-claude-code-aider-cline-opencode-antigravity-codex.md) (Cursor, Claude Code, Aider, Antigravity, Codex CLI, Devin) |
| **Apêndice F — Motores de Raciocínio** | [01](./01-arquitetura-transformer-decoder-llm.md), [09](./09-treinamento-pretraining-sft-dpo-grpo-rlhf.md), [18](./18-reasoning-models-o1-o3-r1-qwq-grpo-test-time-compute.md) |

---

## 📜 Como contribuir / corrigir

- A série é **viva**: erros, sugestões e novos pontos são bem-vindos.
- Se encontrar imprecisão técnica, abra issue/PR no repositório.
- Para propor novos posts, consulte [`ROADMAP-FUTURO.md`](./ROADMAP-FUTURO.md) — temas pré-priorizados aguardando autoria.

---

## ⚖️ Aviso

Os conteúdos referenciados (papers, documentação, repositórios, vídeos) pertencem aos seus respectivos autores. Esta série é uma **compilação didática** em PT-BR com objetivo educacional, citando todas as fontes originais.

---

*Série mantida em ondas iterativas — última atualização da estrutura: Onda 4 concluída (sub-séries hands-on). Próximas frentes em [`ROADMAP-FUTURO.md`](./ROADMAP-FUTURO.md).*
