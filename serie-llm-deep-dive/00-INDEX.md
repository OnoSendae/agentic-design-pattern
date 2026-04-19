# Série: LLMs em Profundidade — Da Atenção ao TurboQuant e Além

Série didática e progressiva de posts técnicos, em estilo *blog técnico / nota de divulgação científica*. Cada post combina **fundamento técnico**, **analogias com o mundo real**, **diagramas Mermaid**, **tabelas comparativas** e **referências oficiais** (papers, documentação, repositórios).

A jornada parte da arquitetura básica de uma LLM, passa pelos gargalos de inferência (atenção, KV cache, memória), aprofunda em **quantização** (pesos e KV), chega ao **TurboQuant** com rigor, e segue **além**: contexto longo, hardware, frameworks e técnicas complementares (sparsity, speculative decoding, MoE).

---

## Mapa da série

| # | Post | Tema central | Pré-requisito |
|---|------|--------------|---------------|
| 01 | [Arquitetura Transformer & LLMs decoder-only](./01-arquitetura-transformer-decoder-llm.md) | Tokens, embeddings, blocos decoder, geração autoregressiva | — |
| 02 | [Atenção em profundidade: MHA, MQA, GQA, MLA, FlashAttention](./02-attention-mha-mqa-gqa-mla-flashattention.md) | Variantes de atenção, custos quadráticos, FlashAttention 1/2/3 | 01 |
| 03 | [KV cache: anatomia, custos e PagedAttention/vLLM](./03-kv-cache-anatomia-pagedattention-vllm.md) | Por que existe, fórmula de tamanho, fragmentação, vLLM | 01, 02 |
| 04 | [Quantização de pesos: GPTQ, AWQ, GGUF, bitsandbytes](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) | INT8, INT4, NF4, formatos GGUF/EXL2, calibração | 01 |
| 05 | [Quantização de KV cache: KIVI, KVQuant, CacheGen](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) | Por que KV é difícil; outliers; per-channel/per-token | 03, 04 |
| 06 | [TurboQuant em profundidade: polar, JL e Lloyd–Max](./06-turboquant-deep-dive-polar-jl-lloydmax.md) | Quantização não-enviesada, MSE/IP, two-stage IP, cota $4^{-b}$ | 04, 05 |
| 07 | [Contexto longo: RoPE, YaRN, Ring/StreamingLLM, Mamba](./07-contexto-longo-rope-yarn-ring-streaming.md) | Estender janela, escalar atenção, alternativas ao Transformer | 02, 03 |
| 08 | [Além da quantização: sparsity, speculative decoding, MoE, distillation](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) | Outras alavancas para acelerar/reduzir custo | 01–06 |

### Apêndices DEEP (Onda 2 — aprofundamentos verticais)

| Apêndice | Foco | Ligado a |
|----------|------|----------|
| [02-DEEP](./02-DEEP-online-softmax-flashattention.md) | Online softmax (prova por indução) + FlashAttention 1/2/3/4 + Triton kernel + backward | Post 02 |
| [04-DEEP](./04-DEEP-gptq-qlora-handson.md) | GPTQ algoritmo passo a passo (OBQ → Cholesky) + QLoRA receita completa em código | Post 04 |
| [05-DEEP](./05-DEEP-outliers-kv-quant-tutorial.md) | Análise empírica de outliers em K (notebook) + tutorial vLLM/llama.cpp/MLX/TRT-LLM | Post 05 |
| [06-DEEP](./06-DEEP-mlx-turboquant-walkthrough.md) | Walkthrough TurboQuant em MLX + provas dos Teoremas 1/2 + censo de implementações comunitárias | Post 06 |
| [07-DEEP](./07-DEEP-yarn-ssm-math.md) | RoPE → PI → NTK → YaRN → LongRoPE matemático + SSM/Mamba/Mamba-2/Jamba completos | Post 07 |
| [08-DEEP](./08-DEEP-speculative-math-eagle.md) | Prova de equivalência distribucional do speculative + EAGLE-1/2/3 + tree decoding + MTP | Post 08 |

### Posts horizontais (Onda 3 — novas frentes)

| # | Post | Tema central | Pré-requisito |
|---|------|--------------|---------------|
| 09 | [Treinamento de LLMs: pretraining, SFT, DPO, GRPO, RLHF](./09-treinamento-pretraining-sft-dpo-grpo-rlhf.md) | Pipeline completo de treino: do next-token-prediction ao alinhamento por RL | 01, 04 |
| 10 | [Hardware para LLMs: H100, H200, B200, MI300X, TPU, Apple Silicon, Groq](./10-hardware-h100-h200-b100-b200-mi300x-tpu-apple-groq.md) | Chips, memória HBM, NVLink, datacenters, edge | 01–03 |
| 11 | [Frameworks de inferência comparados: vLLM, SGLang, TRT-LLM, TGI, llama.cpp, MLX, Ollama, KTransformers](./11-frameworks-vllm-sglang-trtllm-tgi-llamacpp-mlx-ollama.md) | Qual usar e como configurar para produção | 03, 04, 05 |
| 12 | [Embeddings: contrastive, MTEB, Matryoshka, multilingue e multimodal](./12-embeddings-contrastive-mteb-matryoshka-multimodal.md) | Anatomia, treinamento e taxonomia dos modelos de embedding | 01 |
| 13 | [RAG em profundidade: chunking, retrieval híbrido, reranking, GraphRAG, agentic, eval](./13-rag-chunking-retrieval-rerank-graph-agentic-eval.md) | Arquitetura completa de RAG produção 2026 | 06 (PQ↔TurboQuant), 12 |
| 14 | [Agentes, tool use, MCP, ReAct e multi-agent](./14-agentes-tool-use-mcp-react-multiagent.md) | Agentes LLM gerais: function calling, MCP, padrões de loop, multi-agent | 11, 13 |
| 15 | [Avaliação de LLMs: MMLU, GPQA, ARC-AGI, LLM-as-judge, Arena, contaminação](./15-eval-mmlu-gpqa-arc-agi-llm-judge-arena-contamination.md) | Benchmarks, judges, arenas e eval custom de produção | — |
| 16 | [Segurança: jailbreaks, prompt injection, alinhamento, red-team, governança](./16-seguranca-jailbreaks-prompt-injection-alinhamento-redteam.md) | Superfície de ataque, defesas em camadas, RSP/EU AI Act/PL 2338 | 09 (alignment) |
| 17 | [Multimodalidade: VLM, áudio, vídeo, omni — CLIP, LLaVA, Qwen-VL, Gemini](./17-multimodalidade-vlm-audio-video-omni-clip-llava-qwen-vl-gemini.md) | Arquitetura, treino e capabilities de modelos multimodais geradores | 01–03 |
| 18 | [Reasoning models: o1, o3, R1, QwQ, GRPO e test-time compute](./18-reasoning-models-o1-o3-r1-qwq-grpo-test-time-compute.md) | CoT escalado por RL e tempo de inferência | 09 |
| 19 | [Loop agêntico de codificação: Cursor, Antigravity, Claude Code, Aider, Cline, OpenCode, Codex CLI, Devin](./19-loop-agentico-coding-cursor-claude-code-aider-cline-opencode-antigravity-codex.md) | Coding agents IDE/CLI/background, edit formats, MCP em código, vibe coding | 14 |

### Sub-séries especializadas (Onda 4 — verticais hands-on)

#### [`serie-modelos-open-2026/`](./serie-modelos-open-2026/) — Hands-on dos modelos open-weights chineses e Gemma

| # | Post | Foco |
|---|------|------|
| 01 | [Qwen 3 hands-on: dense, MoE, Coder, VL, Omni](./serie-modelos-open-2026/01-qwen3-hands-on-dense-moe-coder-vl-omni.md) | Família Alibaba completa: variantes, quantização, serving, fine-tune, casos de uso PT-BR |
| 02 | [Kimi K2+ MoE 1T agentic hands-on](./serie-modelos-open-2026/02-kimi-k2-moe-1t-agentic-handson.md) | MoE 1T da Moonshot AI: KTransformers em consumer, agentic com SWE-bench/τ-bench |
| 03 | [Gemma 3 e 4 hands-on: multimodal & on-device](./serie-modelos-open-2026/03-gemma-3-4-handson-multimodal-ondevice.md) | Família Google: 1B/4B/12B/27B, vision nativa, edge Android/iOS, ShieldGemma, PaliGemma |
| 04 | [DeepSeek V3, R1, R2 hands-on: MLA, MTP, FP8 training](./serie-modelos-open-2026/04-deepseek-v3-r1-r2-handson-mla-mtp.md) | 671B MoE eficiente: SGLang+FlashMLA, KTransformers, R1-Distill, R2 reasoning |

#### [`serie-inferencia-local/`](./serie-inferencia-local/) — Rodar LLMs em casa, no laptop, no servidor caseiro

| # | Post | Foco |
|---|------|------|
| 01 | [llama.cpp deep workflow: imatrix, quantize, serve](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md) | **Pipeline ponta-a-ponta**: build → HF → GGUF → imatrix → quantize → bench → llama-server → clientes |
| 02 | [MLX no Mac Apple Silicon: mlx-lm, mlx-vlm, fine-tune, distributed](./serie-inferencia-local/02-mlx-mac-silicon-mlx-lm-mlx-vlm-fine-tune.md) | UMA, mlx-lm/mlx-vlm/mlx-audio, LoRA local, cluster Thunderbolt 5 |
| 03 | [Ollama, LM Studio, Open WebUI, Jan, Msty, AnythingLLM](./serie-inferencia-local/03-ollama-lmstudio-openwebui-jan-msty.md) | UX-first: Modelfile, GUI, RAG built-in, MCP, multi-user, voice |
| 04 | [Hardware builds: Mac, PC DDR5, RTX 3090/4090/5090, Pro 6000, MI300X](./serie-inferencia-local/04-hardware-builds-mac-pc-ddr5-rtx-3090-4090-mi300x-amador.md) | 10 builds R\$ 5k → R\$ 400k, ROI vs API hosted, energia/ruído BR |

#### [`serie-llm-math/`](./serie-llm-math/) — Matemática essencial para entender LLMs

| # | Post | Foco |
|---|------|------|
| 01 | [Álgebra linear essencial: vetores, matrizes, SVD, normas](./serie-llm-math/01-algebra-linear-essencial-vetores-matrizes-svd-norms.md) | Da norma L2 ao SVD, com NumPy e aplicação direta a Transformer/LoRA/MLA |
| 02 | [Cálculo, gradientes, autograd e backpropagation](./serie-llm-math/02-calculo-derivadas-gradientes-autograd-backprop.md) | Regra da cadeia matricial, micrograd-style, Adam/AdamW/Lion/Muon, mixed precision, checkpointing |
| 03 | [Probabilidade e info theory: entropia, KL, perplexidade, rate-distortion](./serie-llm-math/03-probabilidade-info-theory-entropia-kl-perplexidade-rate-distortion.md) | Cross-entropy loss, KL no RLHF/DPO/GRPO, sampling (top-k/p/min-p), R(D) e TurboQuant |

---

## Como ler

- **Iniciante** em LLMs internas: leia 01 → 02 → 03, depois pule para 04/05.
- **Curioso por TurboQuant**: 03 → 05 → 06 → 06-DEEP → série formal [`turboquant/`](./turboquant/).
- **Engenheiro de inferência** (LM Studio, vLLM, llama.cpp, MLX): 03 → 04 → 07 → 08 → **10 → 11**.
- **Quem treina/alinha modelos**: 01 → 04 → **09 → 10 → 16 → 18**.
- **Quem constrói RAG**: 01 → 03 → **12 → 13 → 11**.
- **Quem constrói agentes**: 11 → 13 → **14 → 18 → 16** (segurança).
- **Quem usa coding agents** (Cursor, Antigravity, Claude Code, Aider): leia direto **19 → 14 → 16**.
- **Quem trabalha com multimodal/voz/vídeo**: 01 → 12 → **17 → 11**.
- **Quem precisa avaliar LLMs em produção**: **15 → 13 (RAG eval) → 14 (agent eval) → 18 (reasoning eval)**.
- **Quem cuida de segurança/compliance**: **16 → 14 → 19 → 09 (alignment)**.
- **Quem quer rodar Qwen / Kimi K2 / Gemma / DeepSeek em casa**: 04 → 05 → **[serie-modelos-open-2026/](./serie-modelos-open-2026/) → [serie-inferencia-local/](./serie-inferencia-local/)**.
- **Quem quer dominar `llama.cpp` ponta-a-ponta**: 04 → **[serie-inferencia-local/01](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md)** → 05-DEEP → 11.
- **Quem é dev Mac**: **[serie-inferencia-local/02 (MLX)](./serie-inferencia-local/02-mlx-mac-silicon-mlx-lm-mlx-vlm-fine-tune.md)** → [serie-inferencia-local/03](./serie-inferencia-local/03-ollama-lmstudio-openwebui-jan-msty.md) → 19.
- **Quem está montando hardware caseiro/empresa pequena**: **[serie-inferencia-local/04](./serie-inferencia-local/04-hardware-builds-mac-pc-ddr5-rtx-3090-4090-mi300x-amador.md)** → 10 → [serie-inferencia-local/01](./serie-inferencia-local/01-llamacpp-deep-workflow-imatrix-quantize-serve.md).
- **Quem precisa de fundamentos matemáticos para a série**: **[serie-llm-math/01–03](./serie-llm-math/)** em ordem antes de 02-DEEP/06-DEEP/09.
- **Pesquisador**: leia em ordem; consulte também a série acadêmica em [`turboquant/`](./turboquant/) para o tratamento formal do paper.

## Convenções

- **Tom**: técnico primeiro, **analogia** logo em seguida.
- **Diagramas**: Mermaid (`flowchart`, `sequenceDiagram`, `graph`).
- **Comparações**: tabelas com colunas claras (técnica, custo, quando usar, *trade-off*).
- **Referências**: cada post tem seção final com links oficiais (arXiv, docs, repositórios, blogs canônicos).
- **Idioma**: Português (BR), com termos técnicos preservados em inglês quando padrão da área.

## Fontes-base

- Transcrição original do vídeo: [`./video-bX2hsi253QY-transcricao-pt.md`](./video-bX2hsi253QY-transcricao-pt.md)
- Síntese inicial: [`./video-bX2hsi253QY-turboquant-artigo-completo.md`](./video-bX2hsi253QY-turboquant-artigo-completo.md)
- Tratamento formal (paper TurboQuant arXiv:2504.19874): [`./turboquant/`](./turboquant/)

---

## Documentos de apoio (Onda 1 — referência rápida)

| Documento | Conteúdo |
|-----------|----------|
| [GLOSSARY.md](./GLOSSARY.md) | Termos e siglas com link para o post correspondente |
| [CHEATSHEET.md](./CHEATSHEET.md) | Fórmulas (KV), prefill vs decode, comandos de exemplo |
| [DECISION-TREE.md](./DECISION-TREE.md) | Árvores Mermaid: gargalo → técnica sugerida |
| [BIBLIOGRAPHY.md](./BIBLIOGRAPHY.md) | Papers e docs anotados em clusters temáticos |
| [FAQ.md](./FAQ.md) | Perguntas frequentes com ponte para os posts |
| [TIMELINE.md](./TIMELINE.md) | Linha do tempo 2017–2026 (marcos da série) |
| [EXPANSION-ROADMAP.md](./EXPANSION-ROADMAP.md) | Mapa original de expansões (Ondas 1–4) — **todas executadas** |
| [ROADMAP-FUTURO.md](./ROADMAP-FUTURO.md) | **Mapa pós-Onda 4**: Ondas 5+ planejadas (Otimização extrema, DEEP residuais, Labs, Case Studies, séries 2026: Voice/Video/On-device/Diffusion/Synthetic data/Compliance) |
