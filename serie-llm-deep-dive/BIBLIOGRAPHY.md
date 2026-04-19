# Bibliografia anotada — Série LLMs em Profundidade

Lista **curada** de referências citadas ou alinhadas com os posts **01–08**. Cada entrada: **1–2 frases** sobre o que o trabalho provê + link canônico (arXiv, ACM, ou repositório oficial).

**Como usar:** leia na ordem dos *clusters* se for iniciante; salte para o cluster do seu post.

---

## Cluster A — Fundamentos do Transformer

| Referência | Anotação | Link |
|------------|----------|------|
| Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 | Define **Transformer**, **MHA**, **FFN**, **positional encoding** sinusoidal; base de tudo na série. | https://arxiv.org/abs/1706.03762 |
| Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1), 2018 | Pré-treino decoder-only em linguagem; ponte para GPT-2/3. | https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf |
| Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 | Escala e zero-shot; populariza stack decoder-only. | https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS 2020 | **In-context learning**; escala de parâmetros. | https://arxiv.org/abs/2005.14165 |
| Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 | **RMSNorm** usado em Llama e família. | https://arxiv.org/abs/1910.07467 |

---

## Cluster B — Posição, contexto longo, eficiência de atenção

| Referência | Anotação | Link |
|------------|----------|------|
| Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 | Base teórica do **RoPE** (popularizado em GPT-Neo, depois Llama). | https://arxiv.org/abs/2104.09864 |
| Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, 2021 | **ALiBi** — viés linear por distância. | https://arxiv.org/abs/2108.12409 |
| Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023 | **YaRN** — extensão de contexto via RoPE. | https://arxiv.org/abs/2309.00071 |
| Beltagy et al., *Longformer: The Long-Document Transformer*, 2020 | Atenção esparsa por janelas + global. | https://arxiv.org/abs/2004.05150 |
| Zaheer et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020 | Atenção esparsa com blocos aleatórios + global. | https://arxiv.org/abs/2007.14062 |
| Child et al., *Generating Long Sequences with Sparse Transformers*, 2019 | Padrões esparsos em atenção (precursor conceitual). | https://arxiv.org/abs/1904.10509 |
| Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*, 2023 | **Ring Attention** — paralelismo de sequência longa. | https://arxiv.org/abs/2310.01889 |
| Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023 | **StreamingLLM** — *sink tokens* + janela. | https://arxiv.org/abs/2309.17453 |
| Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022 | **FlashAttention** — tiling em SRAM. | https://arxiv.org/abs/2205.14135 |
| Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023 | Melhorias de paralelismo. | https://arxiv.org/abs/2307.08691 |
| Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, 2024 | FP8, async, Hopper. | https://arxiv.org/abs/2407.08608 |

---

## Cluster C — Variantes de atenção (KV)

| Referência | Anotação | Link |
|------------|----------|------|
| Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 | **MQA** — uma cabeça KV. | https://arxiv.org/abs/1911.02150 |
| Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 | **GQA** — grupos de KV compartilhados. | https://arxiv.org/abs/2305.13245 |
| DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024 | Introduz **MLA** (KV latente). | https://arxiv.org/abs/2405.04434 |
| DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 | Escala MoE + MLA; referência de sistema. | https://arxiv.org/abs/2412.19437 |

---

## Cluster D — Serving, KV, batching

| Referência | Anotação | Link |
|------------|----------|------|
| Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 | **Iteration-level scheduling** / continuous batching. | https://www.usenix.org/conference/osdi22/presentation/yu |
| Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 | **PagedAttention** + **vLLM**. | https://arxiv.org/abs/2309.06180 |
| Agrawal et al., *Sarathi-Serve: Efficient LLM Inference by Chunking and Pipelining Prefills*, OSDI 2024 | Chunked prefill + pipeline. | https://arxiv.org/abs/2408.14469 |
| Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, 2024 | Prefill/decode desagregados. | https://arxiv.org/abs/2401.09669 |
| Hu et al., *MemServe: Context Caching for Disaggregated LLM Serving in Memory Clusters*, 2025 | Contexto/memória em cluster (referência de tendência). | Buscar versão arXiv atual se usar em trabalho formal |

---

## Cluster E — Quantização de pesos e treino eficiente

| Referência | Anotação | Link |
|------------|----------|------|
| Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, NeurIPS 2022 | INT8 com *outliers* em FP16. | https://arxiv.org/abs/2208.07339 |
| Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, 2022 | Migra dificuldade de ativação para peso. | https://arxiv.org/abs/2211.10438 |
| Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023 | **GPTQ** — quantização com informação de segunda ordem. | https://arxiv.org/abs/2210.17323 |
| Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024 | **AWQ** — protege pesos “salientes”. | https://arxiv.org/abs/2306.00978 |
| Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023 | **NF4** + adapters — fine-tuning barato. | https://arxiv.org/abs/2305.14314 |
| Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, NeurIPS 2024 | Rotações Hadamard + INT4. | https://arxiv.org/abs/2404.00456 |
| Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, 2024 | Rotações aprendidas para quantização. | https://arxiv.org/abs/2405.16406 |
| Jegou et al., *Product Quantization for Nearest Neighbor Search*, PAMI 2010 | **PQ** — base para compressão de vetores (ligação conceitual a KV). | https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf |

---

## Cluster F — Quantização de KV e compressão de cache

| Referência | Anotação | Link |
|------------|----------|------|
| Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024 | K/V assimétricos, *tuning-free*. | https://arxiv.org/abs/2402.02750 |
| Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference*, 2024 | **KVQuant** — *pre-RoPE*, não uniforme, denso+esparso. | https://arxiv.org/abs/2401.18079 |
| Liu et al., *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*, SIGCOMM 2024 | **CacheGen** — codec para KV. | https://arxiv.org/abs/2404.14294 |
| Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, NeurIPS 2023 | Evicção por *heavy hitters*. | https://arxiv.org/abs/2306.14048 |
| Li et al., *SnapKV: LLMs Know What You are Looking for Before Generation*, 2024 | Seleção de KV por atenção observada. | https://arxiv.org/abs/2404.14469 |

---

## Cluster G — TurboQuant e vizinhos teóricos

| Referência | Anotação | Link |
|------------|----------|------|
| Jiang et al., *Training Large Embeddings for Private Record Linkage with Quantized Johnson-Lindenstrauss Transform*, AAAI 2025 | **QJL** — antecessor próximo. | https://arxiv.org/abs/2406.03482 |
| Zandieh et al., *TurboQuant: Towards Fast and Accurate Quantization for Billion-Scale Vectors*, arXiv 2025 | **TurboQuant** — quantização polar + cotas; alvo KV/embeddings. | https://arxiv.org/abs/2504.19874 |
| Série formal local | Derivations lema Beta, SLB, Teoremas — `../turboquant-docs/` | [INDEX](turboquant/INDEX.md) |

---

## Cluster H — Alternativas ao Transformer (SSM, híbrido)

| Referência | Anotação | Link |
|------------|----------|------|
| Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023 | **Mamba** — SSM seletivo. | https://arxiv.org/abs/2312.00752 |
| Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, 2024 | **Mamba-2** / SSD. | https://arxiv.org/abs/2405.21060 |
| Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*, 2024 | Híbrido Transformer + Mamba. | https://arxiv.org/abs/2403.19887 |

---

## Cluster I — Speculative decoding, MoE, sparsity, distillation

| Referência | Anotação | Link |
|------------|----------|------|
| Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2022 | **Speculative decoding** (draft + verify). | https://arxiv.org/abs/2211.17192 |
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023 | Amostragem especulativa (DeepMind). | https://arxiv.org/abs/2302.01318 |
| Cai et al., *Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads*, 2024 | Cabeças decodificadoras paralelas. | https://arxiv.org/abs/2401.10774 |
| Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024 | **EAGLE** — draft autoregressivo nas camadas. | https://arxiv.org/abs/2401.15077 |
| Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022 | **MoE** *switch* — roteamento simples. | https://arxiv.org/abs/2101.03961 |
| Jiang et al., *Mixtral of Experts*, 2024 | **Mixtral** 8×7B — MoE aberto. | https://arxiv.org/abs/2401.04088 |
| Frantar & Alistarh, *SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot*, 2023 | Podar pesos em um shot. | https://arxiv.org/abs/2301.00774 |
| Sun et al., *A Simple and Effective Pruning Approach for Large Language Models*, ICLR 2024 | **Wanda** — pruning por magnitude × ativação. | https://arxiv.org/abs/2306.11695 |
| Hinton et al., *Distilling the Knowledge in a Neural Network*, NIPS Deep Learning Workshop 2015 | Base de **distillation**. | https://arxiv.org/abs/1503.02531 |

---

## Cluster J — Documentação e ferramentas (não são papers, mas referência operacional)

| Recurso | Anotação | Link |
|---------|----------|------|
| vLLM | Servidor com PagedAttention; docs mudam por versão. | https://docs.vllm.ai |
| llama.cpp | Inferência GGUF; quantização e flags de KV. | https://github.com/ggerganov/llama.cpp |
| Hugging Face Transformers | API padrão de modelos e integração BnB. | https://huggingface.co/docs/transformers |
| TensorRT-LLM | Otimização NVIDIA; FP8/INT4 conforme geração GPU. | https://github.com/NVIDIA/TensorRT-LLM |
| MLX | Apple — arrays e treino em Apple Silicon. | https://github.com/ml-explore/mlx |

---

## Como expandir esta bibliografia

- Adicionar **cluster K — RAG / retrieval** (na Onda 3 planejada).
- Adicionar **cluster L — avaliação** (MMLU, HELM, etc.).
- Para cada post novo da série, incluir 5–10 entradas “**first class**” aqui.

---

*Onda 1 — revisar anualmente; arXiv IDs podem ganhar versões v2/v3.*
