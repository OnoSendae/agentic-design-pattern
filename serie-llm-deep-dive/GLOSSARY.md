# Glossário — Série LLMs em Profundidade

Definições curtas em **Português (BR)**; termos em inglês mantidos quando são padrão da literatura. Cada entrada aponta para o post da série onde o conceito é desenvolvido.

**Convenção:** *itálico* = termo em inglês; **negrito** = conceito-chave.

---

## A

| Termo | Definição | Post |
|-------|-----------|------|
| **ALiBi** (*Attention with Linear Biases*) | Viés posicional por distância (sem embedding posicional explícito); usado em alguns modelos com janela longa. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **Atenção** (*attention*) | Mecanismo que combina vetores **Q**, **K**, **V** para ponderar posições anteriores na sequência. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **Atenção causal / mascarada** | Atenção em que cada posição só “vê” tokens anteriores (geração autoregressiva). | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **Autocompletar autoregressivo** | Modelo prevê o próximo token condicionado a todos os anteriores; repete até critério de parada. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **AWQ** (*Activation-aware Weight Quantization*) | PTQ que protege pesos “sensíveis” às magnitudes de ativação; comum em INT4 servível. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |

## B

| Termo | Definição | Post |
|-------|-----------|------|
| **Beam search** | Busca em árvore mantendo *k* hipóteses; menos usada em chat criativo, mais em tradução/formal. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **bitsandbytes** | Biblioteca (HF) para carregar pesos em INT8/INT4 com *outliers* em FP16 (LLM.int8, NF4). | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |
| **Block table** (vLLM) | Tabela que mapeia sequências lógicas para blocos físicos de KV na **PagedAttention**. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |

## C

| Termo | Definição | Post |
|-------|-----------|------|
| **CacheGen** | Codificação/compressão de KV para transferência ou armazenamento (visão de “codec” do cache). | [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |
| **Continuous batching** | Replanejamento dinâmico de requisições em fila (Orca/vLLM) para melhor utilização da GPU. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **Contexto** (*context*) | Janela de tokens visível ao modelo (prompt + histórico relevante); limitada por arquitetura e memória. | [01](./01-arquitetura-transformer-decoder-llm.md), [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |

## D

| Termo | Definição | Post |
|-------|-----------|------|
| **Decode** | Fase em que se gera **um token de cada vez**, lendo o **KV cache** acumulado. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **Decoder-only** | Arquitetura só com blocos *decoder* (GPT, Llama, Qwen, Mistral); padrão em LLMs de texto. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **Distillation** (*knowledge distillation*) | Treinar modelo menor (“student”) para imitar um maior (“teacher”). | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |
| **Disaggregated serving** | Separar **prefill** e **decode** em processos/GPUs distintos (Splitwise, DistServe, etc.). | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |

## E

| Termo | Definição | Post |
|-------|-----------|------|
| **Embedding** | Vetor numérico que representa token ou posição no espaço de significados. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **Encoder–decoder** | Arquitetura com encoder + decoder (T5, BART); diferente de decoder-only. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **Expert** (MoE) | Sub-rede especializada; o *router* escolhe *top-k* experts por token. | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |

## F

| Termo | Definição | Post |
|-------|-----------|------|
| **FFN** (*Feed-Forward Network*) | MLP por posição após a atenção (ex.: SwiGLU em Llama). | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **FlashAttention** | Implementação I/O-aware da atenção (tiling em SRAM) mantendo resultado matematicamente equivalente ao *naive*. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **FP8 / BF16 / FP16** | Formatos de ponto flutuante; FP8 é comum em inferência acelerada em GPUs recentes. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |

## G

| Termo | Definição | Post |
|-------|-----------|------|
| **GGUF** | Formato de arquivo para modelos quantizados consumidos pelo **llama.cpp** e derivados. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |
| **GQA** (*Grouped-Query Attention*) | Várias cabeças de query compartilham o mesmo grupo de KV; meio-termo entre MHA e MQA. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **GPTQ** | Quantização pós-treino guiada por Hessiana (segunda ordem) para INT4/INT3 em pesos. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |

## H

| Termo | Definição | Post |
|-------|-----------|------|
| **H2O** (*Heavy Hitter Oracle*) | Evicção de KV baseada em “tokens importantes” no histórico. | [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |

## I–J

| Termo | Definição | Post |
|-------|-----------|------|
| **INT4 / INT8** | Inteiros de 4 ou 8 bits; base de muitas quantizações de peso e de KV. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md), [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |

## K

| Termo | Definição | Post |
|-------|-----------|------|
| **K, V** (*Key*, *Value*) | Projeções da atenção; **K** e **V** por posição são o que normalmente se **armazena** no cache. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md), [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **KIVI** | Quantização de KV assimétrica (ex.: K *per-channel*, V *per-token*) para reduzir memória. | [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |
| **KV cache** | Memória dos tensores **K** e **V** já computados para tokens passados; evita recomputar atenção. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **KVQuant** | Quantização de KV com pipeline *pre-RoPE*, quantização não uniforme e componente denso+esparso. | [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |

## L

| Termo | Definição | Post |
|-------|-----------|------|
| **LayerNorm / RMSNorm** | Normalização estabiliza treinamento; **RMSNorm** é variante comum em LLMs (Llama). | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **Lloyd–Max** | Quantizador ótimo para minimizar MSE dada distribuição (base do **Algoritmo 1** no TurboQuant). | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md) |

## M

| Termo | Definição | Post |
|-------|-----------|------|
| **MHA** (*Multi-Head Attention*) | Várias cabeças em paralelo; cada uma com seus Q, K, V. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **MLA** (*Multi-head Latent Attention*) | Comprime KV em espaço latente (DeepSeek-V2/V3); reduz drasticamente memória de KV por token. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md), [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **MoE** (*Mixture of Experts*) | Camadas com múltiplos experts + roteador; *active params* ≪ *total params*. | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |
| **MQA** (*Multi-Query Attention*) | Uma única cabeça KV compartilhada por todas as cabeças de Q. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |

## N–O

| Termo | Definição | Post |
|-------|-----------|------|
| **NF4** | Tipo numérico de 4 bits normalizado (QLoRA); boa relação qualidade/tamanho para fine-tuning. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |
| **Outlier** (ativação) | Valores extremos em ativações/pesos que degradam quantização grosseira se ignorados. | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md), [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |

## P

| Termo | Definição | Post |
|-------|-----------|------|
| **PagedAttention** | Alocação de KV em blocos não contíguos (como páginas de SO), reduzindo fragmentação. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **Polar quantization** | Representar vetores por **módulo** + **direção** (ângulos) antes de quantizar — núcleo intuitivo do TurboQuant. | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md) |
| **Prefix caching / APC** | Reutilizar KV de prefixos idênticos entre requisições (RadixAttention, vLLM, etc.). | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **Prefill** | Fase em que o modelo processa o prompt (muitos tokens de entrada) e materializa KV inicial. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **PTQ** (*Post-Training Quantization*) | Quantizar depois do treino, sem retreinar (GPTQ, AWQ, etc.). | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |

## Q

| Termo | Definição | Post |
|-------|-----------|------|
| **Q** (*Query*) | Projeção da posição atual na atenção; não costuma ser armazenada no **KV cache** da mesma forma que K/V. | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| **QJL** (*Quantized Johnson–Lindenstrauss*) | Embeddings quantizados preservando distâncias aproximadamente; antecessor conceitual próximo ao TurboQuant. | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md) |
| **Quantização** | Mapear valores de alta precisão para menos bits (com erro controlado ou não). | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |

## R

| Termo | Definição | Post |
|-------|-----------|------|
| **Ring Attention** | Particionar sequência longa em GPUs em anel, trocando blocos de KV/atenção. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **RoPE** (*Rotary Position Embedding*) | Embeddings posicionais por rotação em pares de dimensões; padrão em Llama/Qwen/Mistral. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **Router** (MoE) | Rede que escolhe quais experts ativar por token. | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |
| **RAG** (*Retrieval-Augmented Generation*) | Completar contexto com documentos recuperados de um índice externo. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |

## S

| Termo | Definição | Post |
|-------|-----------|------|
| **Shannon lower bound** (no paper TurboQuant) | Limite de taxa–distorção ligado a $4^{-b}$ para certas quantizações esféricas. | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md), [../turboquant-docs/04-shannon-lower-bound.md](turboquant/04-shannon-lower-bound.md) |
| **Sink tokens** (*attention sinks*) | Primeiros tokens que concentram atenção residual; base do **StreamingLLM** para janelas longas. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **Sliding Window Attention** | Atenção restrita a uma janela local (Mistral, etc.). | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md), [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **Speculative decoding** | Modelo pequeno propõe tokens; modelo grande verifica em lote — pode aumentar tokens/s. | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |
| **SSM** (*State Space Model*) | Família incluindo **Mamba**: recorrência com treino paralelizável em sequência. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| **StreamingLLM** | Manter *sink tokens* + janela recente para contexto longo com KV limitado. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |

## T

| Termo | Definição | Post |
|-------|-----------|------|
| **Tensor parallel / Pipeline parallel** | Particionar modelo em várias GPUs (camadas ou fatias de tensor). | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) (contexto) |
| **Token** | Unidade atômica do vocabulário do modelo (subpalavra, não necessariamente palavra humana). | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **Top-k / Top-p** (*nucleus sampling*) | Estratégias de amostragem que restringem o espaço de próximos tokens. | [01](./01-arquitetura-transformer-decoder-llm.md) |
| **TurboQuant** | Algoritmo de compressão de vetores (KV) via quantização polar + ajustes; paper **arXiv:2504.19874**. | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md), [../turboquant-docs/](turboquant) |

## V–Z

| Termo | Definição | Post |
|-------|-----------|------|
| **VRAM** | Memória de vídeo da GPU; onde residem pesos e KV em inferência local acelerada. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **vLLM** | Motor de inferência com **PagedAttention** e *continuous batching*. | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| **YaRN** | Extensão de **RoPE** por reescala/interpolação de frequências para janelas maiores que o treino. | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |

---

## Siglas frequentes

| Sigla | Significado |
|-------|-------------|
| APC | *Automatic Prefix Caching* |
| BPE | *Byte-Pair Encoding* |
| FFN | *Feed-Forward Network* |
| GQA | *Grouped-Query Attention* |
| KV | *Key-Value* (cache de atenção) |
| MHA | *Multi-Head Attention* |
| MLA | *Multi-head Latent Attention* |
| MoE | *Mixture of Experts* |
| MQA | *Multi-Query Attention* |
| PTQ | *Post-Training Quantization* |
| QAT | *Quantization-Aware Training* |
| RAG | *Retrieval-Augmented Generation* |
| RoPE | *Rotary Position Embedding* |
| SSM | *State Space Model* |
| SWA | *Sliding Window Attention* |

---

*Última atualização: Onda 1 — documento vivo; sugerimos PR/issue para novos termos.*
