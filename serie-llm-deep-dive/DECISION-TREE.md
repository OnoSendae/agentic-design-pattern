# Árvore de decisão — qual técnica usar?

Guia prático para **arquitetos** e **engenheiros de inferência**: começa pelo **sintoma** (o que dói) e chega a **famílias de solução**. Não substitui benchmark no **seu** hardware e modelo.

---

## 1. Qual é o gargalo principal?

```mermaid
flowchart TD
  A["Comece aqui: o que falha primeiro?"] --> B["VRAM insuficiente"]
  A --> C["Latência alta por requisição"]
  A --> D["Throughput baixo (req/s)"]
  A --> E["Qualidade / alucinação / drift"]

  B --> B1["Modelo não cabe OU contexto não cabe"]
  C --> C1["Prefill lento OU decode lento"]
  D --> D1["GPU subutilizada OU fila grande"]
  E --> E1["Ver dados, eval, RAG, não só 'mais quant'"]
```

### 1.1 VRAM insuficiente

```mermaid
flowchart TD
  V["VRAM"] --> V1["Pesos não cabem"]
  V --> V2["Pesos cabem, KV não cabe"]

  V1 --> P1["Quantizar pesos: INT4 AWQ/GPTQ/GGUF"]
  V1 --> P2["Modelo menor / MoE com menos experts ativos"]
  V1 --> P3["Offload parcial CPU — último recurso"]

  V2 --> K1["Reduzir max context / truncar histórico"]
  V2 --> K2["Quantizar KV: INT8/FP8/KIVI/KVQuant/TurboQuant*"]
  V2 --> K3["MQA/GQA/MLA já no modelo — trocar checkpoint"]
  V2 --> K4["Eviction: H2O, SnapKV, StreamingLLM"]
  V2 --> K5["RAG em vez de contexto infinito"]

  K2 --> Note["*TurboQuant: pesquisa / implementações em evolução"]
```

**Analogia:** VRAM é o **apartamento**. Pesos são os **móveis fixos**; KV é a **estante que cresce com cada livro novo** (token). Primeiro decida se o problema é móvel ou estante.

---

## 2. Latência: prefill lento vs decode lento

| Sintoma | Hipótese comum | Primeiras alavancas |
|---------|----------------|---------------------|
| **Prefill** longo (prompt enorme) | Compute de atenção em $O(T^2)$ para a fase | FlashAttention; chunked prefill; menos tokens no prompt (sumarizar, RAG); modelo menor na etapa de ingestão |
| **Decode** lento (geração longa) | Memory-bound lendo KV; poucos tokens/s | KV mais compacto; batch menor; speculative decoding; FP8 KV; kernels otimizados (TensorRT-LLM, vLLM) |
| **Ambos** | GPU fraca para o modelo | Quantização agressiva, modelo menor, ou hardware maior |

```mermaid
flowchart LR
  subgraph prefill["Prefill lento"]
    A1["Chunked prefill"]
    A2["FlashAttention"]
    A3["Prompt mais curto / RAG"]
  end
  subgraph decode["Decode lento"]
    B1["Speculative decoding"]
    B2["KV FP8 / quant KV"]
    B3["KV cache menor (GQA/MQA/MLA)"]
  end
```

---

## 3. Throughput: mais requisições por GPU

Ordem típica de impacto em **serving multiusuário**:

1. **Continuous batching** (vLLM, TGI, SGLang) — não deixar GPU ociosa entre requisições.
2. **PagedAttention** — reduzir fragmentação de KV.
3. **Prefix caching** — prompts idênticos compartilham KV.
4. **Disaggregated prefill/decode** — escalar cada fase separadamente (quando custo justificar).

```mermaid
flowchart TD
  T["Quero mais req/s"] --> T1["Continuous batching ativado?"]
  T1 -->|não| T2["vLLM / TGI / SGLang com fila"]
  T1 -->|sim| T3["Prefix caching para prompts repetidos"]
  T3 --> T4["Ainda não bate SLO?"]
  T4 --> T5["Separar prefill e decode em GPUs"]
```

---

## 4. Qualidade vs custo (trade-off explícito)

| Objetivo | Técnica típica | Custo |
|----------|----------------|--------|
| Máxima qualidade | FP16/BF16 pesos, KV alto, modelo grande | VRAM e $ |
| Produção equilibrada | INT4 pesos (AWQ/GPTQ), KV FP8 ou INT8 | Pequena queda de métricas |
| Edge / CPU | GGUF Q4_K_M, llama.cpp, contexto curto | Latência e qualidade variáveis |
| Fine-tuning barato | QLoRA NF4 + adapters | Tempo de treino, não só inferência |

**Regra:** nunca otimize **só** perplexidade em produção — use **tarefas reais** (RAG hit rate, JSON válido, etc.).

---

## 5. Quando NÃO quantizar mais

- Métricas de negócio caem mais que o ganho de VRAM.
- **Outliers** dominam (certas camadas/tensores) — tente **AWQ**, **SmoothQuant**, ou **rotations** (QuaRot/SpinQuant) antes de INT2 “na marra”.
- O problema é **dado** (prompt ruim, RAG errado) — técnica de compressão não corrige.

---

## 6. Mapa “técnica → post da série”

| Decisão | Onde aprofundar |
|---------|-----------------|
| Arquitetura, tokens, sampling | [01](./01-arquitetura-transformer-decoder-llm.md) |
| MHA/MQA/GQA/MLA, FlashAttention | [02](./02-attention-mha-mqa-gqa-mla-flashattention.md) |
| KV, vLLM, PagedAttention | [03](./03-kv-cache-anatomia-pagedattention-vllm.md) |
| Quant pesos | [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md) |
| Quant KV | [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md) |
| TurboQuant | [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md) |
| RoPE, YaRN, Ring, Mamba | [07](./07-contexto-longo-rope-yarn-ring-streaming.md) |
| Speculative, MoE, sparsity | [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md) |

---

## 7. Diagrama único (visão de sistema)

```mermaid
flowchart TB
  subgraph app["Aplicação"]
    U["Usuário / API"]
  end
  subgraph serving["Serving"]
    LB["Roteamento / fila"]
    V["vLLM / TGI / SGLang / TRT-LLM"]
  end
  subgraph model["Modelo"]
    W["Pesos quantizados"]
    KV["KV cache"]
  end
  subgraph hw["Hardware"]
    G["GPU VRAM + bandwidth"]
  end
  U --> LB --> V
  V --> W
  V --> KV
  W --> G
  KV --> G
```

---

*Onda 1 — revisar após novos posts (RAG dedicado, eval, agents) na Onda 3 planejada.*
