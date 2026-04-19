# FAQ — Série LLMs em Profundidade

Respostas curtas com **ponte** para o post certo. Se a sua dúvida não estiver aqui, proponha inclusão no repositório.

---

## Fundamentos

### 1. O que é um “token” e por que não é igual a “palavra”?

**Token** é a unidade do vocabulário do modelo (geralmente **subpalavras** via BPE/SentencePiece). Uma palabra pode virar vários tokens; dois idiomas diferentes podem ter contagens diferentes para a mesma ideia.  
→ Post [01](./01-arquitetura-transformer-decoder-llm.md).

### 2. Por que quase todo modelo de chat é “decoder-only”?

Porque a tarefa principal é **gerar** texto condicionado ao contexto anterior; **encoder–decoder** brilha em tarefas com entrada/saída assimétricas (tradução, sumarização clássica), mas o ecossistema de LLMs generalistas consolidou **GPT-style**.  
→ Post [01](./01-arquitetura-transformer-decoder-llm.md).

### 3. O que é “temperatura” na prática?

Escalona a **logits** antes do softmax: temperatura **maior** = distribuição mais **plana** = mais aleatoriedade; **menor** = mais **gulosa** e repetitiva.  
→ Post [01](./01-arquitetura-transformer-decoder-llm.md).

---

## Atenção e complexidade

### 4. Por que dizem que atenção é O(n²)?

Na forma densa padrão, para \(n\) tokens você materializa scores de tamanho **\(n \times n\)** (por cabeça/camada) — isso explode em memória e FLOPs para sequências longas. Há variantes **esparsas** ou **locais** que reduzem isso.  
→ Post [02](./02-attention-mha-mqa-gqa-mla-flashattention.md).

### 5. Qual a diferença entre MHA, MQA e GQA?

- **MHA:** cada cabeça tem seu K e V.  
- **MQA:** **um** K/V compartilhado por todas as cabeças de Q.  
- **GQA:** **grupos** de cabeças compartilham o mesmo KV (meio-termo).  
→ Post [02](./02-attention-mha-mqa-gqa-mla-flashattention.md).

### 6. O que é FlashAttention e por que não muda o resultado?

É uma **implementação** da mesma atenção (em FP16/BF16 “exata” no sentido matemático usual), reorganizando I/O para usar melhor SRAM — **não** é uma aproximação como atenção esparsa.  
→ Post [02](./02-attention-mha-mqa-gqa-mla-flashattention.md).

---

## KV cache e memória

### 7. O que é KV cache e por que ele existe?

Na geração autoregressiva, recalcular K e V para todos os tokens anteriores a cada passo seria caro. O cache **armazena** K e V já computados.  
→ Post [03](./03-kv-cache-anatomia-pagedattention-vllm.md).

### 8. Minha GPU tem 24 GB. Por que o modelo “de 13B INT4” ainda estoura?

Porque além dos **pesos** entram **ativações**, **buffers**, e o **KV cache** — que cresce com o **tamanho do contexto**.  
→ Posts [03](./03-kv-cache-anatomia-pagedattention-vllm.md), [CHEATSHEET](./CHEATSHEET.md).

### 9. O que é PagedAttention?

Alocação de KV em **blocos** não necessariamente contíguos, como **páginas de memória** virtual — reduz fragmentação e permite batching eficiente (vLLM).  
→ Post [03](./03-kv-cache-anatomia-pagedattention-vllm.md).

### 10. Prefill e decode: qual otimizar primeiro?

Depende do seu gargalo: **prompt enorme** costuma doer no **prefill**; **geração longa** costuma doer no **decode** (leitura de KV, bandwidth).  
→ Post [03](./03-kv-cache-anatomia-pagedattention-vllm.md), [DECISION-TREE](./DECISION-TREE.md).

---

## Quantização

### 11. Quantizar pesos piora qualidade?

**Quase sempre** há trade-off: menos bits ⇒ mais erro. Métodos modernos (GPTQ, AWQ, NF4) tentam minimizar isso em **4 bits**. Sempre valide na **sua tarefa**.  
→ Post [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md).

### 12. GGUF, GPTQ, AWQ: qual escolher?

- **GGUF + llama.cpp:** CPU/edge, ecossistema enorme.  
- **GPTQ/AWQ:** comum em **GPU** com servidores HF/vLLM.  
Não existe “melhor absoluto” — depende de **hardware** e **framework**.  
→ Post [04](./04-quantizacao-pesos-gptq-awq-gguf-bitsandbytes.md).

### 13. Quantizar KV é igual a quantizar pesos?

**Não.** KV é **online**, muda por token, tem **outliers** em K, interage com **RoPE** — os métodos são diferentes (KIVI, KVQuant, etc.).  
→ Post [05](./05-quantizacao-kv-cache-kivi-kvquant-cachegen.md).

---

## TurboQuant

### 14. O que é TurboQuant em uma frase?

Uma proposta de **compressão de vetores** (incluindo KV) usando representação **polar** (módulo + direção) e quantização com cotas ligadas a **rate–distortion** — paper **arXiv:2504.19874**.  
→ Post [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md), [turboquant-docs](turboquant/INDEX.md).

### 15. TurboQuant já está pronto para produção?

Em **abril de 2026**, trate como **pesquisa ativa**: existem implementações comunitárias e resultados mistos em **prefill vs decode**. Valide antes de prometer SLA.  
→ Post [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md).

---

## Contexto longo e alternativas

### 16. O que é RoPE e por que todo mundo usa?

**RoPE** injeta posição via **rotações** em pares de dimensões; funciona bem com **atenção relativa** e virou padrão em Llama/Qwen/Mistral.  
→ Post [07](./07-contexto-longo-rope-yarn-ring-streaming.md).

### 17. Como aumentar contexto além do treino?

Técnicas como **YaRN**, **Position Interpolation**, **LongRoPE**, ou mudar o modelo ( **MLA**, **Ring Attention**). Alternativa pragmática: **RAG**.  
→ Post [07](./07-contexto-longo-rope-yarn-ring-streaming.md).

### 18. Mamba substitui Transformer?

**Depende.** SSMs (ex. **Mamba**) têm custo subquadrático em \(n\) em certos regimes, mas ecossistema, tooling e “qualidade geral” ainda competem com Transformers em muitos cenários. Há **híbridos** (Jamba).  
→ Post [07](./07-contexto-longo-rope-yarn-ring-streaming.md).

---

## Throughput, MoE, speculative

### 19. O que é speculative decoding?

Um modelo **pequeno** propõe vários tokens; o modelo **grande** **verifica** em paralelo — pode aumentar tokens/s se o *acceptance rate* for bom.  
→ Post [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md).

### 20. MoE significa que só parte do modelo roda?

Em **inferência**, o **roteador** escolhe poucos **experts** por token (*top-k*), mas **VRAM** frequentemente precisa comportar **muitos** experts carregados — depende da implementação e do modelo.  
→ Post [08](./08-alem-quantizacao-sparsity-speculative-moe-distillation.md).

### 21. Onde leio o mapa “qual técnica usar”?

→ [DECISION-TREE.md](./DECISION-TREE.md).

---

## Referência rápida

### 22. Onde está a fórmula do KV cache?

→ [CHEATSHEET.md](./CHEATSHEET.md).

### 23. Onde está o glossário de siglas?

→ [GLOSSARY.md](./GLOSSARY.md).

---

*FAQ Onda 1 — meta: 20+ perguntas; expansão contínua.*
