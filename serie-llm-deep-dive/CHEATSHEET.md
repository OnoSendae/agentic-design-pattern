# Cheatsheet — LLMs (inferência e memória)

Referência de **1 página** (expandir conforme necessidade). Valores são **ordens de ideia**; sempre confira no seu modelo/framework.

---

## 1. Tamanho do KV cache (ordem de ideia)

Para **uma** sequência, **sem batch**, precisão **FP16** (2 bytes/elem), atenção **MHA/GQA** clássica:

$$
\text{KV\_bytes} \approx 2 \times L \times H_{kv} \times d_h \times T \times 2
$$

Onde:

- $L$ = número de **camadas** com atenção  
- $H_{kv}$ = número de **cabeças KV** (em GQA, $H_{kv} < H_q$)  
- $d_h$ = **head dim** (tipicamente $d_{\text{model}} / H_q$)  
- $T$ = **tokens** no contexto (prompt + gerados)  
- Fator **2** = tensores **K** e **V**  
- Fator final **2** = bytes por elemento em FP16  

**Regra rápida (FP16):**  

$$
\text{KV\_bytes/token} \approx 4 \times L \times H_{kv} \times d_h
$$

**BF16/FP16:** mesmo tamanho (2 bytes). **FP8:** ~metade se tudo em FP8 de 1 byte por elem (depende de layout e kernels).

**MLA (DeepSeek):** usar fórmula do modelo (dimensão latente do KV); ver Post [03](./03-kv-cache-anatomia-pagedattention-vllm.md).

---

## 2. Tamanho dos pesos (ordem de ideia)

Modelo em **FP16/BF16** (2 bytes/param):  

$$
\text{Pesos\_GB} \approx \frac{N \times 2}{10^9}
$$

($N$ = parâmetros totais.)

**INT4** (~0,5 byte/param médio, depende de empacotamento): ordem de **~0,5 × N** bytes + overhead de metadados.

---

## 3. Prefill vs decode (o que otimizar)

| Fase | Compute | Memória | Nota |
|------|---------|---------|------|
| **Prefill** | Alto (muitos tokens de entrada) | KV cresce rápido | Muitas vezes **compute-bound** em GPUs rápidas |
| **Decode** | Baixo por passo (1 token) | Leitura de KV grande | Muitas vezes **memory-bandwidth-bound** |

**Roofline (lembrete):** se FLOPs/byte < pico da GPU para o kernel, você está **limitado por banda**, não por FLOPs.

---

## 4. Comandos rápidos (copiar com cuidado)

### llama.cpp (exemplo)

```bash
# Modelo GGUF + quantização KV (ajustar paths e flags conforme build)
./llama-cli -m modelo.Q4_K_M.gguf -p "Olá" -ngl 99 \
  --cache-type-k q8_0 --cache-type-v q8_0
```

Tipos comuns de cache: `f16`, `q8_0`, `q5_1`, `q4_1`, `q4_0` (ver documentação da sua versão).

### vLLM (exemplo conceitual)

```bash
# Opções mudam entre versões — ver docs oficiais
python -m vllm.entrypoints.openai.api_server \
  MODEL_NAME \
  --dtype auto \
  --kv-cache-dtype fp8_e4m3   # exemplo: FP8 KV quando suportado
```

Sempre validar: `--max-model-len`, tensor parallel, e dtype do KV no **release** que você usa.

### bitsandbytes (Hugging Face, exemplo)

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
```

---

## 5. Formatos de quantização de pesos (lembrete)

| Família | Onde | Notas |
|---------|------|--------|
| **GGUF** (llama.cpp) | Disco + inferência CPU/GPU | `Q4_K_M`, `Q5_K_M`, `Q8_0`, variantes IQ* |
| **GPTQ** | INT4 com calibração | Arquivos `.safetensors` + config |
| **AWQ** | INT4 activation-aware | Comum em servidores |
| **EXL2** | ExLlamaV2 | Controle fino de bits por grupo |
| **MLX** | Apple Silicon | Arrays quantizados nativos |

---

## 6. Variantes de atenção (efeito no KV)

| Variante | KV por token (vs MHA) |
|----------|-------------------------|
| **MHA** | Baseline |
| **GQA** | Menor (menos cabeças KV) |
| **MQA** | Mínimo entre as clássicas (1 KV compartilhado) |
| **MLA** | Muito menor (latente) |

---

## 7. Sampling (chat)

| Método | Efeito |
|--------|--------|
| **Temperature** ↑ | Mais aleatório |
| **Top-k** | Corta cauda de vocabulário |
| **Top-p** | Nucleus: massa cumulativa |
| **Greedy** | Sempre argmax (repetição, chato) |

---

## 8. TurboQuant (lembrete de 1 linha)

Quantização **polar** (módulo + direção) + correção leve → visa KV compacto com erro controlado; paper **arXiv:2504.19874**. Detalhes: Post [06](./06-turboquant-deep-dive-polar-jl-lloydmax.md).

---

## 9. Links úteis (oficiais)

| Recurso | URL |
|---------|-----|
| vLLM docs | https://docs.vllm.ai |
| llama.cpp | https://github.com/ggerganov/llama.cpp |
| Transformers (HF) | https://huggingface.co/docs/transformers |
| FlashAttention | https://github.com/Dao-AILab/flash-attention |
| TurboQuant (arXiv) | https://arxiv.org/abs/2504.19874 |

---

*Onda 1 — manter este arquivo curto; detalhes nos posts da série.*
