# TurboQuant — Seção 3.3 (limites inferiores) e Seção 4 (experimentos)

Resumo em português (BR) a partir de `paper-turboquant-cp.md` (trecho ~1076–1679), com números-chave em bullets e contexto breve de literatura (web).

---

## Parte 1 — Seção 3.3: limites inferiores (Teorema 3)

### Mensagem principal

O artigo mostra que o **TurboQuant** atinge uma **taxa de distorção ótima a menos de um fator constante pequeno**, para qualquer largura de bits, provando **limites inferiores** sobre a melhor distorção alcançável por **qualquer** algoritmo de compressão.

### Princípio minimax de Yao

- A prova usa o **princípio minimax de Yao** para relacionar:
  - o limite inferior para **algoritmos aleatorizados** com entrada **pior caso** (vetores determinísticos adversariais), e
  - o limite inferior para **algoritmos determinísticos** com entrada **aleatorizada** (distribuição difícil).
- Em seguida, o limite para este segundo cenário é ligado ao **limite inferior de Shannon (SLB)** da Sec. 2.1 (via **Lema 3** no manuscrito: MSE mínima para entradas uniformes na esfera unitária \(S^{d-1}\)).

### Teorema 3 (formas explícitas)

Para qualquer algoritmo de quantização aleatorizado \(Q : S^{d-1} \to \{0,1\}^{b\cdot d}\) com **largura \(b\) bits por coordenada** e qualquer mapa de reconstrução \(Q^{-1}\):

1. **MSE (pior instância \(x \in S^{d-1}\))**  
   \[
   D_{\mathrm{mse}}(Q) \;\ge\; \frac{1}{4^{\,b}}.
   \]

2. **Distorção de produto interno** — existe \(y \in S^{d-1}\) tal que  
   \[
   D_{\mathrm{prod}}(Q) \;\ge\; \frac{1}{d}\cdot \frac{1}{4^{\,b}}.
   \]

### Argumento do **princípio da casa dos pombos** para \(D_{\mathrm{prod}}\)

- Da cota \(D_{\mathrm{mse}} \ge 4^{-b}\) e da decomposição do MSE nas **\(d\) coordenadas** como erros ao longo dos vetores canônicos \(e_j\):
  \[
  D_{\mathrm{mse}} = \sum_{j=1}^{d} \mathbb{E}\big[(\langle e_j,x\rangle - \langle e_j,Q^{-1}(Q(x))\rangle)^2\big] \ge 4^{-b},
  \]
- pelo **princípio da casa dos pombos**, existe um índice \(j \in [d]\) com erro médio **pelo menos** \(\frac{1}{d}\cdot 4^{-b}\), o que conclui a parte de \(D_{\mathrm{prod}}\) ao tomar \(y = e_j\) (no enunciado, \(y \in S^{d-1}\); o texto formaliza a mesma ideia em termos de coordenadas).

### Fator \(\sqrt{3\pi}/2\) **versus** cota inferior (alinhamento com Teoremas 1–2)

- Os **Teoremas 1 e 2** garantem cotas **superiores** do tipo **\(4^{-b}\)** com prefator constante **\(\sqrt{3\pi}/2\)** (MSE e \(D_{\mathrm{prod}}\), com dependência \(\|y\|_2^2/d\) no caso do produto interno).
- O **Teorema 3** dá **\(4^{-b}\)** (e \(\frac{1}{d}4^{-b}\) para \(D_{\mathrm{prod}}\)) **sem** esse prefator.
- Assim, a **folga** entre as garantias do TurboQuant e a **cota informacional** está concentrada no fator **\(\sqrt{3\pi}/2 \approx 1{,}535\)** (não ~2,7): é a constante que aparece nas **Fig. 3** do paper como “Upper Bound” em relação a “Lower Bound: \(1/(d\,4^{b})\)” e \(4^{-b}\)).

### Alternativa por **empacotamento de esferas** (“sphere packing”)

- O artigo observa que cotas inferiores comparáveis para quantização vetorial também podem ser obtidas por argumentos de **empacotamento de esferas** (referência **[26]**, Gersho 1982), em geral com **constantes maiores** por tratarem um problema **mais difícil** (pior caso).
- **Teorema 3** é preferido na análise porque:
  - limita a **distorção esperada** (não só o pior caso), alinhando-se às cotas **superiores esperadas** dos Teoremas 1–2.

---

## Parte 2 — Seção 4: experimentos

**Hardware (global):** todos os experimentos com **1× GPU NVIDIA A100**.

### Contexto externo (não substitui o paper)

- **LongBench** (Bai et al., arXiv:2308.14508): benchmark **bilíngue** e **multitarefa** para compreensão em **contexto longo** (QA single/multi, sumarização, few-shot, sintético, código, etc.).
- **Needle-in-a-haystack (NIAH)** (ex.: Kamradt, 2023): teste em que uma frase única (“agulha”) é inserida em documento longo (“palheiro”); mede-se se o modelo **recupera** a informação — métrica típica: **recall**.
- **PQ / ANN:** em busca aproximada de vizinhos, **recall@k** e variantes (ex.: fração dos verdadeiros top‑\(k\) recuperados) são padrão; benchmarks como **ANN-Benchmarks** costumam reportar recall versus QPS e tempo de construção do índice — útil como **panorama**, enquanto o paper compara **TurboQuant vs PQ vs RabitQ** nos mesmos protocolos.

---

### 4.1 Validação empírica (DBpedia + embeddings OpenAI)

- **Dados:** **DBpedia Entities**, embeddings **OpenAI-3**, dimensão **\(d = 1536\)**.
- **Amostragem:** **100 000** pontos (treino) + **1 000** consultas (query set).
- **Métodos:** **TurboQuant\(_{\mathrm{prod}}\)** (produto interno **não enviesado**) e **TurboQuant\(_{\mathrm{mse}}\)** (otimizado para MSE).
- **Achados (Fig. 1–2, 3):**
  - maior **largura de bits** → menor variância em ambos;
  - para **estimação de produto interno**, **TurboQuant\(_{\mathrm{mse}}\)** introduz **viés** que some com mais bits;
  - **TurboQuant\(_{\mathrm{prod}}\)** permanece **não enviesado** em todos os bitwidths testados;
  - com **\(b=2\)**, a variância do erro de produto interno para **TurboQuant\(_{\mathrm{prod}}\)** é **aproximadamente constante** em relação ao produto interno médio; já **TurboQuant\(_{\mathrm{mse}}\)** mostra viés **crescente** com o produto interno médio (Fig. 2: **Avg IP** = 0,01 / 0,06 / 0,10 / 0,17).
  - curvas de **erro de produto interno** e **MSE** alinhadas às **cotas teóricas** (Fig. 3): limites **inferiores** \(4^{-b}\), \(1/(d\,4^{b})\) e **superiores** com prefator \(\sqrt{3\pi}/2\).

**Bullets numéricos (Teoremas 1–2, valores “finos” no texto):**

- **MSE** para \(b = 1,2,3,4\): **\(D_{\mathrm{mse}} \approx 0{,}36,\; 0{,}117,\; 0{,}03,\; 0{,}009\)**.
- **\(D_{\mathrm{prod}}\)** para \(b = 1,2,3,4\): **\(\approx \dfrac{1{,}57}{d},\; \dfrac{0{,}56}{d},\; \dfrac{0{,}18}{d},\; \dfrac{0{,}047}{d}\)**.

---

### 4.2 Needle-in-a-haystack

- **Referência de benchmark:** Kamradt (**Needle in a Haystack**); setup alinhado a **Fu et al.** (dados/engineering para contexto longo).
- **Modelo:** **Llama-3.1-8B-Instruct**.
- **Comprimentos:** documento de **4k a 104k tokens**.
- **Métrica:** **recall** (recuperação da frase escondida).
- **Compressão de KV:** razão **0,25** (**25%** do cache KV completo) para todos os métodos comparados.
- **Baselines:** **PolarQuant**, **SnapKV**, **PyramidKV**, **KIVI**.
- **Resultado destacado:** métodos com **garantias teóricas** (**PolarQuant**, **TurboQuant**) superam compressões ao nível de token (**SnapKV**, **PyramidKV**) e quantização escalar sem garantia formal (**KIVI**); **TurboQuant** iguala o modelo **precisão total** com compressão **\(> 4\times\)** (Fig. 4 — scores tipo **0,997** para TurboQuant vs Full-Precision **0,997** no gráfico reportado).

---

### 4.3 Geração ponta a ponta — **LongBench**

- **Dataset:** **LongBench**; subconjunto **LongBench-E** (distribuição de comprimentos mais **uniforme** para comparação justa entre tamanhos de contexto).
- **Modelos:** **Llama-3.1-8B-Instruct** e **Ministral-7B-Instruct**.
- **Diferencial do método:** TurboQuant aplica quantização também durante a **geração em streaming**; **KIVI** e **PolarQuant** deixam tokens gerados **não quantizados** (como nos baselines citados).
- **Modos de bits efetivos:** **2,5** e **3,5 bits/canal** no KV, por **split** entre canais **outlier** vs **não-outlier** (duas instâncias TurboQuant). Exemplo **2,5 bits:** **32** canais a **3** bits + **96** canais a **2** bits sobre **128** canais → \((32\cdot 3 + 96\cdot 2)/128 = 2{,}5\).
- **Compressão:** pelo menos **\(4{,}5\times\)** nos vetores quantizados mantendo desempenho comparável ao **não quantizado**.

**Tabela 1 — LongBench-V1 (scores; colunas: SingleQA, MultiQA, Summarization, Few shot, Synthetic, Code, Average):**

| Configuração | KV Size | SingleQA | MultiQA | Summ. | Few shot | Synth. | Code | Avg |
|-------------|---------|----------|---------|-------|----------|--------|------|-----|
| **Llama-3.1-8B-Instruct** |
| Full Cache | **16** | 45,29 | 45,16 | 26,55 | 68,38 | 59,54 | 46,28 | **50,06** |
| KIVI | 3 | 43,38 | 37,99 | 27,16 | 68,38 | 59,50 | 44,68 | 48,50 |
| KIVI | 5 | 45,04 | 45,70 | 26,47 | 68,57 | 59,55 | 46,41 | 50,16 |
| PolarQuant | 3,9 | 45,18 | 44,48 | 26,23 | 68,25 | 60,07 | 45,24 | 49,78 |
| TurboQuant | **2,5** | 44,16 | 44,96 | 24,80 | 68,01 | 59,65 | 45,76 | **49,44** |
| TurboQuant | **3,5** | 45,01 | 45,31 | 26,00 | 68,63 | 59,95 | 46,17 | **50,06** |
| **Ministral-7B-Instruct** |
| Full Cache | **16** | 47,53 | 49,06 | 26,09 | 66,83 | 53,50 | 47,90 | **49,89** |
| TurboQuant | **2,5** | 48,38 | 49,22 | 24,91 | 66,69 | 53,17 | 46,83 | **49,62** |

- Com **3,5 bits/canal** no Llama-3.1-8B, a **média 50,06** **iguala** o **Full Cache** (50,06).

---

### 4.4 Busca de vizinho mais próximo — **TurboQuant vs PQ** (e RabitQ)

- **Dados:** DBpedia Entities com embeddings OpenAI-3 em **\(d \in \{1536,\,3072\}\)**; também **GloVe** em **\(d=200\)** (10 000 queries pré-definidas para GloVe).
- **Treino/avaliação:** 100 000 pontos + 1 000 queries (quando não há query set nativo).
- **Baselines:** **Product Quantization (PQ)** e **RabitQ**.
- **Métrica:** **\(1@k\)** — frequência com que o **verdadeiro** resultado de **maior produto interno** aparece entre os **top‑\(k\)** aproximados.
- **PQ:** codebooks via **k-means**; para comparar com outros métodos, usaram **LUT256** (256 codewords); **LUT16** era mais rápido mas com **queda forte de qualidade**. Mesmo dataset para treino e teste dá **vantagem inerente** ao PQ.
- **Resultado qualitativo (Fig. 5):** **TurboQuant** domina **PQ** e **RabitQ** em **recall** nos cenários reportados (2 e 4 bits).

**Tabela 2 — Tempo de quantização (segundos), quantização 4 bits:**

| Abordagem | \(d=200\) | \(d=1536\) | \(d=3072\) |
|-----------|-----------|------------|------------|
| **PQ** | **37,04** | **239,75** | **494,42** |
| RabitQ | 597,25 | 2267,59 | 3957,19 |
| **TurboQuant** | **0,0007** | **0,0013** | **0,0021** |

- **Ordens de magnitude:** TurboQuant ~**\(10^5\)–\(10^6)\times** mais rápido que PQ na **quantização** nesta tabela (tempo de indexação/quantização do corpus, não confundir com latência de consulta isolada).

---

## Referências rápidas do próprio paper (seções citadas)

- **LongBench / LongBench-E:** [10] Bai et al., arXiv:2308.14508.  
- **Needle-in-a-haystack:** [32] Kamradt; setup [21] Fu et al.  
- **Sphere packing / VQ:** [26] Gersho 1982.  
- **PQ:** [31] Jégou et al.; **RabitQ:** [22] Gao et al., arXiv:2409.09913.
