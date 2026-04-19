# TurboQuant: quantização vetorial *online* com taxa de distorção quase ótima

**Tradução organizada para português brasileiro** · Original: *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv:2504.19874v1 [cs.LG], 28 Apr 2025)  
**Fonte do texto base:** `transcripts/2504.19874v1.md` (extração automática do PDF)

---

## Metadados do artigo

| Campo | Conteúdo |
|--------|-----------|
| Identificador | arXiv:2504.19874v1 |
| Área | cs.LG (aprendizado de máquina) |

### Autores e afiliações

- **Amir Zandieh** — Google Research — zandieh@google.com  
- **Majid Daliri** — New York University — daliri.majid@nyu.edu  
- **Majid Hadian** — Google DeepMind — majidh@google.com  
- **Vahab Mirrokni** — Google Research — mirrokni@google.com  

---

## Resumo

A quantização vetorial, problema enraizado na teoria de codificação de fontes de Shannon, busca quantizar vetores euclidianos de alta dimensão minimizando a distorção na estrutura geométrica. Propõe-se o **TurboQuant** para tratar tanto o erro quadrático médio (**MSE**, *mean-squared error*) quanto a distorção no produto interno, superando limitações de métodos existentes que não atingem taxas de distorção ótimas.

Os algoritmos são **agnósticos aos dados** (*data-oblivious*), adequados a aplicações *online*, e alcançam taxas de distorção **quase ótimas** (a menos de uma constante multiplicativa pequena) para todas as larguras de bits e dimensões. O método:

1. Aplica **rotação aleatória** aos vetores de entrada, induzindo uma distribuição Beta concentrada nas coordenadas.  
2. Explora a **quase independência** entre coordenadas distintas em altas dimensões para usar quantizadores escalares ótimos por coordenada.  

Reconhecendo que quantizadores ótimos em MSE introduzem **viés** na estimativa do produto interno, propõe-se uma abordagem em **dois estágios**: quantização MSE seguida de uma transformada **QJL** (*Quantized Johnson–Lindenstrauss*) de 1 bit sobre o **resíduo**, obtendo um quantizador de produto interno **não viesado**.

O artigo ainda apresenta prova formal dos **limites inferiores** informação-teóricos para a melhor taxa de distorção alcançável por qualquer quantizador vetorial, mostrando que o TurboQuant **aproxima** esses limites com fator de ordem **≈2,7** (constante pequena).

**Resultados experimentais:** para quantização de *cache* KV, neutralidade de qualidade com **3,5 bits por canal** e degradação marginal com **2,5 bits**. Em busca do vizinho mais próximo, o método supera técnicas de **quantização de produto** (*product quantization*, PQ) em *recall*, reduzindo o tempo de indexação para quase zero.

---

## 1 Introdução

A quantização vetorial (**VQ**) no espaço euclidiano é central para lidar com vetores de alta dimensão: treinamento e implantação de modelos de IA e *deep learning*, bases vetoriais para busca/recuperação, etc. O objetivo central é **comprimir** vetores quantizando coordenadas em ponto flutuante para inteiros de poucos bits, minimizando **distorção** (por exemplo, MSE ou erro de produto interno). Preservando essas propriedades, consultas de produto interno podem ser respondidas com baixa latência e menor custo computacional e de comunicação.

As origens remontam ao trabalho de Shannon sobre codificação de fontes [48, 49]: a menor distorção alcançável por códigos de fonte em bloco (quantizadores vetoriais) é descrita pela função distorção–taxa de Shannon, dependendo da fonte e da medida de distorção (ex.: MSE).

Hoje, a VQ é crítica em IA, *deep learning* e sistemas de busca. Modelos de linguagem (**LLMs**) [5, 18, 7, 52] escalam com tamanho e comprimento de contexto [34]; a inferência sofre com gargalos de memória (HBM/SRAM, *clusters*). Quantizar pesos e ativações mitiga esses gargalos. O **produto interno** entre ativações e pesos é o núcleo dos modelos; esquemas de quantização buscam comprimir vetores preservando produtos internos.

Transformadores decodificadores [54] armazenam incorporações **KV** (*key/value*) de tokens anteriores no **cache KV**, cujo tamanho cresce com camadas, cabeças de atenção e comprimento de contexto — gargalo de memória e velocidade, especialmente em modelos de **contexto longo**. Reduzir o cache KV sem perder acurácia exige preservar a geometria euclidiana (produtos internos e distâncias); a VQ é um enquadramento natural.

A busca de **vizinho mais próximo** em altas dimensões com produto interno ou similaridade cosseno [1, 27] sustenta bases vetoriais [4, 2, 3], RAG (*retrieval-augmented generation*) [23, 19] e recuperação de informação [35, 46]. A PQ desempenha papel central: comprime vetores da base, poupa memória e permite estimar produtos internos com consultas rapidamente.

Os algoritmos de VQ existentes fazem **trade-off**: ou não são amigáveis a aceleradores (vetorização) e são lentos — inadequados a KV *online* —, ou têm cotas de distorção subótimas em relação à largura de bits.

**Contribuição:** o **TurboQuant** é **leve**, aplicável *online* (essencial para cache KV) e **compatível com aceleradores**. O núcleo é **dois estágios**:

1. Quantizador vetorial com taxa de distorção ótima em termos de **MSE**.  
2. Quantizador de **1 bit** no resíduo → estimador de produto interno **não viesado** e de baixa distorção.

Quantizadores ótimos em MSE, mostra-se, **não** são estimadores não viesados do produto interno; a solução em dois estágios preenche essa lacuna.

O quantizador MSE-ótimo **rotaciona aleatoriamente** vetores *d*-dimensionais; cada coordenada rotacionada segue uma **Beta**; projetam-se quantizadores **Lloyd–Max** [42, 43] por coordenada via *k-means* contínuo. Para produto interno não viesado, compõe-se com a transformada **QJL** [62] (1 bit por coordenada no resíduo). Há cotas de distorção provadamente ótimas para MSE e produto interno, com melhoria exponencial na dependência da largura de bits em relação a métodos anteriores.

---

## 1.1 Definição do problema

Objetivo: projetar um mapa de quantização \(Q : \mathbb{R}^d \to \{0,1\}^B\). Se \(B = b \cdot d\) para \(b \geq 0\), o quantizador tem **largura de bits** média \(b\) por coordenada real. Exige-se um mapa inverso \(Q^{-1} : \{0,1\}^B \to \mathbb{R}^d\) para **desquantização** aproximada. O mapa é **lossy**; busca-se minimizar distorção com foco em **MSE** e **distorção de produto interno**, **sem** hipótese sobre a distribuição dos dados (**pior caso**). O quantizador pode ser **aleatorizado**; usa-se distorção **esperada** sobre a aleatoriedade.

Definições (vetores \(x, y \in \mathbb{R}^d\), pior caso):

- **MSE:**  
  \[
  D_{\text{mse}} := \mathbb{E}_Q\left[\left\|x - Q^{-1}(Q(x))\right\|_2^2\right] \quad \text{(Eq. 1)}
  \]

- **Erro de produto interno:**  
  \[
  D_{\text{prod}} := \mathbb{E}_Q\left[\left(\langle y,x\rangle - \langle y, Q^{-1}(Q(x))\rangle\right)^2\right] \quad \text{(Eq. 2)}
  \]

Para quantizadores de produto interno, exige-se **não-viés**:

\[
\mathbb{E}_Q\left[\langle y, Q^{-1}(Q(x))\rangle\right] = \langle y, x\rangle
\]

Buscam-se quantizadores computacionalmente eficientes \(Q_{\text{mse}}\) e \(Q_{\text{prod}}\) com cotas ótimas para cada \(b\), com \(Q_{\text{prod}}\) não viesado.

**Primitivas** para \(n\) vetores \(x_1,\ldots,x_n\):

- **Quant:** quantiza o conjunto e calcula \(Q(x_1),\ldots,Q(x_n)\).  
- **DeQuant:** dado o conjunto quantizado, reconstrói \(Q^{-1}(Q(x_i))\) para qualquer \(i\).

---

## 1.2 Trabalhos relacionados

**Início da VQ:** Shannon [48, 49]; Zador [61] (altas taxas); Gersho [25] (teoria de alta resolução, quantização em treliça). A busca exaustiva de vizinho mais próximo era cara, limitando adoção prática.

**Quantização *online* vs. *offline*:** Métodos *online* (agnósticos aos dados) aplicam-se sem ajuste por dados [16, 8, 41, 47, 28]. Métodos *offline* exigem pré-processamento e aprendizado [37]; outros usam informação de segunda ordem (Hessiana) [20, 39, 57, 13].

**Compressão de cache KV:** Modificações arquiteturais [50, 6, 15]; poda/evicção de tokens [11, 66, 40, 58, 64, 38, 29]; quantização do cache [60, 59, 17, 33, 65, 41, 30, 36, 28]. A **QJL** [62] oferece quantização *data-oblivious* de 1 bit com estimativas não viesadas de produto interno; o TurboQuant para distorção de produto interno utiliza essa ideia.

**PQ (*Product Quantization*):** Em busca de vizinhos, o tamanho do índice motiva PQ; muitos métodos constroem *codebooks* com *k-means* na indexação [31, 9, 24, 56, 27] — **não** são *online*. RabitQ em grade na esfera [22] elimina pré-treino, mas análise teórica é subótima e o algoritmo é lento e pouco vetorizável em GPU [22].

---

## 1.3 Visão geral das técnicas e contribuições

### TurboQuant otimizado para MSE

Minimiza a distorção MSE da Eq. (1). **Rotação aleatória** induz distribuição **Beta** por coordenada; em dimensão alta, cada coordenada aproxima **Gaussiana** e coordenadas distintas ficam **quase independentes** [55], permitindo quantização escalar ótima **por coordenada** (*Max–Lloyd*, *codebooks* pré-computados).

**Teorema 1 (escopo):** o TurboQuant MSE \(Q_{\text{mse}} : \mathbb{R}^d \to \{0,1\}^{b\cdot d}\) para \(\|x\|_2=1\) satisfaz, entre outras cotas,

\[
D_{\text{mse}}(Q_{\text{mse}}) \leq \frac{\sqrt{3\pi}}{2} \cdot \frac{1}{4^b}
\]

e valores aproximados para \(b=1,2,3,4\): \(0{,}36,\ 0{,}117,\ 0{,}03,\ 0{,}009\). Norma unitária é padrão; normas gerais podem ser armazenadas em precisão flutuante para reescala.

### TurboQuant para produto interno

Quantizadores MSE são **viesados** para o produto interno. Solução: usar \(Q_{\text{mse}}\) com **\(b-1\)** bits e aplicar **QJL** [62] no resíduo.

**Teorema 2 (escopo):** o TurboQuant produto \(Q_{\text{prod}}\) é **não viesado** e limita \(D_{\text{prod}}\) com fator \(\|y\|_2^2/d\) vezes \(1/4^b\) (com constante \(\sqrt{3}\pi^2\) na forma do artigo). Valores refinados para \(b=1,\ldots,4\) conforme o texto original.

### Limites inferiores

**Teorema 3:** via limite inferior de Shannon e princípio minimax de Yao, para qualquer algoritmo aleatorizado \(Q\) com largura \(b\), existem instâncias difíceis com cotas do tipo \(1/4^b\) para MSE e \(\|y\|^2_2/(d\,4^b)\) para produto interno (ver artigo para enunciado exato).

O TurboQuant fica a fator **\(\sqrt{3\pi}/2 \approx 2{,}7\)** do limite informação-teórico no MSE; para \(b=1\), o fator cai para perto de **1,45**.

### Experimentos (prévia)

- **§4.1:** validação empírica das cotas (DBpedia, *embeddings* 1536-D).  
- **§4.2–4.3:** cache KV *online* — tarefa *needle-in-a-haystack* e **LongBench**; compressão do cache KV **>5×** com forte retenção de desempenho.  
- **§4.4:** busca de vizinho com **PQ** e **RabitQ** — TurboQuant melhor em *recall* e indexação quase sem tempo.

---

## 2 Preliminares

**Notação:** vetores em negrito (\(\mathbf{x}\), \(\mathbf{y}\)); matrizes em negrito maiúsculo (\(\mathbf{M}\)); fatia \(\mathbf{x}_{i:j}\); linha \(\mathbf{M}_{i,:}\); esfera unitária \(\mathbb{S}^{d-1}\); entropia diferencial \(h(x)\); informação mútua \(I(x;y)=h(x)-h(x|y)\).

### Lema 1 (distribuição da coordenada em \(\mathbb{S}^{d-1}\))

Se \(\mathbf{x}\) é uniforme em \(\mathbb{S}^{d-1}\), cada coordenada \(x_j\) segue a densidade Beta (forma fechada no artigo). Em altas dimensões, aproxima-se \(\mathcal{N}(0,1/d)\).

### 2.1 Limite inferior de Shannon (SLB)

**Lema 2 (SLB):** para fonte \(\mathbf{x}\) com entropia \(h(\mathbf{x})\) e taxa de bits total \(B\), a distorção MSE ótima obedece à cota clássica (ver [14]).

**Lema 3 (SLB na esfera):** para \(\mathbf{x}\) uniforme em \(\mathbb{S}^{d-1}\), \(D(B) \geq 2^{-2B/d}\).

### 2.2 QJL: quantização de produto interno com 1 bit

**Definição 1 (QJL):**  
\(Q_{\text{qjl}}(\mathbf{x}) = \operatorname{sign}(\mathbf{S}\mathbf{x})\) com \(\mathbf{S}\) Gaussiana i.i.d.; desquantização \(\mathbf{Q}_{\text{qjl}}^{-1}(\mathbf{z}) \propto \mathbf{S}^\top \mathbf{z}\) (escala \(\sqrt{\pi/2}/\sqrt{d}\) como no artigo).

**Lema 4:** estimador **não viesado** e cota de variância \(\frac{\pi}{2d}\|\mathbf{y}\|_2^2\) (detalhes nas Eqs. (3) do original).

---

## 3 TurboQuant: quantização de alto desempenho

Dois algoritmos: (i) MSE; (ii) produto interno não viesado. **§3.3** prova limites inferiores informação-teóricos.

### 3.1 TurboQuant MSE ótimo

Rotação **\(\boldsymbol{\Pi}\)**, coordenadas com densidade \(f_X\) na \([-1,1]\); particionar \([-1,1]\) em \(2^b\) intervalos (solução de *k-means* 1D / Voronoi [42]). Custo escalar ótimo \(C(f_X,b)\) (Eq. 4 no artigo). Centroides para \(b=1,2\) em termos de \(d\) conforme o texto.

**Algoritmo 1 — TurboQuant\(_{\text{mse}}\)** (resumo em PT-BR)

| Passo | Descrição |
|--------|-----------|
| Entrada | Dimensão \(d\), largura de bits \(b\) |
| 1 | Gerar matriz de rotação aleatória \(\boldsymbol{\Pi} \in \mathbb{R}^{d\times d}\) |
| 2 | Construir *codebook*: centroides \(c_1,\ldots,c_{2^b}\) que minimizam o custo MSE (Eq. 4) |
| **Quant\(_{\text{mse}}\)(\(x\))** | \(\mathbf{y} \leftarrow \boldsymbol{\Pi}x\); para cada \(j\), guardar índice do centróide mais próximo de \(y_j\) |
| **DeQuant\(_{\text{mse}}\)** | Reconstruir \(\tilde{\mathbf{y}}\) pelos centroides; \(\tilde{\mathbf{x}} \leftarrow \boldsymbol{\Pi}^\top \tilde{\mathbf{y}}\) |

*Codificação entrópica dos índices* pode reduzir bits médios (~5% em \(b=4\)); os autores omitiram por simplicidade.

### 3.2 TurboQuant para produto interno

Combina \(Q_{\text{mse}}\) com **\(b-1\)** bits e **QJL** no resíduo \(\mathbf{r} = x - Q_{\text{mse}}^{-1}(Q_{\text{mse}}(x))\). Para \(b=1\), o viés \(2/\pi\) do caminho puramente MSE é ilustrado no artigo.

**Algoritmo 2 — TurboQuant\(_{\text{prod}}\)** (resumo)

| Passo | Descrição |
|--------|-----------|
| Entrada | \(d\), \(b\) |
| 1 | Instanciar TurboQuant\(_{\text{mse}}\) com largura \(b-1\) |
| 2 | Amostrar \(\mathbf{S}\) Gaussiana \(d\times d\) |
| **Quant\(_{\text{prod}}\)** | Obter índices MSE; resíduo \(\mathbf{r}\); \(q_{\text{qjl}} \leftarrow \operatorname{sign}(\mathbf{S}\mathbf{r})\); saída \((\text{idx}, q_{\text{qjl}}, \|\mathbf{r}\|_2)\) |
| **DeQuant\(_{\text{prod}}\)** | Somar reconstrução MSE com termo QJL escalado por \(\|\mathbf{r}\|_2\) |

**Teorema 2:** não-viés de \(\langle y,\tilde{x}\rangle\) e cota de \(D_{\text{prod}}\); prova por esperança condicional (dado \(\tilde{x}_{\text{mse}}\)) e lei da esperança total.

### 3.3 Limites inferiores

**Teorema 3:** cotas para pior caso / instâncias difíceis; segunda parte usa principo da casa dos pombos sobre coordenadas. Comparação com argumentos de *sphere packing* [26].

---

## 4 Experimentos

**Hardware:** 1× GPU NVIDIA A100.  
**Partes:** (A) verificação teórica; (B) KV *cache* e busca de vizinhos.

### 4.1 Validação empírica

Dataset **DBpedia Entities** com *embeddings* 1536-D (OpenAI). 100 mil pontos de treino, 1 mil consultas. Comparar **TurboQuant\(_{\text{prod}}\)** e **TurboQuant\(_{\text{mse}}\)** na estimativa de produto interno. Aumentar \(b\) reduz variância em ambos; apenas **prod** permanece não viesado. Texto extraído do PDF inclui rótulos soltos das **Figuras 1–3** (*histogramas*, cotas superior/inferior teóricas).

### 4.2 *Needle-in-a-haystack*

Benchmark de recuperação de frase oculta em documento longo [32]. Modelo **Llama-3.1 8B-Instruct**; comprimentos de 4k a 104k *tokens*; métrica de *recall*. Baselines: **PolarQuant**, **SnapKV**, **PyramidKV**, **KIVI** [21, 28, 38, 12, 41]. Razão de memória 0,25. **Figura 4:** TurboQuant igual ao modelo em precisão total mesmo com **>4×** quantização (conforme legenda original).

### 4.3 Geração *end-to-end* no LongBench

**LongBench** / **LongBench-E** [10]: QA, resumo, *few-shot*, código, etc. Modelos **Llama-3.1-8B-Instruct** e **Ministral-7B-Instruct**. O método quantiza também durante geração em fluxo (**diferente** de KIVI/PolarQuant em certos detalhes). **Tabela 1** (scores por tarefa): TurboQuant 2,5 e 3,5 bits competitivos; estratégia de **outliers** em canais [63, 51] — ex.: 32 canais a 3 bits e 96 a 2 bits → média 2,5 bits/canal em 128 canais.

**Tabela 2:** tempos de quantização (s) — PQ e RabitQ muito maiores que TurboQuant em \(d\in\{200,1536,3072\}\) com 4 bits.

### 4.4 Vizinho mais próximo

DBpedia em 1536-D e 3072-D [links Hugging Face no original]; **GloVe** \(d=200\) [45]. 100k treino, 1k consultas (10k consultas no GloVe). Comparar **PQ**, **RabitQ** [22], **TurboQuant**; métrica **recall@k** (top-*k* aproximado contém o verdadeiro top-1 de produto interno). **Figura 5:** *recall* superior para TurboQuant em 2 e 4 bits. Discussão de implementação PQ (AVX2, LUT256 vs LUT16) e limitações de vetorização do RabitQ.

---

## Figuras e tabelas (legendas traduzidas)

| ID | Legenda (PT-BR) |
|----|------------------|
| Fig. 1 | Distribuição do erro de **TurboQuant\(_{\text{prod}}\)** vs **\(_{\text{mse}}\)** na estimativa do produto interno. |
| Fig. 2 | Variância do erro de produto interno estável em **prod**; em **mse** depende do produto interno médio (\(b=2\)). |
| Fig. 3 | Erro de produto interno e MSE vs cotas teóricas por largura de bits. |
| Fig. 4 | *Needle-in-a-haystack* para Llama-3.1-8B: TurboQuant iguala *baseline* sem compressão apesar de >4× quantização. |
| Fig. 5 | *Recall* em GloVe (200D) e OpenAI3 (1536D, 3072D). |
| Tab. 1 | Resultados LongBench-V1 de compressão de cache KV (Llama-3.1-8B, etc.). |
| Tab. 2 | Tempo de quantização (s), 4 bits, várias dimensões. |

---

## Referências

As entradas **[1]–[66]** reproduzem o texto em **inglês** do artigo original (títulos de periódicos e nomes próprios inalterados). Consulte o PDF em arXiv para formatação bibliográfica final.

[1] Elastic search., 2025. https://www.elastic.co/enterprise-search/vector-search.  
[2] Qdrant vectore search., 2025. https://qdrant.tech/.  
[3] Pgvector search., 2025. https://github.com/pgvector/pgvector/.  
[4] Pinecone vectore database., 2025. https://www.pinecone.io/.  
[5] Achiam, J., et al. Gpt-4 technical report. arXiv:2303.08774, 2023.  
[6] Ainslie, J., et al. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. EMNLP, 2023.  
[7] Anthropic. Claude, 2024. https://www.anthropic.com/news/claude-3-family.  
[8] Ashkboos, S., et al. Quarot: Outlier-free 4-bit inference in rotated llms. arXiv:2404.00456, 2024.  
[9] Babenko, A. and Lempitsky, V. Additive quantization for extreme vector compression. CVPR, 2014.  
[10] Bai, Y., et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv:2308.14508, 2023.  
[11] Beltagy, I., Peters, M. E., and Cohan, A. Longformer. arXiv:2004.05150, 2020.  
[12] Cai, Z., et al. Pyramidkv. arXiv:2406.02069, 2024.  
[13] Chee, J., et al. Quip: 2-bit quantization of large language models with guarantees. NeurIPS, 2023.  
[14] Cover, T. M. Elements of information theory. Wiley, 1999.  
[15] Dai, D., et al. Deepseekmoe. arXiv:2401.06066, 2024.  
[16] Dettmers, T., et al. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS, 2022.  
[17] Dong, S., et al. Qaq: Quality adaptive quantization for llm kv cache. arXiv:2403.04643, 2024.  
[18] Dubey, A., et al. The llama 3 herd of models. arXiv:2407.21783, 2024.  
[19] Edge, D., et al. Graph rag approach to query-focused summarization. arXiv:2404.16130, 2024.  
[20] Frantar, E., et al. Gptq. arXiv:2210.17323, 2022.  
[21] Fu, Y., et al. Data engineering for scaling language models to 128k context. arXiv:2402.10171, 2024.  
[22] Gao, J., et al. Practical and asymptotically optimal quantization... ANNS. arXiv:2409.09913, 2024.  
[23] Gao, Y., et al. RAG survey. arXiv:2312.10997, 2023.  
[24] Ge, T., et al. Optimized product quantization. CVPR, 2013.  
[25] Gersho, A. Asymptotically optimal block quantization. IEEE TIT, 1979.  
[26] Gersho, A. On the structure of vector quantizers. IEEE TIT, 1982.  
[27] Guo, R., et al. Accelerating large-scale inference with anisotropic vector quantization. ICML, 2020.  
[28] Han, I., et al. Polarquant. arXiv:2502.02617, 2025.  
[29] Han, I., et al. Balancekv. arXiv:2502.07861, 2025.  
[30] Hooper, C., et al. Kvquant. arXiv:2401.18079, 2024.  
[31] Jegou, H., et al. Product quantization for nearest neighbor search. IEEE TPAMI, 2010.  
[32] Kamradt, G. Needle in a haystack. GitHub, 2023.  
[33] Kang, H., et al. Gear. arXiv:2403.05527, 2024.  
[34] Kaplan, J., et al. Scaling laws for neural language models. arXiv:2001.08361, 2020.  
[35] Khattab, O. and Zaharia, M. Colbert. SIGIR, 2020.  
[36] Kim, J., et al. Lexico. arXiv:2412.08890, 2024.  
[37] Kim, S., et al. Squeezellm. arXiv:2306.07629, 2023.  
[38] Li, Y., et al. Snapkv. arXiv:2404.14469, 2024.  
[39] Lin, J., et al. Awq. MLSys, 2024.  
[40] Liu, Z., et al. Scissorhands. NeurIPS, 2024.  
[41] Liu, Z., et al. Kivi. arXiv:2402.02750, 2024.  
[42] Lloyd, S. Least squares quantization in pcm. IEEE TIT, 1982.  
[43] Max, J. Quantizing for minimum distortion. IRE TIT, 1960.  
[44] Panter, P. and Dite, W. Quantization distortion in PCM. Proc. IRE, 1951.  
[45] Pennington, J., Socher, R., and Manning, C. GloVe. EMNLP, 2014.  
[46] Santhanam, K., et al. Colbertv2. arXiv:2112.01488, 2021.  
[47] Shah, J., et al. Flashattention 3. arXiv:2407.08608, 2024.  
[48] Shannon, C. E. A mathematical theory of communication. BSTJ, 1948.  
[49] Shannon, C. E. Coding theorems for a discrete source with a fidelity criterion. IRE Nat. Conv. Rec, 1959.  
[50] Shazeer, N. Fast transformer decoding. arXiv:1911.02150, 2019.  
[51] Su, Z., et al. Rotatekv. arXiv:2501.16383, 2025.  
[52] Team, G., et al. Gemini 1.5. arXiv:2403.05530, 2024.  
[53] Thakur, N., et al. BEIR. NeurIPS Datasets, 2021.  
[54] Vaswani, A., et al. Attention is all you need. NeurIPS, 2017.  
[55] Vershynin, R. High-dimensional probability. CUP, 2018.  
[56] Wang, J., et al. A survey on learning to hash. IEEE TPAMI, 2017.  
[57] Xiao, G., et al. Smoothquant. ICML, 2023.  
[58] Xiao, G., et al. Attention sinks. arXiv:2309.17453, 2023.  
[59] Yang, J. Y., et al. No token left behind. arXiv:2402.18096, 2024.  
[60] Yue, Y., et al. Wkvquant. arXiv:2402.12065, 2024.  
[61] Zador, P. L. Development and evaluation of procedures for quantizing multivariate distributions. PhD thesis, Stanford, 1964.  
[62] Zandieh, A., Daliri, M., and Han, I. Qjl: 1-bit quantized jl transform for kv cache. arXiv:2406.03482, 2024.  
[63] Zandieh, A., Daliri, M., and Han, I. Qjl (preprint duplicate). arXiv:2406.03482, 2024.  
[64] Zandieh, A., et al. Subgen. arXiv:2402.06082, 2024.  
[65] Zhang, T., et al. Kv cache is 1 bit per channel. arXiv:2405.03917, 2024.  
[66] Zhang, Z., et al. H2o. NeurIPS, 2024.  

---

*Fim da versão PT-BR organizada. Para equações e provas completas no detalhe dos autores, use o PDF do arXiv.*
