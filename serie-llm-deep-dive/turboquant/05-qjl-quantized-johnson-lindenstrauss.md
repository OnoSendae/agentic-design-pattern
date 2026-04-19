# QJL (Quantized Johnson–Lindenstrauss): definição, Lema 4 e produto interno

## Contexto no TurboQuant

O texto do paper explica que quantizadores ótimos em **MSE** nem sempre dão **estimadores não enviesados** do produto interno, com viés acentuado em **baixa largura de bits**. A solução em duas etapas usa, na última etapa (1 bit), o **QJL** — “optimal inner product quantizer” com **1 bit por coordenada**, conforme a referência **[62]** (no material que você tem, isso corresponde ao trabalho **QJL: 1-Bit Quantized JL Transform**, publicado no **arXiv [2406.03482](https://arxiv.org/abs/2406.03482)** e nas **AAAI 2025**; autores: Amir Zandieh, Majid Daliri, Insu Han).

---

## Definição 1 (QJL) — o que o mapa faz

Para dimensão \(d \in \mathbb{Z}_{>0}\):

1. **Matriz aleatória** \(S \in \mathbb{R}^{d \times d}\) com entradas **i.i.d.** \(\mathcal{N}(0,1)\).
2. **Quantização (1 bit por coordenada):**  
   \[
   Q_{\mathrm{qjl}}(x) \;=\; \mathrm{sign}(S x),
   \]
   com `sign` aplicado **coordenada a coordenada** ao vetor \(Sx\) (convenciona-se \(\mathrm{sign}(0)\) de forma consistente com [62]; o paper cita o Lema 3.2 deles para o não-viés).
3. **Dequantização / reconstrução linear:**  
   \[
   Q_{\mathrm{qjl}}^{-1}(z) \;=\; \sqrt{\frac{\pi}{2d}}\; S^{\top} z
   \quad\text{para } z \in \{-1,+1\}^d.
   \]

**Interpretação:** primeiro você **mistura** \(x\) com uma transformação tipo JL (projeção aleatória gaussiana em \(\mathbb{R}^d\)), depois guarda só o **sinal** de cada coordenada misturada. Na “volta”, você **desfaz** a mistura com \(S^{\top}\) e **corrige a escala** com \(\sqrt{\pi/(2d)}\) para que combinações lineares (em especial o produto interno com outro vetor) tenham **média correta**.

O fator \(\sqrt{\pi/2}\) está ligado ao momento de \(|\xi|\) quando \(\xi \sim \mathcal{N}(0,1)\): \(\mathbb{E}[|\xi|] = \sqrt{2/\pi}\), o que aparece ao relacionar \(\mathrm{sign}(s_i^\top x)\) com projeções gaussianas e corrigir o encolhimento introduzido pelo corte de 1 bit.

---

## Caixa de algoritmo (codificação e decodificação)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Algoritmo QJL (Definição 1 — como no TurboQuant)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Entrada global: dimensão d; amostrar S ∈ R^{d×d} com N(0,1) i.i.d.      │
│                                                                         │
│ Encode(x ∈ R^d):                                                        │
│   u ← S · x                    ▷ mistura JL (vetor em R^d)              │
│   z ← sign(u)                  ▷ 1 bit por coordenada, z ∈ {±1}^d       │
│   retornar z                                                            │
│                                                                         │
│ Decode(z ∈ {±1}^d):                                                     │
│   x̂ ← sqrt(π / (2d)) · (S^T · z)                                        │
│   retornar x̂                                                           │
│                                                                         │
│ Estimador de produto interno ⟨y, x⟩ (x no paper frequentemente em S^{d-1}):│
│   IP̂(y, x) ← ⟨ y , Decode(Encode(x)) ⟩  = ⟨ y , sqrt(π/(2d)) S^T sign(Sx) ⟩ │
└─────────────────────────────────────────────────────────────────────────┘
```

Forma explícita usada na prova do Lema 4 (média de \(d\) termos i.i.d.):

\[
\Big\langle y,\, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x)) \Big\rangle
= \frac{1}{d}\sum_{i=1}^{d} \sqrt{\frac{\pi}{2}}\, (s_i^{\top} y)\,\mathrm{sign}(s_i^{\top} x),
\]

onde \(s_i^{\top}\) é a \(i\)-ésima linha de \(S\).

---

## Lema 4 (garantia) — não-viés e variância

Para \(x \in \mathcal{S}^{d-1}\) (esfera unitária) e qualquer \(y \in \mathbb{R}^d\):

| Propriedade | Enunciado |
|-------------|-----------|
| **Não-viés** | \(\displaystyle \mathbb{E}\Big[\big\langle y,\, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x))\big\rangle\Big] = \langle y, x\rangle\). |
| **Variância** | \(\displaystyle \mathrm{Var}\Big(\big\langle y,\, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x))\big\rangle\Big) \le \frac{\pi}{2d}\,\|y\|_2^2\). |

**Escalamento da variância:** a cota é **\(O(\|y\|_2^2 / d)\)** — mais precisamente constante \(\pi/2\) vezes \(\|y\|_2^2/d\). O paper decompõe o estimador como média de \(d\) variáveis i.i.d. \(z_i = \sqrt{\pi/2}\,(s_i^{\top}y)\,\mathrm{sign}(s_i^{\top}x)\), usa \(\mathrm{Var}(z_i) \le (\pi/2)\,\mathbb{E}[(s_i^{\top}y)^2] = (\pi/2)\|y\|_2^2\) (Fato 3.4 de [62]), e então \(\mathrm{Var}(\text{média}) \le \frac{1}{d^2}\sum_i \mathrm{Var}(z_i) \le \frac{\pi}{2d}\|y\|_2^2\).

---

## Por que o QJL “conserta” a estimação do produto interno

1. **Quantização MSE-optimal** otimiza erro \(\ell_2\) da reconstrução **ponto a ponto**, não a estatística \(\langle y, \hat{x}\rangle\). Com poucos bits, o **centroide / arredondamento** pode **empurrar** \(\hat{x}\) de modo sistemático; ao formar \(\langle y, \hat{x}\rangle\), isso vira **viés** (erro médio \(\neq 0\)).
2. O QJL escolhe **binarização por sinal** em espaço **pré-misturado** por \(S\) e define **dequantização linear** \(S^{\top}\) com escala \(\sqrt{\pi/(2d)}\) de forma que, **marginalmente**, o acoplamento \(y\)–\(x\) através dos sinais das projeções aleatórias tenha **esperança exata** \(\langle y,x\rangle\) (Lema 4 / Lema 3.2 de [62]).
3. Assim, para **mesmo orçamento extremo de 1 bit/coord**, você troca “reconstrução MSE bonita” por **garantia estatística de IP**: média certa + variância que **decai com \(1/d\)**, o que é o regime usual de **concentração de médias** de estimadores JL.

---

## Ligação com Johnson–Lindenstrauss (JL)

- **JL clássico:** uma matriz aleatória (gaussiana, subgaussiana, etc.) **preserva com alta probabilidade** distâncias (e, por polarização, **produtos internos**) entre um conjunto finito de pontos, após **escala** adequada — “geometria em dimensão menor (ou misturada) permanece fiel”.
- **QJL:** é a **mesma ideia de pré-condicionamento aleatório** (papel de \(S\)), mas a **memória** guardada é **só o sinal** de \(Sx\). O par **\(\mathrm{sign}(Sx)\)** + **reconstrução \(\propto S^{\top}z\)** é calibrado para que o **funcional linear** \(\langle y, \cdot\rangle\) não seja sistematicamente distorcido — ou seja, **JL “quantizado”** no sentido de **1 bit por coordenada após a mistura**, com **estimador de IP não enviesado** em vez de apenas preservação bilinear não-linear sem correção.

Em resumo: **JL** explica *por que* misturar com \(S\) preserva geometria; **QJL** explica *como* manter essa geometria **no nível de estimação de produto interno** depois de uma **quantização brutal** (sinal), via **escala** \(\sqrt{\pi/(2d)}\) e estrutura \(S^{\top}\).

---

## Notas de complexidade e memória (conforme a Def. 1 “densa”)

| Operação | Custo típico (S denso \(d \times d\)) | Comentário |
|----------|--------------------------------------|------------|
| \(Sx\) | \(O(d^2)\) | Uma matvec densa. |
| \(S^{\top}z\) | \(O(d^2)\) | Idem na decodificação. |
| Armazenar \(S\) | \(O(d^2)\) números | Em aplicações grandes, variantes **esparsas / SRHT / blocos** de JL reduzem custo; o texto do TurboQuant usa a forma teórica completa para o Lema. |
| Largura de bits | **1 bit/coord** após mistura | Código \(z \in \{\pm1\}^d\). |

A **variância** do estimador de \(\langle y,x\rangle\) escala como **\(\Theta(1/d)\)** em termos de \(d\) (cota explícita \(\frac{\pi}{2d}\|y\|_2^2\)), o que é o comportamento desejável para **média de \(d\) contribuições independentes**.

---

## Referências citáveis

- **[62] / QJL:** Amir Zandieh, Majid Daliri, Insu Han, *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead*, [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (2024); proceedings AAAI (2025). Repositório: [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL).
- **TurboQuant (seu extrato):** Seção **2.2**, Definição 1 e Lema 4 em `paper-turboquant-cp.md` (linhas ~473–599 no arquivo que você indicou).

---

### Trecho-base no paper (Definição 1 e Lema 4)

*Referência:* `paper-turboquant-cp.md`, sec. 2.2 (por volta das linhas 481–599).

**Definição 1 (QJL).** Para qualquer inteiro positivo \(d\), o mapa \(Q_{\mathrm{qjl}} : \mathbb{R}^d \to \{-1,+1\}^d\) é

$$
Q_{\mathrm{qjl}}(x) := \operatorname{sign}(S\,x), \qquad x \in \mathbb{R}^d,
$$

onde \(S \in \mathbb{R}^{d \times d}\) tem entradas i.i.d. \(\mathcal{N}(0,1)\) e \(\operatorname{sign}(\cdot)\) é aplicado coordenada a coordenada. A **dequantização** é

$$
Q_{\mathrm{qjl}}^{-1}(z) := \sqrt{\frac{\pi}{2d}}\, S^{\top} z, \qquad z \in \{-1,+1\}^d.
$$

**Lema 4 (garantia do QJL).** Para \(Q_{\mathrm{qjl}}\) e \(Q_{\mathrm{qjl}}^{-1}\) como acima, qualquer \(x \in S^{d-1}\) e \(y \in \mathbb{R}^d\) satisfazem:

- **Não-viés:** \(\mathbb{E}\big[\langle y,\, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x))\rangle\big] = \langle y, x\rangle\).
- **Cota de variância:** \(\mathrm{Var}\big(\langle y,\, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x))\rangle\big) \le \dfrac{\pi}{2d}\,\|y\|_2^2\).