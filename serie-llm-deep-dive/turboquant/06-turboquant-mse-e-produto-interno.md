# TurboQuant — Seções 3.1 e 3.2 (notas estruturadas)

Documento em duas partes: **Parte A** — TurboQuant MSE-ótimo; **Parte B** — TurboQuant para produto interno. O texto segue o paper (`paper-turboquant-cp.md`, linhas ~600–1075) e complementa com conceitos clássicos de **Lloyd–Max** e da decomposição **viés–variância** em quantização em estágios.

---

## Parte A — Seção 3.1: MSE Optimal TurboQuant

### Objetivo e ideia geral

Dado **x** na esfera unitária **S^{d−1}**, quer-se quantizar **x** com **b bits por coordenada**, minimizando o **MSE de reconstrução** (Eq. (1) do paper). O método separa três ideias:

1. **Aleatorizar a direção** com uma rotação aleatória, para que cada coordenada tenha uma distribuição tratável.
2. **Quantizar escalar** cada coordenada de forma **MSE-mínima** (Lloyd–Max / k-means 1D contínuo).
3. **Desfazer a rotação** na reconstrução, preservando normas e produtos internos sob rotação.

---

### Passo 1: matriz de rotação aleatória Π

- Gera-se **Π ∈ ℝ^{d×d}** aplicando **decomposição QR** a uma matriz aleatória com entradas **i.i.d. N(0,1)** (ortogonalização de Gram–Schmidt / QR).
- **y = Π x** está **uniformemente distribuído em S^{d−1}** quando **x** é fixo na esfera (rotação é uma isometria que “embaralha” direções de modo uniforme no grupo ortogonal).

**Consequência (Lemma 1 no paper):** cada coordenada **y_j** segue uma lei cuja densidade marginal em **[−1, 1]** é

\[
f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}\,(1-x^2)^{(d-3)/2},\quad x\in[-1,1].
\]

Em **alta dimensão**, essa marginal **aproxima uma normal** (paper: converge para normal; coordenadas tornam-se **quase independentes**), o que justifica tratar cada eixo com um quantizador escalar **independente**.

---

### Passo 2: “Beta por coordenada” (interpretação)

No texto do paper, a distribuição marginal das coordenadas da projeção em **S^{d−1}** é descrita com a densidade acima; em dimensão moderada/alta ela se comporta como uma **Beta “perto da normal”** no limite assintótico. O ponto operacional é: **não é uma Gaussiana genérica arbitrária** — é a **marginal da uniforme na esfera**, e o desenho dos centroides deve respeitar **f_X**.

---

### Passo 3: Lloyd–Max e Eq. (4) — k-means contínuo em 1D

Para uma densidade **f_X** conhecida, o quantizador escalar MSE-ótimo com **2^b níveis** resolve o **problema de Lloyd–Max**:

- Particionar **[−1, 1]** em **2^b** células.
- **Fronteiras de decisão:** pontos médios entre **centroides consecutivos** (tesselação de Voronoi 1D = intervalos entre midpoints) — condição **de Nearest-Neighbor ótimo**.
- **Centroides:** minimizam o erro quadrático médio ponderado por **f_X** — na prática, **média condicional** em cada intervalo (condição **de Lloyd**).

Na literatura (e na busca web), o algoritmo de **Lloyd** para fonte escalar é o mesmo espírito do **k-means em 1D com distribuição contínua**: alternar **fronteiras** e **reprodução (centroides)** até convergência.

O paper formular isso como **minimização explícita** do custo escalar:

**Eq. (4) — custo MSE escalar ótimo C(f_X, b):**

\[
C(f_X,b) := \min_{-1 \le c_1 \le \cdots \le c_{2^b} \le 1} \;
\sum_{i=1}^{2^b} \int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}} (x - c_i)^2 \, f_X(x)\, dx
\]

(com convenção nos extremos para **c_0, c_{2^b+1}** que fecham o intervalo **[−1,1]** como no paper: integrais entre midpoints sucessivos).

- **C(f_X, b)** é o **MSE por coordenada** do melhor quantizador escalar de **b bits** para a marginal **f_X**.
- Resolve-se **numericamente** (iterativo) para cada **b** desejado; os centroides são **pré-computados e armazenados** (código compartilhado).

**Exemplo (dimensão d moderadamente alta, marginal ≈ N(0, 1/d))** — paper dá centroides aproximados para **b = 1** e **b = 2**:

- **b = 1:** níveis da ordem **±√(2/(πd))** (no texto aparece como **±√(2/π)/√d**).
- **b = 2:** quatro níveis escalados com **O(1/√d)** (valores numéricos **~0.453/√d**, **~0.51/√d** no trecho lido).

---

### Algoritmo 1 — `TurboQuant_mse`: pseudocódigo explicado

**Pré-processamento (uma vez):**

1. Gerar **Π** (rotação aleatória).
2. Resolver **Eq. (4)** e obter o codebook **c_1, …, c_{2^b} ⊂ [−1, 1]**.

**`Quant_mse(x)` — codificação:**

3. **y ← Π x**.
4. Para cada coordenada **j**, **idx_j ← argmin_k |y_j − c_k|** (índice **b-bit**).
5. Saída: vetor de índices **idx**.

**`DeQuant_mse(idx)` — decodificação:**

6. **ỹ_j ← c_{idx_j}** (reconstrução por centroides nas coordenadas rotacionadas).
7. **x̃ ← Π^T ỹ** (rotação inversa).

O mapa **Q_mse : ℝ^d → {0,1}^{b d}** guarda apenas os índices; **Q_mse^{-1}** recupera centroides e aplica **Π^T**.

---

### Teorema 1 — limites de distorção (MSE ponta a ponta)

Para **x ∈ S^{d−1}**, com **x̃** produzido por `DeQuant_mse(Quant_mse(x))`:

**Cota geral (alta resolução / Panter–Dite no proof):**

\[
D_{\mathrm{mse}} := \mathbb{E}_{\tilde{x}}\big[\|x - \tilde{x}\|_2^2\big] \;\le\; \frac{\sqrt{3\pi}}{2}\,\frac{1}{4^{b}}
\]

(para **b** tal que a análise de alta resolução se aplica; o paper enuncia para **b ≥ 0** no trecho, com uso da fórmula Panter–Dite para **b > 4** na prova).

**Prova (esboço lógico do paper):**

1. Por ortogonalidade de **Π**: **‖x − x̃‖² = ‖y − ỹ‖²**.
2. Simetria das coordenadas: **D_mse = d · E[|y_1 − ĉ_{idx_1}|²] = d · C(f_X, b)**.
3. Para **b = 1,2,3,4**, **C** é obtido **numericamente** da Eq. (4).
4. Para **b > 4**, usa-se a fórmula de **Panter–Dite** para quantização escalar a taxa fixa, levando ao termo **(√3π)/(2d) · 1/4^b** para **C**, logo **D_mse = d·C ≤ (√3π/2)·1/4^b**.

### Tabela numérica — MSE para **b = 1 … 4** (do paper)

| **b** | **D_mse ≈** |
|------|-------------|
| 1 | 0.36 |
| 2 | 0.117 |
| 3 | 0.03 |
| 4 | 0.009 |

*(Valores “finos” na prova vêm de **C(f_X,b) ≈ (0.36/d, 0.117/d, 0.03/d, 0.009/d)**; multiplicar por **d** recupera a coluna acima.)*

### Encurtamento por entropia (opcional, paper)

Probabilidades **p_ℓ = ∫** sobre a célula **ℓ** com densidade **f_X**; codificação entrópica dos índices pode reduzir o **número médio de bits** sem mudar a distorção; o paper cita ganho modesto (ex.: **b = 4** → entropia **≈ 3.8**; redução média ~**5%**), e optam por simplicidade/velocidade sem isso no método base.

---

## Parte B — Seção 3.2: Inner-product TurboQuant

### Problema: quantizador MSE-ótimo é enviesado para produto interno

Para **ANN / similaridade**, interessa **𝔼[⟨y, x̂⟩] = ⟨y, x⟩** (estimador não enviesado do produto interno). O **TurboQuant_mse** **não** garante isso.

**Exemplo **b = 1** (paper):** codebook tende a **±√(2/(πd))**, logo **Q_mse** ~ **sign(Πx)** e **Q_mse^{-1}(z) ∝ Π^T z**. Pelo **Lemma 4**, para **d** grande:

\[
\mathbb{E}\big[\langle y, Q_{\mathrm{mse}}^{-1}(Q_{\mathrm{mse}}(x))\rangle\big] = \frac{2}{\pi}\,\langle y, x\rangle
\]

— **viés multiplicativo 2/π**. O viés **diminui** quando **b** cresce (comportamento empírico na Seção 4.1 do paper).

**Ligação à literatura / intuição:** quantizadores ótimos no sentido **MSE** reproduzem bem **médias quadráticas** (L2), mas as **médias lineares** dos valores reconstruídos podem **não** coincidir com as do sinal original — daí o **viés** em estimadores de **produto interno** baseados só em **x̃_mse**.

---

### Pipeline em duas etapas (b bits no total)

1. **Primeira etapa — MSE com b−1 bits por coordenada:** instanciar **TurboQuant_mse** com **bit-width b−1** (Algoritmo 1). Obter **x̃_mse = Q_mse^{-1}(Q_mse(x))**.
2. **Residual:** **r = x − x̃_mse**. O paper nota que **𝔼‖r‖** está ligada a **√C(f_X, b−1)** (escala do erro MSE da primeira etapa).
3. **Segunda etapa — QJL no residual:** aplicar **QJL** a **r**, i.e. projeção aleatória **S** com entradas Gaussianas e **sinal** (**q_jl = sign(S r)**), com escala adequada na desquantização.
4. **Orçamento de bits:** a primeira parte usa **(b−1)** bits por coordenada nos índices do **TurboQuant_mse**; o último “bit efetivo” por eixo entra via o esquema **sign** estilo JL no **residual**, mantendo **b bits** no agregado (como no pseudocódigo).

**Forma da estimativa de produto interno (paper):**

\[
\langle y, x\rangle \;\approx\; \big\langle y, Q_{\mathrm{mse}}^{-1}(Q_{\mathrm{mse}}(x))\big\rangle
+ \|r\|_2 \cdot \Big\langle y, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(r))\Big\rangle
\]

onde o segundo termo usa o **QJL** no **r** com reescala por **γ = ‖r‖₂** na reconstrução.

---

### Algoritmo 2 — `TurboQuant_prod` (resumo)

**Setup:**

- Instanciar **TurboQuant_mse** com **b−1**.
- Amostrar **S ∈ ℝ^{d×d}** com **S_{i,j} ~ N(0,1)** (matriz de projeção JL / QJL).

**`Quant_prod(x)`:**

1. **idx ← Quant_mse(x)** (com **b−1** bits/coord).
2. **r ← x − DeQuant_mse(idx)**.
3. **q_jl ← sign(S r)**.
4. Saída **(idx, q_jl, ‖r‖₂)**.

**`DeQuant_prod(idx, q_jl, γ)`:**

1. **x̃_mse ← DeQuant_mse(idx)**.
2. **x̃_qjl ← (√(π/2)/d) · γ · S^T q_jl** (escala compatível com o estimador JL não enviesado; **γ** é a norma do residual).
3. **x̃ ← x̃_mse + x̃_qjl**.

**Mapa composto:** **Q_prod** produz índices MSE, sinais QJL do residual e escalar **γ**.

---

### Teorema 2 — não enviesamento + limitador **D_prod**

Para **x ∈ S^{d−1}**, **y ∈ ℝ^d**, com **x̃** de `DeQuant_prod`:

1. **Não enviesamento do produto interno:**  
   \[
   \mathbb{E}_{\tilde{x}}[\langle y, \tilde{x}\rangle] = \langle y, x\rangle.
   \]

2. **Distorção do produto interno:**  
   \[
   D_{\mathrm{prod}} := \mathbb{E}_{\tilde{x}}\big[|\langle y,x\rangle - \langle y,\tilde{x}\rangle|^2\big]
   \;\le\; \frac{\sqrt{3\pi}}{2}\,\frac{\|y\|_2^2}{d}\,\frac{1}{4^{b}}.
   \]

3. **Valores mais finos para b = 1…4** (paper):  
   **D_prod ≈** **1.57/d**, **0.56/d**, **0.18/d**, **0.047/d**.

**Esboço da prova (condicionamento em x̃_mse):**

- **Passo 1 — esperança:** condicionar a **x̃_mse**. O termo QJL é construído para que  
  **𝔼[⟨y, x̃_qjl⟩ | x̃_mse] = ⟨y, r⟩** com **r = x − x̃_mse** (Lemma 4). Logo **𝔼[⟨y, x̃⟩|x̃_mse] = ⟨y, x̃_mse⟩ + ⟨y, r⟩ = ⟨y, x⟩** — pela lei da esperança total, **𝔼[⟨y, x̃⟩] = ⟨y, x⟩**.

- **Passo 2 — variância:**  
  \[
  \mathbb{E}\big[|\langle y,x\rangle - \langle y,\tilde{x}\rangle|^2 \,\big|\, \tilde{x}_{\mathrm{mse}}\big]
  = \mathrm{Var}_{\tilde{x}_{\mathrm{qjl}}}\big(\langle y, \tilde{x}_{\mathrm{qjl}}\rangle \,\big|\, \tilde{x}_{\mathrm{mse}}\big)
  \le \frac{\pi}{2d}\,\|r\|_2^2\,\|y\|_2^2
  \]
  (cota de variância do estimador QJL / Lemma 4).

- **Passo 3 — total:** **D_prod ≤ (π/(2d)) ‖y‖² · 𝔼‖x − x̃_mse‖² = (π/(2d)) ‖y‖² · D_mse** com **D_mse** para largura **b−1**; aplica-se o **Teorema 1** e obtém-se a cota em **1/4^b**.

**Interpretação viés–variância (residual, conexão conceitual com busca web):**

- A **primeira etapa** reduz energia em **r** (erro MSE pequeno em norma L2).
- A **segunda etapa** adiciona um estimador de **⟨y, r⟩** com **variância** controlada (QJL), escolhido para **corrigir o viés** médio do primeiro estágio no produto interno.
- O **Teorema 2** separa: **sem viés** na média do estimador de **⟨y,x⟩**; **crescimento controlado da variância** via **D_mse** da etapa MSE com **b−1** bits.

---

## Fluxo ponta a ponta (resumo em passos)

| Etapa | MSE (`3.1`) | IP (`3.2`) |
|--------|-------------|------------|
| 1 | Gerar **Π**, pré-computar centroides da Eq. (4) para **f_X** e **b** | Idem com **b−1** para a etapa MSE |
| 2 | **y = Πx**, quantizar cada **y_j** para índice **b-bit** | Obter **x̃_mse**, **r = x − x̃_mse** |
| 3 | **ỹ_j = c_{idx_j}**, **x̃ = Π^T ỹ** | **q_jl = sign(Sr)**, guardar **γ = ‖r‖** |
| 4 | Garantia **D_mse** (Teorema 1 + tabela **b=1…4**) | **x̃ = x̃_mse + x̃_qjl**; Teorema 2: **IP sem viés** e **D_prod** |

---

*Fonte primária: trecho do manuscript TurboQuant em `paper-turboquant-cp.md` (seções 3.1–3.2, Teoremas 1–2, Algoritmos 1–2). Conceitos Lloyd–Max / k-means 1D: literatura clássica de quantização MMSE; residual em dois estágios: decomposição conceitual viés (MSE) + correção estocástica (QJL) conforme o próprio paper.*