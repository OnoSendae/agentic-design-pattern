# Seção 2.1 — *Shannon Lower Bound* (SLB) sobre distorção

## Papel na teoria da informação

O **teorema de codificação com perdas de Shannon** (rate–distortion) descreve o melhor compromisso possível entre **taxa** (quanto de “informação” sobre a fonte você transmite ou retém) e **distorção** (quão longe a reconstrução fica do original), para uma medida de fidelidade fixa. O **Shannon Lower Bound** é uma **cota inferior universal**: para uma dada taxa (aqui medida por **informação mútua** em **bits**), **nenhum** esquema de compressão com perdas pode ficar abaixo dessa distorção média, **independentemente** do algoritmo — desde que a fonte tenha **entropia diferencial finita**.

No paper, a medida é **EQM (MSE)** em \(\mathbb{R}^d\): \(\mathbb{E}\|x - y\|_2^2\). O par \((x,y)\) é qualquer acoplamento (vetor fonte, vetor reconstrução) respeitando o limite de informação mútua.

---

## Lema 2 (SLB geral com entropia diferencial)

**Enunciado (como no paper).** Seja \(x \in \mathbb{R}^d\) com densidade \(p_X\) e **entropia diferencial** \(h(x)\) **finita** (no mesmo base dos logs usados na fórmula — no paper, base \(2\) para bits). Defina a função distorção–taxa com orçamento de **\(B\)** bits:

\[
D(p_X, B) := \inf \Big\{ \mathbb{E}\|x - y\|_2^2 : I(x; y) \le B \Big\},
\]

onde o ínfimo é sobre **todas** as distribuições conjuntas de \((x,y)\) com \(y \in \mathbb{R}^d\). Então, para todo \(B \ge 0\),

\[
D(p_X, B) \;\ge\; \frac{d}{2\pi e} \cdot 2^{\frac{2}{d}\big(h(x) - B\big)}.
\]

### Interpretação em **bits por dimensão**

- **\(B\)** é o número **total** de bits “úteis” que o canal reconstrução–fonte pode usar, no sentido de **\(I(x;y) \le B\)** (não é só contar bits do índice do quantizador; é o limite informacional do acoplamento).
- **Taxa por dimensão:** \(R := B/d\) (bits/dimensão).
- A desigualdade pode ser lida assim: a **distorção mínima possível** escala com \(2^{2(h(x)/d - R)}\) até fatores que dependem só de \(d\) e da base logarítmica embutida em \(2\pi e\).  
  - **Mais bits por dimensão** (\(R\) maior \(\Rightarrow\) \(B\) maior) **reduz** a cota inferior de distorção **exponencialmente** com expoente **\(2/d\)** na diferença \(h(x)/d - R\): cada bit/dimensão “comprado” na informação mútua permite, no máximo, encolher a distorção na escala \(2^{-2/d}\) **por bit total**… na forma exponencial completa entram ainda \(h(x)\) e o fator geométrico \(d/(2\pi e)\).

### Por que aparece \(2\pi e\) e o **canal de teste gaussiano reverso**

A prova clássica (citada no paper, p.ex. Berger/Cover–Thomas) usa o **“backward Gaussian test channel”**:

1. Imagina-se uma reconstrução \(y\) e um ruído gaussiano **aditivo** ligando \(y\) a \(x\) na direção “reversa” da causalidade usual do quantizador (daí *backward*): o acoplamento ótimo ou o acoplamento que **atinge a cota** é comparado ao caso gaussiano, onde a relação taxa–distorção sob MSE é explícita.
2. Para **fonte gaussiana** escalar/vectorial com MSE, a cota de Shannon coincide com a realidade; para **fonte não gaussiana** com a **mesma entropia diferencial**, a **mesma** taxa não pode dar distorção menor — daí “**lower bound**”: a gaussiana é, em certo sentido, a mais “difícil” de comprimir para uma dada entropia sob MSE.

O fator **\(2\pi e\)** é o que liga a **variância** de um Gaussiano à sua entropia diferencial (\(h(\mathcal{N}(0,\sigma^2)) = \tfrac{1}{2}\log(2\pi e \sigma^2)\) em nats, etc.): a cota expressa uma **área mínima de bola de erro** em \(\mathbb{R}^d\) compatível com \(B\) bits de descrição conjunta.

---

## Lema 3 (uniforme na esfera \(\Rightarrow\) \(D(B) \ge 2^{-2B/d}\))

**Enunciado.** Seja \(x\) **uniforme** em \(S^{d-1}\) (esfera unitária em \(\mathbb{R}^d\)), com \(D(B)\) definido como no Lema 2. Então

\[
D(B) \;\ge\; 2^{-2B/d}.
\]

### Esboço da prova (como no paper)

1. A densidade é uniforme na superfície; a entropia diferencial (em base 2) satisfaz \(h(x) = \log_2 A_d\), onde \(A_d\) é a **área** de \(S^{d-1}\) (superfície).
2. Substitui-se no Lema 2:
   \[
   D(B) \;\ge\; \frac{d}{2\pi e} \cdot A_d^{2/d} \cdot 2^{-2B/d}.
   \]
3. Usa-se \(A_d = \dfrac{2\pi^{d/2}}{\Gamma(d/2)}\) e a **aproximação de Stirling** para \(\Gamma(d/2)\), obtendo
   \[
   A_d^{2/d} \;\ge\; \left(\frac{2\pi e}{d}\right)^{d/2 \cdot (2/d)} \times (\text{fator subpolinomial em } d) \;=\; \frac{2\pi e}{d} \times (1 - O(1/d))^{\text{forma simplificada}}.
   \]
   Na manipulação do paper, o produto \(\dfrac{d}{2\pi e} \cdot A_d^{2/d}\) **cancela assintoticamente** o prefactor até **ordem principal**, restando a forma limpa
   \[
   D(B) \;\ge\; 2^{-2B/d}
   \]
   (o paper enuncia a desigualdade final nessa forma; o passo intermediário exibe explicitamente \(\frac{d}{2\pi e} A_d^{2/d} 2^{-2B/d}\)).

### Leitura intuitiva

Para uma fonte **só na casca** da esfera, a “complexidade intrínseca” \(h(x)\) cresce com \(d\) de modo que, **após** normalizar com \(d\) e usar a geometria de alta dimensão, a cota universal do Lema 2 se **colapsa** numa lei simples: **distorção mínima \(\gtrsim 2^{-2 \cdot (\text{bits/dimensão})}\)** quando se escreve \(B = d \cdot R\):

\[
D(B) \ge 2^{-2B/d} = 2^{-2R}.
\]

Ou seja: **cada bit por dimensão** pode, no limite dessa cota, reduzir no máximo a ordem de grandeza da EQM na base **\(4\)** por dimensão (\(2^{-2R} = 4^{-R}\)), neste regime geométrico.

---

## Ligação com o **Teorema 3** (mais adiante no paper)

No **Teorema 3**, o algoritmo tem **largura de bits \(b\)** e mapeia para **\(b \cdot d\)** bits no total: \(Q : S^{d-1} \to \{0,1\}^{b d}\). Tomando **\(B = b d\)** no Lema 3:

\[
D_{\mathrm{mse}} \;\ge\; 2^{-2(bd)/d} \;=\; 2^{-2b} \;=\; \frac{1}{4^b},
\]

que é exatamente a cota **\(D_{\mathrm{mse}}(Q) \ge 1/4^b\)** que o paper prova para instâncias difíceis, usando **Yao (minimax)** para reduzir “pior caso com algoritmo aleatorizado” a “fonte aleatorizada + algoritmo determinístico”, e então aplicar o **SLB na esfera** (Lema 3).

**Resumo da cadeia lógica:**  
**Teoria da informação (Lema 2)** \(\rightarrow\) **geometria da esfera + Stirling (Lema 3)** \(\rightarrow\) **\(2^{-2b}\) para \(B = bd\)** \(\rightarrow\) **Teorema 3** (limite inferior para qualquer quantizador aleatorizado na esfera, além da parte de produto interno que o paper deriva a partir da EQM).

---

## Referências conceituais (leitura extra)

- **Rate–distortion** e **Shannon lower bound** sob MSE: tratamentos padrão em Berger e Cover–Thomas.  
- **Backward Gaussian test channel:** construção clássica para obter cotas e (em fontes gaussianas) igualdade na função distorção–taxa.  
- Artigo da IEEE IT Society sobre *Shannon bounds for quadratic rate-distortion problems*: [Shannon Bounds for Quadratic Rate-Distortion Problems](https://www.itsoc.org/jsait/jsait-article/shannon-bounds-quadratic-rate-distortion-problems).

---

*Trecho base do paper (Seção 2.1, Lemas 2–3): fórmula geral com \(h(x)\) e especialização uniforme em \(S^{d-1}\) com prova via área \(A_d\) e Stirling; Teorema 3 usa \(B = bd\) para obter \(1/4^b\).*