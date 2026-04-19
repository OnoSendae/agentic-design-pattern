# Documentação didática — TurboQuant (arXiv:2504.19874)

Série gerada a partir de `../paper-turboquant-cp.md` e do PDF `../2504.19874v1.pdf`, com pesquisa complementar na web (subagentes em paralelo). Idioma: português (BR).

## Ordem de leitura sugerida

| # | Arquivo | Conteúdo (mapa do paper) |
|---|---------|---------------------------|
| 01 | [01-fundamentos-e-definicao-formal.md](./01-fundamentos-e-definicao-formal.md) | Abstract, introdução (LLM, KV, vector DB), §1.1: $Q$, $Q^{-1}$, $D_{\mathrm{mse}}$, $D_{\mathrm{prod}}$, não-viés, Quant/DeQuant |
| 02 | [02-trabalhos-relacionados.md](./02-trabalhos-relacionados.md) | §1.2: Shannon/Zador/Gersho, online vs offline, KV, PQ vs grid |
| 03 | [03-preliminares-beta-esfera-e-concentracao.md](./03-preliminares-beta-esfera-e-concentracao.md) | Lema 1 (Beta na esfera), concentração, independência aproximada das coordenadas |
| 04 | [04-shannon-lower-bound.md](./04-shannon-lower-bound.md) | §2.1: SLB (Lemas 2–3), ligação com cota $4^{-b}$ |
| 05 | [05-qjl-quantized-johnson-lindenstrauss.md](./05-qjl-quantized-johnson-lindenstrauss.md) | §2.2: definição QJL, Lema 4 (não-viés, variância) |
| 06 | [06-turboquant-mse-e-produto-interno.md](./06-turboquant-mse-e-produto-interno.md) | §3.1 MSE (Lloyd–Max, Algoritmo 1, Teorema 1); §3.2 IP em dois estágios (Teorema 2) |
| 07 | [07-limites-inferiores-e-experimentos.md](./07-limites-inferiores-e-experimentos.md) | §3.3 Teorema 3 (Yao, pigeonhole); §4 experimentos (DBpedia, NIAH, LongBench, NN vs PQ) |

## Nota sobre a constante “≈ 2,7” no resumo do PDF

O texto do artigo escreve algo como $\sqrt{3\pi}/2 \approx 2{,}7$. Numericamente, $\sqrt{3\pi}/2 \approx 1{,}53$. Um valor próximo de **2,7** corresponde melhor a **$\sqrt{3}\,\pi/2$** (ordem dos símbolos na raiz). O documento **07** discute o alinhamento das **cotas superiores** dos Teoremas 1–2 com o termo **$4^{-b}$** da cota inferior e evita confundir com o número do abstract.

## Cópia espelho

O ficheiro `../turboquant-teoria-3.3-e-experimentos-sec4.md` coincide com o **07** (gerado pelo mesmo fluxo de análise).
