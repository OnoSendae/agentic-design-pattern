#!/usr/bin/env python3
"""
escape-currency.py
==================

Escapa o cifrão de símbolos monetários (R$, US$, AU$, CA$, etc.) em arquivos
Markdown para evitar que o GitHub MathJax interprete pares de `$` como
delimitadores de fórmula.

Conversão:
    R$ 5k    →  R\\$ 5k
    US$ 600  →  US\\$ 600
    R$200    →  R\\$200
    R\\$ ... →  inalterado (já escapado)

Preserva blocos de código fenced (``` e ~~~).

Uso:
    python escape-currency.py <root>             # aplica
    python escape-currency.py <root> --dry-run   # apenas relatório
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CURRENCY_RE = re.compile(
    r"(?<!\\)\b((?:R|US|AU|CA|HK|NZ|S|NT|MX|AR|CL|CO|UY)\$)(?=\s*\d)"
)

# Math display $$...$$ (multilinha)
DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]+?\$\$")

# Math inline $...$ válido segundo a heurística do GitHub:
#  - $ de abertura precedido por: início, espaço, pontuação (não letra/dígito/$/\)
#  - $ de abertura NÃO seguido por espaço
#  - conteúdo sem $, \n, e sem espaço encostando no $ de fecho
#  - $ de fecho NÃO seguido por letra/dígito
INLINE_MATH_RE = re.compile(
    r"(?<![\\\w$])"
    r"\$"
    r"(?=\S)"
    r"([^\$\n]+?)"
    r"(?<=\S)"
    r"\$"
    r"(?!\w)"
)

# Cifrão "solto" candidato a escape: $ seguido de dígito (USD implícito)
LOOSE_DOLLAR_RE = re.compile(r"(?<![\\\w$])\$(?=\d)")

FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_by_fences(text: str) -> list[tuple[bool, str]]:
    segments: list[tuple[bool, str]] = []
    lines = text.split("\n")
    buf: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            if not in_fence:
                segments.append((False, "\n".join(buf)))
                buf = [line]
                in_fence = True
            else:
                buf.append(line)
                segments.append((True, "\n".join(buf)))
                buf = []
                in_fence = False
        else:
            buf.append(line)
    if buf:
        segments.append((in_fence, "\n".join(buf)))
    return segments


def escape_in_text(text: str, loose: bool = False) -> tuple[str, int]:
    count = 0

    def repl_currency(m: re.Match[str]) -> str:
        nonlocal count
        prefix = m.group(1)[:-1]
        count += 1
        return f"{prefix}\\$"

    new = CURRENCY_RE.sub(repl_currency, text)

    if loose:
        # Etapa 1: proteger blocos matemáticos válidos com placeholders.
        protected: list[str] = []

        def protect(m: re.Match[str]) -> str:
            protected.append(m.group(0))
            return f"\x00MATH{len(protected) - 1}\x00"

        new = DISPLAY_MATH_RE.sub(protect, new)
        new = INLINE_MATH_RE.sub(protect, new)

        # Etapa 2: escapar cifrões soltos restantes (USD implícito).
        def repl_loose(_m: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return r"\$"

        new = LOOSE_DOLLAR_RE.sub(repl_loose, new)

        # Etapa 3: restaurar placeholders.
        def restore(m: re.Match[str]) -> str:
            return protected[int(m.group(1))]

        new = re.sub(r"\x00MATH(\d+)\x00", restore, new)

    return new, count


def process_text(text: str, loose: bool = False) -> tuple[str, int]:
    segments = split_by_fences(text)
    out_parts: list[str] = []
    total = 0
    for is_code, content in segments:
        if is_code:
            out_parts.append(content)
        else:
            content, n = escape_in_text(content, loose=loose)
            total += n
            out_parts.append(content)
    return "\n".join(out_parts), total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--loose-dollars",
        action="store_true",
        help="também escapa $123 sem prefixo (USD implícito), útil em textos sobre custos",
    )
    args = p.parse_args()

    files = sorted(args.root.rglob("*.md"))
    print(f"Processando {len(files)} arquivos em {args.root}\n")
    print(f"{'Arquivo':<78} {'Escapados':>10}")
    print("-" * 90)

    grand = 0
    changed = 0
    for f in files:
        original = f.read_text(encoding="utf-8")
        new_text, n = process_text(original, loose=args.loose_dollars)
        if n:
            rel = f.relative_to(args.root)
            print(f"{str(rel):<78} {n:>10}")
            grand += n
            changed += 1
            if not args.dry_run:
                f.write_text(new_text, encoding="utf-8")

    print("-" * 90)
    print(f"{'TOTAL':<78} {grand:>10}")
    print(f"\nArquivos {'que seriam alterados' if args.dry_run else 'alterados'}: {changed}")
    if args.dry_run:
        print("Modo dry-run: nenhum arquivo foi modificado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
