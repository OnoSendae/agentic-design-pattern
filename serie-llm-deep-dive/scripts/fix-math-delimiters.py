#!/usr/bin/env python3
"""
fix-math-delimiters.py
======================

Converte delimitadores LaTeX `\\(...\\)` e `\\[...\\]` para os equivalentes
suportados pelo GitHub MathJax (`$...$` e `$$...$$`), preservando blocos de
código fenced (``` ou ~~~).

Uso:
    python fix-math-delimiters.py <root>             # aplica mudanças
    python fix-math-delimiters.py <root> --dry-run   # apenas relatório
    python fix-math-delimiters.py <root> --diff      # mostra diff por arquivo

Regras:
    - Inline:  \\(X\\)  →  $X$            (X strip de espaços)
    - Display: \\[X\\]  →  \\n\\n$$\\nX\\n$$\\n\\n
    - Não toca em conteúdo dentro de blocos ``` ou ~~~
    - Detecta cifrões órfãos remanescentes (sanity check)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_by_fences(text: str) -> list[tuple[bool, str]]:
    """Quebra o texto em segmentos (is_code, content) preservando blocos fenced."""
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


def convert_display(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        inner = m.group(1).strip()
        count += 1
        return f"\n\n$$\n{inner}\n$$\n\n"

    new = DISPLAY_RE.sub(repl, text)
    new = re.sub(r"\n{3,}", "\n\n", new)
    return new, count


def convert_inline(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        inner = m.group(1).strip()
        count += 1
        return f"${inner}$"

    new = INLINE_RE.sub(repl, text)
    return new, count


def process_text(text: str) -> tuple[str, int, int]:
    segments = split_by_fences(text)
    out_parts: list[str] = []
    inline_total = 0
    display_total = 0
    for is_code, content in segments:
        if is_code:
            out_parts.append(content)
        else:
            content, n_disp = convert_display(content)
            content, n_inl = convert_inline(content)
            display_total += n_disp
            inline_total += n_inl
            out_parts.append(content)
    return "\n".join(out_parts), inline_total, display_total


def sanity_orphans(original: str, converted: str) -> tuple[int, int]:
    """Conta quantos `\\(`, `\\)`, `\\[`, `\\]` *suspeitos* restam fora de fences.

    Ignora padrões LaTeX legítimos que aparecem DENTRO de blocos $$...$$:
    - `\\\\[Npt]` — line-break com espaçamento (ex.: `\\\\[3pt]`)
    - `\\\\[Nem]`, `\\\\[Nex]`, `\\\\[Nmm]` — outras unidades de espaçamento
    """
    legit_break = re.compile(r"\\\\\[\d+(?:pt|em|ex|mm|cm|in)\]")
    segs = split_by_fences(converted)
    orphan_paren = 0
    orphan_bracket = 0
    for is_code, content in segs:
        if is_code:
            continue
        cleaned = legit_break.sub("", content)
        orphan_paren += len(re.findall(r"\\[\(\)]", cleaned))
        orphan_bracket += len(re.findall(r"\\[\[\]]", cleaned))
    return orphan_paren, orphan_bracket


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--diff", action="store_true", help="mostra diff dos arquivos alterados")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="padrão glob a excluir (relativo à root)",
    )
    args = p.parse_args()

    files = sorted(args.root.rglob("*.md"))
    for excl in args.exclude:
        files = [f for f in files if not f.match(excl)]

    print(f"Processando {len(files)} arquivos em {args.root}\n")
    print(f"{'Arquivo':<72} {'Inline':>7} {'Display':>8} {'Orfão':>6}")
    print("-" * 96)

    grand_inline = 0
    grand_display = 0
    grand_orphan = 0
    changed_files = 0

    for f in files:
        original = f.read_text(encoding="utf-8")
        new_text, n_inl, n_disp = process_text(original)
        op, ob = sanity_orphans(original, new_text)
        orphan_total = op + ob
        if n_inl or n_disp or orphan_total:
            rel = f.relative_to(args.root)
            print(f"{str(rel):<72} {n_inl:>7} {n_disp:>8} {orphan_total:>6}")
            grand_inline += n_inl
            grand_display += n_disp
            grand_orphan += orphan_total
            if n_inl or n_disp:
                changed_files += 1
                if not args.dry_run:
                    f.write_text(new_text, encoding="utf-8")
                if args.diff:
                    import difflib

                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile=str(rel) + " (antes)",
                        tofile=str(rel) + " (depois)",
                        n=2,
                    )
                    sys.stdout.writelines(diff)
                    print()

    print("-" * 96)
    print(f"{'TOTAL':<72} {grand_inline:>7} {grand_display:>8} {grand_orphan:>6}")
    print(f"\nArquivos {'que seriam alterados' if args.dry_run else 'alterados'}: {changed_files}")
    if grand_orphan:
        print(
            f"\n⚠  {grand_orphan} delimitadores órfãos ainda restam fora de blocos de código."
            "\n   Verifique manualmente — podem ser casos atípicos (parênteses escapados etc.)."
        )
    if args.dry_run:
        print("\nModo dry-run: nenhum arquivo foi modificado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
