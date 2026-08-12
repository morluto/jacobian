#!/usr/bin/env python3
"""Search persistent evaluation files for source, artifact, or root-cause overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".txt", ".yaml", ".yml"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Case-insensitive overlap phrase")
    parser.add_argument("roots", nargs="+", type=Path, help="Report or trajectory roots")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    needle = args.query.casefold()
    found = 0
    for root in args.roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if needle in line.casefold():
                    print(f"{path}:{number}:{line.strip()}")
                    found += 1
                    if found >= args.limit:
                        return 0
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
