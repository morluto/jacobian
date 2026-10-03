"""Rewrite recorded ``pytest.raises(..., match=...)`` sites into code assertions.

Reads the JSON emitted by ``tools/match_recorder.py`` and rewrites a site only
when the recorded evidence justifies it:

* a specific code was observed for that exact site, and
* that code is not a generic catch-all, and
* no *other* site in the same file reaches the same code through a different
  ``match=`` text (otherwise the rewrite would collapse distinguishable guards
  onto one identical assertion).

Anything else keeps its message match. That is deliberate: when the owner
publishes no stable code, the wording is the only discriminator, and the testing
strategy sanctions a message match exactly in that case.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

GENERIC_CODES = frozenset(
    {
        "value_error",
        "polynomial.invariant",
        "probability.model_invariant",
        "polynomial.multivariate_contract",
        "assertion_error",
    }
)
_RAISES = re.compile(r"^(\s*)with pytest\.raises\((?P<err>[A-Za-z_][A-Za-z0-9_]*)\)")


def load(path: Path) -> dict[str, dict[str, object]]:
    recorded: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recorded.update(json.loads(line))
    return recorded


def convertible_sites(
    recorded: dict[str, dict[str, object]],
) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    """Return sites safe to rewrite, plus per-file rejection counts."""
    # A code shared by differing match text within one file is not a
    # discriminator for those sites.
    by_code: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, info in recorded.items():
        path, _, lineno = key.rpartition(":")
        code = info.get("code")
        match = info.get("match")
        if (
            isinstance(code, str)
            and code not in GENERIC_CODES
            and isinstance(match, str)
        ):
            by_code[(path, code)].add(match)

    usable: dict[tuple[str, int], str] = {}
    rejected: dict[str, int] = defaultdict(int)
    for key, info in recorded.items():
        path, _, lineno = key.rpartition(":")
        code = info.get("code")
        match = info.get("match")
        if not isinstance(code, str) or not code:
            rejected["no_code"] += 1
            continue
        if code in GENERIC_CODES:
            rejected["generic_code"] += 1
            continue
        if len(by_code[(path, code)]) > 1:
            rejected["code_shared_by_other_messages"] += 1
            continue
        usable[(path, int(lineno))] = code
    return usable, rejected


def _targets(
    tree: ast.Module, sites: dict[int, str]
) -> list[tuple[int, int, int, str, str]]:
    """Locate each ``with pytest.raises(..., match=...)`` header by AST.

    Returns ``(start_line, end_line, error_class, code)`` 1-indexed and inclusive.
    """
    found: list[tuple[int, int, int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr == "raises"):
                continue
            if not any(keyword.arg == "match" for keyword in call.keywords):
                continue
            code = sites.get(call.lineno)
            if code is None:
                continue
            first = call.args[0] if call.args else None
            name = first.id if isinstance(first, ast.Name) else None
            if name is None:
                continue
            body_end: int = node.body[-1].end_lineno if node.body else call.lineno
            if body_end is None:
                continue
            header_end: int = call.end_lineno or call.lineno
            found.append((call.lineno, header_end, body_end, name, code))
    return found


def rewrite(path: Path, sites: dict[int, str]) -> int:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(path))
    targets = _targets(tree, sites)
    if not targets:
        return 0
    # Process outermost-first and never let one rewrite swallow another.
    targets.sort(key=lambda item: (item[0], -item[2]))
    filtered: list[tuple[int, int, int, str, str]] = []
    reach = 0
    for start, header_end, body_end, error_class, code in targets:
        if start <= reach:
            continue
        filtered.append((start, header_end, body_end, error_class, code))
        reach = body_end
    targets = filtered
    out: list[str] = []
    cursor = 0
    changed = 0
    for start, header_end, body_end, error_class, code in targets:
        out.extend(lines[cursor : start - 1])
        leading = re.match(r"\s*", lines[start - 1])
        indent = leading.group(0) if leading else ""
        out.append(f"{indent}with pytest.raises({error_class}) as exc_info:\n")
        # guarded body, 1-indexed header_end+1 .. body_end
        out.extend(lines[header_end:body_end])
        out.append(f'{indent}assert exc_info.value.errors()[0]["type"] == "{code}"\n')
        cursor = body_end
        changed += 1
    out.extend(lines[cursor:])
    result = "".join(out)
    ast.parse(result, filename=str(path))
    path.write_text(result, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="JSON emitted by match_recorder")
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recorded = load(args.record)
    usable, rejected = convertible_sites(recorded)
    per_file: dict[str, dict[int, str]] = defaultdict(dict)
    for (path, lineno), code in usable.items():
        per_file[path][lineno] = code

    total = 0
    for relative, sites in sorted(per_file.items()):
        target = args.root / relative
        if not target.exists():
            continue
        if args.dry_run:
            total += len(sites)
            continue
        count = rewrite(target, sites)
        total += count
        print(f"{relative}: rewrote {count}")
    print(f"total rewritten: {total}")
    for reason, count in sorted(rejected.items()):
        print(f"  left pinned ({reason}): {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
