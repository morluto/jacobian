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
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypeGuard

try:  # pragma: no cover - import style depends on how the tool is invoked
    from tools.match_records import merge_record
except ImportError:  # invoked as a script: python tools/rewrite_match_pins.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.match_records import merge_record

GENERIC_CODES = frozenset(
    {
        "value_error",
        "polynomial.invariant",
        "probability.model_invariant",
        "polynomial.multivariate_contract",
        "assertion_error",
        "recurrence_solving.invalid_domain",
        "matrix.budget_exceeded",
    }
)
_RAISES = re.compile(r"^(\s*)with pytest\.raises\((?P<err>[A-Za-z_][A-Za-z0-9_]*)\)")
DEFAULT_BINDING = "exc_info"


def load(path: Path) -> dict[str, dict[str, object]]:
    recorded: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # Each xdist worker writes one JSON object per session, so the same
        # site can appear in several lines. Union them rather than letting the
        # last worker overwrite the others: a coded observation must not hide
        # an earlier code-less one, which is what makes an unsafe single-code
        # assert look justified.
        for key, info in json.loads(line).items():
            recorded[key] = merge_record(recorded.get(key), info)
    return recorded


def convertible_sites(
    recorded: dict[str, dict[str, object]],
) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    """Return sites safe to rewrite, plus per-file rejection counts."""
    # A code shared by differing match text within one file is not a
    # discriminator for those sites.
    by_code: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, info in recorded.items():
        path, _, _ = key.rpartition(":")
        code = info.get("code")
        for match in _observed_matches(info):
            if _is_owner_code(code) and code not in GENERIC_CODES:
                by_code[(path, code)].add(match)

    usable: dict[tuple[str, int], str] = {}
    rejected: dict[str, int] = defaultdict(int)
    for key, info in recorded.items():
        path, _, lineno = key.rpartition(":")
        code = info.get("code")
        codes = info.get("codes")
        # A site that raised more than one code across its executions (a
        # parametrized case or a loop) cannot be pinned to a single assert.
        if isinstance(codes, list) and len(codes) > 1:
            rejected["multiple_codes_across_executions"] += 1
            continue
        # Likewise for a site whose executions asserted different messages
        # behind one code: the wording is the only thing telling those guards
        # apart, so it cannot be replaced by a single code assert.
        if len(_observed_matches(info)) > 1:
            rejected["multiple_matches_across_executions"] += 1
            continue
        if not isinstance(code, str) or not code:
            rejected["no_code"] += 1
            continue
        if not _is_owner_code(code):
            rejected["non_owner_code"] += 1
            continue
        if code in GENERIC_CODES:
            rejected["generic_code"] += 1
            continue
        if len(by_code[(path, code)]) > 1:
            rejected["code_shared_by_other_messages"] += 1
            continue
        usable[(path, int(lineno))] = code
    return usable, rejected


def _is_owner_code(code: object) -> TypeGuard[str]:
    """Owner errors use namespaced codes; bare names are generic validators."""

    return isinstance(code, str) and "." in code


def _observed_matches(info: dict[str, object]) -> list[str]:
    """Every ``match`` text recorded for one site, across repeated executions."""

    observed = info.get("matches")
    if isinstance(observed, list):
        return [item for item in observed if isinstance(item, str) and item]
    match = info.get("match")
    return [match] if isinstance(match, str) and match else []


def _targets(
    tree: ast.Module, sites: dict[int, str]
) -> tuple[list[tuple[int, int, int, str, str, str]], int]:
    """Locate each ``with pytest.raises(..., match=...)`` header by AST.

    Returns ``(targets, multi_item_with)``. ``multi_item_with`` counts the
    raised sites deliberately left pinned because the ``with`` also manages
    sibling contexts.
    """

    found: list[tuple[int, int, int, str, str, str]] = []
    multi_item_with = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        raises_items = [item for item in node.items if _is_raises_call(item)]
        if not raises_items:
            continue
        if len(node.items) > 1:
            # `with request_execution(...), pytest.raises(...):` and its
            # parenthesised spelling manage real state alongside the
            # assertion. Rebuilding the header from the raises call alone
            # would silently drop the siblings and change what the test
            # exercises, so these sites keep their message match.
            multi_item_with += len(raises_items)
            continue
        item = raises_items[0]
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        if any(keyword.arg == "check" for keyword in call.keywords):
            # Code pins must not discard pytest's additional exception
            # predicate. Keep the original match assertion for these sites.
            continue
        code = sites.get(call.lineno)
        if code is None:
            continue
        first = call.args[0] if call.args else None
        error_class = first.id if isinstance(first, ast.Name) else None
        if error_class is None:
            continue
        # Reuse an existing `as <name>` binding; introducing a second name
        # would orphan assertions the test already makes on the original.
        if isinstance(item.optional_vars, ast.Name):
            binding = item.optional_vars.id
        else:
            binding = DEFAULT_BINDING
        body_end: int = node.body[-1].end_lineno if node.body else call.lineno
        if body_end is None:
            continue
        header_end: int = call.end_lineno or call.lineno
        found.append((call.lineno, header_end, body_end, error_class, code, binding))
    return found, multi_item_with


def _is_raises_call(item: ast.withitem) -> bool:
    """Whether one ``with`` item is a ``pytest.raises(..., match=...)`` call."""

    call = item.context_expr
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "raises"):
        return False
    return any(keyword.arg == "match" for keyword in call.keywords)


def rewrite(path: Path, sites: dict[int, str]) -> tuple[int, int]:
    """Rewrite the convertible sites in one file; return (rewritten, skipped)."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(path))
    targets, multi_item_with = _targets(tree, sites)
    if not targets:
        return 0, multi_item_with
    # Process outermost-first and never let one rewrite swallow another.
    targets.sort(key=lambda item: (item[0], -item[2]))
    filtered: list[tuple[int, int, int, str, str, str]] = []
    reach = 0
    for start, header_end, body_end, error_class, code, binding in targets:
        if start <= reach:
            continue
        filtered.append((start, header_end, body_end, error_class, code, binding))
        reach = body_end
    targets = filtered
    out: list[str] = []
    cursor = 0
    changed = 0
    for start, header_end, body_end, error_class, code, binding in targets:
        out.extend(lines[cursor : start - 1])
        leading = re.match(r"\s*", lines[start - 1])
        indent = leading.group(0) if leading else ""
        out.append(f"{indent}with pytest.raises({error_class}) as {binding}:\n")
        # guarded body, 1-indexed header_end+1 .. body_end
        out.extend(lines[header_end:body_end])
        out.append(f'{indent}assert {binding}.value.errors()[0]["type"] == "{code}"\n')
        cursor = body_end
        changed += 1
    out.extend(lines[cursor:])
    result = "".join(out)
    ast.parse(result, filename=str(path))
    path.write_text(result, encoding="utf-8")
    return changed, multi_item_with


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
    skipped_multi_item = 0
    for relative, sites in sorted(per_file.items()):
        target = args.root / relative
        if not target.exists():
            continue
        if args.dry_run:
            total += len(sites)
            continue
        try:
            count, multi_item_with = rewrite(target, sites)
        except SyntaxError:
            # A rewritten file must still parse; a decorator that merely
            # references pytest.raises can collide with a recorded line number.
            print(f"{relative}: SKIPPED (rewrite would not parse)")
            continue
        total += count
        skipped_multi_item += multi_item_with
        print(f"{relative}: rewrote {count}")
    print(f"total rewritten: {total}")
    if skipped_multi_item:
        print(f"  left pinned (sibling context managers): {skipped_multi_item}")
    for reason, count in sorted(rejected.items()):
        print(f"  left pinned ({reason}): {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
