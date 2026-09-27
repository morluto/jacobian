"""Require test monkeypatches to carry assertion evidence.

A test that replaces a name with ``monkeypatch`` must assert something about the
result: a plain ``assert``, ``pytest.raises`` (or an owner-local raises helper),
an ``assert_*``/``_assert_*`` helper, a validation-error context manager, or a
sentinel replacement whose body raises (the "this must not run" pattern). A
patch with none of these cannot demonstrate anything about the behavior it
replaces, so it is reported.

This intentionally checks only that *some* assertion evidence exists. It does
not judge whether the assertion depends on the patch: that requires dataflow
analysis, and the deterministic suite-wide evidence is the existence check.

Escape hatch: a test may waive the rule with a rationale comment on any line in
its body::

    # monkeypatch-evidence: <reason>

Prefer a real assertion. The marker exists for the rare case where the
replacement is itself the observable behavior and no separate assertion applies.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = PurePosixPath("tests")
_GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
)
_MARKER = "# monkeypatch-evidence:"
_MONKEYPATCH_METHODS = frozenset(
    {"setattr", "setitem", "delattr", "delitem", "setenv", "delenv"}
)
_ASSERTION_HELPER_PREFIXES = ("assert_", "_assert_")
_ASSERTION_HELPER_SUBSTRINGS = ("raises", "validation_error", "validation_code")


@dataclass(frozen=True)
class Violation:
    """One test-hygiene violation."""

    path: str
    line: int
    code: str
    message: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"

    def __str__(self) -> str:
        return f"{self.location}: [{self.code}] {self.message}"


@dataclass(frozen=True)
class HygieneReport:
    """Result of checking the test suite's monkeypatch evidence."""

    root: Path
    violations: tuple[Violation, ...]
    files_scanned: int

    @property
    def failed(self) -> bool:
        return bool(self.violations)

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        if self.ok:
            return f"test-hygiene: OK ({self.files_scanned} files checked)"
        lines = [
            f"test-hygiene: {len(self.violations)} violation(s) "
            f"({self.files_scanned} files checked)"
        ]
        lines.extend(str(violation) for violation in self.violations)
        return "\n".join(lines)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _test_functions(
    tree: ast.AST,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _monkeypatch_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "monkeypatch"
        and node.func.attr in _MONKEYPATCH_METHODS
    )


def _local_callables(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.AST]:
    """Nested functions and lambdas bound to a local name."""

    callables: dict[str, ast.AST] = {}
    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            callables[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Lambda):
                    callables[target.id] = node.value
    return callables


def _bodies_raise(node: ast.AST) -> bool:
    return any(isinstance(descendant, ast.Raise) for descendant in ast.walk(node))


def _contains_assertion_evidence(node: ast.AST) -> bool:
    for descendant in ast.walk(node):
        if isinstance(descendant, (ast.Assert, ast.Raise)):
            return True
        if isinstance(descendant, (ast.With, ast.AsyncWith)) and any(
            isinstance(item.context_expr, ast.Call)
            and _call_name(item.context_expr.func) == "raises"
            for item in descendant.items
        ):
            return True
    return False


def _sentinel_replacement_raises(
    calls: tuple[ast.Call, ...], callables: dict[str, ast.AST]
) -> bool:
    """A replacement that raises is the forbidden-call evidence itself."""

    for call in calls:
        # ``setattr(obj, name, value)`` and ``setattr("mod.attr", value)`` both
        # place the replacement last; deletion helpers carry no value.
        if _call_name(call.func) not in {"setattr", "setitem", "setenv"}:
            continue
        value: ast.AST | None = call.args[-1] if call.args else None
        if value is None:
            value = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "value"),
                None,
            )
        if value is None:
            continue
        if isinstance(value, ast.Name) and value.id in callables:
            if _bodies_raise(callables[value.id]):
                return True
        elif _bodies_raise(value):
            return True
    return False


def _has_assertion_evidence(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    calls: tuple[ast.Call, ...],
    module_helpers: Mapping[str, ast.AST],
) -> bool:
    local_helpers = dict(module_helpers)
    local_helpers.update(_local_callables(function))
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name is None:
                continue
            is_assertion_helper = name.startswith(_ASSERTION_HELPER_PREFIXES) or any(
                token in name for token in _ASSERTION_HELPER_SUBSTRINGS
            )
            if is_assertion_helper:
                helper = local_helpers.get(name)
                if helper is None or _contains_assertion_evidence(helper):
                    return True
    return _sentinel_replacement_raises(calls, _local_callables(function))


def _waived(lines: list[str], function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    end = function.end_lineno or function.lineno
    return any(_MARKER in line for line in lines[function.lineno - 1 : end])


def _test_files(root: Path) -> tuple[Path, ...]:
    tests_root = root / _TESTS_ROOT
    if not tests_root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(tests_root.rglob("*.py"))
        if not any(part in _GENERATED_DIRECTORIES for part in path.parts)
    )


def _check_file(root: Path, path: Path) -> tuple[Violation, ...]:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, SyntaxError):
        return ()
    lines = source.splitlines()
    module_helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    violations: list[Violation] = []
    for function in _test_functions(tree):
        calls = _monkeypatch_calls(function)
        if not calls:
            continue
        if _waived(lines, function):
            continue
        if _has_assertion_evidence(function, calls, module_helpers):
            continue
        violations.append(
            Violation(
                relative,
                function.lineno,
                "monkeypatch-without-evidence",
                f"{function.name} uses monkeypatch but asserts nothing about the "
                "patched behavior; add an assert, a typed-error check, or a "
                "forbidden-call sentinel, or waive with "
                f"'{_MARKER} <reason>'",
            )
        )
    return tuple(violations)


def check(root: Path | str = ROOT) -> HygieneReport:
    project_root = Path(root).resolve()
    files = _test_files(project_root)
    violations = tuple(
        sorted(
            (
                violation
                for path in files
                for violation in _check_file(project_root, path)
            ),
            key=lambda item: (item.path, item.line),
        )
    )
    return HygieneReport(project_root, violations, len(files))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = check(args.root)
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
