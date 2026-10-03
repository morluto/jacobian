"""Require every module-level test helper to be referenced by its own module.

``vulture`` cannot cover this: at the confidence level that surfaces dead test
helpers it also reports fixtures (referenced only through
``@pytest.mark.usefixtures`` or by pytest's fixture-argument convention) and
hundreds of low-signal unused variables. At the confidence level the repository
gate uses, dead test helpers are invisible.

This check is deliberately narrow and deterministic. For each ``tests/**`` module
it reports a module-level function, class, or assignment whose name is never
referenced anywhere else in that module. Names referenced by ``@pytest.fixture``
consumers, ``usefixtures`` strings, ``parametrize`` arguments, and decorator
arguments all count as references, so live fixtures are not reported.

Escape hatch: a module may declare an intentionally unused helper with a
``# dead-code: <reason>`` comment on the same line as the definition.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = PurePosixPath("tests")
_GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
)
_WAIVER = "# dead-code:"
# pytest collects classes by prefix and reads the module-level ``pytestmark``;
# neither is referenced inside the module that declares it.
_PYTEST_COLLECTED_PREFIX = "Test"
_PYTEST_MODULE_NAMES = frozenset({"pytestmark"})


@dataclass(frozen=True)
class Violation:
    """One unreferenced module-level test helper."""

    path: str
    line: int
    name: str
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.kind}] {self.name} is defined but never referenced"


@dataclass(frozen=True)
class Report:
    """Result of checking test modules for unreferenced helpers."""

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
            return f"test-dead-code: OK ({self.files_scanned} files checked)"
        lines = [
            f"test-dead-code: {len(self.violations)} unreferenced helper(s) "
            f"({self.files_scanned} files checked)"
        ]
        lines.extend(str(item) for item in self.violations)
        return "\n".join(lines)


def _definition_names(tree: ast.Module) -> list[tuple[str, int, str, ast.AST]]:
    """Module-level helpers with their position, kind, and node.

    Test functions are excluded: pytest collects them by name, so they have no
    in-file reference and reporting them would flag the entire suite.
    """
    found: list[tuple[str, int, str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # pytest collects test functions and Test* classes by name, so they
            # have no in-file reference and reporting them would flag the suite.
            if node.name.startswith("test_") or node.name.startswith(
                _PYTEST_COLLECTED_PREFIX
            ):
                continue
            if node.name in _PYTEST_MODULE_NAMES:
                continue
            found.append((node.name, node.lineno, "function", node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id not in _PYTEST_MODULE_NAMES
                ):
                    found.append((target.id, node.lineno, "variable", node))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id not in _PYTEST_MODULE_NAMES
        ):
            found.append((node.target.id, node.lineno, "variable", node))
    return found


def _definition_only_references(tree: ast.Module) -> set[str]:
    """Names referenced by anything other than their own binding site."""
    bound_at: dict[str, set[int]] = {}
    for name, line, _kind, _node in _definition_names(tree):
        bound_at.setdefault(name, set()).add(line)

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # A bare Name in Store context at a definition line is the binding.
            if isinstance(node.ctx, ast.Store) and node.lineno in bound_at.get(
                node.id, set()
            ):
                continue
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Fixture and marker arguments are strings.
            referenced.add(node.value)
        elif isinstance(node, ast.arg):
            referenced.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A definition is not a use of itself, but decorators and base
            # classes inside its header are real references.
            for dec in node.decorator_list:
                for child in ast.walk(dec):
                    if isinstance(child, ast.Name):
                        referenced.add(child.id)
                    elif isinstance(child, ast.Constant) and isinstance(
                        child.value, str
                    ):
                        referenced.add(child.value)
    return referenced


def _waived(lines: list[str], line: int, node: ast.AST) -> bool:
    end = getattr(node, "end_lineno", None) or line
    return any(_WAIVER in text for text in lines[line - 1 : end])


def _check_file(root: Path, path: Path) -> tuple[Violation, ...]:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, SyntaxError):
        return ()
    lines = source.splitlines()
    referenced = _definition_only_references(tree)
    violations: list[Violation] = []
    for name, line, kind, node in _definition_names(tree):
        if name in referenced:
            continue
        if _waived(lines, line, node):
            continue
        violations.append(Violation(relative, line, name, kind))
    return tuple(violations)


def _test_files(root: Path) -> Iterable[Path]:
    """Only collected test modules.

    Shared helper modules such as ``tests/dispatch/_support.py`` and
    ``tests/fixtures/accounting.py`` are imported by other modules, so they
    legitimately have no in-file reference.
    """
    tests_root = root / _TESTS_ROOT
    if not tests_root.is_dir():
        return ()
    return (
        path
        for path in sorted(tests_root.rglob("test_*.py"))
        if not any(part in _GENERATED_DIRECTORIES for part in path.parts)
    )


def check(root: Path | str = ROOT) -> Report:
    project_root = Path(root).resolve()
    files = tuple(_test_files(project_root))
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
    return Report(project_root, violations, len(files))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = check(args.root)
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
