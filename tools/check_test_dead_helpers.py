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
from collections.abc import Iterable, Sequence
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
# Every construct that opens a name scope for the reference walk below.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_COMPREHENSIONS = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


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
    """Module-level names that a *lexically resolved* load actually reaches.

    A flat name set is not enough. A local parameter, assignment, or attribute
    can share a module-level helper's name while Python resolves that
    occurrence elsewhere, which would let an unused helper pass the gate: an
    unused ``value = 1`` survives because some test declares a ``value``
    parameter, and an unused function survives because the code writes
    ``obj.function_name``. References are therefore resolved against the
    scopes that enclose them, and pytest's name-based collection is handled
    separately in :func:`_definition_names`.
    """

    module_names = {name for name, _line, _kind, _node in _definition_names(tree)}
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Fixture and marker arguments are strings, not resolved names.
            referenced.add(node.value)
    _scan(tree, frozenset(), module_names, referenced, _fixture_names(tree))
    return referenced


def _fixture_names(tree: ast.Module) -> frozenset[str]:
    """Module-level helpers pytest injects by parameter name.

    Only these are exempt from lexical resolution. A test parameter that
    happens to share a name with a plain module-level assignment is not a
    fixture reference, so that assignment stays subject to the dead-code gate.
    """

    names = set()
    for node in _definition_names(tree):
        _name, _line, kind, definition = node
        if kind != "function":
            continue
        if any("fixture" in ast.dump(item) for item in definition.decorator_list):  # type: ignore[attr-defined]
            names.add(_name)
    return frozenset(names)


def _scan(
    node: ast.AST | Sequence[ast.AST],
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: frozenset[str],
) -> None:
    """Record module-level loads that no enclosing scope shadows."""

    if isinstance(node, ast.Name):
        _record(node.id, node.ctx, bound, module_names, referenced)
        return
    if isinstance(node, ast.Constant):
        # Fixture and marker arguments are strings, not resolved names.
        if isinstance(node.value, str):
            referenced.add(node.value)
        return
    children: Sequence[ast.AST] = (
        node if isinstance(node, Sequence) else list(ast.iter_child_nodes(node))
    )
    for child in children:
        if isinstance(child, _SCOPE_NODES):
            # Headers evaluate in the enclosing scope; the body does not.
            for decorator in getattr(child, "decorator_list", ()):
                _scan(decorator, bound, module_names, referenced, fixtures)
            for default in _header_expressions(child):
                _scan(default, bound, module_names, referenced, fixtures)
            if _is_pytest_injected(child):
                # pytest resolves a fixture by parameter name, so a requested
                # fixture is referenced without any Name load in the body.
                referenced.update(_argument_names(child) & fixtures)
            _scan(
                child.body,
                bound | _scope_bindings(child),
                module_names,
                referenced,
                fixtures,
            )
            continue
        if isinstance(child, ast.Name):
            _record(child.id, child.ctx, bound, module_names, referenced)
            # A Store or Del name is a binding or rebinding, never a use.
            continue
        if isinstance(child, _COMPREHENSIONS):
            # Comprehensions open their own scope from Python 3 onward.
            inner = bound | _comprehension_targets(child)
            for generator in child.generators:
                _scan(generator.iter, inner, module_names, referenced, fixtures)
                _scan(generator.ifs, inner, module_names, referenced, fixtures)
            if isinstance(child, ast.DictComp):
                _scan(child.key, inner, module_names, referenced, fixtures)
                _scan(child.value, inner, module_names, referenced, fixtures)
            else:
                _scan(child.elt, inner, module_names, referenced, fixtures)
            continue
        # An attribute access never resolves to a module-level binding:
        # `obj.name` reads a member, so it cannot reference a global helper.
        if isinstance(child, ast.Attribute):
            _scan(child.value, bound, module_names, referenced, fixtures)
            continue
        _scan(child, bound, module_names, referenced, fixtures)


def _record(
    name: str,
    context: ast.expr_context,
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
) -> None:
    if isinstance(context, ast.Load) and name in module_names and name not in bound:
        referenced.add(name)


def _is_pytest_injected(node: ast.AST) -> bool:
    """Whether pytest resolves this function's parameters by name."""

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if node.name.startswith("test_"):
        return True
    return any("fixture" in ast.dump(decorator) for decorator in node.decorator_list)


def _argument_names(node: ast.AST) -> frozenset[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return frozenset()
    arguments = node.args
    names = {
        argument.arg
        for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs)
        for argument in group
    }
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return frozenset(names)


def _scope_bindings(node: ast.AST) -> frozenset[str]:
    """Names a function, class, or lambda binds directly, nested scopes aside."""

    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = node.args
        for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
            names.update(argument.arg for argument in group)
        for extra in (arguments.vararg, arguments.kwarg):
            if extra is not None:
                names.add(extra.arg)
    if isinstance(node, ast.ClassDef):
        # Only a class name is bound inside its own body. A `def` binds its name
        # in the *enclosing* scope, so a method that calls a same-named
        # module-level function resolves to that global, not to itself.
        names.add(node.name)
    declared_global: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Global):
            declared_global.update(child.names)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_SCOPE_NODES, *_COMPREHENSIONS)):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound = alias.asname or alias.name
                names.add(bound.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            names.add(child.name)
    # `global name` means the assignment targets the module scope, so it does
    # not shadow the module-level binding it writes to.
    return frozenset(names - declared_global)


def _comprehension_targets(node: ast.AST) -> frozenset[str]:
    return frozenset(
        target.id
        for generator in node.generators  # type: ignore[attr-defined]
        for target in ast.walk(generator.target)
        if isinstance(target, ast.Name)
    )


def _header_expressions(node: ast.AST) -> tuple[ast.AST, ...]:
    """Defaults and annotations, which evaluate outside the function scope."""

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return ()
    arguments = node.args
    found: list[ast.expr] = [*arguments.defaults]
    found.extend(item for item in arguments.kw_defaults if item is not None)
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None and extra.annotation is not None:
            found.append(extra.annotation)
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.returns is not None
    ):
        found.append(node.returns)
    for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
        found.extend(
            argument.annotation for argument in group if argument.annotation is not None
        )
    if isinstance(node, ast.ClassDef):
        found.extend(node.bases)
        found.extend(keyword.value for keyword in node.keywords)
    return tuple(found)


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
