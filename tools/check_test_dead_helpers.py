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
                found.extend(
                    (name, node.lineno, "variable", node)
                    for name in _assignment_names(target)
                    if name not in _PYTEST_MODULE_NAMES
                )
        elif isinstance(node, ast.AnnAssign):
            found.extend(
                (name, node.lineno, "variable", node)
                for name in _assignment_names(node.target)
                if name not in _PYTEST_MODULE_NAMES
            )
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
    fixtures = _fixture_names(tree)
    referenced.update(
        fixtures[name] for name in _pytest_string_references(tree) if name in fixtures
    )
    referenced.update(_autouse_fixture_definitions(tree))
    _scan(tree, frozenset(), module_names, referenced, fixtures)
    return referenced


def _assignment_names(target: ast.expr) -> frozenset[str]:
    """Names bound by a simple or destructuring assignment target."""

    if isinstance(target, ast.Name):
        return frozenset({target.id})
    if isinstance(target, (ast.Tuple, ast.List)):
        return frozenset().union(*(_assignment_names(item) for item in target.elts))
    if isinstance(target, ast.Starred):
        return _assignment_names(target.value)
    return frozenset()


def _pytest_string_references(tree: ast.Module) -> set[str]:
    """Names that pytest resolves from supported string-bearing markers."""

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name == "pytest.mark.usefixtures":
            for argument in node.args:
                referenced.update(_string_values(argument))
        elif name == "pytest.mark.parametrize":
            # Only indirect parameter names identify fixtures. The first
            # argument is a local parameter declaration, not a global helper.
            for keyword in node.keywords:
                if keyword.arg == "indirect":
                    if (
                        isinstance(keyword.value, ast.Constant)
                        and (keyword.value.value is True)
                        and node.args
                    ):
                        referenced.update(_parameter_names(node.args[0]))
                    else:
                        referenced.update(_string_values(keyword.value))
    return referenced


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _string_values(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_string_values(item) for item in node.elts))
    return set()


def _parameter_names(node: ast.AST) -> set[str]:
    return {
        name.strip()
        for value in _string_values(node)
        for name in value.split(",")
        if name.strip()
    }


def _fixture_names(tree: ast.Module) -> dict[str, str]:
    """Module-level helpers pytest injects by parameter name.

    Only these are exempt from lexical resolution. A test parameter that
    happens to share a name with a plain module-level assignment is not a
    fixture reference, so that assignment stays subject to the dead-code gate.
    """

    names: dict[str, str] = {}
    for node in _definition_names(tree):
        _name, _line, kind, definition = node
        if kind != "function":
            continue
        if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in definition.decorator_list:
            if "fixture" not in ast.dump(decorator):
                continue
            fixture_name = definition.name
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if (
                        keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        fixture_name = keyword.value.value
                        break
            names[fixture_name] = definition.name
            break
    return names


def _autouse_fixture_definitions(tree: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for _name, _line, kind, definition in _definition_names(tree):
        if kind != "function" or not isinstance(
            definition, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        for decorator in definition.decorator_list:
            if not isinstance(decorator, ast.Call) or "fixture" not in ast.dump(
                decorator.func
            ):
                continue
            if any(
                keyword.arg == "autouse"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in decorator.keywords
            ):
                names.add(definition.name)
                break
    return frozenset(names)


def _scan(  # noqa: C901
    node: ast.AST | Sequence[ast.AST],
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: dict[str, str],
    class_outer_bound: frozenset[str] | None = None,
) -> None:
    """Record module-level loads that no enclosing scope shadows."""

    if isinstance(node, ast.Name):
        _record(node.id, node.ctx, bound, module_names, referenced)
        return
    if isinstance(node, ast.Constant):
        return
    children: Sequence[ast.AST] = (
        node if isinstance(node, Sequence) else list(ast.iter_child_nodes(node))
    )
    for child in children:
        if isinstance(child, _SCOPE_NODES):
            # Headers evaluate in the enclosing scope; the body does not.
            for decorator in getattr(child, "decorator_list", ()):
                _scan(
                    decorator,
                    bound,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
            for default in _header_expressions(child):
                _scan(
                    default,
                    bound,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
            if _is_pytest_injected(child):
                # pytest resolves a fixture by parameter name, so a requested
                # fixture is referenced without any Name load in the body.
                requested = _argument_names(child) - _direct_parametrize_names(child)
                referenced.update(
                    fixtures[name] for name in requested if name in fixtures
                )
            if isinstance(child, ast.ClassDef):
                _scan_class_body(child.body, bound, module_names, referenced, fixtures)
                continue
            body_bound = bound | _scope_bindings(child)
            body_class_outer_bound = class_outer_bound
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and class_outer_bound is not None
            ):
                # Python methods do not close over the class namespace. They
                # resolve names from the scope that surrounded the class.
                body_bound = class_outer_bound | _scope_bindings(child)
                body_class_outer_bound = None
            elif (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name in module_names
            ):
                # A definition cannot make itself live through recursion alone.
                body_bound = body_bound | {child.name}
            _scan(
                child.body,
                body_bound,
                module_names,
                referenced,
                fixtures,
                body_class_outer_bound,
            )
            continue
        if isinstance(child, ast.Name):
            _record(child.id, child.ctx, bound, module_names, referenced)
            # A Store or Del name is a binding or rebinding, never a use.
            continue
        if isinstance(child, _COMPREHENSIONS):
            # Comprehensions open their own scope from Python 3 onward.
            inner = bound
            for generator in child.generators:
                _scan(
                    generator.iter,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
                inner = inner | _comprehension_target_names(generator.target)
                _scan(
                    generator.ifs,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
            if isinstance(child, ast.DictComp):
                _scan(
                    child.key,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
                _scan(
                    child.value,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
            else:
                _scan(
                    child.elt,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    class_outer_bound,
                )
            continue
        # An attribute access never resolves to a module-level binding:
        # `obj.name` reads a member, so it cannot reference a global helper.
        if isinstance(child, ast.Attribute):
            _scan(
                child.value,
                bound,
                module_names,
                referenced,
                fixtures,
                class_outer_bound,
            )
            continue
        _scan(
            child,
            bound,
            module_names,
            referenced,
            fixtures,
            class_outer_bound,
        )


def _scan_class_body(
    statements: Sequence[ast.stmt],
    outer_bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: dict[str, str],
) -> None:
    """Scan class statements in order because class locals bind when assigned."""

    bound = outer_bound
    for statement in statements:
        _scan(statement, bound, module_names, referenced, fixtures, outer_bound)
        bound = bound | _class_statement_bindings(statement)


def _class_statement_bindings(statement: ast.stmt) -> frozenset[str]:
    names: set[str] = set()
    declared_global: set[str] = set()
    stack = [statement]
    while stack:
        child = stack.pop()
        if isinstance(child, (*_SCOPE_NODES, *_COMPREHENSIONS)):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.Global):
            declared_global.update(child.names)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound = alias.asname or alias.name
                names.add(bound.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            names.add(child.name)
        stack.extend(ast.iter_child_nodes(child))
    return frozenset(names - declared_global)


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


def _direct_parametrize_names(node: ast.AST) -> frozenset[str]:
    """Test arguments supplied as values do not request same-named fixtures."""

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset()
    direct: set[str] = set()
    for decorator in node.decorator_list:
        if (
            not isinstance(decorator, ast.Call)
            or _dotted_name(decorator.func) != ("pytest.mark.parametrize")
            or not decorator.args
        ):
            continue
        parameter_names = _parameter_names(decorator.args[0])
        indirect_keyword = next(
            (item.value for item in decorator.keywords if item.arg == "indirect"),
            None,
        )
        if (
            isinstance(indirect_keyword, ast.Constant)
            and indirect_keyword.value is True
        ):
            indirect = parameter_names
        elif indirect_keyword is not None:
            indirect = _parameter_names(indirect_keyword)
        else:
            indirect = set()
        direct.update(parameter_names - indirect)
    return frozenset(direct)


def _scope_bindings(node: ast.AST) -> frozenset[str]:
    """Names a function, class, or lambda binds directly, nested scopes aside."""

    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
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
    body = getattr(node, "body", ())
    stack = list(body) if isinstance(body, list) else []
    while stack:
        child = stack.pop()
        if isinstance(child, (*_SCOPE_NODES, *_COMPREHENSIONS)):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.Global):
            declared_global.update(child.names)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound = alias.asname or alias.name
                names.add(bound.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            names.add(child.name)
        stack.extend(ast.iter_child_nodes(child))
    # `global name` means the assignment targets the module scope, so it does
    # not shadow the module-level binding it writes to.
    return frozenset(names - declared_global)


def _comprehension_target_names(target: ast.AST) -> frozenset[str]:
    return frozenset(
        child.id
        for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    )


def _header_expressions(node: ast.AST) -> tuple[ast.AST, ...]:
    """Defaults and annotations, which evaluate outside the function scope."""

    if isinstance(node, ast.ClassDef):
        return (*node.bases, *(keyword.value for keyword in node.keywords))
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
    return tuple(found)


def _waived(lines: list[str], line: int) -> bool:
    return _WAIVER in lines[line - 1]


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
    for name, line, kind, _node in _definition_names(tree):
        if name in referenced:
            continue
        if _waived(lines, line):
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
