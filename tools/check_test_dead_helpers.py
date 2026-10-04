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
_PYTEST_MODULE_NAMES = frozenset({"pytestmark", "pytest_plugins"})
_PYTEST_LIFECYCLE_HOOKS = frozenset(
    {
        "setup_module",
        "teardown_module",
        "setUpModule",
        "tearDownModule",
        "setup_function",
        "teardown_function",
        "pytest_addoption",
        "pytest_configure",
        "pytest_unconfigure",
        "pytest_generate_tests",
        "pytest_collection_modifyitems",
        "pytest_collection_finish",
        "pytest_sessionstart",
        "pytest_sessionfinish",
        "pytest_runtest_setup",
        "pytest_runtest_call",
        "pytest_runtest_teardown",
    }
)
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
    callable_test_aliases = _pytest_callable_test_aliases(tree)
    unittest_classes = _pytest_unittest_testcase_classes(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # pytest uses distinct name prefixes for test functions and classes.
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test"):
                continue
            if isinstance(node, ast.ClassDef) and (
                node.name.startswith(_PYTEST_COLLECTED_PREFIX)
                or node.name in unittest_classes
            ):
                continue
            if _is_overload_declaration(node):
                continue
            if node.name in _PYTEST_MODULE_NAMES:
                continue
            if node.name in _PYTEST_LIFECYCLE_HOOKS:
                continue
            found.append((node.name, node.lineno, "function", node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                found.extend(
                    (name, node.lineno, "variable", node)
                    for name in _assignment_names(target)
                    if name not in _PYTEST_MODULE_NAMES
                    and name not in callable_test_aliases
                )
        elif isinstance(node, ast.AnnAssign):
            found.extend(
                (name, node.lineno, "variable", node)
                for name in _assignment_names(node.target)
                if name not in _PYTEST_MODULE_NAMES
                and name not in callable_test_aliases
            )
    return found


def _pytest_callable_test_aliases(tree: ast.Module) -> frozenset[str]:
    """Module assignment aliases that pytest collects as test callables."""

    callable_names: set[str] = set()
    aliases: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            callable_names.add(statement.name)
            continue
        value = (
            statement.value
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            else None
        )
        callable_alias = isinstance(value, ast.Name) and value.id in callable_names
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            targets = []
        for target in targets:
            for name in _assignment_names(target):
                if name.startswith("test") and callable_alias:
                    aliases.add(name)
                    callable_names.add(name)
                else:
                    callable_names.discard(name)
                    aliases.discard(name)
    return frozenset(aliases)


def _pytest_unittest_testcase_classes(tree: ast.Module) -> frozenset[str]:
    module_aliases = {"unittest"}
    testcase_aliases: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for item in statement.names:
                if item.name == "unittest":
                    module_aliases.add(item.asname or "unittest")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "unittest":
            testcase_aliases.update(
                item.asname or item.name
                for item in statement.names
                if item.name == "TestCase"
            )

    collected: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        for base in statement.bases:
            name = _dotted_name(base)
            if name == "unittest.TestCase":
                collected.add(statement.name)
                break
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "TestCase"
                and isinstance(base.value, ast.Name)
                and base.value.id in module_aliases
            ):
                collected.add(statement.name)
                break
            if isinstance(base, ast.Name) and base.id in testcase_aliases | collected:
                collected.add(statement.name)
                break
    return frozenset(collected)


def _is_overload_declaration(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Overload signatures do not create runtime bindings of their own."""

    for decorator in node.decorator_list:
        name = _dotted_name(decorator)
        if isinstance(decorator, ast.Call):
            name = _dotted_name(decorator.func)
        if name in {"overload", "typing.overload", "typing_extensions.overload"}:
            return True
    return False


def _definition_only_references(
    tree: ast.Module,
) -> tuple[set[str], set[tuple[str, int, bool, int | None]]]:
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

    nodes, parents = _ast_nodes_and_parents(tree)
    module_names = {name for name, _line, _kind, _node in _definition_names(tree)}
    referenced: set[str] = set()
    fixture_declarations = _pytest_fixture_declarations(tree)
    fixtures = _fixture_names(fixture_declarations)
    constants_by_line, module_constants = _module_string_constants_at_definitions(tree)
    unittest_classes = _pytest_unittest_testcase_classes(tree)
    collected_class_lines = frozenset(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and (
            node.name.startswith(_PYTEST_COLLECTED_PREFIX)
            or node.name in unittest_classes
        )
    )
    referenced.update(
        fixtures[name]
        for name in _pytest_string_references(tree, nodes, module_constants)
        if name in fixtures
    )
    autouse_fixtures = _autouse_fixture_definitions(fixture_declarations)
    referenced.update(autouse_fixtures)
    aliases = _pytest_import_aliases(tree)
    reference_sites: set[tuple[str, int, bool, int | None]] = set()
    _scan(
        tree, frozenset(), module_names, referenced, fixtures, aliases, reference_sites
    )
    reference_sites.update(
        _pytest_injected_fixture_reference_sites(
            nodes,
            parents,
            fixtures,
            aliases,
            fixture_declarations,
            constants_by_line,
            collected_class_lines,
        )
    )
    synthetic_line = (
        max(
            (
                getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)
                for node in ast.walk(tree)
            ),
            default=0,
        )
        + 1
    )
    for definition in autouse_fixtures:
        reference_sites.add((definition, synthetic_line, True, None))
    reference_sites.update(
        _pytest_fixture_reference_sites(
            nodes, parents, fixtures, aliases, module_constants
        )
    )
    return referenced, reference_sites


def _ast_nodes_and_parents(
    tree: ast.AST,
) -> tuple[list[ast.AST], dict[ast.AST, ast.AST]]:
    nodes: list[ast.AST] = []
    parents: dict[ast.AST, ast.AST] = {}
    pending = [tree]
    while pending:
        node = pending.pop()
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            pending.append(child)
    return nodes, parents


def _assignment_names(target: ast.expr) -> frozenset[str]:
    """Names bound by a simple or destructuring assignment target."""

    if isinstance(target, ast.Name):
        return frozenset({target.id})
    if isinstance(target, (ast.Tuple, ast.List)):
        return frozenset().union(*(_assignment_names(item) for item in target.elts))
    if isinstance(target, ast.Starred):
        return _assignment_names(target.value)
    return frozenset()


def _pytest_string_references(
    tree: ast.Module, nodes: Sequence[ast.AST], constants: dict[str, str]
) -> set[str]:
    """Names that pytest resolves from supported string-bearing markers."""

    referenced: set[str] = set()
    aliases = _pytest_import_aliases(tree)
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _canonical_pytest_name(node.func, aliases)
        if name == "pytest.mark.usefixtures":
            for argument in node.args:
                referenced.update(_string_values(argument, constants))
        elif name is not None and name.endswith(".getfixturevalue") and node.args:
            referenced.update(_string_values(node.args[0], constants))
        elif name == "pytest.mark.parametrize":
            # Only indirect parameter names identify fixtures. The first
            # argument is a local parameter declaration, not a global helper.
            for keyword in node.keywords:
                if keyword.arg == "indirect":
                    if isinstance(keyword.value, ast.Constant) and (
                        keyword.value.value is True
                    ):
                        referenced.update(_parametrize_argnames(node, constants))
                    else:
                        referenced.update(_string_values(keyword.value, constants))
    return referenced


def _pytest_fixture_reference_sites(
    nodes: Sequence[ast.AST],
    parents: dict[ast.AST, ast.AST],
    fixtures: dict[str, str],
    aliases: dict[str, str],
    constants: dict[str, str],
) -> set[tuple[str, int, bool, int | None]]:
    """Preserve the fixture function that owns each string-based request."""

    sites: set[tuple[str, int, bool, int | None]] = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _canonical_pytest_name(node.func, aliases)
        if name == "pytest.mark.usefixtures":
            requested = set().union(
                *(_string_values(arg, constants) for arg in node.args)
            )
        elif name is not None and name.endswith(".getfixturevalue") and node.args:
            requested = _string_values(node.args[0], constants)
        elif name == "pytest.mark.parametrize":
            requested = set()
            for keyword in node.keywords:
                if keyword.arg != "indirect":
                    continue
                if (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    requested.update(_parametrize_argnames(node, constants))
                else:
                    requested.update(_string_values(keyword.value, constants))
        else:
            continue
        owner_line = None
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner_line = current.lineno
                break
            if isinstance(current, ast.ClassDef):
                owner_line = current.lineno
                break
        sites.update(
            (fixtures[fixture_name], node.lineno, True, owner_line)
            for fixture_name in requested
            if fixture_name in fixtures
        )
    return sites


def _pytest_injected_fixture_reference_sites(
    nodes: Sequence[ast.AST],
    parents: dict[ast.AST, ast.AST],
    fixtures: dict[str, str],
    aliases: dict[str, str],
    declarations: Sequence[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, str | None, bool, bool]
    ],
    constants_by_line: dict[int, dict[str, str]],
    collected_class_lines: frozenset[int],
) -> set[tuple[str, int, bool, int | None]]:
    """Resolve fixture arguments only on module-collected tests and fixtures."""

    fixture_lines = {
        definition.lineno
        for definition, _fixture_name, _autouse, _unknown_name in declarations
    }
    fixture_lines.update(
        node.lineno
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _pytest_collected_test_class(node, parents, collected_class_lines)
        is not None
        and _is_pytest_fixture_function(node, aliases)
    )
    sites: set[tuple[str, int, bool, int | None]] = set()
    for node in nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno in fixture_lines:
            class_parameters = frozenset()
            parameter_constants = constants_by_line.get(node.lineno, {})
        elif _is_pytest_collected_test(node, parents, collected_class_lines):
            class_node = _pytest_collected_test_class(
                node, parents, collected_class_lines
            )
            context_line = class_node.lineno if class_node is not None else node.lineno
            parameter_constants = constants_by_line.get(context_line, {})
            class_parameters = _pytest_test_class_parameters(
                node, parents, aliases, parameter_constants, collected_class_lines
            )
        else:
            continue
        requested = (
            _argument_names(node)
            - _direct_parametrize_names(node, aliases, parameter_constants)
            - class_parameters
        )
        sites.update(
            (fixtures[name], node.lineno, True, node.lineno)
            for name in requested
            if name in fixtures
        )
    return sites


def _is_pytest_collected_test(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    collected_class_lines: frozenset[int],
) -> bool:
    if not node.name.startswith("test"):
        return False
    parent = parents.get(node)
    while parent is not None and not isinstance(
        parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        parent = parents.get(parent)
    return (
        isinstance(parent, ast.Module)
        or _pytest_collected_test_class(node, parents, collected_class_lines)
        is not None
    )


def _pytest_collected_test_class(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    collected_class_lines: frozenset[int],
) -> ast.ClassDef | None:
    parent = parents.get(node)
    while parent is not None and not isinstance(
        parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        parent = parents.get(parent)
    if (
        not isinstance(parent, ast.ClassDef)
        or parent.lineno not in collected_class_lines
    ):
        return None
    enclosing = parents.get(parent)
    while enclosing is not None and not isinstance(
        enclosing, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        enclosing = parents.get(enclosing)
    return parent if isinstance(enclosing, ast.Module) else None


def _is_pytest_fixture_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]
) -> bool:
    return any(
        _canonical_pytest_name(
            decorator.func if isinstance(decorator, ast.Call) else decorator,
            aliases,
        )
        == "pytest.fixture"
        for decorator in node.decorator_list
    )


def _pytest_test_class_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    aliases: dict[str, str],
    constants: dict[str, str],
    collected_class_lines: frozenset[int],
) -> frozenset[str]:
    parent = parents.get(node)
    while parent is not None and not isinstance(
        parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        parent = parents.get(parent)
    if isinstance(parent, ast.ClassDef) and parent.lineno in collected_class_lines:
        return _direct_parametrize_names(parent, aliases, constants)
    return frozenset()


def _module_string_constants_at_definitions(
    tree: ast.Module,
) -> tuple[dict[int, dict[str, str]], dict[str, str]]:
    constants: dict[str, str] = {}
    snapshots: dict[int, dict[str, str]] = {}
    for statement in tree.body:
        if isinstance(statement, _SCOPE_NODES):
            snapshots[statement.lineno] = constants.copy()
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                constants.pop(statement.name, None)
        else:
            _update_module_string_constants(statement, constants)
    return snapshots, constants


def _pytest_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local pytest import spellings to their canonical names."""

    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "pytest":
                    local_name = item.asname or item.name
                    aliases[local_name] = "pytest"
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for item in node.names:
                local_name = item.asname or item.name
                aliases[local_name] = f"pytest.{item.name}"
    return aliases


def _canonical_pytest_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a pytest API name through direct import aliases."""

    name = _dotted_name(node)
    if name is None:
        return None
    root, separator, suffix = name.partition(".")
    canonical_root = aliases.get(root, root)
    return canonical_root + (separator + suffix if separator else "")


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _string_values(node: ast.AST, constants: dict[str, str] | None = None) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and constants is not None:
        value = constants.get(node.id)
        return {value} if value is not None else set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_string_values(item, constants) for item in node.elts))
    return set()


def _parameter_names(
    node: ast.AST, constants: dict[str, str] | None = None
) -> set[str]:
    if isinstance(node, ast.Name):
        value = (constants or {}).get(node.id)
        return (
            {part.strip() for part in value.split(",") if part.strip()}
            if value
            else set()
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_parameter_names(item, constants) for item in node.elts))
    return {
        name.strip()
        for value in _string_values(node)
        for name in value.split(",")
        if name.strip()
    }


def _parametrize_argnames(
    node: ast.Call, constants: dict[str, str] | None = None
) -> set[str]:
    argument = (
        node.args[0]
        if node.args
        else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "argnames"),
            None,
        )
    )
    return _parameter_names(argument, constants) if argument is not None else set()


def _fixture_names(
    declarations: Sequence[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, str | None, bool, bool]
    ],
) -> dict[str, str]:
    """Module-level helpers pytest injects by parameter name.

    Only these are exempt from lexical resolution. A test parameter that
    happens to share a name with a plain module-level assignment is not a
    fixture reference, so that assignment stays subject to the dead-code gate.
    """

    return {
        fixture_name or definition.name: definition.name
        for definition, fixture_name, _autouse, _unknown_name in declarations
    }


def _autouse_fixture_definitions(
    declarations: Sequence[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, str | None, bool, bool]
    ],
) -> frozenset[str]:
    return frozenset(
        definition.name
        for definition, _fixture_name, autouse, unknown_name in declarations
        if autouse or unknown_name
    )


def _pytest_fixture_declarations(
    tree: ast.Module,
) -> tuple[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str | None, bool, bool], ...]:
    aliases = _pytest_import_aliases(tree)
    constants: dict[str, str] = {}
    declarations: list[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, str | None, bool, bool]
    ] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                fixture_decorator = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if (
                    _canonical_pytest_name(fixture_decorator, aliases)
                    != "pytest.fixture"
                ):
                    continue
                name_keyword = (
                    next(
                        (
                            keyword.value
                            for keyword in decorator.keywords
                            if keyword.arg == "name"
                        ),
                        None,
                    )
                    if isinstance(decorator, ast.Call)
                    else None
                )
                fixture_name = (
                    _module_string_value(name_keyword, constants)
                    if name_keyword is not None
                    else node.name
                )
                autouse = isinstance(decorator, ast.Call) and any(
                    keyword.arg == "autouse"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in decorator.keywords
                )
                declarations.append(
                    (
                        node,
                        fixture_name,
                        autouse,
                        name_keyword is not None and fixture_name is None,
                    )
                )
                break
            constants.pop(node.name, None)
        elif isinstance(node, ast.ClassDef):
            constants.pop(node.name, None)
        else:
            _update_module_string_constants(node, constants)
    return tuple(declarations)


def _module_string_value(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _update_module_string_constants(
    statement: ast.stmt, constants: dict[str, str]
) -> None:
    bound: set[str] = set()
    pending: list[ast.AST] = [statement]
    while pending:
        node = pending.pop()
        if isinstance(node, _SCOPE_NODES):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(
                (item.asname or item.name).split(".")[0] for item in node.names
            )
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            bound.add(node.name)
        pending.extend(ast.iter_child_nodes(node))

    assigned: dict[str, str] = {}
    if isinstance(statement, ast.Assign):
        value = _module_string_value(statement.value, constants)
        if value is not None:
            assigned = {
                target.id: value
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
    elif isinstance(statement, ast.AnnAssign) and isinstance(
        statement.target, ast.Name
    ):
        value = (
            _module_string_value(statement.value, constants)
            if statement.value
            else None
        )
        if value is not None:
            assigned[statement.target.id] = value
    for name in bound:
        constants.pop(name, None)
    constants.update(assigned)


def _scan(
    node: ast.AST | Sequence[ast.AST],
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: dict[str, str],
    pytest_aliases: dict[str, str],
    reference_sites: set[tuple[str, int, bool, int | None]],
    class_outer_bound: frozenset[str] | None = None,
    deferred: bool = False,
    scope_line: int | None = None,
) -> None:
    """Record module-level loads and the binding each load can reach."""

    if isinstance(node, ast.Name):
        _record(
            node.id,
            node.ctx,
            node.lineno,
            bound,
            module_names,
            referenced,
            reference_sites,
            deferred,
            scope_line,
        )
        return
    if isinstance(node, ast.Constant):
        return
    children: Sequence[ast.AST] = (
        node if isinstance(node, Sequence) else list(ast.iter_child_nodes(node))
    )
    for child in children:
        if isinstance(child, _SCOPE_NODES):
            # Headers evaluate where the definition appears; function bodies
            # resolve module names after module initialization has completed.
            for decorator in getattr(child, "decorator_list", ()):
                _scan(
                    decorator,
                    bound,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    class_outer_bound,
                    deferred,
                    scope_line,
                )
            for default in _header_expressions(child):
                _scan(
                    default,
                    bound,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    class_outer_bound,
                    deferred,
                    scope_line,
                )
            if isinstance(child, ast.ClassDef):
                _scan_class_body(
                    child.body,
                    class_outer_bound if class_outer_bound is not None else bound,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    deferred,
                    class_outer_bound if class_outer_bound is not None else bound,
                )
                continue
            body_bound = bound | _scope_bindings(child)
            body_class_outer_bound = class_outer_bound
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and class_outer_bound is not None
            ):
                body_bound = class_outer_bound | _scope_bindings(child)
                body_class_outer_bound = None
            elif (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name in module_names
            ):
                body_bound = body_bound | {child.name}
            _scan(
                child.body,
                body_bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                body_class_outer_bound,
                True,
                child.lineno
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else scope_line,
            )
            continue
        if isinstance(child, ast.Name):
            _record(
                child.id,
                child.ctx,
                child.lineno,
                bound,
                module_names,
                referenced,
                reference_sites,
                deferred,
                scope_line,
            )
            continue
        if isinstance(child, _COMPREHENSIONS):
            first_iter_bound = bound
            inner = class_outer_bound if class_outer_bound is not None else bound
            lazy = deferred or isinstance(child, ast.GeneratorExp)
            for index, generator in enumerate(child.generators):
                _scan(
                    generator.iter,
                    first_iter_bound if index == 0 else inner,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    class_outer_bound if index == 0 else None,
                    deferred if index == 0 else lazy,
                    scope_line,
                )
                inner = inner | _comprehension_target_names(generator.target)
                _scan(
                    generator.ifs,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    None,
                    lazy,
                    scope_line,
                )
            for expression in (
                (child.key, child.value)
                if isinstance(child, ast.DictComp)
                else (child.elt,)
            ):
                _scan(
                    expression,
                    inner,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    None,
                    lazy,
                    scope_line,
                )
            continue
        if isinstance(child, ast.Attribute):
            _scan(
                child.value,
                bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                class_outer_bound,
                deferred,
                scope_line,
            )
            continue
        _scan(
            child,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            class_outer_bound,
            deferred,
            scope_line,
        )


def _scan_class_body(
    statements: Sequence[ast.stmt],
    outer_bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: dict[str, str],
    pytest_aliases: dict[str, str],
    reference_sites: set[tuple[str, int, bool, int | None]],
    deferred: bool,
    class_outer_bound: frozenset[str] | None = None,
) -> None:
    """Scan class statements in order because class locals bind when assigned."""

    bound = outer_bound
    lexical_outer = class_outer_bound if class_outer_bound is not None else outer_bound
    for statement in statements:
        _scan_class_statement(
            statement,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            lexical_outer,
            deferred,
        )
        bound = bound | _class_statement_bindings(statement)


def _scan_class_statement(
    statement: ast.stmt,
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    fixtures: dict[str, str],
    pytest_aliases: dict[str, str],
    reference_sites: set[tuple[str, int, bool, int | None]],
    outer_bound: frozenset[str],
    deferred: bool,
) -> None:
    if isinstance(statement, ast.If):
        _scan(
            statement.test,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            outer_bound,
            deferred,
        )
        if isinstance(statement.test, ast.Constant) and isinstance(
            statement.test.value, bool
        ):
            branch = statement.body if statement.test.value else statement.orelse
            _scan_class_body(
                branch,
                bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                deferred,
                outer_bound,
            )
            return
        _scan_class_body(
            statement.body,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        _scan_class_body(
            statement.orelse,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        return
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        _scan(
            statement.iter,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            outer_bound,
            deferred,
        )
        loop_bound = bound | _comprehension_target_names(statement.target)
        _scan_class_body(
            statement.body,
            loop_bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        _scan_class_body(
            statement.orelse,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        return
    if isinstance(statement, ast.While):
        _scan(
            statement.test,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            outer_bound,
            deferred,
        )
        _scan_class_body(
            statement.body,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        _scan_class_body(
            statement.orelse,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        return
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        with_bound = bound
        for item in statement.items:
            _scan(
                item.context_expr,
                with_bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                outer_bound,
                deferred,
            )
            if item.optional_vars is not None:
                with_bound |= _comprehension_target_names(item.optional_vars)
        _scan_class_body(
            statement.body,
            with_bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        return
    if isinstance(statement, (ast.Try, ast.TryStar)):
        _scan_class_body(
            statement.body,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        for handler in statement.handlers:
            if handler.type is not None:
                _scan(
                    handler.type,
                    bound,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    outer_bound,
                    deferred,
                )
            handler_bound = bound | ({handler.name} if handler.name else set())
            _scan_class_body(
                handler.body,
                frozenset(handler_bound),
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                deferred,
                outer_bound,
            )
        body_bound = bound | _class_suite_bindings(statement.body)
        _scan_class_body(
            statement.orelse,
            body_bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        final_bound = body_bound | _class_suite_bindings(statement.orelse)
        _scan_class_body(
            statement.finalbody,
            final_bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            deferred,
            outer_bound,
        )
        return
    if isinstance(statement, ast.Match):
        _scan(
            statement.subject,
            bound,
            module_names,
            referenced,
            fixtures,
            pytest_aliases,
            reference_sites,
            outer_bound,
            deferred,
        )
        for case in statement.cases:
            _scan(
                case.pattern,
                bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                outer_bound,
                deferred,
            )
            case_bound = bound | _pattern_binding_names(case.pattern)
            if case.guard is not None:
                _scan(
                    case.guard,
                    case_bound,
                    module_names,
                    referenced,
                    fixtures,
                    pytest_aliases,
                    reference_sites,
                    outer_bound,
                    deferred,
                )
            _scan_class_body(
                case.body,
                case_bound,
                module_names,
                referenced,
                fixtures,
                pytest_aliases,
                reference_sites,
                deferred,
                outer_bound,
            )
        return
    _scan(
        [statement],
        bound,
        module_names,
        referenced,
        fixtures,
        pytest_aliases,
        reference_sites,
        outer_bound,
        deferred,
        None,
    )


def _class_suite_bindings(statements: Sequence[ast.stmt]) -> frozenset[str]:
    return frozenset().union(
        *(_class_statement_bindings(statement) for statement in statements)
    )


def _pattern_binding_names(pattern: ast.pattern) -> frozenset[str]:
    return frozenset(
        name
        for node in ast.walk(pattern)
        if (name := _pattern_binding_name(node)) is not None
    )


def _class_statement_bindings(statement: ast.stmt) -> frozenset[str]:  # noqa: C901
    if isinstance(statement, ast.If):
        if isinstance(statement.test, ast.Constant) and isinstance(
            statement.test.value, bool
        ):
            branch = statement.body if statement.test.value else statement.orelse
            return _class_suite_bindings(branch)
        body_names = _class_suite_bindings(statement.body)
        else_names = _class_suite_bindings(statement.orelse)
        return body_names & else_names
    if isinstance(statement, (ast.Try, ast.TryStar)):
        normal_names = _class_suite_bindings(statement.body) | _class_suite_bindings(
            statement.orelse
        )
        if statement.handlers:
            handler_names = [
                _class_suite_bindings(handler.body) for handler in statement.handlers
            ]
            guaranteed = normal_names.intersection(*handler_names)
        else:
            guaranteed = normal_names
        return frozenset(guaranteed | _class_suite_bindings(statement.finalbody))
    if isinstance(statement, ast.Match):
        possible_case_names: list[frozenset[str]] = []
        for case in statement.cases:
            possible_case_names.append(_class_suite_bindings(case.body))
            if case.guard is None and _pattern_is_irrefutable(case.pattern):
                return frozenset.intersection(*possible_case_names)
        return frozenset()
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return (
            _class_suite_bindings(statement.orelse)
            if not _loop_body_has_break(statement.body)
            else frozenset()
        )
    names: set[str] = set()
    declared_global: set[str] = set()
    stack = [statement]
    while stack:
        child = stack.pop()
        if isinstance(child, _COMPREHENSIONS):
            names.update(_comprehension_walrus_names(child))
            continue
        if isinstance(child, _SCOPE_NODES):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif (pattern_name := _pattern_binding_name(child)) is not None:
            names.add(pattern_name)
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


def _pattern_is_irrefutable(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _pattern_is_irrefutable(pattern.pattern)
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_is_irrefutable(item) for item in pattern.patterns)
    return False


def _loop_body_has_break(statements: Sequence[ast.stmt]) -> bool:
    pending: list[ast.AST] = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(node, (*_SCOPE_NODES, ast.For, ast.AsyncFor, ast.While)):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False


def _record(
    name: str,
    context: ast.expr_context,
    line: int,
    bound: frozenset[str],
    module_names: set[str],
    referenced: set[str],
    reference_sites: set[tuple[str, int, bool, int | None]],
    deferred: bool,
    scope_line: int | None,
) -> None:
    if isinstance(context, ast.Load) and name in module_names and name not in bound:
        referenced.add(name)
        reference_sites.add((name, line, deferred, scope_line))


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


def _direct_parametrize_names(
    node: ast.AST,
    pytest_aliases: dict[str, str],
    constants: dict[str, str] | None = None,
) -> frozenset[str]:
    """Test arguments supplied as values do not request same-named fixtures."""

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return frozenset()
    direct: set[str] = set()
    for decorator in node.decorator_list:
        if (
            not isinstance(decorator, ast.Call)
            or _canonical_pytest_name(decorator.func, pytest_aliases)
            != "pytest.mark.parametrize"
        ):
            continue
        parameter_names = _parametrize_argnames(decorator, constants)
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
            indirect = _parameter_names(indirect_keyword, constants)
        else:
            indirect = set()
        direct.update(parameter_names - indirect)
    return frozenset(direct)


def _scope_bindings(node: ast.AST) -> frozenset[str]:  # noqa: C901
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
        if isinstance(child, _COMPREHENSIONS):
            names.update(_comprehension_walrus_names(child))
            continue
        if isinstance(child, _SCOPE_NODES):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif (pattern_name := _pattern_binding_name(child)) is not None:
            names.add(pattern_name)
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


def _pattern_binding_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, ast.MatchMapping):
        return node.rest
    return None


def _comprehension_target_names(target: ast.AST) -> frozenset[str]:
    return frozenset(
        child.id
        for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    )


def _comprehension_walrus_names(node: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(child, ast.NamedExpr):
            names.update(_assignment_names(child.target))
        stack.extend(ast.iter_child_nodes(child))
    return frozenset(names)


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


def _module_eager_call_map(tree: ast.Module) -> dict[int, tuple[int, ...]]:
    """Map functions to module-time invocation lines in one pass per module."""

    definitions_by_name: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions_by_name.setdefault(node.name, []).append(node)

    def calls(nodes: Sequence[ast.AST]) -> list[tuple[str, int]]:
        found: list[tuple[str, int]] = []
        stack = list(nodes)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(node.decorator_list)
                stack.extend(_header_expressions(node))
                continue
            if isinstance(node, ast.Lambda):
                stack.extend(_header_expressions(node))
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                found.append((node.func.id, node.lineno))
            stack.extend(ast.iter_child_nodes(node))
        return found

    pending = [(name, line) for name, line in calls(tree.body)]
    visited: set[tuple[int, int]] = set()
    invocation_lines: dict[int, set[int]] = {}
    while pending:
        called_name, invocation_line = pending.pop()
        candidates = [
            candidate
            for candidate in definitions_by_name.get(called_name, ())
            if candidate.lineno < invocation_line
        ]
        if not candidates:
            continue
        target = max(candidates, key=lambda candidate: candidate.lineno)
        state = (target.lineno, invocation_line)
        if state in visited:
            continue
        visited.add(state)
        invocation_lines.setdefault(target.lineno, set()).add(invocation_line)
        pending.extend((name, invocation_line) for name, _line in calls(target.body))
    return {line: tuple(sorted(lines)) for line, lines in invocation_lines.items()}


def _reachable_bindings(roots: set[int], edges: dict[int, set[int]]) -> set[int]:
    reachable = set(roots)
    pending = list(roots)
    while pending:
        source = pending.pop()
        for target in edges.get(source, set()) - reachable:
            reachable.add(target)
            pending.append(target)
    return reachable


def _scope_owner_binding(
    line: int | None,
    function_nodes: dict[int, ast.FunctionDef | ast.AsyncFunctionDef],
    parents: dict[ast.AST, ast.AST],
    function_bindings: dict[int, int],
    class_bindings: dict[int, int],
) -> int | None:
    if line in class_bindings:
        return class_bindings[line]
    current = function_nodes.get(line) if line is not None else None
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if parent.lineno in function_bindings:
                return function_bindings[parent.lineno]
        elif isinstance(parent, ast.ClassDef) and parent.lineno in class_bindings:
            return class_bindings[parent.lineno]
        current = parent
    return None


def _binding_before(choices: list[tuple[int, int, int]], line: int) -> int | None:
    eligible = [
        index
        for index, start_line, end_line in choices
        if start_line < line and not start_line <= line <= end_line
    ]
    return eligible[-1] if eligible else None


def _check_file(root: Path, path: Path) -> tuple[Violation, ...]:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, SyntaxError):
        return ()
    lines = source.splitlines()
    _referenced, reference_sites = _definition_only_references(tree)
    definitions = _definition_names(tree)
    referenced_bindings = _referenced_bindings(tree, definitions, reference_sites)
    return tuple(
        Violation(relative, line, name, kind)
        for index, (name, line, kind, _node) in enumerate(definitions)
        if not _waived(lines, line) and index not in referenced_bindings
    )


def _referenced_bindings(
    tree: ast.Module,
    definitions: list[tuple[str, int, str, ast.AST]],
    reference_sites: set[tuple[str, int, bool, int | None]],
) -> set[int]:
    """Resolve loads to bindings reachable from test and module execution roots."""

    bindings: dict[str, list[tuple[int, int, int]]] = {}
    for index, (name, line, _kind, _node) in enumerate(definitions):
        end_line = getattr(_node, "end_lineno", None) or line
        bindings.setdefault(name, []).append((index, line, end_line))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    class_bindings = {
        node.lineno: index
        for index, (_name, _line, kind, node) in enumerate(definitions)
        if kind == "function" and isinstance(node, ast.ClassDef)
    }
    function_nodes = {
        node.lineno: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    function_bindings = {
        node.lineno: index
        for index, (_name, _line, kind, node) in enumerate(definitions)
        if kind == "function"
        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    eager_call_lines = _module_eager_call_map(tree)
    module_roots: set[int] = set()
    runtime_roots: set[int] = set()
    module_edges: dict[int, set[int]] = {}
    runtime_edges: dict[int, set[int]] = {}
    for name, line, deferred, scope_line in reference_sites:
        choices = bindings.get(name, [])
        if not choices:
            continue
        source = function_bindings.get(scope_line) if scope_line is not None else None
        if source is None:
            source = _scope_owner_binding(
                scope_line, function_nodes, parents, function_bindings, class_bindings
            )
        if deferred:
            if source is None:
                runtime_roots.add(choices[-1][0])
            else:
                runtime_edges.setdefault(source, set()).add(choices[-1][0])
            for call_line in eager_call_lines.get(scope_line, ()):
                target = _binding_before(choices, call_line)
                if target is not None:
                    if source is None:
                        module_roots.add(target)
                    else:
                        module_edges.setdefault(source, set()).add(target)
        else:
            target = _binding_before(choices, line)
            if target is not None:
                if source is None:
                    module_roots.add(target)
                else:
                    runtime_edges.setdefault(source, set()).add(target)
    referenced_bindings = _reachable_bindings(
        module_roots, module_edges
    ) | _reachable_bindings(runtime_roots, runtime_edges)
    return referenced_bindings


def _test_files(root: Path) -> Iterable[Path]:
    """Only collected test modules.

    Shared helper modules such as ``tests/dispatch/_support.py`` and
    ``tests/fixtures/accounting.py`` are imported by other modules, so they
    legitimately have no in-file reference.
    """
    tests_root = root / _TESTS_ROOT
    if not tests_root.is_dir():
        return ()
    paths = set(tests_root.rglob("test_*.py")) | set(tests_root.rglob("*_test.py"))
    return (
        path
        for path in sorted(paths)
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
