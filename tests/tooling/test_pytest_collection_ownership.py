"""Static ownership contracts for pytest's configured default roots."""

from __future__ import annotations

import ast
import tomllib
from itertools import combinations
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).parents[2]
SPECIALIST_ROOTS = (
    PurePosixPath("tests/process"),
    PurePosixPath("tests/mcp"),
)
ORDINARY_OWNERS = frozenset(
    {"catalog", "cli", "dispatch", "integration", "math", "tooling"}
)


def _nested_test_functions(path: Path) -> tuple[str, ...]:
    """Return test definitions pytest cannot collect from ``path``.

    Pytest discovers module-level tests and methods on test classes, but a
    ``test_*`` definition below another function is inert.  Parsing the syntax
    lets this ownership check catch that mistake without making collection a
    prerequisite of each ordinary test lane.
    """

    try:
        display_path: str = str(path.relative_to(ROOT))
    except ValueError:
        display_path = path.name

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nested: list[str] = []

    class Visitor(ast.NodeVisitor):
        function_depth = 0

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if node.name.startswith("test_") and self.function_depth:
                nested.append(f"{display_path}:{node.lineno}")
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    Visitor().visit(tree)
    return tuple(nested)


def _configured_testpaths() -> tuple[PurePosixPath, ...]:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    configured = project["tool"]["pytest"]["ini_options"]["testpaths"]
    assert isinstance(configured, list)
    assert all(isinstance(path, str) for path in configured)
    return tuple(PurePosixPath(path) for path in configured)


def _overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def test_default_testpaths_are_normalized_existing_ordinary_roots() -> None:
    for relative in _configured_testpaths():
        assert not relative.is_absolute()
        assert relative.parts[:1] == ("tests",)
        assert "." not in relative.parts and ".." not in relative.parts
        assert (ROOT / relative).is_dir()

        owner = relative.parts[1:2]
        assert owner and owner[0] in ORDINARY_OWNERS


def test_default_testpaths_do_not_overlap_specialist_roots() -> None:
    configured = _configured_testpaths()

    assert all(
        not _overlap(default_root, specialist_root)
        for default_root in configured
        for specialist_root in SPECIALIST_ROOTS
    )


def test_default_testpaths_do_not_overlap_each_other() -> None:
    assert all(
        not _overlap(left, right)
        for left, right in combinations(_configured_testpaths(), 2)
    )


def test_test_functions_are_not_nested_under_functions() -> None:
    violations = tuple(
        location
        for path in sorted((ROOT / "tests").rglob("*.py"))
        for location in _nested_test_functions(path)
    )

    assert violations == ()


def test_nested_test_function_check_rejects_an_uncollected_definition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test_example.py"
    path.write_text(
        "def helper() -> None:\n"
        "    def test_never_collected() -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )

    assert _nested_test_functions(path) == ("test_example.py:2",)


def test_math_tests_do_not_boot_complete_product_boundaries() -> None:
    forbidden = (
        "jacobian.catalog.catalog",
        "jacobian.cli",
        "jacobian.dispatch",
        "jacobian.mcp",
    )
    violations: list[str] = []
    for path in sorted((ROOT / "tests/math").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            else:
                continue
            if any(module.startswith(forbidden) for module in modules):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert violations == []
