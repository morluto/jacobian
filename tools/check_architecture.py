"""Check the few Jacobian source boundaries that need custom AST analysis.

Import Linter owns dependency direction; Ruff, mypy, deptry, and vulture own
their native static checks.  This module is deliberately limited to rules
specific to Jacobian's process and exact-wire-value boundaries.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
_PRODUCT_ROOT = PurePosixPath("src/jacobian")
_PROCESS_OWNER = PurePosixPath("src/jacobian/process.py")
_EXTERNAL_OPERATION_OWNERS = frozenset(
    {
        _PROCESS_OWNER,
        PurePosixPath("src/jacobian/math/_singular.py"),
        PurePosixPath("src/jacobian/math/commutative_algebra_ops/_singular.py"),
        PurePosixPath("src/jacobian/math/logic/_operations.py"),
        PurePosixPath(
            "src/jacobian/math/polynomials/multivariate/_factor_backend.py"
        ),
        PurePosixPath("src/jacobian/math/polynomials/maps/_replay.py"),
    }
)
_GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
)
_EXEC_FUNCTIONS = frozenset({"execlp", "execv", "execve", "execvp", "execvpe"})
_EVALUATOR_CAPABLE_FUNCTIONS = frozenset(
    {"eval", "exec", "lambdify", "parse_expr", "sympify"}
)
_RATIONAL_COMPONENTS = frozenset({"denominator", "numerator", "p", "q"})
_DESCRIPTIVE_RATIONAL_COMPONENTS = frozenset({"denominator", "numerator"})
_EMBEDDED_PROCESS_PATTERNS = (
    "import subprocess",
    "from subprocess import ",
    "subprocess.Popen(",
    "subprocess.call(",
    "subprocess.check_call(",
    "subprocess.check_output(",
    "subprocess.run(",
)


@dataclass(frozen=True)
class Violation:
    """One source-boundary violation."""

    path: str
    code: str
    message: str
    line: int | None = None

    @property
    def location(self) -> str:
        suffix = f":{self.line}" if self.line is not None else ""
        return f"{self.path}{suffix}"

    def __str__(self) -> str:
        return f"{self.location}: [{self.code}] {self.message}"


@dataclass(frozen=True)
class ArchitectureReport:
    """Result of checking the installed product source."""

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
            return f"architecture: OK ({self.files_scanned} files checked)"
        lines = [
            f"architecture: {len(self.violations)} violation(s) "
            f"({self.files_scanned} files checked)"
        ]
        lines.extend(str(violation) for violation in self.violations)
        return "\n".join(lines)


class ArchitecturePolicyError(RuntimeError):
    """Raised when product source violates a custom boundary."""

    def __init__(self, report: ArchitectureReport) -> None:
        self.report = report
        super().__init__(report.render())


def _walk(tree: ast.AST) -> tuple[ast.AST, ...]:
    return tuple(ast.walk(tree))


def _violation(
    relative: PurePosixPath,
    node: ast.AST,
    code: str,
    message: str,
) -> Violation:
    return Violation(str(relative), code, message, getattr(node, "lineno", None))


def _process_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if relative == _PROCESS_OWNER:
        return ()
    violations: list[Violation] = []
    for node in _walk(tree):
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "subprocess" for alias in node.names)
        ) or (isinstance(node, ast.ImportFrom) and node.module == "subprocess"):
            violations.append(
                _violation(
                    relative,
                    node,
                    "subprocess-confined",
                    "direct subprocess use belongs in jacobian.process",
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr in _EXEC_FUNCTIONS
        ):
            violations.append(
                _violation(
                    relative,
                    node,
                    "subprocess-confined",
                    f"os.{node.func.attr} is unbounded process replacement",
                )
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(pattern in node.value for pattern in _EMBEDDED_PROCESS_PATTERNS)
        ):
            violations.append(
                _violation(
                    relative,
                    node,
                    "subprocess-confined",
                    "embedded worker source must not bypass jacobian.process",
                )
            )
    return tuple(violations)


def _bounded_process_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if relative in _EXTERNAL_OPERATION_OWNERS:
        return ()
    violations: list[Violation] = []
    for node in _walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "jacobian.process"
            and any(alias.name == "run_bounded_process" for alias in node.names)
        ) or (
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "run_bounded_process"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run_bounded_process"
                )
            )
        ):
            violations.append(
                _violation(
                    relative,
                    node,
                    "bounded-process-gateway",
                    "run_bounded_process requires a concrete external-tool owner",
                )
            )
    return tuple(violations)


def _resolver_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if relative in _EXTERNAL_OPERATION_OWNERS:
        return ()
    return tuple(
        _violation(
            relative,
            node,
            "shutil-which-resolver",
            "external executable discovery requires a concrete tool owner",
        )
        for node in _walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "shutil"
        and node.attr == "which"
    )


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
    )


def _spreads_environ(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            and len(node.args) == 1
            and _is_os_environ(node.args[0])
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy"
            and _is_os_environ(node.func.value)
        ):
            return True
        return any(
            keyword.arg is None and _is_os_environ(keyword.value)
            for keyword in node.keywords
        )
    return isinstance(node, ast.Dict) and any(
        key is None and _is_os_environ(value)
        for key, value in zip(node.keys, node.values, strict=True)
    )


def _environment_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    return tuple(
        _violation(
            relative,
            node,
            "environ-spreading",
            "copy only explicitly allowed environment variables",
        )
        for node in _walk(tree)
        if _spreads_environ(node)
    )


def _unsafe_wire_conversion_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    return tuple(
        _violation(
            relative,
            node,
            "unsafe-canonical-conversion",
            "use the canonical conversion API for rational wire components",
        )
        for node in _walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"int", "str"}
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr in {"num", "den"}
    )


def _evaluator_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    direct_names, builtin_modules = _imported_evaluator_aliases(tree)
    changed = True
    while changed:
        changed = False
        for node in _walk(tree):
            for target, value in _simple_assignments(node):
                if (
                    target not in direct_names
                    and _evaluator_reference_name(value, direct_names, builtin_modules)
                    is not None
                ):
                    direct_names.add(target)
                    changed = True
    return direct_names, builtin_modules


def _imported_evaluator_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    direct_names = set(_EVALUATOR_CAPABLE_FUNCTIONS)
    builtin_modules = {"builtins"}
    for node in _walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtin_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (node.module == "builtins" and alias.name in {"eval", "exec"}) or (
                    node.module is not None
                    and (node.module == "sympy" or node.module.startswith("sympy."))
                    and alias.name in _EVALUATOR_CAPABLE_FUNCTIONS
                ):
                    direct_names.add(alias.asname or alias.name)
    return direct_names, builtin_modules


def _simple_assignments(node: ast.AST) -> tuple[tuple[str, ast.expr], ...]:
    if isinstance(node, ast.Assign):
        return tuple(
            (target.id, node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        )
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return ((node.target.id, node.value),)
    return ()


def _evaluator_reference_name(
    node: ast.expr,
    direct_names: set[str],
    builtin_modules: set[str],
) -> str | None:
    if isinstance(node, ast.Name):
        return node.id if node.id in direct_names else None
    if (
        isinstance(node, ast.Attribute)
        and node.attr in _EVALUATOR_CAPABLE_FUNCTIONS
        and (
            node.attr not in {"eval", "exec"}
            or (isinstance(node.value, ast.Name) and node.value.id in builtin_modules)
        )
    ):
        return node.attr
    return None


def _evaluator_parser_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    """Keep evaluator-capable parsers out of the mathematical operation tree."""

    if not relative.is_relative_to(PurePosixPath("src/jacobian/math")):
        return ()
    direct_names, builtin_modules = _evaluator_aliases(tree)
    violations: list[Violation] = []
    for node in _walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _evaluator_reference_name(node.func, direct_names, builtin_modules)
        if name is not None:
            violations.append(
                _violation(
                    relative,
                    node,
                    "evaluator-capable-parser",
                    f"{name} is forbidden in public mathematical input flows",
                )
            )
    return tuple(violations)


def _contains_component(node: ast.AST, attributes: frozenset[str]) -> bool:
    return any(
        isinstance(descendant, ast.Attribute) and descendant.attr in attributes
        for descendant in ast.walk(node)
    )


def _uses_canonical_formatter(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "format_canonical_integer"
    )


def _unsafe_render_nodes(
    node: ast.AST, attributes: frozenset[str]
) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.JoinedStr):
        return tuple(
            value
            for value in node.values
            if isinstance(value, ast.FormattedValue)
            and _contains_component(value.value, attributes)
            and not _uses_canonical_formatter(value.value)
        )
    if not isinstance(node, ast.Call):
        return ()
    arguments: tuple[ast.AST, ...]
    if isinstance(node.func, ast.Name) and node.func.id in {"format", "str"}:
        arguments = tuple(node.args)
    elif isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        arguments = (*node.args, *(keyword.value for keyword in node.keywords))
    else:
        return ()
    if any(
        _contains_component(argument, attributes)
        and not _uses_canonical_formatter(argument)
        for argument in arguments
    ):
        return (node,)
    return ()


def _rational_output_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    unsafe: dict[int, ast.AST] = {}
    for node in _walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
            value = node.value
            if value is not None:
                for render in _unsafe_render_nodes(
                    value, _DESCRIPTIVE_RATIONAL_COMPONENTS
                ):
                    unsafe[id(render)] = render
        if isinstance(node, ast.Call):
            sink_values = tuple(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"num", "den"}
            )
        elif isinstance(node, ast.Dict):
            sink_values = tuple(
                value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and key.value in {"num", "den"}
            )
        else:
            sink_values = ()
        for value in sink_values:
            for render in _unsafe_render_nodes(value, _RATIONAL_COMPONENTS):
                unsafe[id(render)] = render
    return tuple(
        _violation(
            relative,
            node,
            "unsafe-canonical-rational-output",
            "format rational result components with format_canonical_integer",
        )
        for node in unsafe.values()
    )


def _source_files(root: Path) -> tuple[Path, ...]:
    source_root = root / _PRODUCT_ROOT
    if not source_root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(source_root.rglob("*.py"))
        if not any(part in _GENERATED_DIRECTORIES for part in path.parts)
    )


def _check_file(root: Path, path: Path) -> tuple[Violation, ...]:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    except (OSError, SyntaxError) as exc:
        return (Violation(str(relative), "parse-error", f"cannot parse file: {exc}"),)
    return (
        *_process_violations(relative, tree),
        *_bounded_process_violations(relative, tree),
        *_resolver_violations(relative, tree),
        *_environment_violations(relative, tree),
        *_evaluator_parser_violations(relative, tree),
        *_unsafe_wire_conversion_violations(relative, tree),
        *_rational_output_violations(relative, tree),
    )


def check_architecture(root: Path | str = ROOT) -> ArchitectureReport:
    """Check installed product source without importing the runtime."""

    project_root = Path(root).resolve()
    files = _source_files(project_root)
    violations = tuple(
        sorted(
            (
                violation
                for path in files
                for violation in _check_file(project_root, path)
            ),
            key=lambda item: (item.path, item.line or 0, item.code),
        )
    )
    return ArchitectureReport(project_root, violations, len(files))


def assert_architecture(root: Path | str = ROOT) -> ArchitectureReport:
    """Check architecture and raise when a boundary is violated."""

    report = check_architecture(root)
    if report.failed:
        raise ArchitecturePolicyError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = check_architecture(args.root)
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
