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
        PurePosixPath("src/jacobian/math/graphs/isomorphism/_operations.py"),
        PurePosixPath("src/jacobian/math/polynomials/ideals/_singular.py"),
        PurePosixPath("src/jacobian/math/logic/_sat.py"),
        PurePosixPath("src/jacobian/math/logic/_smt.py"),
        PurePosixPath("src/jacobian/math/logic/_unsat_core.py"),
        PurePosixPath("src/jacobian/math/hypergraphs/_independence_z3.py"),
        PurePosixPath("src/jacobian/math/graphs/_independence_z3.py"),
        PurePosixPath("src/jacobian/math/graphs/coloring/_operations.py"),
        PurePosixPath("src/jacobian/math/graphs/optimization/_finite_optimization.py"),
        PurePosixPath("src/jacobian/math/graphs/optimization/_invariants.py"),
        PurePosixPath("src/jacobian/math/graphs/optimization/_chromatic_number.py"),
        PurePosixPath("src/jacobian/math/graphs/optimization/_maximum_cut_process.py"),
        PurePosixPath("src/jacobian/math/discrepancy_theory/_optimum_process.py"),
        PurePosixPath("src/jacobian/math/number_theory/_factorization_kernels.py"),
        PurePosixPath("src/jacobian/math/number_field/_operations.py"),
        PurePosixPath("src/jacobian/math/polynomials/multivariate/_factor_backend.py"),
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
_RESULT_VALIDATOR_KERNEL_CALLS = frozenset(
    {
        "Solver",
        "SolverFor",
        "_solve_analysis_by_enumeration",
        "_evaluate_chromatic_number_certificate",
        "_solve_terminal_game_data",
        "betti_data",
        "characteristic_polynomial",
        "delta_periodicity_bound",
        "factor_list",
        "factorization_length_extrema",
        "factorization_lengths",
        "factorizations",
        "factors_of_length",
        "invariant_factors",
        "is_irreducible",
        "minimal_polynomial",
        "parse_smt2_string",
        "periods",
        "primary_decomposition",
    }
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


def _owner_operation_module(relative: PurePosixPath) -> str | None:
    """Return the operation-module prefix forbidden to one contract file.

    Contracts and canonical values may depend on neutral values from another
    owner, but must not re-enter their own public operation path.  This stays
    deliberately narrower than a backend-import rule: it constrains only the
    owner that would otherwise make parsing execute its own kernel.
    """

    if relative.name not in {"_models.py", "values.py"} or not relative.is_relative_to(
        PurePosixPath("src/jacobian/math")
    ):
        return None
    owner_parts = relative.parent.relative_to(PurePosixPath("src")).parts
    return ".".join(owner_parts)


def _is_owner_operation_module(module: str, owner: str) -> bool:
    return module in {f"{owner}.operations", f"{owner}._operations"}


def _resolve_import_from_module(
    node: ast.ImportFrom, relative: PurePosixPath
) -> str | None:
    """Resolve one ``from`` target relative to its source package."""

    if node.level == 0:
        return node.module
    package_parts = relative.parent.relative_to(PurePosixPath("src")).parts
    parents_to_remove = node.level - 1
    if parents_to_remove >= len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - parents_to_remove]
    if node.module is not None:
        base_parts += tuple(node.module.split("."))
    return ".".join(base_parts)


def _dynamic_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return direct ``import_module`` names and imported ``importlib`` names."""

    functions = {"import_module"}
    modules = {"importlib"}
    for node in _walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    functions.add(alias.asname or alias.name)
    return functions, modules


def _is_dynamic_import_call(
    node: ast.Call, functions: set[str], modules: set[str]
) -> bool:
    return (isinstance(node.func, ast.Name) and node.func.id in functions) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in modules
    )


def _owner_operation_reentry_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    """Reject static and literal dynamic re-entry into an owner's kernel."""

    owner = _owner_operation_module(relative)
    if owner is None:
        return ()
    dynamic_functions, dynamic_modules = _dynamic_import_aliases(tree)
    violations: list[Violation] = []
    for node in _walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _resolve_import_from_module(node, relative)
            if (module is not None and _is_owner_operation_module(module, owner)) or (
                module == owner
                and any(
                    alias.name in {"operations", "_operations"} for alias in node.names
                )
            ):
                violations.append(
                    _violation(
                        relative,
                        node,
                        "owner-operation-reentry",
                        "contract and value modules must not import their own operations",
                    )
                )
        elif isinstance(node, ast.Import):
            if any(
                _is_owner_operation_module(alias.name, owner) for alias in node.names
            ):
                violations.append(
                    _violation(
                        relative,
                        node,
                        "owner-operation-reentry",
                        "contract and value modules must not import their own operations",
                    )
                )
        elif (
            isinstance(node, ast.Call)
            and _is_dynamic_import_call(node, dynamic_functions, dynamic_modules)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_owner_operation_module(node.args[0].value, owner)
        ):
            violations.append(
                _violation(
                    relative,
                    node,
                    "owner-operation-reentry",
                    "contract and value modules must not dynamically import their own operations",
                )
            )
    return tuple(violations)


def _is_model_validator(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether one method is a Pydantic model validator."""

    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "model_validator")
        or (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "model_validator"
        )
        for decorator in function.decorator_list
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _result_validator_replay_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    """Keep owner kernels and solver APIs out of public result validators.

    Successful producers establish their invariant once and use a private
    trusted factory.  Independently supplied claims must use an explicit,
    owner-local verifier with a declared envelope instead of hiding replay in
    ordinary Pydantic deserialization.  The named set is deliberately narrow:
    it covers the known expensive kernel/backend entry points without treating
    cheap structural predicates as architecture violations.
    """

    if not relative.is_relative_to(PurePosixPath("src/jacobian/math")):
        return ()
    if not isinstance(tree, ast.Module):
        return ()
    violations: list[Violation] = []
    for result_class in (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Result")
    ):
        for method in result_class.body:
            if not isinstance(
                method, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) or not _is_model_validator(method):
                continue
            for call in (
                node for node in ast.walk(method) if isinstance(node, ast.Call)
            ):
                name = _call_name(call)
                if name in _RESULT_VALIDATOR_KERNEL_CALLS:
                    violations.append(
                        _violation(
                            relative,
                            call,
                            "result-validator-replay",
                            "result validators must not call owner kernels, backends, or solvers; use an explicit bounded verifier",
                        )
                    )
    return tuple(violations)


def _literal_string_sequence(tree: ast.AST, name: str) -> tuple[str, ...]:
    """Return a literal ``list`` or ``tuple`` assignment when one is present."""

    for node in tree.body if isinstance(tree, ast.Module) else ():
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return ()
        values = tuple(
            item.value
            for item in node.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
        return values if len(values) == len(node.value.elts) else ()
    return ()


def _module_name(relative: PurePosixPath) -> str | None:
    """Map one source path below ``src`` to its importable module name."""

    if not relative.is_relative_to(PurePosixPath("src")):
        return None
    parts = relative.relative_to(PurePosixPath("src")).parts
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    if not parts[-1].endswith(".py"):
        return None
    return ".".join((*parts[:-1], parts[-1][:-3]))


def _module_path(root: Path, module: str) -> Path | None:
    """Resolve an installed source module without importing it."""

    base = root / "src" / Path(*module.split("."))
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = base / "__init__.py"
    return package_file if package_file.is_file() else None


def _import_module_name(
    root: Path, node: ast.ImportFrom, source_module: str
) -> str | None:
    """Resolve an ``ImportFrom`` module relative to its source module."""

    if node.level == 0:
        return node.module
    source_parts = source_module.split(".")
    source_path = _module_path(root, source_module)
    if source_path is not None and source_path.name != "__init__.py":
        source_parts = source_parts[:-1]
    parent_parts = source_parts[: len(source_parts) - (node.level - 1)]
    if not parent_parts:
        return None
    return ".".join((*parent_parts, *(node.module or "").split(".")))


def _imports_by_local_name(
    root: Path, tree: ast.AST, source_module: str
) -> dict[str, tuple[str, str]]:
    """Return statically resolvable imported symbols by local binding name."""

    imports: dict[str, tuple[str, str]] = {}
    for node in tree.body if isinstance(tree, ast.Module) else ():
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _import_module_name(root, node, source_module)
        if module is None:
            continue
        for alias in node.names:
            imports[alias.asname or alias.name] = (module, alias.name)
    return imports


def _function_target(
    root: Path,
    module: str,
    symbol: str,
    seen: set[tuple[str, str]] | None = None,
) -> tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Resolve a re-exported function through static ``from`` imports only."""

    seen = seen or set()
    if (module, symbol) in seen:
        return None
    seen.add((module, symbol))
    path = _module_path(root, module)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol
        ):
            return path, node
    imported = _imports_by_local_name(root, tree, module).get(symbol)
    if imported is None:
        return None
    return _function_target(root, *imported, seen)


def _native_public_function_targets(
    root: Path,
) -> tuple[tuple[PurePosixPath, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    """Resolve the functions exposed by root ``jacobian.math`` domains.

    ``jacobian.math.__all__`` and each domain's literal ``__all__`` are the
    supported native surface.  Following only their static re-exports keeps
    this check out of private MCP/catalog adapters and unrelated operation
    modules.
    """

    math_module = "jacobian.math"
    math_path = _module_path(root, math_module)
    if math_path is None:
        return ()
    try:
        math_tree = ast.parse(math_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ()
    targets: dict[
        tuple[str, int], tuple[PurePosixPath, ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for domain in _literal_string_sequence(math_tree, "__all__"):
        module = f"{math_module}.{domain}"
        path = _module_path(root, module)
        if path is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for symbol in _literal_string_sequence(tree, "__all__"):
            target = _function_target(root, module, symbol)
            if target is None:
                continue
            source, function = target
            relative = PurePosixPath(source.relative_to(root).as_posix())
            targets[(str(relative), function.lineno)] = (relative, function)
    return tuple(targets.values())


def _root_native_export_violations(root: Path) -> tuple[Violation, ...]:
    """Keep statically imported native domains listed in the root surface."""

    path = _module_path(root, "jacobian.math")
    if path is None:
        return ()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return ()
    exported = set(_literal_string_sequence(tree, "__all__"))
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "jacobian.math"
        for alias in node.names
        if alias.name != "*"
    }
    return tuple(
        _violation(
            PurePosixPath("src/jacobian/math/__init__.py"),
            tree,
            "native-root-export",
            f"native domain {name!r} is imported but missing from jacobian.math.__all__",
        )
        for name in sorted(imported - exported)
    )


def _wire_model_names(root: Path, tree: ast.AST, source_module: str) -> set[str]:
    """Find local bindings that name a Request/Input wire model."""

    names = set()
    if isinstance(tree, ast.Module):
        names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name.endswith(("Request", "Input"))
        }
    for local, (_, original) in _imports_by_local_name(
        root, tree, source_module
    ).items():
        if original.endswith(("Request", "Input")):
            names.add(local)
    return names


def _is_wire_model_reference(node: ast.AST, names: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in names) or (
        isinstance(node, ast.Attribute) and node.attr.endswith(("Request", "Input"))
    )


def _annotation_contains_wire_model(node: ast.AST, names: set[str]) -> bool:
    """Return whether an annotation names a request/input wire model."""

    return any(
        _is_wire_model_reference(descendant, names) for descendant in ast.walk(node)
    )


def _native_compute_adapter_names(
    root: Path, tree: ast.AST, source_module: str
) -> set[str]:
    """Find public ``operations``-module compute adapters imported by a wrapper."""

    return {
        local
        for local, (module, original) in _imports_by_local_name(
            root, tree, source_module
        ).items()
        if module.endswith(".operations") and original.startswith("compute_")
    }


def _native_public_boundary_violations(root: Path) -> tuple[Violation, ...]:
    """Keep exported native functions on canonical values and direct kernels."""

    violations: list[Violation] = []
    parsed: dict[PurePosixPath, tuple[ast.Module, str]] = {}
    for relative, function in _native_public_function_targets(root):
        if relative not in parsed:
            path = root / relative
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(relative)
                )
            except (OSError, SyntaxError):
                continue
            module = _module_name(relative)
            if module is None:
                continue
            parsed[relative] = tree, module
        tree, module = parsed[relative]
        wire_names = _wire_model_names(root, tree, module)
        compute_adapters = _native_compute_adapter_names(root, tree, module)
        for node in _walk(function):
            if isinstance(node, ast.arg) and node.annotation is not None:
                if _annotation_contains_wire_model(node.annotation, wire_names):
                    violations.append(
                        _violation(
                            relative,
                            node.annotation,
                            "native-wire-boundary",
                            "exported native functions must not accept wire Request/Input models",
                        )
                    )
            elif (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.returns is not None
                and _annotation_contains_wire_model(node.returns, wire_names)
            ):
                violations.append(
                    _violation(
                        relative,
                        node.returns,
                        "native-wire-boundary",
                        "exported native functions must not return wire Request/Input models",
                    )
                )
            if not isinstance(node, ast.Call):
                continue
            constructs_wire = _is_wire_model_reference(node.func, wire_names) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "model_validate"
                and _is_wire_model_reference(node.func.value, wire_names)
            )
            if constructs_wire:
                violations.append(
                    _violation(
                        relative,
                        node,
                        "native-wire-boundary",
                        "exported native functions must not construct wire Request/Input models",
                    )
                )
            elif isinstance(node.func, ast.Name) and node.func.id in compute_adapters:
                violations.append(
                    _violation(
                        relative,
                        node,
                        "native-wire-boundary",
                        "exported native functions must call a private direct kernel, not a public compute adapter",
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
        *_owner_operation_reentry_violations(relative, tree),
        *_result_validator_replay_violations(relative, tree),
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
                *(
                    violation
                    for path in files
                    for violation in _check_file(project_root, path)
                ),
                *_native_public_boundary_violations(project_root),
                *_root_native_export_violations(project_root),
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
