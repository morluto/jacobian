"""Static policy checks for the ownership boundaries of the test suite.

The checker is intentionally a small AST/filesystem tool.  It does not import
Jacobian (or pytest), so running it cannot accidentally construct a runtime or
probe an optional provider.  ``check_test_architecture`` returns a report for
callers such as CI and the command line entry point turns that report into a
useful, location-aware diagnostic.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tools.test_architecture.lanes import owners
from tools.test_architecture.runtime_owners import allows_create_runtime

_TEST_FILE = "test_*.py"

# These are deliberately narrow names.  Importing a typed domain model from
# ``jacobian.portfolio`` is fine; importing the explicit built-in plan or its
# complete installer is the operation that couples a lower tier to the whole
# application.
_BUILTIN_PORTFOLIO_NAMES = frozenset(
    {
        "BUILTIN_PORTFOLIO",
        "build_builtin_portfolio",
        "PortfolioAssembler",
        "install_builtin_portfolio",
        "install_complete_portfolio",
        "install_portfolio",
    }
)
_BUILTIN_PORTFOLIO_MODULES = frozenset(
    {"jacobian.portfolio.builtin", "jacobian.portfolio.assembler"}
)

# Provider implementation modules and third-party engines.  Provider metadata
# and the lazy-loader contracts are intentionally absent from this list.
_PROVIDER_MODULE_PREFIXES = (
    # ``jacobian.providers`` contains only the metadata and lazy-loader
    # contracts.  Keep those importable from unit tests; the concrete
    # provider adapters live in the explicitly named modules below.
    "jacobian.flint_",
    "jacobian.sat_smt.cvc5",
    "jacobian.lean_",
    "jacobian.lean_frontend.service",
    "jacobian.z3_",
    "jacobian.sympy_",
)
_PROVIDER_EXTERNAL_MODULES = frozenset(
    {"sympy", "networkx", "z3", "cvc5", "flint", "z3solver"}
)

_UNIT_PREFIX = PurePosixPath("tests/unit")
_COMPONENT_PREFIX = PurePosixPath("tests/component")
_DOMAIN_PREFIX = PurePosixPath("tests/domain")
_COMPOSITION_PREFIX = PurePosixPath("tests/composition")
_BOUNDARY_PREFIX = PurePosixPath("tests/boundary")
_E2E_PREFIXES = (PurePosixPath("tests/e2e"),)

# Ordinary domain fixtures must use the typed verified-domain seam rather than
# reassembling install_exact_domain_verification by hand.
_EXACT_DOMAIN_INSTALL_ALLOWLIST = frozenset(
    {
        "src/jacobian/portfolio/core_installation.py",
        "src/jacobian/exact_domain_checkers.py",
        "tests/support/exact_domain.py",
        "tests/component/checkers/test_exact_domain_checker_installation.py",
    }
)

_COMPOSITION_ADMISSION_CATEGORIES = frozenset(
    {
        "AUTHORITY",
        "WIRING",
        "LIFECYCLE",
        "DISCOVERY",
        "REFERENCE",
        "MIXED",
    }
)
_COMPLETE_RUNTIME_FIXTURE_NAMES = frozenset(
    {
        "fresh_complete_runtime",
        "attached_complete_runtime",
        "attached_complete_runtime_read_only",
        "authorized_complete_runtime",
        "authorized_complete_runtime_read_only",
        "complete_portfolio_template",
        "authorized_portfolio_template",
    }
)


@dataclass(frozen=True)
class Violation:
    """One architectural policy violation."""

    path: str
    code: str
    message: str
    line: int | None = None
    column: int | None = None

    @property
    def location(self) -> str:
        suffix = ""
        if self.line is not None:
            suffix = f":{self.line}"
            if self.column is not None:
                suffix += f":{self.column}"
        return f"{self.path}{suffix}"

    def __str__(self) -> str:
        return f"{self.location}: [{self.code}] {self.message}"


@dataclass(frozen=True)
class ArchitectureReport:
    """Result of checking a test tree.

    ``violations`` always contains the complete observed set.  In ratchet mode
    ``new_violations`` is the subset not present in the supplied baseline and
    ``failed`` is based on that subset.
    """

    root: Path
    violations: tuple[Violation, ...]
    files_scanned: int
    new_violations: tuple[Violation, ...] | None = None
    mode: str = "strict"

    @property
    def failed(self) -> bool:
        effective = (
            self.violations if self.new_violations is None else self.new_violations
        )
        return bool(effective)

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        effective = (
            self.violations if self.new_violations is None else self.new_violations
        )
        if not effective:
            return f"test architecture: OK ({self.files_scanned} files checked)"
        lines = [
            f"test architecture: {len(effective)} violation(s) "
            f"({self.mode} mode; {self.files_scanned} files checked)"
        ]
        lines.extend(str(item) for item in effective)
        if self.new_violations is not None and len(self.new_violations) != len(
            self.violations
        ):
            lines.append(
                f"{len(self.violations) - len(self.new_violations)} known violation(s) "
                "allowed by ratchet baseline"
            )
        return "\n".join(lines)


class ArchitecturePolicyError(RuntimeError):
    """Raised by :func:`assert_test_architecture` when the policy fails."""

    def __init__(self, report: ArchitectureReport) -> None:
        self.report = report
        super().__init__(report.render())


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


def _tier(path: PurePosixPath) -> str | None:
    """Return the coarse pytest directory, not named-lane ownership."""

    if _is_under(path, _UNIT_PREFIX):
        return "unit"
    if _is_under(path, _COMPONENT_PREFIX):
        return "component"
    if _is_under(path, _DOMAIN_PREFIX):
        return "domain"
    if _is_under(path, _COMPOSITION_PREFIX):
        return "composition"
    if _is_under(path, _BOUNDARY_PREFIX):
        return "boundary"
    if any(_is_under(path, prefix) for prefix in _E2E_PREFIXES):
        return "e2e"
    return None


def _runtime_allowed(path: PurePosixPath, tier_override: str | None = None) -> bool:
    return allows_create_runtime(path.as_posix(), tier=tier_override)


def _provider_allowed(path: PurePosixPath, tier_override: str | None = None) -> bool:
    tier = tier_override or _tier(path)
    if tier == "boundary":
        return True
    if tier == "component":
        return "providers" in path.parts or "provider" in path.stem.lower()
    return False


def _node_location(path: str, node: ast.AST) -> tuple[str, int, int]:
    return path, getattr(node, "lineno", 1), getattr(node, "col_offset", 0)


def _complete_runtime_fixture_import_violations(
    module: str,
    relative: str,
    node: ast.Import | ast.ImportFrom,
) -> tuple[Violation, ...]:
    """Complete-runtime fixture bindings stay out of lower-tier modules."""

    if module != "tests.support.complete_runtime_fixtures":
        return ()
    owners = (
        "tests/composition/conftest.py",
        "tests/e2e/conftest.py",
        "tests/boundary/storage/conftest.py",
        "tests/boundary/providers/conftest.py",
        "tests/support/complete_runtime_fixtures.py",
    )
    if relative in owners:
        return ()
    return (
        Violation(
            relative,
            "complete-runtime-fixture-import",
            "import complete-runtime fixtures only from owning-tier conftest.py",
            node.lineno,
            node.col_offset,
        ),
    )


def _imported_module(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if node.level:
        return ""
    return node.module or ""


def _conftest_import_violations(
    module: str,
    relative: str,
    node: ast.Import | ast.ImportFrom,
) -> tuple[Violation, ...]:
    if module != "conftest" and not module.endswith(".conftest"):
        return ()
    return (
        Violation(
            relative,
            "conftest-import",
            "fixtures must be registered from a support plugin, not imported from conftest.py",
            node.lineno,
            node.col_offset,
        ),
    )


def _provider_module(module: str) -> bool:
    root = module.split(".", 1)[0]
    return (
        module.startswith(_PROVIDER_MODULE_PREFIXES)
        or root in _PROVIDER_EXTERNAL_MODULES
    )


def _is_portfolio_import(module: str, imported: str) -> bool:
    full = f"{module}.{imported}" if imported else module
    return (
        imported in _BUILTIN_PORTFOLIO_NAMES
        or module in _BUILTIN_PORTFOLIO_MODULES
        or full in _BUILTIN_PORTFOLIO_MODULES
    )


def _fixture_is_high_cost(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    name = node.name.casefold()
    if any(token in name for token in ("runtime", "portfolio", "provider", "process")):
        return True
    return name in {
        "durable_store",
        "checker_process",
        "mcp_server_process",
        "lean_environment",
        "available_cvc5",
        "complete_portfolio_template",
        "fresh_complete_runtime",
        "attached_complete_runtime",
        "attached_complete_runtime_read_only",
        "authorized_complete_runtime",
        "authorized_complete_runtime_read_only",
    }


def _is_pytest_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if (
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "pytest"
            and decorator.attr == "fixture"
        ):
            return True
        if isinstance(decorator, ast.Name) and decorator.id == "fixture":
            return True
    return False


def _baseline_keys(source: Path | Iterable[str] | Mapping[str, Any] | None) -> set[str]:
    if source is None:
        return set()
    raw: Any = source
    if isinstance(source, Path):
        raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = raw.get("violations", raw.get("baseline", raw.keys()))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Iterable):
        return set()
    keys: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            keys.add(item)
        elif isinstance(item, Mapping):
            path = str(item.get("path", ""))
            code = str(item.get("code", ""))
            line = item.get("line")
            keys.add(f"{path}:{line}:{code}")
    return keys


def _violation_key(item: Violation) -> str:
    return f"{item.path}:{item.line}:{item.code}"


def _import_violations(
    node: ast.Import | ast.ImportFrom,
    relative: str,
    path: PurePosixPath,
    tier: str | None,
) -> tuple[list[Violation], bool]:
    """Return import violations and whether a runtime import was flagged."""
    violations: list[Violation] = []
    runtime_import_violation = False
    module = _imported_module(node)
    violations.extend(_conftest_import_violations(module, relative, node))
    violations.extend(
        _complete_runtime_fixture_import_violations(module, relative, node)
    )
    for alias in node.names:
        imported = (
            alias.name.split(".", 1)[0]
            if isinstance(node, ast.ImportFrom)
            else alias.name
        )
        imported_full = alias.name
        if (
            (
                isinstance(node, ast.ImportFrom)
                and module == "jacobian.runtime"
                and imported == "create_runtime"
            )
            or (isinstance(node, ast.Import) and alias.name == "jacobian.runtime")
        ) and not _runtime_allowed(path, tier):
            violations.append(
                Violation(
                    relative,
                    "runtime-usage",
                    "create_runtime is reserved for composition, lifecycle boundaries, and end-to-end tests",
                    node.lineno,
                    node.col_offset,
                )
            )
            runtime_import_violation = True
        portfolio_import = _is_portfolio_import(module, imported_full)
        if isinstance(node, ast.Import) and module in _BUILTIN_PORTFOLIO_MODULES:
            portfolio_import = True
        if tier in {"unit", "component", "domain"} and portfolio_import:
            violations.append(
                Violation(
                    relative,
                    "builtin-portfolio",
                    "built-in portfolio installation is not allowed in lower-tier tests",
                    node.lineno,
                    node.col_offset,
                )
            )
        if _provider_module(
            module if isinstance(node, ast.ImportFrom) else imported_full
        ) and not _provider_allowed(path, tier):
            violations.append(
                Violation(
                    relative,
                    "provider-import",
                    "provider implementation imports belong in boundary or focused provider-component tests",
                    node.lineno,
                    node.col_offset,
                )
            )
        sqlite_import = module == "sqlite3" or imported_full == "sqlite3"
        sqlite_store_import = (
            module == "jacobian.storage.repository" and imported == "ArtifactRepository"
        ) or imported_full == "jacobian.storage.repository"
        if tier == "unit" and (sqlite_import or sqlite_store_import):
            violations.append(
                Violation(
                    relative,
                    "sqlite-unit",
                    "SQLite access is not allowed in unit tests",
                    node.lineno,
                    node.col_offset,
                )
            )
        process_import = module == "subprocess" or imported_full == "subprocess"
        if tier == "unit" and process_import:
            violations.append(
                Violation(
                    relative,
                    "process-unit",
                    "subprocess access is not allowed in unit tests",
                    node.lineno,
                    node.col_offset,
                )
            )
    return violations, runtime_import_violation


def _call_violations(
    node: ast.Call,
    relative: str,
    path: PurePosixPath,
    tier: str | None,
    runtime_import_violation: bool,
) -> list[Violation]:
    target = node.func
    called_name = (
        target.id
        if isinstance(target, ast.Name)
        else target.attr
        if isinstance(target, ast.Attribute)
        else ""
    )
    violations: list[Violation] = []
    if (
        called_name == "create_runtime"
        and not _runtime_allowed(path, tier)
        and not runtime_import_violation
    ):
        violations.append(
            Violation(
                relative,
                "runtime-usage",
                "create_runtime is reserved for composition, lifecycle boundaries, and end-to-end tests",
                node.lineno,
                node.col_offset,
            )
        )
    if (
        tier in {"unit", "component", "domain"}
        and called_name in _BUILTIN_PORTFOLIO_NAMES
    ):
        violations.append(
            Violation(
                relative,
                "builtin-portfolio",
                "built-in portfolio installation is not allowed in lower-tier tests",
                node.lineno,
                node.col_offset,
            )
        )
    return violations


def _conftest_fixture_violations(tree: ast.AST, relative: str) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _is_pytest_fixture(node)
            and _fixture_is_high_cost(node)
        ):
            violations.append(
                Violation(
                    relative,
                    "root-high-cost-fixture",
                    f"high-cost fixture '{node.name}' must live in its owning tier",
                    node.lineno,
                    node.col_offset,
                )
            )
    return violations


def _non_root_pytest_plugins_violations(
    tree: ast.AST, relative: str
) -> list[Violation]:
    """Reject non-root ``pytest_plugins`` (deprecated and session-global)."""

    if relative == "tests/conftest.py" or not relative.endswith("/conftest.py"):
        return []
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "pytest_plugins":
                violations.append(
                    Violation(
                        relative,
                        "non-root-pytest-plugins",
                        "declare complete-runtime fixtures in the owning conftest "
                        "instead of non-root pytest_plugins",
                        node.lineno,
                        node.col_offset,
                    )
                )
    return violations


def _exact_domain_install_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _exact_domain_install_violations(tree: ast.AST, relative: str) -> list[Violation]:
    """Ordinary tests must use open_exact_domain_services, not the low-level recipe."""

    if relative in _EXACT_DOMAIN_INSTALL_ALLOWLIST:
        return []
    if not relative.startswith("tests/"):
        return []
    message = (
        "use tests.support.exact_domain.open_exact_domain_services "
        "instead of copying install_exact_domain_verification"
    )
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "jacobian.exact_domain_checkers"
            and any(
                alias.name == "install_exact_domain_verification"
                for alias in node.names
            )
        ):
            violations.append(
                Violation(
                    relative,
                    "exact-domain-install-recipe",
                    message,
                    node.lineno,
                    node.col_offset,
                )
            )
        elif (
            isinstance(node, ast.Call)
            and _exact_domain_install_name(node.func)
            == "install_exact_domain_verification"
        ):
            violations.append(
                Violation(
                    relative,
                    "exact-domain-install-recipe",
                    message,
                    node.lineno,
                    node.col_offset,
                )
            )
    return violations


def _uses_complete_runtime_fixture(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (*node.args.args, *node.args.kwonlyargs):
                if arg.arg in _COMPLETE_RUNTIME_FIXTURE_NAMES:
                    return True
        elif isinstance(node, ast.Name) and node.id in _COMPLETE_RUNTIME_FIXTURE_NAMES:
            return True
    return False


def _composition_admission_value(tree: ast.AST) -> tuple[str | None, int | None]:
    if not isinstance(tree, ast.Module):
        return None, None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "COMPOSITION_ADMISSION":
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value, node.lineno
        return None, node.lineno
    return None, None


def _composition_admission_violations(tree: ast.AST, relative: str) -> list[Violation]:
    """Composition modules that hydrate a complete runtime must declare admission."""

    path = PurePosixPath(relative)
    if not _is_under(path, _COMPOSITION_PREFIX):
        return []
    if path.name == "conftest.py" or not path.name.startswith("test_"):
        return []
    if not _uses_complete_runtime_fixture(tree):
        return []
    value, line = _composition_admission_value(tree)
    if value is None:
        return [
            Violation(
                relative,
                "composition-admission-missing",
                (
                    "declare COMPOSITION_ADMISSION as one of "
                    f"{sorted(_COMPOSITION_ADMISSION_CATEGORIES)} when using "
                    "complete-runtime fixtures"
                ),
                line or 1,
            )
        ]
    if value not in _COMPOSITION_ADMISSION_CATEGORIES:
        return [
            Violation(
                relative,
                "composition-admission-invalid",
                (
                    f"COMPOSITION_ADMISSION={value!r} is not one of "
                    f"{sorted(_COMPOSITION_ADMISSION_CATEGORIES)}"
                ),
                line,
            )
        ]
    return []


def _file_violations(
    file_path: Path,
    project_root: Path,
) -> list[Violation]:
    relative = file_path.relative_to(project_root).as_posix()
    path = PurePosixPath(relative)
    tier = _tier(path)
    violations: list[Violation] = []
    if path.name.startswith("test_"):
        claimed = owners(relative)
        if len(claimed) != 1:
            detail = (
                "no directory lane claims this test file"
                if not claimed
                else (
                    "test file belongs to multiple directory lanes: "
                    + ", ".join(claimed)
                )
            )
            violations.append(Violation(relative, "lane-ownership", detail))
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError) as exc:
        violations.append(
            Violation(relative, "parse-error", f"cannot parse test file: {exc}")
        )
        return violations
    # An imported ``create_runtime`` and its call describe one policy
    # violation.  Record import-level diagnostics so the call walk does
    # not emit a duplicate for the same file; direct qualified calls with
    # no import remain covered below.
    runtime_import_violation = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_violations, flagged = _import_violations(node, relative, path, tier)
            violations.extend(import_violations)
            if flagged:
                runtime_import_violation = True
        if isinstance(node, ast.Call):
            violations.extend(
                _call_violations(node, relative, path, tier, runtime_import_violation)
            )
    if relative == "tests/conftest.py":
        violations.extend(_conftest_fixture_violations(tree, relative))
    violations.extend(_non_root_pytest_plugins_violations(tree, relative))
    violations.extend(_exact_domain_install_violations(tree, relative))
    violations.extend(_composition_admission_violations(tree, relative))
    return violations


def check_test_architecture(
    root: Path | str,
    *,
    mode: str = "strict",
    baseline: Path | Iterable[str] | Mapping[str, Any] | None = None,
) -> ArchitectureReport:
    """Check test imports and ownership boundaries below ``root``.

    ``mode="strict"`` reports every violation.  ``mode="ratchet"`` still
    reports all observations but fails only for observations absent from the
    supplied baseline.  Lane ownership is the test directory layout.
    """

    project_root = Path(root).resolve()
    if mode not in {"strict", "ratchet"}:
        raise ValueError("mode must be 'strict' or 'ratchet'")

    violations: list[Violation] = []
    test_root = project_root / "tests"
    files = (
        sorted(path for path in test_root.rglob("*.py") if path.name != "__init__.py")
        if test_root.is_dir()
        else []
    )
    for file_path in files:
        violations.extend(_file_violations(file_path, project_root))

    all_violations = tuple(
        sorted(violations, key=lambda item: (item.path, item.line or 0, item.code))
    )
    new_violations: tuple[Violation, ...] | None = None
    if mode == "ratchet":
        known = _baseline_keys(baseline)
        new_violations = tuple(
            item
            for item in all_violations
            if _violation_key(item) not in known and str(item) not in known
        )
    return ArchitectureReport(
        project_root, all_violations, len(files), new_violations, mode
    )


def assert_test_architecture(
    root: Path | str,
    *,
    mode: str = "strict",
    baseline: Path | Iterable[str] | Mapping[str, Any] | None = None,
) -> ArchitectureReport:
    report = check_test_architecture(
        root, mode=mode, baseline=baseline
    )
    if report.failed:
        raise ArchitecturePolicyError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("strict", "ratchet"), default="strict")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    report = check_test_architecture(
        args.root, mode=args.mode, baseline=args.baseline
    )
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
