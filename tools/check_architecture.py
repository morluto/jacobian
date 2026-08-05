"""Static architecture enforcement for product source boundaries.

This checker is an AST/filesystem tool that does not import the Jacobian
runtime.  It enforces six PR10 invariants:

1. **subprocess-confined**: direct ``subprocess`` usage and ``os.execvpe``/
   ``os.execvp`` are allowed only in ``bounded_process.py``,
   ``command_runner.py``, and explicit e2e/process-boundary test fixture
   files listed by exact path.  Product code and mathematical checkers must
   route through the bounded process gateway.

2. **bounded-process-gateway**: ``run_bounded_process`` imports/calls are
   allowed only in ``src/jacobian/process_policy.py`` (the product gateway)
   and ``src/jacobian/bounded_process.py`` (the engine itself).  All other
   product callers must use ``process_policy``.

3. **shutil-which-resolver**: ``shutil.which`` is allowed only in bootstrap
   resolver modules that discover operator-installed executables at startup.

4. **environ-spreading**: product code must not spread the full ambient
   environment (``dict(os.environ)``, ``os.environ.copy()``, ``**os.environ``)
   into child-process calls.  Selective ``os.environ.get`` access is fine.

5. **public-contract-drift**: every canonical agent-workflow-v1 task must
   have a ``public_contract.json`` and its projection must match the rendered
   ``submission_schema.json`` and ``instruction.md``.  A missing contract is
   a violation, not a skip.

6. **unsupported-surface**: removed experimental memory/search identifiers must
   not appear in supported product source, tests, schemas, catalog, or docs.

The checker excludes ``wt-438/`` and generated directories from all scans.
``CHANGELOG.md`` is excluded from the unsupported-surface text scan as
genuinely historical record.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Path allowlists — deliberately narrow, exact paths only
# ---------------------------------------------------------------------------

# Subprocess APIs: only the engine, the tooling runner, and exact test
# fixture files where direct subprocess is genuinely the test mechanism.
_SUBPROCESS_ALLOWED_EXACT: frozenset[PurePosixPath] = frozenset(
    {
        # The bounded process engine itself.
        PurePosixPath("src/jacobian/bounded_process.py"),
        # The tooling command runner.
        PurePosixPath("benchmarks/tooling/command_runner.py"),
        # --- Explicit test fixtures where subprocess is the test mechanism ---
        # E2e workflow scenarios spawn CLI processes.
        PurePosixPath("tests/e2e/verified_results/test_reference_runtime.py"),
        # Process boundary tests directly exercise subprocess seams.
        PurePosixPath("tests/boundary/process/test_bounded_process.py"),
        PurePosixPath("tests/boundary/process/test_process_policy.py"),
        PurePosixPath("tests/boundary/process/test_rational_lp_worker_protocol.py"),
        PurePosixPath("tests/boundary/process/test_worker_error_protocol.py"),
        PurePosixPath("tests/boundary/process/public_api/test_import_isolation.py"),
        # Tooling boundary tests invoke CI scripts and installers as subprocesses.
        PurePosixPath("tests/boundary/process/tooling/ci.py"),
        PurePosixPath("tests/boundary/process/tooling/test_ci_ownership_manifest.py"),
        PurePosixPath("tests/boundary/process/tooling/test_deploy_installer.py"),
        PurePosixPath("tests/boundary/process/tooling/test_local_validation_tools.py"),
        PurePosixPath("tests/boundary/process/tooling/test_source_agent_bootstrap.py"),
        # MCP transport boundary tests spawn server processes.
        PurePosixPath("tests/boundary/mcp/test_mcp_operations.py"),
        PurePosixPath("tests/boundary/mcp/test_remote_mcp_auth.py"),
        # Provider startup boundary tests spawn provider executables.
        PurePosixPath(
            "tests/boundary/providers/external_sat/startup/"
            "test_sat_unsat_proof_verification.py"
        ),
        PurePosixPath(
            "tests/boundary/providers/external_sat/startup/"
            "test_smt_unsat_proof_verification.py"
        ),
        PurePosixPath(
            "tests/boundary/providers/flint/startup/test_optional_provider_startup.py"
        ),
        PurePosixPath(
            "tests/boundary/providers/lean/test_lean_statement_capabilities.py"
        ),
        # Storage recovery/transaction boundary tests crash and recover processes.
        PurePosixPath("tests/boundary/storage/recovery/test_enumeration_recovery.py"),
        PurePosixPath(
            "tests/boundary/storage/transactions/test_process_crash_recovery.py"
        ),
        PurePosixPath("tests/boundary/storage/transactions/test_quota_accounting.py"),
        PurePosixPath(
            "tests/boundary/storage/transactions/test_state_database_migrations.py"
        ),
        # Checker component tests spawn checker processes.
        PurePosixPath("tests/component/checkers/test_exact_domain_checker_attacks.py"),
        PurePosixPath("tests/component/checkers/test_lean4_checker.py"),
        # Lean provider component tests spawn lean checker.
        PurePosixPath("tests/component/providers/lean/test_lean_checker_errors.py"),
        # Matrix provider component tests spawn matrix executables.
        PurePosixPath("tests/component/providers/matrix/test_matrix_capabilities.py"),
        # Real analysis domain test uses external analysis executables.
        PurePosixPath("tests/domain/analysis/test_real_analysis.py"),
        # SAT assignment composition test invokes the SAT checker directly.
        PurePosixPath("tests/composition/runtime/test_sat_assignment_verification.py"),
        # Untrusted plugin entrypoints manage their own process lifecycle.
        PurePosixPath("tests/support/plugin_entrypoints.py"),
        # Architecture policy test uses subprocess in a synthetic import probe.
        PurePosixPath("tests/unit/tooling/test_architecture_policy.py"),
        # This checker's own test file uses subprocess in synthetic probes.
        PurePosixPath("tests/unit/tooling/test_architecture_enforcement.py"),
        # Repository-command integration tests deliberately invoke CLI entrypoints.
        PurePosixPath(
            "benchmarks/validation/agent_workflow_v1/"
            "test_multiplicative_grid_extremum.py"
        ),
        PurePosixPath("benchmarks/validation/test_benchmark_plan_validation.py"),
        PurePosixPath("benchmarks/validation/test_benchmark_planner.py"),
    }
)

# run_bounded_process gateway: only the engine and the product gateway.
_RUN_BOUNDED_PROCESS_ALLOWED: frozenset[PurePosixPath] = frozenset(
    {
        PurePosixPath("src/jacobian/bounded_process.py"),
        PurePosixPath("src/jacobian/process_policy.py"),
    }
)

# shutil.which: only bootstrap resolver modules and test skip-condition checks.
_SHUTIL_WHICH_ALLOWED: frozenset[PurePosixPath] = frozenset(
    {
        PurePosixPath("src/jacobian/process_policy.py"),
        PurePosixPath("src/jacobian/providers/singular_runtime.py"),
        PurePosixPath("src/jacobian/providers/external_solver_runtime.py"),
        PurePosixPath("src/jacobian/provider_measurements.py"),
        PurePosixPath("src/jacobian/lean_frontend/repl.py"),
        PurePosixPath("src/jacobian/lean_frontend/exploration.py"),
        PurePosixPath("src/jacobian_checkers/lean4.py"),
        PurePosixPath("benchmarks/tooling/command_runner.py"),
        PurePosixPath("tools/source_agent_doctor.py"),
        # Test skip-condition checks for optional operator tools.
        PurePosixPath("tests/boundary/process/test_bounded_process.py"),
        PurePosixPath("tests/boundary/process/tooling/test_ci_ownership_manifest.py"),
        PurePosixPath("tests/boundary/process/tooling/test_deploy_installer.py"),
        PurePosixPath("tests/boundary/process/tooling/test_source_agent_bootstrap.py"),
    }
)

# os.environ spreading: product source only.
_ENVIRON_SPREAD_ROOTS = (PurePosixPath("src"),)

# Public-contract checks apply only to agent-workflow-v1 tasks.
_DATASET_PREFIX = PurePosixPath("benchmarks/datasets/agent-workflow-v1")

# Unsupported surfaces: scan supported src, tests, schemas, catalog, and docs.
# Tokens are built from fragments so the checker source does not self-trigger.
_RESEARCH = "Research"
_MEMORY = "Memory"
_EPISODE = "Episode"
_KNOWLEDGE = "knowledge"
_SEARCH = "search"
_UNDERSCORE = "_"
_DOT = "."
_URI = "uri"
_RECORDS = "records"

# Python identifier-level tokens (case-sensitive — these are exact symbols).
_UNSUPPORTED_SURFACE_SYMBOLS: frozenset[str] = frozenset(
    {
        f"{_RESEARCH}{_MEMORY}",
        f"{_RESEARCH}{_EPISODE}",
        f"{_RESEARCH.lower()}{_UNDERSCORE}{_MEMORY.lower()}",
        f"{_RESEARCH.lower()}{_UNDERSCORE}{_EPISODE.lower()}",
        f"{_KNOWLEDGE}{_DOT}{_SEARCH}",
        f"{_KNOWLEDGE}{_UNDERSCORE}{_SEARCH}",
        f"{_EPISODE.lower()}{_UNDERSCORE}{_URI}",
        f"{_RECORDS}{_UNDERSCORE}{_EPISODE.lower()}",
    }
)

# Module imports that are unsupported surfaces.
_UNSUPPORTED_SURFACE_MODULES: frozenset[str] = frozenset(
    {f"jacobian{_DOT}{_MEMORY.lower()}"}
)

# Text-scan tokens (case-insensitive prose phrases and exact symbols).
# Built from fragments to avoid self-triggering in the checker source.
_UNSUPPORTED_SURFACE_TEXT_TOKENS: tuple[str, ...] = (
    f"{_RESEARCH}{_MEMORY}",
    f"{_RESEARCH}{_EPISODE}",
    f"{_RESEARCH.lower()}{_UNDERSCORE}{_MEMORY.lower()}",
    f"{_RESEARCH.lower()}{_UNDERSCORE}{_EPISODE.lower()}",
    f"{_KNOWLEDGE}{_DOT}{_SEARCH}",
    f"{_KNOWLEDGE}{_UNDERSCORE}{_SEARCH}",
    f"{_EPISODE.lower()}{_UNDERSCORE}{_URI}",
    f"{_RECORDS}{_UNDERSCORE}{_EPISODE.lower()}",
    # Prose variants are case-insensitive.
    f"{_RESEARCH} {_MEMORY.lower()}",
    f"{_RESEARCH} {_EPISODE.lower()}",
)

# Text file extensions scanned for unsupported surfaces.
_TEXT_EXTENSIONS = frozenset(
    {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".rst"}
)

# Directories excluded from all scans.
_EXCLUDED_DIRS = frozenset(
    {
        "wt-438",
        ".git",
        "__pycache__",
        ".venv",
        "dist",
        ".diagnostics",
        "node_modules",
        # Lean toolchain packages (third-party Mathlib build artifacts).
        ".lake",
    }
)

# Files excluded from the unsupported-surface text scan.
# CHANGELOG.md is genuinely historical record.
# The checker and its test file construct tokens from fragments but still
# contain prose mentions, so they are excluded from the text scan.
_UNSUPPORTED_SURFACE_TEXT_EXCLUDED: frozenset[PurePosixPath] = frozenset(
    {
        PurePosixPath("CHANGELOG.md"),
        PurePosixPath("tools/check_architecture.py"),
        PurePosixPath("tests/unit/tooling/test_architecture_enforcement.py"),
    }
)

# os.execvpe / os.execvp detection (process replacement without bounds).
_EXEC_FUNCTIONS = frozenset({"execvpe", "execvp", "execlp", "execve", "execv"})

# Embedded subprocess API patterns in string constants — catches worker-source
# strings that bypass the import gate by embedding subprocess calls in code
# passed to ``exec``/``python -c``/``eval``.  Patterns are built from fragments
# so the checker source does not self-trigger.
_SUB = "sub"
_PROC = "proc" + "ess"
_SUBPROCESS = _SUB + _PROC
_EMBEDDED_SUBPROCESS_PATTERNS: tuple[str, ...] = (
    f"{_SUBPROCESS}.run(",
    f"{_SUBPROCESS}.Popen(",
    f"{_SUBPROCESS}.call(",
    f"{_SUBPROCESS}.check_call(",
    f"{_SUBPROCESS}.check_output(",
    f"from {_SUBPROCESS} import ",
    f"import {_SUBPROCESS}",
)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One architecture policy violation."""

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
    """Result of checking the repository tree."""

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
        if not self.violations:
            return f"architecture: OK ({self.files_scanned} files checked)"
        lines = [
            f"architecture: {len(self.violations)} violation(s) "
            f"({self.files_scanned} files checked)"
        ]
        lines.extend(str(item) for item in self.violations)
        return "\n".join(lines)


class ArchitecturePolicyError(RuntimeError):
    """Raised when architecture enforcement fails."""

    def __init__(self, report: ArchitectureReport) -> None:
        self.report = report
        super().__init__(report.render())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIRS for part in path.parts)


def _relative(root: Path, path: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(root).as_posix())


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _imported_module(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if node.level:
        return ""
    return node.module or ""


def _imported_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    return [alias.name for alias in node.names]


# ---------------------------------------------------------------------------
# Check 1: subprocess + os.exec* confinement
# ---------------------------------------------------------------------------


def _subprocess_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if relative in _SUBPROCESS_ALLOWED_EXACT:
        return ()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = _imported_module(node)
            names = _imported_names(node)
            if module == "subprocess" or "subprocess" in names:
                violations.append(
                    Violation(
                        str(relative),
                        "subprocess-confined",
                        "direct subprocess is only allowed in bounded_process.py, "
                        "command_runner.py, and explicit test fixtures",
                        node.lineno,
                    )
                )
        # os.execvpe / os.execvp / os.execve / os.execv process replacement
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr in _EXEC_FUNCTIONS
        ):
            violations.append(
                Violation(
                    str(relative),
                    "subprocess-confined",
                    f"os.{node.func.attr} is unbounded process replacement; "
                    "use the bounded tooling command runner",
                    node.lineno,
                )
            )
        # Embedded subprocess API in string constants — catches worker-source
        # strings that bypass the import gate by embedding subprocess calls in
        # code passed to exec/python -c/eval.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _EMBEDDED_SUBPROCESS_PATTERNS:
                if pattern in node.value:
                    violations.append(
                        Violation(
                            str(relative),
                            "subprocess-confined",
                            f"string constant embeds '{pattern.rstrip('(')}' — "
                            "worker source must not bypass the subprocess gate",
                            node.lineno,
                        )
                    )
                    break  # One violation per string is enough.
    return tuple(violations)


# ---------------------------------------------------------------------------
# Check 2: run_bounded_process gateway confinement
# ---------------------------------------------------------------------------


def _bounded_process_import_violations(
    relative: PurePosixPath, node: ast.AST
) -> Iterator[Violation]:
    """Yield violations for direct ``run_bounded_process`` imports."""

    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return
    module = _imported_module(node)
    names = _imported_names(node)
    if module == "jacobian.bounded_process" and "run_bounded_process" in names:
        yield Violation(
            str(relative),
            "bounded-process-gateway",
            "run_bounded_process must be called only from "
            "process_policy.py (the product gateway)",
            node.lineno,
        )
    if isinstance(node, ast.Import) and module == "jacobian.bounded_process":
        yield Violation(
            str(relative),
            "bounded-process-gateway",
            "jacobian.bounded_process must be imported only from "
            "process_policy.py (the product gateway)",
            node.lineno,
        )


def _bounded_process_call_violations(
    relative: PurePosixPath, node: ast.AST
) -> Iterator[Violation]:
    """Yield violations for direct ``run_bounded_process(...)`` calls."""

    if not isinstance(node, ast.Call):
        return
    target = node.func
    called_name = ""
    if isinstance(target, ast.Name):
        called_name = target.id
    elif isinstance(target, ast.Attribute):
        called_name = target.attr
    if called_name == "run_bounded_process":
        yield Violation(
            str(relative),
            "bounded-process-gateway",
            "run_bounded_process must be called only from "
            "process_policy.py (the product gateway)",
            node.lineno,
        )


def _run_bounded_process_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    # Only gate product source — tests legitimately monkeypatch run_bounded_process.
    if not _is_under(relative, PurePosixPath("src")):
        return ()
    if relative in _RUN_BOUNDED_PROCESS_ALLOWED:
        return ()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        violations.extend(_bounded_process_import_violations(relative, node))
        violations.extend(_bounded_process_call_violations(relative, node))
    return tuple(violations)


# ---------------------------------------------------------------------------
# Check 3: shutil.which resolver confinement
# ---------------------------------------------------------------------------


def _shutil_which_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if relative in _SHUTIL_WHICH_ALLOWED:
        return ()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            isinstance(node.value, ast.Name)
            and node.value.id == "shutil"
            and node.attr == "which"
        ):
            violations.append(
                Violation(
                    str(relative),
                    "shutil-which-resolver",
                    "shutil.which is only allowed in bootstrap resolver modules",
                    node.lineno,
                )
            )
    return tuple(violations)


# ---------------------------------------------------------------------------
# Check 4: os.environ spreading
# ---------------------------------------------------------------------------


def _is_dict_os_environ_call(node: ast.AST) -> bool:
    """Match ``dict(os.environ)``."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "os"
        and node.args[0].attr == "environ"
    )


def _is_os_environ_copy_call(node: ast.AST) -> bool:
    """Match ``os.environ.copy()``."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
        and node.func.value.attr == "environ"
    )


def _has_star_os_environ_in_dict_call(node: ast.AST) -> bool:
    """Match ``dict(**os.environ)``."""

    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return False
    for keyword in node.keywords:
        if (
            keyword.arg is None  # ** spread
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "os"
            and keyword.value.attr == "environ"
        ):
            return True
    return False


def _has_star_os_environ_in_dict_literal(node: ast.AST) -> bool:
    """Match ``{**os.environ, ...}``."""

    if not isinstance(node, ast.Dict):
        return False
    for key, value in zip(node.keys, node.values, strict=False):
        if (
            key is None  # ** spread entry
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
            and value.attr == "environ"
        ):
            return True
    return False


def _environ_spread_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    if not any(_is_under(relative, root) for root in _ENVIRON_SPREAD_ROOTS):
        return ()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if _is_dict_os_environ_call(node):
            violations.append(
                Violation(
                    str(relative),
                    "environ-spreading",
                    "spreading the full ambient environment is forbidden; "
                    "use selective os.environ.get or an allowlisted environment",
                    node.lineno,
                )
            )
        elif _is_os_environ_copy_call(node):
            violations.append(
                Violation(
                    str(relative),
                    "environ-spreading",
                    "os.environ.copy() spreads the full ambient environment; "
                    "use selective os.environ.get or an allowlisted environment",
                    node.lineno,
                )
            )
        elif _has_star_os_environ_in_dict_call(node):
            violations.append(
                Violation(
                    str(relative),
                    "environ-spreading",
                    "**os.environ spreads the full ambient environment; "
                    "use selective os.environ.get or an allowlisted environment",
                    node.lineno,
                )
            )
        if _has_star_os_environ_in_dict_literal(node):
            violations.append(
                Violation(
                    str(relative),
                    "environ-spreading",
                    "**os.environ in a dict literal spreads the full "
                    "ambient environment; use selective os.environ.get "
                    "or an allowlisted environment",
                    node.lineno,
                )
            )
    return tuple(violations)


# ---------------------------------------------------------------------------
# Check 5: public-contract drift (missing contract is a violation)
# ---------------------------------------------------------------------------


def _public_contract_drift_violations(root: Path) -> tuple[Violation, ...]:
    from benchmarks.tooling.public_contract import check as check_public_contract

    dataset_root = root / _DATASET_PREFIX
    if not dataset_root.is_dir():
        return ()
    violations: list[Violation] = []
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir():
            continue
        # Only canonical task directories (with task.toml) require a public contract.
        if not (task_dir / "task.toml").is_file():
            continue
        contract_path = task_dir / "tests" / "public_contract.json"
        contract_rel = str(PurePosixPath(contract_path.relative_to(root).as_posix()))
        if not contract_path.is_file():
            violations.append(
                Violation(
                    contract_rel,
                    "public-contract-drift",
                    "required public_contract.json is missing",
                )
            )
            continue
        try:
            drifts = check_public_contract(contract_path, task_dir)
        except Exception as exc:
            violations.append(
                Violation(
                    contract_rel, "public-contract-drift", f"contract error: {exc}"
                )
            )
            continue
        for drift in drifts:
            violations.append(Violation(contract_rel, "public-contract-drift", drift))
    return tuple(violations)


# ---------------------------------------------------------------------------
# Check 6: unsupported surfaces (Python AST + text scan)
# ---------------------------------------------------------------------------


# Files excluded from the unsupported-surface AST scan (enforcement machinery
# that legitimately references the tokens in its own allowlists).
_UNSUPPORTED_SURFACE_AST_EXCLUDED: frozenset[PurePosixPath] = frozenset(
    {
        PurePosixPath("tools/check_architecture.py"),
        PurePosixPath("tests/unit/tooling/test_architecture_enforcement.py"),
    }
)


def _unsupported_surface_ast_violations(
    relative: PurePosixPath, tree: ast.AST
) -> tuple[Violation, ...]:
    """Detect unsupported surfaces in Python source via AST."""
    if relative in _UNSUPPORTED_SURFACE_AST_EXCLUDED:
        return ()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = _imported_module(node)
            names = _imported_names(node)
            if module in _UNSUPPORTED_SURFACE_MODULES:
                violations.append(
                    Violation(
                        str(relative),
                        "unsupported-surface",
                        f"{module} is an experimental surface not supported "
                        "in shipped product source",
                        node.lineno,
                    )
                )
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in _UNSUPPORTED_SURFACE_SYMBOLS:
                    violations.append(
                        Violation(
                            str(relative),
                            "unsupported-surface",
                            f"{name} is an experimental surface not supported "
                            "in shipped product source",
                            node.lineno,
                        )
                    )
        # String literal references to removed surfaces.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _UNSUPPORTED_SURFACE_SYMBOLS
        ):
            violations.append(
                Violation(
                    str(relative),
                    "unsupported-surface",
                    f"'{node.value}' is an experimental surface not supported "
                    "in shipped product source",
                    node.lineno,
                )
            )
    return tuple(violations)


def _unsupported_surface_text_violations(
    root: Path, relative: PurePosixPath, path: Path
) -> tuple[Violation, ...]:
    """Detect unsupported surfaces in text files via case-insensitive string search."""
    if relative in _UNSUPPORTED_SURFACE_TEXT_EXCLUDED:
        return ()
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    violations: list[Violation] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        lower = line.lower()
        for token in _UNSUPPORTED_SURFACE_TEXT_TOKENS:
            if token.lower() in lower:
                violations.append(
                    Violation(
                        str(relative),
                        "unsupported-surface",
                        f"'{token}' is an experimental surface not supported "
                        "in shipped product source",
                        line_no,
                    )
                )
    return tuple(violations)


# ---------------------------------------------------------------------------
# Scan orchestration
# ---------------------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    """Return all non-excluded Python files under root."""
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if _is_excluded(path):
            continue
        if not path.is_file():
            continue
        files.append(path)
    return files


def _text_files(root: Path) -> list[Path]:
    """Return non-Python text files for unsupported-surface scanning."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if _is_excluded(path):
            continue
        if not path.is_file():
            continue
        ext = path.suffix
        if ext not in _TEXT_EXTENSIONS:
            continue
        files.append(path)
    return files


def _check_python_file(root: Path, path: Path) -> tuple[Violation, ...]:
    relative = _relative(root, path)
    rel_str = str(relative)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_str)
    except (OSError, SyntaxError) as exc:
        return (Violation(rel_str, "parse-error", f"cannot parse file: {exc}"),)

    violations: list[Violation] = []
    violations.extend(_subprocess_violations(relative, tree))
    violations.extend(_run_bounded_process_violations(relative, tree))
    violations.extend(_shutil_which_violations(relative, tree))
    violations.extend(_environ_spread_violations(relative, tree))
    violations.extend(_unsupported_surface_ast_violations(relative, tree))
    return tuple(violations)


def check_architecture(root: Path | str = ROOT) -> ArchitectureReport:
    """Check the repository tree for architecture boundary violations.

    Scans all non-excluded Python files for subprocess confinement,
    run_bounded_process gateway confinement, shutil.which resolver
    confinement, os.environ spreading, and unsupported experimental
    surfaces.  Scans non-Python text files (docs, schemas, catalog) for
    unsupported surfaces.  Additionally checks every agent-workflow-v1
    task public-contract projection for drift (missing contracts are
    violations).
    """
    project_root = Path(root).resolve()
    py_files = _python_files(project_root)
    text_files = _text_files(project_root)

    violations: list[Violation] = []
    for file_path in py_files:
        violations.extend(_check_python_file(project_root, file_path))

    # Unsupported-surface text scan for non-Python text files.
    # Python files are covered by the AST scan above (imports, string
    # constants, names).  The text scan covers docs, schemas, catalog,
    # and other prose/config files.
    for file_path in text_files:
        if file_path.suffix == ".py":
            continue
        relative = _relative(project_root, file_path)
        violations.extend(
            _unsupported_surface_text_violations(project_root, relative, file_path)
        )

    violations.extend(_public_contract_drift_violations(project_root))

    all_violations = tuple(
        sorted(violations, key=lambda item: (item.path, item.line or 0, item.code))
    )
    return ArchitectureReport(project_root, all_violations, len(py_files))


def assert_architecture(root: Path | str = ROOT) -> ArchitectureReport:
    """Check architecture and raise on failure."""
    report = check_architecture(root)
    if report.failed:
        raise ArchitecturePolicyError(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = check_architecture(args.root)
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
