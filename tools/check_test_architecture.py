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
import fnmatch
import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

_TEST_FILE = "test_*.py"
_TOPOLOGY_CANDIDATES = (Path("tests/topology.toml"), Path("topology.toml"))

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
class TopologyManifest:
    """The path globs owned by each semantic test lane."""

    lanes: Mapping[str, tuple[str, ...]]
    path: Path
    lane_tiers: Mapping[str, str] = field(default_factory=dict)

    def owners(self, relative_path: str) -> tuple[str, ...]:
        return tuple(
            lane
            for lane, patterns in self.lanes.items()
            if any(_matches_path(relative_path, pattern) for pattern in patterns)
        )

    def tier_for(self, relative_path: str) -> str | None:
        owners = self.owners(relative_path)
        if len(owners) != 1:
            return None
        return self.lane_tiers.get(owners[0])


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
    topology: TopologyManifest | None = None
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


def _collect_lane_entries(raw_lanes: Any, *, list_mode: bool) -> list[tuple[str, Any]]:
    """Collect ``(name, value)`` lane entries from either TOML shape."""
    entries: list[tuple[str, Any]] = []
    if list_mode:
        for value in raw_lanes:
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                continue
            entries.append((value["name"], value))
    else:
        for name, value in raw_lanes.items():
            entries.append((str(name), value))
    return entries


def _lanes_from_entries(
    entries: list[tuple[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    lanes: dict[str, tuple[str, ...]] = {}
    lane_tiers: dict[str, str] = {}
    for name, value in entries:
        patterns = _patterns_from_lane(value)
        if patterns:
            lanes[name] = patterns
            tier = _tier_from_lane(value)
            if tier:
                lane_tiers[name] = tier
    return lanes, lane_tiers


def load_topology_manifest(path: Path) -> TopologyManifest | None:
    """Read a topology manifest, accepting the two common TOML shapes.

    The target manifest uses ``[lanes.<name>]`` tables.  Supporting an array of
    ``[[lanes]]`` records keeps the checker useful while the manifest is being
    introduced, without adding a second execution or selection mechanism.
    """

    if not path.is_file():
        return None
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read topology manifest {path}: {exc}") from exc

    raw_lanes = raw.get("lanes", {})
    if isinstance(raw_lanes, dict):
        entries = _collect_lane_entries(raw_lanes, list_mode=False)
    elif isinstance(raw_lanes, list):
        entries = _collect_lane_entries(raw_lanes, list_mode=True)
    else:
        entries = []
    lanes, lane_tiers = _lanes_from_entries(entries)

    # A few manifests put lane tables at the top level.  Only consume tables
    # with a path-like field, so unrelated metadata is ignored.
    if not lanes:
        top_entries = [
            (str(name), value) for name, value in raw.items() if isinstance(value, dict)
        ]
        lanes, lane_tiers = _lanes_from_entries(top_entries)
    return TopologyManifest(lanes=lanes, path=path, lane_tiers=lane_tiers)


def _patterns_from_lane(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (dict, list, tuple)):
        return ()
    if isinstance(value, dict):
        candidates = value.get("owned_paths", value.get("paths", value.get("path", ())))
    else:
        candidates = value
    if isinstance(candidates, str):
        candidates = (candidates,)
    if not isinstance(candidates, (list, tuple)):
        return ()
    return tuple(
        str(pattern) for pattern in candidates if isinstance(pattern, str) and pattern
    )


def _tier_from_lane(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("tier"), str):
        return str(value["tier"])
    return None


def _matches_path(relative_path: str, pattern: str) -> bool:
    path = relative_path.replace("\\", "/")
    normalized = pattern.replace("\\", "/")
    if fnmatch.fnmatchcase(path, normalized):
        return True
    if not any(char in normalized for char in "*?["):
        return path == normalized or path.startswith(normalized.rstrip("/") + "/")
    # ``fnmatch`` treats ``**`` as ordinary stars.  Match the useful directory
    # ownership spelling explicitly as well.
    if normalized.endswith("/**"):
        prefix = normalized[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return PurePosixPath(path).match(normalized)


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


def _tier(path: PurePosixPath) -> str | None:
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
    if path in {
        PurePosixPath("tests/support/runtime_templates.py"),
        PurePosixPath("tests/support/runtime_instances.py"),
    }:
        return True
    tier = tier_override or _tier(path)
    if tier in {"composition", "e2e"}:
        return True
    if tier == "boundary":
        # Boundary tests are allowed to own complete construction only when
        # their path names the lifecycle/startup/recovery boundary explicitly.
        return any(part in {"runtime", "startup", "recovery"} for part in path.parts)
    return False


def _provider_allowed(path: PurePosixPath, tier_override: str | None = None) -> bool:
    tier = tier_override or _tier(path)
    if tier == "boundary":
        return True
    if tier == "component":
        return "providers" in path.parts or "provider" in path.stem.lower()
    return False


def _node_location(path: str, node: ast.AST) -> tuple[str, int, int]:
    return path, getattr(node, "lineno", 1), getattr(node, "col_offset", 0)


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
        "authorized_complete_runtime",
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


def _file_violations(
    file_path: Path,
    project_root: Path,
    manifest: TopologyManifest | None,
) -> list[Violation]:
    relative = file_path.relative_to(project_root).as_posix()
    path = PurePosixPath(relative)
    tier = _tier(path)
    if tier is None and manifest is not None:
        tier = manifest.tier_for(relative)
    violations: list[Violation] = []
    if manifest is not None:
        owners = manifest.owners(relative)
        if path.name.startswith("test_") and len(owners) != 1:
            detail = (
                "no topology lane claims this test file"
                if not owners
                else (
                    "test file belongs to multiple topology lanes: " + ", ".join(owners)
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
    return violations


def check_test_architecture(
    root: Path | str,
    *,
    mode: str = "strict",
    baseline: Path | Iterable[str] | Mapping[str, Any] | None = None,
    topology: Path | None = None,
) -> ArchitectureReport:
    """Check test imports and ownership boundaries below ``root``.

    ``mode="strict"`` reports every violation.  ``mode="ratchet"`` still
    reports all observations but fails only for observations absent from the
    supplied baseline.  The default topology path is ``tests/topology.toml``;
    no lane check is invented when that manifest is not present yet.
    """

    project_root = Path(root).resolve()
    if mode not in {"strict", "ratchet"}:
        raise ValueError("mode must be 'strict' or 'ratchet'")
    topology_path = (
        topology.resolve()
        if topology is not None
        else next(
            (
                project_root / candidate
                for candidate in _TOPOLOGY_CANDIDATES
                if (project_root / candidate).is_file()
            ),
            None,
        )
    )
    manifest = (
        load_topology_manifest(topology_path) if topology_path is not None else None
    )

    violations: list[Violation] = []
    test_root = project_root / "tests"
    files = (
        sorted(path for path in test_root.rglob("*.py") if path.name != "__init__.py")
        if test_root.is_dir()
        else []
    )
    for file_path in files:
        violations.extend(_file_violations(file_path, project_root, manifest))

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
        project_root, all_violations, len(files), manifest, new_violations, mode
    )


def assert_test_architecture(
    root: Path | str,
    *,
    mode: str = "strict",
    baseline: Path | Iterable[str] | Mapping[str, Any] | None = None,
    topology: Path | None = None,
) -> ArchitectureReport:
    report = check_test_architecture(
        root, mode=mode, baseline=baseline, topology=topology
    )
    if report.failed:
        raise ArchitecturePolicyError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("strict", "ratchet"), default="strict")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--topology", type=Path)
    args = parser.parse_args(argv)
    report = check_test_architecture(
        args.root, mode=args.mode, baseline=args.baseline, topology=args.topology
    )
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
