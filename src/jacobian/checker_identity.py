"""Measured, per-checker execution identity and source import enforcement."""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import stat
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, cast

from jacobian.contracts.checkers import (
    CheckerManifest,
    CheckerPythonDistribution,
    CheckerPythonRuntime,
    CheckerSandboxPolicy,
    CheckerSourceModule,
)
from jacobian.contracts.operations import ProviderObservation
from jacobian.implementation import split_entrypoint

_CHECKER_WORKER_MODULE = "jacobian.checker_worker"
_JACOBIAN_PACKAGE = "jacobian"
_WORKER_DISTRIBUTIONS = ("pydantic", "pydantic-core", "rfc8785")
_ORIGINAL_IMPORT = builtins.__import__
_ORIGINAL_IMPORT_MODULE = importlib.import_module


class CheckerManifestError(RuntimeError):
    """A checker execution identity cannot be measured or remeasured safely."""


class UndeclaredCheckerImportError(ImportError):
    """A checker tried to import first-party code outside its manifest."""


@dataclass(frozen=True, slots=True)
class _ResolvedModule:
    name: str
    path: Path
    is_package: bool


@dataclass(slots=True)
class _ManifestMeasurementBatch:
    source_modules: dict[
        tuple[tuple[str, ...], frozenset[str]], tuple[CheckerSourceModule, ...]
    ]
    distributions: dict[str, CheckerPythonDistribution]
    package_owners: Mapping[str, list[str]] | None = None
    python_runtime: CheckerPythonRuntime | None = None


_ACTIVE_MEASUREMENT_BATCH: ContextVar[_ManifestMeasurementBatch | None] = ContextVar(
    "jacobian_checker_manifest_measurement_batch", default=None
)


@contextmanager
def batch_checker_manifest_measurement() -> Iterator[None]:
    """Share immutable identity measurements across one installation operation."""

    active = _ACTIVE_MEASUREMENT_BATCH.get()
    if active is not None:
        yield
        return
    token = _ACTIVE_MEASUREMENT_BATCH.set(
        _ManifestMeasurementBatch(source_modules={}, distributions={})
    )
    try:
        yield
    finally:
        _ACTIVE_MEASUREMENT_BATCH.reset(token)


class _DeclaredSourceLoader(importlib.abc.Loader):
    """Load one manifest-bound module from remeasured source only."""

    def __init__(self, module: _ResolvedModule, expected_digest: str) -> None:
        self.module = module
        self.expected_digest = expected_digest

    def create_module(self, spec: Any) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self.module.path.read_bytes()
        if _source_digest(source) != self.expected_digest:
            raise CheckerManifestError(f"checker source changed: {self.module.name}")
        code = compile(source, str(self.module.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _DeclaredSourceFinder(importlib.abc.MetaPathFinder):
    """Reject undeclared first-party imports and reload declared source exactly."""

    def __init__(
        self,
        *,
        declared: dict[str, CheckerSourceModule],
        first_party_packages: frozenset[str],
        allowed_third_party_roots: frozenset[str],
    ) -> None:
        self.declared = declared
        self.first_party_packages = first_party_packages
        self.allowed_third_party_roots = allowed_third_party_roots

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        root = fullname.split(".", 1)[0]
        if root not in self.first_party_packages:
            if (
                root not in self.allowed_third_party_roots
                and root not in sys.stdlib_module_names
                and _import_requester_is_declared(frozenset(self.declared))
            ):
                raise UndeclaredCheckerImportError(
                    "checker import is not declared by its manifest: " + fullname
                )
            return None
        expected = self.declared.get(fullname)
        if expected is None:
            if any(name.startswith(fullname + ".") for name in self.declared):
                # Namespace-package parents contain no executable source. Let
                # Python assemble the namespace while every concrete child
                # remains manifest-bound.
                return None
            raise UndeclaredCheckerImportError(
                f"checker import is not declared by its manifest: {fullname}"
            )
        resolved = _resolve_module(fullname)
        return importlib.util.spec_from_file_location(
            fullname,
            resolved.path,
            loader=_DeclaredSourceLoader(resolved, expected.source_digest),
            submodule_search_locations=(
                [str(resolved.path.parent)] if resolved.is_package else None
            ),
        )


def default_checker_sandbox_policy() -> CheckerSandboxPolicy:
    """Return the sole bounded-process policy used by checker workers."""

    return CheckerSandboxPolicy(
        max_wall_seconds=105,
        max_cpu_seconds=106,
        max_address_space_bytes=16 * 1024 * 1024 * 1024,
        max_stdout_bytes=1024 * 1024,
        max_stderr_bytes=1024 * 1024,
    )


def build_checker_manifest(
    entrypoint: str,
    *,
    provider_runtime: ProviderObservation | None,
    passive_contract_uris: Iterable[str],
    sandbox: CheckerSandboxPolicy | None = None,
) -> CheckerManifest:
    """Measure exactly the first-party code and execution policy a checker uses."""

    extra_modules = (
        ("jacobian.providers.lean_runtime",)
        if provider_runtime is not None
        and provider_runtime.provider == "jacobian.lean4"
        else ()
    )
    entrypoint_module, _ = split_entrypoint(entrypoint)
    first_party_packages = frozenset(
        {_JACOBIAN_PACKAGE, entrypoint_module.split(".", 1)[0]}
    )
    checker_source_modules = _batched_source_modules(
        (entrypoint_module,),
        first_party_packages=first_party_packages,
    )
    worker_source_modules = _batched_source_modules(
        (_CHECKER_WORKER_MODULE, *extra_modules),
        first_party_packages=first_party_packages,
    )
    return CheckerManifest(
        entrypoint=entrypoint,
        checker_source_modules=checker_source_modules,
        worker_source_modules=worker_source_modules,
        python_runtime=_batched_python_runtime(),
        python_distributions=_collect_python_distributions(
            (*checker_source_modules, *worker_source_modules),
            first_party_packages=first_party_packages,
        ),
        provider_runtime=provider_runtime,
        passive_contract_uris=tuple(sorted(set(passive_contract_uris))),
        sandbox=default_checker_sandbox_policy() if sandbox is None else sandbox,
    )


def checker_implementation_digest(manifest: CheckerManifest) -> str:
    """Return the identity digest of one versioned checker execution manifest."""

    return manifest.implementation_digest()


def require_manifest_unchanged(manifest: CheckerManifest) -> str:
    """Rebuild a manifest and reject every source, runtime, or policy change."""

    measured = build_checker_manifest(
        manifest.entrypoint,
        provider_runtime=manifest.provider_runtime,
        passive_contract_uris=manifest.passive_contract_uris,
        sandbox=manifest.sandbox,
    )
    if measured != manifest:
        raise CheckerManifestError(
            "checker manifest changed after authorization; authorize the current "
            "checker version"
        )
    return checker_implementation_digest(measured)


def require_manifest_material_unchanged(manifest: CheckerManifest) -> str:
    """Reject changes to every execution artifact already bound by a manifest.

    The worker performs full dependency discovery before loading the checker.  Its
    post-execution check only needs to remeasure that closed set: the import guard
    prevents undeclared code from entering the process, while direct remeasurement
    avoids rebuilding and reparsing the complete import graph a second time.
    """

    for expected in _manifest_source_modules(manifest):
        resolved = _resolve_module(expected.module)
        measured = _source_digest(resolved.path.read_bytes())
        if measured != expected.source_digest:
            raise CheckerManifestError(f"checker source changed: {expected.module}")
    if _python_runtime() != manifest.python_runtime:
        raise CheckerManifestError("checker Python runtime changed")
    measured_distributions = tuple(
        _measure_python_distribution(item.distribution)
        for item in manifest.python_distributions
    )
    if measured_distributions != manifest.python_distributions:
        raise CheckerManifestError("checker Python distribution changed")
    return checker_implementation_digest(manifest)


def install_manifest_import_guard(manifest: CheckerManifest) -> None:
    """Reload the checker through the manifest and reject undeclared code imports."""

    module_name, _ = split_entrypoint(manifest.entrypoint)
    first_party_packages = frozenset({_JACOBIAN_PACKAGE, module_name.split(".", 1)[0]})
    declared = {source.module: source for source in _manifest_source_modules(manifest)}
    declared_names = frozenset(declared)
    allowed_third_party_roots = _manifest_distribution_roots(manifest)
    _purge_first_party_modules(first_party_packages)
    sys.meta_path.insert(
        0,
        _DeclaredSourceFinder(
            declared=declared,
            first_party_packages=first_party_packages,
            allowed_third_party_roots=allowed_third_party_roots,
        ),
    )
    builtins.__import__ = _guarded_import(
        declared_names,
        first_party_packages,
        allowed_third_party_roots,
    )
    importlib.import_module = cast(
        Any,
        _guarded_import_module(
            declared_names,
            first_party_packages,
            allowed_third_party_roots,
        ),
    )


def _purge_first_party_modules(first_party_packages: frozenset[str]) -> None:
    """Prevent preloaded modules from bypassing the declared-source finder."""

    for name in tuple(sys.modules):
        if name == "__main__":
            continue
        if name.split(".", 1)[0] in first_party_packages:
            del sys.modules[name]


def _collect_source_modules(
    roots: tuple[str, ...],
    *,
    first_party_packages: frozenset[str],
) -> tuple[CheckerSourceModule, ...]:
    pending = list(roots)
    resolved: dict[str, _ResolvedModule] = {}
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        chain = _resolve_module_chain(name)
        for item in chain:
            if item.name in resolved:
                continue
            resolved[item.name] = item
            pending.extend(_first_party_imports(item, first_party_packages))
    return tuple(
        CheckerSourceModule(
            module=name, source_digest=_source_digest(item.path.read_bytes())
        )
        for name, item in sorted(resolved.items())
    )


def _batched_source_modules(
    roots: tuple[str, ...],
    *,
    first_party_packages: frozenset[str],
) -> tuple[CheckerSourceModule, ...]:
    batch = _ACTIVE_MEASUREMENT_BATCH.get()
    if batch is None:
        return _collect_source_modules(roots, first_party_packages=first_party_packages)
    key = (roots, first_party_packages)
    measured = batch.source_modules.get(key)
    if measured is None:
        measured = _collect_source_modules(
            roots, first_party_packages=first_party_packages
        )
        batch.source_modules[key] = measured
    return measured


def _manifest_source_modules(
    manifest: CheckerManifest,
) -> tuple[CheckerSourceModule, ...]:
    """Return the one declared source set used by the worker import guard."""

    declared: dict[str, CheckerSourceModule] = {}
    for source in (
        *manifest.checker_source_modules,
        *manifest.worker_source_modules,
    ):
        existing = declared.setdefault(source.module, source)
        if existing.source_digest != source.source_digest:
            raise CheckerManifestError(
                "checker manifest has conflicting source digests for one module"
            )
    return tuple(declared[name] for name in sorted(declared))


def _collect_python_distributions(
    source_modules: tuple[CheckerSourceModule, ...],
    *,
    first_party_packages: frozenset[str],
) -> tuple[CheckerPythonDistribution, ...]:
    """Bind the installed distributions owning static checker-worker imports."""

    import_roots: set[str] = set()
    for bound_source in source_modules:
        source = _resolve_module(bound_source.module)
        import_roots.update(_third_party_import_roots(source, first_party_packages))
    batch = _ACTIVE_MEASUREMENT_BATCH.get()
    if batch is not None and batch.package_owners is not None:
        package_owners = batch.package_owners
    else:
        package_owners = metadata.packages_distributions()
        if batch is not None:
            batch.package_owners = package_owners
    distributions = set(_WORKER_DISTRIBUTIONS)
    for root in import_roots:
        owners = package_owners.get(root)
        if not owners:
            raise CheckerManifestError(
                "cannot bind installed distribution for checker import: " + root
            )
        if len(owners) != 1:
            raise CheckerManifestError(
                "checker import has ambiguous installed distribution owners: " + root
            )
        distributions.add(owners[0])
    measured: dict[str, CheckerPythonDistribution] = {}
    for distribution in sorted(distributions, key=_distribution_key):
        identity = _batched_python_distribution(distribution)
        key = _distribution_key(identity.distribution)
        existing = measured.setdefault(key, identity)
        if existing != identity:
            raise CheckerManifestError(
                "checker Python distribution has conflicting metadata identity: "
                + identity.distribution
            )
    return tuple(measured[key] for key in sorted(measured))


def _third_party_import_roots(
    source: _ResolvedModule,
    first_party_packages: frozenset[str],
) -> tuple[str, ...]:
    """Return static non-stdlib top-level imports from one source module."""

    try:
        tree = ast.parse(
            source.path.read_text(encoding="utf-8"), filename=str(source.path)
        )
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise CheckerManifestError(
            f"cannot parse checker source module {source.name}"
        ) from exc
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_base(node, source)
            if base is not None:
                roots.add(base.partition(".")[0])
    return tuple(
        sorted(
            root
            for root in roots
            if root not in first_party_packages
            and root not in sys.stdlib_module_names
            and root != "__future__"
        )
    )


def _manifest_distribution_roots(manifest: CheckerManifest) -> frozenset[str]:
    allowed_distributions = {
        _distribution_key(item.distribution) for item in manifest.python_distributions
    }
    return frozenset(
        root
        for root, owners in metadata.packages_distributions().items()
        if any(_distribution_key(owner) in allowed_distributions for owner in owners)
    )


def _import_requester_is_declared(declared: frozenset[str]) -> bool:
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if (
            isinstance(module, str)
            and module
            not in {
                __name__,
                "_frozen_importlib",
                "_frozen_importlib_external",
            }
            and not module.startswith("importlib")
        ):
            return module in declared
        frame = frame.f_back
    return False


def _require_declared_import(
    name: str,
    *,
    declared: frozenset[str],
    first_party_packages: frozenset[str],
    allowed_third_party_roots: frozenset[str],
) -> None:
    if not name or not _import_requester_is_declared(declared):
        return
    root = name.split(".", 1)[0]
    if (
        root not in first_party_packages
        and root not in allowed_third_party_roots
        and root not in sys.stdlib_module_names
    ):
        raise UndeclaredCheckerImportError(
            "checker import is not declared by its manifest: " + name
        )


def _guarded_import(
    declared: frozenset[str],
    first_party_packages: frozenset[str],
    allowed_third_party_roots: frozenset[str],
) -> Callable[..., Any]:
    def guarded(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        _require_declared_import(
            name,
            declared=declared,
            first_party_packages=first_party_packages,
            allowed_third_party_roots=allowed_third_party_roots,
        )
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    return guarded


def _guarded_import_module(
    declared: frozenset[str],
    first_party_packages: frozenset[str],
    allowed_third_party_roots: frozenset[str],
) -> Callable[[str, str | None], ModuleType]:
    def guarded(name: str, package: str | None = None) -> ModuleType:
        absolute = (
            importlib.util.resolve_name(name, package) if name.startswith(".") else name
        )
        _require_declared_import(
            absolute,
            declared=declared,
            first_party_packages=first_party_packages,
            allowed_third_party_roots=allowed_third_party_roots,
        )
        return _ORIGINAL_IMPORT_MODULE(name, package)

    return guarded


def _measure_python_distribution(distribution: str) -> CheckerPythonDistribution:
    try:
        installed = metadata.distribution(distribution)
    except metadata.PackageNotFoundError as exc:
        raise CheckerManifestError(
            "checker Python distribution is not installed: " + distribution
        ) from exc
    name = installed.metadata.get("Name")
    if not isinstance(name, str) or not name:
        raise CheckerManifestError(
            "checker Python distribution has no exact name: " + distribution
        )
    file_count, files_digest = _distribution_file_closure(installed)
    return CheckerPythonDistribution(
        distribution=name,
        version=installed.version,
        file_count=file_count,
        files_digest=files_digest,
    )


def _batched_python_distribution(
    distribution: str,
) -> CheckerPythonDistribution:
    batch = _ACTIVE_MEASUREMENT_BATCH.get()
    if batch is None:
        return _measure_python_distribution(distribution)
    key = _distribution_key(distribution)
    measured = batch.distributions.get(key)
    if measured is None:
        measured = _measure_python_distribution(distribution)
        batch.distributions[key] = measured
    return measured


def _batched_python_runtime() -> CheckerPythonRuntime:
    batch = _ACTIVE_MEASUREMENT_BATCH.get()
    if batch is None:
        return _python_runtime()
    if batch.python_runtime is None:
        batch.python_runtime = _python_runtime()
    return batch.python_runtime


def _distribution_file_closure(
    installed: metadata.Distribution,
) -> tuple[int, str]:
    """Hash every installed file named by the distribution's RECORD index."""

    files = installed.files
    if files is None:
        raise CheckerManifestError("checker Python distribution has no RECORD index")
    paths = tuple(sorted(path.as_posix() for path in files))
    if not paths or len(set(paths)) != len(paths):
        raise CheckerManifestError(
            "checker Python distribution has an invalid RECORD file index"
        )
    closure = hashlib.sha256(b"jacobian.checker-distribution-files.v1\x00")
    for relative in paths:
        unresolved = Path(str(installed.locate_file(relative)))
        if unresolved.is_symlink() or not unresolved.is_file():
            raise CheckerManifestError(
                "checker Python distribution contains a missing or non-regular file: "
                + relative
            )
        try:
            content_digest = _hash_installed_regular_file(unresolved)
        except OSError as exc:
            raise CheckerManifestError(
                "cannot read checker Python distribution file: " + relative
            ) from exc
        encoded_path = relative.encode("utf-8")
        closure.update(len(encoded_path).to_bytes(8, "big"))
        closure.update(encoded_path)
        closure.update(bytes.fromhex(content_digest.removeprefix("sha256:")))
    return len(paths), "sha256:" + closure.hexdigest()


def _hash_installed_regular_file(path: Path) -> str:
    """Hash one declared regular file through a single opened descriptor."""

    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CheckerManifestError(
                "checker Python distribution contains a missing or non-regular file"
            )
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
        ):
            raise CheckerManifestError(
                "checker Python distribution file changed during measurement"
            )
        return "sha256:" + hasher.hexdigest()
    finally:
        os.close(fd)


def _distribution_key(name: str) -> str:
    """Return the PEP 503 comparison key without a packaging dependency."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _first_party_imports(
    source: _ResolvedModule,
    first_party_packages: frozenset[str],
) -> tuple[str, ...]:
    try:
        tree = ast.parse(
            source.path.read_text(encoding="utf-8"), filename=str(source.path)
        )
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise CheckerManifestError(
            f"cannot parse checker source module {source.name}"
        ) from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_if_first_party(imported, alias.name, first_party_packages)
        elif isinstance(node, ast.ImportFrom):
            imported.update(
                _from_import_dependencies(node, source, first_party_packages)
            )
    return tuple(sorted(imported))


def _from_import_dependencies(
    node: ast.ImportFrom,
    source: _ResolvedModule,
    first_party_packages: frozenset[str],
) -> tuple[str, ...]:
    base = _absolute_import_base(node, source)
    if base is None or base.split(".", 1)[0] not in first_party_packages:
        return ()
    dependencies = {base}
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = f"{base}.{alias.name}"
        if _try_resolve_module(candidate) is not None:
            dependencies.add(candidate)
    return tuple(sorted(dependencies))


def _absolute_import_base(
    node: ast.ImportFrom,
    source: _ResolvedModule,
) -> str | None:
    if node.level == 0:
        return node.module
    package = source.name if source.is_package else source.name.rpartition(".")[0]
    if not package:
        return None
    parts = package.split(".")
    if node.level > len(parts):
        raise CheckerManifestError(f"relative import escapes package: {source.name}")
    parent = parts[: len(parts) - node.level + 1]
    if node.module:
        parent.extend(node.module.split("."))
    return ".".join(parent)


def _add_if_first_party(
    imported: set[str],
    module_name: str,
    first_party_packages: frozenset[str],
) -> None:
    if module_name.split(".", 1)[0] in first_party_packages:
        imported.add(module_name)


def _resolve_module(name: str) -> _ResolvedModule:
    chain = _resolve_module_chain(name)
    return chain[-1]


def _try_resolve_module(name: str) -> _ResolvedModule | None:
    try:
        return _resolve_module(name)
    except CheckerManifestError:
        return None


def _resolve_module_chain(name: str) -> tuple[_ResolvedModule, ...]:
    parts = name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise CheckerManifestError(f"invalid checker module name: {name!r}")
    specification = importlib.machinery.PathFinder.find_spec(parts[0])
    if specification is None:
        raise CheckerManifestError(f"cannot resolve checker module {name!r}")
    chain: list[_ResolvedModule] = []
    resolved = _resolved_source_module(parts[0], specification)
    if resolved is not None:
        chain.append(resolved)
    for index, part in enumerate(parts[1:], start=1):
        locations = specification.submodule_search_locations
        if locations is None:
            raise CheckerManifestError(f"cannot resolve checker module {name!r}")
        specification = importlib.machinery.PathFinder.find_spec(part, locations)
        if specification is None:
            raise CheckerManifestError(f"cannot resolve checker module {name!r}")
        resolved = _resolved_source_module(".".join(parts[: index + 1]), specification)
        if resolved is not None:
            chain.append(resolved)
    if not chain or chain[-1].name != name:
        raise CheckerManifestError(
            f"checker module must be regular Python source: {name}"
        )
    return tuple(chain)


def _resolved_source_module(
    name: str,
    specification: importlib.machinery.ModuleSpec,
) -> _ResolvedModule | None:
    if specification.origin is None and specification.submodule_search_locations:
        return None
    return _resolved_module(name, specification)


def _resolved_module(
    name: str,
    specification: importlib.machinery.ModuleSpec,
) -> _ResolvedModule:
    if specification.origin is None or not specification.origin.endswith(".py"):
        raise CheckerManifestError(
            f"checker module must be regular Python source: {name}"
        )
    unresolved = Path(specification.origin)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise CheckerManifestError(f"checker module is not a regular file: {name}")
    path = unresolved.resolve(strict=True)
    return _ResolvedModule(
        name=name,
        path=path,
        is_package=specification.submodule_search_locations is not None,
    )


def _python_runtime() -> CheckerPythonRuntime:
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise CheckerManifestError("checker Python executable is not a regular file")
    return CheckerPythonRuntime(
        implementation=sys.implementation.name,
        version=sys.version,
        executable_digest=_source_digest(executable.read_bytes()),
    )


def _source_digest(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


__all__ = [
    "CheckerManifestError",
    "UndeclaredCheckerImportError",
    "batch_checker_manifest_measurement",
    "build_checker_manifest",
    "checker_implementation_digest",
    "default_checker_sandbox_policy",
    "install_manifest_import_guard",
    "require_manifest_material_unchanged",
    "require_manifest_unchanged",
]
