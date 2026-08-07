"""Source identity for installed Python plugin and checker entrypoints."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

_PACKAGE_DIGEST_CACHE: dict[str, str] | None = None
_CHECKER_RUNTIME_ENTRYPOINT = "jacobian.checker_worker:main"


class ImplementationError(RuntimeError):
    """A Python implementation cannot be identified safely."""


class _SourceOnlyLoader(importlib.abc.Loader):
    """Compile one measured module directly from source, bypassing bytecode."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def create_module(self, spec: Any) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self.path.read_bytes()
        code = compile(source, str(self.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _SourceOnlyFinder(importlib.abc.MetaPathFinder):
    """Resolve not-yet-imported modules in one package from measured source."""

    def __init__(self, top_level_package: str) -> None:
        self.top_level_package = top_level_package

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        if fullname != self.top_level_package and not fullname.startswith(
            self.top_level_package + "."
        ):
            return None
        specification = importlib.machinery.PathFinder.find_spec(fullname, path)
        if specification is None or specification.origin is None:
            return specification
        if not specification.origin.endswith(".py"):
            raise ImportError(
                f"plugin package module is not Python source: {specification.origin}"
            )
        locations = specification.submodule_search_locations
        return importlib.util.spec_from_file_location(
            fullname,
            specification.origin,
            loader=_SourceOnlyLoader(Path(specification.origin)),
            submodule_search_locations=(
                list(locations) if locations is not None else None
            ),
        )


def install_source_only_importer(entrypoint: str) -> None:
    """Force the entrypoint package's future imports to compile measured source."""

    module_name, _ = split_entrypoint(entrypoint)
    top_level = module_name.split(".", 1)[0]
    # Purge any pre-imported modules from the target package so the
    # SourceOnlyFinder recompiles them from measured source.  Without
    # this, already-imported modules in sys.modules would shadow the
    # finder and serve potentially stale bytecode.
    stale = [
        name
        for name in list(sys.modules)
        if name == top_level or name.startswith(top_level + ".")
    ]
    for name in stale:
        del sys.modules[name]
    sys.meta_path.insert(0, _SourceOnlyFinder(top_level))


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    try:
        module_name, attribute_name = entrypoint.split(":", 1)
    except ValueError as exc:
        raise ImplementationError(
            "entrypoint must use the form module:attribute"
        ) from exc
    if (
        not module_name
        or not attribute_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute_name.isidentifier()
    ):
        raise ImplementationError("entrypoint must use the form module:attribute")
    return module_name, attribute_name


def _package_entries(module_name: str) -> list[tuple[str, Path]]:
    """Resolve a module from its top-level package without importing parents."""

    top_level, *remaining = module_name.split(".")
    specification = importlib.machinery.PathFinder.find_spec(top_level)
    if specification is None:
        raise ImplementationError(f"cannot resolve package {top_level!r}")

    locations = specification.submodule_search_locations
    if locations:
        roots = [Path(location) for location in locations]
        if not _module_exists_in_roots(roots, remaining):
            raise ImplementationError(f"cannot resolve module {module_name!r}")
        entries: list[tuple[str, Path]] = []
        for root_index, root in enumerate(roots):
            if root.is_symlink() or not root.is_dir():
                raise ImplementationError(
                    f"package root is not a regular directory: {root}"
                )
            for directory, names, files in os.walk(root, followlinks=False):
                directory_path = Path(directory)
                names[:] = [name for name in names if name != "__pycache__"]
                for name in names:
                    child = directory_path / name
                    if child.is_symlink():
                        raise ImplementationError(
                            f"package contains a symlink: {child}"
                        )
                for name in files:
                    entry = directory_path / name
                    if entry.is_symlink() or not entry.is_file():
                        raise ImplementationError(
                            f"package entry is not a regular file: {entry}"
                        )
                    relative = entry.relative_to(root).as_posix()
                    entries.append((f"{root_index}:{top_level}/{relative}", entry))
        if not entries:
            raise ImplementationError(f"package {top_level!r} has no files")
        return sorted(entries)

    if specification.origin is None:
        raise ImplementationError(f"module {top_level!r} has no source")
    source = Path(specification.origin)
    if remaining:
        raise ImplementationError(f"{top_level!r} is not a package")
    if source.is_symlink() or not source.is_file() or source.suffix != ".py":
        raise ImplementationError(
            f"module source is not a regular Python file: {source}"
        )
    return [(f"{top_level}.py", source)]


def _module_exists_in_roots(roots: list[Path], remaining: list[str]) -> bool:
    for root in roots:
        if not remaining:
            if (root / "__init__.py").is_file():
                return True
            continue
        relative = Path(*remaining)
        module_file = root / relative.with_suffix(".py")
        package_file = root / relative / "__init__.py"
        for candidate in (module_file, package_file):
            if candidate.is_file() and not candidate.is_symlink():
                return True
    return False


def package_source_digest(entrypoint: str) -> str:
    """Hash every regular file in an entrypoint's top-level package.

    Binding the package rather than only the named module prevents unchecked
    helper modules and data files from changing an authorized implementation.
    Bytecode caches are excluded because workers execute measured source.

    Callers may enter :func:`cached_package_digests` during a runtime attach so
    repeated authorizations of the same package reuse one digest without
    suppressing later on-disk package edits outside that scope.
    """

    module_name, _ = split_entrypoint(entrypoint)
    top_level, *remaining = module_name.split(".")
    _ensure_module_in_package(top_level, remaining)
    cache = _PACKAGE_DIGEST_CACHE
    if cache is not None and top_level in cache:
        return cache[top_level]
    digest = _digest_top_level_package(top_level)
    if cache is not None:
        cache[top_level] = digest
    return digest


def package_import_path(entrypoint: str) -> str:
    """Return the measured package's exact import roots for a worker process."""

    module_name, _ = split_entrypoint(entrypoint)
    top_level, *remaining = module_name.split(".")
    specification = importlib.machinery.PathFinder.find_spec(top_level)
    if specification is None:
        raise ImplementationError(f"cannot resolve package {top_level!r}")
    locations = specification.submodule_search_locations
    if locations:
        roots = [Path(location) for location in locations]
        if not _module_exists_in_roots(roots, remaining):
            raise ImplementationError(f"cannot resolve module {module_name!r}")
        if any(root.is_symlink() or not root.is_dir() for root in roots):
            raise ImplementationError(
                f"package root is not a regular directory: {roots[0]}"
            )
        parents = tuple(str(root.resolve(strict=True).parent) for root in roots)
        return os.pathsep.join(dict.fromkeys(parents))
    if remaining or specification.origin is None:
        raise ImplementationError(f"cannot resolve module {module_name!r}")
    unresolved_source = Path(specification.origin)
    if unresolved_source.is_symlink() or not unresolved_source.is_file():
        raise ImplementationError(
            f"module source is not a regular file: {unresolved_source}"
        )
    source = unresolved_source.resolve(strict=True)
    if not source.is_file():
        raise ImplementationError(f"module source is not a regular file: {source}")
    return str(source.parent)


def checker_source_digest(entrypoint: str) -> str:
    """Bind checker source together with the complete Jacobian worker runtime."""

    digest = hashlib.sha256()
    digest.update(b"jacobian.checker-source.v1\x00")
    digest.update(package_source_digest(entrypoint).encode("ascii"))
    digest.update(b"\x00")
    digest.update(package_source_digest(_CHECKER_RUNTIME_ENTRYPOINT).encode("ascii"))
    return "sha256:" + digest.hexdigest()


@contextmanager
def cached_package_digests() -> Iterator[None]:
    """Reuse top-level package digests within one attach/authorization batch."""

    global _PACKAGE_DIGEST_CACHE
    if _PACKAGE_DIGEST_CACHE is not None:
        yield
        return
    _PACKAGE_DIGEST_CACHE = {}
    try:
        yield
    finally:
        _PACKAGE_DIGEST_CACHE = None


def _ensure_module_in_package(top_level: str, remaining: list[str]) -> None:
    specification = importlib.machinery.PathFinder.find_spec(top_level)
    if specification is None:
        raise ImplementationError(f"cannot resolve package {top_level!r}")
    locations = specification.submodule_search_locations
    if locations:
        roots = [Path(location) for location in locations]
        if not _module_exists_in_roots(roots, remaining):
            module_name = ".".join([top_level, *remaining])
            raise ImplementationError(f"cannot resolve module {module_name!r}")
        return
    if remaining:
        raise ImplementationError(f"{top_level!r} is not a package")
    if specification.origin is None:
        raise ImplementationError(f"module {top_level!r} has no source")


def _digest_top_level_package(top_level: str) -> str:
    entries = _package_entries(top_level)
    digest = hashlib.sha256()
    digest.update(b"jacobian.python-package.v2\x00")
    digest.update(top_level.encode("utf-8"))
    digest.update(b"\x00")
    for relative_name, source in entries:
        name_bytes = relative_name.encode("utf-8")
        content = source.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()
