"""Read Jacobian's literal, repository-contained Make include graph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_INCLUDE = re.compile(r"^(?:-?include)\s+([^\s#]+)\s*(?:#.*)?$")
_DYNAMIC_MARKERS = ("$", "*", "?", "[", "~")


class MakefileCatalogError(ValueError):
    """Raised when the Make command surface cannot be read safely."""


@dataclass(frozen=True)
class MakefileCatalog:
    """The ordered, de-duplicated files that define a Make entrypoint."""

    files: tuple[Path, ...]

    def text(self) -> str:
        """Return the graph's source in Make parse order."""

        return "\n".join(path.read_text(encoding="utf-8") for path in self.files)


def discover_makefiles(root: Path, entrypoint: Path | None = None) -> MakefileCatalog:
    """Resolve literal ``include`` directives without leaving ``root``.

    Command-surface consumers must agree on exactly which files Make reads.  Dynamic,
    optional, missing, escaping, and cyclic includes are rejected rather than silently
    producing an incomplete catalog.
    """

    root = root.resolve()
    start = (entrypoint or root / "Makefile").resolve()
    files: list[Path] = []
    visiting: set[Path] = set()
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise MakefileCatalogError(f"include escapes repository: {path}") from exc
        if path in visiting:
            raise MakefileCatalogError(f"cyclic Make include: {relative.as_posix()}")
        if path in visited:
            return
        if not path.is_file():
            raise MakefileCatalogError(f"included Makefile is missing: {relative.as_posix()}")
        visiting.add(path)
        files.append(path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _INCLUDE.match(line)
            if match is None:
                continue
            include = match.group(1)
            if any(marker in include for marker in _DYNAMIC_MARKERS):
                raise MakefileCatalogError(
                    f"dynamic Make include at {relative.as_posix()}:{line_number}"
                )
            candidate = (path.parent / include).resolve()
            visit(candidate)
        visiting.remove(path)
        visited.add(path)

    visit(start)
    return MakefileCatalog(tuple(files))
