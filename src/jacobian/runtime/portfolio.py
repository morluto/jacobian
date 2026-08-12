"""Runtime-owned record and resources produced by portfolio assembly."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.lean_frontend.declarations import LeanDeclarationService
from jacobian.lean_frontend.exploration import LeanExplorationInstallation
from jacobian.lean_frontend.service import LeanService


@dataclass(slots=True)
class PortfolioResources:
    """Closeable resources retained by one live runtime."""

    lean: LeanService | None = None
    lean_declarations: LeanDeclarationService | None = None
    lean_exploration: LeanExplorationInstallation | None = None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        for resource in (
            self.lean_declarations,
            self.lean_exploration.repl if self.lean_exploration is not None else None,
            self.lean,
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            exceptions = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exceptions) == len(failures):
                raise ExceptionGroup("portfolio resources failed to close", exceptions)
            raise BaseExceptionGroup("portfolio resources failed to close", failures)
        self._closed = True


__all__ = ["PortfolioResources"]
