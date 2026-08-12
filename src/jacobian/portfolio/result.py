"""Domain-bundle installation observations consumed during portfolio assembly."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from jacobian.operation_installation import InstalledDomainBundle

PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
_DIAGNOSTIC_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class BundleInstallationStatus(StrEnum):
    """The lifecycle of one domain bundle within a portfolio installation."""

    INSTALLED = "INSTALLED"
    SKIPPED_PROVIDER_UNAVAILABLE = "SKIPPED_PROVIDER_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostic:
    """One inspectable, non-conclusive portfolio installation observation."""

    code: str
    component_id: str
    stage: str
    message: str

    def __post_init__(self) -> None:
        if not _DIAGNOSTIC_CODE_PATTERN.fullmatch(self.code):
            raise ValueError("portfolio diagnostic code has an invalid format")


@dataclass(frozen=True, slots=True)
class BundleInstallation:
    """One domain-bundle installation outcome."""

    domain_id: str
    status: BundleInstallationStatus
    capability_ids: tuple[str, ...]
    installed: InstalledDomainBundle | None
    diagnostic: PortfolioDiagnostic | None

    def __post_init__(self) -> None:
        installed = self.status is BundleInstallationStatus.INSTALLED
        if installed != (self.installed is not None):
            raise ValueError("bundle installation status and value disagree")
        if installed == (self.diagnostic is not None):
            raise ValueError("bundle installation status and diagnostic disagree")


@dataclass(frozen=True, slots=True)
class PortfolioInstallationResult:
    """Immutable result for the ordinary domain-bundle plan."""

    installed: Mapping[str, InstalledDomainBundle]
    diagnostics: tuple[PortfolioDiagnostic, ...]
    outcomes: tuple[BundleInstallation, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            outcome.diagnostic
            for outcome in self.outcomes
            if outcome.diagnostic is not None
        )
        if self.diagnostics != expected:
            raise ValueError("portfolio diagnostics do not match bundle outcomes")
        object.__setattr__(self, "installed", MappingProxyType(dict(self.installed)))


__all__ = [
    "PROVIDER_UNAVAILABLE",
    "BundleInstallation",
    "BundleInstallationStatus",
    "PortfolioDiagnostic",
    "PortfolioInstallationResult",
]
