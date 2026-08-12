"""Typed declaration of the explicitly installed portfolio."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.operations import DomainBundle

if TYPE_CHECKING:
    from jacobian.installation.context import InstallationContext
    from jacobian.operation_installation import InstalledDomainBundle


@dataclass(frozen=True, slots=True)
class ManagedPortfolioComponent:
    """Exceptional installation unit owned by the portfolio composition root."""

    domain_id: str
    provider_runtime: CapabilityProviderRuntime
    capability_ids: tuple[str, ...]
    install: Callable[
        [InstallationContext, Mapping[str, InstalledDomainBundle]],
        InstalledDomainBundle,
    ]
    dependency_ids: tuple[str, ...] = ()


type PortfolioComponent = DomainBundle | ManagedPortfolioComponent


@dataclass(frozen=True, slots=True)
class PortfolioPlan:
    """Explicit, ordered built-in portfolio without dynamic discovery.

    The plan is a literal, ordered tuple of ordinary domain bundles and named
    managed components. It performs no discovery, registration, or ranking:
    callers install it through
    :class:`jacobian.portfolio.domain_installation.DomainBundleInstaller`,
    which records every per-component outcome as a typed diagnostic.
    """

    components: tuple[PortfolioComponent, ...]

    def validate(self) -> None:
        """Reject structural portfolio defects before installation.

        Plan-level defects (invalid components, blank IDs, duplicate IDs)
        are programming errors and fail fast. Installation failures other than
        declared provider unavailability also propagate from the assembler.
        """

        domain_ids: list[str] = []
        for bundle in self.components:
            _validate_bundle(bundle)
            dependency_ids = (
                bundle.dependency_ids
                if isinstance(bundle, ManagedPortfolioComponent)
                else ()
            )
            missing = tuple(
                dependency_id
                for dependency_id in dependency_ids
                if dependency_id not in domain_ids
            )
            if missing:
                raise ValueError(
                    f"bundle {bundle.domain_id} dependencies must be declared earlier: "
                    + ", ".join(missing)
                )
            domain_ids.append(bundle.domain_id)
        duplicates = sorted(
            domain_id
            for domain_id in set(domain_ids)
            if domain_ids.count(domain_id) > 1
        )
        if duplicates:
            raise ValueError(
                "portfolio contains duplicate domain bundles: " + ", ".join(duplicates)
            )

    @property
    def domain_ids(self) -> tuple[str, ...]:
        """The ordered domain IDs declared by this plan."""

        return tuple(bundle.domain_id for bundle in self.components)

    def component_for(self, domain_id: str) -> PortfolioComponent | None:
        """Return the component declared for ``domain_id``, or ``None``."""

        for bundle in self.components:
            if bundle.domain_id == domain_id:
                return bundle
        return None


def _validate_bundle(bundle: object) -> None:
    if not isinstance(bundle, (DomainBundle, ManagedPortfolioComponent)):
        raise TypeError(
            "portfolio entries must be domain bundles or managed components, "
            f"not {type(bundle).__name__}"
        )
    if not bundle.domain_id:
        raise ValueError("portfolio contains a bundle with a blank domain id")
    dependency_ids = (
        bundle.dependency_ids if isinstance(bundle, ManagedPortfolioComponent) else ()
    )
    if len(dependency_ids) != len(set(dependency_ids)):
        raise ValueError(f"bundle {bundle.domain_id} has duplicate dependency IDs")
    if bundle.domain_id in dependency_ids:
        raise ValueError(f"bundle {bundle.domain_id} cannot depend on itself")
    if isinstance(bundle, ManagedPortfolioComponent) and not bundle.capability_ids:
        raise ValueError(f"managed component {bundle.domain_id} must declare IDs")
    if len(bundle.capability_ids) != len(set(bundle.capability_ids)):
        raise ValueError(f"component {bundle.domain_id} has duplicate capability IDs")
