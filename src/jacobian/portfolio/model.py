"""Typed declaration of the explicitly installed portfolio."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.domain_bundles import DomainBundle


@dataclass(frozen=True, slots=True)
class PortfolioPlan:
    """Explicit, ordered built-in portfolio without dynamic discovery.

    The plan is a literal, ordered tuple of ordinary domain bundles. It
    performs no discovery, registration, or ranking:
    callers install it through
    :class:`jacobian.portfolio.domain_installation.DomainBundleInstaller`,
    which records every per-component outcome as a typed diagnostic.
    """

    components: tuple[DomainBundle, ...]

    def validate(self) -> None:
        """Reject structural portfolio defects before installation.

        Plan-level defects (invalid components, blank IDs, duplicate IDs)
        are programming errors and fail fast. Installation failures other than
        declared provider unavailability also propagate from the assembler.
        """

        domain_ids: list[str] = []
        for bundle in self.components:
            _validate_bundle(bundle)
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

    def component_for(self, domain_id: str) -> DomainBundle | None:
        """Return the bundle declared for ``domain_id``, or ``None``."""

        for bundle in self.components:
            if bundle.domain_id == domain_id:
                return bundle
        return None


def _validate_bundle(bundle: object) -> None:
    if not isinstance(bundle, DomainBundle):
        raise TypeError(
            f"portfolio entries must be domain bundles, not {type(bundle).__name__}"
        )
    if not bundle.domain_id:
        raise ValueError("portfolio contains a bundle with a blank domain id")
    if len(bundle.capability_ids) != len(set(bundle.capability_ids)):
        raise ValueError(f"bundle {bundle.domain_id} has duplicate capability IDs")
