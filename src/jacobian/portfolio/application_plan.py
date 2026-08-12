"""Typed application installation plans and structural receipts.

Complete and scoped installations share one plan vocabulary. Receipts prove
what was installed and authorized; they are not mathematical evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from jacobian.operation_installation import InstalledDomainBundle
from jacobian.runtime.config import CheckerAuthorityMode

ApplicationKind = Literal["complete", "scoped"]


@dataclass(frozen=True, slots=True)
class ApplicationInstallPlan:
    """Closed description of one application installation.

    ``domain_ids`` is empty for the complete builtin portfolio. Scoped plans
    list explicit domain IDs in install order.
    """

    kind: ApplicationKind
    domain_ids: tuple[str, ...]
    checker_authority: CheckerAuthorityMode
    include_exact_verification: bool = True

    def __post_init__(self) -> None:
        if self.kind == "complete" and self.domain_ids:
            raise ValueError("complete plans must not list scoped domain_ids")
        if self.kind == "scoped" and not self.domain_ids:
            raise ValueError("scoped plans require at least one domain_id")
        if len(set(self.domain_ids)) != len(self.domain_ids):
            raise ValueError("domain_ids must be unique")

    @classmethod
    def complete(
        cls,
        *,
        checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    ) -> ApplicationInstallPlan:
        return cls(
            kind="complete",
            domain_ids=(),
            checker_authority=checker_authority,
            include_exact_verification=True,
        )

    @classmethod
    def scoped(
        cls,
        domain_ids: Sequence[str],
        *,
        checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
        include_exact_verification: bool = True,
    ) -> ApplicationInstallPlan:
        return cls(
            kind="scoped",
            domain_ids=tuple(domain_ids),
            checker_authority=checker_authority,
            include_exact_verification=include_exact_verification,
        )

    def digest(self) -> str:
        payload = {
            "kind": self.kind,
            "domain_ids": list(self.domain_ids),
            "checker_authority": self.checker_authority.value,
            "include_exact_verification": self.include_exact_verification,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InstallationReceipt:
    """Structural evidence for one successful installation."""

    plan_kind: ApplicationKind
    plan_digest: str
    domain_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    checker_ids: tuple[str, ...]
    checker_authority: str
    schema_uris: tuple[str, ...]

    def domain_projection(self) -> dict[str, object]:
        """Project domain-owned fields for scoped↔complete equivalence checks."""

        return {
            "domain_ids": list(self.domain_ids),
            "capability_ids": list(self.capability_ids),
            "checker_ids": list(self.checker_ids),
            "schema_uris": list(self.schema_uris),
            "checker_authority": self.checker_authority,
        }


def receipt_from_installed_bundles(
    plan: ApplicationInstallPlan,
    bundles: Mapping[str, InstalledDomainBundle],
    *,
    checker_ids: Sequence[str] = (),
) -> InstallationReceipt:
    """Build a receipt from installed domain bundles and checker IDs."""

    ordered_ids = plan.domain_ids or tuple(sorted(bundles))
    missing = tuple(domain_id for domain_id in ordered_ids if domain_id not in bundles)
    if missing:
        raise ValueError(
            "installation receipt missing installed bundle(s): " + ", ".join(missing)
        )
    capability_ids: list[str] = []
    schema_uris: list[str] = []
    for domain_id in ordered_ids:
        installed = bundles[domain_id]
        capability_ids.extend(
            sorted({adapter.descriptor.capability_id for adapter in installed.adapters})
        )
        schema_uris.append(installed.semantics_uri)
        schema_uris.extend(installed.input_schema_uris.values())
        schema_uris.extend(installed.result_schema_uris.values())
        schema_uris.extend(installed.named_schema_uris.values())
    return InstallationReceipt(
        plan_kind=plan.kind,
        plan_digest=plan.digest(),
        domain_ids=ordered_ids,
        capability_ids=tuple(capability_ids),
        checker_ids=tuple(sorted(set(checker_ids))),
        checker_authority=plan.checker_authority.value,
        schema_uris=tuple(sorted(set(schema_uris))),
    )


__all__ = [
    "ApplicationInstallPlan",
    "ApplicationKind",
    "InstallationReceipt",
    "receipt_from_installed_bundles",
]
