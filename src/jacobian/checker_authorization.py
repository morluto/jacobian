"""Standalone authorization of retained polytope and Lean checkers.

This module owns the operator-facing installation of the two retained checker
families:

* finite-polytope convex-combination and linear-separator checkers, and
* pinned Lean 4 kernel certificate checkers.

It depends only on the operator-owned :class:`CheckerRegistry`, the shared
:class:`SchemaRegistry`, and the artifact store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from jacobian.checker_operations import CheckerOperation, InstalledChecker
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope
from jacobian.contracts.lean import LeanCandidate, LeanClaim, LeanEnvironment
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository

__all__ = [
    "LeanCheckerInstallation",
    "PolytopeCheckerInstallation",
    "authorize_checker",
    "authorize_checker_operation",
    "install_lean_checkers",
    "install_polytope_checkers",
    "register_lean_checker_contracts",
]


def authorize_checker_operation(
    registry: CheckerRegistry,
    operation: CheckerOperation,
    *,
    authorize: bool,
    bind_existing: bool | None = None,
) -> InstalledChecker:
    """Authorize or bind one declared checker through the operator registry."""

    should_bind_existing = (
        registry.bind_existing_when_omitted if bind_existing is None else bind_existing
    )
    if authorize:
        registration = registry.authorize(
            name=operation.name,
            entrypoint=operation.entrypoint,
            evidence_kind=operation.evidence_kind,
            format_id=operation.format_id,
            format_version=operation.format_version,
            claim_schema_uris=operation.claim_schema_uris,
            semantics_uris=operation.semantics_uris,
            candidate_schema_uris=operation.candidate_schema_uris,
            target_schema_uris=operation.target_schema_uris,
            target_semantics_uris=operation.target_semantics_uris,
            provider_runtime=operation.provider_runtime,
            reason=operation.reason,
        )
        return InstalledChecker(operation=operation, checker_id=registration.checker_id)
    if should_bind_existing:
        return InstalledChecker(
            operation=operation,
            checker_id=registry.bind_existing(
                name=operation.name,
                entrypoint=operation.entrypoint,
                evidence_kind=operation.evidence_kind,
                format_id=operation.format_id,
                format_version=operation.format_version,
                claim_schema_uris=operation.claim_schema_uris,
                semantics_uris=operation.semantics_uris,
                candidate_schema_uris=operation.candidate_schema_uris,
                target_schema_uris=operation.target_schema_uris,
                target_semantics_uris=operation.target_semantics_uris,
                provider_runtime=operation.provider_runtime,
            ),
        )
    return InstalledChecker(operation=operation, checker_id=None)


@dataclass(frozen=True, slots=True)
class PolytopeCheckerInstallation:
    witness_checker_id: str | None
    certificate_checker_id: str | None


@dataclass(frozen=True, slots=True)
class LeanCheckerInstallation:
    environment: LeanEnvironment
    lean_version: str
    lean_commit: str
    import_name: str | None
    mathlib_commit: str | None
    allowed_axioms: tuple[str, ...]
    checker_timeout_seconds: int
    semantics_uri: str
    claim_schema_uri: str
    candidate_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


_LEAN_VERSION = "4.31.0"
_LEAN_COMMIT = "68218e876d2a38b1985b8590fff244a83c321783"
_MATHLIB_COMMIT = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
_LEAN_CONFIGURATIONS: dict[
    LeanEnvironment,
    tuple[str | None, str | None, tuple[str, ...], int],
] = {
    LeanEnvironment.CORE: (None, None, (), 30),
    LeanEnvironment.MATHLIB: (
        "Mathlib",
        _MATHLIB_COMMIT,
        ("Classical.choice", "Quot.sound", "propext"),
        225,
    ),
}


def register_lean_checker_contracts(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    *,
    checker_ids: dict[LeanEnvironment, str | None] | None = None,
) -> tuple[
    dict[LeanEnvironment, LeanCheckerInstallation],
    dict[str, dict[str, Any]],
]:
    """Register passive Lean contracts without probing or authorizing Lean."""

    claim_schema_uri = schemas.register(
        name="jacobian.lean4.claim",
        version="1",
        schema=model_schema(LeanClaim),
    )
    candidate_schema_uri = schemas.register(
        name="jacobian.lean4.candidate",
        version="1",
        schema=model_schema(LeanCandidate),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    selected_ids = checker_ids or {}
    profiles: dict[str, dict[str, Any]] = {}
    installations: dict[LeanEnvironment, LeanCheckerInstallation] = {}
    for environment, (
        import_name,
        pinned_mathlib,
        allowed_axioms,
        checker_timeout_seconds,
    ) in _LEAN_CONFIGURATIONS.items():
        semantics_uri = store.register_descriptor(
            kind="semantics",
            name=f"jacobian.lean4-{environment.value.lower()}",
            version="1",
            definition={
                "description": (
                    "exact Lean proposition checked by the pinned Lean kernel"
                ),
                "environment": environment.value,
                "lean_version": _LEAN_VERSION,
                "lean_commit": _LEAN_COMMIT,
                "import_name": import_name,
                "mathlib_commit": pinned_mathlib,
                "allowed_axioms": list(allowed_axioms),
                "checker_timeout_seconds": checker_timeout_seconds,
                "trust_level": 0,
            },
        )
        profiles[environment.value] = {
            "semantics_uri": semantics_uri,
            "lean_version": _LEAN_VERSION,
            "lean_commit": _LEAN_COMMIT,
            "import_name": import_name,
            "mathlib_commit": pinned_mathlib,
            "allowed_axioms": list(allowed_axioms),
            "checker_timeout_seconds": checker_timeout_seconds,
        }
        installations[environment] = LeanCheckerInstallation(
            environment=environment,
            lean_version=_LEAN_VERSION,
            lean_commit=_LEAN_COMMIT,
            import_name=import_name,
            mathlib_commit=pinned_mathlib,
            allowed_axioms=allowed_axioms,
            checker_timeout_seconds=checker_timeout_seconds,
            semantics_uri=semantics_uri,
            claim_schema_uri=claim_schema_uri,
            candidate_schema_uri=candidate_schema_uri,
            certificate_schema_uri=certificate_schema_uri,
            checker_id=selected_ids.get(environment),
        )
    return installations, profiles


def authorize_checker(
    checkers: CheckerRegistry,
    *,
    name: str,
    entrypoint: str,
    evidence_kind: str,
    format_id: str,
    claim_schema: str,
    semantics: str,
    candidate_schema: str,
    provider_runtime: ProviderObservation | None = None,
) -> str | None:
    """Authorize one checker through the operator-owned registry."""

    installed = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name=name,
            entrypoint=entrypoint,
            evidence_kind=EvidenceKind(evidence_kind),
            format_id=format_id,
            format_version="1",
            claim_schema_uris=(claim_schema,),
            semantics_uris=(semantics,),
            candidate_schema_uris=(candidate_schema,),
            provider_runtime=provider_runtime,
            reason="bundled retained checker",
        ),
        authorize=not checkers.bind_existing_when_omitted,
    )
    if checkers.bind_existing_when_omitted:
        # Hydrate may omit checkers that were never authorized on this store.
        return installed.checker_id
    return installed.require_checker_id()


def install_polytope_checkers(
    checkers: CheckerRegistry,
    *,
    claim_schema_uri: str,
    semantics_uri: str,
    point_schema_uri: str,
) -> PolytopeCheckerInstallation:
    """Authorize the separately implemented finite-polytope replay code."""

    witness_checker_id = authorize_checker(
        checkers,
        name="finite-polytope convex-combination checker",
        entrypoint="jacobian_checkers.polytope:check_convex_combination",
        evidence_kind="WITNESS",
        format_id="polytope.convex_combination",
        claim_schema=claim_schema_uri,
        semantics=semantics_uri,
        candidate_schema=point_schema_uri,
    )
    certificate_checker_id = authorize_checker(
        checkers,
        name="finite-polytope linear-separator checker",
        entrypoint="jacobian_checkers.polytope:check_linear_separator",
        evidence_kind="CERTIFICATE",
        format_id="polytope.linear_separator",
        claim_schema=claim_schema_uri,
        semantics=semantics_uri,
        candidate_schema=point_schema_uri,
    )
    return PolytopeCheckerInstallation(
        witness_checker_id=witness_checker_id,
        certificate_checker_id=certificate_checker_id,
    )


def install_lean_checkers(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    checkers: CheckerRegistry,
    *,
    resolve_provider_runtime: Callable[
        [dict[str, dict[str, Any]]], ProviderObservation
    ],
) -> tuple[dict[LeanEnvironment, LeanCheckerInstallation], ProviderObservation]:
    """Authorize Lean checkers bound to their measured provider runtime."""
    installations, profiles = register_lean_checker_contracts(store, schemas)
    provider_runtime = resolve_provider_runtime(profiles)
    for environment, installation in installations.items():
        checker_id = None
        if provider_runtime.availability is ProviderAvailability.AVAILABLE:
            checker_id = authorize_checker(
                checkers,
                name=f"pinned {environment.value} Lean kernel checker",
                entrypoint="jacobian_checkers.lean4:check_kernel_certificate",
                evidence_kind="CERTIFICATE",
                format_id="lean4.kernel",
                claim_schema=installation.claim_schema_uri,
                semantics=installation.semantics_uri,
                candidate_schema=installation.candidate_schema_uri,
                provider_runtime=provider_runtime,
            )
        installations[environment] = replace(installation, checker_id=checker_id)
    return installations, provider_runtime
