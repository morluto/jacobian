"""Well-formedness validation for stored claims and installed plugins."""

from __future__ import annotations

import logging

from jacobian.contracts.claims import (
    ClaimSpec,
    ClaimValidationResult,
)
from jacobian.contracts.plugins import CapabilityName
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
)
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class ClaimValidationService:
    """Validate a claim's structure and capabilities without proving it."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins

    def validate(
        self,
        *,
        claim_uri: str,
        plugin_id: str,
    ) -> ClaimValidationResult:
        """Check a stored claim against the selected installed plugin."""

        errors: list[str] = []
        warnings: list[str] = []
        claim_digest: str | None = None
        semantics_digest: str | None = None
        required: tuple[CapabilityName, ...] = ()
        available: tuple[CapabilityName, ...] = ()

        try:
            artifact = self.store.get(claim_uri)
            claim_digest = artifact.manifest.object_digest
            manifest = self.plugins.get(plugin_id)
            available = tuple(
                sorted(manifest.capabilities, key=lambda item: item.value)
            )
            if artifact.manifest.schema_uri != manifest.claim_schema_uri:
                errors.append("claim schema does not match the plugin claim schema")
            if artifact.manifest.semantics_uri != manifest.semantics_uri:
                errors.append("claim artifact semantics do not match plugin semantics")
            normalized = self.schemas.validate(
                manifest.claim_schema_uri,
                artifact.payload,
            )
            claim = ClaimSpec.model_validate(normalized)
            required = claim.required_capabilities
            if claim.domain_id != manifest.domain_id:
                errors.append("claim domain does not match plugin domain")
            if claim.domain_version != manifest.domain_version:
                errors.append(
                    "claim domain version does not match plugin domain version"
                )
            if claim.semantics_uri != manifest.semantics_uri:
                errors.append("claim semantics binding does not match plugin semantics")
            semantics = self.store.get_descriptor(
                claim.semantics_uri,
                expected_kind="semantics",
            )
            semantics_artifact = self.store.get(claim.semantics_uri)
            if semantics.get("kind") != "semantics":
                errors.append("claim semantics descriptor is invalid")
            semantics_digest = semantics_artifact.manifest.object_digest
        except (
            StorageError,
            SchemaRegistryError,
            PluginRegistryError,
            ValueError,
        ) as exc:
            _LOGGER.warning("claim validation failed", exc_info=exc)
            errors.append(_claim_validation_error_message(exc))

        missing = tuple(
            sorted(
                set(required) - set(available),
                key=lambda item: item.value,
            )
        )
        if missing:
            errors.append(
                "The selected plugin is missing required capabilities: "
                + ", ".join(item.value for item in missing)
                + ". Call math.find and choose a plugin that advertises "
                "all required capabilities."
            )
        valid = not errors
        return ClaimValidationResult(
            execution=Execution(status=ExecutionStatus.COMPLETED),
            input=InputValidation(
                status=(InputStatus.ACCEPTED if valid else InputStatus.REJECTED),
                errors=tuple(errors),
                warnings=tuple(warnings),
            ),
            valid=valid,
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            claim_digest=claim_digest,
            resolved_semantics_digest=semantics_digest,
            required_capabilities=required,
            available_capabilities=available,
            missing_capabilities=missing,
        )


def _claim_validation_error_message(exc: Exception) -> str:
    if isinstance(exc, StorageError):
        return (
            "The claim or its semantics artifact is unavailable. Check the "
            "artifact URI, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The selected plugin is unavailable. Call math.find, "
            "choose an installed reference domain, and retry."
        )
    return (
        "The claim does not match the selected reference contract. "
        "Recreate it from that contract, then retry."
    )
