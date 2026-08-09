"""Installation and immutable resolution of untrusted domain plugins."""

from __future__ import annotations

import hashlib
import logging
import platform
import sysconfig
from dataclasses import dataclass

from jacobian.canonical import canonicalize_json
from jacobian.contracts.plugins import (
    CapabilityDescriptor,
    CapabilityName,
    PluginManifest,
    PluginRegistrySnapshot,
    PluginRuntimeIdentity,
    SealedCapabilityBinding,
)
from jacobian.implementation import (
    ImplementationError,
    package_source_digest,
)
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


class PluginRegistryError(RuntimeError):
    """A plugin manifest or implementation binding is invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    """A plugin capability bound to the source digest measured at resolution."""

    plugin_id: str
    name: CapabilityName
    descriptor: CapabilityDescriptor
    implementation_digest: str
    registry_snapshot_uri: str


class PluginRegistry:
    """Seal operator-installed packages without granting checker authority.

    Installation binds the manifest and every capability to its implementation
    package digest plus runtime, build, and platform identity. Discovery reads
    source without importing package code; resolution remeasures it before
    execution. This establishes implementation identity, not host isolation or
    mathematical trust.
    """

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry | None = None,
    ) -> None:
        self.store = store
        self.schemas = schemas or SchemaRegistry(store)
        self.snapshot_schema_uri = self.schemas.register(
            name="jacobian.plugin-registry-snapshot",
            version="1",
            schema=model_schema(PluginRegistrySnapshot),
        )
        self.snapshot_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.plugin-registry-snapshot",
            version="1",
            definition={
                "description": (
                    "immutable installed plugin contract, source, runtime, "
                    "and platform binding"
                )
            },
        )

    def register_implementation(self, entrypoint: str) -> str:
        """Record an operator-installed entrypoint and its current source digest."""

        try:
            digest = package_source_digest(entrypoint)
        except ImplementationError as exc:
            _LOGGER.warning(
                "could not register plugin implementation %s",
                entrypoint,
                exc_info=exc,
            )
            raise PluginRegistryError(
                "The plugin entrypoint could not be loaded. Check that the "
                "module:function entrypoint is installed, then retry."
            ) from exc
        return self.store.register_descriptor(
            kind="implementation",
            name=entrypoint,
            version="1",
            definition={
                "entrypoint": entrypoint,
                "module_digest": digest,
            },
        )

    def install(self, plugin_id: str) -> PluginManifest:
        """Validate dependencies and publish one immutable registry snapshot.

        Every capability must resolve to its measured entrypoint, and the
        snapshot freezes that collection of implementation bindings. Plugin
        manifests have no field or side channel that can authorize a checker.
        """

        try:
            artifact = self.store.get(plugin_id)
            manifest = PluginManifest.model_validate(artifact.payload)
            self.store.get_descriptor(
                manifest.semantics_uri,
                expected_kind="semantics",
            )
            self.store.get_descriptor(
                manifest.claim_schema_uri,
                expected_kind="schema",
            )
            self.store.get_descriptor(
                manifest.candidate_schema_uri,
                expected_kind="schema",
            )
            for schema_uri in (
                *manifest.witness_schema_uris,
                *manifest.certificate_schema_uris,
            ):
                self.store.get_descriptor(schema_uri, expected_kind="schema")
            capability_bindings: dict[
                CapabilityName,
                SealedCapabilityBinding,
            ] = {}
            for name, descriptor in manifest.capabilities.items():
                implementation = self.store.get_descriptor(
                    descriptor.implementation_uri,
                    expected_kind="implementation",
                )
                definition = implementation.get("definition")
                if not isinstance(definition, dict):
                    raise PluginRegistryError(
                        "implementation descriptor has no object definition"
                    )
                if definition.get("entrypoint") != descriptor.entrypoint:
                    raise PluginRegistryError(
                        "capability entrypoint differs from implementation binding"
                    )
                expected_digest = definition.get("module_digest")
                if expected_digest != package_source_digest(descriptor.entrypoint):
                    raise PluginRegistryError(
                        "plugin implementation bytes differ from its binding"
                    )
                capability_bindings[name] = SealedCapabilityBinding(
                    descriptor=descriptor,
                    implementation_digest=expected_digest,
                )
        except (StorageError, ValueError, ImplementationError) as exc:
            _LOGGER.warning("plugin installation failed", exc_info=exc)
            raise PluginRegistryError(
                "The plugin manifest or one of its dependencies is invalid or "
                "unavailable. Check the reference contract and registered "
                "descriptors, then retry."
            ) from exc

        runtime_identity = _runtime_identity()
        build_identity_digest = _digest(
            {
                "plugin_manifest_digest": artifact.manifest.object_digest,
                "capabilities": {
                    name.value: binding.model_dump(mode="json")
                    for name, binding in capability_bindings.items()
                },
                "runtime_identity": runtime_identity.model_dump(mode="json"),
            }
        )
        snapshot = PluginRegistrySnapshot(
            plugin_id=plugin_id,
            plugin_manifest_digest=artifact.manifest.object_digest,
            domain_id=manifest.domain_id,
            domain_version=manifest.domain_version,
            claim_schema_uri=manifest.claim_schema_uri,
            candidate_schema_uri=manifest.candidate_schema_uri,
            capabilities=capability_bindings,
            runtime_identity=runtime_identity,
            build_identity_digest=build_identity_digest,
        )
        normalized_snapshot = self.schemas.validate(
            self.snapshot_schema_uri,
            snapshot.model_dump(mode="json"),
        )
        stored_snapshot = self.store.put(
            schema_uri=self.snapshot_schema_uri,
            semantics_uri=self.snapshot_semantics_uri,
            payload=normalized_snapshot,
            parents=tuple(
                dict.fromkeys(
                    (
                        plugin_id,
                        *(
                            binding.descriptor.implementation_uri
                            for binding in capability_bindings.values()
                        ),
                    )
                )
            ),
            summary="sealed plugin registry snapshot",
        )

        with self.store.connection() as connection:
            if not self.store.transaction_active:
                connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO installed_plugins (
                    plugin_id, domain_id, domain_version, registry_snapshot_uri
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    plugin_id,
                    manifest.domain_id,
                    manifest.domain_version,
                    stored_snapshot.artifact_uri,
                ),
            )
            connection.execute(
                """
                UPDATE installed_plugins
                SET registry_snapshot_uri = ?
                WHERE plugin_id = ? AND registry_snapshot_uri IS NULL
                """,
                (stored_snapshot.artifact_uri, plugin_id),
            )
            row = connection.execute(
                """
                SELECT domain_id, domain_version, registry_snapshot_uri
                FROM installed_plugins
                WHERE plugin_id = ?
                """,
                (plugin_id,),
            ).fetchone()
            if row is None or (
                row["domain_id"],
                row["domain_version"],
            ) != (manifest.domain_id, manifest.domain_version):
                raise PluginRegistryError("installed plugin metadata mismatch")
            if row["registry_snapshot_uri"] != stored_snapshot.artifact_uri:
                raise PluginRegistryError("installed plugin snapshot mismatch")
        return manifest

    def has_any_domain(
        self, domain_ids: frozenset[str] | set[str] | tuple[str, ...]
    ) -> bool:
        """Return whether any installed plugin matches one of ``domain_ids``."""

        if not domain_ids:
            return False
        ordered = tuple(sorted(domain_ids))
        placeholders = ", ".join("?" for _ in ordered)
        with self.store.connection() as connection:
            row = connection.execute(
                f"""
                SELECT 1 FROM installed_plugins
                WHERE domain_id IN ({placeholders})
                LIMIT 1
                """,
                ordered,
            ).fetchone()
        return row is not None

    def get(self, plugin_id: str) -> PluginManifest:
        """Return an installed manifest without resolving executable code."""

        with self.store.connection() as connection:
            installed = connection.execute(
                "SELECT 1 FROM installed_plugins WHERE plugin_id = ?",
                (plugin_id,),
            ).fetchone()
        if installed is None:
            raise PluginRegistryError(f"plugin is not installed: {plugin_id}")
        try:
            return PluginManifest.model_validate(self.store.get(plugin_id).payload)
        except (StorageError, ValueError) as exc:
            _LOGGER.warning("installed plugin manifest is unreadable", exc_info=exc)
            raise PluginRegistryError(
                "The installed plugin manifest is invalid or unavailable. Reload "
                "the plugin from its reference contract, then retry."
            ) from exc

    def snapshot_uri(self, plugin_id: str) -> str:
        """Return the immutable installation snapshot URI without importing code."""

        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT registry_snapshot_uri
                FROM installed_plugins
                WHERE plugin_id = ?
                """,
                (plugin_id,),
            ).fetchone()
        if row is None:
            raise PluginRegistryError(f"plugin is not installed: {plugin_id}")
        snapshot_uri = row["registry_snapshot_uri"]
        if not isinstance(snapshot_uri, str):
            raise PluginRegistryError(
                "installed plugin is missing its registry snapshot"
            )
        return snapshot_uri

    def snapshot(self, plugin_id: str) -> PluginRegistrySnapshot:
        """Read a sealed installation snapshot without resolving executable code."""

        try:
            artifact = self.store.get(self.snapshot_uri(plugin_id))
            if (
                artifact.manifest.schema_uri != self.snapshot_schema_uri
                or artifact.manifest.semantics_uri != self.snapshot_semantics_uri
            ):
                raise PluginRegistryError(
                    "installed plugin snapshot uses the wrong contract"
                )
            snapshot = PluginRegistrySnapshot.model_validate(artifact.payload)
            manifest_artifact = self.store.get(plugin_id)
        except (StorageError, ValueError) as exc:
            _LOGGER.warning("installed plugin snapshot is unreadable", exc_info=exc)
            raise PluginRegistryError(
                "The installed plugin snapshot is invalid or unavailable. Reload "
                "the plugin from its reference contract, then retry."
            ) from exc
        if snapshot.plugin_id != plugin_id:
            raise PluginRegistryError("registry snapshot plugin binding mismatch")
        if snapshot.plugin_manifest_digest != manifest_artifact.manifest.object_digest:
            raise PluginRegistryError("registry snapshot manifest digest mismatch")
        expected_parents = tuple(
            sorted(
                {
                    plugin_id,
                    *(
                        binding.descriptor.implementation_uri
                        for binding in snapshot.capabilities.values()
                    ),
                }
            )
        )
        if artifact.manifest.parents != expected_parents:
            raise PluginRegistryError("registry snapshot parent binding mismatch")
        return snapshot

    def resolve(
        self,
        plugin_id: str,
        capability: CapabilityName,
    ) -> ResolvedCapability:
        """Resolve a capability only if its source still matches installation."""

        manifest = self.get(plugin_id)
        snapshot = self.snapshot(plugin_id)
        registry_snapshot_uri = self.snapshot_uri(plugin_id)
        if snapshot.runtime_identity != _runtime_identity():
            raise PluginRegistryError(
                "plugin registry snapshot is incompatible with this runtime"
            )
        descriptor = manifest.capabilities.get(capability)
        if descriptor is None:
            raise PluginRegistryError(f"plugin does not implement {capability.value}")
        sealed_binding = snapshot.capabilities.get(capability)
        if sealed_binding is None or sealed_binding.descriptor != descriptor:
            raise PluginRegistryError(
                "capability differs from the sealed registry snapshot"
            )
        try:
            implementation = self.store.get_descriptor(
                descriptor.implementation_uri,
                expected_kind="implementation",
            )
            definition = implementation.get("definition")
            if not isinstance(definition, dict):
                raise PluginRegistryError(
                    "implementation descriptor has no object definition"
                )
            expected_digest = definition.get("module_digest")
            actual_digest = package_source_digest(descriptor.entrypoint)
            if definition.get("entrypoint") != descriptor.entrypoint:
                raise PluginRegistryError(
                    "capability entrypoint differs from implementation binding"
                )
            if (
                expected_digest != actual_digest
                or sealed_binding.implementation_digest != actual_digest
            ):
                raise PluginRegistryError(
                    "plugin implementation bytes changed after installation"
                )
        except (StorageError, ImplementationError) as exc:
            _LOGGER.warning("plugin resolution failed", exc_info=exc)
            raise PluginRegistryError(
                "The plugin implementation is unavailable. Reload Jacobian to "
                "register the current plugin version, then retry."
            ) from exc
        return ResolvedCapability(
            plugin_id=plugin_id,
            name=capability,
            descriptor=descriptor,
            implementation_digest=actual_digest,
            registry_snapshot_uri=registry_snapshot_uri,
        )


def _runtime_identity() -> PluginRuntimeIdentity:
    return PluginRuntimeIdentity(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_tag=sysconfig.get_platform(),
        system=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
    )


def _digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
