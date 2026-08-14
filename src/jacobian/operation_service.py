"""Operation service composition and operator policy.

The service is deliberately a small composition root. Registration, discovery,
dispatch, verification, validation, and telemetry each live in their owning
module so the public service keeps one stable typed entry point without hiding
those boundaries behind a second compatibility API.
"""

from __future__ import annotations

from typing import Any

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationRequest,
    OperationResult,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_discovery import discover_operations
from jacobian.operation_dispatch import dispatch_operation
from jacobian.operation_errors import OperationError
from jacobian.operation_validation import validator
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.storage.repository import ArtifactRepository


class OperationService:
    """Typed composition root for operation lifecycle and invocation."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        policy: OperationVisibilityPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or OperationVisibilityPolicy()
        self._adapters: dict[str, OperationAdapter[Any]] = {}
        self._descriptors: dict[str, OperationDescriptor] = {}

    def register(self, adapter: OperationAdapter[Any]) -> None:
        descriptor = adapter.descriptor
        if descriptor.operation_id in self._adapters:
            raise OperationError(f"duplicate operation ID: {descriptor.operation_id}")
        validator(descriptor.input_schema)
        validator(descriptor.output_schema)
        for example in descriptor.examples:
            try:
                from jacobian.operation_validation import validate_payload

                validate_payload(descriptor.input_schema, example.input)
            except OperationError as exc:
                raise OperationError(
                    f"operation {descriptor.operation_id} invocation example "
                    f"{example.name!r} does not match its input schema"
                ) from exc
        self._descriptors[descriptor.operation_id] = descriptor.model_copy(deep=True)
        self._adapters[descriptor.operation_id] = adapter

    def catalog(self) -> OperationCatalogSnapshot:
        projected = tuple(
            projected
            for name in sorted(self._adapters)
            if (projected := self.policy.project(self._descriptors[name])) is not None
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=projected,
        )

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        descriptor = self._descriptors.get(operation_id)
        if descriptor is None:
            return None
        return self.policy.project(descriptor)

    def discover(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.catalog(), request)

    def invoke(self, request: OperationRequest) -> OperationResult:
        return dispatch_operation(self, request)

    def _validate_verified_result(self, result: Any) -> None:
        from jacobian.operation_verification import validate_verified_result

        validate_verified_result(self.store, result)


__all__ = [
    "OperationService",
]
