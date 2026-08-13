"""Registration and catalog projection for installed operations."""

from __future__ import annotations

from typing import Any, cast

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    ProviderAvailability,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_errors import OperationError
from jacobian.operation_validation import validator


class OperationRegistryMixin:
    """Own descriptor admission and deterministic catalog projection."""

    def register(self: Any, adapter: OperationAdapter[Any]) -> None:
        descriptor = adapter.descriptor
        if descriptor.operation_id in self._adapters:
            raise OperationError(
                f"duplicate operation ID: {descriptor.operation_id}"
            )
        if descriptor.provider_runtime is None:
            raise OperationError(
                f"operation {descriptor.operation_id} has no provider runtime identity"
            )
        if (
            descriptor.provider_runtime.availability
            is not ProviderAvailability.AVAILABLE
        ):
            raise OperationError(
                f"operation {descriptor.operation_id} is unavailable: "
                f"{descriptor.provider_runtime.diagnostic}"
            )
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

    def catalog(self: Any) -> OperationCatalogSnapshot:
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

    def inspect(self: Any, operation_id: str) -> OperationDescriptor | None:
        """Return one policy-visible descriptor without projecting the catalog."""

        descriptor = self._descriptors.get(operation_id)
        if descriptor is None:
            return None
        return cast(OperationDescriptor | None, self.policy.project(descriptor))


__all__ = ["OperationRegistryMixin"]
