"""Registration and catalog projection for installed capabilities."""

from __future__ import annotations

from typing import Any, cast

from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_errors import CapabilityError
from jacobian.capability_validation import validator
from jacobian.contracts.capabilities import (
    CapabilityCatalog,
    CapabilityDescriptor,
    CapabilityProviderAvailability,
)


class CapabilityRegistryMixin:
    """Own descriptor admission and deterministic catalog projection."""

    def register(self: Any, adapter: CapabilityAdapter[Any]) -> None:
        descriptor = adapter.descriptor
        if descriptor.capability_id in self._adapters:
            raise CapabilityError(
                f"duplicate capability ID: {descriptor.capability_id}"
            )
        if descriptor.provider_runtime is None:
            raise CapabilityError(
                f"capability {descriptor.capability_id} has no provider runtime identity"
            )
        if (
            descriptor.provider_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            raise CapabilityError(
                f"capability {descriptor.capability_id} is unavailable: "
                f"{descriptor.provider_runtime.diagnostic}"
            )
        validator(descriptor.input_schema)
        validator(descriptor.output_schema)
        for example in descriptor.invocation_examples:
            try:
                from jacobian.capability_validation import validate_payload

                validate_payload(descriptor.input_schema, example.input)
            except CapabilityError as exc:
                raise CapabilityError(
                    f"capability {descriptor.capability_id} invocation example "
                    f"{example.name!r} does not match its input schema"
                ) from exc
        self._descriptors[descriptor.capability_id] = descriptor.model_copy(deep=True)
        self._adapters[descriptor.capability_id] = adapter

    def catalog(self: Any) -> CapabilityCatalog:
        projected = tuple(
            projected
            for name in sorted(self._adapters)
            if (projected := self.policy.project(self._descriptors[name])) is not None
        )
        return CapabilityCatalog(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            capabilities=projected,
        )

    def inspect(self: Any, capability_id: str) -> CapabilityDescriptor | None:
        """Return one policy-visible descriptor without projecting the catalog."""

        descriptor = self._descriptors.get(capability_id)
        if descriptor is None:
            return None
        return cast(CapabilityDescriptor | None, self.policy.project(descriptor))


__all__ = ["CapabilityRegistryMixin"]
