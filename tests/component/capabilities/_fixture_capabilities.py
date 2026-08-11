from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.provider_runtime import source_provider_runtime


@dataclass(frozen=True)
class FixtureAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="fixture.increment",
        version="1",
        title="Increment an integer",
        description="External fixture adapter loaded through an operator entrypoint.",
        provider="tests.fixture",
        provider_runtime=source_provider_runtime(
            "tests.fixture",
            version="1",
            entrypoint="tests.component.capabilities._fixture_capabilities:create_adapter",
            install_tier=CapabilityInstallTier.T0,
            license_id="MIT",
            features=("integer-increment",),
        ),
        modes=(CapabilityMode.EXPLORE,),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        invocation_examples=(
            CapabilityInvocationExample(
                name="increment_41",
                description="Increment 41 to obtain 42.",
                mode=CapabilityMode.EXPLORE,
                input={"value": 41},
            ),
        ),
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": int(request.input["value"]) + 1},
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="fixture integer arithmetic",
            ),
        )


def create_adapter(_runtime: Any) -> FixtureAdapter:
    return FixtureAdapter()
