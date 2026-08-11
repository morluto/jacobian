"""Bounded exact rational search for polynomial-system solutions."""

from __future__ import annotations

import time
from itertools import product

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import format_canonical_integer
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.exact import CanonicalRational, bounded_rational_scalars
from jacobian.contracts.polynomial_systems import (
    PolynomialSystemRationalSearchOutput,
    PolynomialSystemRationalSearchRequest,
    PolynomialSystemSolutionRequest,
    RationalPolynomialAssignment,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.polynomial_system_capabilities import (
    PolynomialSystemInstallation,
    _evaluate_request,
)
from jacobian.provider_runtime import known_provider_runtime


class PolynomialSystemRationalSearchAdapter:
    def __init__(
        self, artifacts: ArtifactService, installation: PolynomialSystemInstallation
    ) -> None:
        self.artifacts, self.installation = artifacts, installation
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.system.rational_solution.search",
            version="1",
            title="Search a bounded rational grid for a polynomial-system solution",
            description="Return the first exact satisfying assignment in one declared finite grid.",
            provider="jacobian.exact-polynomial-system-search",
            provider_runtime=known_provider_runtime(
                "jacobian.exact-polynomial-system-search",
                features=("bounded-rational-enumeration",),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=PolynomialSystemRationalSearchRequest.model_json_schema(),
            output_schema=PolynomialSystemRationalSearchOutput.model_json_schema(),
            tags=("polynomial", "system", "solution", "bounded-search"),
            invocation_examples=(
                example(
                    "zero_system",
                    "Find zero for x=0 on the smallest grid.",
                    {
                        "system": {
                            "variables": ["x"],
                            "equations": [
                                {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                }
                            ],
                        },
                        "max_abs_numerator": 0,
                        "max_denominator": 1,
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = PolynomialSystemRationalSearchRequest.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_POLYNOMIAL_SYSTEM_SEARCH_REQUEST",
                    stage="request_validation",
                    message="The bounded rational solution search request is invalid.",
                )
            ) from exc
        started = time.monotonic()
        values = tuple(
            CanonicalRational(
                num=format_canonical_integer(value.numerator),
                den=format_canonical_integer(value.denominator),
            )
            for value in bounded_rational_scalars(
                validated.max_abs_numerator, validated.max_denominator
            )
        )
        grid_assignment_count = len(values) ** len(validated.system.variables)
        system = self.artifacts.put(
            schema_uri=self.installation.system_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=validated.system.model_dump(mode="json"),
            summary="bounded-search rational polynomial system",
        )
        assignment_uri = None
        assignment = None
        examined = 0
        for candidate in product(values, repeat=len(validated.system.variables)):
            examined += 1
            residuals, inequations = _evaluate_request(
                PolynomialSystemSolutionRequest(
                    system=validated.system, assignment=candidate
                )
            )
            if all(value.as_fraction() == 0 for value in residuals) and all(
                value.as_fraction() != 0 for value in inequations
            ):
                assignment = candidate
                assignment_uri = self.artifacts.put(
                    schema_uri=self.installation.assignment_schema_uri,
                    semantics_uri=self.installation.semantics_uri,
                    payload=RationalPolynomialAssignment(values=candidate).model_dump(
                        mode="json"
                    ),
                    parents=(system.artifact_uri,),
                    summary="unverified bounded-search rational solution candidate",
                ).artifact_uri
                break
        output = PolynomialSystemRationalSearchOutput(
            found=assignment is not None,
            system_uri=system.artifact_uri,
            assignment_uri=assignment_uri,
            assignment=assignment,
            examined_assignment_count=examined,
            grid_assignment_count=grid_assignment_count,
            checker_id=self.installation.checker_id,
        )
        uris = (system.artifact_uri,) + (
            (assignment_uri,) if assignment_uri is not None else ()
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one complete declared finite rational grid",
                parameters={
                    "grid_assignment_count": grid_assignment_count,
                    "examined": examined,
                },
                artifact_uri=system.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="the search stopped at its first match or exhausted the declared grid",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic exact rational enumeration; candidate remains unverified",
            ),
            artifact_uris=uris,
        )
