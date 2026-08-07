"""Adapter implementations for sparse rational polynomial-map capabilities."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
)
from jacobian.contracts.polynomials import (
    PolynomialIdentityRequest,
    PolynomialInverseAnsatzSpecification,
    PolynomialInverseCoefficientEquation,
    PolynomialInverseSolverProvenance,
    PolynomialInverseSynthesisStatus,
    PolynomialMapCompositionResiduals,
    PolynomialMapInverseClaim,
    PolynomialMapInverseReplayPayload,
    PolynomialMapInverseSynthesisArtifact,
    PolynomialMapInverseSynthesisOutput,
    PolynomialMapInverseSynthesisRequest,
    PolynomialMapInverseVerifyOutput,
    PolynomialMapInverseVerifyRequest,
    RationalPolynomialMap,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
)
from jacobian.polynomials._support import (
    _inverse_candidate_map,
    _inverse_coefficient_system,
    _inverse_residual_term_bound,
    _inverse_supports,
    _map_inverse_residuals,
    _polynomial_error,
    _solve_inverse_system,
    _validate_request,
)
from jacobian.polynomials.identity import PolynomialIdentityAdapter
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.schema_registry import model_schema


class PolynomialMapInverseSynthesizeAdapter:
    """Solve one bounded exact polynomial inverse ansatz over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.inverse.candidate_synthesize",
            version="1",
            title="Synthesize a bounded polynomial-map inverse candidate",
            description=(
                "Solve an explicit finite polynomial inverse ansatz over QQ, "
                "then submit every found candidate to the independent two-sided "
                "inverse verifier."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("polynomial-map-inverse-ansatz", "exact-equation-solving"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialMapInverseSynthesisRequest),
            output_schema=model_schema(PolynomialMapInverseSynthesisOutput),
            tags=("polynomial", "map", "inverse", "synthesis", "exact-rational"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="triangular_inverse",
                    description=(
                        "Synthesize and independently check the degree-two "
                        "inverse of (x + y^2, y)."
                    ),
                    mode=CapabilityMode.EXPLORE,
                    input=PolynomialMapInverseSynthesisRequest.model_validate(
                        {
                            "forward_map": {
                                "variables": ["x", "y"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1, 0],
                                            },
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [0, 2],
                                            },
                                        ]
                                    },
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [0, 1],
                                            }
                                        ]
                                    },
                                ],
                            },
                            "source_variables": ["x", "y"],
                            "target_variables": ["u", "v"],
                            "inverse_degree_bound": 2,
                            "support_mode": "FULL_TOTAL_DEGREE",
                            "solver": "sympy.solve",
                            "limits": {
                                "timeout_ms": 10000,
                                "max_inverse_degree": 4,
                                "max_composition_degree": 32,
                                "max_unknown_coefficients": 64,
                                "max_coefficient_equations": 512,
                                "max_residual_terms": 1024,
                            },
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def _run_synthesis(
        self,
        validated: PolynomialMapInverseSynthesisRequest,
        supports: tuple[tuple[tuple[int, ...], ...], ...],
        coefficient_names: tuple[tuple[str, ...], ...],
        unknown_names: tuple[str, ...],
        started: float,
    ) -> tuple[
        PolynomialInverseSynthesisStatus,
        tuple[PolynomialInverseCoefficientEquation, ...],
        RationalPolynomialMap | None,
        tuple[SparseRationalPolynomial, ...],
        tuple[SparseRationalPolynomial, ...],
        dict[str, Any] | None,
        str | None,
        str | None,
        int,
    ]:
        """Orchestrate the bounded coefficient-system solve and return outcomes."""

        equations: tuple[PolynomialInverseCoefficientEquation, ...] = ()
        candidate: RationalPolynomialMap | None = None
        left_residuals: tuple[SparseRationalPolynomial, ...] = ()
        right_residuals: tuple[SparseRationalPolynomial, ...] = ()
        verification_output: dict[str, Any] | None = None
        verification_artifact_uri: str | None = None
        verification_failure: str | None = None
        residual_term_count = 0

        if validated.solver != "sympy.solve":
            status = PolynomialInverseSynthesisStatus.UNSUPPORTED
            verification_failure = (
                f"solver {validated.solver!r} is unsupported; use 'sympy.solve'"
            )
        elif validated.limits.timeout_ms == 0:
            status = PolynomialInverseSynthesisStatus.TIMEOUT
        elif len(unknown_names) > validated.limits.max_unknown_coefficients:
            status = PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED
            verification_failure = "ansatz unknown count exceeds the declared limit"
        else:
            (
                pre_status,
                pre_failure,
                equations,
                residual_term_count,
                solve_status,
                solution,
                ansatz_expressions,
                unknown_symbols,
            ) = self._solve_coefficient_system(
                validated, supports, coefficient_names, started
            )
            if pre_status is not None:
                status = pre_status
                verification_failure = pre_failure
            else:
                (
                    status,
                    verification_failure,
                    candidate,
                    left_residuals,
                    right_residuals,
                    verification_output,
                    verification_artifact_uri,
                ) = self._classify_solution(
                    validated,
                    ansatz_expressions,
                    unknown_symbols,
                    solution,
                    solve_status,
                )

        return (
            status,
            equations,
            candidate,
            left_residuals,
            right_residuals,
            verification_output,
            verification_artifact_uri,
            verification_failure,
            residual_term_count,
        )

    def _solve_coefficient_system(
        self,
        validated: PolynomialMapInverseSynthesisRequest,
        supports: tuple[tuple[tuple[int, ...], ...], ...],
        coefficient_names: tuple[tuple[str, ...], ...],
        started: float,
    ) -> tuple[
        PolynomialInverseSynthesisStatus | None,
        str | None,
        tuple[PolynomialInverseCoefficientEquation, ...],
        int,
        str | None,
        dict[str, Any] | None,
        tuple[Any, ...],
        tuple[Any, ...],
    ]:
        """Derive and solve the coefficient system.

        Returns ``(status, failure, equations, residual_term_count,
        solve_status, solution, ansatz_expressions, unknown_symbols)``.
        A non-None *status* means the solve was short-circuited.
        """

        forward_degree = max(
            (
                sum(term.exponents)
                for coordinate in validated.forward_map.coordinates
                for term in coordinate.terms
            ),
            default=0,
        )
        residual_term_bound = _inverse_residual_term_bound(validated, supports)
        precheck_exhausted = (
            forward_degree * validated.inverse_degree_bound
            > validated.limits.max_composition_degree
            or residual_term_bound > validated.limits.max_residual_terms
        )

        ansatz_expressions: tuple[Any, ...] = ()
        unknown_symbols: tuple[Any, ...] = ()
        equations: tuple[PolynomialInverseCoefficientEquation, ...] = ()
        residual_term_count = 0

        if precheck_exhausted:
            return (
                PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED,
                (
                    "conservative composition degree or residual-term bound "
                    "exceeds a declared limit"
                ),
                equations,
                residual_term_count,
                None,
                None,
                ansatz_expressions,
                unknown_symbols,
            )

        (
            ansatz_expressions,
            unknown_symbols,
            equations,
            residual_term_count,
        ) = _inverse_coefficient_system(
            validated,
            supports,
            coefficient_names,
        )
        if (
            residual_term_count > validated.limits.max_residual_terms
            or len(equations) > validated.limits.max_coefficient_equations
        ):
            return (
                PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED,
                (
                    "derived coefficient system exceeds a declared residual "
                    "or equation limit"
                ),
                equations,
                residual_term_count,
                None,
                None,
                ansatz_expressions,
                unknown_symbols,
            )
        if not equations:
            return (
                PolynomialInverseSynthesisStatus.UNDERDETERMINED,
                "the ansatz produced no coefficient equations",
                equations,
                residual_term_count,
                None,
                None,
                ansatz_expressions,
                unknown_symbols,
            )

        remaining_ms = validated.limits.timeout_ms - int(
            (time.monotonic() - started) * 1000
        )
        if remaining_ms <= 0:
            solve_status, solution = "TIMEOUT", None
        else:
            solve_status, solution = _solve_inverse_system(
                equations,
                tuple(name for row in coefficient_names for name in row),
                timeout_ms=remaining_ms,
            )
        return (
            None,
            None,
            equations,
            residual_term_count,
            solve_status,
            solution,
            ansatz_expressions,
            unknown_symbols,
        )

    def _classify_solution(
        self,
        validated: PolynomialMapInverseSynthesisRequest,
        ansatz_expressions: tuple[Any, ...],
        unknown_symbols: tuple[Any, ...],
        solution: dict[str, Any] | None,
        solve_status: str | None,
    ) -> tuple[
        PolynomialInverseSynthesisStatus,
        str | None,
        RationalPolynomialMap | None,
        tuple[SparseRationalPolynomial, ...],
        tuple[SparseRationalPolynomial, ...],
        dict[str, Any] | None,
        str | None,
    ]:
        """Classify the solver result and run independent verification."""

        candidate: RationalPolynomialMap | None = None
        left_residuals: tuple[SparseRationalPolynomial, ...] = ()
        right_residuals: tuple[SparseRationalPolynomial, ...] = ()
        verification_output: dict[str, Any] | None = None
        verification_artifact_uri: str | None = None
        verification_failure: str | None = None

        if solve_status == "TIMEOUT":
            status = PolynomialInverseSynthesisStatus.TIMEOUT
        elif solve_status == "ERROR":
            status = PolynomialInverseSynthesisStatus.UNSUPPORTED
            verification_failure = "the configured exact solver failed"
        elif solution is None:
            status = PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ
        elif any(value.free_symbols for value in solution.values()) or set(
            solution
        ) != set(unknown_symbols):
            status = PolynomialInverseSynthesisStatus.UNDERDETERMINED
            verification_failure = "the coefficient equations have free parameters"
        elif not all(value.is_Rational for value in solution.values()):
            status = PolynomialInverseSynthesisStatus.UNSUPPORTED
            verification_failure = "the selected solution is not rational over QQ"
        else:
            candidate = _inverse_candidate_map(
                validated,
                ansatz_expressions,
                solution,
            )
            verify_request = PolynomialMapInverseVerifyRequest(
                forward_map=validated.forward_map,
                inverse_map=candidate,
                source_variables=validated.source_variables,
                target_variables=validated.target_variables,
            )
            left_residuals, right_residuals = _map_inverse_residuals(verify_request)
            status = PolynomialInverseSynthesisStatus.FOUND
            try:
                verified = PolynomialMapInverseVerifyAdapter(self.resources).invoke(
                    CapabilityRequest(
                        capability_id="polynomial.map.inverse.verify",
                        mode=CapabilityMode.VERIFY,
                        input=verify_request.model_dump(mode="json"),
                    )
                )
                verification_output = verified.output
                artifact = verified.output.get(
                    "verification_record_uri"
                ) or verified.output.get("certificate_uri")
                verification_artifact_uri = (
                    artifact if isinstance(artifact, str) else None
                )
                if verified.output.get("inverse_verified") is not True:
                    verification_failure = (
                        "the independent two-sided verifier rejected "
                        "the synthesized candidate"
                    )
            except CapabilityInvocationError as exc:
                verification_failure = exc.diagnostic.message

        return (
            status,
            verification_failure,
            candidate,
            left_residuals,
            right_residuals,
            verification_output,
            verification_artifact_uri,
        )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        validated = _validate_request(
            PolynomialMapInverseSynthesisRequest,
            request.input,
            code="INVALID_POLYNOMIAL_MAP_INVERSE_SYNTHESIS_REQUEST",
            operation="map inverse candidate synthesis",
        )
        forward_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=self.resources.installation.inverse_semantics_uri,
            payload=validated.forward_map.model_dump(mode="json"),
            summary="forward map for bounded inverse synthesis",
        )
        supports = _inverse_supports(validated)
        coefficient_names = tuple(
            tuple(f"c_{coordinate}_{index}" for index in range(len(support)))
            for coordinate, support in enumerate(supports)
        )
        ansatz = PolynomialInverseAnsatzSpecification(
            support_mode=validated.support_mode,
            inverse_degree_bound=validated.inverse_degree_bound,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            coordinate_supports=supports,
            coefficient_symbols=coefficient_names,
        )
        unknown_names = tuple(name for row in coefficient_names for name in row)

        (
            status,
            equations,
            candidate,
            left_residuals,
            right_residuals,
            verification_output,
            verification_artifact_uri,
            verification_failure,
            residual_term_count,
        ) = self._run_synthesis(
            validated, supports, coefficient_names, unknown_names, started
        )

        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        provenance = PolynomialInverseSolverProvenance(
            solver=validated.solver,
            backend_version=SYMPY_VERSION,
            timeout_ms=validated.limits.timeout_ms,
            unknown_count=len(unknown_names),
            equation_count=len(equations),
            residual_term_count=residual_term_count,
            elapsed_ms=elapsed_ms,
        )
        payload = PolynomialMapInverseSynthesisArtifact(
            status=status,
            forward_map=validated.forward_map,
            ansatz=ansatz,
            coefficient_equations=equations,
            solver_provenance=provenance,
            candidate_inverse_map=candidate,
            inverse_after_forward=left_residuals,
            forward_after_inverse=right_residuals,
            verification_output=verification_output,
            verification_artifact_uri=verification_artifact_uri,
            verification_failure=verification_failure,
        )
        parents = tuple(
            dict.fromkeys(
                (
                    forward_artifact.artifact_uri,
                    *(
                        (verification_artifact_uri,)
                        if verification_artifact_uri
                        else ()
                    ),
                )
            )
        )
        synthesis = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_synthesis_schema_uri,
            semantics_uri=self.resources.installation.inverse_semantics_uri,
            payload=payload.model_dump(mode="json"),
            parents=parents,
            summary=f"bounded polynomial inverse synthesis: {status.value}",
        )
        output = PolynomialMapInverseSynthesisOutput(
            **payload.model_dump(mode="python"),
            synthesis_uri=synthesis.artifact_uri,
            forward_map_uri=forward_artifact.artifact_uri,
        )
        artifact_uris = tuple(
            dict.fromkeys(
                (
                    forward_artifact.artifact_uri,
                    synthesis.artifact_uri,
                    *(
                        (verification_artifact_uri,)
                        if verification_artifact_uri
                        else ()
                    ),
                )
            )
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=(
                    ExecutionStatus.TIMEOUT
                    if status is PolynomialInverseSynthesisStatus.TIMEOUT
                    else ExecutionStatus.COMPLETED
                ),
                runtime_ms=elapsed_ms,
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one finite polynomial inverse coefficient ansatz over QQ",
                parameters={
                    "inverse_degree_bound": validated.inverse_degree_bound,
                    "support_mode": validated.support_mode.value,
                    "unknown_count": len(unknown_names),
                },
                artifact_uri=synthesis.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if status
                    in {
                        PolynomialInverseSynthesisStatus.FOUND,
                        PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ,
                        PolynomialInverseSynthesisStatus.UNDERDETERMINED,
                    }
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=(
                    "the declared finite ansatz and exact coefficient system were solved"
                    if status
                    in {
                        PolynomialInverseSynthesisStatus.FOUND,
                        PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ,
                        PolynomialInverseSynthesisStatus.UNDERDETERMINED,
                    }
                    else "the declared synthesis operation did not complete"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "bounded exact symbolic synthesis; no failure status proves "
                    "noninvertibility and only the separate verifier may certify "
                    "a found candidate"
                ),
            ),
            artifact_uris=artifact_uris,
        )


class PolynomialMapInverseVerifyAdapter:
    """Verify both exact compositions of two square polynomial maps."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.inverse_checker_id
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.inverse.verify",
            version="1",
            title="Verify a two-sided polynomial-map inverse",
            description=(
                "Independently replay both ordered polynomial-map compositions "
                "over QQ and accept an inverse only when both are identity."
            ),
            provider="jacobian.sparse-polynomial-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.sparse-polynomial-checker",
                features=("polynomial-map-composition", "two-sided-inverse"),
                checker_ids=((checker_id,) if checker_id is not None else ()),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(PolynomialMapInverseVerifyRequest),
            output_schema=model_schema(PolynomialMapInverseVerifyOutput),
            tags=("polynomial", "map", "inverse", "verification", "exact-rational"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="identity_map_inverse",
                    description=(
                        "Independently verify the identity map as its own "
                        "two-sided inverse over QQ."
                    ),
                    mode=CapabilityMode.VERIFY,
                    input=PolynomialMapInverseVerifyRequest.model_validate(
                        {
                            "forward_map": {
                                "variables": ["x"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1],
                                            }
                                        ]
                                    }
                                ],
                            },
                            "inverse_map": {
                                "variables": ["u"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1],
                                            }
                                        ]
                                    }
                                ],
                            },
                            "source_variables": ["x"],
                            "target_variables": ["u"],
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialMapInverseVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_MAP_INVERSE_REQUEST",
            operation="map inverse verification",
        )
        checker_id = self.resources.installation.inverse_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_MAP_INVERSE_CHECKER_UNAVAILABLE",
                "inverse_verification",
                "No authorized polynomial-map inverse checker is installed.",
            )
        semantics_uri = self.resources.installation.inverse_semantics_uri
        forward = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=semantics_uri,
            payload=validated.forward_map.model_dump(mode="json"),
            summary="forward sparse rational polynomial map",
        )
        inverse = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=semantics_uri,
            payload=validated.inverse_map.model_dump(mode="json"),
            summary="candidate inverse sparse rational polynomial map",
        )
        left_residuals, right_residuals = _map_inverse_residuals(validated)
        identity_adapter = PolynomialIdentityAdapter(self.resources)
        zero = SparseRationalPolynomial()
        left_records: list[str] = []
        right_records: list[str] = []
        identity_artifacts: list[str] = []
        for variables, residuals, records in (
            (validated.source_variables, left_residuals, left_records),
            (validated.target_variables, right_residuals, right_records),
        ):
            for residual in residuals:
                checked = identity_adapter.invoke(
                    CapabilityRequest(
                        capability_id="polynomial.identity.verify",
                        mode=CapabilityMode.VERIFY,
                        input=PolynomialIdentityRequest(
                            variables=variables,
                            left=residual,
                            right=zero,
                        ).model_dump(mode="json"),
                    )
                )
                record_uri = checked.output.get("verification_record_uri")
                if not isinstance(record_uri, str):
                    raise _polynomial_error(
                        "POLYNOMIAL_IDENTITY_REPLAY_UNAVAILABLE",
                        "composition_identity_replay",
                        "A composition residual could not obtain a checker record.",
                    )
                records.append(record_uri)
                identity_artifacts.extend(checked.artifact_uris)
        residual_payload = PolynomialMapCompositionResiduals(
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            inverse_after_forward=left_residuals,
            forward_after_inverse=right_residuals,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
        )
        residual_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_residual_schema_uri,
            semantics_uri=semantics_uri,
            payload=residual_payload.model_dump(mode="json"),
            parents=tuple(
                dict.fromkeys(
                    (
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        *left_records,
                        *right_records,
                    )
                )
            ),
            summary="both exact polynomial-map composition residual families",
        )
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_claim_schema_uri,
            semantics_uri=semantics_uri,
            payload=PolynomialMapInverseClaim(
                forward_map_uri=forward.artifact_uri,
                inverse_map_uri=inverse.artifact_uri,
                source_variables=validated.source_variables,
                target_variables=validated.target_variables,
            ).model_dump(mode="json"),
            parents=(forward.artifact_uri, inverse.artifact_uri),
            summary="two-sided polynomial-map inverse claim",
        )
        semantics = self.resources.store.get(semantics_uri)
        replay = PolynomialMapInverseReplayPayload(
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            residuals_uri=residual_artifact.artifact_uri,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
        )
        replay_payload = replay.model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.map.inverse.two_sided_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=residual_artifact.object_digest,
                scope_digest=forward.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(replay_payload)).hexdigest()
            ),
            payload=replay_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.certificate_schema_uri,
            semantics_uri=semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=tuple(
                dict.fromkeys(
                    (
                        claim.artifact_uri,
                        residual_artifact.artifact_uri,
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        *left_records,
                        *right_records,
                    )
                )
            ),
            summary="two-sided exact polynomial-map inverse replay certificate",
        )
        supporting = (inverse.artifact_uri, *left_records, *right_records)
        aggregate_checked = self.resources.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
            supporting_artifact_uris=supporting,
        )
        verified = aggregate_checked.verification_record_uri is not None
        conclusion = aggregate_checked.conclusion
        output = PolynomialMapInverseVerifyOutput(
            inverse_verified={
                Conclusion.TRUE: True,
                Conclusion.FALSE: False,
                Conclusion.UNKNOWN: None,
            }[conclusion],
            conclusion=conclusion,
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            residuals_uri=residual_artifact.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
            verification_record_uri=aggregate_checked.verification_record_uri,
            checker_id=checker_id,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=aggregate_checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="both ordered polynomial compositions over QQ",
                parameters={
                    "source_variables": list(validated.source_variables),
                    "target_variables": list(validated.target_variables),
                },
                artifact_uri=forward.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if verified
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "both complete composition residual families were independently replayed"
                    if verified
                    else "the aggregate independent checker did not accept the replay"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.COMPUTED
                ),
                verification_record_uri=aggregate_checked.verification_record_uri,
            ),
            relationships=(),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted by the authorized independent two-sided map checker"
                    if verified
                    else "the independent checker did not accept the inverse request"
                ),
                verification_record_uri=aggregate_checked.verification_record_uri,
            ),
            artifact_uris=tuple(
                dict.fromkeys(
                    (
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        residual_artifact.artifact_uri,
                        claim.artifact_uri,
                        certificate_artifact.artifact_uri,
                        *identity_artifacts,
                        *(
                            (aggregate_checked.verification_record_uri,)
                            if aggregate_checked.verification_record_uri is not None
                            else ()
                        ),
                    )
                )
            ),
        )
