"""Shared helpers for polynomial-map capability adapters."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Iterator
from math import prod as multiply
from queue import Empty
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomials import (
    PolynomialInverseCoefficientEquation,
    PolynomialInverseSupportMode,
    PolynomialMapEvaluation,
    PolynomialMapInverseSynthesisRequest,
    PolynomialMapInverseVerifyRequest,
    RationalPolynomialMap,
    RationalPolynomialPoint,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import (
    ContractModel,
    Execution,
    ExecutionStatus,
)
from jacobian.polynomials._sympy import _sympy
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import SYMPY_VERSION
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact

if TYPE_CHECKING:
    from sympy import Poly

_INVERSE_SOLVER_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def _materialize_map(
    resources: PolynomialResources,
    polynomial_map: RationalPolynomialMap,
) -> tuple[RationalPolynomialMap, str]:
    artifact = resources.artifacts.put(
        schema_uri=resources.installation.map_schema_uri,
        semantics_uri=resources.installation.semantics_uri,
        payload=polynomial_map.model_dump(mode="json"),
        summary="exact sparse rational polynomial map",
    )
    return polynomial_map, artifact.artifact_uri


def _load_evaluation(
    resources: PolynomialResources,
    evaluation_uri: str,
    *,
    path: str,
) -> tuple[PolynomialMapEvaluation, StoredArtifact]:
    try:
        artifact = resources.store.get(evaluation_uri)
    except StorageError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="POLYNOMIAL_EVALUATION_ARTIFACT_NOT_FOUND",
                stage="evaluation_resolution",
                message="The requested polynomial evaluation artifact is unavailable.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Use an evaluation URI returned by polynomial.map.evaluate.",
            )
        ) from exc
    if (
        artifact.manifest.schema_uri != resources.installation.evaluation_schema_uri
        or artifact.manifest.semantics_uri != resources.installation.semantics_uri
        or not isinstance(artifact.payload, dict)
    ):
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The artifact is not a compatible polynomial-map evaluation.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Use an evaluation URI returned by polynomial.map.evaluate.",
            )
        )
    try:
        evaluation = PolynomialMapEvaluation.model_validate(artifact.payload)
    except ValidationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The polynomial-map evaluation artifact payload is malformed.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Recreate the artifact through polynomial.map.evaluate.",
            )
        ) from exc
    if evaluation.map_uri not in artifact.manifest.parents:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="MISBOUND_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The evaluation artifact is not bound to its declared map.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Recreate the artifact through polynomial.map.evaluate.",
            )
        )
    return evaluation, artifact


def _load_polynomial_map(
    resources: PolynomialResources,
    map_uri: str,
) -> tuple[RationalPolynomialMap, StoredArtifact]:
    try:
        artifact = resources.store.get(map_uri)
    except StorageError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="POLYNOMIAL_MAP_ARTIFACT_NOT_FOUND",
                stage="map_resolution",
                message="The polynomial map referenced by an evaluation is unavailable.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        ) from exc
    if (
        artifact.manifest.schema_uri != resources.installation.map_schema_uri
        or artifact.manifest.semantics_uri != resources.installation.semantics_uri
        or not isinstance(artifact.payload, dict)
    ):
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_MAP_ARTIFACT",
                stage="map_validation",
                message="An evaluation references an incompatible polynomial map.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        )
    try:
        polynomial_map = RationalPolynomialMap.model_validate(artifact.payload)
    except ValidationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_MAP_ARTIFACT",
                stage="map_validation",
                message="The referenced polynomial map artifact payload is malformed.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        ) from exc
    return polynomial_map, artifact


def _inverse_solver_worker(
    equation_texts: tuple[str, ...],
    unknown_names: tuple[str, ...],
    result_queue: Any,
) -> None:
    """Solve one exact coefficient system in an isolated process."""

    try:
        sp = _sympy.get()
        unknowns = sp.symbols(" ".join(unknown_names), seq=True)
        locals_by_name = dict(zip(unknown_names, unknowns, strict=True))
        equations = tuple(
            sp.sympify(expression, locals=locals_by_name)
            for expression in equation_texts
        )
        solutions = sp.solve(equations, unknowns, dict=True, simplify=False)
        serialized = tuple(
            tuple(
                (name, str(solution.get(symbol, symbol)))
                for name, symbol in zip(
                    unknown_names,
                    unknowns,
                    strict=True,
                )
            )
            for solution in solutions
        )
        result_queue.put(("OK", tuple(sorted(serialized))))
    except Exception as exc:  # pragma: no cover - defensive child boundary
        result_queue.put(("ERROR", type(exc).__name__))


def _inverse_supports(
    request: PolynomialMapInverseSynthesisRequest,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if request.support_mode is PolynomialInverseSupportMode.EXPLICIT:
        assert request.explicit_support is not None
        return request.explicit_support
    dimension = len(request.target_variables)
    support = tuple(
        _reverse_lex_degree_bounded_exponents(
            dimension=dimension,
            degree_bound=request.inverse_degree_bound,
        )
    )
    return tuple(support for _ in range(dimension))


def _reverse_lex_degree_bounded_exponents(
    *,
    dimension: int,
    degree_bound: int,
) -> Iterator[tuple[int, ...]]:
    if dimension == 1:
        for exponent in range(degree_bound, -1, -1):
            yield (exponent,)
        return
    for exponent in range(degree_bound, -1, -1):
        for suffix in _reverse_lex_degree_bounded_exponents(
            dimension=dimension - 1,
            degree_bound=degree_bound - exponent,
        ):
            yield (exponent, *suffix)


def _inverse_residual_term_bound(
    request: PolynomialMapInverseSynthesisRequest,
    supports: tuple[tuple[tuple[int, ...], ...], ...],
) -> int:
    forward_counts = tuple(
        len(coordinate.terms) for coordinate in request.forward_map.coordinates
    )
    support_counts = tuple(len(support) for support in supports)
    left = sum(
        multiply(
            count**exponent
            for count, exponent in zip(
                forward_counts,
                monomial,
                strict=True,
            )
        )
        for support in supports
        for monomial in support
    )
    right = sum(
        multiply(
            count**exponent
            for count, exponent in zip(
                support_counts,
                term.exponents,
                strict=True,
            )
        )
        for coordinate in request.forward_map.coordinates
        for term in coordinate.terms
    )
    return int(left + right + 2 * len(request.source_variables))


def _inverse_coefficient_system(
    request: PolynomialMapInverseSynthesisRequest,
    supports: tuple[tuple[tuple[int, ...], ...], ...],
    coefficient_names: tuple[tuple[str, ...], ...],
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[PolynomialInverseCoefficientEquation, ...],
    int,
]:
    source_generators, forward = _sympy_map(request.forward_map)
    sp = _sympy.get()
    target_generators = tuple(sp.symbols(request.target_variables))
    flat_names = tuple(name for row in coefficient_names for name in row)
    unknowns = tuple(sp.symbols(" ".join(flat_names), seq=True))
    unknown_by_name = dict(zip(flat_names, unknowns, strict=True))
    ansatz = tuple(
        sp.expand(
            sum(
                unknown_by_name[name]
                * multiply(
                    generator**exponent
                    for generator, exponent in zip(
                        target_generators,
                        exponents,
                        strict=True,
                    )
                )
                for name, exponents in zip(names, support, strict=True)
            )
        )
        for names, support in zip(coefficient_names, supports, strict=True)
    )
    left_substitutions = {
        generator: polynomial.as_expr()
        for generator, polynomial in zip(
            target_generators,
            forward,
            strict=True,
        )
    }
    left = tuple(
        sp.expand(
            expression.subs(left_substitutions, simultaneous=True)
            - source_generators[index]
        )
        for index, expression in enumerate(ansatz)
    )
    right_substitutions = dict(zip(source_generators, ansatz, strict=True))
    right = tuple(
        sp.expand(
            polynomial.as_expr().subs(right_substitutions, simultaneous=True)
            - target_generators[index]
        )
        for index, polynomial in enumerate(forward)
    )
    records: list[PolynomialInverseCoefficientEquation] = []
    residual_term_count = 0
    for direction, generators, residuals in (
        ("INVERSE_AFTER_FORWARD", source_generators, left),
        ("FORWARD_AFTER_INVERSE", target_generators, right),
    ):
        for coordinate, residual in enumerate(residuals):
            polynomial = sp.Poly(residual, *generators)
            terms = polynomial.terms()
            residual_term_count += len(terms)
            records.extend(
                PolynomialInverseCoefficientEquation(
                    direction=cast(Any, direction),
                    coordinate=coordinate,
                    monomial_exponents=monomial,
                    expression=str(coefficient),
                )
                for monomial, coefficient in terms
                if coefficient != 0
            )
    return ansatz, unknowns, tuple(records), residual_term_count


def _solve_inverse_system(
    equations: tuple[PolynomialInverseCoefficientEquation, ...],
    unknown_names: tuple[str, ...],
    *,
    timeout_ms: int,
) -> tuple[str, dict[Any, Any] | None]:
    # Spawn is used (not fork) because the Jacobian runtime is multi-threaded.
    # fork() in a multi-threaded process can deadlock and is deprecated in
    # Python 3.14+.  The spawn cost (~30ms plus SymPy re-import) is accepted
    # as the price of process isolation for this unbounded solver.
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_inverse_solver_worker,
        args=(
            tuple(item.expression for item in equations),
            unknown_names,
            result_queue,
        ),
    )
    process.start()
    process.join(timeout_ms / 1000)
    if process.is_alive():
        process.terminate()
        process.join(timeout=_INVERSE_SOLVER_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=_INVERSE_SOLVER_SHUTDOWN_TIMEOUT_SECONDS)
        return "TIMEOUT", None
    try:
        status, raw = result_queue.get_nowait()
    except Empty:
        return "ERROR", None
    if status != "OK":
        return "ERROR", None
    if not raw:
        return "OK", None
    sp = _sympy.get()
    symbols_by_name = dict(
        zip(unknown_names, sp.symbols(" ".join(unknown_names), seq=True), strict=True)
    )
    first = raw[0]
    solution = {
        symbols_by_name[name]: sp.sympify(value, locals=symbols_by_name)
        for name, value in first
    }
    return "OK", solution


def _inverse_candidate_map(
    request: PolynomialMapInverseSynthesisRequest,
    ansatz: tuple[Any, ...],
    solution: dict[Any, Any],
) -> RationalPolynomialMap:
    sp = _sympy.get()
    target_generators = tuple(sp.symbols(request.target_variables))
    return RationalPolynomialMap(
        variables=request.target_variables,
        coordinates=tuple(
            _wire_polynomial(
                sp.Poly(
                    sp.expand(expression.subs(solution, simultaneous=True)),
                    *target_generators,
                    domain=sp.QQ,
                )
            )
            for expression in ansatz
        ),
    )


def _map_inverse_residuals(
    request: PolynomialMapInverseVerifyRequest,
) -> tuple[
    tuple[SparseRationalPolynomial, ...],
    tuple[SparseRationalPolynomial, ...],
]:
    source_generators, forward = _sympy_map(request.forward_map)
    target_generators, inverse = _sympy_map(request.inverse_map)
    sp = _sympy.get()

    def compose_residuals(
        outer: tuple[Poly, ...],
        outer_generators: tuple[Any, ...],
        inner: tuple[Poly, ...],
        result_generators: tuple[Any, ...],
    ) -> tuple[SparseRationalPolynomial, ...]:
        substitutions = {
            generator: polynomial.as_expr()
            for generator, polynomial in zip(outer_generators, inner, strict=True)
        }
        return tuple(
            _wire_polynomial(
                sp.Poly(
                    sp.expand(
                        polynomial.as_expr().subs(
                            substitutions,
                            simultaneous=True,
                        )
                    )
                    - result_generators[index],
                    *result_generators,
                    domain=sp.QQ,
                )
            )
            for index, polynomial in enumerate(outer)
        )

    return (
        compose_residuals(inverse, target_generators, forward, source_generators),
        compose_residuals(forward, source_generators, inverse, target_generators),
    )


def _validate_request[RequestModel: ContractModel](
    model: type[RequestModel],
    payload: object,
    *,
    code: str,
    operation: str,
) -> RequestModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _polynomial_error(
            code,
            "request_validation",
            f"The complete polynomial {operation} request is invalid.",
        ) from exc


def _sympy_map(
    polynomial_map: RationalPolynomialMap,
) -> tuple[tuple[Any, ...], tuple[Poly, ...]]:
    sp = _sympy.get()
    generators = cast(
        tuple[Any, ...],
        sp.symbols(" ".join(polynomial_map.variables), seq=True),
    )
    coordinates = tuple(
        _sympy_polynomial(polynomial, generators)
        for polynomial in polynomial_map.coordinates
    )
    return generators, coordinates


def _sympy_polynomial(
    polynomial: SparseRationalPolynomial,
    generators: tuple[Any, ...],
) -> Poly:
    sp = _sympy.get()
    terms = {}
    for term in polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        terms[term.exponents] = sp.QQ(coefficient.numerator, coefficient.denominator)
    return sp.Poly.from_dict(terms, generators, domain=sp.QQ)


def _wire_polynomial(polynomial: Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=_wire_rational(coefficient),
                exponents=exponents,
            )
            for exponents, coefficient in polynomial.terms()
            if coefficient != 0
        )
    )


def _wire_rational(value: object) -> CanonicalRational:
    rational = _sympy.get().Rational(value)
    return CanonicalRational(num=str(rational.p), den=str(rational.q))


def _evaluate(
    polynomial_map: RationalPolynomialMap,
    point: RationalPolynomialPoint,
) -> tuple[CanonicalRational, ...]:
    sp = _sympy.get()
    try:
        generators, coordinates = _sympy_map(polynomial_map)
        substitutions = {}
        for generator, value in zip(generators, point.values, strict=True):
            fraction = value.as_fraction()
            substitutions[generator] = sp.QQ(fraction.numerator, fraction.denominator)
        return tuple(_wire_rational(poly.eval(substitutions)) for poly in coordinates)
    except (
        cast(type[BaseException], sp.PolynomialError),
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise _polynomial_error(
            "POLYNOMIAL_EVALUATION_FAILED",
            "evaluation",
            "The exact polynomial-map evaluation failed.",
        ) from exc


def _materialize_evaluation(
    resources: PolynomialResources,
    *,
    map_uri: str,
    point: RationalPolynomialPoint,
    image: tuple[CanonicalRational, ...],
) -> tuple[PolynomialMapEvaluation, str]:
    evaluation = PolynomialMapEvaluation(
        map_uri=map_uri,
        point=point,
        image=image,
        backend_version=SYMPY_VERSION,
    )
    artifact = resources.artifacts.put(
        schema_uri=resources.installation.evaluation_schema_uri,
        semantics_uri=resources.installation.semantics_uri,
        payload=evaluation.model_dump(mode="json"),
        parents=(map_uri,),
        summary="exact rational polynomial-map point evaluation",
    )
    return evaluation, artifact.artifact_uri


def _computed_result(
    *,
    descriptor: CapabilityDescriptor,
    request: CapabilityRequest,
    started: float,
    output: dict[str, Any],
    scope: CapabilityScope,
    relationships: tuple[CapabilityRelationship, ...],
    artifact_uris: tuple[str, ...],
    completeness_basis: str,
    completeness_status: CapabilityCompletenessStatus = (
        CapabilityCompletenessStatus.COMPLETE
    ),
    assurance_basis: str = (
        "deterministic exact SymPy arithmetic over QQ; the computation did not "
        "authorize or invoke an independent checker"
    ),
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        mode=request.mode,
        execution=Execution(
            status=ExecutionStatus.COMPLETED,
            runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        ),
        output=output,
        scope=scope,
        completeness=CapabilityCompleteness(
            status=completeness_status,
            basis=(
                f"{completeness_basis}; no mathematical conclusion or independent "
                "verification is claimed"
            ),
            assurance_level=CapabilityAssuranceLevel.COMPUTED,
        ),
        relationships=relationships,
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis=assurance_basis,
        ),
        artifact_uris=artifact_uris,
    )


def _polynomial_error(
    code: str,
    stage: str,
    message: str,
) -> CapabilityInvocationError:
    return CapabilityInvocationError(
        CapabilityDiagnostic(
            code=code,
            stage=stage,
            message=message,
            hint=(
                "Use the advertised sparse QQ schema with reduced rationals, "
                "matching dimensions, unique exponent vectors in descending "
                "monomial order, and no zero-coefficient terms. Combine duplicate "
                "exponent vectors before invoking."
            ),
        )
    )
