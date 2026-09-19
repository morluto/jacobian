"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import comb, lcm
from typing import Any, Literal, NoReturn

import sympy
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
    request_checkpoint,
    require_execution_deadline,
)
from jacobian.backends import BackendUnavailableError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.rational_linear._models import LinearRationalSystem
from jacobian.math.matrices.rational_linear.operations import (
    _admit_system,
    _LinearPlan,
    _solve_admitted,
)
from jacobian.math.matrices.values import (
    MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.ideals._models import (
    MAX_CERTIFICATE_COFACTOR_DEGREE,
    MAX_CERTIFICATE_INPUT_TERMS,
    MAX_COEFFICIENT_DIGITS,
    MAX_GENERATORS,
    MAX_INPUT_EXPONENT,
    MAX_INPUT_TERMS,
    MAX_OUTPUT_TERMS,
    MAX_VARS,
    EliminationIdealResult,
    GradedBettiNumber,
    GroebnerBasisResult,
    IdealComputationBudget,
    IdealContainmentLedger,
    IdealContainmentResult,
    IdealEqualityResult,
    IdealMembershipCertificateResult,
    IdealMinimalPrimesResult,
    IdealNormalFormResult,
    IdealQuotientResult,
    IdealRadicalMembershipResult,
    IdealRadicalResult,
    IdealSaturationResult,
    LcmLatticeHomologyEntry,
    MonomialIdealBettiResult,
    MultigradedBettiNumber,
    _admit_monomial_ideal,
    _require_computed_minimal_prime_family,
    _require_ideal_budget,
    _require_provable_family_fit,
    _validation_error,
)
from jacobian.math.polynomials.ideals._monomial_betti import (
    compute_monomial_betti_kernel,
)
from jacobian.math.polynomials.ideals._singular import (
    run_singular_ideal_operation,
    run_singular_minimal_primes,
)
from jacobian.math.polynomials.ideals._sympy_process import (
    _ResultLimitExceededError,
    _run_sympy_kernel,
    _SympyKernelCancelledError,
    _SympyKernelError,
    _SympyKernelTimeoutError,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

_MAX_CERTIFICATE_COLUMNS = 1_024
_MAX_CERTIFICATE_ROWS = 2_048
_MAX_CERTIFICATE_NONZEROS = MAX_SPARSE_RATIONAL_MATRIX_NONZEROS


@dataclass(frozen=True)
class _MembershipCertificatePlan:
    columns: tuple[tuple[int, tuple[int, ...]], ...]
    linear_plan: _LinearPlan


@dataclass(frozen=True)
class _ImmediateMembershipCertificate:
    cofactors: tuple[RationalPolynomial, ...]


def _run_admission[T](admission: Callable[[], T]) -> T:
    try:
        return admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.ideal_admission", message=str(exc)
        ) from exc


def monomial_ideal_graded_betti_table(
    ideal: RationalPolynomialIdeal,
) -> MonomialIdealBettiResult:
    """Compute the complete minimal graded Betti profile of a monomial ideal."""

    _run_admission(lambda: _admit_monomial_ideal(ideal))
    computed = compute_monomial_betti_kernel(ideal)
    lattice_homology = tuple(
        LcmLatticeHomologyEntry.model_construct(
            multidegree=entry.multidegree,
            face_counts=entry.face_counts,
            boundary_ranks=entry.boundary_ranks,
            reduced_homology_dimensions=entry.reduced_homology_dimensions,
        )
        for entry in computed.lattice_homology
    )
    multigraded = tuple(
        MultigradedBettiNumber.model_construct(
            homological_degree=homological_degree,
            multidegree=multidegree,
            value=value,
        )
        for homological_degree, multidegree, value in computed.multigraded_betti
    )
    graded = tuple(
        GradedBettiNumber.model_construct(
            homological_degree=homological_degree,
            internal_degree=internal_degree,
            value=value,
        )
        for homological_degree, internal_degree, value in computed.graded_betti
    )
    return MonomialIdealBettiResult._from_kernel(
        ideal,
        lcm_lattice_homology=lattice_homology,
        multigraded_betti_numbers=multigraded,
        graded_betti_numbers=graded,
        regularity=computed.regularity,
        has_linear_resolution=computed.has_linear_resolution,
    )


def _admit_source(ideal: RationalPolynomialIdeal, *, label: str) -> None:
    # A nonzero constant already establishes the unit ideal.  Its coefficient
    # never enters a Groebner expansion, so the kernel coefficient cap must not
    # reject this exact zero-work case.
    if any(
        len(generator.polynomial.terms) == 1
        and not any(generator.polynomial.terms[0].exponents)
        and generator.polynomial.terms[0].coefficient.num != 0
        for generator in ideal.generators
    ):
        if len(ideal.variables) > MAX_VARS:
            raise _validation_error(
                f"{label} exceeds the {MAX_VARS}-variable operation budget"
            )
        if len(ideal.generators) > MAX_GENERATORS:
            raise _validation_error(
                f"{label} exceeds the {MAX_GENERATORS}-generator operation budget"
            )
        if (
            sum(len(generator.polynomial.terms) for generator in ideal.generators)
            > MAX_INPUT_TERMS
        ):
            raise _validation_error(
                f"{label} exceeds the {MAX_INPUT_TERMS}-term aggregate input budget"
            )
        for generator in ideal.generators:
            if (
                len(generator.polynomial.terms) == 1
                and not any(generator.polynomial.terms[0].exponents)
                and generator.polynomial.terms[0].coefficient.num != 0
            ):
                continue
            require_polynomial_budget(
                generator,
                maximum_terms=MAX_INPUT_TERMS,
                maximum_exponent=MAX_INPUT_EXPONENT,
                maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
                label=f"{label} generator",
            )
            if any(
                sum(term.exponents) > MAX_INPUT_EXPONENT
                for term in generator.polynomial.terms
            ):
                raise _validation_error(
                    f"{label} generator exceeds total degree {MAX_INPUT_EXPONENT}"
                )
        return
    _require_ideal_budget(ideal, label=label)


def _admit_membership(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="membership polynomial",
    )


def _admit_saturation(
    ideal: RationalPolynomialIdeal, denominator: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    if not denominator.polynomial.terms:
        raise _validation_error("saturation denominator must be nonzero")
    require_polynomial_budget(
        denominator,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="saturation denominator",
    )


def _admit_quotient(
    dividend: RationalPolynomialIdeal, divisor: RationalPolynomialIdeal
) -> None:
    _admit_source(dividend, label="dividend ideal")
    _admit_source(divisor, label="divisor ideal")


def _admit_relation(
    left: RationalPolynomialIdeal, right: RationalPolynomialIdeal
) -> None:
    _admit_source(left, label="left ideal")
    _admit_source(right, label="right ideal")
    if left.variables != right.variables:
        raise _validation_error(
            "ideal relation operands must use the same ordered ring"
        )


def _admit_minimal_primes(ideal: RationalPolynomialIdeal) -> None:
    _admit_source(ideal, label="ideal")
    _require_provable_family_fit(ideal)


def _admit_groebner(ideal: RationalPolynomialIdeal) -> None:
    _admit_source(ideal, label="ideal")


def _admit_normal_form(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> None:
    _admit_source(ideal, label="ideal")
    if polynomial.variables != ideal.variables:
        raise _validation_error("polynomial must use the ideal's ordered ring")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_INPUT_TERMS,
        maximum_exponent=MAX_INPUT_EXPONENT,
        maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
        label="polynomial",
    )


def _immediate_membership_certificate(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> _ImmediateMembershipCertificate | None:
    if not polynomial.polynomial.terms or polynomial in ideal.generators:
        zero = RationalPolynomial(
            variables=ideal.variables, polynomial=SparseRationalPolynomial(terms=())
        )
        cofactors = [zero] * len(ideal.generators)
        if polynomial.polynomial.terms:
            cofactors[ideal.generators.index(polynomial)] = RationalPolynomial(
                variables=ideal.variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0,) * len(ideal.variables),
                        ),
                    )
                ),
            )
        return _ImmediateMembershipCertificate(tuple(cofactors))
    return None


def _admit_membership_certificate(
    ideal: RationalPolynomialIdeal,
    polynomial: RationalPolynomial,
    cofactor_degree_bound: int,
) -> _MembershipCertificatePlan | _ImmediateMembershipCertificate:
    if type(cofactor_degree_bound) is not int or not (
        0 <= cofactor_degree_bound <= MAX_CERTIFICATE_COFACTOR_DEGREE
    ):
        raise _validation_error(
            f"cofactor degree must be between 0 and {MAX_CERTIFICATE_COFACTOR_DEGREE}"
        )
    if polynomial.variables != ideal.variables:
        raise _validation_error(
            "certificate polynomial must use the ideal's ordered ring"
        )
    aggregate_terms = sum(
        len(generator.polynomial.terms) for generator in ideal.generators
    )
    if len(ideal.variables) > MAX_VARS or len(ideal.generators) > MAX_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="polynomial.ideal_certificate.source_budget_exceeded",
            message="ideal exceeds the certificate variable or generator bound",
        )
    if aggregate_terms > MAX_CERTIFICATE_INPUT_TERMS:
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="polynomial.ideal_certificate.source_budget_exceeded",
            message="ideal exceeds the certificate aggregate source-term bound",
        )
    try:
        for generator in ideal.generators:
            require_polynomial_budget(
                generator,
                maximum_terms=MAX_CERTIFICATE_INPUT_TERMS,
                maximum_exponent=MAX_INPUT_EXPONENT,
                maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
                label="ideal generator",
            )
        require_polynomial_budget(
            polynomial,
            maximum_terms=MAX_CERTIFICATE_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label="polynomial",
        )
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=("ideal",) if "ideal generator" in str(exc) else ("polynomial",),
            code="polynomial.ideal_certificate.source_budget_exceeded",
            message=str(exc),
        ) from exc
    immediate = _immediate_membership_certificate(ideal, polynomial)
    if immediate is not None:
        return immediate
    monomial_count = comb(
        len(ideal.variables) + cofactor_degree_bound,
        cofactor_degree_bound,
    )
    if monomial_count * len(ideal.generators) > _MAX_CERTIFICATE_COLUMNS:
        raise OperationResourceAdmissionError(
            location=("cofactor_degree_bound",),
            code="polynomial.ideal_certificate.work_budget_exceeded",
            message="cofactor search exceeds the 1,024-column certificate envelope",
        )
    if monomial_count * aggregate_terms > _MAX_CERTIFICATE_NONZEROS:
        raise OperationResourceAdmissionError(
            location=("cofactor_degree_bound",),
            code="polynomial.ideal_certificate.work_budget_exceeded",
            message=(
                "cofactor expansion exceeds the "
                f"{_MAX_CERTIFICATE_NONZEROS:,}-nonzero certificate envelope"
            ),
        )
    monomials = _cofactor_monomials(len(ideal.variables), cofactor_degree_bound)
    columns = tuple(
        (generator_index, monomial)
        for generator_index in range(len(ideal.generators))
        for monomial in monomials
    )
    row_exponents = {tuple(term.exponents) for term in polynomial.polynomial.terms}
    entries: list[SparseRationalMatrixEntry] = []
    expanded: list[tuple[int, tuple[int, ...], RationalPolynomialTerm]] = []
    for column, (generator_index, multiplier_exponents) in enumerate(columns):
        for term in ideal.generators[generator_index].polynomial.terms:
            product_exponents = tuple(
                left + right
                for left, right in zip(
                    multiplier_exponents, term.exponents, strict=True
                )
            )
            row_exponents.add(product_exponents)
            expanded.append((column, product_exponents, term))
            if len(row_exponents) > _MAX_CERTIFICATE_ROWS:
                raise OperationResourceAdmissionError(
                    location=("cofactor_degree_bound",),
                    code="polynomial.ideal_certificate.work_budget_exceeded",
                    message=(
                        "cofactor expansion exceeds the 2,048-row certificate envelope"
                    ),
                )
    ordered_rows = tuple(sorted(row_exponents, reverse=True))
    row_index = {exponents: index for index, exponents in enumerate(ordered_rows)}
    for column, product_exponents, term in expanded:
        entries.append(
            SparseRationalMatrixEntry(
                row=row_index[product_exponents],
                column=column,
                value=term.coefficient,
            )
        )
    target = {
        tuple(term.exponents): term.coefficient for term in polynomial.polynomial.terms
    }
    system = LinearRationalSystem(
        variables=tuple(f"c{index}" for index in range(len(columns))),
        coefficients=SparseRationalMatrix(
            row_count=len(ordered_rows),
            column_count=len(columns),
            entries=tuple(sorted(entries, key=lambda entry: (entry.row, entry.column))),
        ),
        rhs=tuple(
            target.get(exponents, CanonicalRational(num=0, den=1))
            for exponents in ordered_rows
        ),
    )
    try:
        linear_plan = _admit_system(system, outcome="solution")
    except OperationDomainValidationError as exc:
        issue = exc.errors()[0]
        raise OperationResourceAdmissionError(
            location=("cofactor_degree_bound",),
            code="polynomial.ideal_certificate.linear_work_budget_exceeded",
            message=str(issue["msg"]),
        ) from exc
    return _MembershipCertificatePlan(
        columns=columns,
        linear_plan=linear_plan,
    )


def _cofactor_monomials(
    variable_count: int, degree_bound: int
) -> tuple[tuple[int, ...], ...]:
    def exact(total: int, slots: int) -> tuple[tuple[int, ...], ...]:
        if slots == 1:
            return ((total,),)
        return tuple(
            (first, *tail)
            for first in range(total + 1)
            for tail in exact(total - first, slots - 1)
        )

    return tuple(
        sorted(
            (
                monomial
                for degree in range(degree_bound + 1)
                for monomial in exact(degree, variable_count)
            ),
            reverse=True,
        )
    )


def ideal_membership_certificate(
    ideal: RationalPolynomialIdeal,
    polynomial: RationalPolynomial,
    cofactor_degree_bound: int,
) -> IdealMembershipCertificateResult:
    """Return an integral source-generator identity within a cofactor bound."""

    plan = _run_admission(
        lambda: _admit_membership_certificate(ideal, polynomial, cofactor_degree_bound)
    )
    if isinstance(plan, _ImmediateMembershipCertificate):
        return IdealMembershipCertificateResult._from_kernel(
            ideal=ideal,
            polynomial=polynomial,
            cofactor_degree_bound=cofactor_degree_bound,
            status="CERTIFICATE",
            multiplier=1,
            cofactors=plan.cofactors,
        )

    solution = _solve_admitted(plan.linear_plan)
    if solution is None:
        return IdealMembershipCertificateResult._from_kernel(
            ideal=ideal,
            polynomial=polynomial,
            cofactor_degree_bound=cofactor_degree_bound,
            status="NO_CERTIFICATE_WITHIN_BOUND",
        )

    fractions = tuple(value.as_fraction() for value in solution)
    multiplier = lcm(*(value.denominator for value in fractions))
    cofactor_terms: list[list[RationalPolynomialTerm]] = [[] for _ in ideal.generators]
    for value, (generator_index, exponents) in zip(
        fractions, plan.columns, strict=True
    ):
        coefficient = value * multiplier
        if coefficient:
            cofactor_terms[generator_index].append(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient.numerator, den=1),
                    exponents=exponents,
                )
            )
    cofactors = tuple(
        RationalPolynomial(
            variables=ideal.variables,
            polynomial=SparseRationalPolynomial(terms=tuple(terms)),
        )
        for terms in cofactor_terms
    )
    return IdealMembershipCertificateResult._from_kernel(
        ideal=ideal,
        polynomial=polynomial,
        cofactor_degree_bound=cofactor_degree_bound,
        status="CERTIFICATE",
        multiplier=multiplier,
        cofactors=cofactors,
    )


def _admit_elimination(
    ideal: RationalPolynomialIdeal, eliminated_variables: tuple[str, ...]
) -> None:
    _admit_source(ideal, label="ideal")
    eliminated = set(eliminated_variables)
    if any(var not in ideal.variables for var in eliminated):
        raise _validation_error(
            "eliminated variables must be a subset of the ideal's variables"
        )
    if not tuple(v for v in ideal.variables if v not in eliminated):
        raise _validation_error(
            "elimination cannot remove every variable; at least one must remain"
        )


def _raise_ideal_backend_failure(
    operation: str, outcome: str, detail: str | None
) -> NoReturn:
    message = detail or f"ideal {operation} backend did not produce an exact result"
    if outcome == "UNAVAILABLE":
        raise BackendUnavailableError("singular", detail=message)
    if outcome == "TIMEOUT":
        raise OperationExecutionTimeoutError(message)
    if outcome == "CANCELLED":
        raise OperationExecutionCancelledError(message)
    raise RuntimeError(message)


def ideal_radical(
    ideal: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealRadicalResult:
    """Compute an exact ideal radical through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_source(ideal, label="ideal"))

    backend = run_singular_ideal_operation(
        "radical",
        ideal,
        None,
        resource_budget,
    )
    if backend.outcome != "COMPUTED" or backend.ideal is None:
        _raise_ideal_backend_failure("radical", backend.outcome, backend.detail)
    return IdealRadicalResult(radical=backend.ideal)


def ideal_minimal_primes(
    ideal: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealMinimalPrimesResult:
    """Compute the complete minimal-prime family over ``QQ``."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_minimal_primes(ideal))

    backend = run_singular_minimal_primes(ideal, resource_budget)
    components = backend.components
    if backend.outcome != "COMPUTED" or components is None:
        _raise_ideal_backend_failure("minimal-prime", backend.outcome, backend.detail)

    try:
        _require_computed_minimal_prime_family(ideal, components)
        return IdealMinimalPrimesResult._from_kernel(ideal=ideal, components=components)
    except _ResultLimitExceededError as error:
        raise RuntimeError(str(error)) from error
    except ValueError as error:
        raise RuntimeError(
            "the computed minimal-prime family violated its public invariant"
        ) from error


def ideal_radical_membership(
    ideal: RationalPolynomialIdeal, polynomial: RationalPolynomial
) -> IdealRadicalMembershipResult:
    """Decide radical membership by the exact Rabinowitsch criterion."""

    _run_admission(lambda: _admit_membership(ideal, polynomial))

    variable_symbols = symbols_for_variables(ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in ideal.generators
    ]
    polynomial_expr = rational_polynomial_to_sympy(polynomial).as_expr()
    auxiliary = sympy.Dummy("jacobian_rabinowitsch")
    basis = sympy.groebner(
        [*ideal_generators, 1 - auxiliary * polynomial_expr],
        *variable_symbols,
        auxiliary,
        order="grevlex",
        domain=sympy.QQ,
    )
    return IdealRadicalMembershipResult(
        ideal=ideal,
        polynomial=polynomial,
        in_radical=len(basis) == 1 and basis[0] == 1,
    )


def verify_ideal_radical_membership(claim: IdealRadicalMembershipResult) -> bool:
    """Check the bounded Rabinowitsch decision asserted by a claim."""

    return (
        ideal_radical_membership(claim.ideal, claim.polynomial).in_radical
        is claim.in_radical
    )


def ideal_quotient(
    dividend: RationalPolynomialIdeal,
    divisor: RationalPolynomialIdeal,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealQuotientResult:
    """Compute an exact ideal quotient through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_quotient(dividend, divisor))

    backend = run_singular_ideal_operation(
        "quotient",
        dividend,
        divisor,
        resource_budget,
    )
    if backend.outcome != "COMPUTED" or backend.ideal is None:
        _raise_ideal_backend_failure("quotient", backend.outcome, backend.detail)
    return IdealQuotientResult(quotient=backend.ideal)


def ideal_saturation(
    ideal: RationalPolynomialIdeal,
    denominator: RationalPolynomial,
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealSaturationResult:
    """Compute I : <d>^infinity through the bounded Singular backend."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_saturation(ideal, denominator))

    denominator_ideal = RationalPolynomialIdeal(
        variables=denominator.variables,
        generators=(denominator,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        ideal,
        denominator_ideal,
        resource_budget,
    )
    if backend.outcome != "COMPUTED" or backend.ideal is None:
        _raise_ideal_backend_failure("saturation", backend.outcome, backend.detail)
    return IdealSaturationResult(saturation=backend.ideal)


def _raise_if_relation_deadline_exceeded(deadline: float) -> None:
    if request_cancelled():
        raise _SympyKernelCancelledError()
    if time.monotonic() > deadline:
        raise _SympyKernelTimeoutError()


def _bind_relation_deadline(
    resource_budget: IdealComputationBudget, outer_deadline: float | None = None
) -> float:
    """Bind the relation kernel to the request's complete deadline."""
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else time.monotonic()
    request_deadline = started_at + resource_budget.wall_seconds
    if execution is not None and execution.deadline is not None:
        request_deadline = min(request_deadline, execution.deadline)
    if outer_deadline is not None:
        # Never restart a fresh sub-window past a caller's absolute deadline.
        request_deadline = min(request_deadline, outer_deadline)
    bind_request_deadline(request_deadline)
    return request_deadline


def _run_relation_kernel_before_deadline(
    payload: dict[str, Any], deadline: float
) -> dict[str, Any]:
    """Launch the worker only when a strictly positive budget remains."""
    _raise_if_relation_deadline_exceeded(deadline)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _SympyKernelTimeoutError()
    return _run_sympy_kernel(payload, remaining, deadline=deadline)


def _relation_ledger(
    payload: dict[str, Any], *, deadline: float
) -> IdealContainmentLedger:
    _raise_if_relation_deadline_exceeded(deadline)
    ledger = IdealContainmentLedger.model_validate_json(json.dumps(payload))
    _raise_if_relation_deadline_exceeded(deadline)
    return ledger


def ideal_containment(
    source: RationalPolynomialIdeal,
    target: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealContainmentResult:
    """Decide ``source subseteq target`` with a source-ordered exact ledger."""

    resource_budget = resource_budget or IdealComputationBudget()
    deadline = _bind_relation_deadline(resource_budget)
    try:
        _raise_if_relation_deadline_exceeded(deadline)
        _run_admission(lambda: _admit_relation(source, target))
        _raise_if_relation_deadline_exceeded(deadline)
        payload = {
            "mode": "ideal_relation",
            "variables": list(source.variables),
            "generators": [item.model_dump(mode="json") for item in source.generators],
            "right_generators": [
                item.model_dump(mode="json") for item in target.generators
            ],
            "order": monomial_order,
            "maximum_terms": MAX_OUTPUT_TERMS,
            "mutual": False,
        }
        result = _run_relation_kernel_before_deadline(payload, deadline)
        ledger = _relation_ledger(result["left_in_right"], deadline=deadline)
        _raise_if_relation_deadline_exceeded(deadline)
        computed = IdealContainmentResult._from_kernel(
            source, target, ledger, monomial_order
        )
        _raise_if_relation_deadline_exceeded(deadline)
        return computed
    except _SympyKernelCancelledError:
        raise OperationExecutionCancelledError(
            "ideal containment was cancelled before producing a result"
        ) from None
    except _SympyKernelTimeoutError:
        raise OperationExecutionTimeoutError(
            "ideal containment exceeded the enforced wall-time budget"
        ) from None
    except _ResultLimitExceededError:
        raise RuntimeError(
            "the exact containment ledger exceeds the declared result bound"
        ) from None
    except OperationDomainValidationError:
        raise
    except (KeyError, TypeError, ValueError, _SympyKernelError) as error:
        raise RuntimeError(
            "the bounded kernel failed without producing an exact containment"
        ) from error


def ideal_equality(
    left: RationalPolynomialIdeal,
    right: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> IdealEqualityResult:
    """Decide equality by two ledgers computed under one request deadline."""

    resource_budget = resource_budget or IdealComputationBudget()
    deadline = _bind_relation_deadline(resource_budget)
    try:
        _raise_if_relation_deadline_exceeded(deadline)
        _run_admission(lambda: _admit_relation(left, right))
        _raise_if_relation_deadline_exceeded(deadline)
        payload = {
            "mode": "ideal_relation",
            "variables": list(left.variables),
            "generators": [item.model_dump(mode="json") for item in left.generators],
            "right_generators": [
                item.model_dump(mode="json") for item in right.generators
            ],
            "order": monomial_order,
            "maximum_terms": MAX_OUTPUT_TERMS,
            "mutual": True,
        }
        result = _run_relation_kernel_before_deadline(payload, deadline)
        left_in_right = _relation_ledger(result["left_in_right"], deadline=deadline)
        right_in_left = _relation_ledger(result["right_in_left"], deadline=deadline)
        _raise_if_relation_deadline_exceeded(deadline)
        computed = IdealEqualityResult._from_kernel(
            left, right, left_in_right, right_in_left, monomial_order
        )
        _raise_if_relation_deadline_exceeded(deadline)
        return computed
    except _SympyKernelCancelledError:
        raise OperationExecutionCancelledError(
            "ideal equality was cancelled before producing a result"
        ) from None
    except _SympyKernelTimeoutError:
        raise OperationExecutionTimeoutError(
            "ideal equality exceeded the enforced wall-time budget"
        ) from None
    except _ResultLimitExceededError:
        raise RuntimeError(
            "the exact equality ledgers exceed the declared result bound"
        ) from None
    except OperationDomainValidationError:
        raise
    except (KeyError, TypeError, ValueError, _SympyKernelError) as error:
        raise RuntimeError(
            "the bounded kernel failed without producing an exact equality result"
        ) from error


def verify_ideal_membership_certificate(
    claim: IdealMembershipCertificateResult,
) -> bool:
    """Verify a serialized ideal-membership identity against its sources."""

    if claim.status == "NO_CERTIFICATE_WITHIN_BOUND":
        return (
            ideal_membership_certificate(
                claim.ideal,
                claim.polynomial,
                claim.cofactor_degree_bound,
            )
            == claim
        )
    try:
        _admit_membership_certificate(
            claim.ideal,
            claim.polynomial,
            claim.cofactor_degree_bound,
        )
        if claim.multiplier is None or claim.cofactors is None:
            return False
        cofactor_terms = 0
        for cofactor in claim.cofactors:
            require_polynomial_budget(
                cofactor,
                maximum_terms=MAX_CERTIFICATE_INPUT_TERMS,
                maximum_exponent=MAX_INPUT_EXPONENT,
                maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
                label="certificate cofactor",
            )
            cofactor_terms += len(cofactor.polynomial.terms)
        if (
            cofactor_terms
            * sum(
                len(generator.polynomial.terms) for generator in claim.ideal.generators
            )
            > _MAX_CERTIFICATE_NONZEROS
        ):
            return False
        if any(
            sum(term.exponents) > claim.cofactor_degree_bound
            for cofactor in claim.cofactors
            for term in cofactor.polynomial.terms
        ):
            return False
    except OperationResourceAdmissionError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, sympy.SympifyError):
        return False
    target = rational_polynomial_to_sympy(claim.polynomial).as_expr()
    identity = sympy.Integer(0)
    for cofactor, generator in zip(
        claim.cofactors, claim.ideal.generators, strict=True
    ):
        identity += rational_polynomial_to_sympy(cofactor).as_expr() * (
            rational_polynomial_to_sympy(generator).as_expr()
        )
    return bool(sympy.expand(identity - claim.multiplier * target) == 0)


def verify_groebner_basis(claim: GroebnerBasisResult) -> bool:
    """Verify a reduced Gröbner basis against its retained ideal source."""

    try:
        _admit_groebner(claim.ideal)
        payload = {
            "mode": "verify_groebner_basis",
            "variables": list(claim.ideal.variables),
            "order": claim.monomial_order,
            "generators": [
                generator.model_dump(mode="json")
                for generator in claim.ideal.generators
            ],
            "basis": [
                generator.model_dump(mode="json")
                for generator in claim.basis.generators
            ],
        }
    except OperationResourceAdmissionError:
        raise
    except _ResultLimitExceededError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, sympy.SympifyError):
        return False

    equal = _run_sympy_kernel(payload, 10).get("equal")
    if type(equal) is not bool:
        raise RuntimeError(
            "the Groebner verification worker omitted its Boolean result"
        )
    return equal


def verify_ideal_normal_form(claim: IdealNormalFormResult) -> bool:
    """Verify a normal-form claim against its ideal, polynomial, and order."""

    try:
        return (
            ideal_normal_form(
                claim.ideal,
                claim.polynomial,
                claim.monomial_order,
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_ideal_containment(claim: IdealContainmentResult) -> bool:
    """Verify a directed containment ledger against both retained ideals."""

    try:
        return (
            ideal_containment(
                claim.source,
                claim.target,
                claim.monomial_order,
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_ideal_equality(claim: IdealEqualityResult) -> bool:
    """Verify both equality ledgers against the retained ideal presentations."""

    try:
        return (
            ideal_equality(
                claim.left,
                claim.right,
                claim.monomial_order,
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


__all__ = [
    "elimination_ideal",
    "groebner_basis",
    "ideal_containment",
    "ideal_equality",
    "ideal_membership_certificate",
    "ideal_minimal_primes",
    "ideal_normal_form",
    "ideal_quotient",
    "ideal_radical",
    "ideal_radical_membership",
    "ideal_saturation",
    "monomial_ideal_graded_betti_table",
    "verify_groebner_basis",
    "verify_ideal_containment",
    "verify_ideal_equality",
    "verify_ideal_membership_certificate",
    "verify_ideal_normal_form",
]


def groebner_basis(
    ideal: RationalPolynomialIdeal,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    *,
    resource_budget: IdealComputationBudget | None = None,
    _outer_deadline: float | None = None,
) -> GroebnerBasisResult:
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_groebner(ideal))
    deadline = _bind_relation_deadline(resource_budget, _outer_deadline)
    source_ideal = ideal
    variables = source_ideal.variables
    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map[monomial_order]
    payload = {
        "mode": "groebner",
        "variables": list(variables),
        "order": order,
        "maximum_terms": MAX_OUTPUT_TERMS,
        "generators": [
            generator.model_dump(mode="json") for generator in source_ideal.generators
        ],
    }

    # The unbounded search runs in a killable worker under the one shared
    # request deadline; result assembly then charges the remaining allowance.
    try:
        result_payload = _run_relation_kernel_before_deadline(payload, deadline)
    except _SympyKernelCancelledError:
        raise OperationExecutionCancelledError(
            "the Groebner computation was cancelled before producing a result"
        ) from None
    except _SympyKernelTimeoutError:
        raise OperationExecutionTimeoutError(
            "Groebner computation exceeded the enforced "
            f"{resource_budget.wall_seconds}s budget"
        ) from None
    except _ResultLimitExceededError as error:
        raise RuntimeError(
            "the exact reduced Gröbner basis exceeds the declared "
            f"exact-result limit: {error}"
        ) from error
    except _SympyKernelError as error:
        raise RuntimeError(
            "the bounded Groebner kernel failed without producing an "
            f"exact basis: {error}"
        ) from error

    basis_generators = []
    request_checkpoint("after Groebner kernel worker")
    # A native call has no request envelope, so request_checkpoint cannot see
    # the absolute deadline. Enforce it explicitly before decoding and result
    # assembly, which happen after the worker and could otherwise return
    # success past the shared wall limit.
    require_execution_deadline(deadline)
    for position, item in enumerate(result_payload["generators"]):
        if position % 256 == 0:
            request_checkpoint("during Groebner basis decoding")
        basis_generators.append(
            RationalPolynomial.model_validate_json(json.dumps(item))
        )
    request_checkpoint("after Groebner basis decoding")
    require_execution_deadline(deadline)
    if not basis_generators:
        from jacobian.math.polynomials.values import SparseRationalPolynomial

        zero = RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        basis_generators.append(zero)

    basis_ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(basis_generators),
    )
    request_checkpoint("before Groebner result construction")
    return GroebnerBasisResult._from_kernel(source_ideal, basis_ideal, monomial_order)


def ideal_normal_form(
    ideal: RationalPolynomialIdeal,
    polynomial: RationalPolynomial,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
) -> IdealNormalFormResult:
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    _run_admission(lambda: _admit_normal_form(ideal, polynomial))
    from jacobian.math.polynomials.ideals._models import IdealNormalFormResult

    payload = {
        "mode": "normal_form",
        "variables": list(ideal.variables),
        "order": monomial_order,
        "generators": [
            generator.model_dump(mode="json") for generator in ideal.generators
        ],
        "polynomial": polynomial.model_dump(mode="json"),
    }

    # A conservative 10-second budget bounds the killable kernel that runs
    # the unbounded Gröbner search; remainder conversion then operates on
    # declared output limits.
    try:
        result_payload = _run_sympy_kernel(payload, 10)
    except _SympyKernelCancelledError:
        raise OperationExecutionCancelledError(
            "the Gröbner reduction was cancelled before producing a result"
        ) from None
    except _SympyKernelTimeoutError:
        raise OperationExecutionTimeoutError(
            "the Gröbner reduction exceeded the enforced 10s wall-time bound"
        ) from None
    except _ResultLimitExceededError as error:
        raise RuntimeError(
            f"the exact normal form exceeds the declared exact-result limit: {error}"
        ) from error
    except _SympyKernelError as error:
        raise RuntimeError(
            "the bounded reduction kernel failed without producing an "
            f"exact remainder: {error}"
        ) from error

    remainder_poly = RationalPolynomial.model_validate_json(
        json.dumps(result_payload["remainder"])
    )
    return IdealNormalFormResult._from_kernel(
        ideal, polynomial, monomial_order, remainder_poly
    )


def _elimination_ideal_from_payload(
    result_payload: dict[str, Any],
) -> RationalPolynomialIdeal:
    """Convert one bounded elimination-kernel payload to its canonical ideal."""

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    remaining_tuple = tuple(result_payload["remaining"])
    if result_payload["unit_ideal"]:
        elimination_generators = [
            RationalPolynomial(
                variables=remaining_tuple,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0,) * len(remaining_tuple),
                        ),
                    )
                ),
            )
        ]
    elif not result_payload["generators"]:
        elimination_generators = [
            RationalPolynomial(
                variables=remaining_tuple,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        ]
    else:
        elimination_generators = [
            RationalPolynomial.model_validate_json(json.dumps(item))
            for item in result_payload["generators"]
        ]
    return RationalPolynomialIdeal(
        variables=remaining_tuple,
        generators=tuple(elimination_generators),
    )


def elimination_ideal(
    ideal: RationalPolynomialIdeal,
    eliminated_variables: tuple[str, ...],
    *,
    resource_budget: IdealComputationBudget | None = None,
) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""

    resource_budget = resource_budget or IdealComputationBudget()
    _run_admission(lambda: _admit_elimination(ideal, eliminated_variables))

    variables = list(ideal.variables)
    payload = {
        "mode": "elimination",
        "variables": variables,
        "eliminated": list(eliminated_variables),
        "generators": [
            generator.model_dump(mode="json") for generator in ideal.generators
        ],
    }

    # The unbounded lex search runs in a killable worker under the declared
    # wall-time budget; canonicalization then operates only on the declared
    # output limits.
    try:
        result_payload = _run_sympy_kernel(payload, resource_budget.wall_seconds)
    except _SympyKernelCancelledError:
        raise OperationExecutionCancelledError(
            "the lex Gröbner elimination was cancelled before producing a result"
        ) from None
    except _SympyKernelTimeoutError:
        raise OperationExecutionTimeoutError(
            "the lex Gröbner elimination exceeded the enforced "
            f"{resource_budget.wall_seconds}s wall-time budget"
        ) from None
    except _ResultLimitExceededError as error:
        raise RuntimeError(
            "the exact elimination ideal exceeds the declared "
            f"exact-result limit: {error}"
        ) from error
    except _SympyKernelError as error:
        raise RuntimeError(
            "the bounded elimination kernel failed without producing "
            f"an exact ideal: {error}"
        ) from error

    return EliminationIdealResult._from_kernel(
        ideal,
        eliminated_variables,
        _elimination_ideal_from_payload(result_payload),
    )
