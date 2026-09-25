"""Exact polynomial coactions from bounded diagonal integer weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
from math import comb, lcm
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import (
    CanonicalLimits,
    decimal_digit_width,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._weight_models import (
    MAX_DIAGONAL_WEIGHT,
    MAX_GM_INVARIANT_DEGREE,
    MAX_GM_INVARIANT_MONOMIALS,
    MAX_GM_SUBREP_BASIS_COEFFICIENT_DIGITS,
    MAX_GM_SUBREP_DIMENSION,
    MAX_GM_SUBREP_TOTAL_TERMS,
    MAX_WEIGHT_ACTION_DEGREE,
    MAX_WEIGHT_ACTION_TERMS,
    MAX_WEIGHT_ACTION_VARIABLES,
    PolynomialWeightAction,
    PolynomialWeightActionResult,
    PolynomialWeightComponent,
    PolynomialWeightDegreeDimension,
    PolynomialWeightInvariantResult,
    PolynomialWeightSubrepresentationRequest,
    PolynomialWeightSubrepresentationResult,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_WEIGHT_PARAMETER_TYPE = TypeAdapter(PolynomialVariable)


def _as_weight_action(
    action: PolynomialWeightAction | Mapping[str, Any], *, code: str
) -> PolynomialWeightAction:
    try:
        payload = (
            action.model_dump()
            if isinstance(action, PolynomialWeightAction)
            else action
        )
        return PolynomialWeightAction.model_validate(payload)
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("action",),
            code=code,
            message="the request must carry one bounded integer weight per variable",
        ) from exc


def _as_decoded_polynomial(
    value: RationalPolynomial | Mapping[str, Any],
) -> RationalPolynomial:
    try:
        payload = value.model_dump() if isinstance(value, RationalPolynomial) else value
        return RationalPolynomial.model_validate(payload)
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_weight_action.request_shape",
            message="the action request must contain a canonical polynomial in its bound ring",
        ) from exc


def _admit_weight_request(
    action: PolynomialWeightAction,
    source: RationalPolynomial,
    parameter: PolynomialVariable,
) -> None:
    try:
        _WEIGHT_PARAMETER_TYPE.validate_python(parameter, strict=True)
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("parameter",),
            code="polynomial_weight_action.request_shape",
            message="the Laurent parameter must be a strict polynomial variable name",
        ) from exc
    if source.variables != action.variables:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_weight_action.request_shape",
            message="polynomial and action must use the same ordered QQ ring",
        )
    if parameter in action.variables:
        raise OperationDomainValidationError(
            location=("parameter",),
            code="polynomial_weight_action.request_shape",
            message="the Laurent parameter must be distinct from ring variables",
        )
    if len(action.variables) > MAX_WEIGHT_ACTION_VARIABLES:
        raise OperationDomainValidationError(
            location=("action",),
            code="polynomial_weight_action.request_shape",
            message=(
                f"the diagonal action is bounded to {MAX_WEIGHT_ACTION_VARIABLES} "
                "source variables because the Laurent coaction carrier reserves "
                "its eighth axis for the parameter"
            ),
        )

    if len(source.polynomial.terms) > MAX_WEIGHT_ACTION_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "polynomial", "terms"),
            code="polynomial_weight_action.term_budget",
            message="source polynomial exceeds the diagonal-action term budget",
        )
    # Admit all growth before deriving Laurent exponent tuples or projections.
    for term in source.polynomial.terms:
        degree = sum(term.exponents)
        if degree > MAX_WEIGHT_ACTION_DEGREE:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.degree_budget",
                message="source polynomial exceeds the total-degree budget",
            )
        weight = sum(w * e for w, e in zip(action.weights, term.exponents, strict=True))
        # This follows from the admitted degree and coordinate-weight bounds,
        # but retain the explicit carrier check at the representation boundary.
        if abs(weight) > MAX_DIAGONAL_WEIGHT * MAX_WEIGHT_ACTION_DEGREE:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.laurent_exponent_budget",
                message="the induced Laurent exponent exceeds the admitted range",
            )
        try:
            require_bounded_rational(
                term.coefficient,
                max_digits=128,
                label="weight-action source coefficient",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("polynomial", "polynomial", "terms"),
                code="polynomial_weight_action.coefficient_budget",
                message=str(exc),
            ) from exc


def diagonal_weight_action(
    action: PolynomialWeightAction | Mapping[str, Any],
    polynomial: RationalPolynomial | Mapping[str, Any],
    parameter: PolynomialVariable = "t",
) -> PolynomialWeightActionResult:
    """Return ``rho(f)`` and its exact integer-weight decomposition.

    Each source monomial ``c*x**e`` maps to ``c*t**(w.e)*x**e``. The sparse
    image has no more terms than the admitted source, so the preflight below
    bounds its complete expansion before the first output term is built.
    """
    checked_action = _as_weight_action(
        action, code="polynomial_weight_action.request_shape"
    )
    checked_source = _as_decoded_polynomial(polynomial)
    _admit_weight_request(checked_action, checked_source, parameter)
    action, source = checked_action, checked_source
    grouped: dict[int, list[RationalPolynomialTerm]] = {}
    coaction_terms: list[RationalLaurentPolynomialTerm] = []
    for term in source.polynomial.terms:
        weight = sum(w * e for w, e in zip(action.weights, term.exponents, strict=True))
        grouped.setdefault(weight, []).append(term)
        coaction_terms.append(
            RationalLaurentPolynomialTerm(
                coefficient=term.coefficient,
                exponents=(*term.exponents, weight),
            )
        )

    components = tuple(
        PolynomialWeightComponent(
            weight=weight,
            polynomial=RationalPolynomial(
                variables=action.variables,
                polynomial=SparseRationalPolynomial(
                    terms=tuple(
                        sorted(terms, key=lambda term: term.exponents, reverse=True)
                    )
                ),
            ),
        )
        for weight, terms in sorted(grouped.items())
    )
    zero = next(
        (component.polynomial for component in components if component.weight == 0),
        RationalPolynomial(
            variables=action.variables, polynomial=SparseRationalPolynomial(terms=())
        ),
    )
    return PolynomialWeightActionResult(
        action=action,
        source=source,
        parameter=parameter,
        coaction=RationalLaurentPolynomial(
            variables=(*action.variables, parameter),
            terms=tuple(
                sorted(coaction_terms, key=lambda term: term.exponents, reverse=True)
            ),
        ),
        components=components,
        weight_zero=zero,
    )


def _degree_compositions(variable_count: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Yield a homogeneous monomial exponent basis in descending lex order."""
    if variable_count == 1:
        yield (degree,)
        return
    for first in range(degree, -1, -1):
        for tail in _degree_compositions(variable_count - 1, degree - first):
            yield (first, *tail)


def gm_invariants_through_degree(
    action: PolynomialWeightAction | Mapping[str, Any], degree: int
) -> PolynomialWeightInvariantResult:
    """Return the complete weight-zero monomial basis through one degree bound."""
    action = _as_weight_action(action, code="polynomial_weight_invariant.request_shape")
    if (
        isinstance(degree, bool)
        or not isinstance(degree, int)
        or not 0 <= degree <= MAX_GM_INVARIANT_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("degree",),
            code="polynomial_weight_invariant.request_shape",
            message=(
                "the invariant-slice degree must be an integer between 0 and "
                f"{MAX_GM_INVARIANT_DEGREE}"
            ),
        )

    variable_count = len(action.variables)
    candidate_count = comb(variable_count + degree, degree)
    if candidate_count > MAX_GM_INVARIANT_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polynomial_weight_invariant.monomial_budget",
            message=(
                "the complete polynomial degree slice has "
                f"{candidate_count} monomials, exceeding the "
                f"{MAX_GM_INVARIANT_MONOMIALS}-monomial envelope"
            ),
        )
    # All candidate tuples and result rows are bounded before enumeration.
    output_cells = candidate_count * (variable_count + 2) + degree + 1
    if output_cells > 50_000:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polynomial_weight_invariant.output_budget",
            message="the exact invariant basis and Hilbert prefix exceed the output envelope",
        )

    basis = []
    profile = []
    for homogeneous_degree in range(degree + 1):
        count = 0
        for exponents in _degree_compositions(variable_count, homogeneous_degree):
            weight = sum(
                variable_weight * exponent
                for variable_weight, exponent in zip(
                    action.weights, exponents, strict=True
                )
            )
            if weight != 0:
                continue
            count += 1
            basis.append(
                RationalPolynomial(
                    variables=action.variables,
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=exponents,
                            ),
                        )
                    ),
                )
            )
        profile.append(
            PolynomialWeightDegreeDimension(degree=homogeneous_degree, dimension=count)
        )
    return PolynomialWeightInvariantResult(
        action=action,
        degree=degree,
        basis=tuple(basis),
        hilbert_prefix=tuple(profile),
        dimension=len(basis),
    )


def _canonical_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(num=value.numerator, den=value.denominator)


def _parse_subrepresentation_request(
    request: PolynomialWeightSubrepresentationRequest | Mapping[str, Any],
) -> tuple[
    PolynomialWeightSubrepresentationRequest,
    PolynomialWeightAction,
    tuple[RationalPolynomial, ...],
]:
    try:
        payload = (
            request.model_dump()
            if isinstance(request, PolynomialWeightSubrepresentationRequest)
            else request
        )
        checked = PolynomialWeightSubrepresentationRequest.model_validate(payload)
        action = _as_weight_action(
            checked.action.model_dump(), code="gm_subrepresentation.request_shape"
        )
        generators = tuple(
            _as_decoded_polynomial(generator.model_dump())
            for generator in checked.generators
        )
    except (ValidationError, TypeError, ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="gm_subrepresentation.request_shape",
            message="the request must bind polynomial generators to one diagonal G_m action",
        ) from exc
    if any(generator.variables != action.variables for generator in generators):
        raise OperationDomainValidationError(
            location=("generators",),
            code="gm_subrepresentation.ordered_ring",
            message="every generator must use the action's ordered polynomial ring",
        )
    return checked, action, generators


def _admit_subrepresentation_support(
    action: PolynomialWeightAction,
    generators: tuple[RationalPolynomial, ...],
    parameter: PolynomialVariable,
) -> tuple[dict[int, tuple[tuple[int, ...], ...]], dict[int, set[int]]]:
    """Preflight projected support, exact elimination work, and output size."""
    grouped_support: dict[int, set[tuple[int, ...]]] = {}
    grouped_sources: dict[int, set[int]] = {}
    total_terms = 0
    for source_index, generator in enumerate(generators):
        if len(generator.polynomial.terms) > 64:
            raise OperationResourceAdmissionError(
                location=("generators", source_index),
                code="gm_subrepresentation.generator_terms",
                message="each polynomial generator is bounded to 64 sparse terms",
            )
        total_terms += len(generator.polynomial.terms)
        for term in generator.polynomial.terms:
            if sum(term.exponents) > MAX_WEIGHT_ACTION_DEGREE:
                raise OperationResourceAdmissionError(
                    location=("generators", source_index),
                    code="gm_subrepresentation.generator_degree",
                    message="polynomial generators exceed total degree 64",
                )
            try:
                require_bounded_rational(
                    term.coefficient,
                    max_digits=8,
                    label="G_m subrepresentation generator coefficient",
                )
            except ValueError as exc:
                raise OperationResourceAdmissionError(
                    location=("generators", source_index),
                    code="gm_subrepresentation.coefficient_budget",
                    message=str(exc),
                ) from exc
            weight = sum(
                variable_weight * exponent
                for variable_weight, exponent in zip(
                    action.weights, term.exponents, strict=True
                )
            )
            grouped_support.setdefault(weight, set()).add(tuple(term.exponents))
            grouped_sources.setdefault(weight, set()).add(source_index)
    if total_terms > MAX_GM_SUBREP_TOTAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="gm_subrepresentation.total_terms",
            message="the combined sparse generators exceed 256 terms",
        )
    work_cells = sum(
        len(grouped_sources[weight]) ** 2 * len(support)
        for weight, support in grouped_support.items()
    )
    if work_cells > 2_000_000:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="gm_subrepresentation.rref_budget",
            message="weight-projection row reduction exceeds its admitted work envelope",
        )
    return (
        {
            weight: tuple(sorted(support, reverse=True))
            for weight, support in grouped_support.items()
        },
        grouped_sources,
    )


def _subrepresentation_result_size_bound(
    action: PolynomialWeightAction,
    generators: tuple[RationalPolynomial, ...],
    parameter: PolynomialVariable,
    *,
    basis_terms: int,
    dimension: int,
    coefficient_digits: int,
) -> int:
    """Bound canonical result bytes from exact source and worst-case values."""
    source_sizes = (
        len(encode_strict_json(action.model_dump(mode="json"))),
        len(
            encode_strict_json(
                [generator.model_dump(mode="json") for generator in generators]
            )
        ),
    )
    variables = list(action.variables)
    empty_polynomial = len(
        encode_strict_json(
            {"domain": "QQ", "variables": variables, "polynomial": {"terms": []}}
        )
    )
    maximum_term = len(
        encode_strict_json(
            {
                "coefficient": {
                    "num": "-" + "9" * coefficient_digits,
                    "den": "9" * coefficient_digits,
                },
                "exponents": [MAX_WEIGHT_ACTION_DEGREE] * len(variables),
            }
        )
    )
    basis_size = (
        2
        + max(dimension - 1, 0)
        + dimension * empty_polynomial
        + basis_terms * (maximum_term + 1)
    )
    coordinate_value = len(encode_strict_json({"num": "-" + "9" * 8, "den": "9" * 8}))
    coordinate_size = (
        2
        + max(len(generators) - 1, 0)
        + len(generators)
        * (2 + max(dimension - 1, 0) + dimension * (coordinate_value + 1))
    )
    weights_size = len(
        encode_strict_json(
            [-MAX_DIAGONAL_WEIGHT * MAX_WEIGHT_ACTION_DEGREE] * dimension
        )
    )
    empty_matrix_entry = len(
        encode_strict_json({"domain": "QQ", "variables": [parameter], "terms": []})
    )
    nonzero_matrix_entry = len(
        encode_strict_json(
            {
                "domain": "QQ",
                "variables": [parameter],
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [-MAX_DIAGONAL_WEIGHT * MAX_WEIGHT_ACTION_DEGREE],
                    }
                ],
            }
        )
    )
    matrix_size = (
        2
        + max(dimension - 1, 0)
        + dimension
        * (
            2
            + max(dimension - 1, 0)
            + dimension * empty_matrix_entry
            + max(dimension - 1, 0)
            + nonzero_matrix_entry
        )
    )
    return strict_json_object_size(
        (
            ("action", source_sizes[0]),
            ("generators", source_sizes[1]),
            ("basis", basis_size),
            ("weights", weights_size),
            ("generator_coordinates", coordinate_size),
            ("parameter", len(encode_strict_json(parameter))),
            ("matrix", matrix_size),
        )
    )


def _rref_coefficient_digit_bound(
    projected_generators: dict[int, dict[int, dict[tuple[int, ...], Fraction]]],
    monomials_by_weight: dict[int, tuple[tuple[int, ...], ...]],
) -> int:
    """Bound RREF coefficients by exact row clearing and Hadamard minors.

    For each projected row, clear its exact common denominator. Row scaling
    multiplies every minor using that row by the same factor, so it cancels in
    the minor ratios that give RREF entries. Hadamard bounds each integer
    minor by ``k**k`` times the product of its row maximum entries. The digit
    estimate below uses the largest ``k`` row maxima, where ``k`` is the
    smaller of row count and support width.
    """
    maximum_digits = 1
    for weight, rows in projected_generators.items():
        minor_order = min(len(rows), len(monomials_by_weight[weight]))
        if minor_order == 0:
            continue
        row_maxima = []
        for row in rows.values():
            common_denominator = lcm(
                *(coefficient.denominator for coefficient in row.values())
            )
            row_maxima.append(
                max(
                    abs(
                        coefficient.numerator
                        * (common_denominator // coefficient.denominator)
                    )
                    for coefficient in row.values()
                )
            )
        largest_row_digits = sum(
            decimal_digit_width(maximum)
            for maximum in sorted(row_maxima, reverse=True)[:minor_order]
        )
        hadamard_digits = largest_row_digits + minor_order * len(str(minor_order)) + 1
        maximum_digits = max(maximum_digits, hadamard_digits)
    return maximum_digits


def _weight_projection_rref(
    projected_generators: dict[int, dict[int, dict[tuple[int, ...], Fraction]]],
    monomials_by_weight: dict[int, tuple[tuple[int, ...], ...]],
) -> dict[int, list[tuple[int, tuple[Fraction, ...]]]]:
    """Return one deterministic RREF basis in each exact character space."""
    basis_by_weight: dict[int, list[tuple[int, tuple[Fraction, ...]]]] = {}
    for weight in sorted(monomials_by_weight):
        monomials = monomials_by_weight[weight]
        pivot_rows: dict[int, list[Fraction]] = {}
        for coefficients in projected_generators.get(weight, {}).values():
            row = [coefficients.get(monomial, Fraction(0)) for monomial in monomials]
            for pivot in sorted(pivot_rows):
                factor = row[pivot]
                if factor:
                    row = [
                        value - factor * pivot_value
                        for value, pivot_value in zip(
                            row, pivot_rows[pivot], strict=True
                        )
                    ]
            pivot = next((index for index, value in enumerate(row) if value), None)
            if pivot is None:
                continue
            scale = row[pivot]
            row = [value / scale for value in row]
            for existing_pivot, existing_row in tuple(pivot_rows.items()):
                factor = existing_row[pivot]
                if factor:
                    pivot_rows[existing_pivot] = [
                        value - factor * pivot_value
                        for value, pivot_value in zip(existing_row, row, strict=True)
                    ]
            pivot_rows[pivot] = row
        if pivot_rows:
            basis_by_weight[weight] = [
                (pivot, tuple(pivot_rows[pivot])) for pivot in sorted(pivot_rows)
            ]
    return basis_by_weight


def _project_generators_by_weight(
    action: PolynomialWeightAction,
    generators: tuple[RationalPolynomial, ...],
) -> dict[int, dict[int, dict[tuple[int, ...], Fraction]]]:
    """Build admitted sparse weight projections in one pass over source terms."""
    projections: dict[int, dict[int, dict[tuple[int, ...], Fraction]]] = {}
    for source_index, generator in enumerate(generators):
        for term in generator.polynomial.terms:
            weight = sum(
                variable_weight * exponent
                for variable_weight, exponent in zip(
                    action.weights, term.exponents, strict=True
                )
            )
            projections.setdefault(weight, {}).setdefault(source_index, {})[
                tuple(term.exponents)
            ] = term.coefficient.as_fraction()
    return projections


def gm_generated_subrepresentation(
    request: PolynomialWeightSubrepresentationRequest | Mapping[str, Any],
) -> PolynomialWeightSubrepresentationResult:
    """Return the smallest G_m-stable span generated by supplied polynomials.

    A diagonal torus coaction separates into integer character spaces. The
    stable span generated by finitely many polynomials is therefore the span
    of their weight-homogeneous projections. Each character-space span is
    reduced to a deterministic RREF basis in the source monomial coordinates.
    """
    checked, action, generators = _parse_subrepresentation_request(request)
    monomials_by_weight, sources_by_weight = _admit_subrepresentation_support(
        action, generators, checked.parameter
    )
    projections = _project_generators_by_weight(action, generators)
    coefficient_digits = _rref_coefficient_digit_bound(projections, monomials_by_weight)
    if coefficient_digits > MAX_GM_SUBREP_BASIS_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="gm_subrepresentation.basis_coefficient_digits",
            message=(
                "Hadamard's exact row-denominator bound exceeds the "
                f"{MAX_GM_SUBREP_BASIS_COEFFICIENT_DIGITS}-digit basis coefficient envelope"
            ),
        )
    basis_term_bound = sum(
        len(monomials_by_weight[weight]) * len(sources)
        for weight, sources in sources_by_weight.items()
    )
    dimension_bound = sum(len(sources) for sources in sources_by_weight.values())
    predicted_output_bytes = _subrepresentation_result_size_bound(
        action,
        generators,
        checked.parameter,
        basis_terms=basis_term_bound,
        dimension=dimension_bound,
        coefficient_digits=coefficient_digits,
    )
    if predicted_output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="gm_subrepresentation.output_budget",
            message=(
                "the exact stable-basis and representation result exceeds the "
                f"{CanonicalLimits().max_output_bytes}-byte canonical output envelope"
            ),
        )
    basis_by_weight = _weight_projection_rref(projections, monomials_by_weight)

    dimension = sum(len(rows) for rows in basis_by_weight.values())
    if dimension > MAX_GM_SUBREP_DIMENSION:
        raise AssertionError("the row-space dimension exceeds its support bound")

    basis_polynomials: list[RationalPolynomial] = []
    basis_weights: list[int] = []
    pivot_monomials: list[tuple[int, ...]] = []
    for weight, rows in basis_by_weight.items():
        monomials = monomials_by_weight[weight]
        for pivot, row in rows:
            terms = tuple(
                RationalPolynomialTerm(
                    coefficient=_canonical_rational(coefficient), exponents=monomial
                )
                for monomial, coefficient in zip(monomials, row, strict=True)
                if coefficient
            )
            basis_polynomials.append(
                RationalPolynomial(
                    variables=action.variables,
                    polynomial=SparseRationalPolynomial(terms=terms),
                )
            )
            basis_weights.append(weight)
            pivot_monomials.append(monomials[pivot])

    # RREF pivots are identity coordinates. Since distinct weight spaces have
    # disjoint monomial supports, reading each source coefficient at the
    # corresponding pivot gives its exact coordinates in the full basis.
    generator_coordinates = []
    for generator in generators:
        coefficients = {
            tuple(term.exponents): term.coefficient.as_fraction()
            for term in generator.polynomial.terms
        }
        generator_coordinates.append(
            tuple(
                _canonical_rational(coefficients.get(pivot, Fraction(0)))
                for pivot in pivot_monomials
            )
        )

    parameter = checked.parameter
    zero_entry = RationalLaurentPolynomial(variables=(parameter,), terms=())
    matrix_rows = []
    for row_index, _weight in enumerate(basis_weights):
        row = []
        for column_index, column_weight in enumerate(basis_weights):
            if row_index == column_index:
                row.append(
                    RationalLaurentPolynomial(
                        variables=(parameter,),
                        terms=(
                            RationalLaurentPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=(column_weight,),
                            ),
                        ),
                    )
                )
            else:
                row.append(zero_entry)
        matrix_rows.append(tuple(row))

    return PolynomialWeightSubrepresentationResult(
        action=action,
        generators=generators,
        basis=tuple(basis_polynomials),
        weights=tuple(basis_weights),
        generator_coordinates=tuple(generator_coordinates),
        parameter=parameter,
        matrix=tuple(matrix_rows),
    )


__all__ = [
    "diagonal_weight_action",
    "gm_generated_subrepresentation",
    "gm_invariants_through_degree",
]
