"""Exact finite polynomial subrepresentation computation for a supplied span."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import comb, prod

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import (
    MAX_DERIVATION_COEFFICIENT_DIGITS,
    MAX_DERIVATION_EXPONENT,
    MAX_GA_ACTION_OUTPUT_BYTES,
    MAX_GA_ACTION_OUTPUT_TERMS,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_models import (
    MAX_GA_SUBREPRESENTATION_DIMENSION,
    PolynomialGaStableSubrepresentation,
    PolynomialGaStableSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations.operations import (
    _encode,
    _multiply,
    _term_map,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_GA_SUBREPRESENTATION_SOURCE_TERMS = 128
MAX_GA_SUBREPRESENTATION_EXPANSION_WORK = 1_000_000
MAX_GA_SUBREPRESENTATION_COORDINATE_WORK = 5_000_000
MAX_GA_SUBREPRESENTATION_COMPOSITION_WORK = 25_000

_Terms = dict[tuple[int, ...], Fraction]


def _reject(code: str, message: str, *, resource: bool = False) -> None:
    error = (
        OperationResourceAdmissionError if resource else OperationDomainValidationError
    )
    raise error(
        location=("basis",),
        code=f"polynomial_ga_subrepresentation.{code}",
        message=message,
    )


def _canonical_action(
    value: PolynomialGaAction | Mapping[str, object],
) -> PolynomialGaAction:
    try:
        data = value.model_dump() if isinstance(value, PolynomialGaAction) else value
        return PolynomialGaAction.model_validate(data)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("action",),
            code="polynomial_ga_subrepresentation.action_shape",
            message="action must be a valid checked polynomial Ga action",
        ) from exc


def _substitute_basis(
    action: PolynomialGaAction, polynomial: RationalPolynomial
) -> _Terms:
    """Expand f(a_t(x)) after the conservative whole-input preflight."""
    axis_size = len(action.source_variables) + 1
    one: _Terms = {tuple(0 for _ in range(axis_size)): Fraction(1)}
    images = tuple(_term_map(image) for image in action.generator_images)
    result: _Terms = {}
    for term in polynomial.polynomial.terms:
        product_terms = dict(one)
        for image, exponent in zip(images, term.exponents, strict=True):
            for _ in range(exponent):
                product_terms = _multiply(product_terms, image)
        scalar = term.coefficient.as_fraction()
        for monomial, coefficient in product_terms.items():
            result[monomial] = result.get(monomial, Fraction(0)) + scalar * coefficient
    return {
        monomial: coefficient for monomial, coefficient in result.items() if coefficient
    }


def _admit_and_verify_ga_action(action: PolynomialGaAction) -> None:
    """Check counit and additive composition on generator images, bounded first."""
    action_terms = tuple(_term_map(image) for image in action.generator_images)
    variables = action.source_variables
    max_digit = max(
        (
            max(len(str(abs(value.numerator))), len(str(value.denominator)))
            for image in action_terms
            for value in image.values()
        ),
        default=1,
    )
    lhs_work = 0
    rhs_work = 0
    lhs_terms = 0
    rhs_terms = 0
    max_source_degree = 0
    max_parameter_degree = 0
    for image in action_terms:
        for exponents in image:
            factor = prod(
                max(1, len(generator_image)) ** exponent
                for generator_image, exponent in zip(
                    action_terms, exponents[:-1], strict=True
                )
            )
            degree = sum(exponents[:-1])
            max_source_degree = max(max_source_degree, degree)
            max_parameter_degree = max(max_parameter_degree, exponents[-1])
            lhs_work += max(1, degree) * factor
            lhs_terms += factor
            rhs_work += exponents[-1] + 1
            rhs_terms += exponents[-1] + 1
            lhs_digits = max_digit + factor * (
                degree * (2 * max_digit + 1) + len(str(max(1, factor)))
            )
            rhs_digits = max_digit + exponents[-1] * 2 + len(str(exponents[-1] + 1))
            if max(lhs_digits, rhs_digits) > MAX_DERIVATION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("action",),
                    code="polynomial_ga_subrepresentation.action_coefficient_growth",
                    message="additive action-law check exceeds the coefficient digit budget",
                )
    lhs_aggregate_digits = max_digit + lhs_terms * (
        max_source_degree * (2 * max_digit + 1) + len(str(max(1, lhs_terms)))
    )
    rhs_aggregate_digits = max_digit + rhs_terms * (
        max_parameter_degree + len(str(max(1, rhs_terms)))
    )
    if (
        max(lhs_aggregate_digits, rhs_aggregate_digits)
        > MAX_DERIVATION_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("action",),
            code="polynomial_ga_subrepresentation.action_coefficient_growth",
            message="combined additive action-law coefficients exceed the digit budget",
        )
    if (
        lhs_work + rhs_work > MAX_GA_SUBREPRESENTATION_COMPOSITION_WORK
        or max(lhs_terms, rhs_terms) > MAX_GA_SUBREPRESENTATION_COMPOSITION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("action",),
            code="polynomial_ga_subrepresentation.action_law_work",
            message="additive action-law check exceeds the admitted work budget",
        )

    for index, image in enumerate(action_terms):
        counit = {
            exponents[:-1]: coefficient
            for exponents, coefficient in image.items()
            if exponents[-1] == 0
        }
        expected_generator = {
            tuple(
                1 if axis == index else 0 for axis in range(len(variables))
            ): Fraction(1)
        }
        if counit != expected_generator:
            raise OperationDomainValidationError(
                location=("action", "generator_images", index),
                code="polynomial_ga_subrepresentation.action_counit",
                message="action does not satisfy the additive identity law",
            )

        lhs: _Terms = {}
        for exponents, coefficient in image.items():
            product_terms: _Terms = {
                tuple(0 for _ in range(len(variables) + 2)): Fraction(1)
            }
            for generator_image, exponent in zip(
                action_terms, exponents[:-1], strict=True
            ):
                lifted = {
                    (*powers[:-1], powers[-1], 0): value
                    for powers, value in generator_image.items()
                }
                for _ in range(exponent):
                    product_terms = _multiply(product_terms, lifted)
            for powers, value in product_terms.items():
                key = (*powers[:-1], powers[-1] + exponents[-1])
                lhs[key] = lhs.get(key, Fraction(0)) + coefficient * value

        rhs: _Terms = {}
        for exponents, coefficient in image.items():
            parameter_degree = exponents[-1]
            for s_degree in range(parameter_degree + 1):
                key = (*exponents[:-1], s_degree, parameter_degree - s_degree)
                rhs[key] = rhs.get(key, Fraction(0)) + coefficient * comb(
                    parameter_degree, s_degree
                )
        lhs = {key: value for key, value in lhs.items() if value}
        rhs = {key: value for key, value in rhs.items() if value}
        if lhs != rhs:
            raise OperationDomainValidationError(
                location=("action", "generator_images", index),
                code="polynomial_ga_subrepresentation.action_composition",
                message="action does not satisfy the additive composition law",
            )


def _coordinate_frame(
    basis: tuple[RationalPolynomial, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[Fraction, ...], ...], int]:
    """Precompute an exact coordinate map from independent monomial rows."""
    size = len(basis)
    support = sorted(
        {term.exponents for value in basis for term in value.polynomial.terms},
        reverse=True,
    )
    basis_term_count = sum(len(value.polynomial.terms) for value in basis)
    coordinate_work = len(support) * (basis_term_count + size * size)
    if coordinate_work > MAX_GA_SUBREPRESENTATION_COORDINATE_WORK:
        _reject(
            "coordinate_work",
            "basis-coordinate computation exceeds the work budget",
            resource=True,
        )
    source_rows = [
        [
            next(
                (
                    term.coefficient.as_fraction()
                    for term in value.polynomial.terms
                    if term.exponents == monomial
                ),
                Fraction(0),
            )
            for value in basis
        ]
        for monomial in support
    ]
    echelon: list[tuple[int, list[Fraction]]] = []
    pivot_monomials: list[tuple[int, ...]] = []
    for monomial, source in zip(support, source_rows, strict=True):
        row = source.copy()
        for pivot, normalized in echelon:
            if row[pivot]:
                scale = row[pivot]
                row = [a - scale * b for a, b in zip(row, normalized, strict=True)]
                if any(
                    max(len(str(abs(value.numerator))), len(str(value.denominator)))
                    > MAX_DERIVATION_COEFFICIENT_DIGITS
                    for value in row
                ):
                    _reject(
                        "basis_coordinate_growth",
                        "basis coordinate elimination exceeds the coefficient digit budget",
                        resource=True,
                    )
        pivot = next((index for index, value in enumerate(row) if value), None)
        if pivot is None:
            continue
        scale = row[pivot]
        echelon.append((pivot, [value / scale for value in row]))
        pivot_monomials.append(monomial)
        if len(pivot_monomials) == size:
            break
    if len(pivot_monomials) != size:
        _reject(
            "dependent_basis", "supplied basis polynomials must be linearly independent"
        )

    pivot_rows = [
        source_rows[support.index(monomial)].copy() for monomial in pivot_monomials
    ]
    augmented = [
        row + [Fraction(int(i == j)) for j in range(size)]
        for i, row in enumerate(pivot_rows)
    ]
    for column in range(size):
        selected = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if selected is None:
            _reject(
                "dependent_basis",
                "supplied basis polynomials must be linearly independent",
            )
        augmented[column], augmented[selected] = augmented[selected], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column]:
                scale = augmented[row][column]
                augmented[row] = [
                    a - scale * b
                    for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
        if any(
            max(len(str(abs(value.numerator))), len(str(value.denominator)))
            > MAX_DERIVATION_COEFFICIENT_DIGITS
            for row in augmented
            for value in row
        ):
            _reject(
                "basis_coordinate_growth",
                "basis coordinate map exceeds the coefficient digit budget",
                resource=True,
            )
    inverse = tuple(tuple(row[size:]) for row in augmented)
    inverse_digits = max(
        (
            max(len(str(abs(value.numerator))), len(str(value.denominator)))
            for row in inverse
            for value in row
        ),
        default=1,
    )
    return tuple(pivot_monomials), inverse, inverse_digits


def _coordinates(
    basis: tuple[RationalPolynomial, ...],
    pivot_monomials: tuple[tuple[int, ...], ...],
    inverse: tuple[tuple[Fraction, ...], ...],
    target: Mapping[tuple[int, ...], Fraction],
) -> tuple[Fraction, ...] | None:
    """Recover pivot coordinates and verify equality on full monomial support."""
    size = len(basis)
    pivot_values = tuple(
        target.get(monomial, Fraction(0)) for monomial in pivot_monomials
    )
    coordinates = tuple(
        sum(
            (inverse[row][column] * pivot_values[column] for column in range(size)),
            Fraction(0),
        )
        for row in range(size)
    )
    reconstructed: dict[tuple[int, ...], Fraction] = {}
    for value, coordinate in zip(basis, coordinates, strict=True):
        for term in value.polynomial.terms:
            exponents = tuple(term.exponents)
            reconstructed[exponents] = (
                reconstructed.get(exponents, Fraction(0))
                + coordinate * term.coefficient.as_fraction()
            )
    reconstructed = {key: value for key, value in reconstructed.items() if value}
    return coordinates if reconstructed == dict(target) else None


def _prepare_input(
    action: PolynomialGaAction
    | PolynomialGaStableSubrepresentationRequest
    | Mapping[str, object],
    basis: tuple[RationalPolynomial, ...] | None,
) -> tuple[PolynomialGaAction, tuple[RationalPolynomial, ...]]:
    if isinstance(action, PolynomialGaStableSubrepresentationRequest):
        if basis is not None:
            raise OperationDomainValidationError(
                location=("basis",),
                code="polynomial_ga_subrepresentation.request",
                message="basis is already carried by request",
            )
        request = PolynomialGaStableSubrepresentationRequest.model_validate(
            action.model_dump()
        )
        return _canonical_action(request.action), request.basis
    if isinstance(action, Mapping) and basis is None:
        request = PolynomialGaStableSubrepresentationRequest.model_validate(action)
        return _canonical_action(request.action), request.basis
    action_value = _canonical_action(action)  # type: ignore[arg-type]
    if basis is None:
        raise OperationDomainValidationError(
            location=("basis",),
            code="polynomial_ga_subrepresentation.basis_missing",
            message="an ordered basis is required",
        )
    try:
        basis_value = tuple(
            RationalPolynomial.model_validate(value.model_dump()) for value in basis
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("basis",),
            code="polynomial_ga_subrepresentation.basis_shape",
            message="basis entries must be typed rational polynomials",
        ) from exc
    return action_value, basis_value


def _admit_basis(
    action: PolynomialGaAction, basis: tuple[RationalPolynomial, ...]
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[Fraction, ...], ...]]:
    if not 1 <= len(basis) <= MAX_GA_SUBREPRESENTATION_DIMENSION:
        _reject(
            "dimension", "basis dimension exceeds the admitted bound", resource=True
        )
    if any(value.variables != action.source_variables for value in basis):
        _reject("ordered_ring", "basis polynomials must use the action source ring")
    action_term_counts = tuple(
        len(image.polynomial.terms) for image in action.generator_images
    )
    max_action_digit = max(
        (
            max(len(str(abs(term.coefficient.num))), len(str(term.coefficient.den)))
            for image in action.generator_images
            for term in image.polynomial.terms
        ),
        default=1,
    )
    expansion_terms = 0
    expansion_work = len(basis) * sum(action_term_counts)
    max_source_digits = 1
    max_source_degree = 0
    for index, value in enumerate(basis):
        try:
            require_polynomial_budget(
                value,
                maximum_terms=MAX_GA_SUBREPRESENTATION_SOURCE_TERMS,
                maximum_exponent=MAX_DERIVATION_EXPONENT,
                maximum_coefficient_digits=MAX_DERIVATION_COEFFICIENT_DIGITS,
                label=f"basis polynomial {index}",
            )
        except (PydanticCustomError, ValueError) as exc:
            raise OperationResourceAdmissionError(
                location=("basis", index),
                code="polynomial_ga_subrepresentation.source_bound",
                message=str(exc),
            ) from exc
        if any(
            sum(term.exponents) > MAX_DERIVATION_EXPONENT
            for term in value.polynomial.terms
        ):
            _reject(
                "degree_bound",
                "basis polynomial exceeds the admitted total degree",
                resource=True,
            )
        for term in value.polynomial.terms:
            max_source_digits = max(
                max_source_digits,
                len(str(abs(term.coefficient.num))),
                len(str(term.coefficient.den)),
            )
            factor = prod(
                max(1, count) ** exponent
                for count, exponent in zip(
                    action_term_counts, term.exponents, strict=True
                )
            )
            degree = sum(term.exponents)
            max_source_degree = max(max_source_degree, degree)
            expansion_terms += factor
            # Sequential multiplication costs at most degree times the final
            # monomial product bound. Charge that before performing it.
            expansion_work += max(1, degree) * factor
            coefficient_digits = max(
                len(str(abs(term.coefficient.num))), len(str(term.coefficient.den))
            ) + factor * (
                degree * (2 * max_action_digit + 1) + len(str(max(1, factor)))
            )
            if coefficient_digits > MAX_DERIVATION_COEFFICIENT_DIGITS:
                _reject(
                    "coefficient_growth",
                    "predicted action expansion coefficient exceeds the digit budget",
                    resource=True,
                )
    if expansion_terms > MAX_GA_ACTION_OUTPUT_TERMS:
        _reject(
            "expanded_terms",
            "predicted basis action output exceeds the term budget",
            resource=True,
        )
    # Terms from distinct source monomials can collide. Treat every bounded
    # product as a possible summand of one output coefficient, so denominator
    # growth from rational addition is also admitted before expansion.
    aggregate_digit_bound = max_source_digits + expansion_terms * (
        max_source_degree * (2 * max_action_digit + 1)
        + len(str(max(1, expansion_terms)))
    )
    if aggregate_digit_bound > MAX_DERIVATION_COEFFICIENT_DIGITS:
        _reject(
            "coefficient_growth",
            "predicted combined action coefficient exceeds the digit budget",
            resource=True,
        )
    pivot_monomials, inverse, inverse_digits = _coordinate_frame(basis)
    coordinate_digit_bound = len(basis) * (
        inverse_digits + aggregate_digit_bound
    ) + len(str(len(basis)))
    if coordinate_digit_bound > MAX_DERIVATION_COEFFICIENT_DIGITS:
        _reject(
            "matrix_coefficient_growth",
            "predicted action matrix coordinate growth exceeds the digit budget",
            resource=True,
        )
    basis_terms = sum(len(value.polynomial.terms) for value in basis)
    matrix_term_bound = len(basis) * expansion_terms
    action_bytes = (
        sum(len(image.polynomial.terms) for image in action.generator_images) * 384
    )
    action_bytes += sum(len(value.encode("utf-8")) for value in action.source_variables)
    action_bytes += len(action.parameter.encode("utf-8")) + 256
    output_bytes = (
        action_bytes + (basis_terms + expansion_terms + matrix_term_bound) * 384
    )
    output_bytes += len(basis) ** 2 * 256
    if output_bytes > MAX_GA_ACTION_OUTPUT_BYTES:
        _reject(
            "output_bytes",
            "basis and action matrix exceed the serialized-output budget",
            resource=True,
        )
    if expansion_work > MAX_GA_SUBREPRESENTATION_EXPANSION_WORK:
        _reject(
            "expansion_work",
            "action expansion exceeds the preflight work budget",
            resource=True,
        )
    return pivot_monomials, inverse


def _compute_matrix(
    action: PolynomialGaAction,
    basis: tuple[RationalPolynomial, ...],
    pivot_monomials: tuple[tuple[int, ...], ...],
    inverse: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[RationalPolynomial, ...], ...]:
    columns: list[list[dict[int, Fraction]]] = []
    for value in basis:
        expanded = _substitute_basis(action, value)
        by_parameter: dict[int, dict[tuple[int, ...], Fraction]] = {}
        for exponents, coefficient in expanded.items():
            by_parameter.setdefault(exponents[-1], {})[exponents[:-1]] = coefficient
        column: list[dict[int, Fraction]] = [{} for _ in basis]
        for parameter_degree, target in by_parameter.items():
            coordinates = _coordinates(basis, pivot_monomials, inverse, target)
            if coordinates is None:
                _reject(
                    "not_stable",
                    f"action image of supplied basis vector is outside the supplied span (parameter degree {parameter_degree})",
                )
            for row, coefficient in enumerate(coordinates):
                if (
                    max(
                        len(str(abs(coefficient.numerator))),
                        len(str(coefficient.denominator)),
                    )
                    > MAX_DERIVATION_COEFFICIENT_DIGITS
                ):
                    _reject(
                        "matrix_coefficient_growth",
                        "action matrix coefficient exceeds the exact digit budget",
                        resource=True,
                    )
                if coefficient:
                    column[row][parameter_degree] = coefficient
        columns.append(column)
    return tuple(
        tuple(
            _encode(
                (action.parameter,),
                {
                    (degree,): coefficient
                    for degree, coefficient in columns[col][row].items()
                },
            )
            for col in range(len(basis))
        )
        for row in range(len(basis))
    )


def ga_stable_subrepresentation(
    action: PolynomialGaAction
    | PolynomialGaStableSubrepresentationRequest
    | Mapping[str, object],
    basis: tuple[RationalPolynomial, ...] | None = None,
) -> PolynomialGaStableSubrepresentation:
    """Return the exact action matrix on the explicitly supplied finite span.

    The contract checks every supplied basis image and does not search for or
    claim to find other finite-dimensional subrepresentations.
    """
    action_value, basis_value = _prepare_input(action, basis)
    _admit_and_verify_ga_action(action_value)
    pivot_monomials, inverse = _admit_basis(action_value, basis_value)
    matrix = _compute_matrix(action_value, basis_value, pivot_monomials, inverse)
    return PolynomialGaStableSubrepresentation.model_construct(
        action=action_value, basis=basis_value, action_matrix=matrix
    )


__all__ = ["ga_stable_subrepresentation"]
