"""Exact finite polynomial subrepresentation computation for a supplied span."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import comb, factorial, lcm, prod

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
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
    PolynomialGaFixedSubspace,
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
MAX_GA_FIXED_KERNEL_WORK = MAX_GA_SUBREPRESENTATION_DIMENSION**3
MAX_GA_FIXED_COORDINATE_DIGITS = MAX_DERIVATION_COEFFICIENT_DIGITS
MAX_GA_FIXED_ROW_DENOMINATOR_DIGITS = 512
MAX_GA_FIXED_MINOR_DIGITS = 10_000
# With the minor bound above, each Gauss-Jordan update combines products of at
# most three minor-bounded ratios; this caps transient exact integer size too.
MAX_GA_FIXED_INTERMEDIATE_DIGITS = 3 * MAX_GA_FIXED_MINOR_DIGITS + 2
MAX_GA_FIXED_MATRIX_TERMS = MAX_GA_ACTION_OUTPUT_BYTES // 384
MAX_GA_FIXED_SUPPORT_TERMS = 4_096
MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS = 512
# A retained action-matrix or basis term carries an admitted 128-digit
# coefficient, while a fixed representative may use the full 512-digit
# numerator and denominator budgets. Charge the admitted widths rather than a
# 128-digit placeholder so the estimate cannot understate the serialized size.
MAX_GA_FIXED_RETAINED_TERM_BYTES = 384
MAX_GA_FIXED_TERM_BYTES = 2 * MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS + 256


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


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


def _ceil_log_count(count: int) -> int:
    return 0 if count <= 1 else len(str(count - 1))


def _check_coefficient_group_heights(
    groups: Mapping[tuple[int, ...], list[tuple[int, int]]],
) -> None:
    """Bound exact rational sums using support collisions and common denominators."""
    for contributions in groups.values():
        denominators = {denominator for _, denominator in contributions}
        if (
            sum(len(str(value)) for value in denominators if value != 1)
            > MAX_DERIVATION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("action",),
                code="polynomial_ga_subrepresentation.action_coefficient_growth",
                message="action-law common denominator exceeds the coefficient digit budget",
            )
        common_denominator = lcm(*denominators)
        if len(str(common_denominator)) > MAX_DERIVATION_COEFFICIENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("action",),
                code="polynomial_ga_subrepresentation.action_coefficient_growth",
                message="action-law common denominator exceeds the coefficient digit budget",
            )
        max_numerator_digits = max(
            numerator_digits
            + (
                len(str(common_denominator // denominator))
                if common_denominator // denominator > 1
                else 0
            )
            for numerator_digits, denominator in contributions
        )
        if (
            max_numerator_digits + _ceil_log_count(len(contributions))
            > MAX_DERIVATION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("action",),
                code="polynomial_ga_subrepresentation.action_coefficient_growth",
                message="action-law numerator growth exceeds the coefficient digit budget",
            )


def _verify_action_laws(
    action_terms: tuple[_Terms, ...], variables: tuple[str, ...]
) -> None:
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
            degree = exponents[-1]
            for s_degree in range(degree + 1):
                key = (*exponents[:-1], s_degree, degree - s_degree)
                rhs[key] = rhs.get(key, Fraction(0)) + coefficient * comb(
                    degree, s_degree
                )
        lhs = {key: value for key, value in lhs.items() if value}
        rhs = {key: value for key, value in rhs.items() if value}
        if lhs != rhs:
            raise OperationDomainValidationError(
                location=("action", "generator_images", index),
                code="polynomial_ga_subrepresentation.action_composition",
                message="action does not satisfy the additive composition law",
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
    lhs_work = 0
    rhs_work = 0
    lhs_terms = 0
    rhs_terms = 0
    for image in action_terms:
        for exponents in image:
            factor = prod(
                max(1, len(generator_image)) ** exponent
                for generator_image, exponent in zip(
                    action_terms, exponents[:-1], strict=True
                )
            )
            degree = sum(exponents[:-1])
            lhs_work += max(1, degree) * factor
            lhs_terms += factor
            rhs_work += exponents[-1] + 1
            rhs_terms += exponents[-1] + 1
    if (
        lhs_work + rhs_work > MAX_GA_SUBREPRESENTATION_COMPOSITION_WORK
        or max(lhs_terms, rhs_terms) > MAX_GA_SUBREPRESENTATION_COMPOSITION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("action",),
            code="polynomial_ga_subrepresentation.action_law_work",
            message="additive action-law check exceeds the admitted work budget",
        )

    # Estimate rational heights per possible output monomial. This keeps a
    # large coefficient attached to one sparse support term from being charged
    # as if it were multiplied by every unrelated term in the action image.
    source_axis_count = len(variables)
    common_denominators: list[int] = []
    max_scaled_numerator_digits: list[int] = []
    support_images: list[tuple[tuple[int, ...], ...]] = []
    for image in action_terms:
        image_denominators = {value.denominator for value in image.values()}
        if (
            sum(len(str(value)) for value in image_denominators if value != 1)
            > MAX_DERIVATION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("action",),
                code="polynomial_ga_subrepresentation.action_coefficient_growth",
                message="action coefficient common denominator exceeds the digit budget",
            )
        denominator = lcm(*image_denominators) if image_denominators else 1
        common_denominators.append(denominator)
        scaled = [
            abs(value.numerator) * (denominator // value.denominator)
            for value in image.values()
        ]
        max_scaled_numerator_digits.append(
            max((len(str(value)) for value in scaled), default=1)
        )
        support_images.append(tuple((*powers[:-1], powers[-1]) for powers in image))

    lhs_height_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    rhs_height_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for image in action_terms:
        for exponents, coefficient in image.items():
            support_product: dict[tuple[int, ...], int] = {
                tuple(0 for _ in range(source_axis_count + 1)): 1
            }
            for support, exponent in zip(support_images, exponents[:-1], strict=True):
                for _ in range(exponent):
                    multiplied: dict[tuple[int, ...], int] = {}
                    for left, left_count in support_product.items():
                        for right in support:
                            key = tuple(a + b for a, b in zip(left, right, strict=True))
                            multiplied[key] = multiplied.get(key, 0) + left_count
                    support_product = multiplied
            denominator_digits = len(str(coefficient.denominator)) + sum(
                exponent * (len(str(base)) if base > 1 else 0)
                for base, exponent in zip(
                    common_denominators, exponents[:-1], strict=True
                )
            )
            if denominator_digits > MAX_DERIVATION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("action",),
                    code="polynomial_ga_subrepresentation.action_coefficient_growth",
                    message="composition coefficient denominator exceeds the digit budget",
                )
            denominator = coefficient.denominator * prod(
                base**exponent
                for base, exponent in zip(
                    common_denominators, exponents[:-1], strict=True
                )
            )
            numerator_digits = len(str(abs(coefficient.numerator))) + sum(
                exponent * digits
                for exponent, digits in zip(
                    exponents[:-1], max_scaled_numerator_digits, strict=True
                )
            )
            for support_powers, multiplicity in support_product.items():
                key = (*support_powers[:-1], support_powers[-1], exponents[-1])
                lhs_height_groups.setdefault(key, []).append(
                    (numerator_digits + _ceil_log_count(multiplicity), denominator)
                )
            for s_degree in range(exponents[-1] + 1):
                binomial = comb(exponents[-1], s_degree)
                key = (*exponents[:-1], s_degree, exponents[-1] - s_degree)
                rhs_height_groups.setdefault(key, []).append(
                    (
                        len(str(abs(coefficient.numerator)))
                        + (len(str(binomial)) if binomial > 1 else 0),
                        coefficient.denominator,
                    )
                )
    _check_coefficient_group_heights(lhs_height_groups)
    _check_coefficient_group_heights(rhs_height_groups)

    _verify_action_laws(action_terms, variables)


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


def _infinitesimal_matrix(
    representation: PolynomialGaStableSubrepresentation,
) -> list[list[Fraction]]:
    size = len(representation.basis)
    matrix = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            for term in representation.action_matrix[row][column].polynomial.terms:
                if term.exponents == (1,):
                    matrix[row][column] = term.coefficient.as_fraction()
                    break
    return matrix


def _admit_and_integerize_kernel_matrix(
    matrix: list[list[Fraction]],
) -> list[list[Fraction]]:
    """Clear row denominators under an exact Hadamard bound.

    Every minor of the integer matrix has at most ``MAX_GA_FIXED_MINOR_DIGITS``
    decimal digits. Gauss-Jordan entries are ratios of such minors. An update
    multiplies two ratios and subtracts a third, so unreduced temporaries have
    at most three times this digit bound plus two digits. The bound on row
    denominators is checked before clearing, and the Hadamard bound is checked
    before elimination.
    """
    size = len(matrix)
    if size**3 > MAX_GA_FIXED_KERNEL_WORK:
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.kernel_work",
            message="exact fixed-space elimination exceeds its admitted work budget",
        )
    if MAX_GA_FIXED_INTERMEDIATE_DIGITS > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.intermediate_growth",
            message="the exact elimination intermediate envelope exceeds the integer limit",
        )

    row_denominators: list[int] = []
    for row in matrix:
        if any(
            max(_integer_digits(value.numerator), _integer_digits(value.denominator))
            > MAX_DERIVATION_COEFFICIENT_DIGITS
            for value in row
        ):
            raise OperationResourceAdmissionError(
                location=("subrepresentation",),
                code="polynomial_ga_fixed_subspace.matrix_coefficient_growth",
                message=(
                    "infinitesimal matrix coefficients exceed the "
                    f"{MAX_DERIVATION_COEFFICIENT_DIGITS}-digit input envelope"
                ),
            )
        denominator_width_bound = sum(len(str(value.denominator)) for value in row)
        if denominator_width_bound > MAX_GA_FIXED_ROW_DENOMINATOR_DIGITS:
            raise OperationResourceAdmissionError(
                location=("subrepresentation",),
                code="polynomial_ga_fixed_subspace.denominator_growth",
                message=(
                    "fixed-space row denominator lcm exceeds the "
                    f"{MAX_GA_FIXED_ROW_DENOMINATOR_DIGITS}-digit preflight bound"
                ),
            )
        row_denominators.append(lcm(*(value.denominator for value in row)))

    integer_matrix: list[list[Fraction]] = []
    maximum_entry_digits = 1
    for row, denominator in zip(matrix, row_denominators, strict=True):
        integer_row = []
        for value in row:
            integer = value.numerator * (denominator // value.denominator)
            maximum_entry_digits = max(maximum_entry_digits, _integer_digits(integer))
            integer_row.append(Fraction(integer))
        integer_matrix.append(integer_row)

    minor_digits_bound = size * maximum_entry_digits + len(str(factorial(size)))
    if minor_digits_bound > MAX_GA_FIXED_MINOR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.minor_growth",
            message=(
                "the Hadamard bound for exact fixed-space minors is "
                f"{minor_digits_bound} digits, exceeding the "
                f"{MAX_GA_FIXED_MINOR_DIGITS}-digit envelope"
            ),
        )
    return integer_matrix


def _rational_kernel_basis(matrix: list[list[Fraction]]) -> list[tuple[Fraction, ...]]:
    """Return a deterministic nullspace basis by exact reduced row elimination."""
    size = len(matrix)
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(size):
        selected = next(
            (row for row in range(pivot_row, size) if matrix[row][column]), None
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot for value in matrix[pivot_row]]
        for row in range(size):
            if row != pivot_row and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        matrix[row], matrix[pivot_row], strict=True
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == size:
            break
    pivot_rows = {column: row for row, column in enumerate(pivot_columns)}
    free_columns = [column for column in range(size) if column not in pivot_rows]
    basis = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(size)]
        vector[free] = Fraction(1)
        for pivot_column, row in pivot_rows.items():
            vector[pivot_column] = -matrix[row][free]
        basis.append(tuple(vector))
    return basis


def _admit_claimed_action_matrix(
    representation: PolynomialGaStableSubrepresentation,
) -> None:
    """Bound a caller-supplied matrix before reconstruction and comparison.

    The shared wire carrier admits 32,768-digit rational components and
    exponents, so a term-count check alone would let a mismatched claim drive
    a full stable-representation reconstruction and an equality comparison
    over hundreds of millions of digits. Every claimed term must fit the
    operation's admitted input widths, and the aggregate encoded size must
    fit the output envelope.
    """
    max_image_parameter_degree = max(
        (
            term.exponents[-1]
            for image in representation.action.generator_images
            for term in image.polynomial.terms
        ),
        default=0,
    )
    max_basis_degree = max(
        (
            sum(term.exponents)
            for value in representation.basis
            for term in value.polynomial.terms
        ),
        default=0,
    )
    degree_bound = max_image_parameter_degree * max_basis_degree
    term_count = 0
    encoded_bytes = 0
    for row in representation.action_matrix:
        for entry in row:
            for term in entry.polynomial.terms:
                numerator_digits = len(str(abs(term.coefficient.num)))
                denominator_digits = len(str(term.coefficient.den))
                if (
                    numerator_digits > MAX_DERIVATION_COEFFICIENT_DIGITS
                    or denominator_digits > MAX_DERIVATION_COEFFICIENT_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=("subrepresentation", "action_matrix"),
                        code="polynomial_ga_fixed_subspace.matrix_coefficient_growth",
                        message=(
                            "supplied action matrix coefficients exceed the "
                            f"{MAX_DERIVATION_COEFFICIENT_DIGITS}-digit input "
                            "envelope"
                        ),
                    )
                if any(exponent > degree_bound for exponent in term.exponents):
                    raise OperationResourceAdmissionError(
                        location=("subrepresentation", "action_matrix"),
                        code="polynomial_ga_fixed_subspace.matrix_degree_growth",
                        message=(
                            "supplied action matrix exponents exceed the "
                            f"{degree_bound}-degree action envelope"
                        ),
                    )
                term_count += 1
                encoded_bytes += numerator_digits + denominator_digits + 128
    if (
        term_count > MAX_GA_FIXED_MATRIX_TERMS
        or encoded_bytes > MAX_GA_ACTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("subrepresentation", "action_matrix"),
            code="polynomial_ga_fixed_subspace.matrix_terms",
            message="the supplied action matrix exceeds the bounded term envelope",
        )


def _admit_fixed_output_support(
    representation: PolynomialGaStableSubrepresentation,
    coordinates: list[tuple[Fraction, ...]],
) -> int:
    candidate_terms = sum(
        len(polynomial.polynomial.terms)
        for vector in coordinates
        for coefficient, polynomial in zip(vector, representation.basis, strict=True)
        if coefficient
    )
    if candidate_terms > MAX_GA_FIXED_SUPPORT_TERMS:
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.output_terms",
            message=(
                "fixed polynomial representatives have an aggregate support "
                "upper bound above the 4096-term envelope"
            ),
        )
    return candidate_terms


def _admit_fixed_result_bytes(
    representation: PolynomialGaStableSubrepresentation,
    coordinates: list[tuple[Fraction, ...]],
    fixed_support_terms: int,
) -> int:
    """Bound the serialized result before representatives are constructed.

    Retained representation terms carry admitted 128-digit coefficients, while
    a fixed representative may use the full admitted 512-digit numerator and
    denominator budgets. Coordinate cells are charged their actual admitted
    component widths rather than a placeholder, so a result near any of these
    ceilings cannot slip past the advertised output-byte envelope.
    """
    retained_terms = (
        sum(
            len(image.polynomial.terms)
            for image in representation.action.generator_images
        )
        + sum(len(value.polynomial.terms) for value in representation.basis)
        + sum(
            len(entry.polynomial.terms)
            for row in representation.action_matrix
            for entry in row
        )
    )
    coordinate_bytes = sum(
        _integer_digits(value.numerator) + _integer_digits(value.denominator) + 128
        for vector in coordinates
        for value in vector
    )
    estimated_result_bytes = retained_terms * MAX_GA_FIXED_RETAINED_TERM_BYTES
    estimated_result_bytes += fixed_support_terms * MAX_GA_FIXED_TERM_BYTES
    estimated_result_bytes += coordinate_bytes
    estimated_result_bytes += len(representation.basis) ** 2 * 256 + 1_024
    if estimated_result_bytes > MAX_GA_ACTION_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.output_bytes",
            message=(
                "the fixed basis and retained representation exceed the "
                "output-byte envelope"
            ),
        )
    return estimated_result_bytes


def _fixed_polynomial_basis(
    representation: PolynomialGaStableSubrepresentation,
    coordinates: list[tuple[Fraction, ...]],
) -> tuple[RationalPolynomial, ...]:
    from math import gcd

    from jacobian.math.polynomials.values import (
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    output = []
    for vector in coordinates:
        grouped: dict[tuple[int, ...], list[Fraction]] = {}
        for coordinate, polynomial in zip(vector, representation.basis, strict=True):
            for term in polynomial.polynomial.terms:
                if coordinate:
                    grouped.setdefault(term.exponents, []).append(
                        coordinate * term.coefficient.as_fraction()
                    )
        terms = []
        for exponents, contributions in grouped.items():
            denominator = 1
            for value in contributions:
                factor = denominator // gcd(denominator, value.denominator)
                if (
                    len(str(factor)) + len(str(value.denominator))
                    > MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=("subrepresentation",),
                        code="polynomial_ga_fixed_subspace.coefficient_growth",
                        message=(
                            "fixed polynomial coefficient denominator exceeds "
                            f"{MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS} digits"
                        ),
                    )
                denominator = factor * value.denominator
            numerator_bound = max(
                len(str(abs(value.numerator)))
                + len(str(denominator // value.denominator))
                for value in contributions
            ) + len(str(len(contributions)))
            if numerator_bound > MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("subrepresentation",),
                    code="polynomial_ga_fixed_subspace.coefficient_growth",
                    message=(
                        "fixed polynomial coefficient numerator exceeds "
                        f"{MAX_GA_FIXED_OUTPUT_COEFFICIENT_DIGITS} digits"
                    ),
                )
            coefficient = sum(contributions, Fraction(0))
            if coefficient:
                terms.append(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_fraction(coefficient),
                        exponents=exponents,
                    )
                )
        output.append(
            RationalPolynomial(
                variables=representation.action.source_variables,
                polynomial=SparseRationalPolynomial(
                    terms=tuple(
                        sorted(terms, key=lambda term: term.exponents, reverse=True)
                    )
                ),
            )
        )
    return tuple(output)


def ga_fixed_subspace(
    subrepresentation: PolynomialGaStableSubrepresentation | Mapping[str, object],
) -> PolynomialGaFixedSubspace:
    """Compute the exact fixed subspace of a checked finite Ga-representation.

    In characteristic zero, the fixed vectors are the kernel of the coefficient
    of ``t`` in the representation matrix: differentiating the group law gives
    ``M'(t)=M(t)M'(0)``, so this kernel is fixed for every parameter value.
    """
    from jacobian.math.polynomials.derivations._stable_models import (
        PolynomialGaStableSubrepresentation,
    )

    try:
        # Native callers can bypass validation through model_construct or
        # model_copy(update=...), so serialize and revalidate typed instances
        # here as the stable-subrepresentation boundary does.
        claim = PolynomialGaStableSubrepresentation.model_validate(
            subrepresentation.model_dump(warnings=False)
            if isinstance(subrepresentation, PolynomialGaStableSubrepresentation)
            else subrepresentation
        )
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.request_shape",
            message="the request must contain a canonical finite Ga-subrepresentation",
        ) from exc

    _admit_claimed_action_matrix(claim)

    # A consumer may rely on the claimed matrix only after checking its defining
    # relation against the action and basis. The producer has operation-specific
    # admission for this exact reconstruction.
    checked = ga_stable_subrepresentation(claim.action, claim.basis)
    if checked.action_matrix != claim.action_matrix:
        raise OperationDomainValidationError(
            location=("subrepresentation", "action_matrix"),
            code="polynomial_ga_fixed_subspace.unverified_subrepresentation",
            message="the supplied matrix is not the action matrix of the supplied basis",
        )

    integer_matrix = _admit_and_integerize_kernel_matrix(_infinitesimal_matrix(checked))
    coordinate_vectors = _rational_kernel_basis(integer_matrix)
    if any(
        max(_integer_digits(value.numerator), _integer_digits(value.denominator))
        > MAX_GA_FIXED_COORDINATE_DIGITS
        for vector in coordinate_vectors
        for value in vector
    ):
        raise OperationResourceAdmissionError(
            location=("subrepresentation",),
            code="polynomial_ga_fixed_subspace.coordinate_growth",
            message=(
                "fixed-space coordinates exceed the "
                f"{MAX_GA_FIXED_COORDINATE_DIGITS}-digit envelope"
            ),
        )
    fixed_candidate_terms = _admit_fixed_output_support(checked, coordinate_vectors)
    _admit_fixed_result_bytes(checked, coordinate_vectors, fixed_candidate_terms)
    fixed_basis = _fixed_polynomial_basis(checked, coordinate_vectors)

    return PolynomialGaFixedSubspace.model_construct(
        subrepresentation=checked,
        coordinates=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in vector)
            for vector in coordinate_vectors
        ),
        basis=tuple(fixed_basis),
    )


__all__ = ["ga_fixed_subspace", "ga_stable_subrepresentation"]
