"""Typed wire contracts for symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import combinations, permutations, product
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalFunction,
    SparseRationalPolynomial,
)

MAX_SYMBOLIC_MATRIX_DIMENSION = 8
MAX_SYMBOLIC_VARIABLES = 8
MAX_SYMBOLIC_MATRIX_TERMS = 512
MAX_SYMBOLIC_RESULT_TERMS = 256
MAX_SYMBOLIC_RESULT_EXPONENT = 64
MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS = 128


def _is_polynomial_entry(value: RationalFunction) -> bool:
    terms = value.denominator.terms
    return (
        len(terms) == 1
        and terms[0].coefficient.num == "1"
        and terms[0].coefficient.den == "1"
        and all(exponent == 0 for exponent in terms[0].exponents)
    )


def _is_scalar_identity(value: RationalFunction) -> bool:
    terms = value.numerator.terms
    return (
        len(terms) == 1
        and terms[0].coefficient.num == "1"
        and terms[0].coefficient.den == "1"
        and all(exponent == 0 for exponent in terms[0].exponents)
        and _is_polynomial_entry(value)
    )


def _principal_minor_term_bounds(
    entries: tuple[tuple[RationalFunction, ...], ...],
) -> tuple[int, ...]:
    """Bound raw terms in each characteristic coefficient by Leibniz expansion."""

    dimension = len(entries)
    bounds = [1]
    for size in range(1, dimension + 1):
        coefficient_terms = 0
        for axes in combinations(range(dimension), size):
            for columns in permutations(axes):
                product_terms = 1
                for row, column in zip(axes, columns, strict=True):
                    product_terms *= len(entries[row][column].numerator.terms)
                coefficient_terms += product_terms
        bounds.append(coefficient_terms)
    return tuple(bounds)


def _require_determinant_family_result_budget(
    matrix: SymbolicMatrix,
    *,
    characteristic_polynomial: bool,
) -> None:
    dimension = len(matrix.entries)
    if dimension == 1:
        return
    values = tuple(value for row in matrix.entries for value in row)
    if any(not _is_polynomial_entry(value) for value in values):
        raise _validation_error(
            "budget_exceeded",
            "multi-dimensional determinant-family requests require polynomial entries",
        )
    term_bounds = _principal_minor_term_bounds(matrix.entries)
    relevant_bounds = term_bounds[1:] if characteristic_polynomial else term_bounds[-1:]
    if any(bound > MAX_SYMBOLIC_RESULT_TERMS for bound in relevant_bounds):
        raise _validation_error(
            "budget_exceeded",
            "determinant-family expansion exceeds the result term budget",
        )
    maximum_exponent = max(
        (
            exponent
            for value in values
            for term in value.numerator.terms
            for exponent in term.exponents
        ),
        default=0,
    )
    if dimension * maximum_exponent > MAX_SYMBOLIC_RESULT_EXPONENT:
        raise _validation_error(
            "budget_exceeded",
            "determinant-family expansion exceeds the result exponent budget",
        )
    coefficient_digits = max(
        (
            len(component.lstrip("-"))
            for value in values
            for term in value.numerator.terms
            for component in (term.coefficient.num, term.coefficient.den)
        ),
        default=1,
    )
    if any(
        bound * dimension * coefficient_digits + len(str(max(bound, 1)))
        > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS
        for bound in relevant_bounds
    ):
        raise _validation_error(
            "budget_exceeded",
            "determinant-family expansion exceeds the result coefficient budget",
        )


class SymbolicMatrix(StrictModel):
    """One nonempty rectangular matrix over a multivariate rational-function field.

    Every entry is a canonical reduced numerator/denominator value over the
    declared ordered variables. For example, the former expression ``a*c`` is
    represented by one numerator term with exponents ``(1, 0, 1, ...)`` and a
    unit denominator; ``f/e`` is represented by numerator ``f`` and denominator
    ``e``. This preserves every element of ``QQ(t_1, ..., t_n)`` without parsing
    caller text with SymPy.
    """

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0,
        max_length=MAX_SYMBOLIC_VARIABLES,
    )
    entries: tuple[tuple[RationalFunction, ...], ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_SYMBOLIC_MATRIX_DIMENSION:
            raise _validation_error(
                "shape_mismatch",
                "matrix rows must contain between 1 and "
                f"{MAX_SYMBOLIC_MATRIX_DIMENSION} entries",
            )
        if any(len(row) != column_count for row in self.entries):
            raise _validation_error(
                "budget_exceeded", "matrix rows must all have the same length"
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "budget_exceeded", "symbolic matrix variables must be unique"
            )
        values = tuple(value for row in self.entries for value in row)
        if any(value.variables != self.variables for value in values):
            raise _validation_error(
                "budget_exceeded",
                "every symbolic matrix entry must use the declared ordered field",
            )
        term_count = sum(
            len(value.numerator.terms) + len(value.denominator.terms)
            for value in values
        )
        if term_count > MAX_SYMBOLIC_MATRIX_TERMS:
            raise _validation_error(
                "budget_exceeded",
                "symbolic matrix exceeds the 512-term operation budget",
            )
        return self


def _maximum_exponents(value: RationalFunction, *, numerator: bool) -> tuple[int, ...]:
    polynomial = value.numerator if numerator else value.denominator
    return tuple(
        max((term.exponents[axis] for term in polynomial.terms), default=0)
        for axis in range(len(value.variables))
    )


def _maximum_coefficient_digits(value: RationalFunction, *, numerator: bool) -> int:
    polynomial = value.numerator if numerator else value.denominator
    return max(
        (
            len(component.lstrip("-"))
            for term in polynomial.terms
            for component in (term.coefficient.num, term.coefficient.den)
        ),
        default=1,
    )


def _sum_coefficient_digit_bound(
    *,
    term_count: int,
    product_coefficient_digits: int,
    integral_coefficients: bool,
) -> int:
    """Bound one coefficient after collecting ``term_count`` rational products.

    Integral products only add an addition carry to the collected
    coefficient. Rational products can also multiply unrelated
    denominators while reaching the common denominator, which the
    conservative rational bound charges for.
    """

    if term_count <= 1:
        return product_coefficient_digits
    if integral_coefficients:
        return product_coefficient_digits + len(str(term_count))
    return term_count * product_coefficient_digits + len(str(term_count))


def _has_integral_coefficients(value: RationalFunction) -> bool:
    """Report whether every coefficient of both polynomial sides is an integer."""

    return all(
        term.coefficient.den == "1"
        for polynomial in (value.numerator, value.denominator)
        for term in polynomial.terms
    )


def _common_denominator_digits(
    left_value: RationalFunction, right_value: RationalFunction
) -> int:
    """Coefficient digits one pair contributes to the product common denominator.

    A canonical unit denominator contributes nothing: multiplying by one
    leaves every coefficient unchanged, just as it performs no expansion
    work.
    """

    if _is_polynomial_entry(left_value) and _is_polynomial_entry(right_value):
        return 0
    return _maximum_coefficient_digits(left_value, numerator=False) + (
        _maximum_coefficient_digits(right_value, numerator=False)
    )


def _scalar_constant_digits(value: RationalFunction) -> int | None:
    """Coefficient height when the value is a nonzero rational constant.

    Every nonzero element of QQ is a unit of the field, so multiplying by
    one rescales coefficients without touching support or denominators.
    """

    terms = value.numerator.terms
    if (
        len(terms) == 1
        and all(exponent == 0 for exponent in terms[0].exponents)
        and _is_polynomial_entry(value)
    ):
        return max(
            len(component.lstrip("-"))
            for component in (terms[0].coefficient.num, terms[0].coefficient.den)
        )
    return None


def _polynomial_pair_exponents(
    first: SparseRationalPolynomial, second: SparseRationalPolynomial
) -> tuple[tuple[int, ...], ...]:
    """Enumerate monomial exponents of every pairwise product of two polynomials."""

    return tuple(
        tuple(
            first_exponent + second_exponent
            for first_exponent, second_exponent in zip(
                first_term.exponents, second_term.exponents, strict=True
            )
        )
        for first_term in first.terms
        for second_term in second.terms
    )


def _product_collision_counts(
    expansions: Iterable[tuple[tuple[tuple[int, ...], ...], ...]],
) -> Counter[tuple[int, ...]]:
    """Count raw products landing on each collected monomial exponent."""

    counts: Counter[tuple[int, ...]] = Counter()
    for groups in expansions:
        for chosen in product(*groups):
            counts[tuple(map(sum, zip(*chosen, strict=True)))] += 1
    return counts


def _maximum_product_collisions(
    expansions: Iterable[tuple[tuple[tuple[int, ...], ...], ...]],
) -> int:
    """Count raw products colliding at the most crowded exponent.

    Each expansion is a finite product of monomial-exponent choices; only
    products landing on one exponent are collected into a single coefficient,
    so the maximum multiplicity, not the total support size, drives the
    coefficient digit bound.
    """

    return max(_product_collision_counts(expansions).values(), default=1)


def _polynomial_budget_violation(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> bool:
    """Report whether one sparse polynomial exceeds the canonical result budgets."""

    return (
        len(polynomial.terms) > MAX_SYMBOLIC_RESULT_TERMS
        or any(
            exponent > MAX_SYMBOLIC_RESULT_EXPONENT
            for term in polynomial.terms
            for exponent in term.exponents
        )
        or any(
            len(component.lstrip("-")) > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS
            for term in polynomial.terms
            for component in (term.coefficient.num, term.coefficient.den)
        )
        or any(len(term.exponents) != variable_count for term in polynomial.terms)
    )


def _shared_common_denominator_bounds(
    factors: tuple[tuple[RationalFunction, RationalFunction], ...],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int, bool, int, bool] | None:
    """Return exact cell bounds for one shared product denominator, or None.

    When every contributing pair ``a_i/g_i * b_i/v_i`` carries one identical
    canonical product denominator ``d = g_i v_i``, the cell value is exactly
    ``(sum a_i b_i) / d``. Admission collects that numerator sum exactly and
    admits the cell only when the sum is coprime to the monic product ``d``
    and both parts fit the canonical result budgets. The backend's reduced
    value then equals this quotient verbatim, so the collected support and
    height are exact bounds and no cancellation estimate is needed. A
    nontrivial common factor, mismatched pairwise product denominators, or
    an out-of-budget part returns None and the conservative rejection
    applies.
    """

    from jacobian.math.polynomials._conversions import (
        sparse_rational_polynomial_from_sympy,
        sparse_rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    variables = factors[0][0].variables
    if not variables:
        return None
    pair_denominator_terms = tuple(
        len(left_value.denominator.terms) * len(right_value.denominator.terms)
        for left_value, right_value in factors
    )
    if any(count > MAX_SYMBOLIC_RESULT_TERMS for count in pair_denominator_terms):
        return None
    raw_products = sum(
        len(left_value.numerator.terms) * len(right_value.numerator.terms)
        for left_value, right_value in factors
    )
    if raw_products > MAX_SYMBOLIC_MATRIX_TERMS:
        return None

    from sympy import Poly

    pair_product_denominators = [
        sparse_rational_polynomial_to_sympy(left_value.denominator, variables)
        * sparse_rational_polynomial_to_sympy(right_value.denominator, variables)
        for left_value, right_value in factors
    ]
    common_denominator = pair_product_denominators[0]
    if any(
        pair_denominator != common_denominator
        for pair_denominator in pair_product_denominators[1:]
    ):
        return None

    generators = symbols_for_variables(variables)
    numerator_sum = Poly(0, *generators, domain="QQ")
    for left_value, right_value in factors:
        numerator_sum += sparse_rational_polynomial_to_sympy(
            left_value.numerator, variables
        ) * sparse_rational_polynomial_to_sympy(right_value.numerator, variables)

    if numerator_sum.is_zero:
        zero_exponents = (0,) * len(variables)
        return (
            raw_products,
            sum(pair_denominator_terms),
            zero_exponents,
            zero_exponents,
            1,
            False,
            1,
            True,
        )

    numerator = sparse_rational_polynomial_from_sympy(numerator_sum, variables)
    common = sparse_rational_polynomial_from_sympy(common_denominator, variables)
    if _polynomial_budget_violation(numerator, len(variables)) or (
        _polynomial_budget_violation(common, len(variables))
    ):
        return None
    if not numerator_sum.gcd(common_denominator).is_one:
        return None
    return (
        raw_products,
        sum(pair_denominator_terms),
        tuple(
            max(term.exponents[axis] for term in numerator.terms)
            for axis in range(len(variables))
        ),
        tuple(
            max(term.exponents[axis] for term in common.terms)
            for axis in range(len(variables))
        ),
        max(
            len(component.lstrip("-"))
            for polynomial in (numerator, common)
            for term in polynomial.terms
            for component in (term.coefficient.num, term.coefficient.den)
        ),
        False,
        len(numerator.terms) + len(common.terms),
        True,
    )


def _product_cell_bounds(
    left: tuple[RationalFunction, ...],
    right: tuple[RationalFunction, ...],
    *,
    exact_shared_bounds: bool = True,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int, bool, int, bool]:
    """Return raw work bounds and cancellation-safe canonical degree bounds.

    The cell is a finite sum of products ``a_i * b_i``. Each nonzero product
    is first written over its own canonical denominator, then all products are
    put over the product common denominator. The raw term and coefficient
    bounds admit that expansion before SymPy receives the request. Coefficient
    digits are bounded by the products colliding at one exponent rather than
    by total support size. The seventh element bounds the cell's canonical
    result support: collected numerator terms plus the retained canonical
    denominator terms, which are exactly one for expanded cells (unit or
    monomial) and the operand's own denominator for the verbatim or
    scalar-scaled copy. The final flag reports a proven cancellation-free
    shape: either the cell stays inside the     admitted cancellation domain
    (unit denominators never cancel, and a monomial common denominator only
    cancels by monomial factors, so the raw bounds stay valid for the
    canonical value), or every pair carries one identical product
    denominator whose exact collected numerator admission proved coprime to
    the retained monic product.
    """

    factors = tuple(
        (left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
        if left_value.numerator.terms and right_value.numerator.terms
    )
    if not factors:
        # The canonical zero has an empty numerator and a one-term denominator.
        zero_exponents = (0,) * len(left[0].variables)
        return 0, 1, zero_exponents, zero_exponents, 1, True, 1, False

    if len(factors) == 1:
        left_value, right_value = factors[0]
        scalar_digits = 0
        if _is_scalar_identity(right_value):
            effective = left_value
        elif _is_scalar_identity(left_value):
            effective = right_value
        else:
            right_digits = _scalar_constant_digits(right_value)
            left_digits = _scalar_constant_digits(left_value)
            if right_digits is not None:
                effective, scalar_digits = left_value, right_digits
            elif left_digits is not None:
                effective, scalar_digits = right_value, left_digits
            else:
                effective = None
        if effective is not None:
            # Identity multiplication returns the other canonical operand
            # verbatim, and a nonzero rational constant only rescales its
            # coefficients by a unit of QQ: neither can cancel or densify,
            # so the operand shape (plus the scalar height when scaling)
            # bounds the result exactly and no expansion occurs.
            return (
                len(effective.numerator.terms),
                len(effective.denominator.terms),
                _maximum_exponents(effective, numerator=True),
                _maximum_exponents(effective, numerator=False),
                max(
                    _maximum_coefficient_digits(effective, numerator=True),
                    _maximum_coefficient_digits(effective, numerator=False),
                )
                + scalar_digits,
                True,
                len(effective.numerator.terms) + len(effective.denominator.terms),
                False,
            )

    return _expanded_product_cell_bounds(
        factors, exact_shared_bounds=exact_shared_bounds
    )


def _expanded_product_cell_bounds(
    factors: tuple[tuple[RationalFunction, RationalFunction], ...],
    *,
    exact_shared_bounds: bool = True,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int, bool, int, bool]:
    """Bound a multi-factor cell by its unreduced common-denominator expansion.

    Each nonzero product is first written over its own canonical denominator,
    then all products are put over the product common denominator; the raw
    term and coefficient bounds admit that expansion before SymPy receives
    the request. Cells the cancellation rejection would refuse get one exact
    second chance through ``_shared_common_denominator_bounds``. With
    ``exact_shared_bounds=False`` that second chance is not executed: an
    eligible cell returns the raw expansion totals the exact fallback charges
    when it admits the cell, so callers can project the aggregate admission
    budget without SymPy work.
    """

    integral_coefficients = all(
        _has_integral_coefficients(left_value)
        and _has_integral_coefficients(right_value)
        for left_value, right_value in factors
    )
    unit_denominator_factors = all(
        _is_polynomial_entry(left_value) and _is_polynomial_entry(right_value)
        for left_value, right_value in factors
    )

    denominator_term_counts = tuple(
        len(left_value.denominator.terms) * len(right_value.denominator.terms)
        for left_value, right_value in factors
    )
    denominator_exponents = tuple(
        tuple(
            left_exponent + right_exponent
            for left_exponent, right_exponent in zip(
                _maximum_exponents(left_value, numerator=False),
                _maximum_exponents(right_value, numerator=False),
                strict=True,
            )
        )
        for left_value, right_value in factors
    )
    denominator_coefficient_digits = tuple(
        _common_denominator_digits(left_value, right_value)
        for left_value, right_value in factors
    )
    denominator_terms = 1
    for count in denominator_term_counts:
        denominator_terms *= count

    # Cells the conservative cancellation rejection would refuse get one
    # exact second chance: when every pair's product denominators coincide,
    # admission can collect the numerator sum and prove it coprime to the
    # retained monic product, which fixes the canonical value verbatim.
    if not (unit_denominator_factors or denominator_terms == 1):
        if not exact_shared_bounds:
            zero_exponents = (0,) * len(factors[0][0].variables)
            return (
                sum(
                    len(left_value.numerator.terms) * len(right_value.numerator.terms)
                    for left_value, right_value in factors
                ),
                sum(denominator_term_counts),
                zero_exponents,
                zero_exponents,
                0,
                False,
                1,
                True,
            )
        shared_bounds = _shared_common_denominator_bounds(factors)
        if shared_bounds is not None:
            return shared_bounds

    denominator_exponent = tuple(
        sum(exponents) for exponents in zip(*denominator_exponents, strict=True)
    )
    denominator_product_digits = sum(denominator_coefficient_digits)

    numerator_terms = 0
    numerator_exponent = (0,) * len(factors[0][0].variables)
    numerator_product_digits = 1
    for index, (left_value, right_value) in enumerate(factors):
        numerator_term_count = len(left_value.numerator.terms) * len(
            right_value.numerator.terms
        )
        for other_index, denominator_term_count in enumerate(denominator_term_counts):
            if other_index != index:
                numerator_term_count *= denominator_term_count
        numerator_terms += numerator_term_count
        numerator_exponent = tuple(
            max(
                accumulated_exponent,
                left_exponent
                + right_exponent
                + common_denominator_exponent
                - own_denominator_exponent,
            )
            for (
                accumulated_exponent,
                left_exponent,
                right_exponent,
                common_denominator_exponent,
                own_denominator_exponent,
            ) in zip(
                numerator_exponent,
                _maximum_exponents(left_value, numerator=True),
                _maximum_exponents(right_value, numerator=True),
                denominator_exponent,
                denominator_exponents[index],
                strict=True,
            )
        )
        numerator_product_digits = max(
            numerator_product_digits,
            _maximum_coefficient_digits(left_value, numerator=True)
            + _maximum_coefficient_digits(right_value, numerator=True)
            + denominator_product_digits
            - denominator_coefficient_digits[index],
        )

    # Only cells whose raw products fit the aggregate expansion budget can
    # survive admission, so enumerating their raw products stays bounded;
    # larger cells keep the conservative support-size estimate and are
    # rejected by the term checks.
    numerator_collision_count = numerator_terms
    denominator_collision_count = denominator_terms
    result_term_count = numerator_terms + 1
    if (
        numerator_terms <= MAX_SYMBOLIC_MATRIX_TERMS
        and denominator_terms <= MAX_SYMBOLIC_MATRIX_TERMS
    ):
        denominator_groups = tuple(
            _polynomial_pair_exponents(left_value.denominator, right_value.denominator)
            for left_value, right_value in factors
        )
        numerator_expansions: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
        for index, (left_value, right_value) in enumerate(factors):
            expansion: list[tuple[tuple[int, ...], ...]] = [
                _polynomial_pair_exponents(left_value.numerator, right_value.numerator)
            ]
            expansion.extend(
                group
                for other_index, group in enumerate(denominator_groups)
                if other_index != index
            )
            numerator_expansions.append(tuple(expansion))
        denominator_collision_count = _maximum_product_collisions((denominator_groups,))
        numerator_collisions = _product_collision_counts(numerator_expansions)
        numerator_collision_count = max(numerator_collisions.values(), default=1)
        result_term_count = len(numerator_collisions) + 1

    maximum_coefficient_digits = max(
        _sum_coefficient_digit_bound(
            term_count=numerator_collision_count,
            product_coefficient_digits=numerator_product_digits,
            integral_coefficients=integral_coefficients,
        ),
        _sum_coefficient_digit_bound(
            term_count=denominator_collision_count,
            product_coefficient_digits=denominator_product_digits,
            integral_coefficients=integral_coefficients,
        ),
    )
    return (
        numerator_terms,
        denominator_terms,
        numerator_exponent,
        denominator_exponent,
        maximum_coefficient_digits,
        unit_denominator_factors,
        result_term_count,
        False,
    )


def _projected_expansion_terms(left: SymbolicMatrix, right: SymbolicMatrix) -> int:
    """Charge every cell the raw expansion its admitted shape must spend.

    A cell admitted through the exact shared-denominator fallback carries
    exactly these raw product totals, and every other cell's cheap bounds
    are already final, so the sum lower-bounds the aggregate expansion of
    any request that survives admission.
    """

    projected_expansion_terms = 0
    for left_row in left.entries:
        for right_column in zip(*right.entries, strict=True):
            (
                numerator_terms,
                denominator_terms,
                _numerator_exponents,
                _denominator_exponents,
                _maximum_coefficient_digits,
                unit_denominator_factors,
                _result_term_count,
                _verified_no_cancellation,
            ) = _product_cell_bounds(left_row, right_column, exact_shared_bounds=False)
            projected_expansion_terms += numerator_terms
            if not unit_denominator_factors:
                projected_expansion_terms += denominator_terms
    return projected_expansion_terms


def _require_symbolic_product_admission(
    left: SymbolicMatrix,
    right: SymbolicMatrix,
) -> None:
    """Prove that every exact product entry fits the canonical result value.

    Cells cancel only through unit or monomial common denominators, where
    reduction cannot grow support or coefficients, or through one identical
    pairwise product denominator whose exact collected numerator admission
    proved coprime to the retained monic product, where reduction cannot
    trigger at all; anything else stays outside the admitted domain
    because no pre-execution bound covers it. A cheap projection pass first
    charges every cell the raw expansion the exact shared-denominator
    admission spends when it succeeds, so a request above the aggregate
    expansion budget is rejected before any SymPy conversion runs.
    """

    if left.variables != right.variables:
        raise _validation_error(
            "budget_exceeded",
            "symbolic matrix multiplication requires identical ordered field variables",
        )
    if len(left.entries[0]) != len(right.entries):
        raise _validation_error(
            "budget_exceeded",
            "symbolic matrix multiplication requires the left column count to equal "
            "the right row count",
        )

    projected_expansion_terms = _projected_expansion_terms(left, right)
    if projected_expansion_terms > MAX_SYMBOLIC_MATRIX_TERMS:
        raise _validation_error(
            "budget_exceeded",
            "symbolic matrix product exceeds the 512-term aggregate expansion budget",
        )

    aggregate_expansion_terms = 0
    aggregate_result_terms = 0
    for left_row in left.entries:
        for right_column in zip(*right.entries, strict=True):
            (
                numerator_terms,
                denominator_terms,
                numerator_exponents,
                denominator_exponents,
                maximum_coefficient_digits,
                unit_denominator_factors,
                result_term_count,
                verified_no_cancellation,
            ) = _product_cell_bounds(left_row, right_column)
            if result_term_count > MAX_SYMBOLIC_RESULT_TERMS:
                # Raw scalar products are governed by the aggregate expansion
                # budget below; this per-entry limit binds the already
                # computed collected support of the canonical cell value.
                raise _validation_error(
                    "budget_exceeded",
                    "symbolic matrix product exceeds the 256-term exact result budget",
                )
            if not (
                unit_denominator_factors
                or denominator_terms == 1
                or verified_no_cancellation
            ):
                # Exact division by a non-monomial greatest common divisor can
                # amplify coefficients far beyond the unreduced expansion
                # (hidden cancellation inside the dividend), and no usable
                # pre-execution height bound exists for that quotient. Cells
                # whose common denominator has several terms therefore lack a
                # coefficient bound and stay outside the admitted domain unless
                # admission collected the shared product denominator's
                # numerator sum and proved it coprime to the retained monic
                # product. Unit
                # denominators never cancel, and a monomial common denominator
                # only loses monomial factors during cancellation (every
                # divisor of a monomial is a monomial), so support and
                # coefficient size stay within the raw expansion bounds.
                raise _validation_error(
                    "budget_exceeded",
                    "symbolic matrix product cannot bound coefficient growth "
                    "under cancellation by a multi-term denominator",
                )
            maximum_exponent = max(
                (*numerator_exponents, *denominator_exponents), default=0
            )
            if maximum_exponent > MAX_SYMBOLIC_RESULT_EXPONENT:
                raise _validation_error(
                    "budget_exceeded",
                    "symbolic matrix product exceeds the result exponent budget",
                )
            if maximum_coefficient_digits > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS:
                raise _validation_error(
                    "budget_exceeded",
                    "symbolic matrix product exceeds the result coefficient budget",
                )
            # Unit denominators produce no denominator work at all, so the
            # expansion charge counts only the scalar products that run.
            aggregate_expansion_terms += numerator_terms
            if not unit_denominator_factors:
                aggregate_expansion_terms += denominator_terms
            # Canonical result support is bounded separately from expansion
            # work: every admitted cell carries exactly one canonical
            # denominator term (unit or monomial), and cancellation in the
            # admitted domain can only shrink collected numerator support,
            # so result_term_count bounds the terms the returned
            # SymbolicMatrix will validate.
            aggregate_result_terms += result_term_count
    if aggregate_expansion_terms > MAX_SYMBOLIC_MATRIX_TERMS:
        raise _validation_error(
            "budget_exceeded",
            "symbolic matrix product exceeds the 512-term aggregate expansion budget",
        )
    if aggregate_result_terms > MAX_SYMBOLIC_MATRIX_TERMS:
        raise _validation_error(
            "budget_exceeded",
            "symbolic matrix product exceeds the 512-term aggregate result budget",
        )


class SymbolicMatrixProductRequest(StrictModel):
    """Two compatible symbolic matrices whose exact product is representable."""

    left: SymbolicMatrix = Field(
        description=(
            "A nonempty symbolic matrix over QQ(t_1, ..., t_n). Its ordered "
            "field variables must exactly match right.variables."
        )
    )
    right: SymbolicMatrix = Field(
        description=(
            "A nonempty symbolic matrix over the same ordered field as left. "
            "Its row count must equal left's column count."
        )
    )

    @model_validator(mode="after")
    def require_bounded_exact_product(self) -> Self:
        _require_symbolic_product_admission(self.left, self.right)
        return self


class SymbolicMatrixRequest(StrictModel):
    """A symbolic matrix over a declared variable list."""

    matrix: SymbolicMatrix

    @model_validator(mode="after")
    def require_request_consistency(self) -> Self:
        return self


class SquareSymbolicMatrixRequest(SymbolicMatrixRequest):
    """A square symbolic matrix for operations requiring square input.

    Operations like determinant, characteristic polynomial, and eigenvalues
    are only defined for square matrices.  This request type enforces
    squareness at the request boundary rather than relying on a backend
    ValueError.
    """

    @model_validator(mode="after")
    def require_square(self) -> Self:
        rows = len(self.matrix.entries)
        cols = len(self.matrix.entries[0])
        if rows != cols:
            raise _validation_error(
                "budget_exceeded", "operation requires a square symbolic matrix"
            )
        return self


class SymbolicDeterminantRequest(SquareSymbolicMatrixRequest):
    """A square matrix whose exact determinant fits the public result type."""

    matrix: SymbolicMatrix = Field(
        description=(
            "A square symbolic matrix. One-dimensional matrices may contain any "
            "accepted rational function; larger matrices require polynomial "
            "entries whose derived determinant expansion has at most 256 terms, "
            "exponent 64, and 128-digit coefficient components."
        )
    )

    @model_validator(mode="after")
    def require_representable_determinant(self) -> Self:
        _require_determinant_family_result_budget(
            self.matrix,
            characteristic_polynomial=False,
        )
        return self


class SymbolicCharacteristicPolynomialRequest(SquareSymbolicMatrixRequest):
    """A square matrix whose characteristic polynomial fits the result type."""

    matrix: SymbolicMatrix = Field(
        description=(
            "A square symbolic matrix. One-dimensional matrices may contain any "
            "accepted rational function; larger matrices require polynomial "
            "entries whose derived principal-minor expansions each have at most "
            "256 terms, exponent 64, and 128-digit coefficient components."
        )
    )

    @model_validator(mode="after")
    def require_representable_characteristic_polynomial(self) -> Self:
        _require_determinant_family_result_budget(
            self.matrix,
            characteristic_polynomial=True,
        )
        return self


class SymbolicDeterminantResult(StrictModel):
    """The exact determinant in the matrix's rational-function field."""

    determinant: RationalFunction
    method: Literal["SYMPY_BAREISS"] = "SYMPY_BAREISS"


class SymbolicRankResult(StrictModel):
    """The exact symbolic rank and the canonical pivot columns."""

    rank: int = Field(ge=0, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_SYMBOLIC_MATRIX_DIMENSION)
    method: Literal["EXACT_SYMBOLIC_ROW_REDUCTION"] = "EXACT_SYMBOLIC_ROW_REDUCTION"


class SymbolicCharacteristicPolynomialResult(StrictModel):
    """The dense monic characteristic polynomial coefficients (descending)."""

    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=1, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    coefficients_descending: tuple[RationalFunction, ...] = Field(
        min_length=2,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION + 1,
    )
    convention: Literal["DET_LAMBDA_I_MINUS_A"] = "DET_LAMBDA_I_MINUS_A"


class SymbolicEigenvaluesResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities.

    The representation discriminates between:
    - EXPLICIT_ROOTS: individual eigenvalue expressions are returned
    - ROOTS_BY_POLYNOMIAL: eigenvalues are the roots of the returned
      characteristic polynomial over QQ(t_1, ..., t_n); individual root
      expressions are not materialized because the backend cannot
      represent them in radicals.
    """

    representation: Literal["EXPLICIT_ROOTS", "ROOTS_BY_POLYNOMIAL"] = "EXPLICIT_ROOTS"
    eigenvalues: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    multiplicities: tuple[int, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    characteristic_polynomial: tuple[RationalFunction, ...] | None = Field(
        default=None,
        min_length=2,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION + 1,
    )
    degree: int | None = Field(default=None, ge=1, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"

    @model_validator(mode="after")
    def require_representation_consistency(self) -> Self:
        if self.representation == "EXPLICIT_ROOTS":
            if self.eigenvalues is None or self.multiplicities is None:
                raise _validation_error(
                    "shape_mismatch",
                    "EXPLICIT_ROOTS must populate eigenvalues and multiplicities",
                )
            if len(self.eigenvalues) != len(self.multiplicities):
                raise _validation_error(
                    "shape_mismatch",
                    "eigenvalues and multiplicities must have the same length",
                )
            if self.characteristic_polynomial is not None or self.degree is not None:
                raise _validation_error(
                    "field_mismatch",
                    "EXPLICIT_ROOTS must not populate characteristic_polynomial or degree",
                )
        else:  # ROOTS_BY_POLYNOMIAL
            if self.eigenvalues is not None or self.multiplicities is not None:
                raise _validation_error(
                    "field_mismatch",
                    "ROOTS_BY_POLYNOMIAL must not populate eigenvalues or multiplicities",
                )
            if self.characteristic_polynomial is None or self.degree is None:
                raise _validation_error(
                    "invariant_mismatch",
                    "ROOTS_BY_POLYNOMIAL must populate characteristic_polynomial and degree",
                )
            if len(self.characteristic_polynomial) != self.degree + 1:
                raise _validation_error(
                    "budget_exceeded",
                    "characteristic polynomial coefficients must equal degree plus one",
                )
        return self


# ---------------------------------------------------------------------------
# Symbolic linear system over QQ(t_1, ..., t_n)
# ---------------------------------------------------------------------------


def _entry_growth_factor(value: RationalFunction) -> int:
    """Term-count factor one entry contributes to an unreduced product.

    Polynomial entries contribute their numerator term count; rational
    entries contribute squared factors because products of fractions
    accumulate numerator and denominator terms on both sides.
    """
    numerator_terms = len(value.numerator.terms)
    denominator_terms = len(value.denominator.terms)
    unit_denominator = denominator_terms == 1 and all(
        term.coefficient.num == "1" and all(e == 0 for e in term.exponents)
        for term in value.denominator.terms
    )
    if unit_denominator:
        return numerator_terms
    return max(numerator_terms, denominator_terms) ** 2


_EXPANSION_ENUMERATION_NODE_BUDGET = 200_000


class _ExpansionBudgetExhaustedError(Exception):
    """Exact Leibniz enumeration exceeded its admission node budget."""


def _augmented_growth_support(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Nonzero ``(column, growth factor)`` cells per row of ``[A | b]``."""
    columns = len(entries[0])
    support: list[tuple[tuple[int, int], ...]] = []
    for row_index, row in enumerate(entries):
        cells = [
            (column, _entry_growth_factor(value)) for column, value in enumerate(row)
        ]
        if row_index < len(rhs):
            cells.append((columns, _entry_growth_factor(rhs[row_index])))
        support.append(tuple(cell for cell in cells if cell[1] > 0))
    return tuple(support)


def _injection_count_bound(
    support: tuple[tuple[tuple[int, int], ...], ...],
    columns_count: int,
    size: int,
) -> int:
    """Closed-form expansion bound over every size-k minor of ``[A | b]``.

    A Leibniz term survives only through an injection of its rows into
    distinct nonzero-growth columns, so the number of surviving terms is
    bounded by the product of the smaller of each largest row and column
    degree, times the largest single-cell growth factor to the size.
    """
    row_degrees = sorted((len(row) for row in support), reverse=True)
    column_degrees = [0] * (columns_count + 1)
    for row in support:
        for column, _ in row:
            column_degrees[column] += 1
    column_degrees.sort(reverse=True)
    growth_factors = [factor for row in support for _, factor in row]
    maximum_growth: int = max(growth_factors, default=0)
    degree_product = 1
    for index in range(size):
        degree_product *= min(row_degrees[index], column_degrees[index])
    expansion_bound: int = degree_product * maximum_growth**size
    return expansion_bound


def _exact_size_expansion(
    support: tuple[tuple[tuple[int, int], ...], ...],
    rows_count: int,
    columns_count: int,
    size: int,
    visited: list[int],
) -> int:
    """Exact maximum Leibniz expansion over every size-k minor.

    The structural walk visits only nonzero-growth cells and charges each
    visit to ``visited[0]``, aborting once the shared admission budget is
    exhausted.
    """

    def walk(
        row_position: int,
        row_indices: tuple[int, ...],
        available: int,
        product: int,
    ) -> int:
        if row_position == len(row_indices):
            return product
        total = 0
        for column, factor in support[row_indices[row_position]]:
            bit = 1 << column
            if available & bit:
                visited[0] += 1
                if visited[0] > _EXPANSION_ENUMERATION_NODE_BUDGET:
                    raise _ExpansionBudgetExhaustedError()
                total += walk(
                    row_position + 1,
                    row_indices,
                    available ^ bit,
                    product * factor,
                )
        return total

    maximum = 0
    for row_indices in combinations(range(rows_count), size):
        for column_indices in combinations(range(columns_count + 1), size):
            column_mask = sum(1 << column for column in column_indices)
            maximum = max(maximum, walk(0, row_indices, column_mask, 1))
    return maximum


def _expansion_bounds_by_size(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> list[int]:
    """Per-size maximum minor expansion, exact within a node budget.

    Sparse systems complete the exact structural enumeration far below
    the budget; once it is exceeded, remaining sizes fall back to
    ``_injection_count_bound``, so request validation never performs
    factorial permutation work.
    """
    rows_count = len(entries)
    columns_count = len(entries[0])
    work = min(rows_count, columns_count)
    support = _augmented_growth_support(entries, rhs)
    bounds = [0] * (work + 1)
    visited = [0]
    exhausted = False
    for size in range(1, work + 1):
        if exhausted:
            bounds[size] = _injection_count_bound(support, columns_count, size)
            continue
        try:
            bounds[size] = _exact_size_expansion(
                support, rows_count, columns_count, size, visited
            )
        except _ExpansionBudgetExhaustedError:
            exhausted = True
            bounds[size] = _injection_count_bound(support, columns_count, size)
    return bounds


def _solution_component_growth_bound(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> tuple[int, int, int]:
    """Conservative (terms, exponent, coefficient digits) for solved components.

    Every solution, particular-solution, and nullspace component is an
    exact ratio of minors of the augmented system ``[A | b]`` over minors
    of ``A`` of every size up to ``work = min(rows, columns)`` (Cramer/RREF
    structure; rank-deficient systems are decided by their largest
    nonvanishing minors). Both sides of such a ratio multiply up to
    ``2 * size`` entry factors, and each unreduced minor numerator expands
    over the Leibniz sum of per-entry term-count products.
    """
    rows = len(entries)
    columns = len(entries[0])
    work = min(rows, columns)

    # Every k-size minor of [A | b] with 1 <= k <= work bounds the
    # expansion work behind some solution component; A's own minors are a
    # subset of these. Lower-rank minors matter when all work-size minors
    # are structurally zero.
    maximum_expansion_by_size = _expansion_bounds_by_size(entries, rhs)

    values = tuple(value for row in entries for value in row) + tuple(rhs)
    maximum_exponent = max(
        (
            exponent
            for value in values
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
            for exponent in term.exponents
        ),
        default=0,
    )
    coefficient_digits = max(
        (
            max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
            for value in values
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
        ),
        default=1,
    )
    terms_bound = max(expansion**2 for expansion in maximum_expansion_by_size)
    exponent_bound = 2 * work * maximum_exponent
    digits_bound = max(
        (
            expansion * 2 * size * coefficient_digits + len(str(max(expansion, 1)))
            for size, expansion in enumerate(maximum_expansion_by_size)
            if size >= 1
        ),
        default=1,
    )
    return terms_bound, exponent_bound, digits_bound


def _require_linear_system_solution_budget(
    request: SymbolicLinearSystemRequest,
) -> None:
    """Admit only systems whose derived solutions fit the result type.

    Runs at request admission so no accepted request can fail inside the
    backend conversion with a host exception instead of returning its
    declared typed result.
    """
    _require_linear_system_growth_admission(request.matrix.entries, request.rhs)


def _require_linear_system_growth_admission(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> None:
    """Admit only systems whose derived solutions fit the result type.

    Shared by the wire request validator and the native solve entry point so
    direct callers cannot bypass the derived-solution bounds.
    """
    growth = _solution_component_growth_bound(entries, rhs)
    if growth[0] > MAX_SYMBOLIC_RESULT_TERMS:
        raise _validation_error(
            "budget_exceeded",
            "linear-system solution exceeds the derived result term budget; "
            "reduce entry term counts or dimension",
        )
    if growth[1] > MAX_SYMBOLIC_RESULT_EXPONENT:
        raise _validation_error(
            "budget_exceeded",
            "linear-system solution exceeds the derived result exponent "
            "budget; reduce entry exponents or dimension",
        )
    if growth[2] > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "linear-system solution exceeds the derived result coefficient "
            "budget; reduce coefficient sizes or dimension",
        )


class SymbolicLinearSystemRequest(StrictModel):
    """Solve one bounded system ``A x = b`` over ``QQ(t_1, ..., t_n)``.

    The declared parameters are algebraically independent: the result is the
    generic solution over the rational-function field, not a case split over
    parameter specializations.  The coefficient matrix ``A`` and right-hand
    side ``b`` must use the same declared ordered variable list.
    """

    matrix: SymbolicMatrix
    rhs: tuple[RationalFunction, ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_consistent_system(self) -> Self:
        rows = len(self.matrix.entries)
        if len(self.rhs) != rows:
            raise _validation_error(
                "budget_exceeded",
                "the right-hand side length must equal the coefficient row count",
            )
        for value in self.rhs:
            if value.variables != self.matrix.variables:
                raise _validation_error(
                    "budget_exceeded",
                    "the right-hand side must use the declared ordered field",
                )
        # Derived-solution admission: bound exponent, term, and coefficient
        # growth before the backend runs so every accepted system returns
        # its declared typed result instead of failing inside conversion.
        _require_linear_system_solution_budget(self)
        return self


def _raw_system_column_bound(system: Any) -> int:
    """Best-effort column count of a not-yet-validated raw system payload."""
    if isinstance(system, dict):
        matrix = system.get("matrix")
        if isinstance(matrix, dict):
            entries = matrix.get("entries")
            if (
                isinstance(entries, (list, tuple))
                and entries
                and isinstance(entries[0], (list, tuple))
            ):
                return len(entries[0])
    return MAX_SYMBOLIC_MATRIX_DIMENSION


class SymbolicLinearSystemResult(StrictModel):
    """Classification and solution data for one symbolic linear system.

    The source system is retained for an explicit bounded verifier. Kernel
    output uses :meth:`_from_kernel`; this transport model checks only the
    source-coupled payload shape and never re-enters a symbolic operation.
    """

    system: SymbolicLinearSystemRequest
    classification: Literal["UNIQUE", "NON_UNIQUE", "INCONSISTENT"]
    solution: tuple[RationalFunction, ...] | None = None
    particular_solution: tuple[RationalFunction, ...] | None = None
    nullspace_basis: tuple[tuple[RationalFunction, ...], ...] | None = None
    consistency: Literal["EXACT_RATIONAL_FUNCTION"] = "EXACT_RATIONAL_FUNCTION"
    field_semantics: Literal["GENERIC_OVER_QQ_FIELD"] = "GENERIC_OVER_QQ_FIELD"

    @model_validator(mode="before")
    @classmethod
    def require_bounded_payload_shapes(cls, data: Any) -> Any:
        # Cap relayed solution payloads against the retained source's column
        # count BEFORE nested RationalFunction parsing; an unbounded tuple of
        # individually valid values would otherwise be fully parsed before
        # any later check rejects it.
        if not isinstance(data, dict):
            return data
        limit = _raw_system_column_bound(data.get("system"))
        for key in ("solution", "particular_solution"):
            value = data.get(key)
            if isinstance(value, (list, tuple)) and len(value) > limit:
                raise _validation_error(
                    "shape_mismatch",
                    f"{key} length {len(value)} exceeds the retained system's "
                    f"column count {limit}",
                )
        basis = data.get("nullspace_basis")
        if isinstance(basis, (list, tuple)):
            if len(basis) > limit:
                raise _validation_error(
                    "shape_mismatch",
                    f"nullspace_basis length {len(basis)} exceeds the "
                    f"retained system's column count {limit}",
                )
            for vector in basis:
                if isinstance(vector, (list, tuple)) and len(vector) > limit:
                    raise _validation_error(
                        "shape_mismatch",
                        "a nullspace basis vector exceeds the retained "
                        f"system's column count {limit}",
                    )
        return canonicalize_json_containers(data)

    def _require_witness_vector_shape(
        self,
        vector: tuple[RationalFunction, ...],
        *,
        label: str,
    ) -> None:
        columns = len(self.system.matrix.entries[0])
        if len(vector) != columns:
            raise _validation_error(
                "shape_mismatch",
                f"{label} must have exactly the retained system's column count",
            )
        if any(value.variables != self.system.matrix.variables for value in vector):
            raise _validation_error(
                "shape_mismatch",
                "witness vectors must use the retained system's declared ordered field",
            )

    def _require_classification_payload_shape(self) -> None:
        if self.classification == "UNIQUE":
            if self.solution is None:
                raise _validation_error(
                    "status_mismatch", "UNIQUE must carry a solution vector"
                )
            if self.particular_solution is not None or self.nullspace_basis is not None:
                raise _validation_error(
                    "status_mismatch",
                    "UNIQUE must not populate particular_solution or nullspace_basis",
                )
            self._require_witness_vector_shape(self.solution, label="solution")
        elif self.classification == "NON_UNIQUE":
            if self.particular_solution is None:
                raise _validation_error(
                    "status_mismatch", "NON_UNIQUE must carry a particular_solution"
                )
            if self.solution is not None:
                raise _validation_error(
                    "budget_exceeded",
                    "NON_UNIQUE must not populate the unique solution",
                )
            self._require_witness_vector_shape(
                self.particular_solution,
                label="particular_solution",
            )
            for vector in self.nullspace_basis or ():
                self._require_witness_vector_shape(
                    vector, label="nullspace basis vector"
                )
        elif (
            self.solution is not None
            or self.particular_solution is not None
            or self.nullspace_basis is not None
        ):
            raise _validation_error(
                "budget_exceeded", "INCONSISTENT must not carry solution data"
            )

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        # Solution growth is bounded at request admission
        # (_require_linear_system_solution_budget), before the backend runs:
        # a parsed canonical RationalFunction already caps each side at
        # MAX_SYMBOLIC_RESULT_TERMS, so per-component term checks here would
        # be ineffective anyway.
        self._require_classification_payload_shape()
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        system: SymbolicLinearSystemRequest,
        classification: Literal["UNIQUE", "NON_UNIQUE", "INCONSISTENT"],
        solution: tuple[RationalFunction, ...] | None,
        particular_solution: tuple[RationalFunction, ...] | None,
        nullspace_basis: tuple[tuple[RationalFunction, ...], ...] | None,
    ) -> Self:
        """Construct a result from the owner-local bounded kernel output."""

        return cls.model_construct(
            system=system,
            classification=classification,
            solution=solution,
            particular_solution=particular_solution,
            nullspace_basis=nullspace_basis,
        )


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
