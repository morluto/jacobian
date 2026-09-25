"""Bounded exact modular-form bases and coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial, gcd, isqrt, lcm
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)
from jacobian.math.number_theory.modular_forms.kernel import expected_coefficients
from jacobian.math.number_theory.modular_forms.operations import space_dimension
from jacobian.math.number_theory.modular_forms.pari_backend import (
    MAX_PARI_BASIS_ALLOCATION_BYTES,
    MAX_PARI_BASIS_DIMENSION,
    MAX_PARI_BASIS_LEVEL,
    MAX_PARI_BASIS_PRECISION,
    MAX_PARI_BASIS_WEIGHT,
    MAX_PARI_BASIS_WORK,
    PARI_STURM_RREF_BASIS_ID,
    _rref_coefficient_digit_bound,
    pari_gamma0_atkin_matrix,
    pari_gamma0_rational_basis,
)
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_GAMMA0_THREE_BASIS_PRECISION,
    MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS,
    MAX_LEVEL_ONE_BASIS_COORDINATES,
    MAX_LEVEL_ONE_BASIS_PRECISION,
    MAX_LEVEL_ONE_BASIS_WEIGHT,
    MAX_MODULAR_FORM_LEVEL,
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
    ModularFormBasis,
    ModularFormBasisElement,
    ModularFormChangeOfBasisFrame,
    ModularFormCoordinates,
    ModularFormFramedCoordinates,
    ModularFormFramedHeckeMatrix,
    ModularFormHeckeMatrix,
    ModularFormOperatorImage,
    ModularFormOperatorImagePrefix,
    ModularFormSpace,
    ModularQExpansion,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

BASIS_ID: _BasisId = "level-one-e4-e6-monomials-v1"
GAMMA0_TWO_BASIS_ID: _BasisId = "gamma0-two-weight-2-4-monomials-v1"
GAMMA0_THREE_BASIS_ID: _BasisId = "gamma0-three-weight-2-4-6-hypersurface-v1"
GAMMA0_FOUR_BASIS_ID: _BasisId = "gamma0-four-weight-2-generators-v1"
GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID: _BasisId = "gamma0-four-chi4-weight-one-v1"
GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID: _BasisId = "gamma0-four-chi4-weight-three-v1"
_BasisId = Literal[
    "level-one-e4-e6-monomials-v1",
    "gamma0-two-weight-2-4-monomials-v1",
    "gamma0-three-weight-2-4-6-hypersurface-v1",
    "gamma0-four-weight-2-generators-v1",
    "gamma0-four-chi4-weight-one-v1",
    "gamma0-four-chi4-weight-three-v1",
    "gamma0-rational-gamma0-sturm-rref-v1",
]
MAX_LEVEL_ONE_BASIS_WORK = 4_000_000
MAX_LEVEL_ONE_BASIS_ALLOCATION_BYTES = 8 * 1024 * 1024
MAX_COORDINATE_RESULT_DIGITS = 4_096
MAX_COORDINATE_HECKE_WORK = 4_000_000
MAX_HECKE_MATRIX_ALLOCATION_BYTES = 8 * 1024 * 1024
MAX_OPERATOR_IMAGE_WORK = 4_000_000
MAX_OPERATOR_IMAGE_ALLOCATION_BYTES = 8 * 1024 * 1024
MAX_MODULAR_FORM_PRODUCT_WORK = 1 << 40
MAX_MODULAR_FORM_PRODUCT_INTERMEDIATE_BYTES = 32 * 1024 * 1024
MAX_CHANGE_OF_BASIS_DIGITS = MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS
_MAX_CHANGE_OF_BASIS_INTEGER = 10**MAX_CHANGE_OF_BASIS_DIGITS
MAX_CHANGE_OF_BASIS_WORK = 32_768
MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES = 8 * 1024 * 1024
MAX_ATKIN_LEHNER_MATRIX_ENTRY_DIGITS = 512
MAX_ATKIN_LEHNER_INTERNAL_DIGITS = 10_000_000
MAX_ATKIN_LEHNER_INTERNAL_BYTES = 256 * 1024 * 1024


def _hecke_coefficient(
    coefficients: tuple[int | Fraction, ...],
    index: int,
    m: int,
    weight: int,
    *,
    chi_minus4: bool = False,
) -> Fraction:
    """Apply the exact T_n coefficient formula to one q-coefficient."""

    value = Fraction(0)
    for divisor in range(1, index + 1):
        if index % divisor == 0 and m % divisor == 0:
            character_factor = (
                0
                if chi_minus4 and divisor % 2 == 0
                else -1
                if chi_minus4 and divisor % 4 == 3
                else 1
            )
            factor = (
                Fraction(1, divisor)
                if weight == 0
                else Fraction(divisor ** (weight - 1))
            )
            value += (
                character_factor
                * factor
                * coefficients[index * m // (divisor * divisor)]
            )
    return value


def _solve_sturm_coordinates(
    basis: tuple[tuple[int | Fraction, ...], ...], image: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Recover unique coordinates and require agreement through the Sturm bound."""

    dimension = len(basis)
    if dimension == 0:
        if any(image):
            raise RuntimeError("Hecke image does not lie in the zero-dimensional space")
        return ()
    rows = [
        [Fraction(vector[row]) for vector in basis] + [image[row]]
        for row in range(len(image))
    ]
    pivot_row = 0
    pivots: list[int] = []
    for column in range(dimension):
        request_checkpoint("during exact Sturm-coordinate elimination")
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]), None
        )
        if pivot is None:
            raise RuntimeError(
                "canonical basis is not independent through the Sturm bound"
            )
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [entry / scale for entry in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                a - scale * b for a, b in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
    if any(not any(row[:dimension]) and row[-1] for row in rows):
        raise RuntimeError("Hecke image fails the exact Sturm-bound membership check")
    solution = [Fraction(0)] * dimension
    for row, column in enumerate(pivots):
        solution[column] = rows[row][-1]
    return tuple(solution)


def _solve_gamma0_four_coordinates(
    basis: tuple[tuple[int | Fraction, ...], ...], image: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Solve in the q-unitriangular Gamma0(4) monomial basis."""

    dimension = len(basis)
    if dimension == 0:
        if any(image):
            raise RuntimeError("V_2 image does not lie in the zero-dimensional space")
        return ()
    if len(image) < dimension:
        raise RuntimeError("V_2 image is shorter than the target Sturm prefix")
    coordinates: list[Fraction] = []
    for row in range(dimension):
        pivot = basis[row][row]
        if pivot != 1:
            raise RuntimeError("Gamma0(4) monomial basis lost its unit leading term")
        residual = image[row] - sum(
            (coordinates[column] * basis[column][row] for column in range(row)),
            Fraction(0),
        )
        coordinates.append(residual)
    if any(
        sum(
            (coordinates[column] * basis[column][row] for column in range(dimension)),
            Fraction(0),
        )
        != image[row]
        for row in range(dimension, len(image))
    ):
        raise RuntimeError("V_2 image fails the exact Gamma0(4) Sturm check")
    return tuple(coordinates)


@dataclass(frozen=True)
class _BasisPlan:
    space: ModularFormSpace
    basis_id: _BasisId
    precision: int
    terms: tuple[tuple[int, int], ...]
    is_cuspidal: bool
    dimension: int
    coefficient_digits: int
    work: int
    basis_vectors: tuple[tuple[Fraction, ...], ...] | None = None
    basis_labels: tuple[str, ...] | None = None
    rref_digit_bound: int = 0


def _is_gamma0_four_chi4(space: ModularFormSpace) -> bool:
    character = space.character
    return (
        space.level == 4
        and space.weight in (1, 3)
        and space.kind == "M"
        and isinstance(character, DirichletCharacter)
        and character.group.modulus == 4
        and character.group.generator_orders == (2,)
        and character.coordinates == (1,)
    )


def _gamma0_four_chi4_basis(
    space: ModularFormSpace,
) -> tuple[_BasisId, tuple[tuple[int, int], ...]] | None:
    if not _is_gamma0_four_chi4(space):
        return None
    if space.weight == 1:
        return GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID, ((0, 0),)
    if space.weight == 3:
        return GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID, ((0, 0), (0, 1))
    return None


def _monomial_terms(weight: int) -> tuple[tuple[int, int], ...]:
    """Exponents for level-one E4/E6 monomials of the requested weight."""
    if weight < 0 or weight % 2:
        return ()
    return tuple(
        (remaining // 4, exponent_e6)
        for exponent_e6 in range(weight // 6 + 1)
        if (remaining := weight - 6 * exponent_e6) % 4 == 0
    )


def _gamma0_two_terms(weight: int) -> tuple[tuple[int, int], ...]:
    """Exponents for A2^a B4^b where 2a + 4b equals the weight."""

    if weight < 0 or weight % 2:
        return ()
    return tuple(
        ((weight - 4 * exponent_b) // 2, exponent_b)
        for exponent_b in range(weight // 4 + 1)
    )


def _gamma0_three_terms(weight: int) -> tuple[tuple[int, int], ...]:
    """Reduced monomials A2^a E4star^b S6^c with b in {0,1}.

    The exact Gamma0(3) graded ring has generators of weights 2, 4, 6
    with its single weight-8 relation eliminating every square of E4star.
    The returned pair records (b,c); a is determined by the requested weight.
    """

    if weight < 0 or weight % 2:
        return ()
    return tuple(
        (exponent_b, exponent_s6)
        for exponent_s6 in range(weight // 6 + 1)
        for exponent_b in (0, 1)
        if weight >= 4 * exponent_b + 6 * exponent_s6
        and (weight - 4 * exponent_b - 6 * exponent_s6) % 2 == 0
    )


def _gamma0_four_terms(weight: int) -> tuple[tuple[int, int], ...]:
    """Exponents for B4^(n-j)D4^j, where 2n is the requested weight."""
    if weight < 0 or weight % 2:
        return ()
    degree = weight // 2
    return tuple((degree - exponent_d, exponent_d) for exponent_d in range(degree + 1))


def _is_prime_level(value: object) -> bool:
    if type(value) is not int or value < 2:
        return False
    return all(value % divisor for divisor in range(2, isqrt(value) + 1))


def _has_admitted_basis_level(space: ModularFormSpace) -> bool:
    return type(space.level) is int and 1 <= space.level <= MAX_PARI_BASIS_LEVEL


def _native_level_requires_holomorphic(space: ModularFormSpace) -> bool:
    return space.level in (2, 4) and space.kind != "M"


def _admit_basis_precision(precision: object) -> int:
    if (
        type(precision) is not int
        or not 1 <= precision <= MAX_LEVEL_ONE_BASIS_PRECISION
    ):
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.basis_precision_bound",
            message=(
                "basis precision must lie in the envelope "
                f"[1, {MAX_LEVEL_ONE_BASIS_PRECISION}]"
            ),
        )
    return precision


def _digit_bound(weight: int, precision: int) -> int:
    """Bound coefficients from divisor-sum bounds and convolution counts."""

    if weight == 0:
        return 1
    factors = (weight + 3) // 4
    bits = (
        (weight + factors) * max(1, precision.bit_length())
        + 9 * factors
        + precision
        + 4
    )
    return (bits * 30_103 + 99_999) // 100_000 + 1


def _basis_terms_for_space(
    space: ModularFormSpace, is_cuspidal: bool
) -> tuple[_BasisId, tuple[tuple[int, int], ...]]:
    """Select one exact basis convention and its ordered monomial indices."""

    character_basis = _gamma0_four_chi4_basis(space)
    if character_basis is not None:
        return character_basis
    if space.level == 1:
        inner_weight = space.weight - 12 if is_cuspidal else space.weight
        return BASIS_ID, _monomial_terms(inner_weight)
    if space.level == 2:
        return GAMMA0_TWO_BASIS_ID, _gamma0_two_terms(space.weight)
    if space.level == 3:
        if space.kind != "M":
            raise OperationDomainValidationError(
                location=("space", "kind"),
                code="modular_form.basis_unsupported_space",
                message="the Gamma0(3) basis currently covers M_k spaces only",
            )
        return GAMMA0_THREE_BASIS_ID, _gamma0_three_terms(space.weight)
    return GAMMA0_FOUR_BASIS_ID, _gamma0_four_terms(space.weight)


def _admit_basis(
    space: object,
    precision: object,
    *,
    materialize_pari: bool = True,
    at_least_sturm: bool = False,
) -> _BasisPlan:
    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.basis_space_type",
            message="basis computation requires a modular-form space value",
        )
    precision = _admit_basis_precision(precision)
    if not _has_admitted_basis_level(space):
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.basis_unsupported_level",
            message=f"deterministic bases are admitted only through level {MAX_PARI_BASIS_LEVEL}",
        )
    if _native_level_requires_holomorphic(space):
        raise OperationDomainValidationError(
            location=("space", "kind"),
            code="modular_form.basis_unsupported_space",
            message="the higher-level bases currently cover holomorphic M_k spaces only",
        )
    if type(space.weight) is not int or space.weight < 0:
        raise OperationDomainValidationError(
            location=("space", "weight"),
            code="modular_form.basis_weight_type",
            message="space weight must be a nonnegative integer",
        )
    if space.weight > MAX_LEVEL_ONE_BASIS_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.basis_weight_bound",
            message=(
                "basis weight exceeds the exact envelope "
                f"[0, {MAX_LEVEL_ONE_BASIS_WEIGHT}]"
            ),
        )
    if space.level > 4:
        return _admit_pari_basis(
            space,
            precision,
            materialize=materialize_pari,
            at_least_sturm=at_least_sturm,
        )
    if space.level == 3 and precision > MAX_GAMMA0_THREE_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.gamma0_three_precision_bound",
            message=(
                "Gamma0(3) basis precision must not exceed "
                f"{MAX_GAMMA0_THREE_BASIS_PRECISION} terms"
            ),
        )
    dimension_result = space_dimension(space)
    dimension = dimension_result.dimension
    is_cuspidal = space.kind == "S"
    basis_id, terms = _basis_terms_for_space(space, is_cuspidal)
    if len(terms) != dimension:
        raise RuntimeError("modular-form monomial count disagrees with exact dimension")
    if dimension > MAX_LEVEL_ONE_BASIS_COORDINATES:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.basis_dimension_bound",
            message="modular-form basis dimension exceeds its coordinate envelope",
        )

    if basis_id == GAMMA0_THREE_BASIS_ID:
        max_a = max(
            ((space.weight - 4 * b - 6 * c) // 2 for b, c in terms),
            default=0,
        )
        max_s6 = max((c for _, c in terms), default=0)
        convolution_count = max_a + 1 + max_s6 + 2 * dimension
    else:
        max_e4 = max((a for a, _ in terms), default=0)
        max_e6 = max((b for _, b in terms), default=0)
        convolution_count = max_e4 + max_e6
        convolution_count += sum(bool(a and b) for a, b in terms)
        if is_cuspidal:
            convolution_count += dimension
    incidences = precision * (precision + 1) // 2
    work = convolution_count * incidences + 2 * precision * isqrt(precision)
    if basis_id == GAMMA0_THREE_BASIS_ID:
        # The finite eta product has at most 4P/3 factors and each bounded
        # factor multiply visits at most 7P coefficients.
        work += 12 * precision * precision
    if work > MAX_LEVEL_ONE_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.basis_work_bound",
            message="basis coefficient work exceeds its exact envelope",
        )

    coefficient_digits = (
        _digit_bound(space.weight, precision)
        if space.level == 1
        else precision.bit_length() + 1
        if basis_id
        in (
            GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
            GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        )
        else _gamma0_three_digit_bound(space.weight, precision)
        if basis_id == GAMMA0_THREE_BASIS_ID
        else _gamma0_two_digit_bound(space.weight, precision)
    )
    output_digits = dimension * precision * (coefficient_digits + 8)
    if (
        coefficient_digits > MAX_COORDINATE_RESULT_DIGITS
        or output_digits > MAX_LEVEL_ONE_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.basis_output_bound",
            message="basis coefficient growth exceeds its output envelope",
        )
    return _BasisPlan(
        space=space,
        basis_id=basis_id,
        precision=precision,
        terms=terms,
        is_cuspidal=is_cuspidal,
        dimension=dimension,
        coefficient_digits=coefficient_digits,
        work=work,
    )


def _admit_pari_basis(
    space: ModularFormSpace,
    precision: int,
    *,
    materialize: bool,
    at_least_sturm: bool = False,
) -> _BasisPlan:
    """Preflight a PARI basis through at least the Sturm determining prefix."""

    if space.character != "TRIVIAL" or space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.pari_basis_parent",
            message="the extended basis backend supports QQ and trivial character only",
        )
    if space.level > MAX_PARI_BASIS_LEVEL:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.pari_basis_level_bound",
            message=f"PARI basis level exceeds {MAX_PARI_BASIS_LEVEL}",
        )
    if space.weight > MAX_PARI_BASIS_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.pari_basis_weight_bound",
            message=f"PARI basis weight exceeds {MAX_PARI_BASIS_WEIGHT}",
        )
    dimension_data = space_dimension(space)
    dimension = dimension_data.dimension
    if dimension > MAX_PARI_BASIS_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.pari_basis_dimension_bound",
            message=f"PARI basis dimension exceeds {MAX_PARI_BASIS_DIMENSION}",
        )
    bound = sturm_bound(space).bound
    sturm_precision = bound + 1
    if sturm_precision > MAX_PARI_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.pari_basis_sturm_precision_bound",
            message="Sturm determining precision exceeds the PARI basis envelope",
        )
    if precision < sturm_precision:
        if not at_least_sturm:
            raise OperationDomainValidationError(
                location=("precision",),
                code="modular_form.pari_basis_requires_sturm_precision",
                message=(
                    "PARI basis precision must include coefficients through the "
                    f"Sturm bound (at least {sturm_precision} terms)"
                ),
            )
        # A coordinate-defined form is already globally identified, so a
        # requested prefix shorter than the determining bound is evaluated
        # from the internal Sturm-determining basis and truncated.
        precision = sturm_precision
    if precision > MAX_PARI_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.pari_basis_precision_bound",
            message=f"PARI basis precision exceeds {MAX_PARI_BASIS_PRECISION}",
        )
    backend_work = (
        max(1, dimension)
        * max(1, precision)
        * dimension_data.index
        * max(1, space.weight)
    )
    rref_digits = _rref_coefficient_digit_bound(dimension)
    rref_work = dimension * dimension * precision * rref_digits
    work = backend_work + rref_work
    if work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_basis_work_bound",
            message="PARI modular-form basis work exceeds its exact envelope",
        )
    allocation_bytes = dimension * precision * (2 * rref_digits + 32)
    if allocation_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_basis_output_bound",
            message="worst-case exact basis encoding exceeds the output envelope",
        )
    plan = _BasisPlan(
        space=space,
        basis_id=PARI_STURM_RREF_BASIS_ID,
        precision=precision,
        terms=(),
        is_cuspidal=space.kind == "S",
        dimension=dimension,
        coefficient_digits=(
            1 if dimension == sturm_precision == precision else rref_digits
        ),
        work=work,
        rref_digit_bound=rref_digits,
    )
    return _materialize_pari_basis(plan) if materialize else plan


def _materialize_pari_basis(plan: _BasisPlan) -> _BasisPlan:
    sturm_precision = sturm_bound(plan.space).bound + 1
    vectors = pari_gamma0_rational_basis(
        plan.space,
        plan.precision,
        sturm_precision,
        plan.dimension,
        admitted_work=plan.work,
        admitted_rref_digits=plan.rref_digit_bound,
    )
    labels = tuple(
        f"q^{next(index for index, value in enumerate(vector[:sturm_precision]) if value)}"
        for vector in vectors
    )
    if len(set(labels)) != len(labels):
        raise RuntimeError("canonical q-Sturm frame has duplicate pivot labels")
    return _BasisPlan(
        space=plan.space,
        basis_id=plan.basis_id,
        precision=plan.precision,
        terms=plan.terms,
        is_cuspidal=plan.is_cuspidal,
        dimension=plan.dimension,
        coefficient_digits=plan.coefficient_digits,
        work=plan.work,
        basis_vectors=vectors,
        basis_labels=labels,
        rref_digit_bound=plan.rref_digit_bound,
    )


def _gamma0_two_digit_bound(weight: int, precision: int) -> int:
    """Upward bound for coefficients of A2^a B4^b at total weight k."""

    if weight == 0 or precision == 1:
        return 1
    # Use log10(P) <= bit_length(P); the integer bound intentionally
    # dominates divisor-sum sizes and each truncated convolution cardinality.
    bit_bound = precision.bit_length()
    a_max = weight // 2
    b_max = weight // 4
    factor_count = a_max + b_max
    if factor_count == 0:
        return 1
    return (
        a_max * (3 + 2 * bit_bound)
        + b_max * (3 + 4 * bit_bound)
        + (factor_count - 1) * (bit_bound + 1)
        + 2
    )


def _gamma0_three_digit_bound(weight: int, precision: int) -> int:
    """Bound coefficients of each A2/E4star/S6 monomial by its exponents."""

    if weight == 0 or precision == 1:
        return 1
    a_digits = len(str(48 * precision**3))
    e4_digits = len(str(300 * precision**5))
    # The coefficient l1 norm is bounded by 2^(8P): the two eta factors
    # contribute at most P and P/3 factors, each with l1 norm 64.
    s6_digits = (8 * precision * 30_103 + 99_999) // 100_000 + 1
    terms = _gamma0_three_terms(weight)
    return max(
        (
            ((weight - 4 * b - 6 * c) // 2) * a_digits
            + b * e4_digits
            + c * s6_digits
            + 2
            for b, c in terms
        ),
        default=1,
    )


def _gamma0_three_eta_product(precision: int) -> tuple[int, ...]:
    """Return q*prod(1-q^n)^6(1-q^(3n))^6 through q^(P-1)."""

    if precision == 1:
        return (0,)
    product = [1] + [0] * (precision - 2)
    for step in (1, 3):
        n = 1
        while step * n <= precision - 2:
            shift = step * n
            next_product = product.copy()
            for degree, value in enumerate(product):
                for exponent in range(1, min(6, (precision - 2 - degree) // shift) + 1):
                    next_product[degree + exponent * shift] += (
                        value * (-1 if exponent % 2 else 1) * comb(6, exponent)
                    )
            product = next_product
            n += 1
    return (0, *product[: precision - 1])


def _gamma0_four_chi_digit_bound(weight: int, precision: int) -> int:
    """Bound coefficients of the reviewed chi_-4 Eisenstein bases."""

    bit_bound = precision.bit_length()
    if weight == 1:
        # The positive coefficients are divisor sums of a sign-valued character.
        return bit_bound + 1
    # At weight three, both divisor-sum families are bounded by n^3, with the
    # first basis vector carrying an additional factor of four.
    return 3 * bit_bound + 3


def _multiply_series(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    order = len(left)
    result = []
    for degree in range(order):
        if degree % 32 == 0:
            request_checkpoint(
                "during exact modular-form basis q-series multiplication"
            )
        result.append(
            sum(left[index] * right[degree - index] for index in range(degree + 1))
        )
    return tuple(result)


def _powers(base: tuple[int, ...], maximum: int) -> tuple[tuple[int, ...], ...]:
    one = (1, *([0] * (len(base) - 1)))
    values = [one]
    for _ in range(maximum):
        values.append(_multiply_series(values[-1], base))
    return tuple(values)


def _gamma0_three_basis_coefficients(plan: _BasisPlan) -> tuple[tuple[int, ...], ...]:
    """Build the exact reduced monomial basis from its three generators."""

    order = plan.precision
    sigma_one = _sigma_coefficients(1, order)
    sigma_three = _sigma_coefficients(3, order)
    a2 = (
        1,
        *(
            12 * sigma_one[n] - (36 * sigma_one[n // 3] if n % 3 == 0 else 0)
            for n in range(1, order)
        ),
    )
    e4star = (
        1,
        *(
            (270 * sigma_three[n // 3] if n % 3 == 0 else 0) - 30 * sigma_three[n]
            for n in range(1, order)
        ),
    )
    s6 = _gamma0_three_eta_product(order)
    maximum_a = max(
        ((plan.space.weight - 4 * b - 6 * c) // 2 for b, c in plan.terms),
        default=0,
    )
    first = _powers(a2, maximum_a)
    second = _powers(e4star, 1)
    third = _powers(s6, max((c for _, c in plan.terms), default=0))
    vectors = []
    for b, c in plan.terms:
        exponent_a = (plan.space.weight - 4 * b - 6 * c) // 2
        vectors.append(
            _multiply_series(_multiply_series(first[exponent_a], second[b]), third[c])
        )
    return tuple(vectors)


def _basis_coefficients(plan: _BasisPlan) -> tuple[tuple[Fraction, ...], ...]:
    return plan.basis_vectors or _formula_basis_coefficients(plan)


def _formula_basis_coefficients(plan: _BasisPlan) -> tuple[tuple[Fraction, ...], ...]:
    order = plan.precision
    terms = plan.terms
    if plan.basis_id == GAMMA0_THREE_BASIS_ID:
        return tuple(
            tuple(Fraction(value) for value in vector)
            for vector in _gamma0_three_basis_coefficients(plan)
        )
    if plan.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID:
        divisor_sums = [0] * order
        for divisor in range(1, order, 2):
            chi = 1 if divisor % 4 == 1 else -1
            for multiple in range(divisor, order, divisor):
                divisor_sums[multiple] += chi
        coefficients = [Fraction(1, 4), *(Fraction(c) for c in divisor_sums[1:])]
        return (tuple(coefficients),)
    if plan.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID:
        chi_first: list[int] = [0] * order
        chi_second: list[int] = [0] * order
        for divisor in range(1, order):
            divisor_square = divisor * divisor
            for multiple in range(divisor, order, divisor):
                quotient = multiple // divisor
                if quotient % 2:
                    chi_quotient = 1 if quotient % 4 == 1 else -1
                    chi_second[multiple] += chi_quotient * divisor_square
            if divisor % 2:
                chi_divisor = 1 if divisor % 4 == 1 else -1
                for multiple in range(divisor, order, divisor):
                    chi_first[multiple] -= 4 * chi_divisor * divisor_square
        chi_first[0] = 1
        return (
            tuple(Fraction(value) for value in chi_first),
            tuple(Fraction(value) for value in chi_second),
        )
    monomial_first: tuple[tuple[int, ...], ...]
    monomial_second: tuple[tuple[int, ...], ...]
    if plan.space.level == 1:
        e4 = tuple(int(value) for value in expected_coefficients("E4", order))
        e6 = tuple(int(value) for value in expected_coefficients("E6", order))
        monomial_first = _powers(e4, max((a for a, _ in terms), default=0))
        monomial_second = _powers(e6, max((b for _, b in terms), default=0))
    elif plan.space.level == 2:
        sigma_one = _sigma_coefficients(1, order)
        a2 = (
            1,
            *(
                24 * sigma_one[n] - (48 * sigma_one[n // 2] if n % 2 == 0 else 0)
                for n in range(1, order)
            ),
        )
        # The level-one normalized E4 is also a modular form on Gamma0(2).
        # Zagier's free-ring description gives generators A2 (weight 2) and
        # E4 (weight 4), so these coefficients are the exact level-two basis.
        e4 = tuple(int(value) for value in expected_coefficients("E4", order))
        monomial_first = _powers(a2, max((a for a, _ in terms), default=0))
        monomial_second = _powers(e4, max((b for _, b in terms), default=0))
    else:
        sigma_one = _sigma_coefficients(1, order)
        a2 = (
            1,
            *(
                24 * sigma_one[n] - (48 * sigma_one[n // 2] if n % 2 == 0 else 0)
                for n in range(1, order)
            ),
        )
        b4 = tuple(
            1 if n == 0 else a2[n // 2] if n % 2 == 0 else 0 for n in range(order)
        )
        d4 = tuple((a2[n] - b4[n]) // 24 for n in range(order))
        monomial_first = _powers(b4, max((a for a, _ in terms), default=0))
        monomial_second = _powers(d4, max((d for _, d in terms), default=0))
    delta = (
        tuple(int(value) for value in expected_coefficients("DELTA", order))
        if plan.is_cuspidal
        else ()
    )
    output: list[tuple[Fraction, ...]] = []
    for a, b in terms:
        int_vector = (
            _multiply_series(monomial_first[a], monomial_second[b])
            if a and b
            else monomial_first[a]
            if a
            else monomial_second[b]
        )
        if plan.is_cuspidal:
            int_vector = _multiply_series(delta, int_vector)
        output.append(tuple(Fraction(value) for value in int_vector))
    return tuple(output)


def _sigma_coefficients(power: int, precision: int) -> tuple[int, ...]:
    """Compute divisor sums below the admitted precision in O(P log P) work."""

    values = [0] * precision
    for divisor in range(1, precision):
        divisor_power = divisor**power
        for multiple in range(divisor, precision, divisor):
            values[multiple] += divisor_power
    return tuple(values)


def modular_form_basis_q_expansions(
    space: ModularFormSpace, precision: int
) -> ModularFormBasis:
    """Return a complete exact basis prefix in a supported space."""

    plan = _admit_basis(space, precision)
    vectors = _basis_coefficients(plan)
    elements = []
    labels = _basis_labels(plan)
    for label, vector in zip(labels, vectors, strict=True):
        expansion = ModularQExpansion.model_construct(
            space=space,
            weight=space.weight,
            basis_id=plan.basis_id,
            q_expansion=TruncatedSeries.model_construct(
                variable="q",
                truncation_order=precision,
                coefficients=tuple(
                    CanonicalRational(
                        num=Fraction(c).numerator, den=Fraction(c).denominator
                    )
                    for c in vector
                ),
            ),
        )
        elements.append(
            ModularFormBasisElement.model_construct(label=label, expansion=expansion)
        )
    return ModularFormBasis.model_construct(
        space=space,
        basis_id=plan.basis_id,
        precision=precision,
        elements=tuple(elements),
    )


def _basis_labels(plan: _BasisPlan) -> tuple[str, ...]:
    if plan.basis_labels is not None:
        return plan.basis_labels
    labels: list[str] = []
    for a, b in plan.terms:
        if plan.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID:
            label = "G1_chi_minus4"
        elif plan.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID:
            label = ("A3_chi_minus4", "B3_chi_minus4")[len(labels)]
        elif plan.basis_id == BASIS_ID:
            label = ("Delta*" if plan.is_cuspidal else "") + _monomial_label(a, b)
        elif plan.basis_id == GAMMA0_TWO_BASIS_ID:
            label = _gamma0_two_monomial_label(a, b)
        elif plan.basis_id == GAMMA0_THREE_BASIS_ID:
            exponent_a = (plan.space.weight - 4 * a - 6 * b) // 2
            label = _gamma0_three_monomial_label(exponent_a, a, b)
        else:
            label = _gamma0_four_monomial_label(a, b)
        labels.append(label)
    return tuple(labels)


def _frame_admission(
    frame: object,
) -> tuple[_BasisPlan, tuple[tuple[Fraction, ...], ...]]:
    """Validate an exact frame and admit matrix arithmetic before elimination."""
    if not isinstance(frame, ModularFormChangeOfBasisFrame):
        raise OperationDomainValidationError(
            location=("frame",),
            code="modular_form.frame_type",
            message="frame must be a typed modular-form change-of-basis value",
        )
    frame_precision = sturm_bound(frame.space).bound + 1
    plan = _admit_basis(frame.space, frame_precision, materialize_pari=False)
    if (
        type(frame.source_labels) is not tuple
        or type(frame.labels) is not tuple
        or any(
            type(label) is not str or not label or len(label) > 96
            for label in frame.labels
        )
        or type(frame.entries) is not tuple
        or any(type(row) is not tuple for row in frame.entries)
    ):
        raise OperationDomainValidationError(
            location=("frame",),
            code="modular_form.frame_shape",
            message="frame axes and matrix rows must be immutable tuples with valid labels",
        )
    if frame.source_basis_id != plan.basis_id or (
        plan.basis_id != PARI_STURM_RREF_BASIS_ID
        and frame.source_labels != _basis_labels(plan)
    ):
        raise OperationDomainValidationError(
            location=("frame", "source_labels"),
            code="modular_form.frame_source_basis",
            message="frame source labels and basis identifier must match the exact canonical basis",
        )
    n = plan.dimension
    if (
        len(frame.labels) != n
        or len(set(frame.labels)) != n
        or len(frame.entries) != n
        or any(len(row) != n for row in frame.entries)
    ):
        raise OperationDomainValidationError(
            location=("frame", "entries"),
            code="modular_form.frame_shape",
            message="frame matrix and its unique labels must match the source dimension",
        )
    matrix: list[tuple[Fraction, ...]] = []
    max_digits = 1
    for row in frame.entries:
        values = []
        for value in row:
            if not isinstance(value, CanonicalRational):
                raise OperationDomainValidationError(
                    location=("frame", "entries"),
                    code="modular_form.frame_rational_type",
                    message="frame entries must be canonical rationals",
                )
            numerator, denominator = (
                getattr(value, "num", None),
                getattr(value, "den", None),
            )
            if (
                type(numerator) is not int
                or type(denominator) is not int
                or denominator <= 0
            ):
                raise OperationDomainValidationError(
                    location=("frame", "entries"),
                    code="modular_form.frame_rational_value",
                    message="frame rationals must have positive reduced denominators",
                )
            if (
                abs(numerator) >= _MAX_CHANGE_OF_BASIS_INTEGER
                or denominator >= _MAX_CHANGE_OF_BASIS_INTEGER
            ):
                raise OperationResourceAdmissionError(
                    location=("frame", "entries"),
                    code="modular_form.frame_entry_digit_bound",
                    message=f"frame entries are limited to {MAX_CHANGE_OF_BASIS_DIGITS} decimal digits",
                )
            digits = max(len(str(abs(numerator))), len(str(denominator)))
            rational = Fraction(numerator, denominator)
            if (rational.numerator, rational.denominator) != (numerator, denominator):
                raise OperationDomainValidationError(
                    location=("frame", "entries"),
                    code="modular_form.frame_rational_value",
                    message="frame rationals must use canonical reduced form",
                )
            max_digits = max(max_digits, digits)
            values.append(rational)
        matrix.append(tuple(values))
    if plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        # Complete every request admission that does not depend on backend
        # output before launching the PARI worker for the canonical basis.
        frame_matrix = tuple(matrix)
        zero = tuple(Fraction(0) for _ in range(n))
        _admit_change_of_basis_arithmetic(n, frame_matrix, zero)
        _solve_frame_matrix(frame_matrix, zero)
        plan = _materialize_pari_basis(plan)
        if frame.source_labels != _basis_labels(plan):
            raise OperationDomainValidationError(
                location=("frame", "source_labels"),
                code="modular_form.frame_source_basis",
                message="frame source labels and basis identifier must match the exact canonical basis",
            )
    return plan, tuple(matrix)


def _admit_change_of_basis_arithmetic(
    dimension: int,
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> None:
    """Conservative Hadamard-style bound for exact elimination and output."""
    max_digits = max(
        (
            max(len(str(abs(v.numerator))), len(str(v.denominator)))
            for row in matrix
            for v in row
        ),
        default=1,
    )
    vector_digits = max(
        (max(len(str(abs(v.numerator))), len(str(v.denominator))) for v in vector),
        default=1,
    )
    determinant_digits = (
        dimension * (max_digits + vector_digits + 2) + dimension * dimension
    )
    estimated_digits = max(determinant_digits, 1)
    work = dimension**3
    frame_bytes = dimension * dimension * (2 * MAX_CHANGE_OF_BASIS_DIGITS + 32)
    coordinate_bytes = dimension * (2 * estimated_digits + 64)
    allocation_bytes = frame_bytes + coordinate_bytes
    if (
        work > MAX_CHANGE_OF_BASIS_WORK
        or estimated_digits > MAX_COORDINATE_RESULT_DIGITS
        or allocation_bytes > MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("frame",),
            code="modular_form.frame_conversion_bound",
            message="exact change-of-basis elimination exceeds its admitted work or rational-growth envelope",
        )


def _solve_frame_matrix(
    matrix: tuple[tuple[Fraction, ...], ...], vector: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Solve M y = x by exact Gauss-Jordan elimination; singularity is semantic."""
    n = len(matrix)
    rows = [[*row, vector[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        request_checkpoint("during exact modular-form change-of-basis elimination")
        pivot = next((row for row in range(column, n) if rows[row][column]), None)
        if pivot is None:
            raise OperationDomainValidationError(
                location=("frame", "entries"),
                code="modular_form.frame_singular",
                message="change-of-basis matrix must be invertible for coordinate conversion",
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [entry / scale for entry in rows[column]]
        for row in range(n):
            if row == column or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                a - scale * b for a, b in zip(rows[row], rows[column], strict=True)
            ]
    return tuple(rows[row][-1] for row in range(n))


def modular_form_basis_frame(
    request: ModularFormBasisFrameRequest,
) -> ModularFormChangeOfBasisFrame:
    """Create a validated source-bound rational basis frame."""
    if not isinstance(request, ModularFormBasisFrameRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="modular_form.frame_request_type",
            message="frame creation requires a typed frame request",
        )
    frame = request.as_frame()
    plan, matrix = _frame_admission(frame)
    zero = tuple(Fraction(0) for _ in range(plan.dimension))
    _admit_change_of_basis_arithmetic(plan.dimension, matrix, zero)
    _solve_frame_matrix(matrix, zero)
    return frame


def modular_form_coordinates_to_frame(
    frame: ModularFormChangeOfBasisFrame, form: ModularFormCoordinates
) -> ModularFormFramedCoordinates:
    """Express canonical coordinates in a validated caller basis."""
    plan, matrix = _frame_admission(frame)
    if (
        not isinstance(form, ModularFormCoordinates)
        or form.space != frame.space
        or form.basis_id != frame.source_basis_id
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.frame_wrong_space",
            message="canonical coordinates must use the frame's exact space and source basis",
        )
    _, canonical = _admit_coordinates(form, 1, admitted_plan=plan)
    _admit_change_of_basis_arithmetic(plan.dimension, matrix, canonical)
    framed = _solve_frame_matrix(matrix, canonical)
    return ModularFormFramedCoordinates.model_construct(
        frame=frame,
        coordinates=tuple(
            CanonicalRational(num=v.numerator, den=v.denominator) for v in framed
        ),
    )


def modular_form_coordinates_from_frame(
    form: ModularFormFramedCoordinates,
) -> ModularFormCoordinates:
    """Return canonical coordinates from an invertible framed coordinate value."""
    if not isinstance(form, ModularFormFramedCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.framed_coordinates_type",
            message="form must be an exact framed modular-form coordinate value",
        )
    plan, matrix = _frame_admission(form.frame)
    if type(form.coordinates) is not tuple or len(form.coordinates) != plan.dimension:
        raise OperationDomainValidationError(
            location=("form", "coordinates"),
            code="modular_form.framed_coordinates_shape",
            message="framed coordinate count must equal the exact source dimension",
        )
    coordinates = []
    for value in form.coordinates:
        if (
            not isinstance(value, CanonicalRational)
            or type(value.num) is not int
            or type(value.den) is not int
            or value.den <= 0
        ):
            raise OperationDomainValidationError(
                location=("form", "coordinates"),
                code="modular_form.framed_coordinate_value",
                message="framed coordinates must be canonical rational values",
            )
        if (
            abs(value.num) >= _MAX_CHANGE_OF_BASIS_INTEGER
            or value.den >= _MAX_CHANGE_OF_BASIS_INTEGER
        ):
            raise OperationResourceAdmissionError(
                location=("form", "coordinates"),
                code="modular_form.framed_coordinate_digit_bound",
                message="framed coordinates exceed the exact decimal digit envelope",
            )
        rational = Fraction(value.num, value.den)
        if (rational.numerator, rational.denominator) != (value.num, value.den):
            raise OperationDomainValidationError(
                location=("form", "coordinates"),
                code="modular_form.framed_coordinate_value",
                message="framed coordinates must use canonical reduced rationals",
            )
        coordinates.append(rational)
    framed = tuple(coordinates)
    _admit_change_of_basis_arithmetic(plan.dimension, matrix, framed)
    # The framed value is only meaningful when its declared basis is invertible.
    _solve_frame_matrix(matrix, tuple(Fraction(0) for _ in range(plan.dimension)))
    canonical = tuple(
        sum(
            (matrix[row][column] * framed[column] for column in range(plan.dimension)),
            Fraction(0),
        )
        for row in range(plan.dimension)
    )
    return ModularFormCoordinates.model_construct(
        space=form.frame.space,
        basis_id=form.frame.source_basis_id,
        coordinates=tuple(
            CanonicalRational(num=v.numerator, den=v.denominator) for v in canonical
        ),
    )


def _gamma0_two_monomial_label(exponent_a2: int, exponent_e4: int) -> str:
    factors = []
    for symbol, exponent in (("A2", exponent_a2), ("E4", exponent_e4)):
        if exponent:
            factors.append(symbol if exponent == 1 else f"{symbol}^{exponent}")
    return "*".join(factors) if factors else "1"


def _gamma0_four_monomial_label(exponent_b4: int, exponent_d4: int) -> str:
    factors = []
    for symbol, exponent in (("B4", exponent_b4), ("D4", exponent_d4)):
        if exponent:
            factors.append(symbol if exponent == 1 else f"{symbol}^{exponent}")
    return "*".join(factors) if factors else "1"


def _gamma0_three_monomial_label(
    exponent_a2: int, exponent_e4: int, exponent_s6: int
) -> str:
    factors = []
    for symbol, exponent in (
        ("A2", exponent_a2),
        ("B4star", exponent_e4),
        ("S6", exponent_s6),
    ):
        if exponent:
            factors.append(symbol if exponent == 1 else f"{symbol}^{exponent}")
    return "*".join(factors) if factors else "1"


def _monomial_label(exponent_e4: int, exponent_e6: int) -> str:
    factors = []
    for symbol, exponent in (("E4", exponent_e4), ("E6", exponent_e6)):
        if exponent:
            factors.append(symbol if exponent == 1 else f"{symbol}^{exponent}")
    return "*".join(factors) if factors else "1"


def _admit_coordinates(
    form: object,
    precision: int,
    *,
    admitted_plan: _BasisPlan | None = None,
    materialize_pari: bool = True,
    check_expansion_growth: bool = True,
    allow_short_prefix: bool = False,
) -> tuple[_BasisPlan, tuple[Fraction, ...]]:
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if form.basis_id not in (
        BASIS_ID,
        GAMMA0_TWO_BASIS_ID,
        GAMMA0_THREE_BASIS_ID,
        GAMMA0_FOUR_BASIS_ID,
        GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
        GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        PARI_STURM_RREF_BASIS_ID,
    ):
        raise OperationDomainValidationError(
            location=("form", "basis_id"),
            code="modular_form.coordinates_basis",
            message="form coordinates use an unsupported basis convention",
        )
    plan = (
        _admit_basis(
            form.space,
            precision,
            materialize_pari=False,
            at_least_sturm=allow_short_prefix,
        )
        if admitted_plan is None
        else admitted_plan
    )
    if form.space != plan.space:
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.coordinates_space_parent",
            message="coordinate space must match its admitted canonical basis",
        )
    if form.basis_id != plan.basis_id:
        raise OperationDomainValidationError(
            location=("form", "basis_id"),
            code="modular_form.coordinates_basis_parent",
            message="basis identifier must match the coordinate space",
        )
    coordinates = getattr(form, "coordinates", None)
    if type(coordinates) is not tuple or len(coordinates) != plan.dimension:
        raise OperationDomainValidationError(
            location=("form", "coordinates"),
            code="modular_form.coordinates_shape",
            message="coordinate count must equal the exact space dimension",
        )
    values = []
    max_coordinate_digits = 1
    for coordinate in coordinates:
        if not isinstance(coordinate, CanonicalRational):
            raise OperationDomainValidationError(
                location=("form", "coordinates"),
                code="modular_form.coordinate_type",
                message="coordinates must be canonical rational values",
            )
        numerator = getattr(coordinate, "num", None)
        denominator = getattr(coordinate, "den", None)
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
        ):
            raise OperationDomainValidationError(
                location=("form", "coordinates"),
                code="modular_form.coordinate_value",
                message="coordinate rationals must have a positive denominator",
            )
        digits = max(
            len(format_canonical_integer(abs(numerator))),
            len(format_canonical_integer(denominator)),
        )
        if digits > MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("form", "coordinates"),
                code="modular_form.coordinate_digit_bound",
                message=(
                    "basis coordinates are limited to "
                    f"{MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS} decimal digits"
                ),
            )
        value = Fraction(numerator, denominator)
        if (value.numerator, value.denominator) != (numerator, denominator):
            raise OperationDomainValidationError(
                location=("form", "coordinates"),
                code="modular_form.coordinate_value",
                message="coordinate rationals must be reduced and canonical",
            )
        max_coordinate_digits = max(max_coordinate_digits, digits)
        values.append(value)
    result_digits = (
        plan.dimension * max_coordinate_digits
        + plan.coefficient_digits
        + len(str(max(1, plan.dimension)))
        + 2
    )
    if check_expansion_growth and result_digits > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.coordinate_result_bound",
            message="coordinate combination exceeds the exact coefficient envelope",
        )
    if materialize_pari and plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        plan = _materialize_pari_basis(plan)
    return plan, tuple(values)


def _require_canonical_coordinate_space(
    form: ModularFormCoordinates,
    side: str,
) -> None:
    if not isinstance(form.space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=(side, "space"),
            code="modular_form.coordinates_space_type",
            message="both coordinate values must carry a canonical modular-form space",
        )


def _admitted_common_level(
    left_space: ModularFormSpace, right_space: ModularFormSpace
) -> int:
    """The exact common Gamma0 level, admitted within the space bound."""

    common_level = (
        left_space.level * right_space.level // gcd(left_space.level, right_space.level)
    )
    if common_level > MAX_MODULAR_FORM_LEVEL:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.equality_level_bound",
            message=(
                "the common Gamma0 level exceeds the modular-form level bound "
                f"{MAX_MODULAR_FORM_LEVEL}"
            ),
        )
    return common_level


def modular_form_coordinates_equal(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> bool:
    """Decide exact equality in a shared supported modular-form ambient space."""

    if not isinstance(left, ModularFormCoordinates) or not isinstance(
        right, ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.coordinates_type",
            message="both operands must be exact modular-form coordinate values",
        )
    for side, form in (("left", left), ("right", right)):
        _require_canonical_coordinate_space(form, side)
    if left.space == right.space:
        if left.space.coefficient_domain != "QQ":
            if left.space.character != "TRIVIAL":
                from jacobian.math.number_theory.modular_forms.character_coordinates import (
                    modular_character_coordinates_equal,
                )

                return modular_character_coordinates_equal(left, right)
            from jacobian.math.number_theory.modular_forms.field_coordinates import (
                modular_form_field_coordinates_equal,
            )

            return modular_form_field_coordinates_equal(left, right)
        plan_precision = (
            sturm_bound(left.space).bound + 1 if left.space.level > 4 else 1
        )
        plan = _admit_basis(left.space, plan_precision, materialize_pari=False)
        _, left_coordinates = _admit_coordinates(
            left,
            plan_precision,
            admitted_plan=plan,
            materialize_pari=False,
            check_expansion_growth=False,
        )
        _, right_coordinates = _admit_coordinates(
            right,
            plan_precision,
            admitted_plan=plan,
            materialize_pari=False,
            check_expansion_growth=False,
        )
        return left_coordinates == right_coordinates

    left_space = left.space
    right_space = right.space
    if (
        left_space.character != "TRIVIAL"
        or right_space.character != "TRIVIAL"
        or left_space.coefficient_domain != "QQ"
        or right_space.coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.equality_parent_unsupported",
            message=(
                "cross-space equality currently requires rational "
                "trivial-character forms"
            ),
        )
    if left_space.weight != right_space.weight:
        raise OperationDomainValidationError(
            location=("right", "space", "weight"),
            code="modular_form.equality_weight_mismatch",
            message="cross-space equality requires equal weights",
        )

    # Both forms embed into M_k(Gamma0(lcm(N1,N2))). The Sturm theorem there
    # makes equality of this finite prefix equivalent to equality of forms;
    # cusp forms embed in its ambient holomorphic space as well.
    common_level = _admitted_common_level(left_space, right_space)
    common_space = ModularFormSpace(
        level=common_level, weight=left_space.weight, kind="M"
    )
    precision = sturm_bound(common_space).bound + 1
    left_plan = _admit_basis(left_space, precision, materialize_pari=False)
    right_plan = _admit_basis(right_space, precision, materialize_pari=False)
    _, left_coordinates = _admit_coordinates(
        left, precision, admitted_plan=left_plan, materialize_pari=False
    )
    _, right_coordinates = _admit_coordinates(
        right, precision, admitted_plan=right_plan, materialize_pari=False
    )

    # Admit both basis materializations and the two exact linear combinations
    # together, before either PARI expansion. Rational sums of d products with
    # coordinate height C and basis coefficient height B have height at most
    # d(C+B)+digits(d)+2.
    combined_work = (
        left_plan.work
        + right_plan.work
        + precision * (left_plan.dimension + right_plan.dimension)
    )
    combined_basis_bytes = sum(
        plan.dimension * precision * (2 * plan.rref_digit_bound + 32)
        if plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else plan.dimension * precision * (plan.coefficient_digits + 8)
        for plan in (left_plan, right_plan)
    )
    if combined_work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.equality_work_bound",
            message="combined equality basis and coefficient work exceeds its envelope",
        )
    if combined_basis_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.equality_basis_output_bound",
            message="combined equality basis output exceeds its exact envelope",
        )
    expansion_digit_bounds = []
    for plan, coordinates in (
        (left_plan, left_coordinates),
        (right_plan, right_coordinates),
    ):
        coordinate_digits = max(
            (
                max(
                    len(format_canonical_integer(abs(value.numerator))),
                    len(format_canonical_integer(value.denominator)),
                )
                for value in coordinates
            ),
            default=1,
        )
        expansion_digit_bounds.append(
            plan.dimension * (coordinate_digits + plan.coefficient_digits)
            + len(str(max(1, plan.dimension)))
            + 2
        )
    max_expansion_digits = max(expansion_digit_bounds, default=1)
    combined_expansion_bytes = 2 * precision * (2 * max_expansion_digits + 32)
    if max_expansion_digits > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.equality_coefficient_growth",
            message="common Sturm prefix coefficient growth exceeds its exact envelope",
        )
    if combined_expansion_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.equality_output_bound",
            message="common Sturm comparison exceeds its exact output envelope",
        )

    request_checkpoint("before common-space equality basis materialization")
    if left_plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        left_plan = _materialize_pari_basis(left_plan)
    if right_plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        right_plan = _materialize_pari_basis(right_plan)
    left_basis = _basis_coefficients(left_plan)
    right_basis = _basis_coefficients(right_plan)
    left_expansion = tuple(
        sum(
            (
                scalar * basis[index]
                for scalar, basis in zip(left_coordinates, left_basis, strict=True)
            ),
            Fraction(0),
        )
        for index in range(precision)
    )
    right_expansion = tuple(
        sum(
            (
                scalar * basis[index]
                for scalar, basis in zip(right_coordinates, right_basis, strict=True)
            ),
            Fraction(0),
        )
        for index in range(precision)
    )
    request_checkpoint("after exact common-space equality comparison")
    return left_expansion == right_expansion


def modular_form_coordinates_transport(
    form: ModularFormCoordinates, target_space: ModularFormSpace
) -> ModularFormCoordinates:
    """Express a rational trivial-character form in a nested Gamma0 space.

    The exact q-prefix through the target Sturm bound determines coordinates
    uniquely in the target's canonical basis. The map is admitted as a whole
    before either backend basis is materialized.
    """

    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.transport_form_type",
            message="form must be an exact modular-form coordinate value",
        )
    if not isinstance(target_space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("target_space",),
            code="modular_form.transport_target_type",
            message="target_space must be an exact modular-form space value",
        )
    source_space = form.space
    if (
        source_space.character != "TRIVIAL"
        or target_space.character != "TRIVIAL"
        or source_space.coefficient_domain != "QQ"
        or target_space.coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("target_space",),
            code="modular_form.transport_parent_unsupported",
            message="coordinate transport currently supports QQ trivial-character spaces only",
        )
    if source_space.weight != target_space.weight:
        raise OperationDomainValidationError(
            location=("target_space", "weight"),
            code="modular_form.transport_weight_mismatch",
            message="source and target weights must agree",
        )
    if source_space.level <= 0 or target_space.level <= 0:
        raise OperationDomainValidationError(
            location=("target_space", "level"),
            code="modular_form.transport_level_value",
            message="source and target levels must be positive",
        )
    if target_space.level % source_space.level:
        raise OperationDomainValidationError(
            location=("target_space", "level"),
            code="modular_form.transport_level_not_nested",
            message="source Gamma0 level must divide the target level",
        )
    if source_space.kind == "M" and target_space.kind == "S":
        raise OperationDomainValidationError(
            location=("target_space", "kind"),
            code="modular_form.transport_kind_not_nested",
            message="the full holomorphic space does not embed into the cuspidal subspace",
        )

    target_precision = sturm_bound(target_space).bound + 1
    # Admitting the source at the target's determining precision also proves
    # the source representation can supply every target comparison term.
    source_plan = _admit_basis(source_space, target_precision, materialize_pari=False)
    target_plan = _admit_basis(target_space, target_precision, materialize_pari=False)
    _, source_coordinates = _admit_coordinates(
        form,
        target_precision,
        admitted_plan=source_plan,
        materialize_pari=False,
    )

    total_work = source_plan.work + target_plan.work
    solve_work = target_precision * (
        source_plan.dimension
        + target_plan.dimension * target_plan.dimension
        + source_plan.dimension * target_plan.dimension
    )
    total_work += solve_work
    if total_work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("target_space",),
            code="modular_form.transport_work_bound",
            message="combined source and target basis work exceeds the transport envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in source_coordinates
        ),
        default=1,
    )
    # A determinant expansion uses at most d! products of d entries. Include
    # source coefficients, caller coordinates, and target pivots in the same
    # integer-digit envelope for Cramer's-rule numerators and denominators.
    dimension = max(1, target_plan.dimension)
    pivot_digit_bound = max(
        target_plan.coefficient_digits,
        target_plan.rref_digit_bound,
        source_plan.coefficient_digits,
    )
    determinant_digit_bound = (
        dimension * (pivot_digit_bound + coordinate_digits + len(str(dimension)) + 2)
        + len(str(max(1, dimension))) * dimension
    )
    result_digit_bound = 2 * determinant_digit_bound + 4
    if result_digit_bound > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.transport_coefficient_growth",
            message="transport coordinate growth exceeds its exact digit envelope",
        )
    allocation_bytes = target_plan.dimension * (2 * result_digit_bound + 32) + 512
    if allocation_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("target_space",),
            code="modular_form.transport_output_bound",
            message="transport coordinates exceed the exact output-byte envelope",
        )

    if source_space == target_space:
        return form
    request_checkpoint("before modular-form transport basis materialization")
    source_plan = (
        _materialize_pari_basis(source_plan)
        if source_plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else source_plan
    )
    target_plan = (
        _materialize_pari_basis(target_plan)
        if target_plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else target_plan
    )
    source_basis = _basis_coefficients(source_plan)
    source_expansion = tuple(
        sum(
            (
                scalar * basis[index]
                for scalar, basis in zip(source_coordinates, source_basis, strict=True)
            ),
            Fraction(0),
        )
        for index in range(target_precision)
    )
    target_coordinates = _solve_sturm_coordinates(
        _basis_coefficients(target_plan), source_expansion
    )
    request_checkpoint("after exact modular-form transport solve")
    return ModularFormCoordinates(
        space=target_space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in target_coordinates
        ),
    )


def modular_form_coordinates_q_expansion(
    form: ModularFormCoordinates, precision: int
) -> ModularQExpansion:
    """Expand an exact form represented in one supported canonical basis."""

    # The coordinate value already identifies one global form, so a prefix
    # shorter than the Sturm-determining bound is admitted at the internal
    # determining precision and truncated to the requested order.
    plan, coordinates = _admit_coordinates(form, precision, allow_short_prefix=True)
    basis_vectors = _basis_coefficients(plan)
    output = []
    for coefficient_index in range(precision):
        request_checkpoint("during modular-form coordinate expansion")
        value = sum(
            (
                scalar * basis[coefficient_index]
                for scalar, basis in zip(coordinates, basis_vectors, strict=True)
            ),
            Fraction(0),
        )
        output.append(CanonicalRational(num=value.numerator, den=value.denominator))
    return ModularQExpansion.model_construct(
        space=form.space,
        weight=form.space.weight,
        basis_id=plan.basis_id,
        q_expansion=TruncatedSeries.model_construct(
            variable="q",
            truncation_order=precision,
            coefficients=tuple(output),
        ),
    )


def modular_form_coordinates_product(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> ModularFormCoordinates:
    """Multiply represented forms and recover coordinates in their product space.

    This bounded slice supports rational trivial-character forms in the
    integer-coefficient native bases through level four. The product's exact
    target is ``Gamma0(lcm(N_left, N_right))``, with added weight and cusp kind
    when either factor is cuspidal.
    """

    if not isinstance(left, ModularFormCoordinates) or not isinstance(
        right, ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.product_form_type",
            message="both factors must be exact modular-form coordinate values",
        )
    for location, form in (("left", left), ("right", right)):
        if form.space.character != "TRIVIAL" or form.space.coefficient_domain != "QQ":
            raise OperationDomainValidationError(
                location=(location, "space"),
                code="modular_form.product_parent_unsupported",
                message="form products currently require trivial-character QQ spaces",
            )
    target_level = lcm(left.space.level, right.space.level)
    if target_level > 4:
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.product_target_level_unsupported",
            message="form products currently require a target Gamma0 level at most 4",
        )
    target_space = ModularFormSpace(
        level=target_level,
        weight=left.space.weight + right.space.weight,
        kind="S" if "S" in (left.space.kind, right.space.kind) else "M",
    )
    precision = sturm_bound(target_space).bound + 1

    # All source and target plans, coordinates, and downstream growth are
    # admitted before any basis coefficient is materialized.
    plans = tuple(
        _admit_basis(form.space, precision, materialize_pari=False)
        for form in (left, right)
    )
    target_plan = _admit_basis(target_space, precision, materialize_pari=False)
    if any(plan.basis_id == PARI_STURM_RREF_BASIS_ID for plan in (*plans, target_plan)):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.product_basis_unsupported",
            message="form products currently require the native integer-coefficient basis family",
        )
    coordinate_vectors = tuple(
        _admit_coordinates(
            form,
            precision,
            admitted_plan=plan,
            materialize_pari=False,
        )[1]
        for form, plan in zip((left, right), plans, strict=True)
    )

    def max_digits(values: tuple[Fraction, ...]) -> int:
        return max(
            (
                max(
                    len(format_canonical_integer(abs(value.numerator))),
                    len(format_canonical_integer(value.denominator)),
                )
                for value in values
            ),
            default=1,
        )

    expansion_digit_bounds = tuple(
        plan.dimension * (max_digits(coordinates) + plan.coefficient_digits)
        + len(str(max(1, plan.dimension)))
        + 2
        for plan, coordinates in zip(plans, coordinate_vectors, strict=True)
    )
    convolution_digits = sum(expansion_digit_bounds) + len(str(max(1, precision))) + 2
    target_dimension = target_plan.dimension
    result_digits = (
        target_dimension
        * (
            convolution_digits
            + target_plan.coefficient_digits
            + target_dimension.bit_length()
            + 3
        )
        + len(str(max(1, target_dimension)))
        + 2
    )
    if result_digits > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("coordinates",),
            code="modular_form.product_coefficient_growth",
            message="exact product coordinates exceed the admitted rational digit bound",
        )

    operation_count = (
        sum(plan.work for plan in (*plans, target_plan))
        + precision * (plans[0].dimension + plans[1].dimension)
        + precision * precision
        + precision * target_dimension * target_dimension
    )
    arithmetic_digits = max(convolution_digits, result_digits)
    work = operation_count * arithmetic_digits * arithmetic_digits * 16
    if work > MAX_MODULAR_FORM_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.product_work_bound",
            message="basis expansion, product, and Sturm reconstruction exceed the work bound",
        )
    basis_bytes = sum(
        plan.dimension * precision * (2 * plan.coefficient_digits + 32)
        for plan in (*plans, target_plan)
    ) + precision * (2 * convolution_digits + 32)
    if basis_bytes > MAX_MODULAR_FORM_PRODUCT_INTERMEDIATE_BYTES:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.product_intermediate_bound",
            message="combined product basis materialization exceeds its memory envelope",
        )
    allocation_bytes = (
        len(encode_strict_json(target_space.model_dump(mode="json")))
        + target_dimension * (2 * result_digits + 64)
        + 1024
    )
    if allocation_bytes > MAX_LEVEL_ONE_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.product_output_bound",
            message="exact modular-form product coordinates exceed the output envelope",
        )

    request_checkpoint("before modular-form product basis materialization")
    basis_vectors = tuple(_basis_coefficients(plan) for plan in plans)
    expansions = tuple(
        tuple(
            sum(
                (
                    scalar * basis[index]
                    for scalar, basis in zip(coordinates, vectors, strict=True)
                ),
                Fraction(0),
            )
            for index in range(precision)
        )
        for coordinates, vectors in zip(coordinate_vectors, basis_vectors, strict=True)
    )
    left_expansion, right_expansion = expansions
    product_expansion = tuple(
        sum(
            (
                left_expansion[index] * right_expansion[degree - index]
                for index in range(degree + 1)
            ),
            Fraction(0),
        )
        for degree in range(precision)
    )
    result_coordinates = _solve_sturm_coordinates(
        _basis_coefficients(target_plan), product_expansion
    )
    request_checkpoint("after exact modular-form product reconstruction")
    return ModularFormCoordinates(
        space=target_space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in result_coordinates
        ),
    )


def _apply_coordinate_operator(
    form: ModularFormCoordinates,
    *,
    operator: str,
    index: int,
) -> ModularFormCoordinates:
    """Apply one supported operator and recover its exact basis coordinates."""

    plan, coordinates = _admit_coordinates(form, 1)
    chi_minus4 = _is_gamma0_four_chi4(form.space)
    if operator == "hecke":
        supported = (
            form.space.level == 1
            or (form.space.level == 2 and index % 2 == 1)
            or (form.space.level == 3 and index % 3 != 0)
            or (
                form.space.level == 4
                and form.space.character == "TRIVIAL"
                and index % 2 == 1
            )
            or (chi_minus4 and index % 2 == 1)
        )
        if not supported:
            raise OperationDomainValidationError(
                location=("form", "space", "level"),
                code="modular_form.coordinates_hecke_unsupported_level",
                message=(
                    "T_n coordinates require level one, or a supported "
                    "positive-level space with n coprime to the level"
                ),
            )
    elif operator == "u2":
        supported_u2 = (
            (form.space.level == 2 and plan.basis_id == GAMMA0_TWO_BASIS_ID)
            or (
                form.space.level == 4
                and form.space.character == "TRIVIAL"
                and plan.basis_id == GAMMA0_FOUR_BASIS_ID
            )
            or (chi_minus4 and form.space.level == 4)
        )
        if not supported_u2:
            raise OperationDomainValidationError(
                location=("form", "space", "level"),
                code="modular_form.coordinates_u2_unsupported_level",
                message=(
                    "U_2 coordinates require a represented M_k(Gamma0(2)) "
                    "or M_k(Gamma0(4), chi_-4) form"
                ),
            )
    else:
        raise RuntimeError("unknown internal modular-form operator")

    level_index = {1: 1, 2: 3, 3: 4, 4: 6}[form.space.level]
    bound = (form.space.weight * level_index) // 12
    precision = bound + 1
    scale = index if operator == "hecke" else 2
    source_order = scale * bound + 1
    if source_order > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.coordinates_operator_source_bound",
            message="operator source order exceeds the bounded q-prefix envelope",
        )
    if source_order > MAX_LEVEL_ONE_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.coordinates_operator_basis_precision",
            message=(
                "operator source order exceeds the current deterministic basis "
                "coefficient envelope"
            ),
        )
    transform_work = (
        max(1, plan.dimension) * precision * index
        + 2 * precision * plan.dimension * plan.dimension
    )
    if transform_work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.coordinates_operator_work_bound",
            message="coordinate operator work exceeds the bounded exact envelope",
        )

    source_digits = (
        _digit_bound(form.space.weight, source_order)
        if form.space.level == 1
        else _gamma0_four_chi_digit_bound(form.space.weight, source_order)
        if chi_minus4
        else _gamma0_three_digit_bound(form.space.weight, source_order)
        if plan.basis_id == GAMMA0_THREE_BASIS_ID
        else _gamma0_two_digit_bound(form.space.weight, source_order)
    )
    term_count = 2 * isqrt(index) + 1 if operator == "hecke" else 1
    power_digits = (
        0
        if operator != "hecke" or form.space.weight <= 1 or index <= 1
        else ((form.space.weight - 1) * max(1, index.bit_length()) * 30_103) // 100_000
        + 2
    )
    if operator == "u2":
        image_digits = source_digits
    elif form.space.weight == 0:
        denominator_digits = source_digits + len(str(index))
        image_digits = max(
            term_count * denominator_digits,
            source_digits
            + power_digits
            + (term_count - 1) * denominator_digits
            + len(str(term_count))
            + 1,
        )
    else:
        image_digits = source_digits + power_digits + len(str(term_count)) + 1
    coordinate_digits = (
        max(
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        )
        if coordinates
        else 1
    )
    basis_digits = (
        _digit_bound(form.space.weight, precision)
        if form.space.level == 1
        else _gamma0_four_chi_digit_bound(form.space.weight, precision)
        if chi_minus4
        else _gamma0_three_digit_bound(form.space.weight, precision)
        if plan.basis_id == GAMMA0_THREE_BASIS_ID
        else _gamma0_two_digit_bound(form.space.weight, precision)
    )
    determinant_factor_digits = plan.dimension * (plan.dimension.bit_length() + 1)
    result_digits = (
        image_digits
        + plan.dimension * basis_digits
        + plan.dimension * coordinate_digits
        + determinant_factor_digits
        + plan.dimension.bit_length()
    )
    if result_digits > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.coordinates_operator_growth_bound",
            message="operator coordinate growth exceeds the bounded exact envelope",
        )

    source_plan = _admit_basis(form.space, source_order)
    basis_vectors = _basis_coefficients(source_plan)
    if operator == "hecke":
        images = tuple(
            tuple(
                _hecke_coefficient(
                    vector,
                    index,
                    m,
                    form.space.weight,
                    chi_minus4=chi_minus4,
                )
                for m in range(precision)
            )
            for vector in basis_vectors
        )
    else:
        images = tuple(
            tuple(vector[2 * m] for m in range(precision)) for vector in basis_vectors
        )
    action_columns = tuple(
        _solve_sturm_coordinates(basis_vectors, image) for image in images
    )
    result_coordinates = tuple(
        sum(
            (
                coordinates[column] * action_columns[column][row]
                for column in range(plan.dimension)
            ),
            Fraction(0),
        )
        for row in range(plan.dimension)
    )
    if any(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for value in result_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.coordinates_operator_result_bound",
            message="exact operator coordinates exceed the bounded result envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=form.space,
        basis_id=form.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in result_coordinates
        ),
    )


def modular_form_coordinates_hecke(
    form: ModularFormCoordinates, index: int
) -> ModularFormCoordinates:
    """Apply T_n to exact coordinates in a reviewed coprime-level space."""

    if type(index) is not int or index < 1 or index > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.coordinates_hecke_index_bound",
            message=(
                "coordinate Hecke index must lie in the bounded envelope "
                f"[1, {MAX_Q_TRANSFORM_SOURCE_ORDER}]"
            ),
        )
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if form.space.level == 2 and index % 2 == 0:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.coordinates_hecke_not_coprime",
            message="T_n on Gamma0(2) is supported only when n is odd",
        )
    if form.space.level == 3 and index % 3 == 0:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.coordinates_hecke_not_coprime",
            message="T_n on Gamma0(3) is supported only when n is coprime to 3",
        )
    if form.space.level == 4 and form.space.character == "TRIVIAL" and index % 2 == 0:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.coordinates_hecke_not_coprime",
            message="T_n on Gamma0(4) is supported only when n is odd",
        )
    if (
        form.space.level == 4
        and isinstance(form.space.character, DirichletCharacter)
        and form.space.character.group.modulus == 4
        and form.space.character.group.generator_orders == (2,)
        and form.space.character.coordinates == (1,)
        and index % 2 == 0
    ):
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.coordinates_hecke_not_coprime",
            message="T_n on Gamma0(4) with chi_-4 is supported only when n is odd",
        )
    return _apply_coordinate_operator(form, operator="hecke", index=index)


def modular_form_coordinates_atkin_lehner(
    form: ModularFormCoordinates, divisor: int
) -> ModularFormCoordinates:
    """Apply exact ``|_k W_Q`` on a supported rational Gamma0 space.

    PARI computes the transformation of the exact modular form represented by
    its q-Sturm RREF coordinates. The returned prefix is solved back into the
    same Jacobian basis through the exact Sturm determining precision.
    """
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    space = form.space
    if (
        space.group != "GAMMA0"
        or space.character != "TRIVIAL"
        or space.coefficient_domain != "QQ"
        or type(space.level) is not int
        or not 1 <= space.level <= MAX_PARI_BASIS_LEVEL
        or type(space.weight) is not int
        or space.weight % 2
    ):
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.atkin_lehner_parent_unsupported",
            message=(
                "Atkin-Lehner transforms currently support even-weight QQ spaces "
                "with trivial character on Gamma0(N)"
            ),
        )
    if type(divisor) is not int or divisor < 1 or divisor > space.level:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="modular_form.atkin_lehner_divisor",
            message="Q must be a positive divisor of the represented level",
        )
    if space.level % divisor or gcd(divisor, space.level // divisor) != 1:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="modular_form.atkin_lehner_exact_divisor",
            message="Q must be an exact divisor of the represented level",
        )
    bound = sturm_bound(space).bound
    precision = bound + 1
    plan = _admit_basis(space, precision, materialize_pari=False)
    plan, coordinates = _admit_coordinates(
        form, precision, admitted_plan=plan, materialize_pari=False
    )
    if divisor == 1:
        return ModularFormCoordinates.model_construct(
            space=space,
            basis_id=plan.basis_id,
            coordinates=tuple(
                CanonicalRational(num=value.numerator, den=value.denominator)
                for value in coordinates
            ),
        )
    if plan.dimension == 0:
        return ModularFormCoordinates.model_construct(
            space=space, basis_id=plan.basis_id, coordinates=()
        )

    dimension = plan.dimension
    dimension_data = space_dimension(space)
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        ),
        default=1,
    )
    # For D=T^-1 A T, the worker caps every returned Jacobian-basis entry
    # of D at H_D=512 decimal digits. Each output coordinate is a sum of d
    # products D_ij*x_j; multiplying rational heights adds and a sum of d
    # fractions is bounded by the sum of their denominator heights.
    output_digit_bound = dimension * (
        coordinate_digits + MAX_ATKIN_LEHNER_MATRIX_ENTRY_DIGITS
    ) + len(str(max(1, dimension)))
    if output_digit_bound > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.atkin_lehner_coordinate_growth_bound",
            message=(
                "worst-case Atkin-Lehner coordinates exceed the exact result "
                "digit envelope under the admitted matrix-height cap"
            ),
        )

    matrix_allocation_bytes = (
        dimension * dimension * (2 * MAX_ATKIN_LEHNER_MATRIX_ENTRY_DIGITS + 32) + 512
    )
    coordinate_allocation_bytes = dimension * (2 * output_digit_bound + 32) + 512
    basis_input_bytes = (
        dimension * precision * (2 * max(1, plan.coefficient_digits) + 32) + 512
    )
    allocation_bytes = matrix_allocation_bytes + coordinate_allocation_bytes
    if basis_input_bytes + matrix_allocation_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("form", "space"),
            code="modular_form.atkin_lehner_backend_output_bound",
            message="Atkin-Lehner basis matrix transport exceeds its byte envelope",
        )

    entry_height = MAX_ATKIN_LEHNER_MATRIX_ENTRY_DIGITS
    # Clearing a common denominator for a d-by-d rational matrix costs at
    # most d^2*H digits. Cofactor determinants therefore cost at most d^3*H;
    # the extra factor two covers normalized Gaussian-elimination ratios.
    inverse_height_bound = (
        2 * dimension**3 * entry_height + len(str(max(1, factorial(dimension)))) + 4
    )
    first_product_bound = (
        dimension * (inverse_height_bound + entry_height)
        + len(str(max(1, dimension)))
        + 2
    )
    internal_digit_bound = (
        dimension * (first_product_bound + entry_height)
        + len(str(max(1, dimension)))
        + 2
    )
    internal_bytes_bound = 8 * dimension * dimension * (internal_digit_bound + 16) // 2
    if (
        internal_digit_bound > MAX_ATKIN_LEHNER_INTERNAL_DIGITS
        or internal_bytes_bound > MAX_ATKIN_LEHNER_INTERNAL_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("form", "space"),
            code="modular_form.atkin_lehner_intermediate_bound",
            message="Atkin-Lehner exact matrix intermediates exceed their envelope",
        )

    backend_work = (
        dimension * precision * max(1, dimension_data.index) * max(1, space.weight)
        + dimension * dimension * precision
        + dimension * dimension * dimension
    )
    total_work = plan.work + backend_work
    if total_work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("form", "space"),
            code="modular_form.atkin_lehner_work_bound",
            message="Atkin-Lehner work exceeds its admitted envelope",
        )
    if allocation_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("form", "space"),
            code="modular_form.atkin_lehner_output_bound",
            message="Atkin-Lehner output exceeds its admitted envelope",
        )
    if plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        plan = _materialize_pari_basis(plan)
    basis_vectors = _basis_coefficients(plan)
    matrix = pari_gamma0_atkin_matrix(
        space,
        divisor,
        precision,
        basis_vectors,
        admitted_work=total_work,
        admitted_allocation_bytes=basis_input_bytes + matrix_allocation_bytes,
    )
    output_coordinates = tuple(
        sum(
            (matrix[row][column] * coordinates[column] for column in range(dimension)),
            Fraction(0),
        )
        for row in range(dimension)
    )
    if any(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for value in output_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.atkin_lehner_coordinate_bound",
            message="Atkin-Lehner coordinates exceed the admitted exact digit envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=space,
        basis_id=plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in output_coordinates
        ),
    )


def modular_form_hecke_matrix(
    space: ModularFormSpace, index: int
) -> ModularFormHeckeMatrix:
    """Return the exact T_n matrix in the canonical basis of an admitted space.

    Matrix rows are output basis coefficients and columns are input basis
    vectors. One basis expansion at the required source order supplies every
    column, and each column is reconstructed and checked through the exact
    Sturm bound.
    """

    if type(index) is not int or not 1 <= index <= MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.hecke_matrix_index_bound",
            message="Hecke matrix index is outside the exact admitted envelope",
        )
    if not isinstance(space, ModularFormSpace):
        _admit_basis(space, 1)
        raise RuntimeError("unreachable invalid modular-form space")
    if type(space.weight) is not int or space.weight < 0:
        _admit_basis(space, 1)
        raise RuntimeError("unreachable invalid modular-form weight")
    chi_minus4 = _is_gamma0_four_chi4(space)
    if space.level > 4:
        # The PARI-backed basis path admits rational trivial-character
        # Gamma0 spaces through MAX_PARI_BASIS_LEVEL. Hecke T_n preserves
        # these spaces when (n, N) = 1.
        if space.character != "TRIVIAL" or space.coefficient_domain != "QQ":
            _admit_basis(space, 1)
        supported = gcd(index, space.level) == 1
    else:
        supported = (
            space.level == 1
            or (space.level == 2 and index % 2 == 1)
            or (space.level == 3 and index % 3 != 0)
            or (space.level == 4 and space.character == "TRIVIAL" and index % 2 == 1)
            or (chi_minus4 and index % 2 == 1)
        )
    if not supported:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.hecke_matrix_not_coprime",
            message="Hecke matrices are supported when the index is coprime to the level",
        )
    bound = sturm_bound(space).bound
    precision = bound + 1
    source_order = index * bound + 1
    if source_order > min(MAX_Q_TRANSFORM_SOURCE_ORDER, MAX_LEVEL_ONE_BASIS_PRECISION):
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.hecke_matrix_source_bound",
            message="Hecke matrix requires basis coefficients beyond the admitted source order",
        )
    plan = _admit_basis(space, source_order)
    term_count = 2 * isqrt(index) + 1
    work = (
        plan.dimension * precision * index
        + 2 * precision * plan.dimension * plan.dimension
        + plan.work
    )
    if work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.hecke_matrix_work_bound",
            message="Hecke matrix computation exceeds the exact work envelope",
        )
    digit_bound = (
        plan.coefficient_digits
        if plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else _digit_bound(space.weight, source_order)
        if space.level == 1
        else _gamma0_four_chi_digit_bound(space.weight, source_order)
        if chi_minus4
        else _gamma0_three_digit_bound(space.weight, source_order)
        if plan.basis_id == GAMMA0_THREE_BASIS_ID
        else _gamma0_two_digit_bound(space.weight, source_order)
    )
    power_digits = (
        0
        if space.weight <= 1 or index <= 1
        else ((space.weight - 1) * max(1, index.bit_length()) * 30_103) // 100_000 + 2
    )
    image_digits = (
        max(
            term_count * (digit_bound + len(str(index))),
            digit_bound
            + power_digits
            + (term_count - 1) * (digit_bound + len(str(index)))
            + len(str(term_count))
            + 1,
        )
        if space.weight == 0
        else digit_bound + power_digits + len(str(term_count)) + 1
    )
    determinant_digits = plan.dimension * (plan.dimension.bit_length() + 1)
    result_digits = (
        image_digits
        + plan.dimension * digit_bound
        + determinant_digits
        + plan.dimension.bit_length()
    )
    matrix_bytes = plan.dimension * plan.dimension * (2 * result_digits + 32)
    if matrix_bytes > MAX_HECKE_MATRIX_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.hecke_matrix_output_bound",
            message="Hecke matrix exact entries exceed the bounded output envelope",
        )

    basis_vectors = _basis_coefficients(plan)
    columns = []
    for vector in basis_vectors:
        image = tuple(
            _hecke_coefficient(
                vector,
                index,
                m,
                space.weight,
                chi_minus4=chi_minus4,
            )
            for m in range(precision)
        )
        columns.append(_solve_sturm_coordinates(basis_vectors, image))
    entries = tuple(
        tuple(
            CanonicalRational(
                num=columns[column][row].numerator,
                den=columns[column][row].denominator,
            )
            for column in range(plan.dimension)
        )
        for row in range(plan.dimension)
    )
    labels = _basis_labels(plan)
    return ModularFormHeckeMatrix.model_construct(
        space=space,
        basis_id=plan.basis_id,
        index=index,
        row_labels=labels,
        column_labels=labels,
        entries=entries,
    )


def _invert_frame_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Invert one admitted rational frame by exact Gauss-Jordan elimination."""
    dimension = len(matrix)
    rows = [
        [
            *row,
            *(Fraction(int(column == row_index)) for column in range(dimension)),
        ]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        request_checkpoint("during framed Hecke matrix conjugation")
        pivot = next(
            (row for row in range(column, dimension) if rows[row][column]), None
        )
        if pivot is None:
            raise OperationDomainValidationError(
                location=("frame", "entries"),
                code="modular_form.frame_singular",
                message="change-of-basis matrix must be invertible",
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for row in range(dimension):
            if row == column or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(rows[row], rows[column], strict=True)
            ]
    return tuple(tuple(row[dimension:]) for row in rows)


def modular_form_hecke_matrix_in_frame(
    frame: ModularFormChangeOfBasisFrame, index: int
) -> ModularFormFramedHeckeMatrix:
    """Conjugate an admitted canonical Hecke matrix into one exact frame."""
    if type(index) is not int or not 1 <= index <= MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.hecke_matrix_index_bound",
            message="Hecke matrix index is outside the exact admitted envelope",
        )
    plan, change = _frame_admission(frame)
    dimension = plan.dimension
    zero = (Fraction(0),) * dimension
    _admit_change_of_basis_arithmetic(dimension, change, zero)
    inverse = _invert_frame_matrix(change)

    canonical = modular_form_hecke_matrix(frame.space, index)
    if (
        canonical.space != frame.space
        or canonical.basis_id != frame.source_basis_id
        or canonical.row_labels != frame.source_labels
        or canonical.column_labels != frame.source_labels
    ):
        raise RuntimeError("canonical Hecke matrix does not match its admitted basis")
    operator = tuple(
        tuple(entry.as_fraction() for entry in row) for row in canonical.entries
    )
    max_change_digits = max(
        (
            max(len(str(abs(value.numerator))), len(str(value.denominator)))
            for row in (*change, *inverse)
            for value in row
        ),
        default=1,
    )
    max_operator_digits = max(
        (
            max(
                len(str(abs(value.numerator))),
                len(str(value.denominator)),
            )
            for row in operator
            for value in row
        ),
        default=1,
    )
    sum_digits = len(str(max(dimension, 1))) + 1
    intermediate_digits = (
        dimension * (max_operator_digits + max_change_digits + 2) + sum_digits
    )
    result_digits = (
        dimension * (max_change_digits + intermediate_digits + 2) + sum_digits
    )
    transform_work = 3 * dimension**3
    frame_bytes = len(encode_strict_json(frame.model_dump(mode="json")))
    allocation_bytes = (
        frame_bytes
        + dimension * dimension * (2 * result_digits + 48)
        + dimension * 768
        + 512
    )
    if (
        transform_work > 100_000
        or result_digits > MAX_COORDINATE_RESULT_DIGITS
        or allocation_bytes > MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("frame",),
            code="modular_form.framed_hecke_matrix_bound",
            message="exact Hecke matrix conjugation exceeds its work, growth, or output envelope",
        )

    framed_entries: list[tuple[Fraction, ...]] = []
    for column in range(dimension):
        request_checkpoint("during framed Hecke matrix conjugation")
        transformed = tuple(
            sum(
                (
                    operator[row][inner] * change[inner][column]
                    for inner in range(dimension)
                ),
                Fraction(0),
            )
            for row in range(dimension)
        )
        framed_column = tuple(
            sum(
                (
                    inverse[row][inner] * transformed[inner]
                    for inner in range(dimension)
                ),
                Fraction(0),
            )
            for row in range(dimension)
        )
        framed_entries.append(framed_column)
    entries = tuple(
        tuple(
            CanonicalRational.from_fraction(framed_entries[column][row])
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    return ModularFormFramedHeckeMatrix(
        frame=frame,
        index=index,
        row_labels=frame.labels,
        column_labels=frame.labels,
        entries=entries,
    )


def modular_form_coordinates_u2(
    form: ModularFormCoordinates,
) -> ModularFormCoordinates:
    """Apply U_2 to exact coordinates in M_k(Gamma0(2))."""

    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    return _apply_coordinate_operator(form, operator="u2", index=2)


def modular_form_coordinates_u_prime(
    form: ModularFormCoordinates, prime: int
) -> ModularFormCoordinates:
    """Apply U_p on exact Gamma0(N) coordinates when p divides N."""

    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if type(prime) is not int or not 2 <= prime <= MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.u_prime_index",
            message=f"prime must be an integer in [2, {MAX_GAMMA0_OPERATION_LEVEL}]",
        )
    if not _is_prime_level(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.u_prime_not_prime",
            message="U_p requires a prime index",
        )
    space = form.space
    if (
        space.group != "GAMMA0"
        or space.kind not in ("M", "S")
        or space.character != "TRIVIAL"
        or space.coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.u_prime_parent",
            message=(
                "U_p coordinates currently support QQ trivial-character M or S "
                "spaces on Gamma0(N)"
            ),
        )
    if space.level % prime:
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.u_prime_level",
            message="U_p is an endomorphism of this Gamma0(N) space only when p divides N",
        )

    bound = sturm_bound(space).bound
    target_precision = bound + 1
    source_precision = prime * bound + 1
    target_plan = _admit_basis(space, target_precision, materialize_pari=False)
    source_plan = _admit_basis(space, source_precision, materialize_pari=False)
    source_plan, coordinates = _admit_coordinates(
        form,
        source_precision,
        admitted_plan=source_plan,
        materialize_pari=False,
        check_expansion_growth=False,
    )
    work = (
        source_plan.work
        + target_plan.work
        + source_plan.dimension * source_precision
        + target_plan.dimension * target_plan.dimension * target_precision
        + target_plan.dimension * source_plan.dimension
    )
    if work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.u_prime_work_bound",
            message="U_p exact coordinate reconstruction exceeds its work envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        ),
        default=1,
    )
    image_digits = (
        source_plan.dimension * coordinate_digits
        + source_plan.coefficient_digits
        + len(str(max(1, source_plan.dimension)))
        + 2
    )
    result_digits = image_digits + target_plan.dimension * (
        target_plan.coefficient_digits + target_plan.dimension.bit_length() + 2
    )
    allocation_bytes = (
        len(encode_strict_json(space.model_dump(mode="json")))
        + target_plan.dimension * (2 * result_digits + 64)
        + 512
    )
    if (
        result_digits > MAX_COORDINATE_RESULT_DIGITS
        or allocation_bytes > MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.u_prime_growth_bound",
            message="U_p rational growth or output exceeds its exact envelope",
        )

    if source_plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        source_plan = _materialize_pari_basis(source_plan)
        target_plan = _materialize_pari_basis(target_plan)
    source_basis = _basis_coefficients(source_plan)
    source_coefficients = tuple(
        sum(
            (
                scalar * vector[n]
                for scalar, vector in zip(coordinates, source_basis, strict=True)
            ),
            Fraction(0),
        )
        for n in range(source_precision)
    )
    image = tuple(source_coefficients[prime * n] for n in range(target_precision))
    target_basis = _basis_coefficients(target_plan)
    result_coordinates = _solve_sturm_coordinates(target_basis, image)
    if any(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for value in result_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.u_prime_result_bound",
            message="exact U_p coordinates exceed the result digit envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in result_coordinates
        ),
    )


def modular_form_coordinates_v2(
    form: ModularFormCoordinates,
) -> ModularFormCoordinates:
    """Apply V_2 from level one to level two or level two to level four."""

    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if form.space.level == 1 and form.space.kind == "M":
        target_level = 2
    elif form.space.level == 2 and form.space.kind == "M":
        target_level = 4
    else:
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.coordinates_v2_source",
            message=(
                "V_2 coordinates currently require M_k(SL2Z) or M_k(Gamma0(2)) forms"
            ),
        )

    weight = form.space.weight
    target_index = 3 if target_level == 2 else 6
    sturm_bound = (target_index * weight) // 12
    target_space = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=target_level,
        weight=weight,
        kind="M",
        character="TRIVIAL",
        coefficient_domain="QQ",
    )
    target_plan = _admit_basis(target_space, sturm_bound + 1)
    source_precision = sturm_bound // 2 + 1
    source_plan, coordinates = _admit_coordinates(form, source_precision)
    work = (
        target_plan.work
        + source_plan.work
        + source_plan.dimension * source_precision
        + target_plan.dimension * target_plan.dimension * (sturm_bound + 1)
        + target_plan.dimension * source_plan.dimension
    )
    if work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.coordinates_v2_work_bound",
            message="V_2 coordinate reconstruction exceeds the exact work envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        ),
        default=1,
    )
    image_digits = (
        source_plan.dimension * coordinate_digits
        + source_plan.coefficient_digits
        + len(str(max(1, source_plan.dimension)))
        + 2
    )
    result_digits = image_digits + target_plan.dimension * (
        target_plan.coefficient_digits + target_plan.dimension.bit_length() + 2
    )
    if result_digits > MAX_COORDINATE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.coordinates_v2_growth_bound",
            message="V_2 coordinate growth exceeds the exact result envelope",
        )

    source_basis = _basis_coefficients(source_plan)
    source_coefficients = tuple(
        sum(
            (
                scalar * vector[n]
                for scalar, vector in zip(coordinates, source_basis, strict=True)
            ),
            Fraction(0),
        )
        for n in range(source_precision)
    )
    target_basis = _basis_coefficients(target_plan)
    image = tuple(
        source_coefficients[m // 2] if m % 2 == 0 else Fraction(0)
        for m in range(sturm_bound + 1)
    )
    result_coordinates = (
        _solve_sturm_coordinates(target_basis, image)
        if target_level == 2
        else _solve_gamma0_four_coordinates(target_basis, image)
    )
    if any(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for value in result_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.coordinates_v2_result_bound",
            message="exact V_2 coordinates exceed the result envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=target_space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in result_coordinates
        ),
    )


def modular_form_coordinates_v3(
    form: ModularFormCoordinates,
) -> ModularFormCoordinates:
    """Apply the level-one degeneracy V_3 into M_k(Gamma0(3))."""

    target_level = 3
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if form.space.level != 1 or form.space.kind != "M":
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.coordinates_v3_source",
            message="V_3 coordinates currently require level-one M_k forms",
        )

    weight = form.space.weight
    sturm_bound = (4 * weight) // 12
    target_space = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=target_level,
        weight=weight,
        kind="M",
        character="TRIVIAL",
        coefficient_domain="QQ",
    )
    target_plan = _admit_basis(target_space, sturm_bound + 1)
    source_precision = sturm_bound // 3 + 1
    source_plan, coordinates = _admit_coordinates(form, source_precision)
    work = (
        target_plan.work
        + source_plan.work
        + source_plan.dimension * source_precision
        + target_plan.dimension * target_plan.dimension * (sturm_bound + 1)
        + target_plan.dimension * source_plan.dimension
    )
    if work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.coordinates_v3_work_bound",
            message="V_3 coordinate reconstruction exceeds the exact work envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        ),
        default=1,
    )
    image_digits = (
        source_plan.dimension * coordinate_digits
        + source_plan.coefficient_digits
        + len(str(max(1, source_plan.dimension)))
        + 2
    )
    result_digits = image_digits + target_plan.dimension * (
        target_plan.coefficient_digits + target_plan.dimension.bit_length() + 2
    )
    allocation_bytes = (
        len(encode_strict_json(target_space.model_dump(mode="json")))
        + target_plan.dimension * (2 * result_digits + 64)
        + 512
    )
    if (
        result_digits > MAX_COORDINATE_RESULT_DIGITS
        or allocation_bytes > MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.coordinates_v3_result_bound",
            message="V_3 coordinate growth or output exceeds the exact result envelope",
        )

    source_basis = _basis_coefficients(source_plan)
    source_coefficients = tuple(
        sum(
            (
                scalar * vector[n]
                for scalar, vector in zip(coordinates, source_basis, strict=True)
            ),
            Fraction(0),
        )
        for n in range(source_precision)
    )
    target_basis = _basis_coefficients(target_plan)
    image = tuple(
        source_coefficients[m // 3] if m % 3 == 0 else Fraction(0)
        for m in range(sturm_bound + 1)
    )
    result_coordinates = _solve_sturm_coordinates(target_basis, image)
    if any(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for value in result_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.coordinates_v3_result_bound",
            message="exact V_3 coordinates exceed the result envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=target_space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in result_coordinates
        ),
    )


def modular_form_coordinates_v_degeneracy(
    form: ModularFormCoordinates, d: int
) -> ModularFormCoordinates:
    """Apply V_d(f)(q)=f(q^d) in the exact Gamma0(Md) coordinate basis."""
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="form must be an exact modular-form coordinate value",
        )
    if type(d) is not int or not 1 <= d <= MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("d",),
            code="modular_form.v_degeneracy_index_bound",
            message=f"V_d requires d in [1, {MAX_GAMMA0_OPERATION_LEVEL}]",
        )
    source_space = form.space
    if not isinstance(source_space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.v_degeneracy_source_space",
            message="V_d source must retain its exact modular-form parent",
        )
    if (
        source_space.group != "GAMMA0"
        or source_space.character != "TRIVIAL"
        or source_space.coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.v_degeneracy_source_parent",
            message="V_d coordinates currently support Gamma0 spaces over QQ with trivial character",
        )
    if (
        type(source_space.level) is not int
        or source_space.level * d > MAX_PARI_BASIS_LEVEL
    ):
        raise OperationResourceAdmissionError(
            location=("d",),
            code="modular_form.v_degeneracy_target_level_bound",
            message="V_d target level exceeds the admitted modular-form basis envelope",
        )
    target_space = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=source_space.level * d,
        weight=source_space.weight,
        kind=source_space.kind,
        character="TRIVIAL",
        coefficient_domain="QQ",
    )
    target_bound = sturm_bound(target_space).bound
    source_precision = target_bound // d + 1

    # Admit both exact basis plans and the supplied coordinates before either
    # backend materializes q-expansions.
    source_plan = _admit_basis(source_space, source_precision, materialize_pari=False)
    target_plan = _admit_basis(target_space, target_bound + 1, materialize_pari=False)
    _, coordinates = _admit_coordinates(
        form,
        source_precision,
        admitted_plan=source_plan,
        materialize_pari=False,
        check_expansion_growth=False,
    )
    if d == 1:
        # V_1 is the identity operator on Gamma0(N): the validated form
        # already carries its exact coordinates in the unchanged basis.
        return form
    work = (
        source_plan.work
        + target_plan.work
        + source_plan.dimension * source_precision
        + target_plan.dimension * target_plan.dimension * (target_bound + 1)
        + target_plan.dimension * source_plan.dimension
    )
    if work > MAX_COORDINATE_HECKE_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.v_degeneracy_work_bound",
            message="V_d exact coordinate reconstruction exceeds its work envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(format_canonical_integer(abs(v.numerator))),
                len(format_canonical_integer(v.denominator)),
            )
            for v in coordinates
        ),
        default=1,
    )
    image_digits = (
        source_plan.dimension * coordinate_digits
        + source_plan.coefficient_digits
        + len(str(max(1, source_plan.dimension)))
        + 2
    )
    result_digits = image_digits + target_plan.dimension * (
        target_plan.coefficient_digits + target_plan.dimension.bit_length() + 2
    )
    allocation_bytes = (
        len(encode_strict_json(target_space.model_dump(mode="json")))
        + target_plan.dimension * (2 * result_digits + 64)
        + 512
    )
    if (
        result_digits > MAX_COORDINATE_RESULT_DIGITS
        or allocation_bytes > MAX_CHANGE_OF_BASIS_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.v_degeneracy_output_bound",
            message="V_d exact coordinate growth or output exceeds its envelope",
        )

    source_plan = (
        _materialize_pari_basis(source_plan)
        if source_plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else source_plan
    )
    target_plan = (
        _materialize_pari_basis(target_plan)
        if target_plan.basis_id == PARI_STURM_RREF_BASIS_ID
        else target_plan
    )
    source_basis = _basis_coefficients(source_plan)
    source_coefficients = tuple(
        sum(
            (
                c * vector[n]
                for c, vector in zip(coordinates, source_basis, strict=True)
            ),
            Fraction(0),
        )
        for n in range(source_precision)
    )
    target_basis = _basis_coefficients(target_plan)
    image = tuple(
        source_coefficients[n // d] if n % d == 0 else Fraction(0)
        for n in range(target_bound + 1)
    )
    result_coordinates = _solve_sturm_coordinates(target_basis, image)
    if any(
        max(
            len(format_canonical_integer(abs(v.numerator))),
            len(format_canonical_integer(v.denominator)),
        )
        > MAX_COORDINATE_RESULT_DIGITS
        for v in result_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("result", "coordinates"),
            code="modular_form.v_degeneracy_output_bound",
            message="exact V_d coordinates exceed the result digit envelope",
        )
    return ModularFormCoordinates.model_construct(
        space=target_space,
        basis_id=target_plan.basis_id,
        coordinates=tuple(
            CanonicalRational(num=v.numerator, den=v.denominator)
            for v in result_coordinates
        ),
    )


def modular_form_operator_image(
    source_form: ModularFormCoordinates, operator: str, prime: int
) -> ModularFormOperatorImage:
    """Bind U_p or V_p to an exact level-one form and its Gamma0(p) parent."""

    _admit_coordinates(source_form, 1)
    if operator not in ("U", "V"):
        raise OperationDomainValidationError(
            location=("operator",),
            code="modular_form.operator_image_operator",
            message="operator must be U or V",
        )
    if type(prime) is not int or prime < 2 or prime > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("prime",),
            code="modular_form.operator_image_prime_bound",
            message=(
                "U_p and V_p require a prime in the admitted Gamma0 level "
                f"envelope [2, {MAX_GAMMA0_OPERATION_LEVEL}]"
            ),
        )
    if not _is_prime_level(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_image_prime",
            message="operator index must be prime",
        )
    codomain = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=prime,
        weight=source_form.space.weight,
        kind=source_form.space.kind,
        character="TRIVIAL",
        coefficient_domain="QQ",
    )
    return ModularFormOperatorImage.model_construct(
        source_form=source_form,
        operator=operator,
        prime=prime,
        codomain=codomain,
    )


def _admit_operator_image_prefix(
    image: object, precision: object
) -> tuple[ModularFormOperatorImage, int, _BasisPlan, tuple[Fraction, ...]]:
    if not isinstance(image, ModularFormOperatorImage):
        raise OperationDomainValidationError(
            location=("image",),
            code="modular_form.operator_image_type",
            message="image must be an exact U_p or V_p operator-image value",
        )
    source_form = getattr(image, "source_form", None)
    operator = getattr(image, "operator", None)
    prime = getattr(image, "prime", None)
    codomain = getattr(image, "codomain", None)
    if operator not in ("U", "V"):
        raise OperationDomainValidationError(
            location=("image", "operator"),
            code="modular_form.operator_image_operator",
            message="operator-image value must name U or V",
        )
    if type(prime) is not int or prime < 2 or prime > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("image", "prime"),
            code="modular_form.operator_image_prime_bound",
            message="operator-image prime is outside the admitted prime envelope",
        )
    if not _is_prime_level(prime):
        raise OperationDomainValidationError(
            location=("image", "prime"),
            code="modular_form.operator_image_prime",
            message="operator-image index must be prime",
        )
    if not isinstance(source_form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("image", "source_form"),
            code="modular_form.operator_image_source_type",
            message="operator-image source must be exact level-one coordinates",
        )
    if not isinstance(source_form.space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("image", "source_form", "space"),
            code="modular_form.operator_image_source_space",
            message="operator-image source must retain its modular-form space",
        )
    if source_form.space.level != 1:
        raise OperationDomainValidationError(
            location=("image", "source_form", "space", "level"),
            code="modular_form.operator_image_source_level",
            message="operator-image source must have level one",
        )
    if not isinstance(codomain, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("image", "codomain"),
            code="modular_form.operator_image_codomain_type",
            message="operator-image codomain must be a modular-form space",
        )
    expected_codomain = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=prime,
        weight=source_form.space.weight,
        kind=source_form.space.kind,
        character="TRIVIAL",
        coefficient_domain="QQ",
    )
    if codomain != expected_codomain:
        raise OperationDomainValidationError(
            location=("image", "codomain"),
            code="modular_form.operator_image_codomain",
            message="operator-image codomain must be the matching Gamma0(prime) space",
        )
    if (
        type(precision) is not int
        or precision < 1
        or precision > MAX_Q_TRANSFORM_OUTPUT_PRECISION
    ):
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.operator_image_precision_bound",
            message=(
                "operator-image precision must lie in the bounded envelope "
                f"[1, {MAX_Q_TRANSFORM_OUTPUT_PRECISION}]"
            ),
        )

    source_order = (
        prime * (precision - 1) + 1 if operator == "U" else (precision - 1) // prime + 1
    )
    if source_order > MAX_LEVEL_ONE_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.operator_image_source_precision_bound",
            message=(
                "operator-image request needs more level-one basis coefficients "
                f"than the admitted source order {MAX_LEVEL_ONE_BASIS_PRECISION}"
            ),
        )
    plan, coordinates = _admit_coordinates(source_form, source_order)
    work = plan.work + precision + source_order * max(1, plan.dimension)
    if work > MAX_OPERATOR_IMAGE_WORK:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.operator_image_work_bound",
            message="operator-image prefix work exceeds its exact envelope",
        )
    coordinate_digits = (
        max(
            max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            )
            for value in coordinates
        )
        if coordinates
        else 1
    )
    result_digits = (
        plan.dimension * coordinate_digits
        + plan.coefficient_digits
        + len(str(max(1, plan.dimension)))
        + 2
    )
    allocation_bytes = precision * (2 * result_digits + 64)
    if allocation_bytes > MAX_OPERATOR_IMAGE_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.operator_image_output_bound",
            message="operator-image prefix exceeds the bounded exact output size",
        )
    return image, source_order, plan, coordinates


def modular_form_operator_image_q_expansion(
    image: ModularFormOperatorImage, precision: int
) -> ModularFormOperatorImagePrefix:
    """Evaluate a bounded U_p or V_p q-prefix from exact source coordinates."""

    image, source_order, plan, coordinates = _admit_operator_image_prefix(
        image, precision
    )
    basis_vectors = _basis_coefficients(plan)
    source_coefficients = tuple(
        sum(
            (
                coordinate * vector[index]
                for coordinate, vector in zip(coordinates, basis_vectors, strict=True)
            ),
            Fraction(0),
        )
        for index in range(source_order)
    )
    coefficients = []
    for index in range(precision):
        if index % 32 == 0:
            request_checkpoint("during modular-form U_p or V_p prefix evaluation")
        if image.operator == "U":
            coefficients.append(source_coefficients[image.prime * index])
        elif index % image.prime == 0:
            coefficients.append(source_coefficients[index // image.prime])
        else:
            coefficients.append(Fraction(0))
    series = TruncatedSeries.model_construct(
        variable="q",
        truncation_order=precision,
        coefficients=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in coefficients
        ),
    )
    return ModularFormOperatorImagePrefix.model_construct(
        image=image,
        q_expansion=series,
    )


__all__ = [
    "BASIS_ID",
    "GAMMA0_FOUR_BASIS_ID",
    "GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID",
    "GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID",
    "GAMMA0_THREE_BASIS_ID",
    "GAMMA0_TWO_BASIS_ID",
    "modular_form_basis_q_expansions",
    "modular_form_coordinates_atkin_lehner",
    "modular_form_coordinates_hecke",
    "modular_form_coordinates_q_expansion",
    "modular_form_coordinates_u2",
    "modular_form_coordinates_u_prime",
    "modular_form_coordinates_v2",
    "modular_form_coordinates_v3",
    "modular_form_coordinates_v_degeneracy",
    "modular_form_hecke_matrix",
    "modular_form_operator_image",
    "modular_form_operator_image_q_expansion",
]
