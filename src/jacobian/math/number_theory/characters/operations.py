"""Native exact operations for bounded principal Dirichlet characters."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Literal, cast

from pydantic import ValidationError
from pydantic_core import PydanticCustomError
from sympy import QQ, Poly, bernoulli, cyclotomic_poly, factorint, symbols

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    MAX_CYCLIC_FIELD_WORK,
    MAX_CYCLIC_PERIOD,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters._models import (
    MAX_GENERALIZED_BERNOULLI_INDEX,
    DirichletCharacterConductorResult,
    DirichletCharacterGaussSumResult,
    DirichletCharacterGeneralizedBernoulliPrefix,
    DirichletCharacterGeneralizedBernoulliResult,
    DirichletCharacterGeneralizedGaussSumResult,
    DirichletCharacterJacobiSumResult,
    DirichletCharacterLValueNonpositiveResult,
    DirichletCharacterOrderResult,
    DirichletCharacterOrthogonalityResult,
    DirichletCharacterParityResult,
    DirichletCharacterPrimitiveGaussNormResult,
    DirichletCharacterResidueIndicatorExpansion,
    DirichletCharacterTableResult,
    DirichletCharacterValueResult,
    PrincipalDirichletCharacterValueResult,
    _require_bounded_digits,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterFamily,
    DirichletCharacterFourierMatrix,
    DirichletCharacterGroup,
    DirichletCharacterInflation,
    DirichletCharacterKernel,
    DirichletCharacterRestrictionObstruction,
    DirichletCharacterRestrictionResult,
    PrincipalDirichletCharacter,
)
from jacobian.math.number_theory.sequences.core._models import (
    FiniteCyclotomicSequence,
    FiniteIntegerSequence,
    FiniteRationalSequence,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
    MAX_SEQUENCE_TOTAL_DIGITS,
)

__all__ = [
    "character_group",
    "dirichlet_character",
    "dirichlet_character_conductor",
    "dirichlet_character_conjugate",
    "dirichlet_character_fourier_matrix",
    "dirichlet_character_gauss_sum",
    "dirichlet_character_generalized_bernoulli",
    "dirichlet_character_generalized_bernoulli_prefix",
    "dirichlet_character_generalized_gauss_sum",
    "dirichlet_character_group_enumerate",
    "dirichlet_character_inflate",
    "dirichlet_character_inverse",
    "dirichlet_character_jacobi_sum",
    "dirichlet_character_kernel",
    "dirichlet_character_l_value_nonpositive_integer",
    "dirichlet_character_order",
    "dirichlet_character_orthogonality",
    "dirichlet_character_parity",
    "dirichlet_character_power",
    "dirichlet_character_primitive_gauss_norm",
    "dirichlet_character_product",
    "dirichlet_character_residue_indicator_expansion",
    "dirichlet_character_restrict_modulus",
    "dirichlet_character_sequence_twist",
    "dirichlet_character_table",
    "dirichlet_character_value",
    "principal_dirichlet_character",
    "principal_dirichlet_character_value",
    "require_complete_character_group",
    "require_complete_principal_dirichlet_character",
    "require_principal_dirichlet_character_value_result",
]

MAX_CHARACTER_GROUP_WORK = 500_000
MAX_CHARACTER_ENUMERATION_WORK = 250_000
MAX_CHARACTER_ENUMERATION_TOTAL_WORK = MAX_CHARACTER_GROUP_WORK + 100_000
MAX_CHARACTER_RESTRICTION_WORK = (
    3 * MAX_CHARACTER_GROUP_WORK + MAX_CHARACTER_ENUMERATION_WORK
)
MAX_CONDUCTOR_KERNEL_WORK = 4_194_304
MAX_CHARACTER_SUM_WORK = 1_000_000
MAX_CHARACTER_ORTHOGONALITY_WORK = 500_000
MAX_GENERALIZED_BERNOULLI_WORK = 500_000
MAX_CHARACTER_RESULT_CELLS = 1_000_000
MAX_CHARACTER_RESULT_DIGITS = 5_000_000
MAX_CHARACTER_SEQUENCE_TWIST_WORK = 10_000_000
MAX_CHARACTER_SEQUENCE_TWIST_COEFFICIENTS = 1_000_000


def _admit_result_allocation(
    *,
    cells: int,
    digits: int,
    location: tuple[str | int, ...],
    code: str,
    message: str,
) -> None:
    """Bound exact result cardinality and integer growth before allocation."""

    if cells > MAX_CHARACTER_RESULT_CELLS or digits > MAX_CHARACTER_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=location,
            code=code,
            message=message,
        )


def _jacobi_sum_field(order: int, modulus: int) -> tuple[int, tuple[int, ...]]:
    """Admit and materialize the tiny defining polynomial needed for a sum."""

    from jacobian.math.matrices.cyclic_linear._models import MAX_CYCLIC_PERIOD

    if order > MAX_CYCLIC_PERIOD:
        raise OperationResourceAdmissionError(
            location=("left", "group", "exponent"),
            code="dirichlet_character.jacobi_sum.cyclotomic_order_bound",
            message=(
                "Jacobi-sum cyclotomic order exceeds the exact value bound "
                f"{MAX_CYCLIC_PERIOD}"
            ),
        )
    degree = _euler_phi(order)
    work = modulus + order * degree
    if work > MAX_CHARACTER_SUM_WORK or work > MAX_CYCLIC_FIELD_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "group", "modulus"),
            code="dirichlet_character.jacobi_sum.work_bound",
            message="Jacobi sum exceeds the admitted residue and field work envelope",
        )
    # The defining polynomial is admitted before construction: order <= 256,
    # degree <= order, and the bounded exact polynomial work is <= 65,536.
    x = symbols("x")
    coefficients = tuple(
        int(value) for value in Poly(cyclotomic_poly(order, x), x).all_coeffs()
    )
    if len(coefficients) != degree + 1 or coefficients[0] != 1:
        raise RuntimeError("cyclotomic polynomial has an unexpected canonical shape")
    return degree, tuple(reversed(coefficients))


def _admit_jacobi_result_growth(
    *, modulus: int, order: int, degree: int, cyclotomic_coefficients: tuple[int, ...]
) -> None:
    """Prove a coefficient envelope before enumerating the residue sum."""

    digit_limit = MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    limit = 10**digit_limit - 1
    reduction_norm = sum(abs(value) for value in cyclotomic_coefficients[:-1])
    # A reduced monomial x^j, j < order, needs at most order-degree
    # substitutions of x^degree = -sum(c_i*x^i). The l1 norm multiplies by
    # at most reduction_norm at each substitution, and there are at most N
    # terms in the Jacobi sum.
    bound = modulus
    for _ in range(max(0, order - degree)):
        if reduction_norm and bound > limit // reduction_norm:
            raise OperationResourceAdmissionError(
                location=("right",),
                code="dirichlet_character.jacobi_sum.coefficient_bound",
                message=(
                    "Jacobi sum may exceed the "
                    f"{digit_limit}-digit exact cyclotomic coefficient bound"
                ),
            )
        bound *= reduction_norm
    if bound > limit:
        raise OperationResourceAdmissionError(
            location=("right",),
            code="dirichlet_character.jacobi_sum.coefficient_bound",
            message=(
                "Jacobi sum may exceed the "
                f"{digit_limit}-digit exact cyclotomic coefficient bound"
            ),
        )


def dirichlet_character_jacobi_sum(
    left: DirichletCharacter, right: DirichletCharacter
) -> DirichletCharacterJacobiSumResult:
    """Return J(chi, psi)=sum_{a mod N} chi(a) psi(1-a) exactly."""

    left = _require_character(left)
    right = _require_character(right)
    if left.group != right.group:
        raise OperationDomainValidationError(
            location=("right", "group"),
            code="dirichlet_character.jacobi_sum.parent_mismatch",
            message="Jacobi-sum characters must use the identical group parent",
        )
    group = left.group
    order = group.exponent
    degree, phi_coefficients = _jacobi_sum_field(order, group.modulus)
    _admit_jacobi_result_growth(
        modulus=group.modulus,
        order=order,
        degree=degree,
        cyclotomic_coefficients=phi_coefficients,
    )

    row_by_residue = dict(zip(group.unit_residues, group.unit_coordinates, strict=True))

    def value_exponent(character: DirichletCharacter, row: tuple[int, ...]) -> int:
        return (
            sum(
                coordinate * (order // generator_order) * unit_coordinate
                for coordinate, generator_order, unit_coordinate in zip(
                    character.coordinates,
                    group.generator_orders,
                    row,
                    strict=True,
                )
            )
            % order
        )

    power_coefficients = [0] * order
    for residue in range(group.modulus):
        left_row = row_by_residue.get(residue)
        if left_row is None:
            continue
        right_row = row_by_residue.get((1 - residue) % group.modulus)
        if right_row is None:
            continue
        exponent = (
            value_exponent(left, left_row) + value_exponent(right, right_row)
        ) % order
        power_coefficients[exponent] += 1

    reduced = power_coefficients
    for power in range(order - 1, degree - 1, -1):
        coefficient = reduced[power]
        if coefficient:
            reduced[power] = 0
            for lower_power in range(degree):
                reduced[power - degree + lower_power] -= (
                    coefficient * phi_coefficients[lower_power]
                )
    if any(
        len(str(abs(value))) > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        for value in reduced[:degree]
    ):
        raise RuntimeError("admitted Jacobi coefficient bound was violated")
    value = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=order),
        coefficients_ascending=tuple(
            CanonicalRational.from_integer_ratio(value, 1) for value in reduced[:degree]
        ),
    )
    return DirichletCharacterJacobiSumResult(
        left=left,
        right=right,
        value=value,
    )


def dirichlet_character_orthogonality(
    left: DirichletCharacter, right: DirichletCharacter
) -> DirichletCharacterOrthogonalityResult:
    r"""Return ``sum_{a mod N} chi(a) conjugate(psi(a))`` exactly.

    The validated residue-to-unit-coordinate table identifies the residue sum
    with a product of finite geometric sums over the cyclic unit-group axes.
    Each axis contributes its order when the two character coordinates agree,
    and zero otherwise. Nonunits contribute zero by the Dirichlet extension.
    """

    left = _require_character(left)
    right = _require_character(right)
    if left.group != right.group:
        raise OperationDomainValidationError(
            location=("right", "group"),
            code="dirichlet_character.orthogonality.parent_mismatch",
            message="orthogonality characters must use the identical group parent",
        )
    group = left.group
    # The parent admission above bounds the complete unit table by 2,048 rows.
    # Price the coordinate product before doing any orthogonality arithmetic.
    work = len(group.unit_residues) * max(1, len(group.generator_orders))
    if work > MAX_CHARACTER_ORTHOGONALITY_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "group"),
            code="dirichlet_character.orthogonality.work_bound",
            message=(
                "character orthogonality exceeds the admitted unit-coordinate "
                "work envelope"
            ),
        )

    value = 1
    for left_coordinate, right_coordinate, axis_order in zip(
        left.coordinates, right.coordinates, group.generator_orders, strict=True
    ):
        if (left_coordinate - right_coordinate) % axis_order:
            value = 0
            break
        value *= axis_order
    return DirichletCharacterOrthogonalityResult._from_kernel(
        left=left,
        right=right,
        value=value,
    )


def _admit_character_integer(
    integer: int, *, type_code: str = "dirichlet_character.integer_type"
) -> None:
    """Admit the exact integer envelope shared by native value operations."""

    if type(integer) is not int:
        raise OperationDomainValidationError(
            location=("integer",),
            code=type_code,
            message="integer must be an exact integer",
        )
    try:
        _require_bounded_digits(integer)
    except PydanticCustomError as exc:
        raise OperationResourceAdmissionError(
            location=("integer",), code=exc.type, message=exc.message()
        ) from exc


def _admit_character_group(modulus: int) -> None:
    """Admit one character-group modulus, shared by native and catalog paths."""

    if type(modulus) is not int:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_type",
            message="character-group modulus must be an integer",
        )
    if modulus < 1:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_sign",
            message="character-group modulus must be positive",
        )
    if modulus > MAX_CHARACTER_GROUP_MODULUS:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_bound",
            message=(
                "character-group modulus exceeds the bound "
                f"{MAX_CHARACTER_GROUP_MODULUS}"
            ),
        )
    phi = _euler_phi(modulus)
    table_work = modulus + phi * max(1, len(f"{modulus}"))
    if phi > MAX_CHARACTER_GROUP_MODULUS or table_work > MAX_CHARACTER_GROUP_WORK:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="dirichlet_character.group.table_bound",
            message="character-group unit table exceeds the admitted work envelope",
        )


def _euler_phi(modulus: int) -> int:
    """Return Euler's totient using exact prime factorization."""

    if modulus <= 1:
        return 1
    result = 1
    for prime, exponent in factorint(modulus).items():
        result *= (prime - 1) * prime ** (exponent - 1)
    return result


def _multiplicative_order(residue: int, order_group: int, modulus: int) -> int:
    """Return the exact multiplicative order of a unit modulo ``modulus``."""

    order = order_group
    for prime, _ in factorint(order).items():
        while order % prime == 0 and pow(residue, order // prime, modulus) == 1:
            order //= prime
    return order


def _primitive_root_prime_power(prime: int, exponent: int) -> tuple[int, int]:
    """Return the canonical generator and order of an odd prime-power group."""

    modulus = prime**exponent
    group_order = (prime - 1) * prime ** (exponent - 1)
    for candidate in range(2, modulus):
        if gcd(candidate, modulus) != 1:
            continue
        if _multiplicative_order(candidate, group_order, modulus) == group_order:
            return candidate, group_order
    raise RuntimeError("odd prime-power unit group has no generator")


def _admit_principal_modulus(modulus: int) -> None:
    if type(modulus) is not int:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.principal.modulus_type",
            message="principal-character modulus must be an integer",
        )
    if not 1 <= modulus <= MAX_PRINCIPAL_CHARACTER_MODULUS:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.principal.modulus_bound",
            message=(
                "principal-character modulus must be between 1 and "
                f"{MAX_PRINCIPAL_CHARACTER_MODULUS}"
            ),
        )


def _require_character(character: DirichletCharacter) -> DirichletCharacter:
    if not isinstance(character, DirichletCharacter):
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.character_type",
            message="character must be a Dirichlet character value",
        )
    group = require_complete_character_group(
        cast(DirichletCharacterGroup, getattr(character, "group", None))
    )
    orders = group.generator_orders
    coordinates = getattr(character, "coordinates", None)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != len(orders)
        or any(
            type(coordinate) is not int or coordinate < 0 or coordinate >= order
            for coordinate, order in zip(coordinates, orders, strict=True)
        )
    ):
        raise OperationDomainValidationError(
            location=("character", "coordinates"),
            code="dirichlet_character.coordinates_invalid",
            message="character coordinates must lie on the complete dual-group axes",
        )
    return DirichletCharacter.model_construct(group=group, coordinates=coordinates)


def require_complete_principal_dirichlet_character(
    character: PrincipalDirichletCharacter,
) -> PrincipalDirichletCharacter:
    """Check the mathematical table claimed by a caller-supplied character."""
    if not isinstance(character, PrincipalDirichletCharacter):
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.principal.character_type",
            message="character must be a principal Dirichlet character value",
        )
    try:
        character = PrincipalDirichletCharacter.model_validate(character.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.principal.invalid_character",
            message="principal character has malformed authored fields",
        ) from exc
    _admit_principal_modulus(character.modulus)

    expected_units = tuple(
        residue
        for residue in range(character.modulus)
        if math.gcd(residue, character.modulus) == 1
    )
    if character.unit_residues != expected_units:
        raise OperationDomainValidationError(
            location=("character", "unit_residues"),
            code="dirichlet_character.unit_residues_mismatch",
            message=(
                "unit residues must be the complete canonical unit group modulo modulus"
            ),
        )
    units = frozenset(expected_units)
    expected_values = tuple(
        1 if residue in units else 0 for residue in range(character.modulus)
    )
    if character.values != expected_values:
        raise OperationDomainValidationError(
            location=("character", "values"),
            code="dirichlet_character.values_table_mismatch",
            message=(
                "values must be the complete extension-by-zero principal character table"
            ),
        )
    return character


def require_principal_dirichlet_character_value_result(
    result: PrincipalDirichletCharacterValueResult,
) -> None:
    """Check the character and source-evaluation relation of a claimed result."""

    if not isinstance(result, PrincipalDirichletCharacterValueResult):
        raise OperationDomainValidationError(
            location=("result",),
            code="dirichlet_character.principal.result_type",
            message="result must be a principal-character evaluation value",
        )
    # This admission is deliberately performed before int() so bools, lists,
    # and oversized authored carriers cannot become a successful source claim.
    _admit_character_integer(
        cast(int, getattr(result, "integer", None)),
        type_code="dirichlet_character.principal.integer_type",
    )
    try:
        canonical = PrincipalDirichletCharacterValueResult.model_validate(
            result.model_dump()
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("result",),
            code="dirichlet_character.principal.invalid_result",
            message="principal-character result has malformed authored fields",
        ) from exc
    result = canonical
    require_complete_principal_dirichlet_character(result.character)
    residue = int(result.integer) % result.character.modulus
    if result.canonical_residue != residue:
        raise ValueError("canonical residue does not match the source integer")
    expected_is_unit = math.gcd(residue, result.character.modulus) == 1
    if result.is_unit != expected_is_unit:
        raise ValueError("unit status does not match the source character modulus")
    if result.value != result.character.values[residue]:
        raise ValueError("value does not match the source principal-character table")


def dirichlet_character(
    group: DirichletCharacterGroup, coordinates: tuple[int, ...]
) -> DirichletCharacter:
    group = require_complete_character_group(group)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != len(group.generator_orders)
        or any(
            type(c) is not int or c < 0 or c >= order
            for c, order in zip(coordinates, group.generator_orders, strict=False)
        )
    ):
        raise OperationDomainValidationError(
            location=("coordinates",),
            code="dirichlet_character.coordinates_shape",
            message="character coordinates must match the group's dual axes",
        )
    return DirichletCharacter(group=group, coordinates=coordinates)


def _character_value(
    character: DirichletCharacter, integer: int
) -> CyclotomicValue | None:
    """Evaluate after the caller has performed one character admission."""
    residue = integer % character.group.modulus
    if residue not in character.group.unit_residues:
        return None
    row = character.group.unit_coordinates[character.group.unit_residues.index(residue)]
    order = character.group.exponent
    exponent = (
        sum(
            c * (order // o) * r
            for c, o, r in zip(
                character.coordinates,
                character.group.generator_orders,
                row,
                strict=True,
            )
        )
        % order
    )
    return CyclotomicValue(order=order, exponent=exponent)


def dirichlet_character_value(
    character: DirichletCharacter, integer: int
) -> DirichletCharacterValueResult:
    _admit_character_integer(integer)
    character = _require_character(character)
    value = _character_value(character, integer)
    residue = integer % character.group.modulus
    return DirichletCharacterValueResult(
        character=character,
        integer=integer,
        canonical_residue=residue,
        is_unit=value is not None,
        value=value,
    )


def dirichlet_character_product(
    left: DirichletCharacter, right: DirichletCharacter
) -> DirichletCharacter:
    left = _require_character(left)
    right = _require_character(right)
    if left.group != right.group:
        raise OperationDomainValidationError(
            location=("right", "group"),
            code="dirichlet_character.parent_mismatch",
            message="characters must use the identical group parent",
        )
    coords = tuple(
        (a + b) % order
        for a, b, order in zip(
            left.coordinates,
            right.coordinates,
            left.group.generator_orders,
            strict=True,
        )
    )
    return DirichletCharacter(group=left.group, coordinates=coords)


def dirichlet_character_order(
    character: DirichletCharacter,
) -> DirichletCharacterOrderResult:
    """Return the exact order from the character's invariant-factor axes."""
    character = _require_character(character)
    order = math.lcm(
        *(
            axis_order // gcd(coordinate, axis_order)
            for coordinate, axis_order in zip(
                character.coordinates,
                character.group.generator_orders,
                strict=True,
            )
        )
    )
    return DirichletCharacterOrderResult(character=character, order=order)


def dirichlet_character_kernel(
    character: DirichletCharacter,
) -> DirichletCharacterKernel:
    """Return all units with exact character value one."""
    character = _require_character(character)
    work = len(character.group.unit_residues) * max(
        1, len(character.group.generator_orders)
    )
    if work > MAX_CHARACTER_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("character", "group"),
            code="dirichlet_character.kernel.work_bound",
            message="kernel evaluation exceeds its bounded dual-coordinate work",
        )
    residues = tuple(
        residue
        for residue, row in zip(
            character.group.unit_residues,
            character.group.unit_coordinates,
            strict=True,
        )
        if sum(
            coordinate * (character.group.exponent // axis_order) * unit_coordinate
            for coordinate, axis_order, unit_coordinate in zip(
                character.coordinates,
                character.group.generator_orders,
                row,
                strict=True,
            )
        )
        % character.group.exponent
        == 0
    )
    return DirichletCharacterKernel(
        character=character,
        residues=residues,
        index=character.group.character_count // len(residues),
    )


def dirichlet_character_parity(
    character: DirichletCharacter,
) -> DirichletCharacterParityResult:
    """Return exact chi(-1), labelled EVEN or ODD by its sign."""
    character = _require_character(character)
    value = _character_value(character, -1)
    if value is None:
        raise RuntimeError("minus one must be a unit modulo every positive modulus")
    if value.exponent == 0:
        parity: Literal["EVEN", "ODD"] = "EVEN"
    elif (
        character.group.exponent % 2 == 0
        and value.exponent == character.group.exponent // 2
    ):
        parity = "ODD"
    else:
        raise RuntimeError("a character maps the order-two unit -1 to a sign")
    return DirichletCharacterParityResult(
        character=character,
        value=value,
        parity=parity,
    )


def dirichlet_character_power(
    character: DirichletCharacter, exponent: int
) -> DirichletCharacter:
    """Return an exact signed power by scaling each dual coordinate."""
    _admit_character_integer(exponent, type_code="dirichlet_character.exponent_type")
    character = _require_character(character)
    return DirichletCharacter(
        group=character.group,
        coordinates=tuple(
            (coordinate * exponent) % axis_order
            for coordinate, axis_order in zip(
                character.coordinates,
                character.group.generator_orders,
                strict=True,
            )
        ),
    )


def dirichlet_character_table(
    character: DirichletCharacter,
) -> DirichletCharacterTableResult:
    character = _require_character(character)
    residues = tuple(range(character.group.modulus))
    return DirichletCharacterTableResult(
        character=character,
        residues=residues,
        values=tuple(_character_value(character, residue) for residue in residues),
    )


def dirichlet_character_conjugate(character: DirichletCharacter) -> DirichletCharacter:
    character = _require_character(character)
    return DirichletCharacter(
        group=character.group,
        coordinates=tuple(
            (-c) % order
            for c, order in zip(
                character.coordinates, character.group.generator_orders, strict=True
            )
        ),
    )


def dirichlet_character_inverse(character: DirichletCharacter) -> DirichletCharacter:
    """Return the inverse of one exact character in its finite dual group."""
    character = _require_character(character)
    return DirichletCharacter(
        group=character.group,
        coordinates=tuple(
            (-coordinate) % axis_order
            for coordinate, axis_order in zip(
                character.coordinates, character.group.generator_orders, strict=True
            )
        ),
    )


def dirichlet_character_conductor(
    character: DirichletCharacter,
) -> DirichletCharacterConductorResult:
    """Compute the least divisor through which ``character`` factors.

    For each ``d | N``, reduction ``(Z/NZ)^* -> (Z/dZ)^*`` is surjective.
    The character factors through it exactly when it is trivial on its kernel,
    the units congruent to 1 modulo ``d``. This checks that criterion using the
    supplied complete unit coordinates and exact cyclotomic exponents.
    """
    character = _require_character(character)
    group = character.group
    divisors = tuple(
        divisor
        for divisor in range(1, group.modulus + 1)
        if group.modulus % divisor == 0
    )
    work = len(divisors) * len(group.unit_residues)
    if work > MAX_CONDUCTOR_KERNEL_WORK:
        raise OperationResourceAdmissionError(
            location=("character", "group", "modulus"),
            code="dirichlet_character.conductor.work_bound",
            message="conductor kernel test exceeds its admitted work envelope",
        )

    common_order = group.exponent
    character_kernel: set[int] = set()
    for residue, row in zip(group.unit_residues, group.unit_coordinates, strict=True):
        value_exponent = (
            sum(
                coordinate * (common_order // generator_order) * unit_coordinate
                for coordinate, generator_order, unit_coordinate in zip(
                    character.coordinates,
                    group.generator_orders,
                    row,
                    strict=True,
                )
            )
            % common_order
        )
        if value_exponent == 0:
            character_kernel.add(residue)

    conductor = next(
        divisor
        for divisor in divisors
        if all(
            residue in character_kernel
            for residue in group.unit_residues
            if residue % divisor == 1 % divisor
        )
    )
    primitive_group = character_group(conductor)
    # The reduction on unit groups is surjective. Lift each canonical target
    # generator to a source unit, then read its exact character value to recover
    # the target dual coordinates. The already-proved conductor criterion
    # guarantees that the lift choice does not affect the result.
    source_lifts = {
        residue % conductor: (residue, row)
        for residue, row in zip(
            group.unit_residues, group.unit_coordinates, strict=True
        )
    }
    primitive_coordinates: list[int] = []
    for generator, target_order in zip(
        primitive_group.generators,
        primitive_group.generator_orders,
        strict=True,
    ):
        lift = source_lifts.get(generator % conductor)
        if lift is None:
            raise RuntimeError("unit reduction failed to provide a generator lift")
        _, source_row = lift
        source_value_exponent = (
            sum(
                coordinate * (common_order // generator_order) * unit_coordinate
                for coordinate, generator_order, unit_coordinate in zip(
                    character.coordinates,
                    group.generator_orders,
                    source_row,
                    strict=True,
                )
            )
            % common_order
        )
        embedding_scale = common_order // primitive_group.exponent
        if source_value_exponent % embedding_scale:
            raise RuntimeError("induced character value is outside the target field")
        target_exponent = source_value_exponent // embedding_scale
        scale = primitive_group.exponent // target_order
        if target_exponent % scale:
            raise RuntimeError("induced character value is outside the target field")
        primitive_coordinates.append((target_exponent // scale) % target_order)
    primitive_character = DirichletCharacter.model_construct(
        group=primitive_group,
        coordinates=tuple(primitive_coordinates),
    )
    return DirichletCharacterConductorResult._from_kernel(
        character, conductor, primitive_character
    )


def principal_dirichlet_character(modulus: int) -> PrincipalDirichletCharacter:
    """Return the complete extension-by-zero principal character modulo ``modulus``."""

    _admit_principal_modulus(modulus)
    unit_residues = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    units = frozenset(unit_residues)
    return PrincipalDirichletCharacter._from_kernel(
        modulus=modulus,
        unit_residues=unit_residues,
        values=tuple(1 if residue in units else 0 for residue in range(modulus)),
    )


def principal_dirichlet_character_value(
    character: PrincipalDirichletCharacter, integer: int
) -> int:
    """Return the exact value of ``character`` at one integer."""

    _admit_character_integer(
        integer, type_code="dirichlet_character.principal.integer_type"
    )
    character = require_complete_principal_dirichlet_character(character)
    return character.values[integer % character.modulus]


def _canonical_blocks(modulus: int) -> tuple[tuple[int, int, int], ...]:
    """Return the nontrivial canonical prime-power generator blocks.

    Blocks of order one generate nothing and carry no coordinates; they are
    dropped so the generator tuple is a genuine generating set.
    """

    if modulus <= 1:
        return ()
    return tuple(
        block
        for prime, exponent in sorted(factorint(modulus).items())
        for block in _prime_power_blocks(prime, exponent)
        if block[2] > 1
    )


def _prime_power_blocks(prime: int, exponent: int) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(block modulus, generator, order)`` rows.

    Odd prime powers are cyclic with the smallest primitive-root generator.
    Powers of two use the standard ``{-1, 5}`` presentation.
    """

    modulus = prime**exponent
    if modulus <= 2:
        return ((modulus, 1 % modulus, 1),)
    if prime == 2 and modulus == 4:
        return ((4, 3, 2),)
    if prime == 2:
        return ((modulus, modulus - 1, 2), (modulus, 5, 2 ** (exponent - 2)))
    generator, order = _primitive_root_prime_power(prime, exponent)
    return ((modulus, generator, order),)


def _crt_lift(modulus: int, block_modulus: int, block_residue: int) -> int:
    """Lift one block residue to ``modulus`` as 1 outside its block."""

    cofactor = modulus // block_modulus
    first = block_residue * cofactor * pow(cofactor % block_modulus, -1, block_modulus)
    if cofactor == 1:
        return first % modulus
    second = block_modulus * pow(block_modulus % cofactor, -1, cofactor)
    return (first + second) % modulus


def _invariant_factors(component_orders: tuple[int, ...]) -> tuple[int, ...]:
    """Combine cyclic component orders into the divisibility chain."""

    if not component_orders or all(order == 1 for order in component_orders):
        return ()
    prime_exponents: dict[int, list[int]] = {}
    for order in component_orders:
        if order == 1:
            continue
        for prime, exponent in factorint(order).items():
            prime_exponents.setdefault(prime, []).append(prime**exponent)
    rank = max(len(powers) for powers in prime_exponents.values())
    for powers in prime_exponents.values():
        powers.sort()
        while len(powers) < rank:
            powers.insert(0, 1)
    factors = tuple(
        math.prod(powers[index] for powers in prime_exponents.values())
        for index in range(rank)
    )
    return tuple(sorted(factors))


def _discrete_log(generator: int, target: int, order: int, modulus: int) -> int:
    """Return the bounded discrete logarithm of ``target`` to ``generator``."""

    power = 1 % modulus
    for exponent in range(order):
        if power == target:
            return exponent
        power = (power * generator) % modulus
    raise ValueError("target is not in the cyclic subgroup")


def _component_log_table(
    block_modulus: int, generators_orders: tuple[tuple[int, int], ...]
) -> dict[int, tuple[int, ...]]:
    """Tabulate joint discrete logarithms over one prime-power block.

    Blocks with several generators (powers of two) are solved jointly by
    enumerating the bounded product exactly once; first enumeration order is
    canonical.
    """

    table: dict[int, tuple[int, ...]] = {}
    ranges = [range(order) for _, order in generators_orders]
    for coordinates in product(*ranges):
        value = 1 % block_modulus
        for (generator, _), coordinate in zip(
            generators_orders, coordinates, strict=True
        ):
            value = (value * pow(generator, coordinate, block_modulus)) % block_modulus
        table.setdefault(value, coordinates)
    return table


def require_complete_character_group(
    group: DirichletCharacterGroup,
) -> DirichletCharacterGroup:
    """Check the mathematical decomposition claimed by a caller-supplied group."""

    if not isinstance(group, DirichletCharacterGroup):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.type",
            message="group must be a Dirichlet-character group value",
        )
    modulus = cast(int, getattr(group, "modulus", None))
    _admit_character_group(modulus)
    expected_units = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    if getattr(group, "unit_residues", None) != expected_units:
        raise OperationDomainValidationError(
            location=("group", "unit_residues"),
            code="dirichlet_character.group.unit_residues_mismatch",
            message=(
                "unit residues must be the complete canonical unit group modulo modulus"
            ),
        )
    structural_fields = (
        getattr(group, "generator_orders", None),
        getattr(group, "generators", None),
        getattr(group, "invariant_factors", None),
        getattr(group, "unit_coordinates", None),
    )
    if any(type(value) is not tuple for value in structural_fields):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.invalid_group",
            message="character group has malformed authored fields",
        )
    generator_orders = cast(tuple[object, ...], structural_fields[0])
    generators = cast(tuple[object, ...], structural_fields[1])
    invariant_factors = cast(tuple[object, ...], structural_fields[2])
    unit_coordinates = cast(tuple[object, ...], structural_fields[3])
    if (
        type(getattr(group, "character_count", None)) is not int
        or type(getattr(group, "exponent", None)) is not int
        or any(type(value) is not int or value <= 0 for value in generator_orders)
        or any(type(value) is not int for value in generators)
        or any(type(value) is not int or value <= 0 for value in invariant_factors)
        or len(unit_coordinates) != len(expected_units)
        or any(
            type(row) is not tuple or any(type(value) is not int for value in row)
            for row in unit_coordinates
        )
    ):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.invalid_group",
            message="character group has malformed authored fields",
        )
    if group.character_count != len(expected_units):
        raise OperationDomainValidationError(
            location=("group", "character_count"),
            code="dirichlet_character.group.character_count_mismatch",
            message="character count must equal phi(modulus)",
        )
    expected_blocks = _canonical_blocks(group.modulus)
    expected_orders = tuple(block[2] for block in expected_blocks)
    if tuple(group.generator_orders) != expected_orders:
        raise OperationDomainValidationError(
            location=("group", "generator_orders"),
            code="dirichlet_character.group.generator_order_mismatch",
            message="generator orders must match the canonical prime-power blocks",
        )
    expected_generators = tuple(
        _crt_lift(group.modulus, block_modulus, generator)
        for block_modulus, generator, _ in expected_blocks
    )
    if tuple(group.generators) != expected_generators:
        raise OperationDomainValidationError(
            location=("group", "generators"),
            code="dirichlet_character.group.generator_mismatch",
            message="generators must be the canonical CRT lifts of block generators",
        )
    if tuple(group.invariant_factors) != _invariant_factors(expected_orders):
        raise OperationDomainValidationError(
            location=("group", "invariant_factors"),
            code="dirichlet_character.group.invariant_factor_mismatch",
            message="invariant factors must be the canonical divisibility chain",
        )
    expected_exponent = math.lcm(*expected_orders) if expected_orders else 1
    if group.exponent != expected_exponent:
        raise OperationDomainValidationError(
            location=("group", "exponent"),
            code="dirichlet_character.group.exponent_mismatch",
            message="the common exponent must be the least common multiple of orders",
        )
    _require_generator_coordinate_round_trip(group)
    return group


def _require_generator_coordinate_round_trip(group: DirichletCharacterGroup) -> None:
    """Replay the generator-coordinate identity on every claimed unit."""

    unit_index = dict(zip(group.unit_residues, group.unit_coordinates, strict=True))
    for residue, row in zip(group.unit_residues, group.unit_coordinates, strict=True):
        rebuilt = 1 % group.modulus
        for generator, coordinate, _order in zip(
            group.generators, row, group.generator_orders, strict=True
        ):
            rebuilt = (
                rebuilt * pow(generator, coordinate, group.modulus)
            ) % group.modulus
        if rebuilt != residue or unit_index[residue] != row:
            raise OperationDomainValidationError(
                location=("group", "unit_coordinates"),
                code="dirichlet_character.group.coordinate_mismatch",
                message="generator coordinates must rebuild every unit residue",
            )


def character_group(modulus: int) -> DirichletCharacterGroup:
    """Return the finite unit-group decomposition and character coordinates."""

    _admit_character_group(modulus)
    return _character_group_from_admitted(modulus)


def _character_group_from_admitted(modulus: int) -> DirichletCharacterGroup:
    """Construct a canonical group after its modulus admission has passed."""

    if modulus == 1:
        return DirichletCharacterGroup._from_kernel(
            modulus=1,
            unit_residues=(0,),
            character_count=1,
            invariant_factors=(),
            generators=(),
            generator_orders=(),
            unit_coordinates=(((),)),
            exponent=1,
        )
    units = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    blocks = _canonical_blocks(modulus)
    generators = tuple(
        _crt_lift(modulus, block_modulus, generator)
        for block_modulus, generator, _ in blocks
    )
    orders = tuple(block[2] for block in blocks)
    for generator, order in zip(generators, orders, strict=True):
        if math.gcd(generator, modulus) != 1:
            raise RuntimeError("lifted generator is not a unit")
        if _multiplicative_order(generator, order, modulus) != order:
            raise RuntimeError("lifted generator order failed its defining check")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for block_modulus, block_generator, order in blocks:
        grouped.setdefault(block_modulus, []).append((block_generator, order))
    tables = {}
    for block_modulus, generators_orders in grouped.items():
        table = _component_log_table(block_modulus, tuple(generators_orders))
        if len(table) != _euler_phi(block_modulus):
            raise RuntimeError("block generators failed their completeness check")
        tables[block_modulus] = table
    coordinates = tuple(
        tuple(
            coordinate
            for block_modulus in grouped
            for coordinate in tables[block_modulus][residue % block_modulus]
        )
        for residue in units
    )
    group = DirichletCharacterGroup._from_kernel(
        modulus=modulus,
        unit_residues=units,
        character_count=len(units),
        invariant_factors=_invariant_factors(orders),
        generators=generators,
        generator_orders=orders,
        unit_coordinates=coordinates,
        exponent=math.lcm(*orders) if orders else 1,
    )
    _require_generator_coordinate_round_trip(group)
    if group.character_count != _euler_phi(modulus):
        raise RuntimeError("character count failed its phi(modulus) identity")
    return group


def dirichlet_character_group_enumerate(
    group: DirichletCharacterGroup,
) -> DirichletCharacterFamily:
    """Return every dual coordinate once, retaining the group parent once."""
    group = require_complete_character_group(group)
    count = group.character_count
    rank = len(group.generator_orders)
    enumeration_work = count * max(rank, 1)
    total_work_bound = MAX_CHARACTER_GROUP_WORK + enumeration_work
    if (
        enumeration_work > MAX_CHARACTER_ENUMERATION_WORK
        or total_work_bound > MAX_CHARACTER_ENUMERATION_TOTAL_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("group",),
            code="dirichlet_character.enumeration_work_bound",
            message="complete dual character enumeration exceeds the admitted work envelope",
        )

    coordinate_cells = count * rank
    coordinate_digits = count * sum(
        len(format_canonical_integer(max(order - 1, 0)))
        for order in group.generator_orders
    )
    _admit_result_allocation(
        cells=coordinate_cells,
        digits=coordinate_digits,
        location=("group",),
        code="dirichlet_character.enumeration_allocation_bound",
        message="complete dual character family exceeds its admitted coordinate allocation",
    )

    coordinates = tuple(product(*(range(order) for order in group.generator_orders)))
    if len(coordinates) != count:
        raise RuntimeError("dual coordinate product failed its group-order identity")
    return DirichletCharacterFamily(group=group, coordinates=coordinates)


def dirichlet_character_fourier_matrix(
    group: DirichletCharacterGroup,
) -> DirichletCharacterFourierMatrix:
    """Return the exact complete character table over the canonical unit axis.

    Each entry is the exponent of the group's common primitive root. Complete
    work and exact result allocation are admitted before constructing rows.
    """

    group = require_complete_character_group(group)
    count = group.character_count
    rank = len(group.generator_orders)
    work = count * count * max(rank, 1)
    if work > MAX_CHARACTER_ORTHOGONALITY_WORK:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="dirichlet_character.fourier.work_bound",
            message="complete character Fourier matrix exceeds the admitted work envelope",
        )

    matrix_cells = count * count + count * rank
    exponent_digits = len(format_canonical_integer(max(group.exponent - 1, 0)))
    coordinate_digits = count * sum(
        len(format_canonical_integer(max(axis_order - 1, 0)))
        for axis_order in group.generator_orders
    )
    matrix_digits = count * count * exponent_digits + coordinate_digits
    _admit_result_allocation(
        cells=matrix_cells,
        digits=matrix_digits,
        location=("group",),
        code="dirichlet_character.fourier_allocation_bound",
        message="complete character Fourier matrix exceeds its admitted exact allocation",
    )

    character_coordinates = tuple(
        product(*(range(order) for order in group.generator_orders))
    )
    common_order = group.exponent
    entries = tuple(
        tuple(
            sum(
                character_coordinate * unit_coordinate * (common_order // axis_order)
                for character_coordinate, unit_coordinate, axis_order in zip(
                    character, unit_row, group.generator_orders, strict=True
                )
            )
            % common_order
            for unit_row in group.unit_coordinates
        )
        for character in character_coordinates
    )
    if len(character_coordinates) != count:
        raise RuntimeError("dual coordinate product failed its group-order identity")
    return DirichletCharacterFourierMatrix._from_kernel(
        group=group,
        character_coordinates=character_coordinates,
        unit_residues=group.unit_residues,
        cyclotomic_order=common_order,
        entries=entries,
    )


def dirichlet_character_residue_indicator_expansion(
    group: DirichletCharacterGroup, residue: int
) -> DirichletCharacterResidueIndicatorExpansion:
    """Expand one unit point indicator by finite character orthogonality.

    On units, ``1_{x=a} = h^-1 sum_chi chi(x) conjugate(chi(a))``.
    Nonunit residues are outside this indicator's domain.
    """

    group = require_complete_character_group(group)
    if type(residue) is not int or not 0 <= residue < group.modulus:
        raise OperationDomainValidationError(
            location=("residue",),
            code="dirichlet_character.residue_indicator.residue_range",
            message="residue must be the canonical representative of a unit modulo the group modulus",
        )
    try:
        unit_index = group.unit_residues.index(residue)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("residue",),
            code="dirichlet_character.residue_indicator.requires_unit",
            message="residue-class indicator expansion is defined for unit residues",
        ) from exc

    count = group.character_count
    rank = len(group.generator_orders)
    work = group.modulus + count * max(rank, 1)
    if work > MAX_CHARACTER_ORTHOGONALITY_WORK:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="dirichlet_character.residue_indicator.work_bound",
            message="residue-class indicator expansion exceeds the admitted work envelope",
        )
    result_cells = count * (rank + 1)
    result_digits = count * (
        len(format_canonical_integer(max(group.exponent - 1, 0)))
        + sum(
            len(format_canonical_integer(max(axis_order - 1, 0)))
            for axis_order in group.generator_orders
        )
    )
    _admit_result_allocation(
        cells=result_cells,
        digits=result_digits,
        location=("group",),
        code="dirichlet_character.residue_indicator_allocation_bound",
        message="residue-class indicator expansion exceeds its admitted exact allocation",
    )

    target_unit_coordinates = group.unit_coordinates[unit_index]
    character_coordinates = tuple(
        product(*(range(order) for order in group.generator_orders))
    )
    exponents = tuple(
        -sum(
            character_coordinate * unit_coordinate * (group.exponent // axis_order)
            for character_coordinate, unit_coordinate, axis_order in zip(
                character, target_unit_coordinates, group.generator_orders, strict=True
            )
        )
        % group.exponent
        for character in character_coordinates
    )
    return DirichletCharacterResidueIndicatorExpansion(
        group=group,
        residue=residue,
        character_coordinates=character_coordinates,
        coefficient_exponents=exponents,
        cyclotomic_order=group.exponent,
        denominator=count,
    )


def dirichlet_character_inflate(
    character: DirichletCharacter, target_modulus: int
) -> DirichletCharacterInflation:
    """Induce a character along reduction of unit groups.

    On target units the value is the source character at the canonical
    reduction modulo the source modulus. The induced Dirichlet character is
    zero on target nonunits.
    """

    character = _require_character(character)
    if type(target_modulus) is not int:
        raise OperationDomainValidationError(
            location=("target_modulus",),
            code="dirichlet_character.inflation.modulus_type",
            message="target modulus must be an exact integer",
        )
    source_group = character.group
    source_modulus = source_group.modulus
    if target_modulus % source_modulus:
        raise OperationDomainValidationError(
            location=("target_modulus",),
            code="dirichlet_character.inflation.modulus_not_multiple",
            message="target modulus must be a multiple of the source modulus",
        )

    _admit_character_group(target_modulus)
    source_rank = len(source_group.generator_orders)
    target_units_bound = target_modulus
    target_rank_bound = target_modulus.bit_length()
    work_bound = target_units_bound * max(1, source_rank) + target_rank_bound * max(
        1, source_rank
    )
    if work_bound > MAX_CHARACTER_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("target_modulus",),
            code="dirichlet_character.inflation.work_bound",
            message="inflation exceeds its admitted coordinate-map work bound",
        )
    result_cells = target_rank_bound + 2 * target_units_bound
    target_digits = len(format_canonical_integer(max(target_modulus - 1, 0)))
    result_digits = target_rank_bound * target_digits + target_units_bound * (
        target_digits + len(format_canonical_integer(source_modulus - 1))
    )
    _admit_result_allocation(
        cells=result_cells,
        digits=result_digits,
        location=("target_modulus",),
        code="dirichlet_character.inflation_allocation_bound",
        message="inflation exceeds its admitted unit-map allocation",
    )

    target_group = _character_group_from_admitted(target_modulus)
    source_exponent = source_group.exponent
    target_exponent = target_group.exponent
    if target_exponent % source_exponent:
        raise RuntimeError(
            "surjective unit reduction must make the source exponent divide the target exponent"
        )
    target_units = target_group.unit_residues

    source_coordinates_by_residue = dict(
        zip(source_group.unit_residues, source_group.unit_coordinates, strict=True)
    )
    root_embedding = target_exponent // source_exponent
    target_coordinates: list[int] = []
    for generator, target_axis_order in zip(
        target_group.generators, target_group.generator_orders, strict=True
    ):
        source_residue = generator % source_modulus
        source_unit_coordinates = source_coordinates_by_residue.get(source_residue)
        if source_unit_coordinates is None:
            raise RuntimeError("target unit generator did not reduce to a source unit")
        source_value_exponent = (
            sum(
                character_coordinate
                * (source_exponent // source_axis_order)
                * unit_coordinate
                for character_coordinate, source_axis_order, unit_coordinate in zip(
                    character.coordinates,
                    source_group.generator_orders,
                    source_unit_coordinates,
                    strict=True,
                )
            )
            % source_exponent
        )
        target_value_exponent = source_value_exponent * root_embedding
        target_axis_step = target_exponent // target_axis_order
        if target_value_exponent % target_axis_step:
            raise RuntimeError(
                "inflated generator value is incompatible with its target dual axis"
            )
        target_coordinates.append(
            (target_value_exponent // target_axis_step) % target_axis_order
        )

    source_units = tuple(residue % source_modulus for residue in target_units)
    target_character = DirichletCharacter.model_construct(
        group=target_group, coordinates=tuple(target_coordinates)
    )
    return DirichletCharacterInflation._from_kernel(
        source=character,
        target=target_character,
        target_unit_residues=target_units,
        source_unit_residues=source_units,
    )


def _admit_character_restriction(source_modulus: int, target_modulus: int) -> None:
    """Admit both groups, complete fiber work, and largest output from moduli."""

    source_units = source_modulus
    source_rank = source_modulus.bit_length()
    target_units = target_modulus
    target_rank = target_modulus.bit_length()
    work_bound = (
        3 * MAX_CHARACTER_GROUP_WORK
        + 2 * source_units * max(1, source_rank)
        + 2 * target_units * max(1, target_rank)
        + source_units * max(1, source_rank)
        + source_units
        + target_units
        + target_rank * max(1, source_rank)
    )
    if work_bound > MAX_CHARACTER_RESTRICTION_WORK:
        raise OperationResourceAdmissionError(
            location=("target_modulus",),
            code="dirichlet_character.restriction.work_bound",
            message="factor-down unit mapping exceeds its admitted work bound",
        )

    result_cells = 2 * target_units + target_rank
    target_digits = len(format_canonical_integer(max(target_modulus - 1, 0)))
    source_digits = len(format_canonical_integer(max(source_modulus - 1, 0)))
    result_digits = (
        target_units * (target_digits + source_digits)
        + target_rank * target_digits
        + 4 * source_digits
        + target_digits
    )
    _admit_result_allocation(
        cells=result_cells,
        digits=result_digits,
        location=("target_modulus",),
        code="dirichlet_character.restriction_allocation_bound",
        message="factor-down result exceeds its admitted unit-map allocation",
    )


def _first_restriction_fiber_obstruction(
    target_group: DirichletCharacterGroup,
    fibers: dict[int, list[tuple[int, CyclotomicValue]]],
) -> DirichletCharacterRestrictionObstruction | None:
    """Return the first canonical same-fiber pair with unequal values."""

    for target_residue in target_group.unit_residues:
        fiber = fibers[target_residue]
        if not fiber:
            raise RuntimeError("unit reduction from source to target is not surjective")
        first_residue, first_value = fiber[0]
        differing = next(
            ((residue, value) for residue, value in fiber[1:] if value != first_value),
            None,
        )
        if differing is not None:
            second_residue, second_value = differing
            return DirichletCharacterRestrictionObstruction.model_construct(
                target_group=target_group,
                target_unit_residue=target_residue,
                source_unit_residues=(first_residue, second_residue),
                values=(first_value, second_value),
            )
    return None


def _restriction_source_modulus(
    character: DirichletCharacter, target_modulus: int
) -> int:
    """Validate native source/target tags and return the source modulus."""

    if not isinstance(character, DirichletCharacter):
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.character_type",
            message="character must be a Dirichlet character value",
        )
    if type(target_modulus) is not int:
        raise OperationDomainValidationError(
            location=("target_modulus",),
            code="dirichlet_character.restriction.modulus_type",
            message="target modulus must be an exact integer",
        )
    source_group = getattr(character, "group", None)
    source_modulus = getattr(source_group, "modulus", None)
    if type(source_modulus) is not int:
        raise OperationDomainValidationError(
            location=("character", "group", "modulus"),
            code="dirichlet_character.group.invalid_group",
            message="source character must carry a bounded canonical group modulus",
        )
    if source_modulus < 1:
        raise OperationDomainValidationError(
            location=("character", "group", "modulus"),
            code="dirichlet_character.group.modulus_sign",
            message="source modulus must be positive",
        )
    if source_modulus > MAX_CHARACTER_GROUP_MODULUS:
        raise OperationResourceAdmissionError(
            location=("character", "group", "modulus"),
            code="dirichlet_character.group.modulus_bound",
            message="source modulus exceeds the 2,048 character-group bound",
        )
    if not 1 <= target_modulus <= MAX_CHARACTER_GROUP_MODULUS:
        raise OperationResourceAdmissionError(
            location=("target_modulus",),
            code="dirichlet_character.group.modulus_bound",
            message="target modulus exceeds the 2,048 character-group bound",
        )
    if source_modulus % target_modulus:
        raise OperationDomainValidationError(
            location=("target_modulus",),
            code="dirichlet_character.restriction.modulus_not_divisor",
            message="target modulus must be a positive divisor of the source modulus",
        )
    return source_modulus


def dirichlet_character_restrict_modulus(
    character: DirichletCharacter, target_modulus: int
) -> DirichletCharacterRestrictionResult:
    """Factor a character through reduction to a divisor modulus when possible.

    A character descends exactly when it is constant on every fiber of the
    surjection from source units to target units. Nonfactorization returns two
    units in one fiber with their distinct exact roots of unity.
    """

    source_modulus = _restriction_source_modulus(character, target_modulus)
    # Preflight from only moduli before authenticating source coordinates or
    # constructing the target unit group and its map.
    _admit_character_restriction(source_modulus, target_modulus)

    character = _require_character(character)
    source_group = character.group
    source_units = source_group.unit_residues
    target_group = character_group(target_modulus)
    target_units = target_group.unit_residues
    exponent = source_group.exponent
    values_by_source_residue: dict[int, CyclotomicValue] = {}
    fibers: dict[int, list[tuple[int, CyclotomicValue]]] = {
        residue: [] for residue in target_units
    }
    for residue, unit_coordinates in zip(
        source_units, source_group.unit_coordinates, strict=True
    ):
        value_exponent = (
            sum(
                character_coordinate * (exponent // axis_order) * unit_coordinate
                for character_coordinate, axis_order, unit_coordinate in zip(
                    character.coordinates,
                    source_group.generator_orders,
                    unit_coordinates,
                    strict=True,
                )
            )
            % exponent
        )
        value = CyclotomicValue(order=exponent, exponent=value_exponent)
        values_by_source_residue[residue] = value
        target_residue = residue % target_modulus
        bucket = fibers.get(target_residue)
        if bucket is None:
            raise RuntimeError("unit reduction did not land in the target unit group")
        bucket.append((residue, value))

    obstruction = _first_restriction_fiber_obstruction(target_group, fibers)
    if obstruction is not None:
        return DirichletCharacterRestrictionResult._obstructed_from_kernel(
            source=character,
            target_modulus=target_modulus,
            obstruction=obstruction,
        )

    source_lifts = tuple(fibers[residue][0][0] for residue in target_units)
    source_to_target_root_scale = exponent // target_group.exponent
    if exponent % target_group.exponent:
        raise RuntimeError(
            "surjective unit reduction must make the target exponent divide the source exponent"
        )
    target_coordinates: list[int] = []
    for generator, target_axis_order in zip(
        target_group.generators, target_group.generator_orders, strict=True
    ):
        fiber = fibers.get(generator)
        if not fiber:
            raise RuntimeError("target generator has no source-unit lift")
        lift = fiber[0][0]
        source_value = values_by_source_residue[lift]
        if source_value.exponent % source_to_target_root_scale:
            raise RuntimeError(
                "factored character value does not descend into the target root group"
            )
        target_value_exponent = (
            source_value.exponent // source_to_target_root_scale
        ) % target_group.exponent
        target_axis_step = target_group.exponent // target_axis_order
        if target_value_exponent % target_axis_step:
            raise RuntimeError(
                "descended generator value is incompatible with its target dual axis"
            )
        target_coordinates.append(
            (target_value_exponent // target_axis_step) % target_axis_order
        )

    target_character = DirichletCharacter.model_construct(
        group=target_group, coordinates=tuple(target_coordinates)
    )
    return DirichletCharacterRestrictionResult._descended_from_kernel(
        source=character,
        target_modulus=target_modulus,
        target=target_character,
        target_unit_residues=target_units,
        source_unit_lifts=source_lifts,
    )


def _character_value_order(character: DirichletCharacter) -> int:
    order = 1
    for coordinate, generator_order in zip(
        character.coordinates, character.group.generator_orders, strict=True
    ):
        order = math.lcm(
            order, generator_order // math.gcd(coordinate, generator_order)
        )
    return order


def _generalized_bernoulli_cyclotomic_polynomial(
    order: int, degree: int
) -> tuple[int, ...]:
    """Build one already admitted defining polynomial in ascending order."""

    variable = symbols("x")
    coefficients = tuple(
        int(coefficient)
        for coefficient in Poly(cyclotomic_poly(order, variable), variable).all_coeffs()
    )
    if len(coefficients) != degree + 1 or coefficients[0] != 1:
        raise RuntimeError("cyclotomic polynomial has an unexpected canonical shape")
    return tuple(reversed(coefficients))


def _admit_generalized_bernoulli_coefficient_growth(
    *,
    index: int,
    modulus: int,
    order: int,
    degree: int,
    cyclotomic_coefficients: tuple[int, ...],
    error_scope: str = "generalized_bernoulli",
    denominator_multiplier: int = 1,
) -> None:
    """Prove common numerator and denominator bounds before the character sum."""

    # B_j has denominator dividing (j+1)! by von Staudt-Clausen and
    # |B_j| <= 2*j!. Hence B_k(x)'s coefficient l1 norm is < 6*k! on [0,1].
    denominator_bound = (
        modulus * math.factorial(index + 1) if index else modulus
    ) * denominator_multiplier
    scalar_limit = 10**MAX_CYCLIC_FIELD_ELEMENT_DIGITS - 1
    if denominator_bound > scalar_limit:
        raise OperationResourceAdmissionError(
            location=("index",),
            code=f"dirichlet_character.{error_scope}.coefficient_bound",
            message="generalized Bernoulli denominator exceeds the exact coefficient bound",
        )
    if index == 0:
        coefficient_bound = modulus
    else:
        coefficient_bound = 6 * math.factorial(index) * math.factorial(index + 1)
        for _ in range(index + 1):
            if coefficient_bound > scalar_limit // modulus:
                raise OperationResourceAdmissionError(
                    location=("index",),
                    code=f"dirichlet_character.{error_scope}.coefficient_bound",
                    message="generalized Bernoulli numerator may exceed the exact coefficient bound",
                )
            coefficient_bound *= modulus
    reduction_norm = sum(abs(value) for value in cyclotomic_coefficients[:-1])
    for _ in range(max(0, order - degree)):
        if coefficient_bound > scalar_limit // reduction_norm:
            raise OperationResourceAdmissionError(
                location=("index",),
                code=f"dirichlet_character.{error_scope}.coefficient_bound",
                message="cyclotomic reduction may exceed the exact coefficient bound",
            )
        coefficient_bound *= reduction_norm


def _bernoulli_polynomial_coefficients(index: int) -> tuple[Fraction, ...]:
    bernoulli_values = []
    for bernoulli_index in range(index + 1):
        if bernoulli_index == 1:
            # SymPy stores B_1=+1/2; this polynomial convention uses -1/2.
            bernoulli_values.append(Fraction(-1, 2))
            continue
        value = bernoulli(bernoulli_index)
        numerator, denominator = value.as_numer_denom()
        bernoulli_values.append(Fraction(int(numerator), int(denominator)))
    return tuple(
        math.comb(index, power) * bernoulli_values[index - power]
        for power in range(index + 1)
    )


def _character_root_exponent(
    character: DirichletCharacter,
    row: tuple[int, ...],
    value_order: int,
) -> int:
    common_order = character.group.exponent
    exponent = (
        sum(
            coordinate * (common_order // axis_order) * unit_coordinate
            for coordinate, axis_order, unit_coordinate in zip(
                character.coordinates,
                character.group.generator_orders,
                row,
                strict=True,
            )
        )
        % common_order
    )
    value_scale = common_order // value_order
    if exponent % value_scale:
        raise RuntimeError("character value did not embed in its canonical value field")
    return exponent // value_scale


def _evaluate_generalized_bernoulli(
    character: DirichletCharacter,
    index: int,
    value_order: int,
    degree: int,
    cyclotomic_coefficients: tuple[int, ...],
) -> RationalCyclotomicElement:
    """Evaluate admitted rational Bernoulli weights and reduce in QQ[zeta_d]."""

    group = character.group
    modulus = group.modulus
    polynomial_coefficients = _bernoulli_polynomial_coefficients(index)
    exponent_by_residue = dict(
        zip(group.unit_residues, group.unit_coordinates, strict=True)
    )
    values = [Fraction(0)] * value_order
    scale = Fraction(1, modulus) if index == 0 else Fraction(modulus ** (index - 1))
    for integer in range(1, modulus + 1):
        row = exponent_by_residue.get(integer % modulus)
        if row is None:
            continue
        rational_argument = Fraction(integer, modulus)
        bernoulli_value = Fraction(0)
        for coefficient in reversed(polynomial_coefficients):
            bernoulli_value = bernoulli_value * rational_argument + coefficient
        root_exponent = _character_root_exponent(character, row, value_order)
        values[root_exponent] += scale * bernoulli_value

    for power in range(value_order - 1, degree - 1, -1):
        coefficient = values[power]
        if coefficient:
            values[power] = Fraction(0)
            for lower_power in range(degree):
                values[power - degree + lower_power] -= (
                    coefficient * cyclotomic_coefficients[lower_power]
                )
    return RationalCyclotomicElement(
        field=RationalCyclotomicField(order=value_order),
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(coefficient)
            for coefficient in values[:degree]
        ),
    )


def _dirichlet_character_generalized_bernoulli(
    character: DirichletCharacter,
    index: int,
    *,
    denominator_multiplier: int = 1,
) -> DirichletCharacterGeneralizedBernoulliResult:
    r"""Return ``N^(k-1) sum_{a=1}^N chi(a) B_k(a/N)`` exactly.

    The Bernoulli convention is generated by ``t*exp(x*t)/(exp(t)-1)``;
    hence ``B_1(x)=x-1/2`` and ``B_1(1)=+1/2``.
    """

    if type(index) is not int or index < 0:
        raise OperationDomainValidationError(
            location=("index",),
            code="dirichlet_character.generalized_bernoulli.index_domain",
            message="generalized Bernoulli index must be a nonnegative integer",
        )
    if index > MAX_GENERALIZED_BERNOULLI_INDEX:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="dirichlet_character.generalized_bernoulli.index_bound",
            message=(
                "generalized Bernoulli index exceeds the admitted bound "
                f"{MAX_GENERALIZED_BERNOULLI_INDEX}"
            ),
        )

    character = _require_character(character)
    group = character.group
    modulus = group.modulus
    character_order = _character_value_order(character)
    if character_order > MAX_CYCLIC_PERIOD:
        raise OperationResourceAdmissionError(
            location=("character", "group", "exponent"),
            code="dirichlet_character.generalized_bernoulli.field_order_bound",
            message=(
                "generalized Bernoulli value field order exceeds the exact "
                f"cyclotomic bound {MAX_CYCLIC_PERIOD}"
            ),
        )

    degree = _euler_phi(character_order)
    rank = len(group.generator_orders)
    work = (
        modulus * (index + rank + 1)
        + character_order * degree
        + character_order * character_order
        + (index + 1) * (index + 1)
    )
    if work > MAX_GENERALIZED_BERNOULLI_WORK or work > MAX_CYCLIC_FIELD_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="dirichlet_character.generalized_bernoulli.work_bound",
            message="generalized Bernoulli evaluation exceeds its admitted exact work envelope",
        )
    _admit_result_allocation(
        cells=2 * degree,
        digits=2 * degree * MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
        location=("character", "group"),
        code="dirichlet_character.generalized_bernoulli.result_allocation_bound",
        message="generalized Bernoulli value exceeds its admitted coefficient allocation",
    )

    # The defining polynomial has degree at most 128, coefficients bounded by
    # 2**degree (Vieta's formula for unit-circle roots), and construction work
    # bounded by character_order*degree. Admit it before materializing Phi.
    cyclotomic_coefficients = _generalized_bernoulli_cyclotomic_polynomial(
        character_order, degree
    )
    _admit_generalized_bernoulli_coefficient_growth(
        index=index,
        modulus=modulus,
        order=character_order,
        degree=degree,
        cyclotomic_coefficients=cyclotomic_coefficients,
        denominator_multiplier=denominator_multiplier,
    )
    result_value = _evaluate_generalized_bernoulli(
        character,
        index,
        character_order,
        degree,
        cyclotomic_coefficients,
    )
    return DirichletCharacterGeneralizedBernoulliResult._from_kernel(
        character=character,
        index=index,
        value=result_value,
    )


def dirichlet_character_generalized_bernoulli(
    character: DirichletCharacter, index: int
) -> DirichletCharacterGeneralizedBernoulliResult:
    """Return one exact generalized Bernoulli number."""

    return _dirichlet_character_generalized_bernoulli(character, index)


def dirichlet_character_l_value_nonpositive_integer(
    character: DirichletCharacter, bernoulli_index: int
) -> DirichletCharacterLValueNonpositiveResult:
    r"""Return ``L(1-k, chi)=-B(k, chi)/k`` for bounded ``k >= 1``."""

    if type(bernoulli_index) is not int or bernoulli_index < 1:
        raise OperationDomainValidationError(
            location=("bernoulli_index",),
            code="dirichlet_character.l_value.index_domain",
            message="Bernoulli index for a nonpositive L-value must be positive",
        )
    if bernoulli_index > MAX_GENERALIZED_BERNOULLI_INDEX:
        raise OperationResourceAdmissionError(
            location=("bernoulli_index",),
            code="dirichlet_character.l_value.index_bound",
            message=(
                "Bernoulli index exceeds the admitted bound "
                f"{MAX_GENERALIZED_BERNOULLI_INDEX}"
            ),
        )
    bernoulli = _dirichlet_character_generalized_bernoulli(
        character,
        bernoulli_index,
        denominator_multiplier=bernoulli_index,
    )
    value = RationalCyclotomicElement(
        field=bernoulli.value.field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(
                -Fraction(coefficient.num, coefficient.den) / bernoulli_index
            )
            for coefficient in bernoulli.value.coefficients_ascending
        ),
    )
    return DirichletCharacterLValueNonpositiveResult._from_kernel(
        character=character,
        bernoulli_index=bernoulli_index,
        value=value,
    )


def dirichlet_character_generalized_bernoulli_prefix(
    character: DirichletCharacter, maximum_index: int
) -> DirichletCharacterGeneralizedBernoulliPrefix:
    """Return the exact generalized Bernoulli prefix B(0,chi)..B(k,chi)."""

    if type(maximum_index) is not int or maximum_index < 0:
        raise OperationDomainValidationError(
            location=("maximum_index",),
            code="dirichlet_character.generalized_bernoulli_prefix.index_domain",
            message="maximum_index must be a nonnegative integer",
        )
    if maximum_index > MAX_GENERALIZED_BERNOULLI_INDEX:
        raise OperationResourceAdmissionError(
            location=("maximum_index",),
            code="dirichlet_character.generalized_bernoulli_prefix.index_bound",
            message=(
                "maximum_index exceeds the admitted bound "
                f"{MAX_GENERALIZED_BERNOULLI_INDEX}"
            ),
        )

    character = _require_character(character)
    modulus = character.group.modulus
    value_order = _character_value_order(character)
    if value_order > MAX_CYCLIC_PERIOD:
        raise OperationResourceAdmissionError(
            location=("character", "group", "exponent"),
            code="dirichlet_character.generalized_bernoulli_prefix.field_order_bound",
            message="generalized Bernoulli prefix field order exceeds the exact cyclotomic bound",
        )
    degree = _euler_phi(value_order)
    rank = len(character.group.generator_orders)
    count = maximum_index + 1
    work = sum(
        modulus * (index + rank + 1)
        + value_order * degree
        + value_order * value_order
        + (index + 1) * (index + 1)
        for index in range(count)
    )
    if work > MAX_GENERALIZED_BERNOULLI_WORK or work > MAX_CYCLIC_FIELD_WORK:
        raise OperationResourceAdmissionError(
            location=("maximum_index",),
            code="dirichlet_character.generalized_bernoulli_prefix.work_bound",
            message="generalized Bernoulli prefix exceeds its admitted exact work envelope",
        )
    _admit_result_allocation(
        cells=2 * count * degree,
        digits=2 * count * degree * MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
        location=("character", "group"),
        code="dirichlet_character.generalized_bernoulli_prefix.result_allocation_bound",
        message="generalized Bernoulli prefix exceeds its admitted coefficient allocation",
    )

    cyclotomic_coefficients = _generalized_bernoulli_cyclotomic_polynomial(
        value_order, degree
    )
    # Admit coefficient growth for every member before evaluating any prefix item.
    for index in range(count):
        _admit_generalized_bernoulli_coefficient_growth(
            index=index,
            modulus=modulus,
            order=value_order,
            degree=degree,
            cyclotomic_coefficients=cyclotomic_coefficients,
            error_scope="generalized_bernoulli_prefix",
        )
    values = tuple(
        _evaluate_generalized_bernoulli(
            character,
            index,
            value_order,
            degree,
            cyclotomic_coefficients,
        )
        for index in range(count)
    )
    return DirichletCharacterGeneralizedBernoulliPrefix._from_kernel(
        character=character,
        maximum_index=maximum_index,
        values=values,
    )


def _cyclotomic_root_power_coordinates(
    *, order: int, degree: int, exponent: int, polynomial: tuple[int, ...]
) -> tuple[int, ...]:
    coordinates = [0] * order
    coordinates[exponent] = 1
    for power in range(order - 1, degree - 1, -1):
        coefficient = coordinates[power]
        if coefficient:
            coordinates[power] = 0
            for lower_power in range(degree):
                coordinates[power - degree + lower_power] -= (
                    coefficient * polynomial[lower_power]
                )
    return tuple(coordinates[:degree])


def _sequence_twist_source_coefficients(
    sequence: FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
    value: object,
) -> tuple[Fraction, ...]:
    if isinstance(sequence, FiniteIntegerSequence):
        if type(value) is not int:
            raise OperationDomainValidationError(
                location=("sequence", "values"),
                code="dirichlet_character.sequence_twist.sequence_value",
                message="integer sequence entries must be strict integers",
            )
        return (Fraction(value),)
    if isinstance(sequence, FiniteRationalSequence):
        if not isinstance(value, CanonicalRational):
            raise OperationDomainValidationError(
                location=("sequence", "values"),
                code="dirichlet_character.sequence_twist.sequence_value",
                message="rational sequence entries must be canonical rationals",
            )
        return (value.as_fraction(),)
    if (
        not isinstance(value, RationalCyclotomicElement)
        or value.field != sequence.field
    ):
        raise OperationDomainValidationError(
            location=("sequence", "values"),
            code="dirichlet_character.sequence_twist.sequence_value",
            message="cyclotomic entries must retain their declared exact field",
        )
    return tuple(
        coefficient.as_fraction() for coefficient in value.coefficients_ascending
    )


def _sequence_twist_input_digit_bounds(
    sequence: FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
) -> tuple[int, int, int]:
    source_digits = 0
    max_numerator_digits = 1
    max_denominator_digits = 1
    for value in sequence.values:
        coefficients = _sequence_twist_source_coefficients(sequence, value)
        if isinstance(value, RationalCyclotomicElement):
            coordinates = value.coefficients_ascending
        else:
            coordinates = (CanonicalRational.from_fraction(coefficients[0]),)
        for coordinate in coordinates:
            numerator_digits = len(format_canonical_integer(abs(coordinate.num)))
            denominator_digits = len(format_canonical_integer(coordinate.den))
            source_digits += numerator_digits + denominator_digits
            max_numerator_digits = max(max_numerator_digits, numerator_digits)
            max_denominator_digits = max(max_denominator_digits, denominator_digits)
    if source_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="dirichlet_character.sequence_twist.input_digits_bound",
            message="source sequence exceeds the shared exact digit envelope",
        )
    return source_digits, max_numerator_digits, max_denominator_digits


def _admit_sequence_twist_source(
    sequence: FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
    index_origin: int | None,
) -> tuple[
    FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence, int
]:
    if not isinstance(
        sequence,
        (FiniteIntegerSequence, FiniteRationalSequence, FiniteCyclotomicSequence),
    ):
        raise OperationDomainValidationError(
            location=("sequence",),
            code="dirichlet_character.sequence_twist.sequence_type",
            message="source must be a canonical finite exact sequence",
        )
    if type(sequence.values) is not tuple or len(sequence.values) > MAX_SEQUENCE_LENGTH:
        raise OperationDomainValidationError(
            location=("sequence", "values"),
            code="dirichlet_character.sequence_twist.sequence_shape",
            message="source sequence values are malformed",
        )
    if index_origin is not None and (
        type(index_origin) is not int or not -(2**31) <= index_origin <= 2**31 - 1
    ):
        raise OperationDomainValidationError(
            location=("index_origin",),
            code="dirichlet_character.sequence_twist.index_origin",
            message="index origin must be a signed 32-bit integer",
        )
    if isinstance(sequence, FiniteCyclotomicSequence):
        if index_origin is not None and index_origin != sequence.index_origin:
            raise OperationDomainValidationError(
                location=("index_origin",),
                code="dirichlet_character.sequence_twist.index_origin_mismatch",
                message="an existing cyclotomic sequence keeps its authored index origin",
            )
        index_origin = sequence.index_origin
    if index_origin is None:
        raise OperationDomainValidationError(
            location=("index_origin",),
            code="dirichlet_character.sequence_twist.index_origin_required",
            message="integer and rational sequences require an explicit index origin",
        )
    return sequence, index_origin


def _admit_sequence_twist_field(
    sequence: FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
    character: DirichletCharacter,
) -> tuple[int, int, int]:
    value_order = _character_value_order(character)
    source_order = (
        sequence.field.order if isinstance(sequence, FiniteCyclotomicSequence) else 1
    )
    target_order = math.lcm(source_order, value_order)
    if target_order > MAX_CYCLIC_PERIOD:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="dirichlet_character.sequence_twist.field_order_bound",
            message=(
                "twisted sequence cyclotomic field order exceeds the exact field "
                f"bound {MAX_CYCLIC_PERIOD}"
            ),
        )
    return target_order, value_order, _euler_phi(target_order)


def dirichlet_character_sequence_twist(
    sequence: FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
    character: DirichletCharacter,
    *,
    index_origin: int | None = None,
) -> FiniteCyclotomicSequence:
    r"""Return ``chi(n) a_n`` in the minimal exact cyclotomic coefficient field."""
    sequence, index_origin = _admit_sequence_twist_source(sequence, index_origin)
    character = _require_character(character)
    target_order, value_order, degree = _admit_sequence_twist_field(sequence, character)
    values = sequence.values
    _, max_numerator_digits, max_denominator_digits = (
        _sequence_twist_input_digit_bounds(sequence)
    )
    source_degree = (
        sequence.field.degree if isinstance(sequence, FiniteCyclotomicSequence) else 1
    )

    coefficient_cells = len(values) * degree
    if coefficient_cells > MAX_CHARACTER_SEQUENCE_TWIST_COEFFICIENTS:
        raise OperationResourceAdmissionError(
            location=("sequence", "values"),
            code="dirichlet_character.sequence_twist.coefficient_cells_bound",
            message="twisted sequence exceeds its admitted cyclotomic coefficient-cell bound",
        )
    work = (
        len(values)
        * (source_degree * degree + degree + len(character.group.generator_orders) + 1)
        + character.group.character_count
        + value_order * degree
        + value_order * value_order
    )
    if work > MAX_CHARACTER_SEQUENCE_TWIST_WORK or work > MAX_CYCLIC_FIELD_WORK:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="dirichlet_character.sequence_twist.work_bound",
            message="twist exceeds its admitted exact lookup and coefficient work",
        )
    # The field polynomial has order <= 128 and degree <= 128; its construction
    # is therefore admitted before materializing the exact defining polynomial.
    polynomial = _generalized_bernoulli_cyclotomic_polynomial(target_order, degree)
    reduction_norm = max(1, sum(abs(value) for value in polynomial[:-1]))
    reductions = max(0, target_order - degree)
    growth_digits = reductions * len(str(reduction_norm)) if values else 0
    max_output_digits = (
        max_numerator_digits
        + growth_digits
        + source_degree * max_denominator_digits
        + len(str(source_degree))
        + 2
    )
    max_output_denominator_digits = source_degree * max_denominator_digits
    if (
        max_output_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        or max_output_denominator_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("character",),
            code="dirichlet_character.sequence_twist.coefficient_growth_bound",
            message="twist may exceed the exact cyclotomic coefficient digit bound",
        )
    total_output_digits = coefficient_cells * (
        max_output_digits + max_output_denominator_digits
    )
    if total_output_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="dirichlet_character.sequence_twist.result_digits_bound",
            message="twisted sequence exceeds the shared exact digit envelope",
        )

    group = character.group
    rows_by_residue = dict(
        zip(group.unit_residues, group.unit_coordinates, strict=True)
    )
    common_order = group.exponent
    embedding_scale = common_order // value_order
    character_to_target = target_order // value_order
    source_order = (
        sequence.field.order if isinstance(sequence, FiniteCyclotomicSequence) else 1
    )
    source_to_target = target_order // source_order
    field = RationalCyclotomicField(order=target_order)
    root_coordinates: dict[int, tuple[int, ...]] = {}
    output_values: list[RationalCyclotomicElement] = []
    zero_coordinates = tuple(Fraction(0) for _ in range(degree))
    for offset, value in enumerate(values):
        source_coefficients = _sequence_twist_source_coefficients(sequence, value)
        residue = (index_origin + offset) % group.modulus
        row = rows_by_residue.get(residue)
        if row is None or all(coefficient == 0 for coefficient in source_coefficients):
            twisted = zero_coordinates
        else:
            exponent = (
                sum(
                    coordinate * (common_order // axis_order) * unit_coordinate
                    for coordinate, axis_order, unit_coordinate in zip(
                        character.coordinates,
                        group.generator_orders,
                        row,
                        strict=True,
                    )
                )
                % common_order
            )
            if exponent % embedding_scale:
                raise RuntimeError(
                    "character exponent is outside its exact value field"
                )
            root_exponent = (exponent // embedding_scale) * character_to_target
            twisted_coordinates = [Fraction(0) for _ in range(degree)]
            for source_power, coefficient in enumerate(source_coefficients):
                if coefficient == 0:
                    continue
                power = (source_power * source_to_target + root_exponent) % target_order
                root = root_coordinates.get(power)
                if root is None:
                    root = _cyclotomic_root_power_coordinates(
                        order=target_order,
                        degree=degree,
                        exponent=power,
                        polynomial=polynomial,
                    )
                    root_coordinates[power] = root
                for coordinate_index, coordinate in enumerate(root):
                    twisted_coordinates[coordinate_index] += coefficient * coordinate
            twisted = tuple(twisted_coordinates)
        output_values.append(
            RationalCyclotomicElement(
                field=field,
                coefficients_ascending=tuple(
                    CanonicalRational.from_fraction(coordinate)
                    for coordinate in twisted
                ),
            )
        )
    return FiniteCyclotomicSequence._from_kernel(
        index_origin=index_origin,
        field=field,
        values=tuple(output_values),
    )


def _compute_generalized_gauss_sum(
    character: DirichletCharacter,
    frequency: int,
    *,
    error_name: str,
) -> DirichletCharacterGeneralizedGaussSumResult:
    r"""Return ``sum_{a mod N} chi(a) zeta_N^(frequency*a)`` exactly.

    Character values and additive roots are embedded in the canonical field
    whose order is their least common multiple. Frequency is reduced modulo N;
    nonunit and zero frequencies are part of the operation's domain.
    """
    character = _require_character(character)
    if type(frequency) is not int:
        raise OperationDomainValidationError(
            location=("frequency",),
            code=f"dirichlet_character.{error_name}.frequency_type",
            message="frequency must be a strict integer",
        )
    _admit_character_integer(
        frequency, type_code=f"dirichlet_character.{error_name}.frequency_type"
    )
    group = character.group
    modulus = group.modulus
    frequency_residue = frequency % modulus
    value_order = _character_value_order(character)
    order = math.lcm(value_order, modulus)
    from jacobian.math.matrices.cyclic_linear._models import MAX_CYCLIC_PERIOD

    if order > MAX_CYCLIC_PERIOD:
        raise OperationResourceAdmissionError(
            location=("character", "group", "modulus"),
            code=f"dirichlet_character.{error_name}.cyclotomic_order_bound",
            message=(
                "Gauss-sum cyclotomic order exceeds the exact value bound "
                f"{MAX_CYCLIC_PERIOD}"
            ),
        )
    degree = _euler_phi(order)
    work = modulus + order * degree
    if work > MAX_CHARACTER_SUM_WORK or work > MAX_CYCLIC_FIELD_WORK:
        raise OperationResourceAdmissionError(
            location=("character", "group", "modulus"),
            code=f"dirichlet_character.{error_name}.work_bound",
            message="Gauss sum exceeds the admitted residue and field work envelope",
        )

    # Phi_n is monic with roots on the unit circle, so its coefficient l1 norm
    # is at most 2**degree. This deliberately conservative recurrence bound is
    # checked before constructing Phi_n or enumerating residues.
    digit_limit = MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    coefficient_bound = modulus
    reduction_l1_bound = 1 << degree
    for _ in range(max(0, order - degree)):
        if coefficient_bound > (10**digit_limit - 1) // reduction_l1_bound:
            raise OperationResourceAdmissionError(
                location=("character",),
                code=f"dirichlet_character.{error_name}.coefficient_bound",
                message=(
                    "Gauss sum may exceed the "
                    f"{digit_limit}-digit exact cyclotomic coefficient bound"
                ),
            )
        coefficient_bound *= reduction_l1_bound
    if coefficient_bound > 10**digit_limit - 1:
        raise OperationResourceAdmissionError(
            location=("character",),
            code=f"dirichlet_character.{error_name}.coefficient_bound",
            message=(
                "Gauss sum may exceed the "
                f"{digit_limit}-digit exact cyclotomic coefficient bound"
            ),
        )

    x = symbols("x")
    phi_descending = tuple(
        int(value) for value in Poly(cyclotomic_poly(order, x), x).all_coeffs()
    )
    if len(phi_descending) != degree + 1 or phi_descending[0] != 1:
        raise RuntimeError("cyclotomic polynomial has an unexpected canonical shape")
    phi_ascending = tuple(reversed(phi_descending))

    exponent_by_residue = dict(
        zip(group.unit_residues, group.unit_coordinates, strict=True)
    )
    values = [0] * order
    group_exponent = group.exponent
    value_embedding = order // value_order
    additive_embedding = order // modulus
    for residue, row in exponent_by_residue.items():
        exponent_in_group = (
            sum(
                coordinate * (group_exponent // generator_order) * unit_coordinate
                for coordinate, generator_order, unit_coordinate in zip(
                    character.coordinates, group.generator_orders, row, strict=True
                )
            )
            % group_exponent
        )
        character_exponent = exponent_in_group // (group_exponent // value_order)
        values[
            (
                character_exponent * value_embedding
                + residue * additive_embedding * frequency_residue
            )
            % order
        ] += 1

    for power in range(order - 1, degree - 1, -1):
        coefficient = values[power]
        if coefficient:
            values[power] = 0
            for lower in range(degree):
                values[power - degree + lower] -= coefficient * phi_ascending[lower]
    value = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=order),
        coefficients_ascending=tuple(
            CanonicalRational.from_integer_ratio(coefficient, 1)
            for coefficient in values[:degree]
        ),
    )
    return DirichletCharacterGeneralizedGaussSumResult(
        character=character,
        frequency=frequency,
        frequency_residue=frequency_residue,
        value=value,
    )


def dirichlet_character_generalized_gauss_sum(
    character: DirichletCharacter,
    frequency: int,
) -> DirichletCharacterGeneralizedGaussSumResult:
    """Compute the exact Gauss sum at any bounded integer frequency."""
    return _compute_generalized_gauss_sum(
        character, frequency, error_name="generalized_gauss_sum"
    )


def dirichlet_character_gauss_sum(
    character: DirichletCharacter,
) -> DirichletCharacterGaussSumResult:
    """Return the ordinary Gauss sum, the frequency-one case."""
    result = _compute_generalized_gauss_sum(character, 1, error_name="gauss_sum")
    return DirichletCharacterGaussSumResult(
        character=result.character, value=result.value
    )


def dirichlet_character_primitive_gauss_norm(
    character: DirichletCharacter,
) -> DirichletCharacterPrimitiveGaussNormResult:
    """Compute |tau(chi)|^2 after deriving primitivity from the source character.

    The complex absolute value is represented exactly as tau(chi) times its
    cyclotomic conjugate. The operation accepts only characters whose computed
    conductor equals their modulus and checks that this product is that modulus.
    """
    character = _require_character(character)
    conductor = dirichlet_character_conductor(character)
    modulus = character.group.modulus
    if conductor.conductor != modulus:
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.primitive_gauss_norm.requires_primitive",
            message="squared Gauss norm theorem requires a primitive character",
        )
    gauss_sum = dirichlet_character_gauss_sum(character).value
    order = gauss_sum.field.order
    x = symbols("x")
    phi = Poly(cyclotomic_poly(order, x), x, domain=QQ)
    degree = phi.degree()
    source = Poly.from_list(
        [
            QQ(value.num, value.den)
            for value in reversed(gauss_sum.coefficients_ascending)
        ],
        gens=x,
        domain=QQ,
    )
    conjugate = Poly.from_dict(
        {
            (0 if power == 0 else order - power,): QQ(value.num, value.den)
            for power, value in enumerate(gauss_sum.coefficients_ascending)
            if value.num
        },
        gens=x,
        domain=QQ,
    ).rem(phi)
    product_value = (source * conjugate).rem(phi)
    coefficients = tuple(product_value.nth(power) for power in range(degree))
    if coefficients != (QQ(modulus),) + (QQ(0),) * (degree - 1):
        raise RuntimeError("primitive Gauss sum failed the exact squared-norm identity")
    norm = RationalCyclotomicElement(
        field=gauss_sum.field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(value)) for value in coefficients
        ),
    )
    return DirichletCharacterPrimitiveGaussNormResult(
        character=character,
        conductor=modulus,
        gauss_sum=gauss_sum,
        norm_squared=norm,
    )
