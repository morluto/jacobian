"""Exact bounded constructors and natural actions for prime-field GL and SL."""

from __future__ import annotations

from itertools import product
from typing import NoReturn, overload

from pydantic import ValidationError

from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.groups.finite_matrix._models import (
    MAX_LINEAR_GROUP_DIMENSION,
    MAX_LINEAR_GROUP_GENERATOR_CELLS,
    MAX_LINEAR_GROUP_ORDER_DIGITS,
    MAX_LINEAR_GROUP_PRIME,
    MAX_NATURAL_VECTOR_ACTION_SIZE,
    PrimeFieldGeneralLinearGroup,
    PrimeFieldGeneralLinearNaturalAction,
    PrimeFieldSpecialLinearGroup,
    PrimeFieldSpecialLinearNaturalAction,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _domain_error(
    location: tuple[str | int, ...], reason: str, message: str
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_matrix_group.{reason}",
        message=message,
    )


def _resource_error(
    location: tuple[str | int, ...], reason: str, message: str
) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"finite_matrix_group.{reason}",
        message=message,
    )


def _admit_parameters(prime: object, dimension: object) -> tuple[int, int]:
    if not isinstance(prime, int) or isinstance(prime, bool):
        _domain_error(("prime",), "prime_type", "prime must be an exact integer")
    if not 2 <= prime <= MAX_LINEAR_GROUP_PRIME:
        _resource_error(
            ("prime",),
            "prime_bound",
            f"prime must not exceed {MAX_LINEAR_GROUP_PRIME}",
        )
    from flint import fmpz

    if not fmpz(prime).is_prime():
        _domain_error(
            ("prime",),
            "characteristic_not_prime",
            "prime must be a prime characteristic",
        )
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        _domain_error(
            ("dimension",), "dimension_type", "dimension must be an exact integer"
        )
    if not 1 <= dimension <= MAX_LINEAR_GROUP_DIMENSION:
        _resource_error(
            ("dimension",),
            "dimension_bound",
            f"dimension must not exceed {MAX_LINEAR_GROUP_DIMENSION}",
        )
    generator_count = max(1, 2 * (dimension - 1)) + int(prime > 2)
    if generator_count * dimension * dimension > MAX_LINEAR_GROUP_GENERATOR_CELLS:
        _resource_error(
            ("dimension",),
            "generator_output_bound",
            "the canonical generator matrices exceed the aggregate output envelope",
        )
    order_digit_bound = dimension * dimension * decimal_digit_width(prime) + 1
    if order_digit_bound > MAX_LINEAR_GROUP_ORDER_DIGITS:
        _resource_error(
            ("dimension",),
            "order_output_bound",
            "the exact group-order envelope exceeds its canonical digit budget",
        )
    return prime, dimension


def _identity(prime: int, dimension: int) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple(
            tuple(1 if row == column else 0 for column in range(dimension))
            for row in range(dimension)
        ),
        columns=dimension,
    )


def _elementary(
    prime: int, dimension: int, source: int, target: int
) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple(
            tuple(
                1 if row == column else 1 if row == target and column == source else 0
                for column in range(dimension)
            )
            for row in range(dimension)
        ),
        columns=dimension,
    )


def _factor_distinct(value: int) -> tuple[int, ...]:
    factors: list[int] = []
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def _least_primitive_root(prime: int) -> int:
    if prime == 2:
        return 1
    factors = _factor_distinct(prime - 1)
    return next(
        candidate
        for candidate in range(2, prime)
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors)
    )


def _special_generators(prime: int, dimension: int) -> tuple[PrimeFieldMatrix, ...]:
    if dimension == 1:
        return (_identity(prime, dimension),)
    generators: list[PrimeFieldMatrix] = []
    for index in range(dimension - 1):
        generators.append(_elementary(prime, dimension, index + 1, index))
        generators.append(_elementary(prime, dimension, index, index + 1))
    return tuple(generators)


def _general_generators(prime: int, dimension: int) -> tuple[PrimeFieldMatrix, ...]:
    if prime == 2:
        return _special_generators(prime, dimension)
    primitive = _least_primitive_root(prime)
    diagonal = [list(row) for row in _identity(prime, dimension).entries]
    diagonal[0][0] = primitive
    scalar_generator = PrimeFieldMatrix(
        prime=prime,
        entries=tuple(tuple(row) for row in diagonal),
        columns=dimension,
    )
    if dimension == 1:
        return (scalar_generator,)
    return (*_special_generators(prime, dimension), scalar_generator)


def _general_linear_order(prime: int, dimension: int) -> int:
    full_space = prime**dimension
    return_value = 1
    for rank in range(dimension):
        return_value *= full_space - prime**rank
    return return_value


def construct_general_linear_group(
    prime: int, dimension: int
) -> PrimeFieldGeneralLinearGroup:
    """Construct the full standard matrix group ``GL(n,p)``.

    Adjacent elementary transvections generate ``SL(n,p)`` over the prime
    field. Adding the least primitive scalar in the first coordinate maps the
    generated group onto ``GF(p)^*`` under determinant, so the retained family
    generates all of ``GL(n,p)``. The exact order is the ordered-basis count.
    """

    admitted_prime, admitted_dimension = _admit_parameters(prime, dimension)
    return PrimeFieldGeneralLinearGroup(
        prime=admitted_prime,
        dimension=admitted_dimension,
        order=_general_linear_order(admitted_prime, admitted_dimension),
        generators=_general_generators(admitted_prime, admitted_dimension),
    )


def construct_special_linear_group(
    prime: int, dimension: int
) -> PrimeFieldSpecialLinearGroup:
    """Construct the determinant kernel ``SL(n,p)`` with complete generators."""

    admitted_prime, admitted_dimension = _admit_parameters(prime, dimension)
    general_order = _general_linear_order(admitted_prime, admitted_dimension)
    determinant_index = admitted_prime - 1
    return PrimeFieldSpecialLinearGroup(
        prime=admitted_prime,
        dimension=admitted_dimension,
        order=general_order // determinant_index,
        generators=_special_generators(admitted_prime, admitted_dimension),
        ambient_general_linear_order=general_order,
        determinant_index=determinant_index,
    )


@overload
def _revalidate_group(
    group: object, expected_type: type[PrimeFieldGeneralLinearGroup]
) -> PrimeFieldGeneralLinearGroup: ...


@overload
def _revalidate_group(
    group: object, expected_type: type[PrimeFieldSpecialLinearGroup]
) -> PrimeFieldSpecialLinearGroup: ...


def _revalidate_group(
    group: object,
    expected_type: type[PrimeFieldGeneralLinearGroup]
    | type[PrimeFieldSpecialLinearGroup],
) -> PrimeFieldGeneralLinearGroup | PrimeFieldSpecialLinearGroup:
    if not isinstance(group, expected_type):
        _domain_error(
            ("group",),
            "group_type",
            f"group must be a canonical {expected_type.__name__} value",
        )
    try:
        admitted = expected_type.model_validate_json(
            group.model_dump_json(warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("group",),
            code="finite_matrix_group.group_shape",
            message="group must satisfy its complete canonical carrier contract",
        ) from exc
    expected: PrimeFieldGeneralLinearGroup | PrimeFieldSpecialLinearGroup
    if expected_type is PrimeFieldGeneralLinearGroup:
        expected = construct_general_linear_group(admitted.prime, admitted.dimension)
    else:
        expected = construct_special_linear_group(admitted.prime, admitted.dimension)
    if admitted != expected:
        _domain_error(
            ("group",),
            "authored_group_mismatch",
            "authored order and generators must equal the canonical full named group",
        )
    return admitted


def _natural_action(
    group: PrimeFieldGeneralLinearGroup | PrimeFieldSpecialLinearGroup,
) -> tuple[tuple[tuple[int, ...], ...], FinitePermutationAction]:
    domain_size = group.prime**group.dimension - 1
    if domain_size > MAX_NATURAL_VECTOR_ACTION_SIZE:
        _resource_error(
            ("group",),
            "natural_action_size_bound",
            "the nonzero-vector axis exceeds the 50-point materialization envelope",
        )
    vectors = tuple(
        vector
        for vector in product(range(group.prime), repeat=group.dimension)
        if any(vector)
    )
    positions = {vector: index for index, vector in enumerate(vectors)}
    permutations = tuple(
        tuple(
            positions[
                tuple(
                    sum(
                        generator.entries[row][column] * vector[column]
                        for column in range(group.dimension)
                    )
                    % group.prime
                    for row in range(group.dimension)
                )
            ]
            for vector in vectors
        )
        for generator in group.generators
    )
    labels = tuple(
        "[" + ",".join(str(value) for value in vector) + "]" for vector in vectors
    )
    return vectors, FinitePermutationAction(domain=labels, generators=permutations)


def general_linear_nonzero_vector_action(
    group: PrimeFieldGeneralLinearGroup,
) -> PrimeFieldGeneralLinearNaturalAction:
    """Materialize the faithful natural GL action on nonzero vectors."""

    admitted = _revalidate_group(group, PrimeFieldGeneralLinearGroup)
    vectors, action = _natural_action(admitted)
    return PrimeFieldGeneralLinearNaturalAction(
        group=admitted,
        vectors=vectors,
        action=action,
    )


def special_linear_nonzero_vector_action(
    group: PrimeFieldSpecialLinearGroup,
) -> PrimeFieldSpecialLinearNaturalAction:
    """Materialize the natural SL action on nonzero vectors."""

    admitted = _revalidate_group(group, PrimeFieldSpecialLinearGroup)
    vectors, action = _natural_action(admitted)
    return PrimeFieldSpecialLinearNaturalAction(
        group=admitted,
        vectors=vectors,
        action=action,
    )


__all__ = [
    "construct_general_linear_group",
    "construct_special_linear_group",
    "general_linear_nonzero_vector_action",
    "special_linear_nonzero_vector_action",
]
