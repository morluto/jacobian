"""Bounded exact Hecke matrices in canonical character q-Sturm bases."""

from __future__ import annotations

from math import factorial, gcd, lcm
from typing import Literal, NamedTuple, cast

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    _character_root_of_unity,
    _character_sturm_precision,
    _rref_character_prefix,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    MAX_CHARACTER_BASIS_PRECISION,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
    _admit_coordinate_space,
    _CoordinateSpace,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.models import (
    ModularCharacterHeckeMatrixResult,
)
from jacobian.math.number_theory.modular_forms.pari_backend import (
    MAX_PARI_BASIS_WORK,
    _pari_character_request,
    pari_character_basis,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

MAX_MULTIDIMENSIONAL_HECKE_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_MULTIDIMENSIONAL_HECKE_WORK = 50_000_000
MAX_MULTIDIMENSIONAL_HECKE_INTERMEDIATE_BYTES = 512 * 1024 * 1024


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=0, den=1) for _ in range(field.degree)
        ),
    )


def _is_zero(value: RationalCyclotomicElement) -> bool:
    return all(coefficient.num == 0 for coefficient in value.coefficients_ascending)


def _integer(field: RationalCyclotomicField, value: int) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=value if i == 0 else 0, den=1)
            for i in range(field.degree)
        ),
    )


def _hecke_coefficient(
    coefficients: tuple[RationalCyclotomicElement, ...],
    character: DirichletCharacter,
    field: RationalCyclotomicField,
    index: int,
    output_index: int,
) -> RationalCyclotomicElement:
    result = _zero(field)
    for divisor in range(1, index + 1):
        if gcd(output_index, index) % divisor:
            continue
        source_index = output_index * index // (divisor * divisor)
        character_value = _character_root_of_unity(character, divisor, field)
        term = cyclotomic.multiply(character_value, coefficients[source_index])
        result = cyclotomic.add(
            result, cyclotomic.multiply(_integer(field, divisor), term)
        )
    return result


def _apply_tn(
    coefficients: tuple[RationalCyclotomicElement, ...],
    character: DirichletCharacter,
    field: RationalCyclotomicField,
    index: int,
    output_precision: int,
) -> tuple[RationalCyclotomicElement, ...]:
    return tuple(
        _hecke_coefficient(coefficients, character, field, index, coefficient_index)
        for coefficient_index in range(output_precision)
    )


class _HeckeAdmission(NamedTuple):
    context: _CoordinateSpace
    character: DirichletCharacter
    field: RationalCyclotomicField
    target_precision: int
    source_precision: int
    basis_digit_bound: int
    matrix_digit_bound: int
    character_request: dict[str, object]


def _admit_hecke_request(space: ModularFormSpace, index: int) -> _HeckeAdmission:
    context = _admit_coordinate_space(space)
    if context.basis_id != CHARACTER_RREF_BASIS_ID:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.multidimensional_hecke_basis",
            message="multidimensional Hecke matrices require the canonical q-Sturm RREF basis",
        )
    dimension = context.dimension
    if dimension < 2:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.multidimensional_hecke_dimension",
            message="this operation is reserved for spaces of dimension at least two",
        )
    if type(index) is not int or not 1 <= index <= 8:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.multidimensional_hecke_index_bound",
            message="multidimensional character Hecke indices must lie in [1, 8]",
        )
    if gcd(index, context.space.level) != 1:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.multidimensional_hecke_coprime_level",
            message="T_n is admitted only when the index is coprime to the level",
        )
    character = context.space.character
    if type(character) is not DirichletCharacter:
        raise RuntimeError("admitted character space lost its exact character")
    character_order = lcm(
        *(
            order // gcd(coordinate, order)
            for coordinate, order in zip(
                character.coordinates, character.group.generator_orders, strict=True
            )
        )
    )
    if character_order not in (1, 2, 3, 6):
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.multidimensional_hecke_character_order",
            message="the coefficient field must contain the character values",
        )
    field = context.field
    if type(field) is not RationalCyclotomicField or field.degree != 2:
        raise RuntimeError("admitted character space lost its order-six field")

    target_precision = _character_sturm_precision(context.space)
    source_precision = index * (target_precision - 1) + 1
    if source_precision > MAX_CHARACTER_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.multidimensional_hecke_precision_bound",
            message="Hecke source prefix exceeds the canonical character basis envelope",
        )
    work = (
        dimension * dimension * source_precision * field.degree * 16
        + dimension * target_precision * index * field.degree * 16
        + dimension * dimension * target_precision * field.degree * 8
    )
    # Raw PARI coordinates have four-digit numerators and denominators. A
    # determinant expansion over at most dimension rows bounds RREF growth;
    # the margins cover field inversion and divisor-sum reconstruction.
    determinant_term_count = factorial(dimension) * (2**dimension)
    basis_digit_bound = 8 * dimension + 2 * len(str(determinant_term_count)) + 24
    matrix_digit_bound = basis_digit_bound + len(str(index)) + len(str(dimension)) + 8
    worker_bytes = dimension * source_precision * field.degree * 48
    intermediate_digits = 2 * (dimension * field.degree) ** 2 * 30 + 128
    rref_bytes = (
        dimension * source_precision * field.degree * (2 * intermediate_digits + 32)
    )
    basis_bytes = (
        dimension * source_precision * field.degree * (2 * basis_digit_bound + 32)
    )
    image_bytes = (
        dimension * target_precision * field.degree * (2 * matrix_digit_bound + 32)
    )
    matrix_bytes = dimension * dimension * field.degree * (2 * matrix_digit_bound + 32)
    if (
        work > MAX_MULTIDIMENSIONAL_HECKE_WORK
        or work > MAX_PARI_BASIS_WORK
        or matrix_bytes > MAX_MULTIDIMENSIONAL_HECKE_OUTPUT_BYTES
        or worker_bytes + rref_bytes + basis_bytes + image_bytes + matrix_bytes
        > MAX_MULTIDIMENSIONAL_HECKE_INTERMEDIATE_BYTES
        or matrix_digit_bound > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.multidimensional_hecke_admission",
            message="Hecke basis work, coefficient growth, or exact output exceeds its admitted envelope",
        )
    return _HeckeAdmission(
        context=context,
        character=character,
        field=field,
        target_precision=target_precision,
        source_precision=source_precision,
        basis_digit_bound=basis_digit_bound,
        matrix_digit_bound=matrix_digit_bound,
        character_request=_pari_character_request(context.space),
    )


def _canonical_basis(
    admission: _HeckeAdmission,
) -> tuple[tuple[RationalCyclotomicElement, ...], ...]:
    context = admission.context
    request_checkpoint("before multidimensional character Hecke basis expansion")
    raw = pari_character_basis(
        context.space,
        admission.source_precision,
        context.dimension,
        character_request=admission.character_request,
    )
    basis = _rref_character_prefix(
        raw,
        admission.field,
        admission.source_precision,
        normalization_precision=admission.target_precision,
    )
    if len(basis) != context.dimension or any(
        len(row) != admission.source_precision for row in basis
    ):
        raise RuntimeError("canonical Hecke basis has an unexpected exact shape")
    if any(
        cyclotomic._validate_element(value)[2] > admission.basis_digit_bound
        for row in basis
        for value in row
    ):
        raise RuntimeError("canonical Hecke RREF basis exceeded its admitted height")
    return basis


def _matrix_from_basis(
    admission: _HeckeAdmission,
    basis: tuple[tuple[RationalCyclotomicElement, ...], ...],
    index: int,
) -> tuple[tuple[RationalCyclotomicElement, ...], ...]:
    dimension = len(basis)
    pivots_optional = tuple(
        next((i for i, value in enumerate(row) if not _is_zero(value)), None)
        for row in basis
    )
    if any(pivot is None for pivot in pivots_optional):
        raise RuntimeError("canonical RREF basis has a zero row")
    if len(set(pivots_optional)) != dimension:
        raise RuntimeError("canonical RREF basis has duplicate pivot columns")
    pivots = cast(tuple[int, ...], pivots_optional)
    one = _one(admission.field)
    zero = _zero(admission.field)
    if any(
        basis[row][pivot] != (one if row == column else zero)
        for column, pivot in enumerate(pivots)
        for row in range(dimension)
    ):
        raise RuntimeError("canonical RREF pivots do not form an identity minor")

    images = tuple(
        _apply_tn(
            row, admission.character, admission.field, index, admission.target_precision
        )
        for row in basis
    )
    matrix = tuple(
        tuple(images[column][pivots[row]] for column in range(dimension))
        for row in range(dimension)
    )
    if any(
        cyclotomic._validate_element(value)[2] > admission.matrix_digit_bound
        for row in matrix
        for value in row
    ):
        raise RuntimeError(
            "Hecke matrix entry exceeded its admitted coefficient height"
        )
    for column, image in enumerate(images):
        for coefficient_index in range(admission.target_precision):
            reconstructed = zero
            for row in range(dimension):
                reconstructed = cyclotomic.add(
                    reconstructed,
                    cyclotomic.multiply(
                        matrix[row][column], basis[row][coefficient_index]
                    ),
                )
            if reconstructed != image[coefficient_index]:
                raise RuntimeError(
                    "Hecke image failed full Sturm-prefix reconstruction"
                )
    return matrix


def modular_character_hecke_matrix_multidimensional(
    space: ModularFormSpace, index: int
) -> ModularCharacterHeckeMatrixResult:
    """Compute exact T_n on a multidimensional cyclotomic-character space.

    The bounded slice is weight two, level 13/26/39, Q(zeta_6)-valued, and
    has character values in Q(zeta_6). The index is at most eight and coprime
    to the level, so T_n is an endomorphism. The returned matrix uses the
    canonical q-Sturm RREF basis and is reconstructed through every Sturm
    coefficient after admission of the required source prefix ``n(B-1)+1``.
    """
    admission = _admit_hecke_request(space, index)
    basis = _canonical_basis(admission)
    matrix = _matrix_from_basis(admission, basis, index)
    request_checkpoint("after multidimensional character Hecke reconstruction")
    labels = tuple(
        f"q^{next(i for i, value in enumerate(row) if not _is_zero(value))}"
        for row in basis
    )
    return ModularCharacterHeckeMatrixResult(
        space=space,
        basis_id=cast(
            Literal["gamma0-cyclotomic-character-sturm-rref-v1"],
            CHARACTER_RREF_BASIS_ID,
        ),
        index=index,
        labels=labels,
        entries=matrix,
    )


def _one(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=int(i == 0), den=1) for i in range(field.degree)
        ),
    )
