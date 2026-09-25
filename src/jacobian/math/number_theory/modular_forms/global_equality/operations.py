"""Bounded equality across exact character spaces and levels."""

from __future__ import annotations

from math import gcd, isqrt, lcm
from typing import cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import _character_value
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    _character_sturm_precision,
    _rref_character_prefix,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    _admit_coordinate_space,
    _admit_coordinate_vector,
    _CoordinateSpace,
)
from jacobian.math.number_theory.modular_forms.global_equality.models import (
    CyclotomicFieldEmbedding,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
    MAX_PARI_BASIS_WORK,
    _pari_character_request,
    pari_character_basis,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

MAX_GLOBAL_EQUALITY_PRECISION = 1024
MAX_GLOBAL_EQUALITY_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_GLOBAL_EQUALITY_INTERMEDIATE_BYTES = 512 * 1024 * 1024


def _fail_domain(code: str, message: str, location: tuple[str, ...] = ()) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=0, den=1) for _ in range(field.degree)
        ),
    )


def _one(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=int(index == 0), den=1)
            for index in range(field.degree)
        ),
    )


def _embedding_image(
    embedding: CyclotomicFieldEmbedding, source: RationalCyclotomicField, side: str
) -> RationalCyclotomicElement:
    target = embedding.target_field
    image = embedding.generator_image
    if (
        type(embedding.source_order) is not int
        or embedding.source_order != source.order
        or source.order != 6
        or target.order not in (6, 12)
        or target.order % source.order
        or image.field != target
    ):
        _fail_domain(
            "modular_form.global_equality_embedding_parent",
            "each source field needs an explicit compatible cyclotomic embedding",
            (side, "embedding"),
        )
    cyclotomic._validate_element(image)
    one = _one(target)
    power = one
    for _ in range(source.order):
        power = cyclotomic.multiply(power, image)
    if power != one:
        _fail_domain(
            "modular_form.global_equality_embedding_root",
            "the generator image must have the declared source root order",
            (side, "embedding", "generator_image"),
        )
    for prime in (2, 3, 5, 7, 11, 13):
        if source.order % prime == 0:
            power = one
            for _ in range(source.order // prime):
                power = cyclotomic.multiply(power, image)
            if power == one:
                _fail_domain(
                    "modular_form.global_equality_embedding_root",
                    "the generator image must be a primitive root of the declared order",
                    (side, "embedding", "generator_image"),
                )
    return image


def _map_element(
    value: RationalCyclotomicElement,
    image: RationalCyclotomicElement,
    target: RationalCyclotomicField,
) -> RationalCyclotomicElement:
    _, coefficients, _ = cyclotomic._validate_element(value)
    result = _zero(target)
    power = _one(target)
    for coefficient in coefficients:
        scalar = RationalCyclotomicElement(
            field=target,
            coefficients_ascending=tuple(
                CanonicalRational.from_fraction(coefficient)
                if index == 0
                else CanonicalRational(num=0, den=1)
                for index in range(target.degree)
            ),
        )
        result = cyclotomic.add(result, cyclotomic.multiply(scalar, power))
        power = cyclotomic.multiply(power, image)
    return result


def _mapped_character_value(
    character: DirichletCharacter,
    residue: int,
    image: RationalCyclotomicElement,
    target: RationalCyclotomicField,
) -> RationalCyclotomicElement:
    value = _character_value(character, residue)
    if value is None:
        raise RuntimeError("a unit modulo the common level was not a source unit")
    exponent_numerator = value.exponent * 6
    if exponent_numerator % value.order:
        _fail_domain(
            "modular_form.global_equality_character_field",
            "the declared source field does not contain a character value",
        )
    power = _one(target)
    for _ in range(exponent_numerator // value.order):
        power = cyclotomic.multiply(power, image)
    return power


def _require_distinct_mapped_characters(
    left: ModularFormSpace,
    right: ModularFormSpace,
    left_image: RationalCyclotomicElement,
    right_image: RationalCyclotomicElement,
    target: RationalCyclotomicField,
) -> None:
    """Keep same-Nebentypus transport/equality in its existing owner lane."""
    level = lcm(left.level, right.level)
    work = 2 * level * max(left.level, right.level) * 32
    if work > 500_000:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.global_equality_character_work_bound",
            message="common character comparison exceeds its exact work envelope",
        )
    left_character = left.character
    right_character = right.character
    if (
        type(left_character) is not DirichletCharacter
        or type(right_character) is not DirichletCharacter
    ):
        raise RuntimeError("admitted character source lost its exact character")
    for residue in range(1, level):
        if gcd(residue, level) != 1:
            continue
        left_value = _mapped_character_value(
            left_character, residue, left_image, target
        )
        right_value = _mapped_character_value(
            right_character, residue, right_image, target
        )
        if left_value != right_value:
            return
    _fail_domain(
        "modular_form.global_equality_same_character",
        "the explicit embeddings give the same character at the common level; "
        "use the existing same-character transport and equality contract",
    )


def _mapped_prefix(
    form: ModularFormCoordinates,
    context: _CoordinateSpace,
    image: RationalCyclotomicElement,
    target: RationalCyclotomicField,
    precision: int,
) -> tuple[RationalCyclotomicElement, ...]:
    field = context.field
    dimension = context.dimension
    if dimension == 0:
        return tuple(_zero(target) for _ in range(precision))
    request = _pari_character_request(context.space)
    raw = pari_character_basis(
        context.space,
        precision,
        dimension,
        character_request=request,
    )
    source_precision = _character_sturm_precision(context.space)
    basis = _rref_character_prefix(
        raw,
        field,
        precision,
        normalization_precision=source_precision,
    )
    coordinates = tuple(
        _map_element(cast(RationalCyclotomicElement, value), image, target)
        for value in form.coordinates
    )
    mapped_basis = tuple(
        tuple(_map_element(value, image, target) for value in vector)
        for vector in basis
    )
    result = []
    for column in range(precision):
        coefficient = _zero(target)
        for scalar, vector in zip(coordinates, mapped_basis, strict=True):
            coefficient = cyclotomic.add(
                coefficient, cyclotomic.multiply(scalar, vector[column])
            )
        result.append(coefficient)
    return tuple(result)


def modular_form_coordinates_global_equal(
    left: ModularFormCoordinates,
    left_embedding: CyclotomicFieldEmbedding,
    right: ModularFormCoordinates,
    right_embedding: CyclotomicFieldEmbedding,
) -> bool:
    """Compare exact source coordinates through a Gamma1(lcm) Sturm prefix.

    This slice compares distinct transported Nebentypus characters in the
    order-six character families at levels 13, 26, and 39. Field maps are
    supplied on each cyclotomic generator. The common ambient form need not be
    materialized: each exact source basis is expanded only through the
    determining prefix of Gamma1(lcm(N_left,N_right)).
    """
    if (
        type(left) is ModularFormCoordinates
        and type(right) is ModularFormCoordinates
        and type(left.space) is ModularFormSpace
        and type(right.space) is ModularFormSpace
        and type(left.space.weight) is int
        and type(right.space.weight) is int
        and left.space.weight != right.space.weight
    ):
        _fail_domain(
            "modular_form.global_equality_weight",
            "modular-form equality requires equal weights",
            ("right", "space", "weight"),
        )
    left_context = _admit_coordinate_space(left.space)
    right_context = _admit_coordinate_space(right.space)
    _admit_coordinate_vector(left, left_context, "left")
    _admit_coordinate_vector(right, right_context, "right")
    left_image = _embedding_image(left_embedding, left_context.field, "left")
    right_image = _embedding_image(right_embedding, right_context.field, "right")
    if left_embedding.target_field != right_embedding.target_field:
        _fail_domain(
            "modular_form.global_equality_common_field",
            "both explicit field maps must have the identical target field",
        )
    common_field = left_embedding.target_field
    _require_distinct_mapped_characters(
        left.space, right.space, left_image, right_image, common_field
    )

    # The upper bound is the closed formula [SL2(Z):Gamma1(L)].
    level = lcm(left.space.level, right.space.level)
    index = level * level
    for prime in range(2, level + 1):
        if level % prime == 0 and all(
            prime % divisor for divisor in range(2, isqrt(prime) + 1)
        ):
            index = index * (prime * prime - 1) // (prime * prime)
    precision = (left.space.weight * index) // 12 + 1
    if precision > MAX_GLOBAL_EQUALITY_PRECISION:
        raise OperationResourceAdmissionError(
            location=("right",),
            code="modular_form.global_equality_precision_bound",
            message="Gamma1 Sturm prefix exceeds the global equality envelope",
        )
    left_dimension, right_dimension = left_context.dimension, right_context.dimension
    work = precision * (left_dimension**2 + right_dimension**2) * 64
    if work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.global_equality_work_bound",
            message="common Sturm basis expansion exceeds its combined work envelope",
        )
    dimension = max(left_dimension, right_dimension, 1)
    source_degree = max(left_context.field.degree, right_context.field.degree)
    intermediate_digits = 2 * (dimension * source_degree) ** 2 * 30 + 128
    worker_bytes = (left_dimension + right_dimension) * precision * source_degree * 48
    rref_bytes = (
        precision
        * (left_dimension + right_dimension)
        * source_degree
        * (2 * intermediate_digits + 32)
    )
    mapped_bytes = (
        precision
        * (left_dimension + right_dimension)
        * common_field.degree
        * (2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS + 32)
    )
    if (
        intermediate_digits > 100_000
        or worker_bytes > MAX_GLOBAL_EQUALITY_OUTPUT_BYTES
        or rref_bytes + mapped_bytes > MAX_GLOBAL_EQUALITY_INTERMEDIATE_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.global_equality_output_bound",
            message="common Sturm comparison exceeds its exact coefficient envelope",
        )
    if left_dimension == 0 and right_dimension == 0:
        return True
    left_prefix = _mapped_prefix(
        left, left_context, left_image, common_field, precision
    )
    right_prefix = _mapped_prefix(
        right, right_context, right_image, common_field, precision
    )
    return left_prefix == right_prefix


__all__ = ["modular_form_coordinates_global_equal"]
