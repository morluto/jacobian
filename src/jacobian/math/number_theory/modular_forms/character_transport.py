"""Explicit bounded transport and common-target equality for character forms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

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
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    CHARACTER_BASIS_ID,
    _character_basis_from_admission,
    _character_sturm_precision,
    _domain,
    _require_basis_space,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterEqualityResult,
    ModularCharacterQExpansion,
    ModularCharacterSpaceInclusion,
    ModularCharacterTransportedForm,
)
from jacobian.math.number_theory.modular_forms.character_dimensions import (
    character_space_dimensions,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_MAX_TRANSPORT_WORK = 5_000_000
_MAX_TRANSPORT_OUTPUT_BYTES = 1_000_000


@dataclass(frozen=True)
class _AdmittedTransport:
    form: ModularFormCoordinates
    inclusion: ModularCharacterSpaceInclusion
    source_dimension: int
    target_dimension: int
    precision: int
    field: RationalCyclotomicField
    source_character_dimensions: tuple[int, int]
    target_character_dimensions: tuple[int, int]


def _linear_combination_digit_bound(coordinate_digits: int, terms: int) -> int:
    """Bound a cyclotomic linear combination with integral, one-digit basis rows.

    In Q(zeta_6), multiplication by an integral basis coefficient combines
    at most three products of scalar coordinates. Clearing the two scalar
    denominators costs at most twice their digit height; summing ``terms``
    coordinates multiplies that common-denominator envelope by ``terms``.
    The extra two digits cover the coefficients (whose absolute values are
    at most 9) and the final numerator addition.
    """
    return terms * (2 * coordinate_digits + 2) + 1


def _is_zero(value: RationalCyclotomicElement) -> bool:
    return all(coefficient.num == 0 for coefficient in value.coefficients_ascending)


def _require_inflation_map(
    inclusion: ModularCharacterSpaceInclusion,
) -> ModularCharacterSpaceInclusion:
    if type(inclusion) is not ModularCharacterSpaceInclusion:
        _domain("character transport requires a canonical explicit inclusion value")
    try:
        inclusion = ModularCharacterSpaceInclusion.model_validate(
            inclusion.model_dump()
        )
    except (TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="modular_form.character_inclusion_invalid",
            message="the explicit character-space inclusion is malformed",
        ) from error
    source = inclusion.source_space
    target = inclusion.target_space
    fields = inclusion.coefficient_field_map
    if (
        type(source) is not ModularFormSpace
        or type(target) is not ModularFormSpace
        or source.group != "GAMMA0"
        or target.group != "GAMMA0"
        or type(source.level) is not int
        or type(target.level) is not int
        or source.level not in (13, 26, 39)
        or target.level not in (26, 39)
        or target.level % source.level
        or source.weight != 2
        or target.weight != 2
        or source.kind != "S"
        or target.kind != "S"
        or type(fields.source) is not RationalCyclotomicField
        or type(fields.target) is not RationalCyclotomicField
        or fields.source != fields.target
        or fields.source != source.coefficient_domain
        or fields.target != target.coefficient_domain
        or fields.source.order != 6
        or fields.source.generator != "CLASS_OF_X"
        or type(inclusion.character_map.source) is not DirichletCharacter
        or type(inclusion.character_map.target) is not DirichletCharacter
        or inclusion.character_map.source != source.character
        or inclusion.character_map.target != target.character
    ):
        _domain(
            "character transport supports explicit S2 inclusions between levels 13, 26, and 39 over the identical Q(zeta_6) parent"
        )
    source_char = inclusion.character_map.source
    target_char = inclusion.character_map.target
    if (
        source_char.group.modulus != source.level
        or target_char.group.modulus != target.level
    ):
        _domain("character inflation moduli must bind their exact source and target levels")
    # Prove the authored map is pullback along residue reduction, on every
    # target unit. This is bounded by the admitted target level (at most 39).
    for residue in range(target.level):
        if gcd(residue, target.level) != 1:
            continue
        source_value = dirichlet_character_value(source_char, residue).value
        target_value = dirichlet_character_value(target_char, residue).value
        if source_value != target_value:
            raise OperationDomainValidationError(
                location=("inclusion", "character_map"),
                code="modular_form.character_inflation_mismatch",
                message="target character is not the explicit inflation of the source character",
            )
    return inclusion


def _admit_transport(
    form: ModularFormCoordinates,
    inclusion: ModularCharacterSpaceInclusion,
) -> _AdmittedTransport:
    inclusion = _require_inflation_map(inclusion)
    if type(form) is not ModularFormCoordinates:
        _domain("character transport requires a canonical character coordinate form")
    if form.space != inclusion.source_space:
        _domain("source form parent must equal the explicit inclusion source")
    source_space = inclusion.source_space
    field = inclusion.coefficient_field_map.source
    source_cusp, _ = character_space_dimensions(
        source_space.level, source_space.weight, source_space.character, field
    )
    target_cusp, _ = character_space_dimensions(
        inclusion.target_space.level,
        inclusion.target_space.weight,
        inclusion.target_space.character,
        field,
    )
    if form.basis_id == CHARACTER_BASIS_ID:
        if (
            source_space.level != 13
            or source_space.kind != "S"
            or form.basis_id != CHARACTER_BASIS_ID
            or type(form.coordinates) is not tuple
            or len(form.coordinates) != 1
            or type(form.coordinates[0]) is not RationalCyclotomicElement
            or form.coordinates[0].field != field
        ):
            _domain("legacy character coordinates are admitted only in the level-13 cusp source")
        for coordinate in form.coordinates:
            cyclotomic._validate_element(coordinate)
    elif form.basis_id == "gamma0-cyclotomic-character-sturm-rref-v1":
        if (
            type(form.coordinates) is not tuple
            or len(form.coordinates) != source_cusp
            or source_space.level == 13
        ):
            _domain("general character coordinates must match a level-26 or level-39 cusp basis")
        for coordinate in form.coordinates:
            if type(coordinate) is not RationalCyclotomicElement or coordinate.field != field:
                _domain("every character coordinate must use the exact declared field")
            cyclotomic._validate_element(coordinate)
    else:
        _domain("character transport requires an admitted character coordinate basis")

    target_precision = _character_sturm_precision(inclusion.target_space)
    source_precision = _character_sturm_precision(source_space)
    if not source_precision <= target_precision <= 128:
        raise OperationResourceAdmissionError(
            location=("inclusion",),
            code="modular_form.character_transport_precision_bound",
            message="target Sturm precision exceeds the admitted source expansion envelope",
        )
    coordinate_digits = max(
        (
            max(
                len(str(abs(int(value.num)))),
                len(str(int(value.den))),
            )
            for coordinate in form.coordinates
            for value in coordinate.coefficients_ascending
        ),
        default=1,
    )
    same_space = source_space == inclusion.target_space
    source_expansion_digits = _linear_combination_digit_bound(
        coordinate_digits, source_cusp
    )
    target_coordinate_digits = (
        coordinate_digits if same_space else source_expansion_digits
    )
    target_expansion_digits = (
        source_expansion_digits
        if same_space
        else _linear_combination_digit_bound(
            source_expansion_digits, target_cusp
        )
    )
    if max(source_expansion_digits, target_expansion_digits) > MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_transport_height_admission",
            message="character transport expansion or target solve exceeds the exact coefficient-height bound",
        )
    work = target_precision * (source_cusp + target_cusp**2 + source_cusp * target_cusp)
    output_bytes = target_precision * field.degree * (
        2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS + 32
    ) + target_cusp * field.degree * (2 * target_coordinate_digits + 32)
    if work > _MAX_TRANSPORT_WORK or output_bytes > _MAX_TRANSPORT_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_transport_admission",
            message="character transport work or exact output exceeds its admitted envelope",
        )
    return _AdmittedTransport(
        form=form,
        inclusion=inclusion,
        source_dimension=source_cusp,
        target_dimension=target_cusp,
        precision=target_precision,
        field=field,
        source_character_dimensions=(source_cusp, source_cusp),
        target_character_dimensions=(target_cusp, target_cusp),
    )


def _basis(
    space: ModularFormSpace,
    precision: int,
    admitted_dimensions: tuple[int, int],
):
    space, field, character_request = _require_basis_space(space)
    return _character_basis_from_admission(
        space,
        field,
        character_request,
        precision=precision,
        admitted_dimensions=admitted_dimensions,
    )


def _expand_coordinates(
    form: ModularFormCoordinates,
    basis,
    field: RationalCyclotomicField,
) -> tuple[RationalCyclotomicElement, ...]:
    return _expand_coordinate_tuple(form.coordinates, basis, field)


def _expand_coordinate_tuple(
    coordinates: tuple[RationalCyclotomicElement, ...], basis, field
) -> tuple[RationalCyclotomicElement, ...]:
    if len(coordinates) != len(basis.elements):
        _domain("character coordinate count differs from its exact basis dimension")
    result = [[Fraction(0), Fraction(0)] for _ in range(basis.precision)]
    for scalar, element in zip(coordinates, basis.elements, strict=True):
        a, b = (
            Fraction(coefficient.num, coefficient.den)
            for coefficient in scalar.coefficients_ascending
        )
        for index, coefficient in enumerate(element.expansion.coefficients):
            c, d = (int(value.num) for value in coefficient.coefficients_ascending)
            result[index][0] += a * c - b * d
            result[index][1] += a * d + b * c + b * d
    return tuple(cyclotomic._canonical(field, tuple(pair)) for pair in result)


def _coordinates_from_prefix(
    prefix: tuple[RationalCyclotomicElement, ...], basis
) -> ModularFormCoordinates:
    if len(prefix) != basis.precision:
        _domain("q-prefix precision must equal the common target Sturm precision")
    pivots = tuple(
        next(
            index
            for index, coefficient in enumerate(element.expansion.coefficients)
            if not _is_zero(coefficient)
        )
        for element in basis.elements
    )
    coordinates = tuple(prefix[pivot] for pivot in pivots)
    reconstructed = _linear_combination_from_coordinates(coordinates, basis)
    if reconstructed != prefix:
        raise OperationDomainValidationError(
            location=("inclusion", "target_space"),
            code="modular_form.character_transport_not_in_target",
            message="the source q-expansion is not in the target character space",
        )
    return ModularFormCoordinates(
        space=basis.space,
        basis_id="gamma0-cyclotomic-character-sturm-rref-v1",
        coordinates=coordinates,
    )


def _linear_combination_from_coordinates(coordinates, basis):
    return _expand_coordinate_tuple(
        coordinates, basis, basis.space.coefficient_domain
    )


def _transport_from_bases(
    admitted: _AdmittedTransport, source_basis, target_basis
) -> ModularCharacterTransportedForm:
    source_prefix = _expand_coordinates(admitted.form, source_basis, admitted.field)
    if len(source_prefix) != admitted.precision:
        _domain("source basis must extend through the target Sturm precision")
    if admitted.inclusion.source_space == admitted.inclusion.target_space:
        if admitted.form.basis_id == "gamma0-cyclotomic-character-sturm-rref-v1":
            target_form = admitted.form
        else:
            _domain("identity transport requires generalized character coordinates")
        target_expansion = source_prefix
    else:
        target_form = _coordinates_from_prefix(source_prefix, target_basis)
        target_expansion = source_prefix
    return ModularCharacterTransportedForm(
        source_form=admitted.form,
        inclusion=admitted.inclusion,
        target_form=target_form,
        target_q_expansion=ModularCharacterQExpansion(
            space=admitted.inclusion.target_space,
            basis_id=target_basis.basis_id,
            coefficients=target_expansion,
        ),
    )


def modular_character_coordinates_transport(
    form: ModularFormCoordinates,
    inclusion: ModularCharacterSpaceInclusion,
) -> ModularCharacterTransportedForm:
    """Transport a represented cusp form through explicit character inflation."""
    admitted = _admit_transport(form, inclusion)
    request_checkpoint("before character-space inclusion basis materialization")
    source_basis = _basis(
        inclusion.source_space,
        admitted.precision,
        admitted.source_character_dimensions,
    )
    target_basis = (
        source_basis
        if inclusion.target_space == inclusion.source_space
        else _basis(
            inclusion.target_space,
            admitted.precision,
            admitted.target_character_dimensions,
        )
    )
    result = _transport_from_bases(admitted, source_basis, target_basis)
    request_checkpoint("after exact character-space inclusion solve")
    return result


def _revalidate_transport(
    value: ModularCharacterTransportedForm,
) -> tuple[ModularCharacterTransportedForm, _AdmittedTransport]:
    if type(value) is not ModularCharacterTransportedForm:
        _domain("global character equality requires canonical transported forms")
    try:
        value = ModularCharacterTransportedForm.model_validate(value.model_dump())
    except (TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.character_transport_value_invalid",
            message="transported character form is malformed",
        ) from error
    return value, _admit_transport(value.source_form, value.inclusion)


def modular_character_coordinates_equal_in_common_space(
    left: ModularCharacterTransportedForm,
    right: ModularCharacterTransportedForm,
) -> ModularCharacterEqualityResult:
    """Decide equality after validating both explicit maps into one target."""
    left, left_admitted = _revalidate_transport(left)
    right, right_admitted = _revalidate_transport(right)
    target = left_admitted.inclusion.target_space
    if right_admitted.inclusion.target_space != target:
        raise OperationDomainValidationError(
            location=("right", "inclusion", "target_space"),
            code="modular_form.character_equality_common_target",
            message="both forms must land in the identical common target space",
        )
    if left_admitted.field != right_admitted.field:
        _domain("common-target character equality requires the identical coefficient field")
    request_checkpoint("before common character target basis materialization")
    target_basis = _basis(
        target,
        left_admitted.precision,
        left_admitted.target_character_dimensions,
    )
    bases = [(target, target_basis)]
    transported = []
    for value, admitted in ((left, left_admitted), (right, right_admitted)):
        source_space = admitted.inclusion.source_space
        source_basis = next(
            (basis for space, basis in bases if space == source_space), None
        )
        if source_basis is None:
            source_basis = _basis(
                source_space,
                admitted.precision,
                admitted.source_character_dimensions,
            )
            bases.append((source_space, source_basis))
        canonical = _transport_from_bases(admitted, source_basis, target_basis)
        if (
            canonical.target_form != value.target_form
            or canonical.target_q_expansion != value.target_q_expansion
        ):
            raise OperationDomainValidationError(
                location=("form",),
                code="modular_form.character_transport_claim_mismatch",
                message="retained target coordinates or q-prefix do not match the explicit source inclusion",
            )
        transported.append(canonical.target_q_expansion.coefficients)
    request_checkpoint("after exact common character equality comparison")
    return ModularCharacterEqualityResult(equal=transported[0] == transported[1])


__all__ = [
    "modular_character_coordinates_equal_in_common_space",
    "modular_character_coordinates_transport",
]
