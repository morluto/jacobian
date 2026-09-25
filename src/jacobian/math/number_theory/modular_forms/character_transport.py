"""Explicit bounded transport and common-target equality for character forms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear import (
    apply_cyclotomic_field_inclusion,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.matrices.cyclic_linear.operations import (
    CyclicRankKernelAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    _TRANSPORT_STURM_BASIS_ENVELOPE,
    CHARACTER_BASIS_ID,
    _character_basis_from_admission,
    _character_root_of_unity,
    _character_sturm_precision,
    _domain,
    _require_basis_space,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterCommonTargetPrefix,
    ModularCharacterCoordinates,
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
    form: ModularFormCoordinates | ModularCharacterCoordinates
    inclusion: ModularCharacterSpaceInclusion
    source_dimension: int
    target_dimension: int
    precision: int
    field: RationalCyclotomicField
    target_field: RationalCyclotomicField
    source_character_dimensions: tuple[int, int]
    target_character_dimensions: tuple[int, int]


def _linear_combination_digit_bound(
    coordinate_digits: int, terms: int, basis_coefficient_digits: int
) -> int:
    """Bound a cyclotomic linear combination with integral, one-digit basis rows.

    In Q(zeta_6), multiplication by an integral basis coefficient combines
    at most three rational products. Their common denominator is bounded by
    the product of the ``terms`` coordinate denominators and ``terms`` basis
    denominators; the numerator adds at most ``3 * terms`` such products.
    This covers the integral one-digit bases at levels 13, 26, and 39.
    """
    return (
        terms * (coordinate_digits + basis_coefficient_digits) + len(str(3 * terms)) + 2
    )


def _is_zero(value: RationalCyclotomicElement) -> bool:
    return all(coefficient.num == 0 for coefficient in value.coefficients_ascending)


def _require_inflation_map(
    inclusion: ModularCharacterSpaceInclusion,
) -> ModularCharacterSpaceInclusion:
    if type(inclusion) is not ModularCharacterSpaceInclusion:
        _domain("character transport requires a canonical explicit inclusion value")
    try:
        inclusion = ModularCharacterSpaceInclusion.model_validate(
            inclusion.model_dump(warnings=False)
        )
    except (AttributeError, TypeError, ValueError) as error:
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
        or target.level not in (13, 26, 39, 78)
        or target.level % source.level
        or source.weight != 2
        or target.weight != 2
        or source.kind != "S"
        or target.kind != "S"
        or type(fields.source) is not RationalCyclotomicField
        or type(fields.target) is not RationalCyclotomicField
        or fields.source != source.coefficient_domain
        or fields.target != target.coefficient_domain
        or fields.source.order != 6
        or fields.source.generator != "CLASS_OF_X"
        or fields.target.order not in (6, 12)
        or fields.target.generator != "CLASS_OF_X"
        or type(inclusion.character_map.source) is not DirichletCharacter
        or type(inclusion.character_map.target) is not DirichletCharacter
        or inclusion.character_map.source != source.character
        or inclusion.character_map.target != target.character
    ):
        _domain(
            "character transport supports S2 inclusions over Q(zeta_6) or its standard extension to Q(zeta_12)"
        )
    source_char = inclusion.character_map.source
    target_char = inclusion.character_map.target
    if (
        source_char.group.modulus != source.level
        or target_char.group.modulus != target.level
    ):
        _domain(
            "character inflation moduli must bind their exact source and target levels"
        )
    # Validate the authored field map through its canonical owner operation.
    source_generator = RationalCyclotomicElement(
        field=fields.source,
        coefficients_ascending=(
            CanonicalRational.from_fraction(Fraction(0)),
            CanonicalRational.from_fraction(Fraction(1)),
        ),
    )
    try:
        apply_cyclotomic_field_inclusion(fields, source_generator)
    except CyclicRankKernelAdmissionError as error:
        raise OperationDomainValidationError(
            location=("inclusion", "coefficient_field_map"),
            code="modular_form.character_field_map_invalid",
            message="the coefficient field map is not the canonical standard inclusion",
        ) from error
    # Prove the authored character map is pullback along residue reduction on
    # every target unit, and that its coefficient values commute with the
    # standard field inclusion. The target level is at most 78 and each
    # character has at most six distinct values in this admitted family.
    checked_character_values: dict[int, int] = {}
    for residue in range(target.level):
        if gcd(residue, target.level) != 1:
            continue
        source_value = dirichlet_character_value(source_char, residue).value
        target_value = dirichlet_character_value(target_char, residue).value
        if source_value is None or target_value is None:
            raise OperationDomainValidationError(
                location=("inclusion", "character_map"),
                code="modular_form.character_inflation_value_missing",
                message="unit residues must have exact character values",
            )
        if source_value != target_value:
            raise OperationDomainValidationError(
                location=("inclusion", "character_map"),
                code="modular_form.character_inflation_mismatch",
                message="target character is not the explicit inflation of the source character",
            )
        checked_character_values.setdefault(source_value.exponent, residue)
    for residue in checked_character_values.values():
        source_coefficient = _character_root_of_unity(
            source_char, residue, fields.source
        )
        target_coefficient = _character_root_of_unity(
            target_char, residue, fields.target
        )
        try:
            mapped_value = apply_cyclotomic_field_inclusion(fields, source_coefficient)
        except CyclicRankKernelAdmissionError as error:
            raise OperationDomainValidationError(
                location=("inclusion", "coefficient_field_map"),
                code="modular_form.character_field_map_invalid",
                message="the coefficient field map is not the canonical standard inclusion",
            ) from error
        if mapped_value != target_coefficient:
            raise OperationDomainValidationError(
                location=("inclusion", "coefficient_field_map"),
                code="modular_form.character_field_map_mismatch",
                message="the standard coefficient-field map does not preserve the inflated character values",
            )
    return inclusion


def _canonical_transport_form(
    form: ModularFormCoordinates | ModularCharacterCoordinates,
) -> ModularFormCoordinates | ModularCharacterCoordinates:
    if type(form) not in (ModularFormCoordinates, ModularCharacterCoordinates):
        _domain("character transport requires a canonical character coordinate form")
    try:
        return type(form).model_validate(form.model_dump(warnings=False))
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.character_coordinates_invalid",
            message="character coordinate form is malformed",
        ) from error


def _admit_transport(
    form: ModularFormCoordinates | ModularCharacterCoordinates,
    inclusion: ModularCharacterSpaceInclusion,
) -> _AdmittedTransport:
    inclusion = _require_inflation_map(inclusion)
    form = _canonical_transport_form(form)
    if form.space != inclusion.source_space:
        _domain("source form parent must equal the explicit inclusion source")
    source_space = inclusion.source_space
    field = inclusion.coefficient_field_map.source
    target_field = inclusion.coefficient_field_map.target
    source_character = inclusion.character_map.source
    target_character = inclusion.character_map.target
    source_cusp, _ = character_space_dimensions(
        source_space.level, source_space.weight, source_character, field
    )
    target_cusp = (
        0
        if inclusion.target_space.level in (13, 78)
        else character_space_dimensions(
            inclusion.target_space.level,
            inclusion.target_space.weight,
            target_character,
            field,
        )[0]
    )
    if type(form) is ModularFormCoordinates:
        if (
            source_space.level != 13
            or source_space.kind != "S"
            or form.basis_id != CHARACTER_BASIS_ID
            or type(form.coordinates) is not tuple
            or len(form.coordinates) != 1
            or type(form.coordinates[0]) is not RationalCyclotomicElement
            or form.coordinates[0].field != field
        ):
            _domain(
                "legacy character coordinates are admitted only in the level-13 cusp source"
            )
        for coordinate in form.coordinates:
            cyclotomic._validate_element(coordinate)
    else:
        if (
            form.basis_id != "gamma0-cyclotomic-character-sturm-rref-v1"
            or len(form.coordinates) != source_cusp
            or source_space.level == 13
        ):
            _domain(
                "general character coordinates must match a level-26 or level-39 cusp basis"
            )
        for coordinate in form.coordinates:
            if (
                type(coordinate) is not RationalCyclotomicElement
                or coordinate.field != field
            ):
                _domain("every character coordinate must use the exact declared field")
            cyclotomic._validate_element(coordinate)

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
    source_basis_digits = _TRANSPORT_STURM_BASIS_ENVELOPE[
        (source_space.level, target_precision)
    ][1]
    target_basis_digits = (
        0
        if inclusion.target_space.level in (13, 78)
        else _TRANSPORT_STURM_BASIS_ENVELOPE[
            (inclusion.target_space.level, target_precision)
        ][1]
    )
    source_expansion_digits = _linear_combination_digit_bound(
        coordinate_digits, source_cusp, source_basis_digits
    )
    target_coordinate_digits = (
        coordinate_digits if same_space else source_expansion_digits
    )
    target_expansion_digits = (
        source_expansion_digits
        if same_space or field != target_field
        else source_expansion_digits
        if inclusion.target_space.level in (13, 78)
        else _linear_combination_digit_bound(
            source_expansion_digits, target_cusp, target_basis_digits
        )
    )
    if (
        max(source_expansion_digits, target_expansion_digits)
        > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_transport_height_admission",
            message="character transport expansion or target solve exceeds the exact coefficient-height bound",
        )
    field_map_work = (
        (target_precision + len(target_character.group.unit_residues) + 7)
        * field.degree
        * target_field.degree**2
    )
    work = (
        target_precision * (source_cusp + target_cusp**2 + source_cusp * target_cusp)
        + field_map_work
    )
    output_bytes = target_precision * target_field.degree * (
        2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS + 32
    ) + target_cusp * target_field.degree * (2 * target_coordinate_digits + 32)
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
        target_field=target_field,
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
    form: ModularFormCoordinates | ModularCharacterCoordinates,
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
) -> ModularCharacterCoordinates:
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
    return ModularCharacterCoordinates(
        space=basis.space,
        basis_id="gamma0-cyclotomic-character-sturm-rref-v1",
        coordinates=coordinates,
    )


def _linear_combination_from_coordinates(coordinates, basis):
    return _expand_coordinate_tuple(coordinates, basis, basis.space.coefficient_domain)


def _transport_from_bases(
    admitted: _AdmittedTransport, source_basis, target_basis=None
) -> ModularCharacterTransportedForm:
    source_prefix = _expand_coordinates(admitted.form, source_basis, admitted.field)
    if len(source_prefix) != admitted.precision:
        _domain("source basis must extend through the target Sturm precision")
    coefficient_map = admitted.inclusion.coefficient_field_map
    target_prefix = (
        source_prefix
        if admitted.field == admitted.target_field
        else tuple(
            apply_cyclotomic_field_inclusion(coefficient_map, coefficient)
            for coefficient in source_prefix
        )
    )
    if (
        admitted.inclusion.target_space.level in (13, 78)
        or admitted.field != admitted.target_field
    ):
        target_form = None
        target_expansion = ModularCharacterCommonTargetPrefix(
            space=admitted.inclusion.target_space,
            precision=admitted.precision,
            coefficients=target_prefix,
        )
    elif admitted.inclusion.source_space == admitted.inclusion.target_space:
        if type(admitted.form) is ModularCharacterCoordinates:
            target_form = admitted.form
        else:
            _domain("identity transport requires generalized character coordinates")
        target_expansion = ModularCharacterQExpansion(
            space=admitted.inclusion.target_space,
            basis_id=target_basis.basis_id,
            coefficients=target_prefix,
        )
    else:
        target_form = _coordinates_from_prefix(target_prefix, target_basis)
        target_expansion = ModularCharacterQExpansion(
            space=admitted.inclusion.target_space,
            basis_id=target_basis.basis_id,
            coefficients=target_prefix,
        )
    return ModularCharacterTransportedForm(
        source_form=admitted.form,
        inclusion=admitted.inclusion,
        target_form=target_form,
        target_q_expansion=target_expansion,
    )


def modular_character_coordinates_transport(
    form: ModularFormCoordinates | ModularCharacterCoordinates,
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
        None
        if inclusion.target_space.level in (13, 78)
        or admitted.field != admitted.target_field
        else source_basis
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
        value = ModularCharacterTransportedForm.model_validate(
            value.model_dump(warnings=False)
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.character_transport_value_invalid",
            message="transported character form is malformed",
        ) from error
    return value, _admit_transport(value.source_form, value.inclusion)


def _transport_equality_work(admitted: _AdmittedTransport) -> int:
    """Bound one retransport and target-field coefficient map in equality."""
    target_character = admitted.inclusion.character_map.target
    coefficient_map_work = (
        (admitted.precision + len(target_character.group.unit_residues) + 7)
        * admitted.field.degree
        * admitted.target_field.degree**2
    )
    comparison_work = admitted.precision * (
        admitted.source_dimension * admitted.field.degree**2
        + admitted.target_dimension**2 * admitted.target_field.degree
        + admitted.source_dimension * admitted.target_dimension
    )
    return coefficient_map_work + comparison_work


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
    expected_level = lcm(
        left_admitted.inclusion.source_space.level,
        right_admitted.inclusion.source_space.level,
    )
    if target.level != expected_level:
        raise OperationDomainValidationError(
            location=("right", "inclusion", "target_space", "level"),
            code="modular_form.character_equality_target_not_lcm",
            message="the common target level must be the least common multiple of source levels",
        )
    if left_admitted.target_field != right_admitted.target_field:
        _domain(
            "common-target character equality requires the identical target coefficient field"
        )
    combined_work = _transport_equality_work(left_admitted) + _transport_equality_work(
        right_admitted
    )
    combined_output_bytes = sum(
        admitted.precision
        * admitted.target_field.degree
        * (2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS + 32)
        + admitted.target_dimension
        * admitted.target_field.degree
        * (2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS + 32)
        for admitted in (left_admitted, right_admitted)
    )
    if (
        combined_work > _MAX_TRANSPORT_WORK
        or combined_output_bytes > _MAX_TRANSPORT_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.character_equality_admission",
            message="combined common-space equality work or output exceeds its admitted envelope",
        )

    request_checkpoint("before common character target basis materialization")
    target_basis = (
        None
        if target.level in (13, 78) or left_admitted.field != left_admitted.target_field
        else _basis(
            target,
            left_admitted.precision,
            left_admitted.target_character_dimensions,
        )
    )
    bases = [] if target_basis is None else [(target, target_basis)]
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
