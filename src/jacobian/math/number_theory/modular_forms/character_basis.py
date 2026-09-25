"""Bounded field-valued modular-form bases for one admitted character family."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal, NoReturn

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
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterBasis,
    ModularCharacterBasisElement,
    ModularCharacterHeckeMatrix,
    ModularCharacterQExpansion,
)
from jacobian.math.number_theory.modular_forms.pari_backend import (
    MAX_PARI_BASIS_PRECISION,
    _pari_character_request,
    pari_character_basis,
)
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
    ModularFormSpace,
)

CHARACTER_BASIS_ID: Literal["gamma0-13-even-order6-character-sturm-v1"] = (
    "gamma0-13-even-order6-character-sturm-v1"
)
_DIMENSION = 1
_STURM_PRECISION: Literal[3] = 3  # floor(2 * [SL2(Z):Gamma0(13)] / 12) + 1 = 3
_MAX_WORK = 1_000_000
_MAX_ALLOCATION_BYTES = 1_000_000
_MAX_NORMALIZED_COORDINATE_DIGITS = 1
MAX_CHARACTER_HECKE_INDEX = 32
MAX_CHARACTER_HECKE_SOURCE_PRECISION = 2 * MAX_CHARACTER_HECKE_INDEX + 1
_MAX_CHARACTER_HECKE_COEFFICIENT_DIGITS = 4


def _domain(message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("space",),
        code="modular_form.character_basis_unsupported_space",
        message=message,
    )


def _require_space(
    space: ModularFormSpace,
) -> tuple[ModularFormSpace, RationalCyclotomicField, dict[str, object]]:
    if type(space) is not ModularFormSpace:
        _domain("character basis requires a canonical modular-form space")
    group_name = getattr(space, "group", None)
    level = getattr(space, "level", None)
    weight = getattr(space, "weight", None)
    kind = getattr(space, "kind", None)
    raw_field = getattr(space, "coefficient_domain", None)
    character = getattr(space, "character", None)
    if (
        group_name != "GAMMA0"
        or type(level) is not int
        or level != 13
        or type(weight) is not int
        or weight != 2
        or kind != "S"
        or type(raw_field) is not RationalCyclotomicField
        or getattr(raw_field, "domain", None) != "QQ_CYCLOTOMIC"
        or type(getattr(raw_field, "order", None)) is not int
        or getattr(raw_field, "order", None) != 6
        or getattr(raw_field, "generator", None) != "CLASS_OF_X"
        or type(character) is not DirichletCharacter
        or getattr(character, "group", None) is None
    ):
        _domain(
            "this exact basis currently supports S2(Gamma0(13), chi) for an "
            "even order-6 character over Q(zeta_6)"
        )
    field = raw_field
    coordinates = getattr(character, "coordinates", None)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != 1
        or type(coordinates[0]) is not int
        or coordinates[0] not in (2, 10)
    ):
        _domain("character coordinates must be exactly (2,) or (10,)")
    # The adapter repeats these claim checks before it builds its worker payload.
    character_request = _pari_character_request(space)
    return space, field, character_request


def _coefficient(
    field: RationalCyclotomicField, coordinates: tuple[Fraction, ...]
) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=value.numerator, den=value.denominator)
            for value in coordinates
        ),
    )


def modular_character_basis_q_expansions(
    space: ModularFormSpace,
) -> ModularCharacterBasis:
    """Construct the exact Sturm-determining q-prefix basis for the admitted space."""
    space, field, character_request = _require_space(space)
    return _character_basis_from_admission(space, field, character_request)


def _character_basis_from_admission(
    space: ModularFormSpace,
    field: RationalCyclotomicField,
    character_request: dict[str, object],
) -> ModularCharacterBasis:
    """Construct the basis after source and character admission has completed."""
    # The Cohen-Oesterle formula gives dimension 1 for coordinate 2 modulo 13;
    # coordinate 10 is its Galois conjugate. Quer, Thm. 2.3, gives the
    # independent Gamma0 character dimension formula.
    # Admission is complete before entering PARI or constructing coefficients.
    precision = _STURM_PRECISION
    work = precision * 14 * 2 + precision * field.degree * 16
    if field.degree != 2:
        _domain("the admitted order-6 coefficient field must have degree 2")
    # This line is a normalized weight-2 newform because the admitted cusp
    # space is one-dimensional. Its Hecke coefficients are algebraic integers.
    # For n <= 2, Deligne gives |a_n^sigma| <= 2*sqrt(2) in both embeddings.
    # Write a_n = u + v*zeta_6 with integers u,v. Subtracting conjugates and
    # using |zeta_6-zeta_6_bar|=sqrt(3) gives |v| <= 4*sqrt(2)/sqrt(3) < 3.27,
    # hence |v| <= 3. Then |u| <= 2*sqrt(2)+3 < 5.83, hence |u| <= 5.
    # Thus every power-basis coordinate through q^2 has one decimal digit.
    # The isolated worker enforces that output bound before framing the response.
    normalized_digits = _MAX_NORMALIZED_COORDINATE_DIGITS
    if field.degree != 2 or normalized_digits > 256:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_basis_height_admission",
            message="normalized character coefficients exceed the exact height envelope",
        )
    allocation_bytes = precision * field.degree * (2 * normalized_digits + 32)
    if work > _MAX_WORK or allocation_bytes > _MAX_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_basis_admission",
            message="character-valued basis work or output exceeds its exact envelope",
        )

    # The adapter canonicalizes character coordinates once; the isolated PARI
    # worker independently compares that character on every unit residue.
    raw_basis = pari_character_basis(
        space,
        precision,
        _DIMENSION,
        character_request=character_request,
    )
    if len(raw_basis) != _DIMENSION:
        raise RuntimeError("PARI returned a character basis of the wrong dimension")

    vector = raw_basis[0]
    pivot = next((i for i, term in enumerate(vector) if any(term)), None)
    if pivot is None or pivot >= _STURM_PRECISION:
        raise RuntimeError("PARI character basis has no Sturm-visible pivot")
    normalized = tuple(_coefficient(field, term) for term in vector)
    pivot_value = normalized[pivot]
    one = _coefficient(field, (Fraction(1),) + (Fraction(0),) * (field.degree - 1))
    if pivot != 1 or pivot_value != one:
        raise RuntimeError("character basis normalization failed exact pivot check")

    expansion = ModularCharacterQExpansion(
        space=space,
        basis_id=CHARACTER_BASIS_ID,
        coefficients=normalized,
    )
    return ModularCharacterBasis(
        space=space,
        basis_id=CHARACTER_BASIS_ID,
        precision=precision,
        elements=(
            ModularCharacterBasisElement(label=f"q^{pivot}", expansion=expansion),
        ),
    )


def _admit_character_form(
    form: ModularFormCoordinates,
    admitted_space: tuple[ModularFormSpace, RationalCyclotomicField, dict[str, object]]
    | None = None,
) -> tuple[
    ModularFormSpace,
    RationalCyclotomicField,
    RationalCyclotomicElement,
    dict[str, object],
]:
    if type(form) is not ModularFormCoordinates:
        _domain("character form must be a canonical ModularFormCoordinates value")
    if admitted_space is None:
        space, field, character_request = _require_space(form.space)
    else:
        space, field, character_request = admitted_space
        if form.space != space:
            _domain("character form must use the already-admitted exact space")
    if (
        form.basis_id != CHARACTER_BASIS_ID
        or type(form.coordinates) is not tuple
        or len(form.coordinates) != 1
    ):
        _domain(
            "character form basis or coordinate axis is incompatible with the admitted space"
        )
    scalar = form.coordinates[0]
    if type(scalar) is not RationalCyclotomicElement:
        _domain("character form coordinate must be a canonical cyclotomic element")
    if scalar.field != field:
        _domain("character form coordinate must use the exact space coefficient field")
    _, _, digits = cyclotomic._validate_element(scalar)
    # The product kernel's sound input bound is D*(2*d+2)+log10(d)+2.
    # Account for each basis coefficient's one-digit envelope before invoking it.
    predicted_digits = (2 * field.degree + 2) * max(digits, 1) + 4
    if predicted_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates", 0),
            code="modular_form.character_coordinate_height_admission",
            message="character coordinate height exceeds the exact multiplication envelope",
        )
    work = 3 * field.degree**2 * 3
    allocation_bytes = 3 * field.degree * (2 * predicted_digits + 32)
    if work > _MAX_WORK or allocation_bytes > _MAX_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_coordinate_admission",
            message="character q-prefix work or output exceeds its admitted envelope",
        )
    return space, field, scalar, character_request


def _character_form_prefix(
    form: ModularFormCoordinates,
    admitted: tuple[
        ModularFormSpace,
        RationalCyclotomicField,
        RationalCyclotomicElement,
        dict[str, object],
    ],
    basis: ModularCharacterBasis,
) -> ModularCharacterQExpansion:
    space, field, scalar, _ = admitted
    if basis.space != space or basis.basis_id != form.basis_id:
        _domain("character basis must match the exact form space and basis identifier")
    if form.basis_id != CHARACTER_BASIS_ID:
        _domain("character basis identifier must match the admitted character basis")
    if not any(value.num for value in scalar.coefficients_ascending):
        zero = RationalCyclotomicElement(
            field=field,
            coefficients_ascending=tuple(
                CanonicalRational(num=0, den=1) for _ in range(field.degree)
            ),
        )
        return ModularCharacterQExpansion(
            space=space,
            basis_id=CHARACTER_BASIS_ID,
            coefficients=(zero, zero, zero),
        )
    basis_values = basis.elements[0].expansion.coefficients
    coefficients = tuple(cyclotomic.multiply(scalar, value) for value in basis_values)
    return ModularCharacterQExpansion(
        space=space, basis_id=CHARACTER_BASIS_ID, coefficients=coefficients
    )


def modular_character_coordinates_q_expansion(
    form: ModularFormCoordinates,
) -> ModularCharacterQExpansion:
    """Realize an exact character coordinate through its Sturm-determining prefix."""
    admitted = _admit_character_form(form)
    if not any(value.num for value in admitted[2].coefficients_ascending):
        space, field, _, _ = admitted
        zero = RationalCyclotomicElement(
            field=field,
            coefficients_ascending=tuple(
                CanonicalRational(num=0, den=1) for _ in range(field.degree)
            ),
        )
        return ModularCharacterQExpansion(
            space=space,
            basis_id=CHARACTER_BASIS_ID,
            coefficients=(zero, zero, zero),
        )
    basis = _character_basis_from_admission(*admitted[:2], admitted[3])
    return _character_form_prefix(form, admitted, basis)


def modular_character_coordinates_product(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> ModularFormFieldQExpansion:
    """Return the target Sturm prefix of the conjugate character-form product."""
    left_admitted = _admit_character_form(left)
    right_admitted = _admit_character_form(right)
    left_space, field, left_scalar, left_request = left_admitted
    right_space, right_field, right_scalar, right_request = right_admitted
    left_character = left_space.character
    right_character = right_space.character
    if not isinstance(left_character, DirichletCharacter) or not isinstance(
        right_character, DirichletCharacter
    ):
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.character_product_parent",
            message=(
                "the supported character product requires the two conjugate "
                "order-6 spaces S2(Gamma0(13), chi_2) and S2(Gamma0(13), chi_10)"
            ),
        )
    if (
        field != right_field
        or left_space.level != right_space.level
        or left_space.weight != right_space.weight
        or left_space.kind != right_space.kind
        or left_space.level != 13
        or left_space.weight != 2
        or left_space.kind != "S"
        or (left_character.coordinates, right_character.coordinates)
        not in (((2,), (10,)), ((10,), (2,)))
    ):
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.character_product_parent",
            message=(
                "the supported character product requires the two conjugate "
                "order-6 spaces S2(Gamma0(13), chi_2) and S2(Gamma0(13), chi_10)"
            ),
        )

    # Rational scalar multiples preserve Galois conjugacy of the two
    # normalized eigenforms, so their product has rational q-coefficients.
    left_scalar_values = cyclotomic._validate_element(left_scalar)[1]
    right_scalar_values = cyclotomic._validate_element(right_scalar)[1]
    if left_scalar_values[1] or right_scalar_values[1]:
        raise OperationDomainValidationError(
            location=("form", "coordinates"),
            code="modular_form.character_product_scalar_field",
            message="conjugate character products currently require rational scalar coordinates",
        )

    rational_target_space = ModularFormSpace(level=13, weight=4, kind="S")
    target_space = ModularFormSpace(
        level=13,
        weight=4,
        kind="S",
        coefficient_domain=field,
    )
    precision = sturm_bound(rational_target_space).bound + 1
    if precision > MAX_PARI_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("target_space",),
            code="modular_form.character_product_precision_bound",
            message="target Sturm prefix exceeds the exact PARI basis envelope",
        )
    left_digits = cyclotomic._validate_element(left_scalar)[2]
    right_digits = cyclotomic._validate_element(right_scalar)[2]
    # The private worker admits four-digit power-basis coordinates. Use that
    # full cap here too: Deligne bounds the embeddings, while the worker's
    # independent height guard is the actual encoded-coordinate envelope.
    coefficient_digits = _MAX_CHARACTER_HECKE_COEFFICIENT_DIGITS
    scalar_product_digits = left_digits + right_digits
    # In Q(zeta_6), (a+b*zeta)(c+d*zeta) has coordinates
    # (ac-bd, ad+bc+bd), since zeta^2=zeta-1. Clearing the six input
    # denominators bounds each product coordinate by 6*h+2 digits. Summing
    # at most `precision` terms and scaling by rational source coordinates
    # therefore gives the admitted output height below.
    product_digits = (
        precision * (6 * coefficient_digits + 2 + scalar_product_digits)
        + len(str(precision))
        + 2
    )
    work = precision * precision * field.degree**2 * 16
    allocation_bytes = (
        2 * precision * field.degree * (2 * coefficient_digits + 32)
        + precision * field.degree * (2 * product_digits + 32)
        + 2_048
    )
    if (
        work > _MAX_WORK
        or allocation_bytes > _MAX_ALLOCATION_BYTES
        or product_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_product_admission",
            message="character product exceeds its exact work, height, or output envelope",
        )
    if left_space == right_space and left_scalar == right_scalar:
        request_checkpoint("before character coordinate equality")
        return ModularFormFieldQExpansion(
            space=target_space,
            coefficients=tuple(
                _coefficient(field, cyclotomic._validate_element(value)[1])
                for value in left_scalar.coefficients_ascending
            )
            + tuple(
                _coefficient(field, (Fraction(0),) * field.degree) for _ in range(2)
            ),
        )
    if not any(value.num for value in left_scalar.coefficients_ascending) or not any(
        value.num for value in right_scalar.coefficients_ascending
    ):
        zero = _coefficient(field, (Fraction(0),) * field.degree)
        return ModularFormFieldQExpansion(
            space=target_space,
            coefficients=tuple(zero for _ in range(precision)),
        )

    request_checkpoint("before conjugate character basis expansion for product")
    left_basis = pari_character_basis(
        left_space, precision, _DIMENSION, character_request=left_request
    )
    right_basis = pari_character_basis(
        right_space, precision, _DIMENSION, character_request=right_request
    )
    if (
        len(left_basis) != 1
        or len(right_basis) != 1
        or len(left_basis[0]) != precision
        or len(right_basis[0]) != precision
    ):
        raise RuntimeError("PARI returned a malformed character product prefix")
    left_coefficients = tuple(
        cyclotomic._validate_element(_coefficient(field, term))[1]
        for term in left_basis[0]
    )
    right_coefficients = tuple(
        cyclotomic._validate_element(_coefficient(field, term))[1]
        for term in right_basis[0]
    )
    scalar_product = left_scalar_values[0] * right_scalar_values[0]
    product = []
    for degree in range(precision):
        request_checkpoint("during exact conjugate character q-series product")
        rational = [Fraction(0), Fraction(0)]
        for left_index in range(degree + 1):
            a, b = left_coefficients[left_index]
            c, d = right_coefficients[degree - left_index]
            rational[0] += a * c - b * d
            rational[1] += a * d + b * c + b * d
        rational = [entry * scalar_product for entry in rational]
        if rational[1]:
            raise RuntimeError(
                "conjugate character-form product failed exact rational descent"
            )
        product.append(_coefficient(field, tuple(rational)))
    request_checkpoint("after exact character product Sturm-prefix convolution")
    return ModularFormFieldQExpansion(
        space=target_space,
        coefficients=tuple(product),
    )


def _character_root_of_unity(
    character: DirichletCharacter,
    integer: int,
    field: RationalCyclotomicField,
) -> RationalCyclotomicElement:
    """Embed one admitted character value into the declared power basis."""
    group = character.group
    residue = integer % group.modulus
    if residue not in group.unit_residues:
        return _coefficient(field, (Fraction(0),) * field.degree)
    row_index = group.unit_residues.index(residue)
    coordinates = group.unit_coordinates[row_index]
    exponent = (
        sum(
            coordinate * (group.exponent // order) * unit_coordinate
            for coordinate, order, unit_coordinate in zip(
                character.coordinates, group.generator_orders, coordinates, strict=True
            )
        )
        % group.exponent
    )
    embedded_numerator = exponent * field.order
    if embedded_numerator % group.exponent:
        raise OperationDomainValidationError(
            location=("form", "space", "coefficient_domain"),
            code="modular_form.character_hecke_field",
            message="the coefficient field does not contain this Hecke character value",
        )
    power = (embedded_numerator // group.exponent) % field.order
    result = _coefficient(
        field,
        (Fraction(1),) + (Fraction(0),) * (field.degree - 1),
    )
    root = _coefficient(
        field,
        (Fraction(0), Fraction(1)) + (Fraction(0),) * (field.degree - 2),
    )
    for _ in range(power):
        result = cyclotomic.multiply(result, root)
    return result


def _scale_cyclotomic(
    value: RationalCyclotomicElement, scalar: int
) -> RationalCyclotomicElement:
    field, coefficients, _ = cyclotomic._validate_element(value)
    return _coefficient(
        field, tuple(coefficient * scalar for coefficient in coefficients)
    )


def _hecke_character_coefficient(
    coefficients: tuple[RationalCyclotomicElement, ...],
    character: DirichletCharacter,
    field: RationalCyclotomicField,
    index: int,
    output_index: int,
) -> RationalCyclotomicElement:
    zero = _coefficient(field, (Fraction(0),) * field.degree)
    result = zero
    for divisor in range(1, index + 1):
        if gcd(output_index, index) % divisor:
            continue
        source_index = output_index * index // (divisor * divisor)
        character_value = _character_root_of_unity(character, divisor, field)
        term = cyclotomic.multiply(character_value, coefficients[source_index])
        term = _scale_cyclotomic(term, divisor)
        result = cyclotomic.add(result, term)
    return result


def modular_character_coordinates_hecke(
    form: ModularFormCoordinates, index: int
) -> ModularFormCoordinates:
    """Apply ``T_n`` to the admitted one-dimensional ``S_2(Gamma0(13), chi)``.

    The image is reconstructed in the same exact cyclotomic coordinate parent
    and checked through its Sturm bound. The private PARI prefix extends to
    ``n * B + 1`` terms, where this space has ``B = 2``.
    """
    admitted = _admit_character_form(form)
    if type(index) is not int or not 1 <= index <= MAX_CHARACTER_HECKE_INDEX:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.character_hecke_index_bound",
            message=(
                "character-valued Hecke indices must lie in "
                f"[1, {MAX_CHARACTER_HECKE_INDEX}]"
            ),
        )
    if gcd(index, admitted[0].level) != 1:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.character_hecke_coprime_level",
            message="T_n on this character space currently requires gcd(n, 13) = 1",
        )
    if index == 1:
        return form
    scalar = admitted[2]
    if not any(value.num for value in scalar.coefficients_ascending):
        return form

    precision = 2 * index + 1
    maximum_index = precision - 1
    # Deligne plus the Hecke recurrence gives an embedding bound
    # |sigma(a_m)| <= sigma_1(m) <= 1 + ... + m. PARI returns power-basis
    # coordinates a_m=u+v*zeta_6, which are not themselves bounded by that
    # embedding bound. Since both conjugates have size <= B and
    # |zeta_6-zeta_6_bar|=sqrt(3), we have |v| < 2B and |u| < 3B.
    embedding_magnitude_bound = maximum_index * (maximum_index + 1) // 2
    coordinate_magnitude_bound = 3 * embedding_magnitude_bound
    coordinate_digits = len(str(max(1, coordinate_magnitude_bound)))
    predicted_coordinate_digits = (
        max(cyclotomic._validate_element(scalar)[2], coordinate_digits)
        * (2 * admitted[1].degree + 2)
        + len(str(admitted[1].degree))
        + 2
    )
    hecke_intermediate_digits = (
        coordinate_digits * (2 * admitted[1].degree + 2) + 2 * len(str(index)) + 8
    )
    work = precision * admitted[1].degree * 64 + index * 16
    allocation_bytes = (
        precision * admitted[1].degree * (2 * coordinate_digits + 32)
        + 3 * admitted[1].degree * (2 * hecke_intermediate_digits + 48)
        + admitted[1].degree * (2 * predicted_coordinate_digits + 48)
        + 2_048
    )
    if (
        precision > MAX_CHARACTER_HECKE_SOURCE_PRECISION
        or coordinate_digits > _MAX_CHARACTER_HECKE_COEFFICIENT_DIGITS
        or hecke_intermediate_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        or predicted_coordinate_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        or work > _MAX_WORK
        or allocation_bytes > _MAX_ALLOCATION_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.character_hecke_admission",
            message="character-valued Hecke source, work, or output exceeds its exact envelope",
        )

    request_checkpoint("before character-valued Hecke basis expansion")
    raw_basis = pari_character_basis(
        admitted[0],
        precision,
        _DIMENSION,
        character_request=admitted[3],
    )
    if len(raw_basis) != 1 or len(raw_basis[0]) != precision:
        raise RuntimeError("PARI returned a malformed character Hecke source prefix")
    basis_coefficients = tuple(_coefficient(admitted[1], term) for term in raw_basis[0])
    if basis_coefficients[1] != _coefficient(admitted[1], (Fraction(1), Fraction(0))):
        raise RuntimeError("normalized character basis must have q coefficient one")
    if any(
        denominator != 1 or abs(numerator) > coordinate_magnitude_bound
        for coefficient in basis_coefficients
        for numerator, denominator in (
            (part.num, part.den) for part in coefficient.coefficients_ascending
        )
    ):
        raise RuntimeError(
            "PARI character coefficients exceeded the admitted power-basis bound"
        )

    character = admitted[0].character
    if not isinstance(character, DirichletCharacter):
        raise OperationDomainValidationError(
            location=("form", "space", "character"),
            code="modular_form.character_hecke_character",
            message="character Hecke action requires an exact Dirichlet character",
        )
    transformed_values = []
    for output_index in range(3):
        request_checkpoint("during character-valued Hecke coefficient reconstruction")
        transformed_values.append(
            _hecke_character_coefficient(
                basis_coefficients, character, admitted[1], index, output_index
            )
        )
    transformed = tuple(transformed_values)
    eigenvalue = transformed[1]
    if any(
        transformed[output_index]
        != cyclotomic.multiply(eigenvalue, basis_coefficients[output_index])
        for output_index in range(3)
    ):
        raise RuntimeError(
            "Hecke image failed exact reconstruction through the Sturm bound"
        )
    result_coordinate = cyclotomic.multiply(eigenvalue, scalar)
    request_checkpoint("after character-valued Hecke reconstruction")
    return ModularFormCoordinates(
        space=admitted[0],
        basis_id=CHARACTER_BASIS_ID,
        coordinates=(result_coordinate,),
    )


def modular_character_hecke_matrix(
    space: ModularFormSpace, index: int
) -> ModularCharacterHeckeMatrix:
    """Return the exact 1-by-1 T_n matrix in the represented order-six basis.

    The single basis vector is the normalized character newform. The scalar is
    obtained by applying the already bounded coefficient reconstruction to
    that vector, so the source precision and coefficient-height admission is
    shared with the coordinate action.
    """
    admitted_space, field, _ = _require_space(space)
    if type(index) is not int or not 1 <= index <= MAX_CHARACTER_HECKE_INDEX:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.character_hecke_index_bound",
            message=(
                "character-valued Hecke indices must lie in "
                f"[1, {MAX_CHARACTER_HECKE_INDEX}]"
            ),
        )
    if gcd(index, admitted_space.level) != 1:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.character_hecke_coprime_level",
            message="T_n on this character space currently requires gcd(n, 13) = 1",
        )
    one = _coefficient(field, (Fraction(1),) + (Fraction(0),) * (field.degree - 1))
    normalized = ModularFormCoordinates(
        space=admitted_space,
        basis_id=CHARACTER_BASIS_ID,
        coordinates=(one,),
    )
    # The delegated coordinate operation pre-admits n*B+1 source precision,
    # exact coefficient height, work, and output before entering the PARI
    # basis worker or reconstructing the Hecke image.
    image = modular_character_coordinates_hecke(normalized, index)
    image_scalar = image.coordinates[0]
    if type(image_scalar) is not RationalCyclotomicElement:
        raise RuntimeError("character Hecke image has a non-cyclotomic coordinate")
    return ModularCharacterHeckeMatrix(
        space=admitted_space,
        basis_id=CHARACTER_BASIS_ID,
        index=index,
        entries=((image_scalar,),),
    )


__all__ = [
    "CHARACTER_BASIS_ID",
    "modular_character_basis_q_expansions",
    "modular_character_coordinates_hecke",
    "modular_character_coordinates_product",
    "modular_character_coordinates_q_expansion",
    "modular_character_hecke_matrix",
]
