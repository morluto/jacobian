"""Bounded field-valued modular-form bases for one admitted character family."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal

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
    MAX_CHARACTER_BASIS_PRECISION,
    ModularCharacterBasis,
    ModularCharacterBasisElement,
    ModularCharacterHeckeMatrix,
    ModularCharacterQExpansion,
)
from jacobian.math.number_theory.modular_forms.character_dimensions import (
    character_space_dimensions,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
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

CHARACTER_BASIS_ID = "gamma0-13-even-order6-character-sturm-v1"
_DIMENSION = 1
_STURM_PRECISION = 3  # floor(2 * [SL2(Z):Gamma0(13)] / 12) + 1 = 3
_MAX_WORK = 50_000_000
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_CHARACTER_HECKE_INDEX = 32
MAX_CHARACTER_HECKE_SOURCE_PRECISION = 2 * MAX_CHARACTER_HECKE_INDEX + 1
_MAX_CHARACTER_HECKE_COEFFICIENT_DIGITS = 4


def _domain(message: str) -> None:
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
    field = getattr(space, "coefficient_domain", None)
    character = getattr(space, "character", None)
    if (
        group_name != "GAMMA0"
        or type(level) is not int
        or level != 13
        or type(weight) is not int
        or weight != 2
        or kind != "S"
        or type(field) is not RationalCyclotomicField
        or getattr(field, "domain", None) != "QQ_CYCLOTOMIC"
        or type(getattr(field, "order", None)) is not int
        or getattr(field, "order", None) != 6
        or getattr(field, "generator", None) != "CLASS_OF_X"
        or type(character) is not DirichletCharacter
        or getattr(character, "group", None) is None
    ):
        _domain(
            "this character-form operation supports S2(Gamma0(13), chi) over Q(zeta_6)"
        )
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


def _require_basis_space(
    space: ModularFormSpace,
) -> tuple[ModularFormSpace, RationalCyclotomicField, dict[str, object]]:
    """Admit the exact level-13 character and its explicit level inflations."""
    if type(space) is not ModularFormSpace:
        _domain("character basis requires a canonical modular-form space")
    field = space.coefficient_domain
    character = space.character
    if (
        space.group != "GAMMA0"
        or type(space.level) is not int
        or space.level not in (13, 26, 39)
        or type(space.weight) is not int
        or space.weight != 2
        or space.kind not in ("M", "S")
        or type(field) is not RationalCyclotomicField
        or field.domain != "QQ_CYCLOTOMIC"
        or type(field.order) is not int
        or field.order != 6
        or field.generator != "CLASS_OF_X"
        or type(character) is not DirichletCharacter
        or character.group.modulus != space.level
    ):
        _domain(
            "character basis supports weight-two conductor-13 characters at levels 13, 26, and 39 over Q(zeta_6)"
        )
    request = _pari_character_request(space)
    return space, field, request


def _character_sturm_precision(space: ModularFormSpace) -> int:
    level = space.level
    index = level
    for prime in (2, 3, 13):
        if level % prime == 0:
            index = index * (prime + 1) // prime
    return (space.weight * index) // 12 + 1


def _is_zero(value: RationalCyclotomicElement) -> bool:
    return all(
        int(coefficient.num) == 0 for coefficient in value.coefficients_ascending
    )


def _rref_character_prefix(
    vectors: tuple[tuple[tuple[Fraction, ...], ...], ...],
    field: RationalCyclotomicField,
    precision: int,
    *,
    normalization_precision: int | None = None,
) -> tuple[tuple[RationalCyclotomicElement, ...], ...]:
    """Canonical row frame of the backend subspace over its declared field."""
    retain_precision = normalization_precision is not None
    if (
        normalization_precision is not None
        and not 1 <= normalization_precision <= precision
    ):
        raise ValueError(
            "Sturm normalization precision must fit in the retained prefix"
        )
    if retain_precision and any(len(vector) < precision for vector in vectors):
        raise RuntimeError("PARI character basis is shorter than the requested prefix")
    rows = [
        [
            _coefficient(field, term)
            for term in (vector if retain_precision else vector[:precision])
        ]
        for vector in vectors
    ]
    pivot_row = 0
    for column in range(normalization_precision or precision):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if not _is_zero(rows[row][column])
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][column]
        rows[pivot_row] = [
            cyclotomic.divide(value, pivot_value) for value in rows[pivot_row]
        ]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            scale = rows[row][column]
            if _is_zero(scale):
                continue
            rows[row] = [
                cyclotomic.subtract(value, cyclotomic.multiply(scale, pivot_value))
                for value, pivot_value in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    if pivot_row != len(rows):
        raise RuntimeError("PARI character basis is dependent through the Sturm bound")
    return tuple(tuple(row) for row in rows)


def _coefficient(
    field: RationalCyclotomicField, coordinates: tuple[Fraction, ...]
) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            {"num": value.numerator, "den": value.denominator} for value in coordinates
        ),
    )


def modular_character_basis_q_expansions(
    space: ModularFormSpace,
    precision: int | None = None,
) -> ModularCharacterBasis:
    """Construct a canonical basis prefix of at least Sturm-determining precision."""
    space, field, character_request = _require_basis_space(space)
    return _character_basis_from_admission(
        space, field, character_request, requested_precision=precision
    )


def _character_basis_from_admission(
    space: ModularFormSpace,
    field: RationalCyclotomicField,
    character_request: dict[str, object],
    *,
    requested_precision: int | None = None,
) -> ModularCharacterBasis:
    """Construct the basis after source and character admission has completed."""
    # Quer, Thm. 2.3, gives the exact independent dimension formula for this
    # bounded character family. Admission is complete before entering PARI.
    character = space.character
    if type(character) is not DirichletCharacter:
        raise RuntimeError("admitted character basis lost its explicit character")
    cusp_dimension, full_dimension = character_space_dimensions(
        space.level, space.weight, character, field
    )
    dimension = cusp_dimension if space.kind == "S" else full_dimension
    sturm_precision = _character_sturm_precision(space)
    if requested_precision is None:
        precision = sturm_precision
    elif type(requested_precision) is not int:
        raise OperationDomainValidationError(
            location=("precision",),
            code="modular_form.character_basis_precision_type",
            message="requested precision must be an exact integer",
        )
    elif requested_precision < sturm_precision:
        raise OperationDomainValidationError(
            location=("precision",),
            code="modular_form.character_basis_precision_below_sturm",
            message="the returned basis prefix must include every Sturm pivot",
        )
    elif requested_precision > MAX_CHARACTER_BASIS_PRECISION:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.character_basis_precision_bound",
            message="requested basis prefix exceeds the admitted precision bound",
        )
    else:
        precision = requested_precision
    work = precision * max(1, dimension) ** 2 * field.degree * 16
    if field.degree != 2 or precision > MAX_PARI_BASIS_PRECISION or dimension > 32:
        _domain(
            "the admitted character basis exceeds its field, precision, or dimension bound"
        )
    # Reserve the complete value-type coefficient envelope before materializing
    # the backend basis and the cyclotomic RREF output.
    normalized_digits = MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    if field.degree != 2:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_basis_height_admission",
            message="normalized character coefficients exceed the exact height envelope",
        )
    output_bytes = (
        max(1, dimension) * precision * field.degree * (2 * normalized_digits + 32)
    )
    if work > _MAX_WORK or output_bytes > _MAX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_basis_admission",
            message="character-valued basis work or output exceeds its exact envelope",
        )

    # The adapter canonicalizes character coordinates once; the isolated PARI
    # worker independently compares that character on every unit residue.
    # Row reduction performs at most dimension pivots. Each pivot normalizes
    # one dimension-by-precision row, then eliminates at most dimension - 1
    # rows across that same precision. `work` above bounds these coefficient
    # updates with a conservative field-degree factor. Pivot columns are all
    # inside the Sturm prefix, so the height of every transformed coefficient
    # is controlled by a dimension-by-degree pivot minor and is independent of
    # the retained trailing prefix length. Admit that exact intermediate
    # envelope before the backend runs; canonical result values retain the
    # separate 256-digit coefficient cap.
    intermediate_digits = 2 * (dimension * field.degree) ** 2 * 30 + 128
    if intermediate_digits > 100_000:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_basis_height_admission",
            message="cyclotomic Sturm row reduction exceeds its exact intermediate height envelope",
        )
    raw_basis = pari_character_basis(
        space,
        precision,
        dimension,
        character_request=character_request,
    )
    if len(raw_basis) != dimension:
        raise RuntimeError("PARI returned a character basis of the wrong dimension")
    normalized = _rref_character_prefix(
        raw_basis,
        field,
        precision,
        normalization_precision=sturm_precision,
    )
    basis_id: Literal[
        "gamma0-13-even-order6-character-sturm-v1",
        "gamma0-cyclotomic-character-sturm-rref-v1",
    ] = (
        "gamma0-13-even-order6-character-sturm-v1"
        if (
            space.level == 13
            and space.kind == "S"
            and type(space.character) is DirichletCharacter
            and space.character.coordinates in ((2,), (10,))
        )
        else "gamma0-cyclotomic-character-sturm-rref-v1"
    )
    elements = tuple(
        ModularCharacterBasisElement(
            label=f"q^{next(index for index, coefficient in enumerate(vector) if not _is_zero(coefficient))}",
            expansion=ModularCharacterQExpansion(
                space=space, basis_id=basis_id, coefficients=vector
            ),
        )
        for vector in normalized
    )
    return ModularCharacterBasis(
        space=space,
        basis_id=basis_id,
        precision=precision,
        elements=elements,
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
    output_bytes = 3 * field.degree * (2 * predicted_digits + 32)
    if work > _MAX_WORK or output_bytes > _MAX_OUTPUT_BYTES:
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
    if not any(value.num for value in scalar.coefficients_ascending):
        zero = RationalCyclotomicElement(
            field=field,
            coefficients_ascending=tuple(
                {"num": 0, "den": 1} for _ in range(field.degree)
            ),
        )
        return ModularCharacterQExpansion(
            space=space,
            basis_id=form.basis_id,
            coefficients=(zero, zero, zero),
        )
    basis_values = basis.elements[0].expansion.coefficients
    coefficients = tuple(cyclotomic.multiply(scalar, value) for value in basis_values)
    return ModularCharacterQExpansion(
        space=space, basis_id=form.basis_id, coefficients=coefficients
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
                {"num": 0, "den": 1} for _ in range(field.degree)
            ),
        )
        return ModularCharacterQExpansion(
            space=space,
            basis_id=form.basis_id,
            coefficients=(zero, zero, zero),
        )
    basis = _character_basis_from_admission(*admitted[:2], admitted[3])
    return _character_form_prefix(form, admitted, basis)


def modular_character_coordinates_equal(
    left: ModularFormCoordinates,
    right: ModularFormCoordinates,
) -> bool:
    """Decide global equality by comparing the common exact Sturm prefix."""
    left_admitted = _admit_character_form(left)
    if type(right) is not ModularFormCoordinates:
        _domain("right character form must be a canonical ModularFormCoordinates value")
    if right.space != left_admitted[0] or right.basis_id != left.basis_id:
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.character_equality_parent",
            message="character equality requires the identical space and basis; no implicit embedding is defined",
        )
    right_admitted = _admit_character_form(
        right, (*left_admitted[:2], left_admitted[3])
    )
    left_space = left_admitted[0]
    right_space = right_admitted[0]
    if left_space != right_space or left.basis_id != right.basis_id:
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.character_equality_parent",
            message="character equality requires the identical space and basis; no implicit embedding is defined",
        )
    left_zero = not any(value.num for value in left_admitted[2].coefficients_ascending)
    right_zero = not any(
        value.num for value in right_admitted[2].coefficients_ascending
    )
    if left_zero and right_zero:
        return True
    basis = _character_basis_from_admission(
        left_space, left_admitted[1], left_admitted[3]
    )
    left_prefix = _character_form_prefix(left, left_admitted, basis)
    right_prefix = _character_form_prefix(right, right_admitted, basis)
    return left_prefix.coefficients == right_prefix.coefficients


def modular_character_coordinates_product(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> ModularFormFieldQExpansion:
    """Return the target Sturm prefix of the conjugate character-form product."""
    left_admitted = _admit_character_form(left)
    right_admitted = _admit_character_form(right)
    left_space, field, left_scalar, left_request = left_admitted
    right_space, right_field, right_scalar, right_request = right_admitted
    if (
        field != right_field
        or left_space.level != right_space.level
        or left_space.weight != right_space.weight
        or left_space.kind != right_space.kind
        or left_space.level != 13
        or left_space.weight != 2
        or left_space.kind != "S"
        or (left_space.character.coordinates, right_space.character.coordinates)
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
    output_bytes = (
        2 * precision * field.degree * (2 * coefficient_digits + 32)
        + precision * field.degree * (2 * product_digits + 32)
        + 2_048
    )
    if (
        work > _MAX_WORK
        or output_bytes > _MAX_OUTPUT_BYTES
        or product_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_product_admission",
            message="character product exceeds its exact work, height, or output envelope",
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


def _order_six_product_coordinates(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, Fraction]:
    """Multiply two Q(zeta_6) values using zeta_6^2 = zeta_6 - 1."""
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c + b * d


def _order_six_character_hecke_coefficient(
    coefficients: tuple[RationalCyclotomicElement, ...],
    character: DirichletCharacter,
    field: RationalCyclotomicField,
    index: int,
    output_index: int,
) -> RationalCyclotomicElement:
    """Apply the weight-two divisor formula with admitted exact arithmetic."""
    result = (Fraction(0), Fraction(0))
    for divisor in range(1, index + 1):
        if gcd(output_index, index) % divisor:
            continue
        source_index = output_index * index // (divisor * divisor)
        source = cyclotomic._validate_element(coefficients[source_index])[1]
        root = cyclotomic._validate_element(
            _character_root_of_unity(character, divisor, field)
        )[1]
        term = _order_six_product_coordinates(source, root)
        result = tuple(
            left + divisor * right for left, right in zip(result, term, strict=True)
        )
    return _coefficient(field, result)


def _rref_character_coordinates_hecke(
    form: ModularFormCoordinates, index: int
) -> ModularFormCoordinates:
    """Apply T_n to a one-dimensional canonical RREF character space.

    The operator is an endomorphism: the exact space and basis identifier are
    retained. A multidimensional character matrix requires its own admitted
    matrix result contract and is not inferred from this scalar action.
    """
    if type(form) is not ModularFormCoordinates:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.character_coordinates_type",
            message="character Hecke requires a canonical ModularFormCoordinates value",
        )
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        CHARACTER_RREF_BASIS_ID,
        _admit_coordinate_space,
        _admit_coordinate_vector,
    )

    context = _admit_coordinate_space(form.space)
    _admit_coordinate_vector(form, context, "form")
    if form.basis_id != CHARACTER_RREF_BASIS_ID:
        raise OperationDomainValidationError(
            location=("form", "basis_id"),
            code="modular_form.character_hecke_basis",
            message="the generalized character Hecke action requires the canonical q-Sturm RREF basis",
        )
    if type(index) is not int or not 1 <= index <= MAX_CHARACTER_HECKE_INDEX:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.character_hecke_index_bound",
            message=(
                "character-valued Hecke indices must lie in "
                f"[1, {MAX_CHARACTER_HECKE_INDEX}]"
            ),
        )
    if gcd(index, context.space.level) != 1:
        raise OperationDomainValidationError(
            location=("index",),
            code="modular_form.character_hecke_coprime_level",
            message="T_n on a character space currently requires gcd(n, level) = 1",
        )
    if context.dimension == 0 or index == 1:
        return form
    if context.dimension != 1:
        raise OperationDomainValidationError(
            location=("form", "space"),
            code="modular_form.character_hecke_dimension",
            message="the generalized character Hecke action currently supports one-dimensional q-Sturm RREF spaces",
        )

    space, field, character_request = _require_basis_space(context.space)
    character = space.character
    if type(character) is not DirichletCharacter:
        raise RuntimeError("admitted character space lost its exact character")
    input_digits = max(
        cyclotomic._validate_element(value)[2] for value in form.coordinates
    )
    if not any(
        coefficient.num for coefficient in form.coordinates[0].coefficients_ascending
    ):
        return form

    sturm_precision = _character_sturm_precision(space)
    source_precision = index * (sturm_precision - 1) + 1
    divisor_count = sum(index % divisor == 0 for divisor in range(1, index + 1))
    # PARI's character worker caps each raw power-basis coordinate at four
    # digits. Dividing one raw vector by its first nonzero Sturm coefficient
    # has a 30-digit exact inverse bound in degree two. The following budget
    # includes form scaling, every divisor term, and final eigenvalue scaling.
    basis_digits = 30
    scaled_source_digits = input_digits + basis_digits + 2
    term_digits = scaled_source_digits + len(str(index)) + 4
    hecke_digits = divisor_count * term_digits + len(str(divisor_count)) + 2
    result_digits = hecke_digits + input_digits + 2
    work = (
        source_precision * field.degree * 24
        + sturm_precision * index * field.degree * 12
        + divisor_count * sturm_precision * field.degree * 8
    )
    output_bytes = (
        source_precision * field.degree * (2 * basis_digits + 32)
        + sturm_precision * field.degree * (2 * hecke_digits + 32)
        + field.degree * (2 * result_digits + 32)
        + 2_048
    )
    if (
        source_precision > MAX_CHARACTER_BASIS_PRECISION
        or result_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        or work > _MAX_WORK
        or output_bytes > _MAX_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.character_hecke_admission",
            message="character Hecke precision, work, coefficient growth, or output exceeds its exact envelope",
        )

    request_checkpoint("before canonical character Hecke basis expansion")
    basis = _character_basis_from_admission(
        space, field, character_request, requested_precision=source_precision
    )
    if (
        len(basis.elements) != 1
        or basis.basis_id != CHARACTER_RREF_BASIS_ID
        or basis.precision != source_precision
    ):
        raise RuntimeError("canonical RREF basis changed its admitted Hecke parent")
    basis_coefficients = basis.elements[0].expansion.coefficients
    if any(
        cyclotomic._validate_element(value)[2] > basis_digits
        for value in basis_coefficients
    ):
        raise RuntimeError("canonical RREF basis exceeded its admitted Hecke height")
    pivot = next(
        (term for term, value in enumerate(basis_coefficients) if not _is_zero(value)),
        None,
    )
    if pivot is None or basis_coefficients[pivot] != _coefficient(
        field, (Fraction(1), Fraction(0))
    ):
        raise RuntimeError("canonical one-dimensional RREF basis has no unit pivot")

    scalar = cyclotomic._validate_element(form.coordinates[0])[1]
    source_coefficients = tuple(
        _coefficient(field, _order_six_product_coordinates(scalar, coefficient))
        for coefficient in (
            cyclotomic._validate_element(value)[1] for value in basis_coefficients
        )
    )
    transformed = tuple(
        _order_six_character_hecke_coefficient(
            source_coefficients, character, field, index, output_index
        )
        for output_index in range(sturm_precision)
    )
    eigenvalue = transformed[pivot]
    eigenvalue_coordinates = cyclotomic._validate_element(eigenvalue)[1]
    if any(
        transformed[output_index]
        != _coefficient(
            field,
            _order_six_product_coordinates(
                eigenvalue_coordinates,
                cyclotomic._validate_element(basis_coefficients[output_index])[1],
            ),
        )
        for output_index in range(sturm_precision)
    ):
        raise RuntimeError("Hecke image failed exact RREF Sturm reconstruction")

    result_coordinate = _coefficient(
        field, _order_six_product_coordinates(eigenvalue_coordinates, scalar)
    )
    request_checkpoint("after canonical character Hecke reconstruction")
    return ModularFormCoordinates(
        space=space,
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(result_coordinate,),
    )


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
    """Apply ``T_n`` within an admitted exact character space.

    The legacy order-six level-13 cusp spaces and one-dimensional generalized
    q-Sturm RREF spaces are supported. ``T_n`` is taken at the represented
    level, preserves its exact character space when ``gcd(n, level) = 1``, and
    returns coordinates with the identical space and basis identifier.
    """
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        CHARACTER_RREF_BASIS_ID,
    )

    if (
        type(form) is ModularFormCoordinates
        and form.basis_id == CHARACTER_RREF_BASIS_ID
    ):
        return _rref_character_coordinates_hecke(form, index)
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
    output_bytes = (
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
        or output_bytes > _MAX_OUTPUT_BYTES
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
    assert isinstance(character, DirichletCharacter)
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
    # coefficient height, work, and output before entering the PARI basis worker.
    image = modular_character_coordinates_hecke(normalized, index)
    return ModularCharacterHeckeMatrix(
        space=admitted_space,
        basis_id=CHARACTER_BASIS_ID,
        index=index,
        entries=((image.coordinates[0],),),
    )


__all__ = [
    "CHARACTER_BASIS_ID",
    "modular_character_basis_q_expansions",
    "modular_character_coordinates_hecke",
    "modular_character_coordinates_product",
    "modular_character_coordinates_q_expansion",
    "modular_character_hecke_matrix",
]
