"""Character-valued V-degeneracies into explicit inflated target spaces."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    _admit_character_form,
    _character_basis_from_admission,
    _character_form_prefix,
)
from jacobian.math.number_theory.modular_forms.character_degeneracy_models import (
    ModularCharacterVDegeneracyRequest,
)
from jacobian.math.number_theory.modular_forms.space_maps import (
    modular_form_character_space_inclusion,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormFieldQExpansion,
    ModularFormSpace,
)

_MAX_WORK = 50_000
_MAX_OUTPUT_BYTES = 1_000_000


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple({"num": 0, "den": 1} for _ in range(field.degree)),
    )


def modular_character_coordinates_v_degeneracy(
    request: ModularCharacterVDegeneracyRequest,
) -> ModularFormFieldQExpansion:
    """Return the known finite prefix of ``V_d(f)(q)=f(q^d)``.

    The source is either normalized form in ``S_2(Gamma0(13), chi)`` for the
    admitted order-six characters over ``Q(zeta_6)``. The request supplies its
    exact target space at level 26 or 39. The character-inflation inclusion is
    checked before the source q-prefix is materialized. The returned prefix
    ends at the image of the source's Sturm-determining prefix, so no unknown
    source coefficient is treated as zero.
    """
    if type(request) is not ModularCharacterVDegeneracyRequest:
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.character_v_degeneracy_request_type",
            message="character V-degeneracy requires a canonical request",
        )
    form = request.form
    target_space = request.target_space
    if type(target_space) is not ModularFormSpace:
        raise OperationDomainValidationError(
            location=("target_space",),
            code="modular_form.character_v_degeneracy_target_type",
            message="V-degeneracy requires an exact modular-form target space",
        )

    admitted = _admit_character_form(form)
    source_space, field, scalar, _ = admitted
    if type(target_space.level) is not int or target_space.level not in (26, 39):
        raise OperationDomainValidationError(
            location=("target_space", "level"),
            code="modular_form.character_v_degeneracy_target_level",
            message="character V-degeneracy target level must be 26 or 39",
        )
    try:
        target_space = ModularFormSpace.model_validate(target_space.model_dump())
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("target_space",),
            code="modular_form.character_v_degeneracy_target_invalid",
            message="target space must be a canonical exact modular-form space",
        ) from error
    d, remainder = divmod(target_space.level, source_space.level)
    if (
        remainder
        or d not in (2, 3)
        or source_space.level != 13
        or source_space.weight != 2
        or source_space.kind != "S"
        or target_space.group != "GAMMA0"
        or target_space.weight != source_space.weight
        or target_space.kind != source_space.kind
        or target_space.coefficient_domain != field
    ):
        raise OperationDomainValidationError(
            location=("target_space",),
            code="modular_form.character_v_degeneracy_target_space",
            message=(
                "character V-degeneracy requires an explicit S2 target at level "
                "26 or 39 over the identical Q(zeta_6) parent"
            ),
        )

    # The source basis is Sturm-determining at three coefficients. V_d has an
    # exact output prefix through q^(2d), since the last known source term is q^2.
    output_precision = 2 * d + 1
    _, _, coordinate_digits = cyclotomic._validate_element(scalar)
    result_digits = (2 * field.degree + 2) * max(coordinate_digits, 1) + 4
    output_bytes = (
        len(encode_strict_json(target_space.model_dump(mode="json")))
        + output_precision * field.degree * (2 * result_digits + 32)
        + 512
    )
    work = 10_000 + 3 * field.degree**2 + output_precision
    if (
        result_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        or output_bytes > _MAX_OUTPUT_BYTES
        or work > _MAX_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.character_v_degeneracy_admission",
            message="character V-degeneracy work, coefficient growth, or output exceeds its exact envelope",
        )

    # Build the exact typed space map and check the pullback character relation
    # on each target unit after all request work and output have been admitted.
    modular_form_character_space_inclusion(source_space, target_space)
    request_checkpoint("before character V-degeneracy source expansion")
    basis = _character_basis_from_admission(source_space, field, admitted[3])
    source_expansion = _character_form_prefix(form, admitted, basis)
    if len(source_expansion.coefficients) != 3:
        raise RuntimeError("admitted character source must have Sturm precision three")
    zero = _zero(field)
    coefficients = tuple(
        source_expansion.coefficients[index // d] if index % d == 0 else zero
        for index in range(output_precision)
    )
    request_checkpoint("after character V-degeneracy source expansion")
    return ModularFormFieldQExpansion(space=target_space, coefficients=coefficients)
