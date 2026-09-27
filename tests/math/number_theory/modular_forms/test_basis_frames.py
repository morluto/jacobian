from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    modular_form_basis_frame,
    modular_form_coordinates_from_frame,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_to_frame,
)
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormChangeOfBasisFrame,
    ModularFormFramedCoordinates,
)


def _rat(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _m4_level_two() -> ModularFormSpace:
    return ModularFormSpace(level=2, weight=4, kind="M")


def _shifted_frame_request() -> ModularFormBasisFrameRequest:
    return ModularFormBasisFrameRequest(
        space=_m4_level_two(),
        source_basis_id="gamma0-two-weight-2-4-monomials-v1",
        source_labels=("A2^2", "E4"),
        labels=("c1", "c2"),
        # c1=A2^2+E4; c2=E4, with columns expressed in canonical basis.
        entries=((_rat(1), _rat(0)), (_rat(1), _rat(1))),
    )


def _coordinate_form(space: ModularFormSpace, basis_id: str, values: tuple[int, ...]):
    return ModularFormCoordinates(
        space=space,
        basis_id=basis_id,
        coordinates=tuple(_rat(value) for value in values),
    )


def test_m4_gamma0_two_shifted_basis_roundtrips_and_preserves_q_expansion() -> None:
    frame = modular_form_basis_frame(_shifted_frame_request())
    canonical = _coordinate_form(
        _m4_level_two(), "gamma0-two-weight-2-4-monomials-v1", (2, 3)
    )
    framed = modular_form_coordinates_to_frame(frame, canonical)
    assert tuple(value.as_fraction() for value in framed.coordinates) == (
        Fraction(2),
        Fraction(1),
    )
    restored = modular_form_coordinates_from_frame(framed)
    assert restored == canonical

    # Independent q-expansion fixture: A2=1+24q+24q^2+96q^3+24q^4+...
    # so A2^2=(1,48,624,1344,5232), while E4=(1,240,2160,6720,17520).
    # The framed vector 2*c1+c2 therefore has these exact first five terms.
    expanded = modular_form_coordinates_q_expansion(restored, 5)
    assert tuple(
        value.as_fraction() for value in expanded.q_expansion.coefficients
    ) == (Fraction(5), Fraction(816), Fraction(7728), Fraction(22848), Fraction(63024))


def test_frame_conversion_revalidates_after_json_roundtrip() -> None:
    frame = modular_form_basis_frame(_shifted_frame_request())
    restored_frame = ModularFormChangeOfBasisFrame.model_validate_json(
        frame.model_dump_json()
    )
    canonical = _coordinate_form(
        _m4_level_two(), "gamma0-two-weight-2-4-monomials-v1", (2, 3)
    )
    framed = modular_form_coordinates_to_frame(restored_frame, canonical)
    restored_framed = ModularFormFramedCoordinates.model_validate_json(
        framed.model_dump_json()
    )
    assert modular_form_coordinates_from_frame(restored_framed) == canonical


def test_conversion_rejects_wrong_space_and_singular_frame() -> None:
    frame = modular_form_basis_frame(_shifted_frame_request())
    wrong_space = _coordinate_form(
        ModularFormSpace(level=1, weight=4, kind="M"),
        "level-one-e4-e6-monomials-v1",
        (1,),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_to_frame(frame, wrong_space)
    assert error.value.errors()[0]["type"] == "modular_form.frame_wrong_space"

    request = _shifted_frame_request().model_copy(
        update={"entries": ((_rat(1), _rat(1)), (_rat(1), _rat(1)))}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_basis_frame(request)
    assert error.value.errors()[0]["type"] == "modular_form.frame_singular"

    # Serialized or directly constructed values are checked again by consumers.
    singular = request.as_frame()
    canonical = _coordinate_form(
        _m4_level_two(), "gamma0-two-weight-2-4-monomials-v1", (2, 3)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_to_frame(singular, canonical)
    assert error.value.errors()[0]["type"] == "modular_form.frame_singular"


def test_native_frame_functions_reject_wrong_value_types() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_basis_frame(object())  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "modular_form.frame_request_type"

    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_from_frame(object())  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "modular_form.framed_coordinates_type"


def test_frame_source_axis_and_size_are_bound_to_the_canonical_space() -> None:
    request = _shifted_frame_request().model_copy(
        update={"source_labels": ("E4", "A2^2")}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_basis_frame(request)
    assert error.value.errors()[0]["type"] == "modular_form.frame_source_basis"

    with pytest.raises(ValueError):
        _shifted_frame_request().model_copy(
            update={"entries": ((_rat(1),), (_rat(0),))}
        ).as_frame()


def test_frame_request_rejects_malformed_axes_and_matrix_during_parsing() -> None:
    valid = _shifted_frame_request().model_dump(mode="python")
    malformed_requests = (
        {**valid, "source_labels": ()},
        {**valid, "labels": ("", "c2")},
        {**valid, "labels": ("c1", "c1")},
        {**valid, "entries": ((_rat(1),), (_rat(0),))},
    )
    for payload in malformed_requests:
        with pytest.raises(ValidationError):
            ModularFormBasisFrameRequest.model_validate(payload)

    # Empty square frames remain valid for zero-dimensional spaces.
    empty = ModularFormBasisFrameRequest(
        space=ModularFormSpace(level=1, weight=4, kind="S"),
        source_basis_id="level-one-e4-e6-monomials-v1",
        source_labels=(),
        labels=(),
        entries=(),
    )
    assert empty.as_frame().entries == ()


def test_zero_dimensional_space_uses_empty_invertible_frame() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="S")
    frame = modular_form_basis_frame(
        ModularFormBasisFrameRequest(
            space=space,
            source_basis_id="level-one-e4-e6-monomials-v1",
            source_labels=(),
            labels=(),
            entries=(),
        )
    )
    form = _coordinate_form(space, "level-one-e4-e6-monomials-v1", ())
    framed = modular_form_coordinates_to_frame(frame, form)
    assert framed.coordinates == ()
    restored = ModularFormFramedCoordinates.model_validate_json(
        framed.model_dump_json()
    )
    assert modular_form_coordinates_from_frame(restored) == form
