import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_coordinates_atkin_lehner,
    modular_form_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.operations import space_dimension
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _form(
    level: int, weight: int, basis_id: str, values: tuple[int, ...]
) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=level, weight=weight, kind="M"),
        basis_id=basis_id,
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in values),
    )


def _coefficients(form: ModularFormCoordinates, precision: int) -> tuple[Fraction, ...]:
    expansion = modular_form_coordinates_q_expansion(form, precision)
    return tuple(value.as_fraction() for value in expansion.q_expansion.coefficients)


def test_fricke_sends_e4_to_four_times_e4_at_two_tau() -> None:
    # E4 is represented by the second vector of (A2^2, E4).
    e4 = _form(2, 4, "gamma0-two-weight-2-4-monomials-v1", (0, 1))
    image = modular_form_coordinates_atkin_lehner(e4, 2)

    # From E4(-1/z)=z^4 E4(z), normalized slash by W_2 gives
    # 2^(4/2) E4(2z); its coefficients are 4*240*sigma_3(n/2).
    assert _coefficients(image, 6) == (
        Fraction(4),
        Fraction(0),
        Fraction(960),
        Fraction(0),
        Fraction(8_640),
        Fraction(0),
    )
    assert modular_form_coordinates_atkin_lehner(image, 2) == e4


def test_exact_divisor_one_is_the_identity() -> None:
    form = _form(2, 4, "gamma0-two-weight-2-4-monomials-v1", (2, 3))
    assert modular_form_coordinates_atkin_lehner(form, 1) == form


def test_pari_backed_gamma0_level_five_space_composes_after_serialization() -> None:
    # This exercises the generic exact Gamma0 basis path, beyond explicit
    # levels 1 through 4, and the ordinary coordinate transport result.
    space = ModularFormSpace(level=5, weight=4, kind="M")
    dimension = space_dimension(space).dimension
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-rational-gamma0-sturm-rref-v1",
        coordinates=tuple(
            CanonicalRational(num=int(i == 0), den=1) for i in range(dimension)
        ),
    )
    image = modular_form_coordinates_atkin_lehner(form, 5)
    restored = ModularFormCoordinates.model_validate_json(image.model_dump_json())
    assert modular_form_coordinates_atkin_lehner(restored, 5) == form


def test_non_exact_divisor_is_rejected() -> None:
    form = ModularFormCoordinates.model_construct(
        space=ModularFormSpace(level=12, weight=4, kind="M"),
        basis_id="gamma0-rational-gamma0-sturm-rref-v1",
        coordinates=(),
    )
    with pytest.raises(OperationDomainValidationError, match="exact divisor"):
        modular_form_coordinates_atkin_lehner(form, 2)


def test_output_height_is_admitted_before_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    form = _form(
        2,
        28,
        "gamma0-two-weight-2-4-monomials-v1",
        (10**127, 0, 0, 0, 0, 0, 0, 0),
    )

    def backend_must_not_run(*args: object, **kwargs: object) -> None:
        pytest.fail("output-height admission must precede PARI")

    monkeypatch.setattr(
        "jacobian.math.number_theory.modular_forms.basis.pari_gamma0_atkin_matrix",
        backend_must_not_run,
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="worst-case Atkin-Lehner"
    ):
        modular_form_coordinates_atkin_lehner(form, 2)


def test_catalog_declares_the_fricke_transform() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.atkin_lehner.apply"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert tool.run(request).space.level == 2
