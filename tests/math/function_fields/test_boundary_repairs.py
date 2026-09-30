"""Boundary and trust regressions for the function-field owner.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldPlace,
    HyperellipticInfinityPlace,
    HyperellipticInfinityPlaceValuationResult,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_element_norm,
    function_field_hyperelliptic_infinity_valuation,
)
from jacobian.math.function_fields import operations as ff_operations
from jacobian.math.function_fields._models import (
    MAX_RIEMANN_ROCH_MEMBERSHIP_OUTPUT_CELLS,
    MAX_RIEMANN_ROCH_MEMBERSHIP_PROFILE_ROWS,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldFiniteValuation,
)
from jacobian.math.function_fields._tools import TOOLS


def _rf(
    numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=denominator),
    )


def _field(branch: tuple[int, ...] = (0, 1, 0, 4)) -> FiniteFunctionField:
    """``y^2 = f(x)`` over GF(5) for the given branch polynomial."""
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rf(tuple(-value % 5 for value in branch)),
            _rf((0,)),
            _rf((1,)),
        ),
    )


def _residue() -> FiniteFieldPresentation:
    return FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="z"
    )


def _typed_infinity(field: FiniteFunctionField) -> HyperellipticInfinityPlace:
    return HyperellipticInfinityPlace(field=field, residue_field=_residue())


def _element(
    field: FiniteFunctionField,
    c0: tuple[int, ...] = (0,),
    c1: tuple[int, ...] = (0,),
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=(_rf(c0), _rf(c1)))


def test_norm_work_envelope_is_decided_before_field_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused norm must not run an exact Gauss-lemma factorization.

    Field admission answers a domain question through the exact
    irreducibility path, while the norm envelope is a cheap growth question
    about coordinates the operand preflight has already bounded.
    """
    order: list[str] = []
    admit_field = ff_operations.__dict__["_admit_field"]
    admit_growth = ff_operations.__dict__["_admit_norm_growth"]

    def recording_field(field: FiniteFunctionField) -> None:
        order.append("admit_field")
        admit_field(field)

    def recording_growth(field: FiniteFunctionField, canonical: object) -> None:
        order.append("admit_norm_growth")
        admit_growth(field, canonical)

    monkeypatch.setattr(ff_operations, "_admit_field", recording_field)
    monkeypatch.setattr(ff_operations, "_admit_norm_growth", recording_growth)
    field = _field()
    element = _element(field, (0,), (1,))  # y, whose norm is -x

    function_field_element_norm(element)

    assert order == ["admit_norm_growth", "admit_field"]

    order.clear()
    monkeypatch.setattr(ff_operations, "MAX_NORM_WORK", 0)

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_norm(element)

    assert (
        error.value.errors()[0]["type"] == "function_field.norm_work_exceeds_envelope"
    )
    assert order == ["admit_norm_growth"]


def test_generic_infinity_place_is_admitted_on_an_extension_field() -> None:
    """A generic INFINITE place must reach the infinity kind check.

    Rational-place admission refuses every supported ``y^2 = f(x)``
    extension, so delegating to it made the documented generic place form
    unreachable on exactly the fields this operation exists for.
    """
    field = _field()
    element = _element(field, (0,), (1,))  # y
    generic = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)

    result = function_field_hyperelliptic_infinity_valuation(generic, element)

    assert isinstance(result.valuation, FunctionFieldFiniteValuation)
    assert result.valuation.value == -3

    wrong_kind = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_hyperelliptic_infinity_valuation(wrong_kind, element)
    assert error.value.errors()[0]["type"] == "function_field.infinity_place_type"


def test_infinity_valuation_result_rejects_mismatched_parents() -> None:
    """A valuation is meaningless without a place-element pairing."""
    cubic = _field()
    quintic = _field((0, 1, 0, 0, 0, 4))
    element = _element(quintic, (0, 1))

    with pytest.raises(ValueError, match="function_field"):
        HyperellipticInfinityPlaceValuationResult.model_validate(
            {
                "place": _typed_infinity(cubic).model_dump(),
                "element": element.model_dump(),
                "valuation": FunctionFieldFiniteValuation(
                    kind="FINITE", value=0
                ).model_dump(),
            }
        )


def test_infinity_valuation_result_accepts_matching_parents() -> None:
    field = _field()
    result = function_field_hyperelliptic_infinity_valuation(
        _typed_infinity(field), _element(field, (0,), (1,))
    )

    revived = HyperellipticInfinityPlaceValuationResult.model_validate(
        result.model_dump()
    )

    assert revived == result
    assert revived.place.field == revived.element.field


def test_divisor_support_composes_with_the_infinity_valuation() -> None:
    """The support a divisor reports must be usable by its own operation.

    ``_preflight_hyperelliptic_infinity_divisor`` returns generic
    ``FunctionFieldPlace`` values, so a caller could not feed them to the
    published infinity valuation without rebuilding the place by hand.
    """
    field = _field()
    divisor = FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(
                place=FunctionFieldPlace(field=field, kind="INFINITE", degree=1),
                multiplicity=3,
            ),
        ),
    )
    space = ff_operations.function_field_riemann_roch_space(divisor)
    support = space.divisor.terms[0].place

    result = function_field_hyperelliptic_infinity_valuation(
        support, _element(field, (0,), (1,))
    )

    assert isinstance(result.valuation, FunctionFieldFiniteValuation)
    assert result.valuation.value == -3


def test_published_operations_are_exported_from_the_native_package() -> None:
    """Native and catalog clients must reach the same operations."""
    import jacobian.math.function_fields as package

    for tool in TOOLS:
        assert tool.operation_id.startswith("function_field.")
    for name in (
        "function_field_hyperelliptic_affine_valuation",
        "function_field_hyperelliptic_infinity_valuation",
        "function_field_riemann_roch_membership",
        "HyperellipticInfinityPlace",
        "HyperellipticInfinityPlaceValuationResult",
        "FunctionFieldRiemannRochMembership",
    ):
        assert name in package.__all__, name
        assert getattr(package, name) is not None
    # Wire models stay private to the delivery boundary.
    assert not any(name.endswith(("Request", "Input")) for name in package.__all__)


def test_norm_description_no_longer_promises_pre_expansion_output_size() -> None:
    """The exact norm degree cannot be known before the determinant."""
    description = next(
        tool.description
        for tool in TOOLS
        if tool.operation_id == "function_field.element.norm.compute"
    )

    assert "output size are admitted before" not in description
    assert "exact reduced norm degree is checked on the normalized result" in (
        description
    )


def test_membership_description_reports_cells_instead_of_encoded_bytes() -> None:
    """The operation bounds profile cells; encoded size is delivery's job."""
    description = next(
        tool.description
        for tool in TOOLS
        if tool.operation_id == "function_field.riemann_roch.membership.compute"
    )

    assert "MiB" not in description
    assert str(MAX_RIEMANN_ROCH_MEMBERSHIP_PROFILE_ROWS) in description
    assert str(MAX_RIEMANN_ROCH_MEMBERSHIP_OUTPUT_CELLS) in description
