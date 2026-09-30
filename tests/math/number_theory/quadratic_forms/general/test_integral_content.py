"""Integral coefficient content of rational quadratic forms."""

from fractions import Fraction
from itertools import combinations, islice

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._models import (
    IntegralContentRequest,
    IntegralContentResult,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    integral_coefficient_content,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _q(value: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, denominator)


def test_integral_content_and_primitive_part_include_cross_coefficients() -> None:
    form = RationalQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_q(-12), _q(0), _q(18)),
        cross_terms=(QuadraticCrossTerm(left=0, right=2, coefficient=_q(30)),),
    )

    result = integral_coefficient_content(form)

    assert result.content == 6
    assert tuple(
        value.as_fraction() for value in result.primitive_part.diagonal_coefficients
    ) == (Fraction(-2), Fraction(0), Fraction(3))
    assert result.primitive_part.cross_terms[0].coefficient.as_fraction() == 5
    assert result.primitive_part.axis == form.axis


def test_large_content_uses_decimal_string_json_encoding() -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(10**20),))
    result = integral_coefficient_content(form)
    assert result.content == 10**20
    assert result.model_dump(mode="json")["content"] == "100000000000000000000"


@pytest.mark.parametrize("axis", [(), ("x",)])
def test_zero_form_has_zero_content_and_zero_primitive_part(
    axis: tuple[str, ...],
) -> None:
    form = RationalQuadraticForm(
        axis=axis, diagonal_coefficients=tuple(_q(0) for _ in axis)
    )
    result = integral_coefficient_content(form)
    result = IntegralContentResult.model_validate_json(result.model_dump_json())
    assert result.content == 0
    assert result.primitive_part == form
    assert integral_coefficient_content(result.primitive_part).content == 0


def test_integral_content_accepts_coefficient_digit_boundary() -> None:
    edge = 10**256 - 1
    form = RationalQuadraticForm(
        axis=("x", "y"), diagonal_coefficients=(_q(-edge), _q(edge))
    )
    result = integral_coefficient_content(form)
    restored = IntegralContentResult.model_validate_json(result.model_dump_json())
    assert restored.content == edge
    assert restored.form == form
    assert restored.primitive_part.diagonal_coefficients == (_q(-1), _q(1))


def test_integral_content_rejects_rational_coefficients() -> None:
    rational = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(1, 2),))
    request = IntegralContentRequest(form=rational)
    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(request.form)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.nonintegral_form"


def test_integral_content_refuses_oversized_support_as_a_resource_bound() -> None:
    """The retained-support ceiling is a capacity, not a request-validity rule.

    A form past the traversal limit is a well-formed integral form that this
    operation declines to enumerate, so the refusal carries the operation's
    resource-admission classification rather than a request validation error.
    """
    axis = tuple(f"x{i}" for i in range(4097))
    oversized = RationalQuadraticForm(
        axis=axis, diagonal_coefficients=tuple(_q(1) for _ in axis)
    )
    admitted = IntegralContentRequest(form=oversized)
    with pytest.raises(OperationResourceAdmissionError) as refusal:
        integral_coefficient_content(admitted.form)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.invariant_support_bound"


@pytest.mark.parametrize("diagonal_count,cross_count", [(4097, 0), (91, 4006)])
def test_integral_content_refuses_support_before_serializing(
    monkeypatch: pytest.MonkeyPatch, diagonal_count: int, cross_count: int
) -> None:
    axis = tuple(f"x{i}" for i in range(diagonal_count))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(_q(6) for _ in axis),
        cross_terms=tuple(
            QuadraticCrossTerm(left=left, right=right, coefficient=_q(10))
            for left, right in islice(
                combinations(range(diagonal_count), 2), cross_count
            )
        ),
    )

    def refuse_dump(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("over-budget forms must not be serialized")

    monkeypatch.setattr(RationalQuadraticForm, "model_dump", refuse_dump)
    with pytest.raises(OperationResourceAdmissionError) as refusal:
        integral_coefficient_content(form)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.invariant_support_bound"


def test_integral_content_at_support_boundary_round_trips_to_primitive_content() -> (
    None
):
    axis = tuple(f"x{i}" for i in range(4093))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(_q(6) for _ in axis),
        cross_terms=tuple(
            QuadraticCrossTerm(left=0, right=right, coefficient=_q(10))
            for right in (1, 2, 3)
        ),
    )

    result = integral_coefficient_content(form)
    restored = IntegralContentResult.model_validate_json(result.model_dump_json())

    assert restored.content == 2
    assert restored.form == form
    assert restored.primitive_part.axis == axis
    assert all(
        value.as_fraction() == 3
        for value in restored.primitive_part.diagonal_coefficients
    )
    assert all(
        term.coefficient.as_fraction() == 5
        for term in restored.primitive_part.cross_terms
    )
    assert integral_coefficient_content(restored.primitive_part).content == 1


def test_integral_content_refuses_a_forged_form_before_reading_coefficients() -> None:
    """A native caller can build a form whose collections disagree.

    The kernel reads the diagonal positionally, so a forged short diagonal
    would otherwise yield a plausible but meaningless content. The carrier's own
    structural invariants are re-established at this trust boundary first.
    """
    forged = RationalQuadraticForm.model_construct(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_q(6), _q(10)),
        cross_terms=(),
    )
    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(forged)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


@pytest.mark.parametrize("field", ["axis", "diagonal_coefficients", "cross_terms"])
def test_integral_content_refuses_missing_native_collections(field: str) -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(6),))
    forged = form.model_copy(update={field: None})

    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(forged)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("axis", (["x"] * 4097, "y")),
        ("axis", ("x" * 65, "y")),
        ("diagonal_coefficients", ((_q(6),) * 4097, _q(10))),
        (
            "diagonal_coefficients",
            (CanonicalRational.model_construct(num=[6], den=1), _q(10)),
        ),
        (
            "diagonal_coefficients",
            (CanonicalRational.model_construct(num=10**256, den=1), _q(10)),
        ),
        (
            "diagonal_coefficients",
            (CanonicalRational.model_construct(num=6, den=None), _q(10)),
        ),
        ("cross_terms", (("nested",) * 4097,)),
        (
            "cross_terms",
            (QuadraticCrossTerm.model_construct(left=[], right=1, coefficient=_q(2)),),
        ),
        (
            "cross_terms",
            (QuadraticCrossTerm.model_construct(left=0, right=1, coefficient=[2]),),
        ),
        ("domain", ["QQ"]),
    ],
)
def test_integral_content_rejects_forged_leaves_before_serializing(
    field: str, replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    form = RationalQuadraticForm(axis=("x", "y"), diagonal_coefficients=(_q(6), _q(10)))
    forged = form.model_copy(update={field: replacement})

    def refuse_dump(*_args: object, **_kwargs: object) -> None:
        pytest.fail("malformed native leaves must not be serialized")

    monkeypatch.setattr(RationalQuadraticForm, "model_dump", refuse_dump)
    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(forged)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


@pytest.mark.parametrize(
    "coefficient",
    [
        CanonicalRational.model_construct(num=6, den=0),
        CanonicalRational.model_construct(num=6, den=2),
        CanonicalRational.model_construct(num=True, den=1),
    ],
)
def test_integral_content_rejects_forged_scalar_invariants(
    coefficient: CanonicalRational,
) -> None:
    forged = RationalQuadraticForm.model_construct(
        axis=("x",), diagonal_coefficients=(coefficient,)
    )
    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(forged)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


def test_integral_content_refuses_mismatched_axis_before_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = RationalQuadraticForm.model_construct(
        axis=tuple(f"x{i}" for i in range(4097)), diagonal_coefficients=(_q(6),)
    )

    def refuse_dump(*_args: object, **_kwargs: object) -> None:
        pytest.fail("inconsistent form collections must not be serialized")

    monkeypatch.setattr(RationalQuadraticForm, "model_dump", refuse_dump)
    with pytest.raises(OperationDomainValidationError) as refusal:
        integral_coefficient_content(forged)
    assert refusal.value.errors()[0]["type"] == "quadratic_form.form_structure"


def test_integral_content_still_returns_the_content_of_a_well_formed_form() -> None:
    """Negative control: revalidation must not change an accepted result."""
    form = RationalQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_q(6), _q(10), _q(15)),
        cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_q(21)),),
    )
    result = integral_coefficient_content(form)
    assert result.content == 1
    assert result.primitive_part == form
