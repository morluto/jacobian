"""Exact exterior-calculus results remain canonical across producer handoffs."""

from fractions import Fraction
from typing import Literal

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    PolynomialMap,
    PolynomialVectorField,
    PrimitiveResult,
    affine_homotopy_primitive,
    exterior_derivative,
    interior_product,
    lie_derivative,
    pullback,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    axis: tuple[str, ...], *terms: tuple[int | Fraction, tuple[int, ...]]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=axis,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(c)),
                    exponents=e,
                )
                for c, e in terms
                if c
            )
        ),
    )


def _form(
    axis: tuple[str, ...],
    degree: int,
    *components: tuple[tuple[int, ...], RationalPolynomial],
) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm(
        variables=axis,
        degree=degree,
        components=tuple(
            FormComponent(indices=i, coefficient=p) for i, p in components
        ),
    )


def _decode(form: PolynomialDifferentialForm) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm.model_validate_json(form.model_dump_json())


@pytest.mark.parametrize("kind", ("contract", "pullback"))
@pytest.mark.parametrize(
    "coefficient",
    (10**2048, 10**4096 - 1),
    ids=("reported", "old-carrier-product-edge"),
)
def test_multiplicative_producer_output_decodes_and_reaches_consumer(
    kind: str, coefficient: int
) -> None:
    target, source = ("x",), ("t",)
    form = _decode(_form(target, 1, ((0,), _poly(target, (coefficient, (0,))))))
    if kind == "contract":
        field = PolynomialVectorField(
            variables=target, components=(_poly(target, (coefficient, (0,))),)
        )
        result = interior_product(field, form)
        expected = _form(target, 0, ((), _poly(target, (coefficient**2, (0,)))))
    else:
        mapping = PolynomialMap(
            source_variables=source,
            target_variables=target,
            images=(_poly(source, (coefficient, (1,))),),
        )
        result = pullback(mapping, form)
        expected = _form(source, 1, ((0,), _poly(source, (coefficient**2, (0,)))))
    assert result == expected
    decoded = _decode(result)
    assert decoded == expected
    # Apply the actual native and serialized outputs unchanged to a consumer.
    assert (
        exterior_derivative(result)
        == exterior_derivative(decoded)
        == _form(expected.variables, int(expected.degree) + 1)
    )


def test_reported_primitive_round_trips_with_source_and_replays_derivative() -> None:
    axis = ("x",)
    denominator = 6 * 10**4095
    source = _decode(
        _form(axis, 1, ((0,), _poly(axis, (Fraction(1, denominator), (1,)))))
    )
    result = affine_homotopy_primitive(source)
    decoded = PrimitiveResult.model_validate_json(result.model_dump_json())
    assert decoded.source == source
    assert decoded.outcome == "CONSTRUCTED"
    assert decoded.primitive == _form(
        axis, 0, ((), _poly(axis, (Fraction(1, 2 * denominator), (2,))))
    )
    assert result.primitive is not None
    assert decoded.primitive is not None
    assert (
        exterior_derivative(result.primitive)
        == exterior_derivative(decoded.primitive)
        == source
    )


def test_lie_sum_is_admitted_before_trusted_construction() -> None:
    axis = ("x", "y")
    coefficient = 5 * 10**4095
    source = _decode(_form(axis, 1, ((1,), _poly(axis, (coefficient, (1, 0))))))
    field = PolynomialVectorField(
        variables=axis, components=(_poly(axis, (1, (1, 0))), _poly(axis, (1, (0, 1))))
    )
    result = lie_derivative(field, source)
    expected = _form(axis, 1, ((1,), _poly(axis, (2 * coefficient, (1, 0)))))
    assert result == _decode(result) == expected
    assert exterior_derivative(result) == _form(
        axis, 2, ((0, 1), _poly(axis, (2 * coefficient, (0, 0))))
    )


@pytest.mark.parametrize("kind", ("contract", "pullback", "primitive", "lie"))
def test_actual_current_carrier_overflow_is_a_resource_nonconclusion(
    kind: Literal["contract", "pullback", "primitive", "lie"],
) -> None:
    digits = MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
    axis = ("x",)
    with pytest.raises(OperationResourceAdmissionError) as error:
        if kind == "primitive":
            denominator = 6 * 10 ** (digits - 1)
            source = _form(
                axis, 1, ((0,), _poly(axis, (Fraction(1, denominator), (1,))))
            )
            affine_homotopy_primitive(_decode(source))
        elif kind == "lie":
            axes = ("x", "y")
            source = _form(
                axes, 1, ((1,), _poly(axes, (5 * 10 ** (digits - 1), (1, 0))))
            )
            field = PolynomialVectorField(
                variables=axes,
                components=(_poly(axes, (1, (1, 0))), _poly(axes, (1, (0, 1)))),
            )
            lie_derivative(field, _decode(source))
        else:
            coefficient = 10 ** (digits // 2)
            source = _form(axis, 1, ((0,), _poly(axis, (coefficient, (0,)))))
            if kind == "contract":
                field = PolynomialVectorField(
                    variables=axis, components=(_poly(axis, (coefficient, (0,))),)
                )
                interior_product(field, _decode(source))
            else:
                mapping = PolynomialMap(
                    source_variables=axis,
                    target_variables=axis,
                    images=(_poly(axis, (coefficient, (1,))),),
                )
                pullback(mapping, _decode(source))
    assert (
        error.value.errors()[0]["type"]
        == "differential_form.calculus.coefficient_budget"
    )


def test_cross_component_cancellation_precedes_final_height_cap() -> None:
    axis = ("x", "y")
    coefficient = 10 ** (MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS - 1)
    source = _form(
        axis,
        1,
        ((0,), _poly(axis, (coefficient, (0, 0)))),
        ((1,), _poly(axis, (-coefficient, (0, 0)))),
    )
    field = PolynomialVectorField(
        variables=axis, components=(_poly(axis, (coefficient, (0, 0))),) * 2
    )
    # Each c² term exceeds the final carrier, but their exact sum is zero.
    assert _decode(interior_product(field, _decode(source))) == _form(axis, 0)


@pytest.mark.parametrize("power,reason", ((16, "coefficient_budget"), (256, "height")))
def test_constant_pullback_growth_never_reaches_unadmitted_canonical_materialization(
    power: int, reason: str
) -> None:
    source = _form(("x",), 0, ((), _poly(("x",), (1, (power,)))))
    mapping = PolynomialMap(
        source_variables=(),
        target_variables=("x",),
        images=(_poly((), (10**2048, ())),),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        pullback(mapping, _decode(source))
    assert error.value.errors()[0]["type"] == f"differential_form.calculus.{reason}"


def test_maximum_carrier_reciprocals_still_fit_the_binary_wedge_work_budget() -> None:
    axis = ("x",)
    coefficient = 10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS - 1
    left = _form(axis, 0, ((), _poly(axis, (coefficient, (0,)))))
    right = _form(axis, 0, ((), _poly(axis, (Fraction(1, coefficient), (0,)))))
    assert _decode(wedge(_decode(left), _decode(right))) == _form(
        axis, 0, ((), _poly(axis, (1, (0,))))
    )


def test_many_components_keep_linear_aggregate_accounting() -> None:
    from itertools import combinations

    axis = tuple(f"x{i}" for i in range(8))
    polynomial = _poly(
        axis,
        *(
            (1, tuple(p if i == k else 0 for i in range(8)))
            for k in range(8)
            for p in range(32, 0, -1)
        ),
    )
    source = _form(
        axis, 3, *((indices, polynomial) for indices in combinations(range(8), 3))
    )
    expected = _form(
        axis,
        4,
        *(
            (
                indices,
                _poly(
                    axis,
                    *(
                        (
                            (-1 if position % 2 else 1) * p,
                            tuple(p - 1 if i == k else 0 for i in range(8)),
                        )
                        for position, k in enumerate(indices)
                        for p in range(32, 1, -1)
                    ),
                ),
            )
            for indices in combinations(range(8), 4)
        ),
    )
    # Constant derivatives cancel in each 4-basis component; the four axes
    # each contribute the remaining 31 monomials with alternating signs.
    result = exterior_derivative(_decode(source))
    assert _decode(result) == expected
    assert len(result.components) == 70
    assert sum(len(c.coefficient.polynomial.terms) for c in result.components) == 8680


def test_general_calculus_work_boundary_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials.differential_forms import operations

    axis = ("x",)
    source = _form(axis, 1, ((0,), _poly(axis, (2, (0,)))))
    field = PolynomialVectorField(variables=axis, components=(_poly(axis, (2, (0,))),))
    original = operations._CalculusArithmeticBudget
    monkeypatch.setattr(
        operations, "_CalculusArithmeticBudget", lambda: original(work_limit=1)
    )
    assert interior_product(field, source) == _form(
        axis, 0, ((), _poly(axis, (4, (0,))))
    )
    monkeypatch.setattr(
        operations, "_CalculusArithmeticBudget", lambda: original(work_limit=0)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        interior_product(field, source)
    assert error.value.errors()[0]["type"] == "differential_form.calculus.work"


@pytest.mark.parametrize("mutation", ("components", "height", "noncanonical"))
def test_native_field_is_admitted_before_zero_contraction(mutation: str) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    axis = ("x",)
    polynomial = _poly(axis, (1, (0,)))
    field = PolynomialVectorField(variables=axis, components=(polynomial,))
    if mutation == "components":
        field = field.model_copy(update={"components": ()})
    else:
        coefficient = CanonicalRational.model_construct(
            num=10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
            if mutation == "height"
            else 2,
            den=1 if mutation == "height" else 4,
        )
        term = polynomial.polynomial.terms[0].model_copy(
            update={"coefficient": coefficient}
        )
        polynomial = polynomial.model_copy(
            update={
                "polynomial": SparseRationalPolynomial.model_construct(terms=(term,))
            }
        )
        field = field.model_copy(update={"components": (polynomial,)})
    with pytest.raises(OperationDomainValidationError) as error:
        interior_product(field, _form(axis, 0))
    assert error.value.errors()[0]["type"] == "differential_form.field_shape"


@pytest.mark.parametrize("missing", ("variables", "components"))
def test_missing_native_field_attribute_has_typed_admission(missing: str) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    field = (
        PolynomialVectorField.model_construct(components=())
        if missing == "variables"
        else PolynomialVectorField.model_construct(variables=("x",))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        interior_product(field, _form(("x",), 0))
    assert error.value.errors()[0]["type"] == (
        "differential_form.field_axis"
        if missing == "variables"
        else "differential_form.field_shape"
    )
