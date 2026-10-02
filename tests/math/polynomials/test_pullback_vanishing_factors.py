"""Pullback cancels alternating factors before substituting coefficients."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    PolynomialMap,
    operations,
    pullback,
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
    axis: tuple[str, ...], indices: tuple[int, ...], coefficient: RationalPolynomial
) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm(
        variables=axis,
        degree=len(indices),
        components=(FormComponent(indices=indices, coefficient=coefficient),),
    )


def _roundtrip(value: PolynomialDifferentialForm) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm.model_validate_json(value.model_dump_json())


@pytest.mark.parametrize("power", (127, 128, 256))
@pytest.mark.parametrize("same_dimension", (False, True))
def test_zero_alternating_factor_precedes_large_substitution(
    power: int, same_dimension: bool
) -> None:
    target = ("x", "y")
    source = ("s", "t") if same_dimension else ("s",)
    image = _poly(source, (1, (2, 0) if same_dimension else (2,)))
    mapping = PolynomialMap(
        source_variables=source, target_variables=target, images=(image, image)
    )
    form = _form(target, (0, 1), _poly(target, (1, (power, 0))))
    result = pullback(
        PolynomialMap.model_validate_json(mapping.model_dump_json()), _roundtrip(form)
    )
    # dF_x=dF_y, so their alternating product vanishes identically.
    assert result.variables == source
    assert result.degree == 2
    assert result.components == ()
    assert _roundtrip(result) == result


def test_canonical_exterior_sum_cancels_distinct_coordinate_choices() -> None:
    source, target = ("s", "t"), ("x", "y")
    image = _poly(source, (1, (2, 0)), (1, (0, 1)))
    mapping = PolynomialMap(
        source_variables=source, target_variables=target, images=(image, image)
    )
    result = pullback(mapping, _form(target, (0, 1), _poly(target, (1, (256, 0)))))
    # (2s ds+dt) wedge itself: the two ds∧dt terms cancel.
    assert result.components == ()


def test_rank_deficient_three_form_preserves_complete_axis() -> None:
    source, target = ("s", "t", "unused"), ("x", "y", "z")
    first, second = _poly(source, (1, (2, 0, 0))), _poly(source, (1, (0, 2, 0)))
    mapping = PolynomialMap(
        source_variables=source,
        target_variables=target,
        images=(first, second, _poly(source, (1, (2, 0, 0)), (1, (0, 2, 0)))),
    )
    result = pullback(
        mapping, _form(target, (0, 1, 2), _poly(target, (1, (256, 0, 0))))
    )
    assert result.variables == source
    assert result.degree == 3
    assert result.components == ()


@pytest.mark.parametrize("zero_coefficient", (False, True))
def test_presolve_refusal_still_checks_for_zero_substituted_coefficient(
    zero_coefficient: bool,
) -> None:
    source, target = ("s", "t"), ("u", "v", "w")
    mapping = PolynomialMap(
        source_variables=source,
        target_variables=target,
        images=(
            _poly(source, (1, (256, 1))),
            _poly(source, (1, (256, 0))),
            _poly(source) if zero_coefficient else _poly(source, (1, (0, 0))),
        ),
    )
    form = _form(target, (0, 1), _poly(target, (1, (0, 0, 1))))
    if zero_coefficient:
        assert pullback(mapping, form).components == ()
    else:
        with pytest.raises(OperationResourceAdmissionError) as error:
            pullback(mapping, form)
        assert (
            error.value.errors()[0]["type"]
            == "differential_form.calculus.exponent_budget"
        )


@pytest.mark.parametrize("power", (127, 128))
def test_genuine_nonzero_output_exponent_boundary(power: int) -> None:
    source, target = ("s", "t"), ("x", "y")
    mapping = PolynomialMap(
        source_variables=source,
        target_variables=target,
        images=(
            _poly(source, (1, (2, 0))),
            _poly(source, (1, (0, 1))),
        ),
    )
    form = _form(target, (0, 1), _poly(target, (1, (power, 0))))
    if power == 127:
        assert _roundtrip(pullback(mapping, form)) == _form(
            source, (0, 1), _poly(source, (2, (255, 0)))
        )
    else:
        with pytest.raises(OperationResourceAdmissionError) as error:
            pullback(mapping, form)
        assert (
            error.value.errors()[0]["type"]
            == "differential_form.calculus.exponent_budget"
        )


def test_identity_pullback_retains_maximum_coefficient_exponent() -> None:
    axis = ("x",)
    mapping = PolynomialMap(
        source_variables=axis, target_variables=axis, images=(_poly(axis, (1, (1,))),)
    )
    form = _form(axis, (0,), _poly(axis, (1, (256,))))
    assert _roundtrip(pullback(mapping, _roundtrip(form))) == form


def test_private_derivative_height_is_not_the_output_carrier_limit() -> None:
    source, target = ("t",), ("x",)
    coefficient = 10**4095
    mapping = PolynomialMap(
        source_variables=source,
        target_variables=target,
        images=(_poly(source, (coefficient, (256,))),),
    )
    form = _form(target, (0,), _poly(target, (Fraction(1, coefficient), (0,))))
    assert _roundtrip(pullback(mapping, _roundtrip(form))) == _form(
        source, (0,), _poly(source, (256, (255,)))
    )


def test_zero_shortcut_preserves_ordered_target_axis_validation() -> None:
    mapping = PolynomialMap(
        source_variables=(),
        target_variables=("x", "y"),
        images=(_poly((), (1, ())), _poly((), (1, ()))),
    )
    form = _form(("y", "x"), (0, 1), _poly(("y", "x"), (1, (0, 0))))
    with pytest.raises(OperationDomainValidationError) as error:
        pullback(mapping, form)
    assert error.value.errors()[0]["type"] == "differential_form.pullback_axis"


def test_presolve_pair_budget_is_cumulative_and_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = ("s", "t")
    mapping = PolynomialMap(
        source_variables=axis,
        target_variables=axis,
        images=(_poly(axis, (2, (1, 0))), _poly(axis, (3, (0, 1)))),
    )
    form = _form(axis, (0, 1), _poly(axis, (1, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_TERM_PAIRS", 4)
    assert pullback(mapping, form) == _form(axis, (0, 1), _poly(axis, (6, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_TERM_PAIRS", 3)
    with pytest.raises(OperationResourceAdmissionError) as error:
        pullback(mapping, form)
    assert error.value.errors()[0]["type"] == "differential_form.pullback.presolve_work"


def test_presolve_scalar_work_is_cumulative_and_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = ("s", "t")
    mapping = PolynomialMap(
        source_variables=axis,
        target_variables=axis,
        images=(_poly(axis, (2, (1, 0))), _poly(axis, (3, (0, 1)))),
    )
    form = _form(axis, (0, 1), _poly(axis, (1, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_DIGIT_WORK", 1)
    assert pullback(mapping, form) == _form(axis, (0, 1), _poly(axis, (6, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_DIGIT_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as error:
        pullback(mapping, form)
    assert error.value.errors()[0]["type"] == "differential_form.pullback.presolve_work"


@pytest.mark.parametrize("size", (4, 8))
def test_scaled_hadamard_volume_cancels_large_scalar_content(size: int) -> None:
    # Sylvester H_n has det(H_n)=n^(n/2) for n=4,8. The map A H_n
    # and coefficient A^-n cancel exactly without changing that volume.
    source = tuple(f"s{i}" for i in range(size))
    target = tuple(f"x{i}" for i in range(size))
    scale = 10 ** (4095 // size)
    mapping = PolynomialMap(
        source_variables=source,
        target_variables=target,
        images=tuple(
            _poly(
                source,
                *(
                    (
                        (-1 if (i & j).bit_count() % 2 else 1) * scale,
                        tuple(int(k == j) for k in range(size)),
                    )
                    for j in range(size)
                ),
            )
            for i in range(size)
        ),
    )
    form = _form(
        target,
        tuple(range(size)),
        _poly(target, (Fraction(1, scale**size), (0,) * size)),
    )
    expected = _form(
        source, tuple(range(size)), _poly(source, (size ** (size // 2), (0,) * size))
    )
    assert _roundtrip(pullback(mapping, _roundtrip(form))) == expected


@pytest.mark.parametrize("mutation", ("images", "axes", "exponent", "coefficient"))
def test_dimensional_zero_does_not_accept_malformed_native_map(mutation: str) -> None:
    source, target = ("s",), ("x", "y")
    image = _poly(source, (1, (2,)))
    mapping = PolynomialMap(
        source_variables=source, target_variables=target, images=(image, image)
    )
    if mutation == "images":
        mapping = mapping.model_copy(update={"images": ()})
    elif mutation == "axes":
        mapping = mapping.model_copy(update={"source_variables": ("s",) * 9})
    else:
        term = image.polynomial.terms[0]
        if mutation == "exponent":
            term = term.model_copy(update={"exponents": (257,)})
        else:
            term = term.model_copy(
                update={
                    "coefficient": CanonicalRational.model_construct(
                        num=1 << (4 * MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS),
                        den=1,
                    )
                }
            )
        image = image.model_copy(
            update={
                "polynomial": SparseRationalPolynomial.model_construct(terms=(term,))
            }
        )
        mapping = mapping.model_copy(update={"images": (image, image)})
    form = _form(target, (0, 1), _poly(target, (1, (256, 0))))
    with pytest.raises(OperationDomainValidationError) as error:
        pullback(mapping, form)
    assert error.value.errors()[0]["type"] == "differential_form.map_shape"


def test_coefficient_first_fallback_preserves_scaled_sparse_cancellation() -> None:
    source, target = ("s", "t"), ("u", "v")
    scale = 10**4095
    exponents = tuple((i, j) for i in range(9, -1, -1) for j in range(9, -1, -1))
    first = _poly(source, *((scale, e) for e in exponents))
    second = _poly(source, *((scale + int(e == (0, 1)), e) for e in exponents))
    mapping = PolynomialMap(
        source_variables=source, target_variables=target, images=(first, second)
    )
    form = _form(target, (0, 1), _poly(target, (Fraction(1, scale), (0, 0))))
    # F=(A*S,A*S+t), so det dF=A*∂S/∂s; A^-1 cancels first.
    expected = _form(
        source,
        (0, 1),
        _poly(
            source,
            *((i, (i - 1, j)) for i in range(9, 0, -1) for j in range(9, -1, -1)),
        ),
    )
    assert _roundtrip(pullback(mapping, _roundtrip(form))) == expected


def test_dense_repeated_differentials_vanish_before_scalar_arithmetic() -> None:
    source, target = ("s", "t"), ("u", "v")
    scale = 10**4095
    image = _poly(
        source, *((scale, (i, j)) for i in range(9, -1, -1) for j in range(9, -1, -1))
    )
    mapping = PolynomialMap(
        source_variables=source, target_variables=target, images=(image, image)
    )
    form = _form(target, (0, 1), _poly(target, (1, (0, 0))))
    assert _roundtrip(pullback(mapping, _roundtrip(form))).components == ()


def test_presolve_scalar_storage_boundary_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = ("s", "t")
    mapping = PolynomialMap(
        source_variables=axis,
        target_variables=axis,
        images=(
            _poly(axis, (2, (1, 0))),
            _poly(axis, (3, (0, 1))),
        ),
    )
    form = _form(axis, (0, 1), _poly(axis, (1, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_STORAGE_DIGITS", 1)
    assert pullback(mapping, form) == _form(axis, (0, 1), _poly(axis, (6, (0, 0))))
    monkeypatch.setattr(operations, "MAX_CALCULUS_STORAGE_DIGITS", 0)
    with pytest.raises(OperationResourceAdmissionError) as error:
        pullback(mapping, form)
    assert (
        error.value.errors()[0]["type"] == "differential_form.pullback.presolve_storage"
    )
