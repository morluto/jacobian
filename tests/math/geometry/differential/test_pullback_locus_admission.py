"""Construction loci use bounded monic factors rather than expanded fractions."""

from fractions import Fraction
from typing import Any

import pytest
from sympy import symbols

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._dag import Expression
from jacobian.math.geometry.differential.pullback import pullback_metric
from jacobian.math.geometry.differential.pullback._locus import PullbackDag
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap


def _metric(
    entries: tuple[Any, ...], axis: tuple[str, ...]
) -> RationalCoordinateMetric:
    components = tuple(rational_function_from_sympy(value, axis) for value in entries)
    guards = tuple(
        dict.fromkeys(
            c.denominator
            for c in components
            if any(any(t.exponents) for t in c.denominator.terms)
        )
    )
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=components,
            retained_nonzero_denominators=guards,
        )
    )


def _map(
    images: tuple[Any, ...], source: tuple[str, ...], target: tuple[str, ...]
) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=source,
        target_coordinates=target,
        components=tuple(
            rational_function_from_sympy(value, source) for value in images
        ),
    )


@pytest.mark.parametrize("degree", (32, 33, 64))
@pytest.mark.parametrize("line", (False, True))
def test_repeated_determinant_factor_is_a_compact_locus(
    degree: int, line: bool
) -> None:
    u, x, y = symbols("u x y")
    metric = _metric((u**degree, 0, 0, u**degree), ("u", "v"))
    axis = ("x",) if line else ("x", "y")
    result = pullback_metric(metric, _map((x, 0 if line else y), axis, ("u", "v")))
    expected = (x**degree,) if line else (x**degree, 0, 0, x**degree)
    assert (
        tuple(rational_function_to_sympy(c) for c in result.pullback.components)
        == expected
    )
    assert result.pullback_locus_guard == (
        rational_function_from_sympy(x, axis).numerator,
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    if not line:
        # The producer's exact tensor feeds a second identity pullback unchanged.
        repeated = pullback_metric(
            RationalCoordinateMetric(tensor=result.pullback), _map((x, y), axis, axis)
        )
        assert repeated.pullback == result.pullback


def test_compact_locus_preserves_permuted_ordered_axes() -> None:
    u, x, y = symbols("u x y")
    result = pullback_metric(
        _metric((u**33, 0, 0, u**33), ("v", "u")), _map((y, x), ("y", "x"), ("v", "u"))
    )
    assert result.pullback.coordinate_axis == ("y", "x")
    assert tuple(rational_function_to_sympy(c) for c in result.pullback.components) == (
        x**33,
        0,
        0,
        x**33,
    )
    assert result.pullback_locus_guard == (
        rational_function_from_sympy(x, ("y", "x")).numerator,
    )


@pytest.mark.parametrize(
    "degree,power", ((2, 63), (2, 64), (32, 127), (64, 127), (-2, 127), (-64, 127))
)
def test_large_nonzero_constant_guard_is_omitted(degree: int, power: int) -> None:
    u = symbols("u")
    result = pullback_metric(
        _metric((u**degree,), ("u",)), _map((10**power,), ("x",), ("u",))
    )
    assert not result.pullback.components[0].numerator.terms
    assert result.pullback_locus_guard == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("reciprocal", (False, True))
def test_zero_map_still_checks_singular_or_undefined_metric(reciprocal: bool) -> None:
    u = symbols("u")
    with pytest.raises(OperationDomainValidationError) as error:
        pullback_metric(
            _metric((1 / u if reciprocal else u**2,), ("u",)),
            _map((0,), ("x",), ("u",)),
        )
    assert (
        error.value.errors()[0]["type"]
        == f"differential_geometry.rational_metric.pullback.{'undefined_metric_locus' if reciprocal else 'singular_metric'}"
    )


@pytest.mark.parametrize("height", (False, True))
def test_actual_component_growth_is_still_refused(height: bool) -> None:
    u, x = symbols("u x")
    with pytest.raises(OperationResourceAdmissionError):
        pullback_metric(
            _metric((10**127 * u if height else u**33,), ("u",)),
            _map((10**127 * x if height else x**2,), ("x",), ("u",)),
        )


def test_scalar_work_boundary_is_inclusive() -> None:
    values = (Expression(Fraction(2)), Expression(Fraction(3)))
    admitted = PullbackDag(1)
    admitted.ledger.work = 50_000_000 - 8
    assert admitted.multiply_checked(*values) == Expression(Fraction(6))
    assert admitted.ledger.work == 50_000_000
    refused = PullbackDag(1)
    refused.ledger.work = 50_000_000 - 7
    with pytest.raises(OperationResourceAdmissionError):
        refused.multiply_checked(*values)


def test_scalar_intermediate_boundary_is_checked_before_constant_locus_omission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.differential.pullback import _locus

    u = symbols("u")
    metric, mapping = _metric((u**2,), ("u",)), _map((8,), ("x",), ("u",))
    monkeypatch.setattr(_locus, "_MAX_SCALAR_BITS", 8)
    assert pullback_metric(metric, mapping).pullback_locus_guard == ()
    monkeypatch.setattr(_locus, "_MAX_SCALAR_BITS", 7)
    with pytest.raises(OperationResourceAdmissionError) as error:
        pullback_metric(metric, mapping)
    assert (
        error.value.errors()[0]["type"]
        == "differential_geometry.rational_metric.pullback.scalar_height"
    )


def test_large_private_scale_crosses_worker_without_host_decimal_limit() -> None:
    u, v, x = symbols("u v x")
    source = _metric((1, 0, 0, u**2), ("u", "v"))
    guard = rational_function_from_sympy(
        u**64 * v**32 + u**64 * v**31, ("u", "v")
    ).numerator
    source = source.model_copy(
        update={
            "tensor": source.tensor.model_copy(
                update={"retained_nonzero_denominators": (guard,)}
            )
        }
    )
    result = pullback_metric(source, _map((10**94, x / 10**80), ("x",), ("u", "v")))
    assert rational_function_to_sympy(result.pullback.components[0]) == 10**28
    expected_guard = rational_function_from_sympy(
        x**32 + 10**80 * x**31, ("x",)
    ).numerator
    assert result.pullback_locus_guard == (expected_guard,)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_expanded_undefined_guard_precedes_structural_singular_determinant() -> None:
    u, v, w, x, y = symbols("u v w x y")
    source = _metric((1, 0, 0, 0, 0, 0, 0, 0, 1), ("u", "v", "w"))
    guard = rational_function_from_sympy(u - v - w, ("u", "v", "w")).numerator
    source = source.model_copy(
        update={
            "tensor": source.tensor.model_copy(
                update={"retained_nonzero_denominators": (guard,)}
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        pullback_metric(source, _map((x + y, x, y), ("x", "y"), ("u", "v", "w")))
    assert (
        error.value.errors()[0]["type"]
        == "differential_geometry.rational_metric.pullback.undefined_metric_locus"
    )


def test_existing_maximal_guard_family_is_not_expanded_past_its_cap() -> None:
    u, v, x, y = symbols("u v x y")
    source = _metric((1, 0, 0, 1), ("u", "v"))
    guards = tuple(
        rational_function_from_sympy(g, ("u", "v")).numerator
        for g in (u * v, *(u + i for i in range(1, 768)))
    )
    source = source.model_copy(
        update={
            "tensor": RationalCoordinateTensor(
                coordinate_axis=("u", "v"),
                variance=("COVARIANT", "COVARIANT"),
                components=source.tensor.components,
                retained_nonzero_denominators=canonical_locus_guards(
                    guards, variable_count=2
                ),
            )
        }
    )
    result = pullback_metric(source, _map((x, y), ("x", "y"), ("u", "v")))
    assert len(result.pullback_locus_guard) == 768
    assert (
        rational_function_from_sympy(x * y, ("x", "y")).numerator
        in result.pullback_locus_guard
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_noncanonical_source_is_rejected_before_zero_determinant() -> None:
    x = symbols("x")
    mapping = _map((x,), ("x",), ("u",))
    component = mapping.components[0]
    mapping = mapping.model_copy(
        update={
            "components": (
                component.model_copy(update={"denominator": component.numerator}),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        pullback_metric(_metric((0,), ("u",)), mapping)
    assert (
        error.value.errors()[0]["type"]
        == "differential_geometry.rational_metric.pullback.noncanonical_source"
    )


@pytest.mark.parametrize("sign", (-1, 1))
def test_private_scale_codec_preserves_large_signed_scalars(sign: int) -> None:
    import sys

    from jacobian.canonical import format_canonical_integer
    from jacobian.math.geometry.differential.metrics._dag_worker import (
        _parse_scalar_integer,
    )

    limit = sys.get_int_max_str_digits()
    value = sign * 10**6016
    assert _parse_scalar_integer(format_canonical_integer(value)) == value
    assert sys.get_int_max_str_digits() == limit


@pytest.mark.parametrize("text", ("", "-", "-0", "+1", "01", "1_0", "٣", "1" * 32769))
def test_private_scale_codec_rejects_noncanonical_or_unbounded_text(text: str) -> None:
    from jacobian.math.geometry.differential.metrics._dag_worker import (
        _parse_scalar_integer,
    )

    with pytest.raises(ValueError, match="SCALE integer"):
        _parse_scalar_integer(text)


def test_compact_guards_do_not_expand_an_unneeded_four_axis_product() -> None:
    u, _v, w, _q, x, y, z, t = symbols("u v w q x y z t")
    target, axis = ("u", "v", "w", "q"), ("x", "y", "z", "t")
    entries = tuple(
        int((i, j) in {(0, 1), (1, 0), (2, 3), (3, 2)})
        for i in range(4)
        for j in range(4)
    )
    source = _metric(entries, target)
    guard = rational_function_from_sympy(u * w, target).numerator
    source = source.model_copy(
        update={
            "tensor": source.tensor.model_copy(
                update={"retained_nonzero_denominators": (guard,)}
            )
        }
    )
    result = pullback_metric(
        source, _map((x**32 * y**32, 1, z**32 * t**32, 1), axis, target)
    )
    assert all(
        not component.numerator.terms for component in result.pullback.components
    )
    assert result.pullback_locus_guard == canonical_locus_guards(
        tuple(
            rational_function_from_sympy(g, axis).numerator
            for g in (x**32 * y**32, z**32 * t**32)
        ),
        variable_count=4,
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_zero_determinant_sentinel_does_not_consume_a_returned_guard_slot() -> None:
    u, x = symbols("u x")
    source = _metric((0,), ("u",))
    guards = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(u + i, ("u",)).numerator for i in range(1, 769)
        ),
        variable_count=1,
    )
    source = source.model_copy(
        update={
            "tensor": source.tensor.model_copy(
                update={"retained_nonzero_denominators": guards}
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        pullback_metric(source, _map((x,), ("x",), ("u",)))
    assert (
        error.value.errors()[0]["type"]
        == "differential_geometry.rational_metric.pullback.singular_metric"
    )
