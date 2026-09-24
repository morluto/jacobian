"""Exact rational-map composition checked by sparse evaluation oracles."""

from fractions import Fraction
from itertools import product
from time import monotonic

import pytest
import sympy
from sympy import symbols

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions import _bounds as rational_bounds
from jacobian.math.polynomials.rational_functions.composition import (
    RationalFunctionMapComposition,
    compose_maps,
)
from jacobian.math.polynomials.rational_functions.composition import (
    operations as composition_ops,
)
from jacobian.math.polynomials.rational_functions.composition._models import (
    RationalMapCompositionRequest,
)
from jacobian.math.polynomials.rational_functions.composition._tools import TOOLS
from jacobian.math.polynomials.rational_functions.gradient import _gcd_process
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    KernelBatchInputLimitError,
    normalize_admitted_fractions,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial


def _rf(expression: object, variables: tuple[object, ...]) -> RationalFunction:
    return rational_function_from_sympy(expression, tuple(str(v) for v in variables))


def _map(
    source: tuple[str, ...],
    target: tuple[str, ...],
    components: tuple[RationalFunction, ...],
) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=source,
        target_coordinates=target,
        components=components,
    )


def _evaluate_polynomial(
    polynomial: SparseRationalPolynomial, point: tuple[Fraction, ...]
) -> Fraction:
    result = Fraction(0)
    for term in polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        for coordinate, exponent in zip(point, term.exponents, strict=True):
            coefficient *= coordinate**exponent
        result += coefficient
    return result


def _evaluate(value: RationalFunction, point: tuple[Fraction, ...]) -> Fraction:
    numerator = _evaluate_polynomial(value.numerator, point)
    denominator = _evaluate_polynomial(value.denominator, point)
    if not denominator:
        raise ZeroDivisionError
    return numerator / denominator


def test_example_states_the_intermediate_axis_precondition() -> None:
    description = TOOLS[0].examples[0].description
    assert "source_variables equal the inner target_coordinates" in description


def test_native_composition_is_exported_without_exposing_wire_request() -> None:
    import jacobian.math.polynomials.rational_functions.composition as composition

    assert composition.compose_maps is compose_maps
    assert "compose_maps" in composition.__all__
    assert "RationalMapCompositionRequest" not in composition.__all__


def test_native_composition_rejects_non_map_outer() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        compose_maps(object(), object())  # type: ignore[arg-type]

    assert error.value.errors()[0]["loc"] == ("outer",)


def test_native_composition_rejects_non_map_inner() -> None:
    y = symbols("y")
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with pytest.raises(OperationDomainValidationError) as error:
        compose_maps(outer, object())  # type: ignore[arg-type]

    assert error.value.errors()[0]["loc"] == ("inner",)


def test_example_composition_retains_all_construction_guards() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf(x / (x - 1), (x,)), _rf((x + 1) / (x - 2), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z1", "z2"),
        (_rf(y1 + y2, (y1, y2)), _rf(y1 / y2, (y1, y2))),
    )

    result = compose_maps(outer, inner)

    assert result.composite.source_variables == ("x",)
    assert result.composite.target_coordinates == ("z1", "z2")
    assert result.composite.components[0] == _rf(
        (2 * x**2 - 2 * x - 1) / (x**2 - 3 * x + 2), (x,)
    )
    assert result.composite.components[1] == _rf((x**2 - 2 * x) / (x**2 - 1), (x,))
    guard_polynomials = tuple(
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in guard.polynomial.terms
        )
        for guard in result.construction_locus_guard
    )
    assert guard_polynomials == (
        (((1,), Fraction(1)), ((0,), Fraction(-1))),
        (((1,), Fraction(1)), ((0,), Fraction(-2))),
        (((1,), Fraction(1)), ((0,), Fraction(1))),
    )
    assert (
        RationalFunctionMapComposition.model_validate_json(result.model_dump_json())
        == result
    )


def test_cancellation_does_not_erase_stricter_locus() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(x / (x - 1), (x,)),))
    outer = _map(("y",), ("z",), (_rf(1 / y, (y,)),))
    result = compose_maps(outer, inner)

    assert result.composite.components[0] == _rf((x - 1) / x, (x,))
    assert {
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in guard.polynomial.terms
        )
        for guard in result.construction_locus_guard
    } == {
        (((1,), Fraction(1)),),
        (((1,), Fraction(1)), ((0,), Fraction(-1))),
    }


def test_outer_denominator_zero_after_substitution_is_domain_error() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(1, (x,)),))
    outer = _map(("y",), ("z",), (_rf(1 / (y - 1), (y,)),))

    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        compose_maps(outer, inner)


def test_permuted_scaled_monomial_map_preserves_sparse_structure() -> None:
    x, y, u, v = symbols("x y u v")
    inner = _map(
        ("x", "y"),
        ("u", "v"),
        (_rf(y, (x, y)), _rf(2 * x, (x, y))),
    )
    outer = _map(("u", "v"), ("z",), (_rf(u**64 + v, (u, v)),))

    result = compose_maps(outer, inner)
    expected = _rf(y**64 + 2 * x, (x, y))
    assert result.composite.components == (expected,)


def test_monomial_fast_path_handles_every_outer_component() -> None:
    x, y, u, v = symbols("x y u v")
    inner = _map(
        ("x", "y"),
        ("u", "v"),
        (_rf(y, (x, y)), _rf(2 * x, (x, y))),
    )
    outer = _map(
        ("u", "v"),
        ("z1", "z2"),
        (_rf(u**3 + v, (u, v)), _rf(u - v**2, (u, v))),
    )

    result = compose_maps(outer, inner)

    assert result.composite.components == (
        _rf(y**3 + 2 * x, (x, y)),
        _rf(y - 4 * x**2, (x, y)),
    )


def test_substitution_bound_reuses_repeated_exact_powers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.rational_functions.composition.operations as operations

    x, u, v = symbols("x u v")
    polynomial = _rf(u**2 * v**3 + u * v**3, (u, v)).numerator
    ledger = operations._Ledger()
    inner = (
        rational_bounds._fraction_bound(_rf((x + 1) / (x + 2), (x,)), ledger),
        rational_bounds._fraction_bound(_rf((x + 2) / (x + 3), (x,)), ledger),
    )
    calls: list[tuple[object, int]] = []
    original_power = operations._power

    def count_power(
        source: rational_bounds.PolynomialBound,
        exponent: int,
        variable_count: int,
        work: operations._Ledger,
    ) -> rational_bounds.PolynomialBound:
        calls.append((source, exponent))
        return original_power(source, exponent, variable_count, work)

    monkeypatch.setattr(operations, "_power", count_power)
    operations._substitute_bound(polynomial, inner, ledger)
    # The numerator and denominator bounds happen to be structurally equal,
    # but represent distinct backend polynomials; each is powered once.
    repeated = (inner[1].numerator, 3)
    assert calls.count(repeated) == 2

    backend_powers: list[tuple[object, int]] = []
    original_poly_power = sympy.Poly.__pow__

    def count_backend_power(poly: sympy.Poly, exponent: int) -> sympy.Poly:
        backend_powers.append((poly.as_expr(), exponent))
        return original_poly_power(poly, exponent)

    monkeypatch.setattr(sympy.Poly, "__pow__", count_backend_power)
    inner_map = _map(
        ("x",),
        ("u", "v"),
        (_rf((x + 1) / (x + 2), (x,)), _rf((x + 2) / (x + 3), (x,))),
    )
    outer_map = _map(
        ("u", "v"),
        ("z",),
        (_rf(u**2 * v**3 + u * v**3, (u, v)),),
    )
    result = compose_maps(outer_map, inner_map)
    assert backend_powers.count((x + 2, 3)) == 1
    expected = ((x + 1) ** 2 / (x + 2) ** 2 + (x + 1) / (x + 2)) * (
        (x + 2) / (x + 3)
    ) ** 3
    assert result.composite.components == (_rf(expected, (x,)),)


def test_sparse_eight_axis_degree_sixty_four_composition_is_admitted() -> None:
    source = symbols("x0:8")
    intermediate = symbols("y0:8")
    target = symbols("z0:8")
    inner = _map(
        tuple(map(str, source)),
        tuple(map(str, intermediate)),
        tuple(_rf(value, source) for value in source),
    )
    outer = _map(
        tuple(map(str, intermediate)),
        tuple(map(str, target)),
        tuple(_rf(value**64, intermediate) for value in intermediate),
    )

    result = compose_maps(outer, inner)

    assert all(
        component.numerator.terms[0].exponents
        == tuple(64 * (axis == index) for axis in range(8))
        for index, component in enumerate(result.composite.components)
    )


def test_empty_inner_axis_composes_exact_constants() -> None:
    y = symbols("y")
    inner = _map((), ("y",), (_rf(2, ()),))
    outer = _map(("y",), ("z",), (_rf(y**3 - y, (y,)),))

    result = compose_maps(outer, inner)
    assert result.composite.source_variables == ()
    assert result.composite.components[0] == _rf(6, ())
    assert result.construction_locus_guard == ()


def test_zero_dimensional_intermediate_axis_retains_inner_source() -> None:
    x = symbols("x")
    inner = _map(("x",), (), ())
    outer = _map((), ("z",), (_rf(2, ()),))

    result = compose_maps(outer, inner)

    assert result.composite.source_variables == ("x",)
    assert result.composite.target_coordinates == ("z",)
    assert result.composite.components == (_rf(2, (x,)),)
    assert result.construction_locus_guard == ()


def test_composition_matches_independent_fraction_oracle() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) / (x - 3), (x,)), _rf((x - 2) / (x + 2), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z1", "z2"),
        (_rf(y1**2 + y2, (y1, y2)), _rf((y1 - y2) / (y1 + y2), (y1, y2))),
    )
    result = compose_maps(outer, inner)

    for numerator, denominator in product(range(-4, 5), repeat=2):
        point = (Fraction(numerator, denominator or 1),)
        try:
            inner_values = tuple(
                _evaluate(component, point) for component in inner.components
            )
            expected = (
                inner_values[0] ** 2 + inner_values[1],
                (inner_values[0] - inner_values[1])
                / (inner_values[0] + inner_values[1]),
            )
            actual = tuple(
                _evaluate(component, point) for component in result.composite.components
            )
        except ZeroDivisionError:
            continue
        assert actual == expected


def test_axis_mismatch_and_authored_noncanonical_source_are_rejected() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("u",), (_rf(x, (x,)),))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with pytest.raises(ValueError, match="must equal"):
        RationalMapCompositionRequest(outer=outer, inner=inner)

    authored = RationalFunction(
        variables=("x",),
        numerator=_rf((x - 1) * (x + 1), (x,)).numerator,
        denominator=_rf(x - 1, (x,)).numerator,
    )
    bad_inner = _map(("x",), ("y",), (authored,))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with pytest.raises(OperationDomainValidationError, match="coprime"):
        compose_maps(outer, bad_inner)


def test_equal_inner_coordinates_make_an_outer_denominator_undefined() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) ** 64, (x,)), _rf((x + 1) ** 64, (x,))),
    )
    outer = _map(("y1", "y2"), ("z",), (_rf(1 / (y1**2 - y2**2), (y1, y2)),))
    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        compose_maps(outer, inner)


def test_identical_inner_denominators_are_one_construction_guard() -> None:
    x = symbols("x")
    coords = tuple(f"u{index}" for index in range(8))
    inner = _map(
        ("x",),
        coords,
        tuple(_rf(1 / (x + 1), (x,)) for _ in coords),
    )
    outer = _map(coords, ("z",), (_rf(symbols(coords[0]), coords),))
    result = compose_maps(outer, inner)
    assert len(result.construction_locus_guard) == 1


def test_shared_deadline_is_honored() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(x, (x,)),))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with (
        request_execution(monotonic() - 100),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        compose_maps(outer, inner)


def test_scalar_scaled_inner_coordinates_vanish_before_exponent_bound() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) ** 64, (x,)), _rf(-((x + 1) ** 64), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z",),
        (_rf(1 / (y1**2 - y2**2), (y1, y2)),),
    )
    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        compose_maps(outer, inner)


def test_general_rational_scalar_equivalence_vanishes() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) / (x + 2), (x,)), _rf(-2 * (x + 1) / (x + 2), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z",),
        (_rf(1 / (y1 + y2 / 2), (y1, y2)),),
    )
    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        compose_maps(outer, inner)


def test_catalog_composition_batches_source_recognition_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "rational_function_map.compose.compute"
    )
    example = tool.examples[0]
    request = RationalMapCompositionRequest.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    recognition_calls: list[int] = []
    normalization_calls: list[int] = []
    recognize = composition_ops.recognize_canonical_rational_functions
    normalize = normalize_admitted_fractions

    def record_recognition(candidates: tuple[object, ...], *, deadline: float) -> object:
        recognition_calls.append(len(candidates))
        return recognize(candidates, deadline=deadline)  # type: ignore[arg-type]

    def record_normalization(
        pairs: tuple[tuple[object, object], ...], variables: tuple[str, ...]
    ) -> object:
        normalization_calls.append(len(pairs))
        return normalize(pairs, variables)  # type: ignore[arg-type]

    monkeypatch.setattr(
        composition_ops,
        "recognize_canonical_rational_functions",
        record_recognition,
    )
    monkeypatch.setattr(
        composition_ops, "normalize_admitted_fractions", record_normalization
    )

    result = compose_maps(request.outer, request.inner)

    assert recognition_calls == [3]
    assert normalization_calls == [4]
    x = symbols("x")
    assert result.composite.components == (
        _rf((2 * x**2 - 2 * x - 1) / (x**2 - 3 * x + 2), (x,)),
        _rf((x**2 - 2 * x) / (3 * x - 3), (x,)),
    )
    assert len(result.construction_locus_guard) == 2


def test_composition_normalization_batches_respect_row_limit_and_output_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(x / (x - 1), (x,)),))
    outer = _map(
        ("y",),
        tuple(f"z{i}" for i in range(9)),
        tuple(_rf((y + i) / (y + 1), (y,)) for i in range(9)),
    )
    batch_sizes: list[int] = []
    normalize = normalize_admitted_fractions

    def record_normalization(
        pairs: tuple[tuple[object, object], ...], variables: tuple[str, ...]
    ) -> object:
        batch_sizes.append(len(pairs))
        return normalize(pairs, variables)  # type: ignore[arg-type]

    monkeypatch.setattr(
        composition_ops, "normalize_admitted_fractions", record_normalization
    )

    result = compose_maps(outer, inner)

    assert batch_sizes == [16, 2]
    assert result.composite.components == tuple(
        _rf(((index + 1) * x - index) / (2 * x - 1), (x,))
        for index in range(9)
    )
    assert len(result.construction_locus_guard) == 2


def test_batch_fraction_normalization_matches_separate_exact_worker_results() -> None:
    from sympy import Poly

    x = symbols("x")
    pairs = (
        (Poly(x**2 - 1, x, domain="QQ"), Poly(x**2 - 2 * x + 1, x, domain="QQ")),
        (Poly(0, x, domain="QQ"), Poly(x + 1, x, domain="QQ")),
    )
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 60)
        batch = normalize_admitted_fractions(pairs, ("x",))
        separate = tuple(
            _gcd_process.normalize_admitted_fraction(numerator, denominator, ("x",))
            for numerator, denominator in pairs
        )
    assert batch == separate


def test_oversize_normalization_batch_falls_back_without_changing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = symbols("x")
    inner = _map(("x",), ("y",), (_rf(x / (x - 1), (x,)),))
    y = symbols("y")
    outer = _map(("y",), ("z",), (_rf(y + 1, (y,)),))
    fallback_calls = 0
    original = composition_ops._normalize_fraction

    def reject_batch(*_args: object, **_kwargs: object) -> object:
        raise KernelBatchInputLimitError("test aggregate payload threshold")

    def count_fallback(*args: object, **kwargs: object) -> object:
        nonlocal fallback_calls
        fallback_calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(composition_ops, "normalize_admitted_fractions", reject_batch)
    monkeypatch.setattr(composition_ops, "_normalize_fraction", count_fallback)

    result = compose_maps(outer, inner)

    assert fallback_calls == 2
    assert result.composite.components == (_rf((2 * x - 1) / (x - 1), (x,)),)


def test_same_shape_non_canceling_scale_composes() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf(x + 1, (x,)), _rf(2 * (x + 1), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z",),
        (_rf(1 / (y1**2 + y2**2), (y1, y2)),),
    )
    result = compose_maps(outer, inner)
    assert result.composite.components[0] == _rf(1 / (5 * (x + 1) ** 2), (x,))


def test_nonzero_scaled_near_miss_does_not_report_vanishing_denominator() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) ** 2, (x,)), _rf(-((x + 1) ** 2) + 1, (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z",),
        (_rf(1 / (y1**2 - y2**2), (y1, y2)),),
    )
    result = compose_maps(outer, inner)
    assert result.composite.components[0] == _rf(1 / (2 * (x + 1) ** 2 - 1), (x,))
