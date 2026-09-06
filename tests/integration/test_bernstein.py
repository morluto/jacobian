"""Independent coefficient reconstruction and public boundary evidence."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
from math import comb
from typing import Any

import pytest
import sympy as sp

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.analysis.intervals import RationalBox
from jacobian.math.polynomials.bernstein import (
    RationalBernsteinPolynomial,
    bernstein_coefficients,
    restrict_bernstein,
    verify_bernstein_coefficients,
    verify_bernstein_restriction,
)
from jacobian.math.polynomials.bernstein._tools import BernsteinRequest

ID = "polynomial.bernstein.coefficients.compute"


def _q(n: int, d: int = 1) -> dict[str, str]:
    value = Fraction(n, d)
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _fixture() -> dict[str, Any]:
    return {
        "polynomial": {
            "domain": "QQ",
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [
                    {"coefficient": _q(c), "exponents": list(e)}
                    for c, e in [(1, (2, 0)), (-1, (1, 0)), (1, (0, 1)), (1, (0, 0))]
                ]
            },
        },
        "box": {
            "domain": "QQ",
            "variables": ["x", "y"],
            "intervals": [{"lower": _q(0), "upper": _q(1)} for _ in range(2)],
        },
        "multidegree": [2, 1],
    }


def _run(payload: dict[str, Any]) -> RationalBernsteinPolynomial:
    output = invoke_operation(ID, payload, Catalog.open()).output
    result = RationalBernsteinPolynomial.model_validate(output)
    request = BernsteinRequest.model_validate(payload)
    assert result == bernstein_coefficients(
        request.polynomial, request.box, request.multidegree
    )
    return result


def _reconstruct(result: RationalBernsteinPolynomial) -> None:
    # Expand the Bernstein basis directly in the original x variables, not
    # through the forward power-to-Bernstein conversion used by the kernel.
    xs = tuple(sp.Symbol(name) for name in result.polynomial.variables)
    original = sum(
        (
            sp.Rational(*term.coefficient.as_integer_ratio())
            * sp.prod(x**e for x, e in zip(xs, term.exponents, strict=True))
            for term in result.polynomial.polynomial.terms
        ),
        sp.Integer(0),
    )
    reconstructed = sp.Integer(0)
    for index, coefficient in zip(
        product(*(range(m + 1) for m in result.multidegree)),
        result.coefficients,
        strict=True,
    ):
        basis = sp.Rational(*coefficient.as_integer_ratio())
        for x, interval, m, k in zip(
            xs, result.box.intervals, result.multidegree, index, strict=True
        ):
            a, b = (
                sp.Rational(*q.as_integer_ratio())
                for q in (interval.lower, interval.upper)
            )
            t = (x - a) / (b - a)
            basis *= comb(m, k) * t**k * (1 - t) ** (m - k)
        reconstructed += basis
    assert sp.Poly(reconstructed - original, *xs, domain=sp.QQ).is_zero


def test_reported_tensor_and_coefficientwise_identity() -> None:
    result = _run(_fixture())
    assert [c.as_fraction() for c in result.coefficients] == [
        1,
        2,
        Fraction(1, 2),
        Fraction(3, 2),
        1,
        2,
    ]
    _reconstruct(result)


@pytest.mark.parametrize("multidegree", [(2, 1), (4, 3)])
def test_rational_translated_box_and_degree_elevation(
    multidegree: tuple[int, int],
) -> None:
    payload = _fixture()
    payload["multidegree"] = list(multidegree)
    payload["box"]["intervals"] = [
        {"lower": _q(-2, 3), "upper": _q(5, 7)},
        {"lower": _q(1, 4), "upper": _q(7, 3)},
    ]
    _reconstruct(_run(payload))


@pytest.mark.parametrize("constant", [0, 3])
@pytest.mark.parametrize("multidegree", [[0, 0], [3, 2]])
def test_zero_and_constant_keep_all_axes(constant: int, multidegree: list[int]) -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = (
        [] if constant == 0 else [{"coefficient": _q(constant), "exponents": [0, 0]}]
    )
    payload["multidegree"] = multidegree
    result = _run(payload)
    assert all(c.as_fraction() == constant for c in result.coefficients)
    _reconstruct(result)


def test_transported_axis_permutation_transposes_the_tensor() -> None:
    payload = _fixture()
    result = _run(payload)
    payload["polynomial"]["variables"].reverse()
    payload["box"]["variables"].reverse()
    payload["box"]["intervals"].reverse()
    payload["multidegree"].reverse()
    terms = payload["polynomial"]["polynomial"]["terms"]
    for term in terms:
        term["exponents"].reverse()
    terms.sort(key=lambda term: term["exponents"], reverse=True)
    permuted = _run(payload)
    assert permuted.coefficients == tuple(
        result.coefficients[i * 2 + j] for j in range(2) for i in range(3)
    )
    _reconstruct(permuted)


def test_serialized_coefficients_are_claims_with_complete_source_context() -> None:
    result = _run(_fixture())
    assert verify_bernstein_coefficients(
        RationalBernsteinPolynomial.model_validate_json(result.model_dump_json())
    )
    wire = deepcopy(result.model_dump(mode="json"))
    wire["coefficients"][0] = _q(999)
    claim = RationalBernsteinPolynomial.model_validate(wire)
    assert claim.polynomial == result.polynomial
    assert claim.box == result.box
    assert not verify_bernstein_coefficients(claim)
    with pytest.raises(AssertionError):
        _reconstruct(claim)


@pytest.mark.parametrize("mutation", ["degree", "point", "axis", "tensor", "height"])
def test_invalid_or_unbounded_requests_reject_before_expansion(mutation: str) -> None:
    payload = _fixture()
    if mutation == "degree":
        payload["multidegree"] = [1, 1]
    elif mutation == "point":
        payload["box"]["intervals"][0]["upper"] = _q(0)
    elif mutation == "axis":
        payload["box"]["variables"].reverse()
    elif mutation == "tensor":
        payload["multidegree"] = [65535, 65535]
    else:
        payload["box"]["intervals"][0]["upper"] = {"num": "1" + "0" * 8192, "den": "1"}
    with pytest.raises(OperationDomainValidationError):
        invoke_operation(ID, payload, Catalog.open())


def test_large_degree_elevation_uses_axis_maps_not_a_dense_global_matrix() -> None:
    payload = _fixture()
    payload["multidegree"] = [4096, 0]
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q(1), "exponents": [1, 0]}
    ]
    result = _run(payload)
    assert tuple(c.as_fraction() for c in result.coefficients) == tuple(
        Fraction(k, 4096) for k in range(4097)
    )


def test_large_constant_tensor_is_charged_for_its_actual_source_degree() -> None:
    payload = _fixture()
    payload["multidegree"] = [4500, 0]
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q(3), "exponents": [0, 0]}
    ]
    result = _run(payload)
    assert len(result.coefficients) == 4501
    assert all(coefficient.as_fraction() == 3 for coefficient in result.coefficients)


def test_rejected_large_degree_does_not_expand_a_binomial_lcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.bernstein.operations as operations

    payload = _fixture()
    payload["multidegree"] = [16384, 0]
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q(1), "exponents": [8192, 0]}
    ]
    request = BernsteinRequest.model_validate(payload)
    monkeypatch.setattr(
        operations,
        "comb",
        lambda *_args: pytest.fail("admission expanded a binomial row"),
    )
    with pytest.raises(OperationDomainValidationError):
        operations._admit(request.polynomial, request.box, request.multidegree)


def test_carrier_rejects_a_point_box() -> None:
    result = _run(_fixture())
    wire = result.model_dump(mode="json")
    wire["box"]["intervals"][0]["upper"] = _q(0)
    claim = RationalBernsteinPolynomial.model_validate(wire)
    assert not verify_bernstein_coefficients(claim)


@pytest.mark.parametrize("constant", [0, 3])
def test_inactive_axes_do_not_expand_large_endpoint_rationals(
    constant: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jacobian.math.polynomials.bernstein.operations as operations

    payload = _fixture()
    payload["multidegree"] = [2, 1]
    payload["polynomial"]["polynomial"]["terms"] = (
        [] if constant == 0 else [{"coefficient": _q(constant), "exponents": [0, 0]}]
    )
    payload["box"]["intervals"][0] = {
        "lower": _q(1, 10**1000),
        "upper": _q(1, 10**1000 - 1),
    }
    request = BernsteinRequest.model_validate(payload)
    from flint import fmpq as original

    operands: list[Any] = []

    def counted(*args: Any) -> Any:
        operands.extend(args)
        return original(*args)

    monkeypatch.setattr(operations, "fmpq", counted)
    result = bernstein_coefficients(
        request.polynomial, request.box, request.multidegree
    )
    assert all(c.as_fraction() == constant for c in result.coefficients)
    assert all(abs(int(value)) <= 3 for value in operands)


def test_expired_request_deadline_is_not_restarted() -> None:
    from time import monotonic

    from jacobian._execution import OperationExecutionTimeoutError, request_execution

    request = BernsteinRequest.model_validate(_fixture())
    with (
        request_execution(monotonic() - 121),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        bernstein_coefficients(request.polynomial, request.box, request.multidegree)


def test_dense_cubic_on_a_rational_rectangle_admits_a_large_elevated_tensor() -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q((-1) ** (i + j) * (i + 1) * (j + 1), 7), "exponents": [i, j]}
        for i in range(3, -1, -1)
        for j in range(3, -1, -1)
    ]
    payload["box"]["intervals"] = [
        {"lower": _q(17, 32), "upper": _q(18, 32)},
        {"lower": _q(-1, 3), "upper": _q(2, 3)},
    ]
    payload["multidegree"] = [3, 3]
    base = _run(payload)
    _reconstruct(base)
    # Independent repeated degree elevation from the reconstructed small
    # tensor: beta'_k = k/(m+1)*beta_(k-1) + (1-k/(m+1))*beta_k.
    tensor = [
        [base.coefficients[i * 4 + j].as_fraction() for j in range(4)] for i in range(4)
    ]
    for m in range(3, 60):
        tensor = (
            [tensor[0]]
            + [
                [
                    Fraction(k, m + 1) * tensor[k - 1][j]
                    + Fraction(m + 1 - k, m + 1) * tensor[k][j]
                    for j in range(4)
                ]
                for k in range(1, m + 1)
            ]
            + [tensor[-1]]
        )
    for m in range(3, 60):
        tensor = [
            [row[0]]
            + [
                Fraction(k, m + 1) * row[k - 1] + Fraction(m + 1 - k, m + 1) * row[k]
                for k in range(1, m + 1)
            ]
            + [row[-1]]
            for row in tensor
        ]
    payload["multidegree"] = [60, 60]
    elevated = _run(payload)
    assert [c.as_fraction() for c in elevated.coefficients] == [
        value for row in tensor for value in row
    ]


def test_dense_bidegree_16_conversion_is_admitted_and_exact() -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = [
        {
            "coefficient": _q((-1) ** (i + j) * (i + 1) * (j + 1), 7),
            "exponents": [i, j],
        }
        for i in range(16, -1, -1)
        for j in range(16, -1, -1)
    ]
    payload["multidegree"] = [16, 16]
    result = _run(payload)
    assert len(result.coefficients) == 289
    _reconstruct(result)


def _box(
    intervals: list[tuple[tuple[int, int], tuple[int, int]]],
    variables: list[str] | None = None,
) -> Any:
    names = variables or ["x", "y"][: len(intervals)]
    return {
        "domain": "QQ",
        "variables": names,
        "intervals": [
            {"lower": _q(lower[0], lower[1]), "upper": _q(upper[0], upper[1])}
            for lower, upper in intervals
        ],
    }


def test_restriction_reuses_exact_tensor_and_matches_direct_conversion() -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q(4), "exponents": [2, 0]},
        {"coefficient": _q(-4), "exponents": [1, 0]},
        {"coefficient": _q(1), "exponents": [0, 0]},
    ]
    payload["multidegree"] = [2, 0]
    payload["box"] = _box([((0, 1), (1, 1)), ((0, 1), (1, 1))])
    parent = _run(payload)
    child = RationalBox.model_validate(_box([((1, 4), (3, 4)), ((0, 1), (1, 1))]))
    restricted = restrict_bernstein(parent, child)
    direct = bernstein_coefficients(parent.polynomial, child, parent.multidegree)
    assert [value.as_fraction() for value in restricted.coefficients] == [
        Fraction(1, 4),
        Fraction(-1, 4),
        Fraction(1, 4),
    ]
    assert restricted.coefficients == direct.coefficients
    assert verify_bernstein_restriction(parent, restricted)


def test_restriction_identity_nested_and_independent_axes_commute() -> None:
    parent = _run(_fixture())
    assert restrict_bernstein(parent, parent.box) == parent
    first_box = RationalBox.model_validate(_box([((1, 8), (7, 8)), ((1, 8), (7, 8))]))
    final_box = RationalBox.model_validate(_box([((1, 4), (3, 4)), ((1, 4), (3, 4))]))
    direct = restrict_bernstein(parent, final_box)
    nested = restrict_bernstein(restrict_bernstein(parent, first_box), final_box)
    assert nested.coefficients == direct.coefficients

    x_box = RationalBox.model_validate(_box([((1, 4), (3, 4)), ((0, 1), (1, 1))]))
    y_box = RationalBox.model_validate(_box([((0, 1), (1, 1)), ((1, 4), (3, 4))]))
    xy_box = RationalBox.model_validate(_box([((1, 4), (3, 4)), ((1, 4), (3, 4))]))
    x_then_y = restrict_bernstein(restrict_bernstein(parent, x_box), xy_box)
    y_then_x = restrict_bernstein(restrict_bernstein(parent, y_box), xy_box)
    assert x_then_y.coefficients == y_then_x.coefficients


def test_restriction_handles_a_realistic_dense_289_entry_tensor() -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = [
        {
            "coefficient": _q((-1) ** (i + j) * (i + 1) * (j + 1), 7),
            "exponents": [i, j],
        }
        for i in range(16, -1, -1)
        for j in range(16, -1, -1)
    ]
    payload["multidegree"] = [16, 16]
    parent = _run(payload)
    child = RationalBox.model_validate(_box([((1, 3), (2, 3)), ((1, 5), (4, 5))]))
    restricted = restrict_bernstein(parent, child)
    direct = bernstein_coefficients(parent.polynomial, child, parent.multidegree)
    assert restricted.coefficients == direct.coefficients


def test_restriction_preserves_zero_tensor_on_a_degenerate_degree_profile() -> None:
    payload = _fixture()
    payload["polynomial"]["polynomial"]["terms"] = []
    payload["multidegree"] = [3, 2]
    parent = _run(payload)
    child = RationalBox.model_validate(_box([((1, 3), (2, 3)), ((1, 5), (4, 5))]))
    restricted = restrict_bernstein(parent, child)
    assert len(restricted.coefficients) == 12
    assert all(value.as_fraction() == 0 for value in restricted.coefficients)


def test_inactive_restriction_axis_does_not_construct_large_split_ratios() -> None:
    payload = _fixture()
    payload["polynomial"]["variables"] = ["x"]
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": _q(1), "exponents": [0]}
    ]
    payload["multidegree"] = [0]
    scale = "1" + "0" * 9000
    payload["box"] = {
        "domain": "QQ",
        "variables": ["x"],
        "intervals": [{"lower": _q(0), "upper": {"num": "1", "den": scale}}],
    }
    request = BernsteinRequest.model_validate(payload)
    parent = bernstein_coefficients(
        request.polynomial, request.box, request.multidegree
    )
    child = RationalBox.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "intervals": [
                {
                    "lower": {"num": "1", "den": "4" + "0" * 9000},
                    "upper": {"num": "1", "den": "2" + "0" * 9000},
                }
            ],
        }
    )
    restricted = restrict_bernstein(parent, child)
    assert restricted.coefficients == parent.coefficients


def test_native_restriction_reuses_the_trusted_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.bernstein.operations as operations

    request = BernsteinRequest.model_validate(_fixture())
    parent = bernstein_coefficients(
        request.polynomial, request.box, request.multidegree
    )
    child = RationalBox.model_validate(_box([((1, 3), (2, 3)), ((1, 5), (4, 5))]))
    monkeypatch.setattr(
        operations,
        "verify_bernstein_coefficients",
        lambda _claim: pytest.fail("native tensor was recomputed"),
    )
    restricted = operations.restrict_bernstein(parent, child)
    assert restricted.polynomial == parent.polynomial


def test_copies_and_serialized_tensors_cross_the_recognition_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.bernstein.operations as operations

    request = BernsteinRequest.model_validate(_fixture())
    parent = bernstein_coefficients(
        request.polynomial, request.box, request.multidegree
    )
    child = RationalBox.model_validate(_box([((1, 3), (2, 3)), ((1, 5), (4, 5))]))
    calls = 0

    def recognize(_claim: RationalBernsteinPolynomial) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(operations, "verify_bernstein_coefficients", recognize)
    operations.restrict_bernstein(parent.model_copy(), child)
    operations.restrict_bernstein(
        RationalBernsteinPolynomial.model_validate_json(parent.model_dump_json()), child
    )
    assert calls == 2


@pytest.mark.parametrize("mutation", ["parent", "child", "outside", "axis"])
def test_restriction_rejects_forged_claims_and_invalid_children(mutation: str) -> None:
    parent = _run(_fixture())
    child_box = RationalBox.model_validate(_box([((1, 4), (3, 4)), ((1, 4), (3, 4))]))
    if mutation == "parent":
        wire = parent.model_dump(mode="json")
        wire["coefficients"][0] = _q(999)
        forged = RationalBernsteinPolynomial.model_validate(wire)
        with pytest.raises(OperationDomainValidationError):
            restrict_bernstein(forged, child_box)
    elif mutation == "child":
        claim = restrict_bernstein(parent, child_box)
        wire = claim.model_dump(mode="json")
        wire["coefficients"][0] = _q(999)
        forged = RationalBernsteinPolynomial.model_validate(wire)
        assert not verify_bernstein_restriction(parent, forged)
    elif mutation == "outside":
        outside = RationalBox.model_validate(_box([((0, 1), (2, 1)), ((1, 4), (3, 4))]))
        with pytest.raises(OperationDomainValidationError):
            restrict_bernstein(parent, outside)
    else:
        wrong_axes = RationalBox.model_validate(
            _box([((1, 4), (3, 4)), ((1, 4), (3, 4))], ["y", "x"])
        )
        with pytest.raises(OperationDomainValidationError):
            restrict_bernstein(parent, wrong_axes)


def test_restriction_rejects_degenerate_parent_and_child() -> None:
    parent = _run(_fixture())
    point = RationalBox.model_validate(_box([((1, 2), (1, 2)), ((0, 1), (1, 1))]))
    with pytest.raises(OperationDomainValidationError):
        restrict_bernstein(parent, point)
    parent_point = parent.model_copy(update={"box": point})
    assert not verify_bernstein_restriction(parent_point, parent)
