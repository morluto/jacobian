"""Tests for arithmetic counting operations."""

from itertools import product
from typing import TypedDict

from jacobian.math.number_theory.counting import congruence_box_count, floor_sum
from jacobian.math.number_theory.counting._models import (
    _MAX_BOX_LINEAR_COEFFICIENT,
    CongruenceBoxCountRequest,
    FloorSumRequest,
)
from jacobian.math.number_theory.counting._tools import (
    compute_congruence_box_count,
    compute_floor_sum,
)


class FloorSumPayload(TypedDict):
    """Raw wire payload used to exercise request validation."""

    n: int
    m: int
    a: int
    b: int


class CongruenceBoxCountPayload(TypedDict):
    """Raw wire payload used to exercise request validation."""

    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    u: int
    v: int
    c: int
    modulus: int


class TestFloorSum:
    def test_basic(self) -> None:
        # sum_{i=0}^{4} floor((2*i + 1) / 3)
        # i=0: floor(1/3)=0, i=1: floor(3/3)=1, i=2: floor(5/3)=1
        # i=3: floor(7/3)=2, i=4: floor(9/3)=3 => total=0+1+1+2+3=7
        req = FloorSumRequest(n=5, m=3, a=2, b=1)
        result = compute_floor_sum(req)
        assert result.value == "7"
        assert floor_sum(5, 3, 2, 1) == 7

    def test_zero_n(self) -> None:
        req = FloorSumRequest(n=0, m=3, a=2, b=1)
        result = compute_floor_sum(req)
        assert result.value == "0"

    def test_simple(self) -> None:
        # sum_{i=0}^{3} floor(i / 2) = 0+0+1+1 = 2
        req = FloorSumRequest(n=4, m=2, a=1, b=0)
        result = compute_floor_sum(req)
        assert result.value == "2"

    def test_rejects_negative_and_unbounded_inputs(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FloorSumRequest.model_validate(FloorSumPayload(n=-1, m=3, a=2, b=1))
        with pytest.raises(ValidationError):
            FloorSumRequest.model_validate(FloorSumPayload(n=5, m=0, a=2, b=1))
        with pytest.raises(ValidationError):
            FloorSumRequest.model_validate(FloorSumPayload(n=10**18 + 1, m=1, a=0, b=0))

    def test_matches_bruteforce_on_bounded_inputs(self) -> None:
        import random

        generator = random.Random(20260824)
        for _ in range(64):
            n = generator.randint(0, 400)
            m = generator.randint(1, 40)
            a = generator.randint(0, 120)
            b = generator.randint(0, 90)
            expected = sum((a * index + b) // m for index in range(n))
            result = compute_floor_sum(FloorSumRequest(n=n, m=m, a=a, b=b))
            assert int(result.value) == expected

    def test_number_theory_scale_is_admitted(self) -> None:
        # n = 10^12 with the Euclidean kernel completes in microseconds; the
        # brute-force loop this replaced would need hours.  Each term is
        # (a*i+b)/m within 1, so the exact sum is pinned between two
        # closed-form rational bounds of width n.
        from fractions import Fraction

        n, m, a, b = 10**12, 999_983, 123_456, 789
        result = compute_floor_sum(FloorSumRequest(n=n, m=m, a=a, b=b))
        value = int(result.value)
        linear = Fraction(a * n * (n - 1) // 2 + b * n, m)
        assert Fraction(value) <= linear
        assert Fraction(value) >= linear - n


class TestCongruenceBoxCount:
    def test_simple(self) -> None:
        req = CongruenceBoxCountRequest(
            x_lo=0,
            x_hi=5,
            y_lo=0,
            y_hi=5,
            u=1,
            v=1,
            c=0,
            modulus=3,
        )
        result = compute_congruence_box_count(req)
        # Count (x+y) % 3 == 0 for x,y in [0,5]
        # For each x, y must satisfy y = (-x) mod 3
        # x=0: y=0,3; x=1: y=2,5; x=2: y=1,4; x=3: y=0,3; x=4: y=2,5; x=5: y=1,4
        # => 2 per x => 12 total
        assert result.count == 12
        assert (
            congruence_box_count(
                x_lo=0,
                x_hi=5,
                y_lo=0,
                y_hi=5,
                u=1,
                v=1,
                c=0,
                modulus=3,
            )
            == 12
        )

    def test_admits_full_coordinate_box(self) -> None:
        request = CongruenceBoxCountRequest.model_validate(
            CongruenceBoxCountPayload(
                x_lo=-10_000,
                x_hi=10_000,
                y_lo=-10_000,
                y_hi=10_000,
                u=7331,
                v=4096,
                c=17,
                modulus=9991,
            )
        )
        assert compute_congruence_box_count(request).count == 40_040

    def test_matches_exhaustive_small_boxes(self) -> None:
        intervals = ((-2, 2), (-1, -1), (0, 2))
        for (x_lo, x_hi), (y_lo, y_hi), u, v, c, modulus in product(
            intervals,
            intervals,
            range(-2, 3),
            range(-2, 3),
            range(-2, 3),
            range(1, 6),
        ):
            expected = sum(
                (u * x + v * y - c) % modulus == 0
                for x in range(x_lo, x_hi + 1)
                for y in range(y_lo, y_hi + 1)
            )
            assert (
                congruence_box_count(
                    x_lo=x_lo,
                    x_hi=x_hi,
                    y_lo=y_lo,
                    y_hi=y_hi,
                    u=u,
                    v=v,
                    c=c,
                    modulus=modulus,
                )
                == expected
            )

    def test_axis_swap_preserves_count(self) -> None:
        request = CongruenceBoxCountPayload(
            x_lo=-20,
            x_hi=19,
            y_lo=-30,
            y_hi=31,
            u=6,
            v=10,
            c=4,
            modulus=14,
        )
        expected = congruence_box_count(**request)
        assert expected == 354
        assert (
            congruence_box_count(
                x_lo=request["y_lo"],
                x_hi=request["y_hi"],
                y_lo=request["x_lo"],
                y_hi=request["x_hi"],
                u=request["v"],
                v=request["u"],
                c=request["c"],
                modulus=request["modulus"],
            )
            == expected
        )

    def test_degenerate_boxes(self) -> None:
        assert (
            congruence_box_count(
                x_lo=0,
                x_hi=0,
                y_lo=-3,
                y_hi=3,
                u=2,
                v=4,
                c=1,
                modulus=6,
            )
            == 0
        )
        assert (
            congruence_box_count(
                x_lo=-3,
                x_hi=3,
                y_lo=0,
                y_hi=0,
                u=2,
                v=4,
                c=2,
                modulus=6,
            )
            == 2
        )
        assert (
            congruence_box_count(
                x_lo=1,
                x_hi=1,
                y_lo=1,
                y_hi=1,
                u=2,
                v=4,
                c=0,
                modulus=6,
            )
            == 1
        )

    def test_rejects_linear_coefficient_above_digit_budget(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CongruenceBoxCountRequest.model_validate(
                CongruenceBoxCountPayload(
                    x_lo=0,
                    x_hi=0,
                    y_lo=0,
                    y_hi=0,
                    u=_MAX_BOX_LINEAR_COEFFICIENT + 1,
                    v=0,
                    c=0,
                    modulus=1,
                )
            )
