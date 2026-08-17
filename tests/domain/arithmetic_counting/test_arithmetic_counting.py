"""Tests for arithmetic counting operations."""

from jacobian.contracts.arithmetic_counting import (
    CongruenceBoxCountRequest,
    FloorSumRequest,
)
from jacobian.domains.arithmetic_counting.operations import (
    compute_congruence_box_count,
    compute_floor_sum,
)


class TestFloorSum:
    def test_basic(self):
        # sum_{i=0}^{4} floor((2*i + 1) / 3)
        # i=0: floor(1/3)=0, i=1: floor(3/3)=1, i=2: floor(5/3)=1
        # i=3: floor(7/3)=2, i=4: floor(9/3)=3 => total=0+1+1+2+3=7
        req = FloorSumRequest(n=5, m=3, a=2, b=1)
        result = compute_floor_sum(req)
        assert result.value == "7"

    def test_zero_n(self):
        req = FloorSumRequest(n=0, m=3, a=2, b=1)
        result = compute_floor_sum(req)
        assert result.value == "0"

    def test_simple(self):
        # sum_{i=0}^{3} floor(i / 2) = 0+0+1+1 = 2
        req = FloorSumRequest(n=4, m=2, a=1, b=0)
        result = compute_floor_sum(req)
        assert result.value == "2"

    def test_rejects_negative_and_unbounded_inputs(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FloorSumRequest(n=-1, m=3, a=2, b=1)
        with pytest.raises(ValidationError):
            FloorSumRequest(n=5, m=0, a=2, b=1)
        with pytest.raises(ValidationError):
            FloorSumRequest(n=1_000_001, m=1, a=0, b=0)


class TestCongruenceBoxCount:
    def test_simple(self):
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

    def test_rejects_oversized_box(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="box area"):
            CongruenceBoxCountRequest(
                x_lo=-10_000,
                x_hi=10_000,
                y_lo=-10_000,
                y_hi=10_000,
                u=1,
                v=1,
                c=0,
                modulus=3,
            )
