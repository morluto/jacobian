"""Known-answer and adversarial tests for finite semigroup operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_semigroups._models import (
    FiniteSemigroup,
    GeneratedSubsemigroupRequest,
    PowerProfileRequest,
)
from jacobian.math.finite_semigroups._operations import (
    compute_generated_subsemigroup,
    compute_power_profile,
)

# Z/3Z as a semigroup under addition mod 3
Z3 = {
    "elements": ["0", "1", "2"],
    "multiplication": [
        ["0", "1", "2"],
        ["1", "2", "0"],
        ["2", "0", "1"],
    ],
}

# A band semigroup (idempotent): {a, b} with a*a=a, b*b=b, a*b=a, b*a=b
BAND = {
    "elements": ["a", "b"],
    "multiplication": [
        ["a", "a"],
        ["b", "b"],
    ],
}

# Null semigroup: x*y = 0 for all x, y
NULL_SG = {
    "elements": ["0", "x", "y"],
    "multiplication": [
        ["0", "0", "0"],
        ["0", "0", "0"],
        ["0", "0", "0"],
    ],
}


class TestFiniteSemigroup:
    def test_z3_is_valid(self) -> None:
        sg = FiniteSemigroup(**Z3)
        assert sg.elements == ("0", "1", "2")

    def test_band_is_valid(self) -> None:
        sg = FiniteSemigroup(**BAND)
        assert sg.elements == ("a", "b")

    def test_null_is_valid(self) -> None:
        sg = FiniteSemigroup(**NULL_SG)
        assert sg.elements == ("0", "x", "y")

    def test_non_associative_rejected(self) -> None:
        # (a*b)*a = b*a = c, but a*(b*a) = a*c = a, so non-associative
        with pytest.raises(ValidationError, match="associative"):
            FiniteSemigroup(
                elements=["a", "b", "c"],
                multiplication=[
                    ["a", "b", "a"],
                    ["c", "a", "b"],
                    ["c", "b", "c"],
                ],
            )

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError, match="declared element"):
            FiniteSemigroup(
                elements=["a", "b"],
                multiplication=[
                    ["a", "z"],
                    ["a", "b"],
                ],
            )


class TestPowerProfile:
    def test_z3_element_1(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        assert result.element == "1"
        assert result.powers == ("1", "2", "0")
        assert result.index == 0
        assert result.period == 3
        assert result.idempotent == "1"

    def test_z3_element_0_is_identity(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="0"))
        assert result.powers == ("0",)
        assert result.index == 0
        assert result.period == 1
        assert result.idempotent == "0"

    def test_band_element_a(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=BAND, element="a"))
        assert result.powers == ("a",)
        assert result.index == 0
        assert result.period == 1
        assert result.idempotent == "a"

    def test_null_element_x(self) -> None:
        result = compute_power_profile(
            PowerProfileRequest(semigroup=NULL_SG, element="x")
        )
        assert result.powers == ("x", "0")
        assert result.idempotent == "0"

    def test_cyclic_subsemigroup(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        assert result.cyclic_subsemigroup == ("1", "2", "0")


class TestGeneratedSubsemigroup:
    def test_z3_generated_by_1(self) -> None:
        result = compute_generated_subsemigroup(
            GeneratedSubsemigroupRequest(semigroup=Z3, generators=["1"])
        )
        assert set(result.elements) == {"0", "1", "2"}

    def test_band_generated_by_a(self) -> None:
        result = compute_generated_subsemigroup(
            GeneratedSubsemigroupRequest(semigroup=BAND, generators=["a"])
        )
        assert result.elements == ("a",)

    def test_band_generated_by_both(self) -> None:
        result = compute_generated_subsemigroup(
            GeneratedSubsemigroupRequest(semigroup=BAND, generators=["a", "b"])
        )
        assert set(result.elements) == {"a", "b"}

    def test_null_generated_by_x(self) -> None:
        result = compute_generated_subsemigroup(
            GeneratedSubsemigroupRequest(semigroup=NULL_SG, generators=["x"])
        )
        assert set(result.elements) == {"x", "0"}

    def test_generators_preserved(self) -> None:
        result = compute_generated_subsemigroup(
            GeneratedSubsemigroupRequest(semigroup=Z3, generators=["1", "2"])
        )
        assert set(result.generators) == {"1", "2"}
