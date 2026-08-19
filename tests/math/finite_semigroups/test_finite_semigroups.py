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

# Monogenic semigroup <a> with a^1=x, a^2=y, a^3=z, a^4=y (index 2, period 2).
CYCLIC_TAIL = {
    "elements": ["x", "y", "z"],
    "multiplication": [
        ["y", "z", "y"],
        ["z", "y", "z"],
        ["y", "z", "y"],
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

    def test_overlong_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="length budget"):
            FiniteSemigroup(
                elements=["a", "x" * 65],
                multiplication=[
                    ["a", "a"],
                    ["a", "a"],
                ],
            )


class TestPowerProfile:
    def test_z3_element_1(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        assert result.element == "1"
        assert result.powers == ("1", "2", "0")
        # a^1 = 1, a^2 = 2, a^3 = 0, a^4 = 1: index 1, period 3.
        assert result.index == 1
        assert result.period == 3
        assert result.idempotent == "0"

    def test_z3_element_0_is_idempotent(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="0"))
        assert result.powers == ("0",)
        assert result.index == 1
        assert result.period == 1
        assert result.idempotent == "0"

    def test_band_element_a(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=BAND, element="a"))
        assert result.powers == ("a",)
        assert result.index == 1
        assert result.period == 1
        assert result.idempotent == "a"

    def test_null_element_x(self) -> None:
        result = compute_power_profile(
            PowerProfileRequest(semigroup=NULL_SG, element="x")
        )
        # a^1 = x, a^2 = 0, a^3 = 0: index 2, period 1.
        assert result.powers == ("x", "0")
        assert result.index == 2
        assert result.period == 1
        assert result.idempotent == "0"

    def test_cyclic_subsemigroup(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        assert result.cyclic_subsemigroup == ("1", "2", "0")

    def test_nontrivial_tail_index_and_period(self) -> None:
        result = compute_power_profile(
            PowerProfileRequest(semigroup=CYCLIC_TAIL, element="x")
        )
        assert result.powers == ("x", "y", "z")
        assert result.index == 2
        assert result.period == 2
        assert result.idempotent == "y"

    def test_powers_replay_from_table(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        mult = Z3["multiplication"]
        elements = Z3["elements"]
        idx = {label: i for i, label in enumerate(elements)}
        a = result.element
        running = a
        for power in result.powers:
            assert power == running
            running = mult[idx[running]][idx[a]]

    def test_idempotent_is_idempotent(self) -> None:
        for sg, element in [(Z3, "1"), (Z3, "0"), (BAND, "a"), (NULL_SG, "x")]:
            result = compute_power_profile(
                PowerProfileRequest(semigroup=sg, element=element)
            )
            mult = sg["multiplication"]
            elements = sg["elements"]
            idx = {label: i for i, label in enumerate(elements)}
            e = result.idempotent
            assert mult[idx[e]][idx[e]] == e

    def test_cyclic_subsemigroup_is_exact_closure(self) -> None:
        result = compute_power_profile(PowerProfileRequest(semigroup=Z3, element="1"))
        assert set(result.cyclic_subsemigroup) == set(result.powers)


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
