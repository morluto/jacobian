"""Known-answer and adversarial tests for finite semigroup operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_semigroups._models import (
    ElementPowerRequest,
    FiniteSemigroup,
    GeneratedSubsemigroupRequest,
    IdempotentsRequest,
    PowerProfileRequest,
    PrincipalIdealsRequest,
)
from jacobian.math.finite_semigroups._operations import (
    compute_element_power,
    compute_generated_subsemigroup,
    compute_idempotents,
    compute_power_profile,
    compute_principal_ideals,
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

# The five matrix units of the 2x2 Brandt semigroup, with zero.
MATRIX_UNITS = {
    "elements": ["0", "e11", "e12", "e21", "e22"],
    "multiplication": [
        ["0", "0", "0", "0", "0"],
        ["0", "e11", "e12", "0", "0"],
        ["0", "0", "0", "e11", "e12"],
        ["0", "e21", "e22", "0", "0"],
        ["0", "0", "0", "e21", "e22"],
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
        with pytest.raises(ValidationError, match="at most 64 characters"):
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


class TestElementPower:
    def test_z3_power_2(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=Z3, element="1", exponent=2)
        )
        assert result.power == "2"

    def test_z3_power_identity(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=Z3, element="1", exponent=1)
        )
        assert result.power == "1"

    def test_z3_power_4_cycles(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=Z3, element="2", exponent=4)
        )
        assert result.power == "2"

    def test_null_semigroup_power(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=NULL_SG, element="x", exponent=2)
        )
        assert result.power == "0"

    def test_exponent_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exponent"):
            ElementPowerRequest(semigroup=Z3, element="1", exponent=0)

    def test_missing_element_rejected(self) -> None:
        with pytest.raises(ValidationError, match="element must be in the semigroup"):
            ElementPowerRequest(semigroup=Z3, element="9", exponent=2)

    def test_power_replays_from_table(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=Z3, element="1", exponent=5)
        )
        mult = Z3["multiplication"]
        elements = Z3["elements"]
        idx = {label: i for i, label in enumerate(elements)}
        running = "1"
        for _ in range(4):
            running = mult[idx[running]][idx["1"]]
        assert result.power == running

    def test_huge_exponent_uses_the_finite_power_profile(self) -> None:
        result = compute_element_power(
            ElementPowerRequest(semigroup=Z3, element="1", exponent=10**100)
        )
        assert result.power == "1"


class TestIdempotents:
    def test_z3_only_zero(self) -> None:
        result = compute_idempotents(IdempotentsRequest(semigroup=Z3))
        assert result.idempotents == ("0",)

    def test_band_both_idempotent(self) -> None:
        result = compute_idempotents(IdempotentsRequest(semigroup=BAND))
        assert result.idempotents == ("a", "b")

    def test_null_semigroup_only_zero(self) -> None:
        result = compute_idempotents(IdempotentsRequest(semigroup=NULL_SG))
        assert result.idempotents == ("0",)

    def test_every_reported_element_is_idempotent(self) -> None:
        for sg in (Z3, BAND, NULL_SG):
            result = compute_idempotents(IdempotentsRequest(semigroup=sg))
            mult = sg["multiplication"]
            elements = sg["elements"]
            idx = {label: i for i, label in enumerate(elements)}
            for e in result.idempotents:
                assert mult[idx[e]][idx[e]] == e


class TestPrincipalIdeals:
    def test_z3_ideal_of_1_is_whole_semigroup(self) -> None:
        result = compute_principal_ideals(
            PrincipalIdealsRequest(semigroup=Z3, elements=["1"])
        )
        assert set(result.ideals[0]) == {"0", "1", "2"}

    def test_band_ideals(self) -> None:
        result = compute_principal_ideals(
            PrincipalIdealsRequest(semigroup=BAND, elements=["a", "b"])
        )
        assert result.ideals == (("a", "b"), ("a", "b"))

    def test_null_ideal_of_x(self) -> None:
        result = compute_principal_ideals(
            PrincipalIdealsRequest(semigroup=NULL_SG, elements=["x"])
        )
        assert set(result.ideals[0]) == {"0", "x"}

    def test_ideal_contains_the_element(self) -> None:
        for sg, element in [(Z3, "1"), (BAND, "a"), (NULL_SG, "x")]:
            result = compute_principal_ideals(
                PrincipalIdealsRequest(semigroup=sg, elements=[element])
            )
            assert element in result.ideals[0]

    def test_principal_two_sided_ideal_contains_triple_products(self) -> None:
        result = compute_principal_ideals(
            PrincipalIdealsRequest(semigroup=MATRIX_UNITS, elements=["e11"])
        )
        assert result.ideals == (("0", "e11", "e12", "e21", "e22"),)

    def test_principal_two_sided_ideal_is_closed_on_both_sides(self) -> None:
        for element in MATRIX_UNITS["elements"]:
            ideal = compute_principal_ideals(
                PrincipalIdealsRequest(semigroup=MATRIX_UNITS, elements=[element])
            ).ideals[0]
            labels = MATRIX_UNITS["elements"]
            table = MATRIX_UNITS["multiplication"]
            index = {label: i for i, label in enumerate(labels)}
            for member in ideal:
                for multiplier in labels:
                    assert table[index[multiplier]][index[member]] in ideal
                    assert table[index[member]][index[multiplier]] in ideal

    def test_duplicate_or_out_of_order_elements_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            PrincipalIdealsRequest(semigroup=Z3, elements=["1", "1"])
        with pytest.raises(ValidationError, match="declared semigroup order"):
            PrincipalIdealsRequest(semigroup=Z3, elements=["2", "1"])

    def test_missing_element_rejected(self) -> None:
        with pytest.raises(
            ValidationError, match="every element must be in the semigroup"
        ):
            PrincipalIdealsRequest(semigroup=Z3, elements=["nope"])


class TestGreenRelations:
    """Tests for Green relations computation."""

    def test_group_has_universal_green_relations(self) -> None:
        """In a group, all Green relations are the universal relation."""
        from jacobian.math.finite_semigroups._models import (
            GreenRelationsRequest,
        )
        from jacobian.math.finite_semigroups._operations import compute_green_relations

        sg = FiniteSemigroup(**Z3)
        result = compute_green_relations(GreenRelationsRequest(semigroup=sg))
        # In a group, all elements are J-equivalent (and hence L, R, H, D)
        assert len(result.L) == 1
        assert len(result.R) == 1
        assert len(result.H) == 1
        assert len(result.D) == 1
        assert len(result.J) == 1

    def test_null_semigroup_relations(self) -> None:
        """In a null semigroup, each non-zero element is alone in its Green classes."""
        from jacobian.math.finite_semigroups._models import (
            GreenRelationsRequest,
        )
        from jacobian.math.finite_semigroups._operations import compute_green_relations

        sg = FiniteSemigroup(**NULL_SG)
        result = compute_green_relations(GreenRelationsRequest(semigroup=sg))
        # Each element has distinct left/right ideals, so all singletons
        assert result.L == (("0",), ("x",), ("y",))
        assert result.R == (("0",), ("x",), ("y",))

    def test_band_semigroup_relations(self) -> None:
        """In a left-zero band (a*b=a), L is universal, R is discrete."""
        from jacobian.math.finite_semigroups._models import (
            GreenRelationsRequest,
        )
        from jacobian.math.finite_semigroups._operations import compute_green_relations

        # left-zero band: a*b=a for all a,b
        # left ideal of a = {a, ...} but since a*b=a, left ideal of a = {a}
        # Actually for left-zero band: a*b=a, so Sa = {a} (left mult gives a),
        # aS = {a, b} (right mult of a by everything gives a). So left ideals
        # are all singletons, right ideals are universal.
        band = FiniteSemigroup(
            elements=("a", "b"),
            multiplication=(("a", "a"), ("b", "b")),
        )
        result = compute_green_relations(GreenRelationsRequest(semigroup=band))
        # left ideal of a = {a} (elements that left-multiply a), left ideal of b = {b}
        # right ideal of a = {a, b} (a*a=a, a*b=a; but a is right-multiplied by all -> a)
        # Actually: left ideal S^1 a = {s*a : s in S} union {a} = {a}
        # right ideal a S^1 = {a*s : s in S} union {a} = {a, b}... wait a*a=a, a*b=a so right ideal of a = {a}
        # Wait, the table is a*b=a (row a, col b), so a multiplied on right by a gives a, by b gives a. So right ideal of a = {a}.
        # Similarly right ideal of b = {b}. So R is also discrete.
        # Actually let me recheck: band table is [[a,a],[b,b]] meaning a*a=a, a*b=a, b*a=b, b*b=b
        # This is a left-zero band where the LEFT argument wins.
        # Left ideal of a: {s*a : s} = {a*a, b*a} = {a, b} = universal
        # Right ideal of a: {a*s : s} = {a*a, a*b} = {a}
        assert len(result.L) == 1  # universal
        assert result.R == (("a",), ("b",))  # discrete

    def test_green_relations_result_binds(self) -> None:
        """GreenRelationsResult validates correctly."""
        from jacobian.math.finite_semigroups._models import (
            GreenRelationsRequest,
            GreenRelationsResult,
        )
        from jacobian.math.finite_semigroups._operations import compute_green_relations

        sg = FiniteSemigroup(**Z3)
        req = GreenRelationsRequest(semigroup=sg)
        result = compute_green_relations(req)
        # Reconstruct with same values should work
        reconstructed = GreenRelationsResult(
            semigroup=result.semigroup,
            L=result.L,
            R=result.R,
            H=result.H,
            D=result.D,
            J=result.J,
        )
        assert reconstructed.L == result.L

    def test_green_relations_wrong_value_rejected(self) -> None:
        """GreenRelationsResult rejects incorrect Green relations."""
        from jacobian.math.finite_semigroups._models import (
            GreenRelationsRequest,
            GreenRelationsResult,
        )
        from jacobian.math.finite_semigroups._operations import compute_green_relations

        sg = FiniteSemigroup(**Z3)
        result = compute_green_relations(GreenRelationsRequest(semigroup=sg))
        with pytest.raises(ValueError, match="L must be"):
            GreenRelationsResult(
                semigroup=sg,
                L=(("0",), ("1",), ("2",)),  # wrong
                R=result.R,
                H=result.H,
                D=result.D,
                J=result.J,
            )
