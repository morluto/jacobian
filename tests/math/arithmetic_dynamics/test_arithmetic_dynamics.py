"""Known-answer and adversarial tests for arithmetic dynamics."""

from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.arithmetic_dynamics import (
    finite_field_functional_graph,
    fixed_point_equation,
    iterate_polynomial,
    polynomial_coefficients,
    polynomial_from_coefficients,
)
from jacobian.math.arithmetic_dynamics._models import (
    MAX_FIELD_PRIME,
    CycleMultiplierRequest,
    DynatomicPolynomialRequest,
    FiniteFieldMapRequest,
    FiniteFieldMapResult,
    MapIterateRequest,
    OrbitPrefixRequest,
    OrbitPrefixResult,
)
from jacobian.math.arithmetic_dynamics._operations import (
    compute_cycle_multiplier,
    compute_dynatomic_polynomial,
    compute_finite_field_map,
    compute_map_iterate,
    compute_orbit_prefix,
)
from jacobian.math.arithmetic_dynamics._tools import TOOLS


def _r(value: int | str) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


class TestMapIterate:
    def test_zero_iterate_is_identity(self) -> None:
        result = compute_map_iterate(
            MapIterateRequest(coefficients=(_r(1), _r(0), _r(1)), n=0)
        )

        assert result.coefficients == (_r(0), _r(1))
        assert result.degree == 1
        assert result.complete is True

    def test_second_iterate_is_exact(self) -> None:
        result = compute_map_iterate(
            MapIterateRequest(coefficients=(_r(1), _r(0), _r(1)), n=2)
        )

        assert result.coefficients == (_r(2), _r(0), _r(2), _r(0), _r(1))
        assert result.degree == 4

    def test_zero_polynomial_iterates_without_backend_degree_coercion(self) -> None:
        result = compute_map_iterate(MapIterateRequest(coefficients=(_r(0),), n=3))

        assert result.coefficients == (_r(0),)
        assert result.degree == 0

    def test_degree_growth_beyond_output_bound_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MapIterateRequest(
                coefficients=(_r(1), _r(0), _r(0), _r(0), _r(0), _r(1)), n=5
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.iterate_degree_exceeds_bound"
        )


class TestOrbitPrefix:
    def test_repeat_proves_preperiod_and_period(self) -> None:
        result = compute_orbit_prefix(
            OrbitPrefixRequest(
                coefficients=(_r(0), _r(0), _r(1)), start=_r(0), max_steps=5
            )
        )

        assert result.orbit == (_r(0), _r(0))
        assert result.termination == "REPEAT_FOUND"
        assert result.repeat is not None
        assert result.repeat.preperiod == 0
        assert result.repeat.period == 1
        assert result.eventual_behavior_complete is True
        assert result.truncated is False

    def test_finite_nonrepeating_prefix_does_not_imply_eventual_behavior(self) -> None:
        result = compute_orbit_prefix(
            OrbitPrefixRequest(coefficients=(_r(1), _r(1)), start=_r(0), max_steps=3)
        )

        assert result.orbit == (_r(0), _r(1), _r(2), _r(3))
        assert result.termination == "STEP_BOUND_REACHED"
        assert result.repeat is None
        assert result.eventual_behavior_complete is False
        assert result.truncated is True

    def test_zero_step_request_is_an_explicit_truncated_prefix(self) -> None:
        result = compute_orbit_prefix(
            OrbitPrefixRequest(coefficients=(_r(1), _r(1)), start=_r(0), max_steps=0)
        )

        assert result.orbit == (_r(0),)
        assert result.computed_steps == 0
        assert result.termination == "STEP_BOUND_REACHED"
        assert result.truncated is True

    def test_output_growth_stops_with_nonconcluding_typed_boundary(self) -> None:
        degree_thirty = (_r(0),) * 30 + (_r(1),)
        result = compute_orbit_prefix(
            OrbitPrefixRequest(
                coefficients=degree_thirty,
                start=_r("1" + "0" * 127),
                max_steps=2,
            )
        )

        assert result.termination == "OUTPUT_BOUND_REACHED"
        assert result.computed_steps < result.requested_steps
        assert result.repeat is None
        assert result.eventual_behavior_complete is False
        assert result.truncated is True

    def test_result_model_rejects_completion_without_repeat_evidence(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrbitPrefixResult(
                source_coefficients=(_r(1), _r(1)),
                start=_r(0),
                orbit=(_r(0), _r(1)),
                requested_steps=1,
                computed_steps=1,
                termination="STEP_BOUND_REACHED",
                repeat=None,
                eventual_behavior_complete=True,
                truncated=False,
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.bounded_termination_claims_eventual_behavior"
        )

    def test_result_model_rejects_orbit_not_bound_to_source_map(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrbitPrefixResult(
                source_coefficients=(_r(1), _r(1)),
                start=_r(0),
                orbit=(_r(0), _r(2)),
                requested_steps=1,
                computed_steps=1,
                termination="STEP_BOUND_REACHED",
                repeat=None,
                eventual_behavior_complete=False,
                truncated=True,
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.orbit_map_mismatch"
        )


class TestDynatomicPolynomial:
    def test_first_dynatomic_polynomial(self) -> None:
        result = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(coefficients=(_r(0), _r(0), _r(1)), n=1)
        )

        assert result.coefficients == (_r(0), _r(-1), _r(1))

    def test_square_factor_mobius_case_and_divisor_product_identity(self) -> None:
        source = polynomial_from_coefficients((0, 0, 1))
        phi_1 = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(coefficients=(_r(0), _r(0), _r(1)), n=1)
        )
        phi_2 = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(coefficients=(_r(0), _r(0), _r(1)), n=2)
        )
        phi_4 = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(coefficients=(_r(0), _r(0), _r(1)), n=4)
        )
        product = polynomial_from_coefficients(
            tuple(value.as_fraction() for value in phi_1.coefficients)
        )
        product *= polynomial_from_coefficients(
            tuple(value.as_fraction() for value in phi_2.coefficients)
        )
        product *= polynomial_from_coefficients(
            tuple(value.as_fraction() for value in phi_4.coefficients)
        )

        assert phi_4.coefficients == tuple(
            _r(value) for value in (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1)
        )
        assert product == fixed_point_equation(source, 4)

    def test_linear_map_is_outside_dynatomic_contract(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DynatomicPolynomialRequest(coefficients=(_r(1), _r(1)), n=2)
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.dynatomic_degree_too_small"
        )


class TestCycleMultiplier:
    def test_validated_two_cycle_multiplier(self) -> None:
        result = compute_cycle_multiplier(
            CycleMultiplierRequest(coefficients=(_r(1), _r(-1)), cycle=(_r(0), _r(1)))
        )

        assert result.multiplier == _r(1)
        assert result.period == 2
        assert result.validated_cycle is True

    def test_arbitrary_points_cannot_be_labeled_a_cycle(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CycleMultiplierRequest(
                coefficients=(_r(0), _r(0), _r(1)), cycle=(_r(0), _r(1))
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.cycle_map_mismatch"
        )

    def test_repeated_cycle_points_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CycleMultiplierRequest(
                coefficients=(_r(0), _r(0), _r(1)), cycle=(_r(0), _r(0))
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.cycle_points_not_distinct"
        )


class TestFiniteFieldFunctionalGraph:
    def test_x_squared_mod_five_has_complete_canonical_graph(self) -> None:
        result = compute_finite_field_map(
            FiniteFieldMapRequest(prime=5, coefficients=("0", "0", "1"))
        )

        assert result.edges == ((0, 0), (1, 1), (2, 4), (3, 4), (4, 1))
        assert result.cycles == ((0,), (1,))
        assert result.tail_lengths == (0, 0, 2, 2, 1)
        assert result.complete is True

    @pytest.mark.parametrize(
        ("prime", "coefficients"),
        [(2, (1, 1, 1)), (3, (2, 0, 1)), (5, (1, 1)), (7, (3, 2, 1))],
    )
    def test_edges_cycles_and_tail_lengths_replay(
        self, prime: int, coefficients: tuple[int, ...]
    ) -> None:
        result = compute_finite_field_map(
            FiniteFieldMapRequest(
                prime=prime, coefficients=tuple(str(value) for value in coefficients)
            )
        )
        targets = dict(result.edges)
        cycle_nodes = {node for cycle in result.cycles for node in cycle}

        assert targets == {
            point: sum(
                coefficient * pow(point, exponent, prime)
                for exponent, coefficient in enumerate(coefficients)
            )
            % prime
            for point in range(prime)
        }
        for cycle in result.cycles:
            assert cycle[0] == min(cycle)
            assert all(
                targets[node] == cycle[(index + 1) % len(cycle)]
                for index, node in enumerate(cycle)
            )
        assert cycle_nodes == {
            node for node, length in enumerate(result.tail_lengths) if length == 0
        }
        assert all(
            result.tail_lengths[source] == result.tail_lengths[target] + 1
            for source, target in result.edges
            if source not in cycle_nodes
        )

    def test_result_contract_rejects_false_tail_evidence(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapResult(
                prime=2,
                coefficients=("0",),
                edges=((0, 0), (1, 0)),
                cycles=((0,),),
                tail_lengths=(0, 0),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.cycle_tail_mismatch"
        )

    def test_result_contract_rejects_edges_not_bound_to_polynomial(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapResult(
                prime=3,
                coefficients=("0", "0", "1"),
                edges=((0, 0), (1, 2), (2, 1)),
                cycles=((0,), (1, 2)),
                tail_lengths=(0, 0, 0),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.functional_graph_edge_mismatch"
        )

    def test_nonprime_modulus_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapRequest(prime=4, coefficients=("1",))
        assert (
            exc_info.value.errors()[0]["type"] == "arithmetic_dynamics.prime_not_prime"
        )

    def test_noncanonical_integer_coefficients_have_an_owner_code(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapRequest(prime=5, coefficients=("01",))
        assert exc_info.value.errors()[0]["type"] == (
            "arithmetic_dynamics.coefficient_not_canonical"
        )

    def test_wire_and_native_paths_share_the_field_prime_bound(self) -> None:
        oversized_prime = MAX_FIELD_PRIME + 1

        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapRequest(prime=oversized_prime, coefficients=("1",))
        assert exc_info.value.errors()[0]["type"] == "less_than_equal"
        with pytest.raises(ValueError, match="prime number"):
            finite_field_functional_graph((1,), oversized_prime)

    def test_trailing_zero_mod_prime_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteFieldMapRequest(prime=5, coefficients=("1", "5"))
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.trailing_zero_coefficients"
        )


class TestCanonicalAndPortfolioContracts:
    @pytest.mark.parametrize(
        "coefficient",
        ["1", {"num": "01", "den": "1"}, {"num": "2", "den": "4"}],
    )
    def test_noncanonical_rational_coefficients_are_rejected(
        self, coefficient: object
    ) -> None:
        with pytest.raises(ValidationError):
            MapIterateRequest.model_validate({"coefficients": [coefficient], "n": 1})

    def test_trailing_zero_coefficients_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MapIterateRequest(coefficients=(_r(1), _r(0)), n=1)
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.trailing_zero_coefficients"
        )

    def test_fixed_point_equation_is_native_not_a_catalog_slot(self) -> None:
        operation_ids = {tool.operation_id for tool in TOOLS}
        source = polynomial_from_coefficients((0, 0, 1))

        assert "arithmetic_dynamics.fixed_point_equation.compute" not in operation_ids
        assert polynomial_coefficients(fixed_point_equation(source, 1)) == (
            Fraction(0),
            Fraction(-1),
            Fraction(1),
        )

    def test_native_polynomial_rejects_non_qq_domain(self) -> None:
        x = sympy.Symbol("x")

        with pytest.raises(ValueError, match="over QQ"):
            fixed_point_equation(sympy.Poly(x**2 + 1, x, modulus=5), 1)

    def test_native_iterate_enforces_output_degree_bound(self) -> None:
        source = polynomial_from_coefficients((1,) + (0,) * 4 + (1,))

        with pytest.raises(ValueError, match="output degree"):
            iterate_polynomial(source, 5)

    def test_native_finite_field_graph_rejects_composite_modulus(self) -> None:
        with pytest.raises(ValueError, match="prime number"):
            finite_field_functional_graph((1,), 4)
