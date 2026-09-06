"""Known-answer and adversarial tests for arithmetic dynamics."""

from copy import deepcopy
from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.dynamics.arithmetic import (
    finite_field_functional_graph,
    fixed_point_equation,
    iterate_polynomial,
    polynomial_coefficients,
    polynomial_from_coefficients,
)
from jacobian.math.dynamics.arithmetic._models import (
    MAX_FIELD_PRIME,
    CycleMultiplierRequest,
    DynatomicPolynomialRequest,
    FiniteFieldMapRequest,
    FiniteFieldMapResult,
    MapIterateRequest,
    OrbitPrefixRequest,
    OrbitPrefixResult,
)
from jacobian.math.dynamics.arithmetic._tools import (
    TOOLS,
    compute_cycle_multiplier,
    compute_dynatomic_polynomial,
    compute_finite_field_map,
    compute_map_iterate,
    compute_orbit_prefix,
    verify_cycle_multiplier,
    verify_dynatomic_polynomial,
    verify_finite_field_map,
    verify_map_iterate,
    verify_orbit_prefix,
)
from jacobian.math.finite_fields.operations import (
    finite_polynomial,
    finite_polynomial_map,
)
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)


def _r(value: int | str) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _p(*values: CanonicalRational):
    return polynomial_from_coefficients(tuple(value.as_fraction() for value in values))


def _coefficients(polynomial):
    return tuple(CanonicalRational.from_fraction(value) for value in polynomial_coefficients(polynomial))


def _fm(prime: int, *values: int):
    presentation = FiniteFieldPresentation(
        characteristic=prime, modulus_coefficients=(0, 1), generator="x"
    )
    coefficients = tuple(
        FiniteFieldElement(presentation=presentation, coordinates=(value % prime,))
        for value in values
    )
    return finite_polynomial_map(finite_polynomial(presentation, coefficients, variable="x"))


class TestMapIterate:
    def test_zero_iterate_is_identity(self) -> None:
        result = compute_map_iterate(
            MapIterateRequest(polynomial=_p(_r(1), _r(0), _r(1)), n=0)
        )

        assert _coefficients(result.polynomial) == (_r(0), _r(1))
        assert result.degree == 1

    def test_second_iterate_is_exact(self) -> None:
        result = compute_map_iterate(
            MapIterateRequest(polynomial=_p(_r(1), _r(0), _r(1)), n=2)
        )

        assert _coefficients(result.polynomial) == (_r(2), _r(0), _r(2), _r(0), _r(1))
        assert result.degree == 4

    def test_zero_polynomial_iterates_without_backend_degree_coercion(self) -> None:
        result = compute_map_iterate(MapIterateRequest(polynomial=_p(_r(0),), n=3))

        assert _coefficients(result.polynomial) == (_r(0),)
        assert result.degree == 0

    def test_degree_growth_beyond_output_bound_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_map_iterate(
                MapIterateRequest(
                    polynomial=_p(_r(1), _r(0), _r(0), _r(0), _r(0), _r(1)), n=5
                )
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.iterate_degree_exceeds_bound"
        )


class TestOrbitPrefix:
    def test_repeat_proves_preperiod_and_period(self) -> None:
        result = compute_orbit_prefix(
            OrbitPrefixRequest(
                polynomial=_p(_r(0), _r(0), _r(1)), start=_r(0), max_steps=5
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
            OrbitPrefixRequest(polynomial=_p(_r(1), _r(1)), start=_r(0), max_steps=3)
        )

        assert result.orbit == (_r(0), _r(1), _r(2), _r(3))
        assert result.termination == "STEP_BOUND_REACHED"
        assert result.repeat is None
        assert result.eventual_behavior_complete is False
        assert result.truncated is True

    def test_zero_step_request_is_an_explicit_truncated_prefix(self) -> None:
        result = compute_orbit_prefix(
            OrbitPrefixRequest(polynomial=_p(_r(1), _r(1)), start=_r(0), max_steps=0)
        )

        assert result.orbit == (_r(0),)
        assert result.computed_steps == 0
        assert result.termination == "STEP_BOUND_REACHED"
        assert result.truncated is True

    def test_output_growth_stops_with_nonconcluding_typed_boundary(self) -> None:
        degree_thirty = (_r(0),) * 30 + (_r(1),)
        result = compute_orbit_prefix(
            OrbitPrefixRequest(
                polynomial=_p(*degree_thirty),
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
        claim = OrbitPrefixResult(
            source_polynomial=_p(_r(1), _r(1)),
            start=_r(0),
            orbit=(_r(0), _r(1)),
            requested_steps=1,
            computed_steps=1,
            termination="STEP_BOUND_REACHED",
            repeat=None,
            eventual_behavior_complete=True,
            truncated=False,
        )
        assert not verify_orbit_prefix(claim)


class TestDynatomicPolynomial:
    def test_first_dynatomic_polynomial(self) -> None:
        result = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(polynomial=_p(_r(0), _r(0), _r(1)), n=1)
        )

        assert _coefficients(result.polynomial) == (_r(0), _r(-1), _r(1))

    def test_square_factor_mobius_case_and_divisor_product_identity(self) -> None:
        source = polynomial_from_coefficients((0, 0, 1))
        compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(polynomial=_p(_r(0), _r(0), _r(1)), n=1)
        )
        compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(polynomial=_p(_r(0), _r(0), _r(1)), n=2)
        )
        phi_4 = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(polynomial=_p(_r(0), _r(0), _r(1)), n=4)
        )
        assert _coefficients(phi_4.polynomial) == tuple(
            _r(value) for value in (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1)
        )
        assert polynomial_coefficients(fixed_point_equation(source, 4)) == (
            Fraction(0),
            Fraction(-1),
            *(Fraction(0),) * 14,
            Fraction(1),
        )

    def test_linear_map_is_outside_dynatomic_contract(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_dynatomic_polynomial(
                DynatomicPolynomialRequest(polynomial=_p(_r(1), _r(1)), n=2)
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.dynatomic_degree_too_small"
        )


class TestCycleMultiplier:
    def test_validated_two_cycle_multiplier(self) -> None:
        result = compute_cycle_multiplier(
            CycleMultiplierRequest(polynomial=_p(_r(1), _r(-1)), cycle=(_r(0), _r(1)))
        )

        assert result.multiplier == _r(1)
        assert result.period == 2

    def test_arbitrary_points_cannot_be_labeled_a_cycle(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_cycle_multiplier(
                CycleMultiplierRequest(
                    polynomial=_p(_r(0), _r(0), _r(1)), cycle=(_r(0), _r(1))
                )
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.cycle_map_mismatch"
        )

    def test_repeated_cycle_points_are_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_cycle_multiplier(
                CycleMultiplierRequest(
                    polynomial=_p(_r(0), _r(0), _r(1)), cycle=(_r(0), _r(0))
                )
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.cycle_points_not_distinct"
        )


class TestFiniteFieldFunctionalGraph:
    def test_x_squared_mod_five_has_complete_canonical_graph(self) -> None:
        result = compute_finite_field_map(
            FiniteFieldMapRequest(polynomial_map=_fm(5, 0, 0, 1))
        )

        assert result.edges == ((0, 0), (1, 1), (2, 4), (3, 4), (4, 1))
        assert result.cycles == ((0,), (1,))
        assert result.tail_lengths == (0, 0, 2, 2, 1)

    @pytest.mark.parametrize(
        ("prime", "coefficients"),
        [(2, (1, 1, 1)), (3, (2, 0, 1)), (5, (1, 1)), (7, (3, 2, 1))],
    )
    def test_edges_cycles_and_tail_lengths_replay(
        self, prime: int, coefficients: tuple[int, ...]
    ) -> None:
        result = compute_finite_field_map(
            FiniteFieldMapRequest(
                polynomial_map=_fm(prime, *coefficients)
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

    def test_result_parse_retains_structural_claims_without_replaying_graph(
        self,
    ) -> None:
        result = FiniteFieldMapResult(
            polynomial_map=_fm(2, 0),
            edges=((0, 0), (1, 0)),
            cycles=((0,),),
            tail_lengths=(0, 0),
        )

        assert FiniteFieldMapResult.model_validate_json(result.model_dump_json()) == result
        assert not verify_finite_field_map(result)
        restored = FiniteFieldMapResult.model_validate_json(result.model_dump_json())
        assert restored.polynomial_map == result.polynomial_map
        forged = deepcopy(restored.model_dump(mode="json"))
        forged["edges"][0][1] = 1
        assert not verify_finite_field_map(FiniteFieldMapResult.model_validate(forged))

    def test_nonprime_modulus_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            FiniteFieldMapRequest(polynomial_map=_fm(4, 1))

    def test_noncanonical_integer_coefficients_have_an_owner_code(self) -> None:
        with pytest.raises(ValidationError):
            FiniteFieldMapRequest.model_validate({"polynomial_map": {"bad": 1}})

    def test_wire_and_native_paths_share_the_field_prime_bound(self) -> None:
        oversized_prime = MAX_FIELD_PRIME + 1

        with pytest.raises(OperationDomainValidationError):
            FiniteFieldMapRequest(polynomial_map=_fm(oversized_prime, 1))
        with pytest.raises(ValueError, match="prime number"):
            finite_field_functional_graph((1,), oversized_prime)

    def test_trailing_zero_mod_prime_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="trailing zeros"):
            finite_field_functional_graph((1, 5), 5)


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
        with pytest.raises(ValueError, match="trailing zeros"):
            _p(_r(1), _r(0))

    def test_serialized_claims_retain_sources_and_verify_forgery(self) -> None:
        iterate = compute_map_iterate(
            MapIterateRequest(polynomial=_p(_r(1), _r(0), _r(1)), n=1)
        )
        restored_iterate = type(iterate).model_validate_json(iterate.model_dump_json())
        assert restored_iterate.source_polynomial == iterate.source_polynomial
        assert verify_map_iterate(restored_iterate)
        forged_iterate = deepcopy(restored_iterate.model_dump(mode="json"))
        forged_iterate["degree"] = 0
        assert not verify_map_iterate(type(iterate).model_validate(forged_iterate))

        dynatomic = compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(polynomial=_p(_r(0), _r(0), _r(1)), n=1)
        )
        restored_dynatomic = type(dynatomic).model_validate_json(
            dynatomic.model_dump_json()
        )
        assert restored_dynatomic.source_polynomial == dynatomic.source_polynomial
        assert verify_dynatomic_polynomial(restored_dynatomic)
        forged_dynatomic = deepcopy(restored_dynatomic.model_dump(mode="json"))
        forged_dynatomic["degree"] = 0
        assert not verify_dynatomic_polynomial(
            type(dynatomic).model_validate(forged_dynatomic)
        )

        cycle = compute_cycle_multiplier(
            CycleMultiplierRequest(polynomial=_p(_r(1), _r(-1)), cycle=(_r(0), _r(1)))
        )
        restored_cycle = type(cycle).model_validate_json(cycle.model_dump_json())
        assert restored_cycle.source_polynomial == cycle.source_polynomial
        assert verify_cycle_multiplier(restored_cycle)
        forged_cycle = deepcopy(restored_cycle.model_dump(mode="json"))
        forged_cycle["multiplier"] = {"num": "2", "den": "1"}
        assert not verify_cycle_multiplier(type(cycle).model_validate(forged_cycle))

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

        with pytest.raises(ValueError):
            fixed_point_equation(sympy.Poly(x**2 + 1, x, modulus=5), 1)

    def test_native_iterate_enforces_output_degree_bound(self) -> None:
        source = polynomial_from_coefficients((1,) + (0,) * 4 + (1,))

        with pytest.raises(ValueError, match="output degree"):
            iterate_polynomial(source, 5)

    def test_native_finite_field_graph_rejects_composite_modulus(self) -> None:
        with pytest.raises(ValueError, match="prime number"):
            finite_field_functional_graph((1,), 4)
