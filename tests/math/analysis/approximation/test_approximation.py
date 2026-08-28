"""Tests for approximation theory operations."""

from __future__ import annotations

from fractions import Fraction
from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisRequest,
    LagrangeBasisResult,
    LagrangeInterpolationRequest,
    LagrangeInterpolationResult,
    RationalNodeSet,
)
from jacobian.math.analysis.approximation._operations import (
    compute_lagrange_basis,
    compute_lagrange_interpolation,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


class RationalWire(TypedDict):
    num: str
    den: str


def _node(num: str, den: str = "1") -> RationalWire:
    return {"num": num, "den": den}


def _canonical_node(node: RationalWire) -> CanonicalRational:
    return CanonicalRational.model_validate(node)


def _node_set(*nodes: RationalWire) -> RationalNodeSet:
    return RationalNodeSet(nodes=tuple(_canonical_node(node) for node in nodes))


def _canonical_values(*values: RationalWire) -> tuple[CanonicalRational, ...]:
    return tuple(_canonical_node(value) for value in values)


def _polynomial_terms(
    polynomial: RationalPolynomial,
) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _assert_validation_code(
    exc_info: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert any(error["type"] == code for error in exc_info.value.errors())


class TestDerivedGrowthBudget:
    """Admission must keep derived results inside the canonical digit limit."""

    def test_huge_denominator_nodes_rejected(self) -> None:
        # (10^2000 + k)/10^2000 is already reduced and has a 2001-digit
        # denominator per node; four such nodes blow the component budget.
        nodes = [_node(str(10**2000 + k), "1" + "0" * 2000) for k in (1, 3, 7, 9)]
        with pytest.raises(ValidationError) as exc_info:
            _node_set(*nodes)
        _assert_validation_code(
            exc_info, "approximation_theory.node_component_budget_exceeded"
        )

    def test_many_moderate_nodes_accepted(self) -> None:
        primes = [
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
        ]
        nodes = [_node(str(p), str(10**14)) for p in primes]
        request = LagrangeBasisRequest(nodes=_node_set(*nodes))
        result = compute_lagrange_basis(request)
        assert result.node_count == 32


class TestLagrangeBasis:
    """Test Lagrange basis computation."""

    def test_two_nodes(self) -> None:
        """Basis for {0, 1} should give l_0 = 1 - x, l_1 = x."""
        nodes = _node_set(_node("0"), _node("1"))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        assert result.node_count == 2
        assert result.basis[0].index == 0
        assert result.basis[1].index == 1

    def test_three_nodes(self) -> None:
        """Basis for {0, 1, 2} has three polynomials of degree 2."""
        nodes = _node_set(_node("0"), _node("1"), _node("2"))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        assert result.node_count == 3
        for bp in result.basis:
            assert (
                max(term.exponents[0] for term in bp.polynomial.polynomial.terms) == 2
            )

    def test_barycentric_weights(self) -> None:
        """Barycentric weights for {0, 1, 2} are 1/2, -1, 1/2."""
        nodes = _node_set(_node("0"), _node("1"), _node("2"))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        weights = [bp.barycentric_weight.as_fraction() for bp in result.basis]
        assert weights == [Fraction(1, 2), Fraction(-1), Fraction(1, 2)]

    def test_basis_partition_of_unity(self) -> None:
        """Sum of basis polynomials equals 1."""
        nodes = _node_set(_node("0"), _node("1"), _node("2"))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        total: dict[tuple[int, ...], Fraction] = {}
        for bp in result.basis:
            for exponents, coefficient in _polynomial_terms(bp.polynomial).items():
                total[exponents] = total.get(exponents, Fraction(0)) + coefficient
        assert total[(0,)] == 1
        assert all(
            coefficient == 0
            for exponents, coefficient in total.items()
            if exponents != (0,)
        )


class TestLagrangeInterpolation:
    """Test Lagrange interpolation."""

    def test_two_points(self) -> None:
        """Interpolate through (0, 1), (1, 2) → x + 1."""
        nodes = _node_set(_node("0"), _node("1"))
        values = _canonical_values(_node("1"), _node("2"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        assert _polynomial_terms(result.polynomial) == {
            (0,): Fraction(1),
            (1,): Fraction(1),
        }

    def test_three_points(self) -> None:
        """Interpolate through (0, 1), (1, 3), (2, 9) → 2x^2 + 1."""
        nodes = _node_set(_node("0"), _node("1"), _node("2"))
        values = _canonical_values(_node("1"), _node("3"), _node("9"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        assert _polynomial_terms(result.polynomial) == {
            (0,): Fraction(1),
            (2,): Fraction(2),
        }

    def test_rational_nodes(self) -> None:
        """Interpolate through (0, 0), (1/2, 1/4), (1, 1) → x^2."""
        nodes = _node_set(_node("0"), _node("1", "2"), _node("1"))
        values = _canonical_values(_node("0"), _node("1", "4"), _node("1"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        assert _polynomial_terms(result.polynomial) == {(2,): Fraction(1)}

    def test_constant_interpolation(self) -> None:
        """Interpolate constant values → constant polynomial."""
        nodes = _node_set(_node("0"), _node("1"), _node("2"))
        values = _canonical_values(_node("5"), _node("5"), _node("5"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        assert _polynomial_terms(result.polynomial) == {(0,): Fraction(5)}

    def test_mismatched_lengths_rejected(self) -> None:
        """Values length must match nodes length."""
        nodes = _node_set(_node("0"), _node("1"))
        request = LagrangeInterpolationRequest(
            nodes=nodes,
            values=_canonical_values(_node("1"), _node("2"), _node("3")),
        )
        with pytest.raises(ValueError, match="same length"):
            compute_lagrange_interpolation(request)


class TestLagrangeInterpolationAxisBinding:
    def test_produced_result_uses_x_axis(self) -> None:
        nodes = _node_set(_node("0"), _node("1"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(
                nodes=nodes, values=_canonical_values(_node("1"), _node("2"))
            )
        )
        assert result.polynomial.variables == ("x",)
        LagrangeInterpolationResult.model_validate(result.model_dump())

    def test_foreign_variable_axis_rejected(self) -> None:
        """A revalidated result cannot carry a different parent ring."""
        from jacobian.math.polynomials.values import (
            RationalPolynomial,
            RationalPolynomialTerm,
            SparseRationalPolynomial,
        )

        ypoly = RationalPolynomial(
            variables=("y",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,),
                    ),
                )
            ),
        )
        with pytest.raises(ValidationError) as exc_info:
            LagrangeInterpolationResult(polynomial=ypoly)
        _assert_validation_code(
            exc_info, "approximation_theory.interpolation_variable_mismatch"
        )


class TestLagrangeBasisSourceBinding:
    def _nodes(self, *values: str) -> RationalNodeSet:
        return _node_set(*(_node(value) for value in values))

    def test_genuine_result_round_trips(self) -> None:
        request = LagrangeBasisRequest(nodes=self._nodes("0", "1", "2"))
        result = compute_lagrange_basis(request)
        assert result.nodes.nodes == request.nodes.nodes
        LagrangeBasisResult.model_validate(result.model_dump(mode="json"))

    def test_duplicate_index_rejected(self) -> None:
        request = LagrangeBasisRequest(nodes=self._nodes("0", "1"))
        genuine = compute_lagrange_basis(request)
        payload = genuine.model_dump()
        payload["basis"][1]["index"] = 0
        with pytest.raises(ValidationError) as exc_info:
            LagrangeBasisResult.model_validate(payload)
        _assert_validation_code(exc_info, "approximation_theory.basis_indices_invalid")

    def test_missing_node_set_rejected(self) -> None:
        """A result without the retained nodes cannot revalidate."""
        request = LagrangeBasisRequest(nodes=self._nodes("0", "1"))
        genuine = compute_lagrange_basis(request)
        payload = genuine.model_dump()
        del payload["nodes"]
        with pytest.raises(ValidationError):
            LagrangeBasisResult.model_validate(payload)

    def test_swapped_nodes_rejected_by_delta_replay(self) -> None:
        """A basis computed on other nodes fails l_k(x_j) = delta_kj."""
        basis_a = compute_lagrange_basis(
            LagrangeBasisRequest(nodes=self._nodes("0", "1"))
        )
        payload = basis_a.model_dump()
        payload["nodes"] = {
            "nodes": [
                {"num": "2", "den": "1"},
                {"num": "5", "den": "1"},
            ]
        }
        with pytest.raises(ValidationError):
            LagrangeBasisResult.model_validate(payload)

    def test_forged_higher_degree_basis_rejected(self) -> None:
        """(x-1)^2 passes the delta replay on nodes (0, 1) but is not l_0.

        The degree bound (degree < node_count) is what pins the cardinal
        polynomial, so a degree-2 forgery with correct node values and weight
        must be rejected.
        """
        request = LagrangeBasisRequest(nodes=self._nodes("0", "1"))
        payload = {
            "nodes": request.model_dump(mode="json")["nodes"],
            "node_count": 2,
            "basis": [
                {
                    "index": 0,
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "barycentric_weight": {"num": "-1", "den": "1"},
                },
                {
                    "index": 1,
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                            ]
                        },
                    },
                    "barycentric_weight": {"num": "1", "den": "1"},
                },
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            LagrangeBasisResult.model_validate(payload)
        _assert_validation_code(exc_info, "approximation_theory.basis_degree_exceeded")

    def test_tampered_weight_rejected(self) -> None:
        request = LagrangeBasisRequest(nodes=self._nodes("0", "1", "2"))
        genuine = compute_lagrange_basis(request)
        payload = genuine.model_dump()
        payload["basis"][1]["barycentric_weight"] = {"num": "7", "den": "3"}
        with pytest.raises(ValidationError) as exc_info:
            LagrangeBasisResult.model_validate(payload)
        _assert_validation_code(
            exc_info, "approximation_theory.barycentric_weight_mismatch"
        )


def _canonical(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _dense_polynomial(coeffs: list[Fraction]) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(c),
            exponents=(exp,),
        )
        for exp, c in sorted(enumerate(coeffs), key=lambda pair: pair[0], reverse=True)
        if c != 0
    )
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(terms=terms),
    )


class TestLagrangeInterpolateNative:
    """The native interpolation surface over canonical domain values."""

    def test_reviewer_counterexample_returns_exact_quadratic(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        result = lagrange_interpolate(
            (_canonical("0"), _canonical("1"), _canonical("2")),
            (_canonical("1"), _canonical("3"), _canonical("9")),
        )
        assert result == _dense_polynomial([Fraction(1), Fraction(0), Fraction(2)])

    def test_returned_value_is_a_canonical_consumer_input(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        nodes = (_canonical("0"), _canonical("1", "2"), _canonical("3"))
        values = (_canonical("-1"), _canonical("1", "4"), _canonical("5"))
        result = RationalPolynomial.model_validate(
            lagrange_interpolate(nodes, values).model_dump()
        )
        assert result.variables == ("x",)

    def test_single_node_constant(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        result = lagrange_interpolate(
            (_canonical("7", "3"),), (_canonical("-11", "5"),)
        )
        assert result == _dense_polynomial([Fraction(-11, 5)])

    def test_unsorted_nodes_rejected(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        with pytest.raises(ValidationError) as exc_info:
            lagrange_interpolate(
                (_canonical("1"), _canonical("0")), (_canonical("1"), _canonical("2"))
            )
        _assert_validation_code(exc_info, "approximation_theory.nodes_not_increasing")

    def test_duplicate_nodes_rejected(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        with pytest.raises(ValidationError) as exc_info:
            lagrange_interpolate(
                (_canonical("0"), _canonical("0")), (_canonical("1"), _canonical("2"))
            )
        _assert_validation_code(exc_info, "approximation_theory.nodes_not_distinct")

    def test_mismatched_values_rejected(self) -> None:
        from jacobian.math.analysis.approximation import lagrange_interpolate

        with pytest.raises(
            ValueError, match="values must have the same length as nodes"
        ):
            lagrange_interpolate((_canonical("0"), _canonical("1")), (_canonical("1"),))


class TestInterpolationAdmissionDisposition:
    """The duplicate interpolant must stay out of discovery, native only."""

    def test_interpolation_is_native_only_with_supported_symbol(self) -> None:
        import importlib

        from jacobian.catalog.admission import AdmissionDecision
        from jacobian.math.analysis.approximation._admission import REGISTRATION

        record = next(
            admission
            for admission in REGISTRATION.admissions
            if admission.operation_id == "approximation.lagrange.interpolate.compute"
        )
        assert record.decision is AdmissionDecision.NATIVE_ONLY
        assert record.native_symbol is not None
        module_name, _, symbol_name = record.native_symbol.rpartition(".")
        module = importlib.import_module(module_name)
        assert symbol_name in module.__all__
        assert callable(getattr(module, symbol_name))

    def test_duplicate_interpolant_is_not_served(self) -> None:
        from jacobian.catalog.builtins import BUILTIN_TOOLS

        ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert "approximation.lagrange.interpolate.compute" not in ids
        assert "approximation.lagrange.basis.compute" in ids
