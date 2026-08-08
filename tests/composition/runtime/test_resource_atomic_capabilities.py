"""Portfolio and smoke checks for resource-mined domain atomics."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityAssuranceLevel, CapabilityRequest
from jacobian.contracts.results import ExecutionStatus


def test_consolidated_domain_results_are_exact_computed_evidence(
    attached_complete_runtime,
) -> None:
    cases = (
        (
            "integer.compute.prime_count",
            {"n": 100},
            {"value": "25"},
        ),
        (
            "integer.compute.floor_square_root",
            {"n": 10},
            {"root": 3},
        ),
        (
            "integer.compute.floor_square_root",
            {"n": 1_000_000_000_000},
            {"root": 1_000_000},
        ),
        (
            "number_theory.compute.legendre_symbol",
            {"a": 2, "prime": 7},
            {"a": 2, "prime": 7, "symbol": 1},
        ),
        (
            "number_theory.compute.factorial_valuation",
            {"n": 10, "base": 12},
            {"n": 10, "base": 12, "valuation": 4},
        ),
        (
            "combinatorics.compute.fibonacci_pair",
            {"n": 10},
            {"n": 10, "f_n": "55", "f_n_plus_one": "89"},
        ),
        (
            "combinatorics.compute.multinomial",
            {"values": ["2", "1", "1"]},
            {"value": "12"},
        ),
        (
            "polynomial.integer.compute.shift",
            {
                "polynomial": {
                    "coefficient_order": "DESCENDING_DEGREE",
                    "coefficients": ["1", "0"],
                },
                "shift": 2,
            },
            {
                "shift": 2,
                "shifted": {
                    "coefficient_order": "DESCENDING_DEGREE",
                    "coefficients": ["1", "2"],
                },
                "convention": "SUBSTITUTE_X_PLUS_SHIFT",
            },
        ),
        (
            "matrix.rational_linear_system.solve",
            {
                "matrix": {
                    "domain": "QQ",
                    "entries": [
                        [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                        [
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                        ],
                    ],
                },
                "rhs": [
                    {"num": "5", "den": "1"},
                    {"num": "11", "den": "1"},
                ],
            },
            {
                "solution": [
                    {"num": "1", "den": "1"},
                    {"num": "2", "den": "1"},
                ],
                "convention": "UNIQUE_SOLUTION_OVER_QQ",
            },
        ),
        (
            "matrix.adjugate.compute",
            {
                "matrix": {
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["3", "4"]],
                }
            },
            {
                "adjugate": {
                    "matrix_schema_version": "1",
                    "domain": "ZZ",
                    "entries": [["4", "-2"], ["-3", "1"]],
                },
                "convention": "CLASSICAL_ADJUGATE",
            },
        ),
        (
            "graph.invariant.triangle_count.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                }
            },
            {"triangle_count": 1},
        ),
        (
            "graph.invariant.radius.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            },
            {
                "status": "COMPUTED",
                "radius": 1,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
        (
            "graph.invariant.radius.compute",
            {"graph": {"vertices": [], "edges": []}},
            {
                "status": "NOT_APPLICABLE",
                "radius": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "radius requires a nonempty connected graph",
            },
        ),
        (
            "graph.k_core.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]],
                },
                "k": 2,
            },
            {"k": 2, "vertices": ["a", "b", "c"]},
        ),
    )
    for capability_id, payload, expected in cases:
        result = attached_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert result.output["result"] == expected


def test_domain_atomic_input_failure_is_not_a_mathematical_conclusion(
    attached_complete_runtime,
) -> None:
    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="number_theory.compute.legendre_symbol",
            input={"a": 2, "prime": 9},
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "NUMBER_THEORY_OPERATION_NOT_APPLICABLE"
    assert result.assurance.verification_record_uri is None
