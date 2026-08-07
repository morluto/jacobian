from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.probability import build_finite_probability_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_finite_probability_bundle()
    ) as services:
        yield services


def _rational(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def _complex(
    real: int,
    imaginary: int = 0,
) -> dict[str, dict[str, str]]:
    return {
        "real": _rational(real),
        "imaginary": _rational(imaginary),
    }


def _distribution(
    *atoms: tuple[int, int, int],
) -> dict[str, list[dict[str, dict[str, str]]]]:
    return {
        "atoms": [
            {
                "value": _rational(value),
                "probability": _rational(numerator, denominator),
            }
            for value, numerator, denominator in atoms
        ]
    }


def test_finite_raw_moment_preserves_exact_contributions(
    domain_services: DomainTestServices,
) -> None:
    runtime = domain_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.raw_moment.compute",
            input={
                "atoms": [
                    {"value": _rational(-1), "probability": _rational(1, 2)},
                    {"value": _rational(3), "probability": _rational(1, 2)},
                ],
                "order": 2,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["moment"] == _rational(5)
    assert [
        item["contribution"] for item in result.output["result"]["contributions"]
    ] == [_rational(1, 2), _rational(9, 2)]
    assert result.output["result"]["verification"] == "UNVERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()


def test_invalid_finite_distribution_fails_before_artifact_writes(
    domain_services: DomainTestServices,
) -> None:
    runtime = domain_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.raw_moment.compute",
            input={
                "atoms": [
                    {"value": _rational(0), "probability": _rational(1, 3)},
                    {"value": _rational(1), "probability": _rational(1, 3)},
                ],
                "order": 1,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()


def test_finite_event_probability_preserves_selected_atom_contributions(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("probability.finite_distribution.event_probability.compute"),
            input={
                "distribution": _distribution(
                    (-1, 1, 4),
                    (0, 1, 2),
                    (2, 1, 4),
                ),
                "event_values": [_rational(-1), _rational(2)],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["event_probability"] == _rational(1, 2)
    assert result.output["result"]["selected_atoms"] == [
        {
            "value": _rational(-1),
            "probability": _rational(1, 4),
        },
        {
            "value": _rational(2),
            "probability": _rational(1, 4),
        },
    ]
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()


def test_finite_conditioning_returns_one_normalized_distribution(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.condition.compute",
            input={
                "distribution": _distribution(
                    (-1, 1, 6),
                    (0, 1, 3),
                    (2, 1, 2),
                ),
                "event_values": [_rational(-1), _rational(2)],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["event_probability"] == _rational(2, 3)
    assert result.output["result"]["distribution"] == _distribution(
        (-1, 1, 4),
        (2, 3, 4),
    )
    assert [
        contribution["conditioned_probability"]
        for contribution in result.output["result"]["contributions"]
    ] == [_rational(1, 4), _rational(3, 4)]


def test_zero_mass_conditioning_is_a_non_conclusion_without_artifacts(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.condition.compute",
            input={
                "distribution": _distribution(
                    (0, 0, 1),
                    (1, 1, 1),
                ),
                "event_values": [_rational(0)],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "FINITE_CONDITIONING_ZERO_MASS"
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.completeness.status is CapabilityCompletenessStatus.NOT_APPLICABLE
    assert result.artifact_uris == ()


def test_finite_pushforward_collapses_equal_target_atoms(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.pushforward.compute",
            input={
                "distribution": _distribution(
                    (-1, 1, 4),
                    (0, 1, 2),
                    (1, 1, 4),
                ),
                "mapping": [
                    {"source": _rational(-1), "target": _rational(1)},
                    {"source": _rational(0), "target": _rational(0)},
                    {"source": _rational(1), "target": _rational(1)},
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["distribution"] == _distribution(
        (0, 1, 2),
        (1, 1, 2),
    )
    assert len(result.output["result"]["contributions"]) == 3


def test_finite_convolution_aggregates_all_independent_pairs(
    domain_services: DomainTestServices,
) -> None:
    fair_bit = _distribution((0, 1, 2), (1, 1, 2))
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.convolution.compute",
            input={"left": fair_bit, "right": fair_bit},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["distribution"] == _distribution(
        (0, 1, 4),
        (1, 1, 2),
        (2, 1, 4),
    )
    assert len(result.output["result"]["contributions"]) == 4


def test_finite_convolution_rejects_distinct_support_above_result_bound(
    domain_services: DomainTestServices,
) -> None:
    left_values = tuple(range(17))
    right_values = tuple(value * 100 for value in range(16))
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.convolution.compute",
            input={
                "left": _distribution(
                    *((value, 1, 17) for value in left_values),
                ),
                "right": _distribution(
                    *((value, 1, 16) for value in right_values),
                ),
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()


def test_incomplete_pushforward_mapping_fails_before_artifact_writes(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.pushforward.compute",
            input={
                "distribution": _distribution((0, 1, 2), (1, 1, 2)),
                "mapping": [
                    {"source": _rational(0), "target": _rational(1)},
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()


def test_gaussian_polynomial_moment_preserves_complete_complex_contraction(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 1,
                    "terms": [
                        {"coefficient": _complex(1), "exponents": [0]},
                        {"coefficient": _complex(0, 1), "exponents": [1]},
                    ],
                },
                "order": 2,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = result.output["result"]
    assert computed["moment"] == _complex(0)
    assert computed["expansion_path_count"] == 4
    assert computed["expanded_monomial_count"] == 3
    assert [item["exponents"] for item in computed["contractions"]] == [
        [0],
        [1],
        [2],
    ]
    assert [item["gaussian_moment_factor"] for item in computed["contractions"]] == [
        "1",
        "0",
        "1",
    ]
    assert [item["contribution"] for item in computed["contractions"]] == [
        _complex(1),
        _complex(0),
        _complex(-1),
    ]
    assert computed["completeness"] == "COMPLETE_BOUNDED_EXPANSION"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()


def test_multivariate_gaussian_polynomial_moment_uses_independence(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 2,
                    "terms": [
                        {"coefficient": _complex(1), "exponents": [0, 1]},
                        {"coefficient": _complex(1), "exponents": [1, 0]},
                    ],
                },
                "order": 4,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["moment"] == _complex(12)
    assert result.output["result"]["expansion_path_count"] == 16
    assert result.output["result"]["expanded_monomial_count"] == 5


def test_gaussian_polynomial_zero_order_is_the_constant_one(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 1,
                    "terms": [{"coefficient": _complex(7, -3), "exponents": [5]}],
                },
                "order": 0,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["moment"] == _complex(1)
    assert result.output["result"]["contractions"][0]["exponents"] == [0]


def test_gaussian_expansion_above_bound_fails_before_artifact_writes(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 1,
                    "terms": [
                        {"coefficient": _complex(1), "exponents": [0]},
                        {"coefficient": _complex(1), "exponents": [1]},
                        {"coefficient": _complex(1), "exponents": [2]},
                        {"coefficient": _complex(1), "exponents": [3]},
                        {"coefficient": _complex(1), "exponents": [4]},
                    ],
                },
                "order": 8,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()


def test_gaussian_expansion_at_raised_bound_succeeds(
    domain_services: DomainTestServices,
) -> None:
    """4 terms at order 8 = 65536 paths, exactly the raised bound."""
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 1,
                    "terms": [
                        {"coefficient": _complex(1), "exponents": [0]},
                        {"coefficient": _complex(1), "exponents": [1]},
                        {"coefficient": _complex(1), "exponents": [2]},
                        {"coefficient": _complex(1), "exponents": [3]},
                    ],
                },
                "order": 8,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = result.output["result"]
    assert computed["expansion_path_count"] == 65536
    assert computed["expanded_monomial_count"] == len(computed["contractions"])
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()


def test_gaussian_complex_coefficient_ledger_matches_stdlib_replay(
    domain_services: DomainTestServices,
) -> None:
    """Nontrivial complex coefficients: FLINT fmpq_mpoly pair must match the
    independent stdlib Fraction contraction for the full moment and ledger."""
    from fractions import Fraction

    def _gaussian_univariate_moment(exponent: int) -> int:
        if exponent % 2:
            return 0
        result = 1
        for factor in range(1, exponent, 2):
            result *= factor
        return result

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 2,
                    "terms": [
                        {"coefficient": _complex(1, -1), "exponents": [0, 1]},
                        {"coefficient": _complex(2, 1), "exponents": [1, 0]},
                    ],
                },
                "order": 4,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = result.output["result"]

    # Independent stdlib replay
    base = [
        ((0, 1), (Fraction(1), Fraction(-1))),
        ((1, 0), (Fraction(2), Fraction(1))),
    ]
    expanded: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {
        (0, 0): (Fraction(1), Fraction(0)),
    }
    for _ in range(4):
        nxt: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {}
        for left_exponents, left_coefficient in expanded.items():
            for right_exponents, right_coefficient in base:
                exponents = tuple(
                    left + right
                    for left, right in zip(left_exponents, right_exponents, strict=True)
                )
                product = (
                    left_coefficient[0] * right_coefficient[0]
                    - left_coefficient[1] * right_coefficient[1],
                    left_coefficient[0] * right_coefficient[1]
                    + left_coefficient[1] * right_coefficient[0],
                )
                previous = nxt.get(exponents, (Fraction(), Fraction()))
                nxt[exponents] = (previous[0] + product[0], previous[1] + product[1])
        expanded = {
            exponents: coefficient
            for exponents, coefficient in nxt.items()
            if coefficient != (Fraction(), Fraction())
        }

    total = (Fraction(), Fraction())
    for exponents, coefficient in sorted(expanded.items()):
        gaussian_factor = 1
        for exponent in exponents:
            gaussian_factor *= _gaussian_univariate_moment(exponent)
        total = (
            total[0] + coefficient[0] * gaussian_factor,
            total[1] + coefficient[1] * gaussian_factor,
        )

    flint_real = Fraction(
        int(computed["moment"]["real"]["num"]), int(computed["moment"]["real"]["den"])
    )
    flint_imag = Fraction(
        int(computed["moment"]["imaginary"]["num"]),
        int(computed["moment"]["imaginary"]["den"]),
    )
    assert flint_real == total[0]
    assert flint_imag == total[1]

    flint_exponents = [tuple(item["exponents"]) for item in computed["contractions"]]
    stdlib_exponents = sorted(expanded.keys())
    assert flint_exponents == stdlib_exponents
    for item, exponents in zip(computed["contractions"], stdlib_exponents, strict=True):
        stdlib_coeff = expanded[exponents]
        flint_coeff_real = Fraction(
            int(item["expanded_coefficient"]["real"]["num"]),
            int(item["expanded_coefficient"]["real"]["den"]),
        )
        flint_coeff_imag = Fraction(
            int(item["expanded_coefficient"]["imaginary"]["num"]),
            int(item["expanded_coefficient"]["imaginary"]["den"]),
        )
        assert flint_coeff_real == stdlib_coeff[0]
        assert flint_coeff_imag == stdlib_coeff[1]


def test_gaussian_denominator_growth_fails_before_artifact_writes(
    domain_services: DomainTestServices,
) -> None:
    denominators = [str(10**127 + offset) for offset in range(1, 17)]
    exponents = [
        [(index >> bit) & 1 for bit in reversed(range(4))] for index in range(16)
    ]
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.gaussian_polynomial.moment.compute",
            input={
                "polynomial": {
                    "variable_count": 4,
                    "terms": [
                        {
                            "coefficient": {
                                "real": {"num": "1", "den": denominator},
                                "imaginary": _rational(0),
                            },
                            "exponents": exponent,
                        }
                        for denominator, exponent in zip(
                            denominators, exponents, strict=True
                        )
                    ],
                },
                "order": 3,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()


def test_graph_reliability_exhausts_all_edge_states_exactly(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.compute"
            ),
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                },
                "edge_probabilities": [
                    {"edge": edge, "open_probability": _rational(1, 2)}
                    for edge in (["a", "b"], ["a", "c"], ["b", "c"])
                ],
                "terminals": ["a", "c"],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = result.output["result"]
    assert computed["connection_probability"] == _rational(5, 8)
    assert computed["visited_states"] == 8
    assert len(computed["states"]) == 8
    assert computed["completeness"] == "COMPLETE"
    assert computed["truncated"] is False
    assert computed["termination_reason"] == "EXHAUSTED"
    assert sum(1 for state in computed["states"] if state["terminals_connected"]) == 5
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_graph_reliability_disconnected_terminals_have_zero_probability(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.compute"
            ),
            input={
                "graph": {"vertices": ["a", "b"], "edges": []},
                "edge_probabilities": [],
                "terminals": ["a", "b"],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["connection_probability"] == _rational(0)
    assert result.output["result"]["visited_states"] == 1


def test_graph_reliability_rejects_incomplete_edge_probability_binding(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.compute"
            ),
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "edge_probabilities": [
                    {
                        "edge": ["a", "b"],
                        "open_probability": _rational(1, 2),
                    }
                ],
                "terminals": ["a", "c"],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()


def test_graph_reliability_rejects_ledger_above_artifact_budget(
    domain_services: DomainTestServices,
) -> None:
    vertices = [f"{index}" + "x" * 255 for index in range(6)]
    edges = [
        [vertices[left], vertices[right]]
        for left in range(6)
        for right in range(left + 1, 6)
    ][:12]
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.compute"
            ),
            input={
                "graph": {"vertices": vertices, "edges": edges},
                "edge_probabilities": [
                    {"edge": edge, "open_probability": _rational(1, 2)}
                    for edge in edges
                ],
                "terminals": [vertices[0], vertices[-1]],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_PROBABILITY_REQUEST"
    assert result.artifact_uris == ()
