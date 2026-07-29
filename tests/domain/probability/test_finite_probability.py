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
from jacobian.domains.probability import FINITE_PROBABILITY_BUNDLE


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", FINITE_PROBABILITY_BUNDLE
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
    assert len(result.artifact_uris) == 2
    persisted = runtime.core.store.get(result.artifact_uris[1])
    assert persisted.payload == result.output["result"]


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
    assert len(result.artifact_uris) == 2


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
    assert result.episode_uri is None


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
    assert len(result.artifact_uris) == 2


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
                    ],
                },
                "order": 8,
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
