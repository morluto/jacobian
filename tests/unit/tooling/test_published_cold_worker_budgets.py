from __future__ import annotations

from jacobian.domains.analysis.operations import POINT_ENCLOSURE_OPERATIONS
from jacobian.domains.polynomial.groebner import POLYNOMIAL_GROEBNER_OPERATION


def test_arb_example_and_vocabulary_match_supported_functions() -> None:
    spec = POINT_ENCLOSURE_OPERATIONS[0]
    assert spec.examples[0].input["wall_seconds"] == 10
    assert {
        "square-root",
        "sqrt",
        "logarithm",
        "log",
        "exponential",
        "exp",
        "sine",
        "sin",
        "cosine",
        "cos",
    } <= set(spec.tags)
    assert "square root, logarithm, exponential, sine, or cosine" in spec.description


def test_groebner_example_preserves_cold_worker_startup_budget() -> None:
    assert (
        POLYNOMIAL_GROEBNER_OPERATION.examples[0].input["resource_budget"][
            "wall_seconds"
        ]
        == 10
    )
