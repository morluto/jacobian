"""Dispatch-level publication checks for the factor-avoidance DFA operation.

These exercise the published operation boundary, so they live in the catalog
lane rather than under ``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.languages.regular.operations import dfa_run
from jacobian.math.logic.languages.regular.values import DFA


def test_published_operation_output_is_usable_by_regular_language_consumer() -> None:
    invocation = invoke_operation(
        "free_algebra.factor_avoidance.dfa.compute",
        {"alphabet": ["x", "y"], "forbidden_factors": [["x", "y"]]},
        Catalog.open(),
    )
    carrier = DFA.model_validate(invocation.output)
    assert carrier.alphabet is not None
    assert carrier.alphabet.symbols == ("x", "y")
    assert dfa_run(carrier, (1, 0))[0]
    assert not dfa_run(carrier, (0, 1))[0]


def test_named_dfa_composes_unchanged_through_run_count_and_complement():
    invocation = invoke_operation(
        "free_algebra.factor_avoidance.dfa.compute",
        {"alphabet": ["x", "y"], "forbidden_factors": [["x", "y"]]},
        Catalog.open(),
    )
    dfa = invocation.output
    run = invoke_operation(
        "regular_language.run.check",
        {"dfa": dfa, "word": [1, 0]},
        Catalog.open(),
    )
    count = invoke_operation(
        "regular_language.count_words.compute",
        {"dfa": dfa, "word_length": 2},
        Catalog.open(),
    )
    complement = invoke_operation(
        "regular_language.complement.compute", {"dfa": dfa}, Catalog.open()
    )
    assert run.output["accepted"] is True
    assert count.output["dfa"]["alphabet"] == {"symbols": ["x", "y"]}
    assert complement.output["dfa"]["alphabet"] == {"symbols": ["x", "y"]}
