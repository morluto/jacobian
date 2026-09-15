"""Factor-complexity and Rauzy slice (#3713)."""

from __future__ import annotations

import pytest

from jacobian.math.logic.languages.words import (
    FiniteWord,
    factor_complexity_prefix,
    rauzy_graph,
)
from jacobian.math.logic.languages.words._models import (
    FactorComplexityRequest,
    RauzyGraphRequest,
    SubstitutionFactorComplexityRequest,
)
from jacobian.math.logic.languages.words._tools import (
    compute_factor_complexity,
    compute_rauzy_graph,
    compute_substitution_factor_complexity,
)


def _word() -> FiniteWord:
    return FiniteWord(alphabet=("a", "b"), letters=("a", "b", "a", "a", "b"))


def test_complexity_known_answer() -> None:
    analysis = factor_complexity_prefix(_word(), 2)
    assert analysis.complexity == (1, 2, 3)
    assert analysis.families[2] == (("a", "b"), ("b", "a"), ("a", "a"))
    result = compute_factor_complexity(
        FactorComplexityRequest(word=_word(), max_order=2)
    )
    assert result.complexity == (1, 2, 3)
    assert result.scope == "FACTORS_OF_SUPPLIED_PREFIX_ONLY"


def test_rauzy_graph_known_answer() -> None:
    analysis = rauzy_graph(_word(), 1)
    assert analysis.vertices == (("a",), ("b",))
    assert tuple(label for _, label, _ in analysis.edges) == (
        ("a", "b"),
        ("b", "a"),
        ("a", "a"),
    )
    result = compute_rauzy_graph(RauzyGraphRequest(word=_word(), order=1))
    assert len(result.edges) == 3
    for edge in result.edges:
        assert edge.source == edge.label[:-1]
        assert edge.target == edge.label[1:]
        assert edge.occurrences


def test_sturmian_prefix_complexity() -> None:
    from jacobian.math.logic.languages.words.values import ProlongableSubstitution

    source = ProlongableSubstitution.model_validate(
        {
            "substitution": {
                "morphism": {
                    "source_alphabet": ["0", "1"],
                    "target_alphabet": ["0", "1"],
                    "images": [["0", "1"], ["0"]],
                }
            },
            "seed": "0",
        }
    )
    result = compute_substitution_factor_complexity(
        SubstitutionFactorComplexityRequest(source=source, prefix_length=8, max_order=3)
    )
    assert result.complexity == (1, 2, 3, 4)
    assert len(result.prefix.letters) == 8
    assert result.scope == "FACTORS_OF_RETAINED_PREFIX_ONLY"


def test_order_budget_is_resource_refusal() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError, match="between 0"):
        factor_complexity_prefix(_word(), 100)
    with pytest.raises(OperationDomainValidationError, match="between 0"):
        rauzy_graph(_word(), 5)
