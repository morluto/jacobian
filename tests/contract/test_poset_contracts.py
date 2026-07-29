from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.posets import (
    FinitePosetRequest,
    LinearExtensionRequest,
    MobiusFunctionRequest,
)
from jacobian.domains.posets.operations import _materialized_poset


def _materialize(elements: list[str], relation: list[tuple[str, str]]):
    return _materialized_poset(
        FinitePosetRequest(
            elements=elements,
            relation=[{"lower": lower, "upper": upper} for lower, upper in relation],
            interpretation="COVER_EDGES",
        )
    )


def test_cover_relation_rejects_cycles_and_redundant_edges() -> None:
    with pytest.raises(ValidationError, match="antisymmetric"):
        FinitePosetRequest(
            elements=["a", "b"],
            relation=[
                {"lower": "a", "upper": "b"},
                {"lower": "b", "upper": "a"},
            ],
            interpretation="COVER_EDGES",
        )
    with pytest.raises(ValidationError, match="redundant"):
        FinitePosetRequest(
            elements=["a", "b", "c"],
            relation=[
                {"lower": "a", "upper": "b"},
                {"lower": "b", "upper": "c"},
                {"lower": "a", "upper": "c"},
            ],
            interpretation="COVER_EDGES",
        )


def test_comparable_pairs_require_complete_transitive_relation() -> None:
    with pytest.raises(ValidationError, match="complete strict order"):
        FinitePosetRequest(
            elements=["a", "b", "c"],
            relation=[
                {"lower": "a", "upper": "b"},
                {"lower": "b", "upper": "c"},
            ],
            interpretation="COMPARABLE_PAIRS",
        )


def test_required_reflexive_policy_binds_the_entire_diagonal() -> None:
    with pytest.raises(ValidationError, match="full carrier"):
        FinitePosetRequest(
            elements=["a", "b"],
            relation=[{"lower": "a", "upper": "a"}],
            interpretation="COMPARABLE_PAIRS",
            reflexive_pairs="REQUIRED",
        )


def test_linear_extension_contract_has_a_separate_exponential_bound() -> None:
    antichain = _materialize([f"x{index}" for index in range(17)], [])
    with pytest.raises(ValidationError, match="at most 16"):
        LinearExtensionRequest(poset=antichain)


def test_selected_mobius_scope_rejects_nonintervals_and_empty_selection() -> None:
    poset = _materialize(["a", "b"], [])
    with pytest.raises(ValidationError, match="at least one"):
        MobiusFunctionRequest(
            poset=poset,
            scope="SELECTED_INTERVALS",
        )
    with pytest.raises(ValidationError, match="lower <= upper"):
        MobiusFunctionRequest(
            poset=poset,
            scope="SELECTED_INTERVALS",
            intervals=[{"lower": "a", "upper": "b"}],
        )
