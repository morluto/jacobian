"""Discovery coverage for exact DFA language equivalence."""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


def test_dfa_equivalence_need_routes_to_the_decision_operation() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(
            need="exact deterministic finite automata language equivalence "
            "shortest distinguishing word",
            limit=5,
        )
    )

    assert result.matches
    assert result.matches[0].operation_id == ("regular_language.dfa.equivalence.decide")
