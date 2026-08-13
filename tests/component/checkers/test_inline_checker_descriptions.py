"""Model-facing lineage invariants for inline exact replay checkers."""

import pytest

from jacobian.domains.graph_optimization.checkers import (
    GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.number_theory.checkers import (
    NUMBER_THEORY_EXACT_REPLAY_CHECKERS,
)

_INLINE_REPLAY_CHECKERS = (
    *GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
    *NUMBER_THEORY_EXACT_REPLAY_CHECKERS,
)


@pytest.mark.parametrize(
    "declaration",
    _INLINE_REPLAY_CHECKERS,
    ids=lambda declaration: declaration.verification_capability_id,
)
def test_inline_checker_cards_do_not_claim_stored_producer_lineage(declaration) -> None:
    """Inline checkers bind submitted values, not an unavailable result URI."""

    description = declaration.verification_description
    assert description is not None
    assert "stored" not in description.lower()
    assert "result_uri" not in description.lower()
