"""Model-facing lineage invariants for inline exact replay checkers."""

import pytest

from jacobian.domain_bundles import DomainBundle
from jacobian.portfolio.builtin import build_builtin_portfolio_components

_INLINE_REPLAY_CHECKERS = tuple(
    declaration
    for component in build_builtin_portfolio_components()
    if isinstance(component, DomainBundle)
    for declaration in component.checker_declarations
    if ".materialize" not in declaration.capability_id
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
