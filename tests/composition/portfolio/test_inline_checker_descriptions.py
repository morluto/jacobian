"""Model-facing lineage invariants for inline exact replay checkers."""

import pytest

from jacobian.builtin_operation_modules import load_builtin_operation_modules

_INLINE_REPLAY_CHECKERS = tuple(
    declaration
    for _module_name, _operations, checker_declarations in load_builtin_operation_modules()
    for declaration in checker_declarations
    if ".materialize" not in declaration.operation_id
)


@pytest.mark.parametrize(
    "declaration",
    _INLINE_REPLAY_CHECKERS,
    ids=lambda declaration: declaration.verification_operation_id,
)
def test_inline_checker_cards_do_not_claim_stored_producer_lineage(declaration) -> None:
    """Inline checkers bind submitted values, not an unavailable result URI."""

    description = declaration.verification_description
    assert description is not None
    assert "stored" not in description.lower()
    assert "result_uri" not in description.lower()
