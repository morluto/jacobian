"""End-to-end caller-visible journey fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.operator_lifecycle import CheckerAuthorization, initialize_state


@pytest.fixture(scope="session")
def initialized_authorized_state_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Publish one operator-initialized bundled-checker MCP snapshot.

    MCP serving reads ``OperationCatalog``. Complete catalog-build templates do
    not persist that snapshot, so this fixture uses ``initialize_state`` once
    and every non-install journey copies the result.
    """

    root = tmp_path_factory.mktemp("initialized-authorized-state")
    initialize_state(
        root,
        checker_authorization=CheckerAuthorization.BUNDLED,
    )
    return root


__all__ = ("initialized_authorized_state_template",)
