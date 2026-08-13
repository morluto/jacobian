from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.lean import LeanCheckRequest, LeanEnvironment


def test_lean_check_request_defaults_to_the_core_environment() -> None:
    request = LeanCheckRequest(statement="True", proof="by trivial")

    assert request.environment is LeanEnvironment.CORE


def test_lean_check_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LeanCheckRequest(
            statement="True",
            proof="by trivial",
            hidden_mode=True,  # type: ignore[call-arg]
        )
