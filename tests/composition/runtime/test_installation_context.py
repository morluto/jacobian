from __future__ import annotations

from types import SimpleNamespace

import pytest

from jacobian.installation import create_installation_context
from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions


def _graphs() -> tuple[SimpleNamespace, SimpleNamespace, list[object]]:
    registered: list[object] = []
    core = SimpleNamespace(
        store=object(),
        schemas=object(),
        artifacts=object(),
        values=object(),
        operations=object(),
        capabilities=SimpleNamespace(register=registered.append),
        checkers=object(),
    )
    application = SimpleNamespace(core=core, verification=object())
    return core, application, registered


def test_context_wires_application_verification_and_filters_excluded_adapters() -> None:
    core, application, registered = _graphs()
    context = create_installation_context(
        core,
        application,
        RuntimeOptions(
            checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
            capability_exclusions=frozenset({"excluded"}),
        ),
    )

    context.register_capability(
        SimpleNamespace(descriptor=SimpleNamespace(capability_id="excluded"))
    )
    included = SimpleNamespace(descriptor=SimpleNamespace(capability_id="included"))
    context.register_capability(included)

    assert context.verification is application.verification
    assert context.values is core.values
    assert context.checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED
    assert registered == [included]
    assert context.authorizes_bundled_checkers


def test_context_rejects_application_built_from_another_core() -> None:
    core, application, _ = _graphs()

    with pytest.raises(ValueError, match="built from the supplied core"):
        create_installation_context(
            core,
            SimpleNamespace(
                core=SimpleNamespace(), verification=application.verification
            ),
            RuntimeOptions(),
        )
