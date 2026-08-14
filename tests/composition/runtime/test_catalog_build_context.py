from __future__ import annotations

from types import SimpleNamespace

import pytest

from jacobian.catalog_build_context import create_catalog_build_context


def _graphs() -> tuple[SimpleNamespace, SimpleNamespace, list[object]]:
    registered: list[object] = []
    core = SimpleNamespace(
        store=object(),
        schemas=object(),
        artifacts=object(),
        values=object(),
        binder=object(),
        operations=SimpleNamespace(register=registered.append),
        checkers=object(),
    )
    application = SimpleNamespace(core=core, verification=object())
    return core, application, registered


def test_context_wires_application_verification_and_registers_adapters() -> None:
    core, application, registered = _graphs()
    context = create_catalog_build_context(
        core,
        application,
        authorize_bundled_checkers=True,
    )

    first = SimpleNamespace(descriptor=SimpleNamespace(operation_id="first"))
    context.register_operation(first)
    included = SimpleNamespace(descriptor=SimpleNamespace(operation_id="included"))
    context.register_operation(included)

    assert context.verification is application.verification
    assert context.values is core.values
    assert context.authorize_bundled_checkers
    assert registered == [first, included]


def test_context_rejects_application_built_from_another_core() -> None:
    core, application, _ = _graphs()

    with pytest.raises(ValueError, match="built from the supplied core"):
        create_catalog_build_context(
            core,
            SimpleNamespace(
                core=SimpleNamespace(), verification=application.verification
            ),
        )
