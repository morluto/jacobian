from __future__ import annotations

from types import SimpleNamespace

from jacobian.catalog.build import create_catalog_build_context


def _graphs() -> tuple[SimpleNamespace, object, list[object]]:
    registered: list[object] = []
    core = SimpleNamespace(
        store=object(),
        schemas=object(),
        artifacts=object(),
        values=object(),
        binder=object(),
        operations=SimpleNamespace(register=registered.append),
        checkers=object(),
        sat=object(),
        smt=object(),
        polynomial_expressions=object(),
    )
    verification = object()
    return core, verification, registered


def test_context_wires_application_verification_and_registers_adapters() -> None:
    core, verification, registered = _graphs()
    context = create_catalog_build_context(
        core,
        verification,
        authorize_bundled_checkers=True,
    )

    first = SimpleNamespace(descriptor=SimpleNamespace(operation_id="first"))
    context.register_operation(first)
    included = SimpleNamespace(descriptor=SimpleNamespace(operation_id="included"))
    context.register_operation(included)

    assert context.verification is verification
    assert context.values is core.values
    assert context.authorize_bundled_checkers
    assert registered == [first, included]
    assert context.sat is core.sat
    assert context.smt is core.smt
