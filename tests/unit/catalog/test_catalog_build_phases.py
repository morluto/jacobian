"""Direct seam tests for explicit catalog build phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jacobian.catalog.build import (
    CatalogBuildContext,
    CatalogBuildResources,
    CatalogOperationBuilder,
)
from jacobian.contracts.operations import ProviderAvailability
from jacobian.lean_frontend.selected import install_selected_lean_catalog


def test_core_domain_verification_phase_accepts_empty_declarations() -> None:
    assert (
        CatalogOperationBuilder(
            cast(CatalogBuildContext, object())
        ).bind_domain_verification({})
        is None
    )


@dataclass
class _UnauthorizedContext:
    authorize_bundled_checkers: bool = False
    checkers: object = field(
        default_factory=lambda: SimpleNamespace(bind_existing_when_omitted=False)
    )
    registered: list[object] = field(default_factory=list)

    def register_operation(self, adapter: object) -> None:
        self.registered.append(adapter)


def test_lean_catalog_omits_checkers_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.lean_frontend.selected.lean_frontend_provider_runtime",
        lambda: SimpleNamespace(availability=ProviderAvailability.UNAVAILABLE),
    )
    context = _UnauthorizedContext()
    install_selected_lean_catalog(
        cast(CatalogBuildContext, context),
        resources=CatalogBuildResources(),
    )

    assert context.registered == []


def test_catalog_close_releases_every_owned_lean_resource_once() -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    result = CatalogBuildResources()
    result.lean_declarations = cast(Any, Resource("declarations"))
    result.lean_exploration = cast(
        Any,
        SimpleNamespace(repl=Resource("exploration")),
    )
    result.lean = cast(Any, Resource("verification"))

    result.close()
    result.close()

    assert closed == ["declarations", "exploration", "verification"]


def test_catalog_close_continues_after_keyboard_interrupt() -> None:
    closed: list[str] = []

    class InterruptingResource:
        def close(self) -> None:
            closed.append("declarations")
            raise KeyboardInterrupt("declarations close interrupted")

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    result = CatalogBuildResources()
    result.lean_declarations = cast(Any, InterruptingResource())
    result.lean_exploration = cast(
        Any,
        SimpleNamespace(repl=Resource("exploration")),
    )
    result.lean = cast(Any, Resource("verification"))

    with pytest.raises(
        BaseExceptionGroup, match="catalog resources failed to close"
    ) as exc:
        result.close()

    assert closed == ["declarations", "exploration", "verification"]
    assert [str(failure) for failure in exc.value.exceptions] == [
        "declarations close interrupted",
    ]
