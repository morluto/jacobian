"""Direct seam tests for explicit catalog build phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

import jacobian.catalog_foundations as foundation_binding
from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.catalog_checkers import CatalogCheckerBuilder
from jacobian.catalog_foundations import CatalogFoundationBuilder
from jacobian.catalog_operations import CatalogOperationBuilder
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.polytope import PolytopeService
from jacobian.provider_inventory import (
    ProviderInventory,
    ProviderInventoryLoader,
)


def _unavailable_runtime(provider: str) -> ProviderObservation:
    return ProviderObservation(
        provider=provider,
        availability=ProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T0,
        license_id="MIT",
        diagnostic=f"{provider} is unavailable",
    )


def _available_runtime(provider: str) -> ProviderObservation:
    return ProviderObservation(
        provider=provider,
        availability=ProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "0" * 64,
        digest_kind=ProviderDigestKind.SOURCE_TREE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T0,
        license_id="MIT",
    )


def _provider_plan_with_unavailable_external_solvers() -> ProviderInventory:
    return ProviderInventory(
        cadical=_unavailable_runtime("cadical"),
        carcara=_unavailable_runtime("carcara"),
        cvc5=_available_runtime("cvc5"),
        drat_trim=_unavailable_runtime("drat-trim"),
        sympy_polynomial_normalization=_available_runtime(
            "sympy-polynomial-normalization"
        ),
    )


def test_foundation_solver_phase_skips_unavailable_external_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    context = SimpleNamespace(register_operation=registered.append, smt=object())
    adapter = object()
    monkeypatch.setattr(
        foundation_binding,
        "install_cvc5_operation",
        lambda _smt, _runtime: adapter,
    )

    CatalogFoundationBuilder(cast(CatalogBuildContext, context)).bind_solver_components(
        _provider_plan_with_unavailable_external_solvers(),
    )

    assert (
        _provider_plan_with_unavailable_external_solvers().cadical.availability
        is ProviderAvailability.UNAVAILABLE
    )
    assert registered == [adapter]


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


def test_checker_phase_derives_authority_from_its_context() -> None:
    context = _UnauthorizedContext()
    CatalogCheckerBuilder(
        cast(CatalogBuildContext, context),
        cast(ProviderInventoryLoader, object()),
    ).bind(cast(PolytopeService, object()), CatalogBuildResources())

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
