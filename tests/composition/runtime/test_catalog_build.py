from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import jacobian.catalog_build as assembler
from jacobian.catalog_build_resources import CatalogBuildResources


class _RecordingContext:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def __enter__(self) -> None:
        self.events.append(f"enter:{self.name}")

    def __exit__(self, *_exc: object) -> None:
        self.events.append(f"exit:{self.name}")


class _Resolver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        events.append("resolver:init")

    def resolve(self) -> str:
        self.events.append("resolver:resolve")
        return "runtimes"


def _binder(
    events: list[str], name: str, *, bind_event: str | None = None
) -> SimpleNamespace:
    events.append(f"{name}:init")
    return SimpleNamespace(
        bind=lambda *_args, **_kwargs: events.append(bind_event or f"{name}:bind")
    )


def test_build_catalog_operations_owns_transaction_and_phase_order(monkeypatch) -> None:
    events: list[str] = []
    store = SimpleNamespace(
        transaction=lambda: _RecordingContext(events, "store"),
    )
    checkers = SimpleNamespace(
        policy_transaction=lambda: _RecordingContext(events, "policy"),
    )
    core = SimpleNamespace(store=store, checkers=checkers)
    application = SimpleNamespace(core=core)
    context = SimpleNamespace(store=store)

    monkeypatch.setattr(assembler, "ProviderInventoryLoader", lambda: _Resolver(events))
    monkeypatch.setattr(
        assembler,
        "CatalogFoundationBuilder",
        lambda _context: _binder(events, "foundation"),
    )
    monkeypatch.setattr(
        assembler,
        "CatalogOperationBuilder",
        lambda _context: _binder(events, "core"),
    )
    monkeypatch.setattr(
        assembler,
        "CatalogResourceBuilder",
        lambda _context: _binder(events, "resource"),
    )

    def checker_binder(_context, _resolver):
        events.append("checker:init")
        return SimpleNamespace(
            bind=lambda *_args: (
                events.append("checker:bind"),
                SimpleNamespace(),
            )[1]
        )

    monkeypatch.setattr(assembler, "CatalogCheckerBuilder", checker_binder)
    monkeypatch.setattr(assembler, "cached_package_digests", lambda: nullcontext())

    resources = assembler.build_catalog_operations(context, application)

    assert isinstance(resources, CatalogBuildResources)
    resources.close()
    assert events == [
        "resolver:init",
        "enter:policy",
        "enter:store",
        "resolver:resolve",
        "foundation:init",
        "foundation:bind",
        "core:init",
        "core:bind",
        "resource:init",
        "resource:bind",
        "checker:init",
        "checker:bind",
        "exit:store",
        "exit:policy",
    ]
