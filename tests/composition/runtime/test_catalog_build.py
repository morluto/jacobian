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
    context = SimpleNamespace(store=store, checkers=checkers)

    monkeypatch.setattr(
        assembler,
        "require_maintained_math_backends",
        lambda: events.append("backends:check"),
    )
    for probe_name in (
        "cadical_provider_runtime",
        "carcara_provider_runtime",
        "cvc5_provider_runtime",
        "drat_trim_provider_runtime",
        "sympy_polynomial_normalization_provider_runtime",
    ):
        monkeypatch.setattr(
            assembler,
            probe_name,
            lambda name=probe_name: events.append(name) or name,
        )
    monkeypatch.setattr(
        assembler,
        "bind_catalog_foundations",
        lambda *_args, **_kwargs: events.append("foundation:bind"),
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

    def checker_binder(_context):
        events.append("checker:init")
        return SimpleNamespace(
            bind=lambda *_args: (
                events.append("checker:bind"),
                SimpleNamespace(),
            )[1]
        )

    monkeypatch.setattr(assembler, "CatalogCheckerBuilder", checker_binder)
    monkeypatch.setattr(assembler, "cached_package_digests", lambda: nullcontext())

    resources = assembler.build_catalog_operations(context, object())

    assert isinstance(resources, CatalogBuildResources)
    resources.close()
    assert events == [
        "enter:policy",
        "enter:store",
        "backends:check",
        "cadical_provider_runtime",
        "carcara_provider_runtime",
        "cvc5_provider_runtime",
        "drat_trim_provider_runtime",
        "sympy_polynomial_normalization_provider_runtime",
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
