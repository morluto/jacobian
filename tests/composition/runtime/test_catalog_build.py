from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import jacobian.catalog.build as assembler
from jacobian.catalog.build import CatalogBuildResources


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
    family_origins = (
        "graph",
        "polynomial",
        "lean",
        "sat-smt",
        "core",
    )

    monkeypatch.setattr(
        assembler,
        "require_maintained_math_backends",
        lambda: events.append("backends:check"),
    )
    monkeypatch.setattr(
        assembler,
        "CatalogOperationBuilder",
        lambda _context: _binder(events, "builtin"),
    )
    monkeypatch.setattr(
        assembler,
        "selected_family_specs",
        lambda: tuple(SimpleNamespace(origin=origin) for origin in family_origins),
    )
    monkeypatch.setattr(
        assembler,
        "selected_family_catalog_installers",
        lambda: {
            origin: (lambda *_args, origin=origin, **_kwargs: events.append(origin))
            for origin in family_origins
        },
    )
    monkeypatch.setattr(assembler, "cached_package_digests", lambda: nullcontext())

    resources = assembler.build_catalog_operations(context, object())

    assert isinstance(resources, CatalogBuildResources)
    resources.close()
    assert events == [
        "enter:policy",
        "enter:store",
        "backends:check",
        "builtin:init",
        "builtin:bind",
        *family_origins,
        "exit:store",
        "exit:policy",
    ]
