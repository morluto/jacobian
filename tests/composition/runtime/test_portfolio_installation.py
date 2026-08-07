from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from jacobian.portfolio import assembler
from jacobian.portfolio.result import PortfolioInstallation


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


def _installer(
    events: list[str], name: str, *, install_event: str | None = None
) -> SimpleNamespace:
    events.append(f"{name}:init")
    return SimpleNamespace(
        install=lambda *_args, **_kwargs: events.append(
            install_event or f"{name}:install"
        )
    )


def test_install_portfolio_owns_transaction_and_phase_order(monkeypatch) -> None:
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

    monkeypatch.setattr(
        assembler, "ProviderAvailabilityResolver", lambda: _Resolver(events)
    )
    monkeypatch.setattr(
        assembler,
        "FoundationInstaller",
        lambda _context: _installer(events, "foundation"),
    )
    monkeypatch.setattr(
        assembler,
        "CoreApplicationInstaller",
        lambda _context: _installer(events, "core"),
    )
    monkeypatch.setattr(
        assembler,
        "ResourceCapabilityInstaller",
        lambda _context: _installer(events, "resource"),
    )
    monkeypatch.setattr(
        assembler,
        "ReferenceLeanInstaller",
        lambda _context, _resolver: _installer(
            events,
            "reference",
            install_event="reference:install:('fixture:adapter',)",
        ),
    )
    monkeypatch.setattr(assembler, "cached_package_digests", lambda: nullcontext())

    result = assembler.install_portfolio(
        context,
        application,
        capability_adapter_entrypoints=("fixture:adapter",),
    )

    assert isinstance(result, PortfolioInstallation)
    assert events == [
        "resolver:init",
        "enter:policy",
        "enter:store",
        "resolver:resolve",
        "foundation:init",
        "foundation:install",
        "core:init",
        "core:install",
        "resource:init",
        "resource:install",
        "reference:init",
        "reference:install:('fixture:adapter',)",
        "exit:store",
        "exit:policy",
    ]
