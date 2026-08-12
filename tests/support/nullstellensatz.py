"""Frozen exact Nullstellensatz evidence shared across semantic test lanes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from jacobian.contracts.nullstellensatz import NullstellensatzChartCertificate
from jacobian.domains.polynomial_nullstellensatz.core import (
    install_nullstellensatz_core,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.runtime.config import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_CERTIFICATE = (
    _ROOT
    / "benchmarks"
    / "datasets"
    / "research-diagnostics-v1"
    / "jcb-postdoc-019"
    / "solution"
    / "nullstellensatz-certificate.json"
)


@lru_cache(maxsize=1)
def load_chart_certificates() -> tuple[NullstellensatzChartCertificate, ...]:
    """Load the checked-in public reproduction without recomputing its proof."""

    payload = json.loads(_PUBLIC_CERTIFICATE.read_text(encoding="utf-8"))
    return tuple(
        NullstellensatzChartCertificate.model_validate(chart)
        for chart in payload["charts"]
    )


@contextmanager
def open_nullstellensatz_services(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode,
) -> Iterator[DomainTestServices]:
    """Open test services with the direct Nullstellensatz core installation."""

    with open_domain_services(root, checker_authority=checker_authority) as services:
        runtime = known_provider_runtime(
            "jacobian.nullstellensatz-core",
            features=(
                "normalized-jacobian-degree-slice",
                "rabinowitsch-chart-cover",
                "independent-exact-replay",
            ),
        )
        with atomic_installation(services.core):
            installed = install_nullstellensatz_core(services.installation, runtime)
            for adapter in installed.adapters:
                services.installation.register_capability(adapter)
        yield services


__all__ = ["load_chart_certificates", "open_nullstellensatz_services"]
