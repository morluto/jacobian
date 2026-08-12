"""Cheap provider metadata inspection without importing implementations.

These helpers read installed distribution metadata through ``importlib.metadata``
only. They never import the provider package itself during runtime assembly;
missing distributions are reported through metadata instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from threading import RLock


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Typed installed-distribution metadata, without importing the package."""

    name: str
    version: str


_cache_lock = RLock()
_version_cache: dict[str, str] = {}
_summary_cache: dict[str, DistributionSummary] = {}


def distribution_version(distribution_name: str) -> str | None:
    """Return installed distribution metadata without importing its package.

    Successful identity lookups are cached for the process. Missing
    distributions are deliberately not cached: a long-lived process may gain
    an optional provider after its environment or import path changes.
    """

    with _cache_lock:
        cached = _version_cache.get(distribution_name)
        if cached is not None:
            return cached
        try:
            installed_version = version(distribution_name)
        except PackageNotFoundError:
            return None
        _version_cache[distribution_name] = installed_version
        return installed_version


def distribution_summary(distribution_name: str) -> DistributionSummary | None:
    """Return a typed summary of an installed distribution, or ``None``.

    The summary is read from distribution metadata only; the provider package
    is never imported. ``name`` falls back to the requested lookup name when
    the recorded ``Name`` header is missing or empty.
    """

    with _cache_lock:
        cached = _summary_cache.get(distribution_name)
        if cached is not None:
            return cached
        try:
            dist = distribution(distribution_name)
        except PackageNotFoundError:
            return None
        recorded_name = dist.metadata["Name"]
        if not isinstance(recorded_name, str) or not recorded_name:
            recorded_name = distribution_name
        summary = DistributionSummary(
            name=recorded_name,
            version=dist.version,
        )
        _summary_cache[distribution_name] = summary
        return summary
