"""Provider inspection and lazy implementation loading.

This package provides two independent, composable primitives:

* :func:`distribution_version` and :func:`distribution_summary` read installed
  distribution metadata through ``importlib.metadata`` without importing the
  provider implementation.
* :class:`LazyLoader` defers importing or constructing a heavy implementation
  until first use, caches success and failure, owns the implementation
  lifecycle, and exposes a typed :class:`LoaderState`.

The package deliberately avoids registries, package discovery, import-time
registration, and compatibility shims. Each loader is an independent, owned
object; successful metadata identities are deduplicated process-locally.
"""

from __future__ import annotations

from jacobian.providers.loader import LazyLoader, LazyLoadError, LoaderState
from jacobian.providers.metadata import (
    DistributionSummary,
    distribution_summary,
    distribution_version,
)

__all__ = [
    "DistributionSummary",
    "LazyLoadError",
    "LazyLoader",
    "LoaderState",
    "distribution_summary",
    "distribution_version",
]
