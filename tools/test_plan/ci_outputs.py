"""Derive CI plan boolean keys from the compiled impact catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def catalog_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = manifest.get("catalog", {})
    if not isinstance(catalog, dict):
        raise ValueError("CI catalog must be an object")
    return {
        str(name): entry
        for name, entry in catalog.items()
        if isinstance(entry, dict) and not entry.get("local_only", False)
    }


def suite_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    suites = manifest.get("suites", ())
    if not isinstance(suites, list) or not all(isinstance(item, str) for item in suites):
        raise ValueError("CI suites must be an array of strings")
    catalog = catalog_entries(manifest)
    return tuple(name for name in suites if name in catalog)


def python_lane_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Pytest topology lanes excluding lean (hosted as its own gate)."""

    catalog = catalog_entries(manifest)
    return tuple(
        name
        for name in suite_names(manifest)
        if catalog[name].get("topology_lane") and name != "lean"
    )


def matrix_lane_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    catalog = catalog_entries(manifest)
    return tuple(
        name for name in suite_names(manifest) if catalog[name].get("matrix")
    )


def boolean_run_keys(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Stable CI plan boolean keys derived from the catalog."""

    python = python_lane_names(manifest)
    keys = ["run-python", *[f"run-{name}" for name in python]]
    keys.extend(["run-coverage", "run-compatibility"])
    seen = set(python)
    for name in suite_names(manifest):
        if name in seen:
            continue
        keys.append(f"run-{name}")
    return tuple(keys)


def python_run_keys(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f"run-{name}" for name in python_lane_names(manifest))
