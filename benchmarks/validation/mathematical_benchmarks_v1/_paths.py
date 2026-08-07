"""Shared dataset path and suite references for mathematical-benchmarks-v1.

Kept separate from metadata, fixture, and verifier-execution logic so the
focused modules can depend on one small path module without importing each
other.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[3]
TASKS = ROOT / "benchmarks" / "datasets" / "mathematical-benchmarks-v1"
AGENT_TASKS = get_suite("mathematical-benchmarks-v1").tasks
