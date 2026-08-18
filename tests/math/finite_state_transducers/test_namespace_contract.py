"""Owner-local exact public API contract for finite_state_transducers."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.finite_state_transducers")
    expected = (
        "RationalEdge",
        "RationalTransducer",
        "SubseqFinalOutput",
        "SubseqTransition",
        "SubsequentialTransducer",
        "coaccessible_states",
        "compose_subsequential",
        "identity_transducer",
        "invert_rational",
        "reachable_states",
        "replay_rational_path",
        "run_subsequential",
        "trim_subsequential",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
