"""Small supported-Python smoke tests used by the compatibility lane.

The Python 3.13 job is intentionally not a second copy of the complete test
suite.  Core behavior is exercised on the primary interpreter; this file only
checks that the public, dependency-light surface imports and validates a wire
value on every supported Python version.
"""

from __future__ import annotations


def test_public_contract_and_canonicalization_surface_imports() -> None:
    from jacobian import ResultEnvelope
    from jacobian.canonical import canonicalize_json

    assert ResultEnvelope.__name__ == "ResultEnvelope"
    assert canonicalize_json({"version": 1}) == b'{"version":1}'


def test_cli_help_surface_does_not_require_runtime_materialization() -> None:
    from jacobian.cli import app

    assert app.info.name == "jacobian"
    assert app.info.help
