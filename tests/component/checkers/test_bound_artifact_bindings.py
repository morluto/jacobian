from __future__ import annotations

import pytest

from jacobian_checkers.bound_artifacts import valid_unscoped_unencoded_bindings


def _bindings() -> dict[str, object]:
    return {
        "claim_digest": "sha256:" + "1" * 64,
        "semantics_digest": "sha256:" + "2" * 64,
        "candidate_digest": "sha256:" + "3" * 64,
        "scope_digest": None,
        "encoding_digest": None,
    }


def test_unscoped_bindings_accept_exact_canonical_shape() -> None:
    assert valid_unscoped_unencoded_bindings(_bindings())


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_unscoped_bindings_reject_inexact_fields(mutation: str) -> None:
    bindings = _bindings()
    if mutation == "missing":
        bindings.pop("candidate_digest")
    else:
        bindings["unexpected"] = None

    assert not valid_unscoped_unencoded_bindings(bindings)


@pytest.mark.parametrize(
    "field", ["claim_digest", "semantics_digest", "candidate_digest"]
)
@pytest.mark.parametrize(
    "value",
    [
        "sha256:" + "A" * 64,
        "sha256:" + "4" * 63,
        True,
        7,
    ],
    ids=["uppercase", "wrong-length", "boolean", "non-string"],
)
def test_unscoped_bindings_reject_malformed_digests(field: str, value: object) -> None:
    bindings = _bindings()
    bindings[field] = value

    assert not valid_unscoped_unencoded_bindings(bindings)


@pytest.mark.parametrize("field", ["scope_digest", "encoding_digest"])
def test_unscoped_bindings_reject_scope_or_encoding_digest(field: str) -> None:
    bindings = _bindings()
    bindings[field] = "sha256:" + "4" * 64

    assert not valid_unscoped_unencoded_bindings(bindings)
