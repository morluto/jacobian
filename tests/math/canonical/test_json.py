from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    canonicalize_json,
    encode_strict_json,
    sha256_digest,
)


def test_sha256_digest_uses_the_canonical_prefixed_format() -> None:
    for value in (b"", b"\x00", b"jacobian", b"\xde\xad\xbe\xef" * 100):
        assert sha256_digest(value) == "sha256:" + hashlib.sha256(value).hexdigest()


def test_equivalent_rationals_have_identical_canonical_bytes() -> None:
    first = canonicalize_json({"weight": {"num": "2", "den": "4"}})
    second = canonicalize_json({"weight": {"num": "1", "den": "2"}})

    assert first == second == b'{"weight":{"den":"2","num":"1"}}'


def test_strict_json_encoding_preserves_unreduced_rationals_and_unicode() -> None:
    decomposed = "e\u0301"

    encoded = encode_strict_json(
        {"weight": {"num": "2", "den": "4"}, "label": decomposed}
    )

    assert encoded.decode("utf-8") == (
        f'{{"label":"{decomposed}","weight":{{"den":"4","num":"2"}}}}'
    )
    assert encode_strict_json(decomposed) == b'"e\xcc\x81"'


@pytest.mark.parametrize(
    "value",
    [
        {"weight": 0.5},
        '{"x": 1, "x": 2}',
        {"weight": {"num": "1", "den": "0"}},
    ],
)
def test_ambiguous_or_inexact_json_is_rejected(value: object) -> None:
    with pytest.raises(CanonicalizationError, match=r"not allowed|duplicate|zero"):
        canonicalize_json(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('{"value": NaN}', "non-finite JSON value is not allowed"),
        (
            '{"secret": 1, "secret": 2}',
            "duplicate JSON object key: 'secret'. Remove or rename one occurrence.",
        ),
        ({"value": (1, 2)}, "unsupported JSON value type"),
    ],
)
def test_canonical_errors_preserve_only_repair_relevant_context(
    value: object,
    message: str,
) -> None:
    with pytest.raises(CanonicalizationError) as raised:
        canonicalize_json(value)

    assert str(raised.value) == message
    assert "NaN" not in str(raised.value)
    assert "tuple" not in str(raised.value)


def test_canonical_rational_wire_model_rejects_unreduced_input() -> None:
    with pytest.raises(ValidationError, match=r"rational must be reduced"):
        CanonicalRational.model_validate({"num": "2", "den": "4"})


@pytest.mark.parametrize(
    "value",
    [
        {"num": "-101", "den": "1"},
        {"num": "1", "den": "101"},
    ],
)
def test_bounded_rational_rejects_oversized_canonical_components(
    value: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="test rational exceeds the 2-digit bound"):
        require_bounded_rational(
            CanonicalRational.model_validate(value),
            max_digits=2,
            label="test rational",
        )


def test_negative_zero_is_not_a_canonical_integer_encoding() -> None:
    with pytest.raises(CanonicalizationError, match=r"canonical decimal"):
        canonicalize_json({"num": "-0", "den": "1"})
    with pytest.raises(ValidationError, match=r"num"):
        CanonicalRational.model_validate({"num": "-0", "den": "1"})


def test_num_den_is_a_reserved_exact_rational_shape() -> None:
    assert canonicalize_json({"num": "2", "den": "4"}) == canonicalize_json(
        {"num": "1", "den": "2"}
    )


def test_num_den_schema_property_map_is_regular_json() -> None:
    encoded = canonicalize_json(
        {
            "num": {"type": "string"},
            "den": {"type": "string"},
        }
    )

    assert encoded == b'{"den":{"type":"string"},"num":{"type":"string"}}'


def test_unicode_bom_and_non_json_tuples_are_rejected() -> None:
    with pytest.raises(CanonicalizationError, match=r"BOM"):
        canonicalize_json(b'\xef\xbb\xbf{"value":1}')
    with pytest.raises(CanonicalizationError, match=r"unsupported JSON value type"):
        canonicalize_json({"value": (1, 2)})


def test_nesting_beyond_the_configured_limit_is_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="depth"):
        canonicalize_json(
            {"a": {"b": {"c": {"d": 1}}}},
            limits=CanonicalLimits(max_depth=2),
        )


@pytest.mark.property
@settings(max_examples=100)
@given(
    numerator=st.integers(min_value=-(10**100), max_value=10**100),
    denominator=st.integers(min_value=1, max_value=10**50),
    scale=st.integers(min_value=1, max_value=10**12),
)
def test_scaled_rationals_have_the_same_canonical_bytes(
    numerator: int,
    denominator: int,
    scale: int,
) -> None:
    reduced = canonicalize_json({"num": str(numerator), "den": str(denominator)})
    scaled = canonicalize_json(
        {
            "num": str(numerator * scale),
            "den": str(denominator * scale),
        }
    )

    assert scaled == reduced
