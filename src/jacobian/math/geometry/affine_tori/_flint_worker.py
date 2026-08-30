"""One-shot python-flint worker for affine-torus fixed loci."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.geometry.affine_tori._flint import (
    AffineTorusKernelSource,
    EmptyFixedLocusKernel,
    NonemptyFixedLocusKernel,
    compute_fixed_locus_kernel,
)
from jacobian.math.geometry.affine_tori.values import (
    MAX_AFFINE_TORUS_DIMENSION,
    MAX_AFFINE_TORUS_INPUT_DIGITS,
)

_PROTOCOL_VERSION = 1
_SUPPORTED_PYTHON_FLINT_VERSION = "0.9.0"
_MAX_WORKER_STDIN_BYTES = 64 * 1024


def _strict_integer(value: Any, *, maximum_digits: int, positive: bool = False) -> int:
    if not isinstance(value, str) or len(value.lstrip("-")) > maximum_digits:
        raise ValueError("worker integer is outside its canonical digit envelope")
    parsed = parse_canonical_integer(value)
    if value != format_canonical_integer(parsed) or (positive and parsed <= 0):
        raise ValueError("worker integer is not canonical")
    return parsed


def _strict_fraction(value: Any) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("worker rational has invalid fields")
    numerator = _strict_integer(
        value["num"], maximum_digits=MAX_AFFINE_TORUS_INPUT_DIGITS
    )
    denominator = _strict_integer(
        value["den"],
        maximum_digits=MAX_AFFINE_TORUS_INPUT_DIGITS,
        positive=True,
    )
    result = Fraction(numerator, denominator)
    if (
        value["num"] != format_canonical_integer(result.numerator)
        or value["den"] != format_canonical_integer(result.denominator)
        or not 0 <= result < 1
    ):
        raise ValueError("worker rational is not a canonical torus coordinate")
    return result


def _decode_source(payload: Any) -> AffineTorusKernelSource:
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "dimension",
        "linear_part",
        "translation",
    }:
        raise ValueError("worker request has invalid fields")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != _PROTOCOL_VERSION
    ):
        raise ValueError("worker request has an unsupported protocol version")
    dimension = payload["dimension"]
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or not 0 <= dimension <= MAX_AFFINE_TORUS_DIMENSION
    ):
        raise ValueError("worker request has an invalid torus dimension")
    linear_value = payload["linear_part"]
    if not isinstance(linear_value, list) or len(linear_value) != dimension:
        raise ValueError("worker linear part has an invalid row count")
    linear_rows: list[tuple[int, ...]] = []
    for candidate_row in linear_value:
        if not isinstance(candidate_row, list) or len(candidate_row) != dimension:
            raise ValueError("worker linear part has an invalid column count")
        linear_rows.append(
            tuple(
                _strict_integer(entry, maximum_digits=MAX_AFFINE_TORUS_INPUT_DIGITS)
                for entry in candidate_row
            )
        )
    translation_value = payload["translation"]
    if not isinstance(translation_value, list) or len(translation_value) != dimension:
        raise ValueError("worker translation has an invalid dimension")
    return AffineTorusKernelSource(
        dimension=dimension,
        linear_part=tuple(linear_rows),
        translation=tuple(_strict_fraction(value) for value in translation_value),
    )


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _kernel_payload(
    kernel: EmptyFixedLocusKernel | NonemptyFixedLocusKernel,
) -> dict[str, Any]:
    if isinstance(kernel, EmptyFixedLocusKernel):
        return {
            "status": "EMPTY",
            "character": [
                format_canonical_integer(value) for value in kernel.character
            ],
            "pairing": _fraction_payload(kernel.pairing),
        }
    identity_dimension = (
        len(kernel.identity_embedding[0]) if kernel.identity_embedding else 0
    )
    return {
        "status": "NONEMPTY",
        "rank": len(kernel.component_generators),
        "nullity": identity_dimension,
        "base_point": [_fraction_payload(value) for value in kernel.base_point],
        "identity_embedding": [
            [format_canonical_integer(value) for value in row]
            for row in kernel.identity_embedding
        ],
        "component_generators": [
            [_fraction_payload(value) for value in generator]
            for generator in kernel.component_generators
        ],
        "relation_matrix": [
            [format_canonical_integer(value) for value in row]
            for row in kernel.relation_matrix
        ],
        "generator_orders": [
            format_canonical_integer(value) for value in kernel.generator_orders
        ],
        "invariant_factors": [
            format_canonical_integer(value) for value in kernel.invariant_factors
        ],
        "component_count": format_canonical_integer(kernel.component_count),
    }


def main() -> int:
    import flint

    if flint.__version__ != _SUPPORTED_PYTHON_FLINT_VERSION:
        raise RuntimeError("unsupported python-flint worker version")
    input_bytes = sys.stdin.buffer.read(_MAX_WORKER_STDIN_BYTES + 1)
    if len(input_bytes) > _MAX_WORKER_STDIN_BYTES:
        raise ValueError("worker request exceeds its input envelope")
    source = _decode_source(json.loads(input_bytes))
    response = {
        "protocol_version": _PROTOCOL_VERSION,
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        **_kernel_payload(compute_fixed_locus_kernel(source)),
    }
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
