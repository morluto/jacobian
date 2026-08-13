"""Independent graph6 replay using only the standard library."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def check_graph6_decode(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="graph.encoding.graph6.decode.compute",
            witness_format="graph.graph6-decode.standard-library-v1",
        )
        if set(source) != {"graph6"} or not isinstance(source["graph6"], str):
            raise ValueError("graph6 source is malformed")
        value = source["graph6"]
        value = value[10:] if value.startswith(">>graph6<<") else value
        if not value or value[0] in {":", "&"}:
            raise ValueError("unsupported graph encoding")
        codes = [ord(character) - 63 for character in value]
        if any(code < 0 or code > 63 for code in codes) or codes[0] == 63:
            raise ValueError("malformed or extended graph6 encoding")
        order = codes[0]
        bit_count = order * (order - 1) // 2
        if len(codes) != 1 + (bit_count + 5) // 6:
            raise ValueError("graph6 length does not match order")
        bits = [(code >> shift) & 1 for code in codes[1:] for shift in range(5, -1, -1)]
        if any(bits[bit_count:]):
            raise ValueError("graph6 padding bits are nonzero")
        pairs = [
            (first, second) for second in range(1, order) for first in range(second)
        ]
        edges = sorted(pair for pair, bit in zip(pairs, bits, strict=False) if bit)
        degrees = [0] * order
        for first, second in edges:
            degrees[first] += 1
            degrees[second] += 1
        digest_payload = json.dumps(
            {"edges": [list(edge) for edge in edges], "order": order},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        expected = {
            "graph6": value,
            "order": order,
            "edges": [{"first": first, "second": second} for first, second in edges],
            "degrees": degrees,
            "graph_digest": "sha256:" + hashlib.sha256(digest_payload).hexdigest(),
            "format": "GRAPH6_SMALL_ORDER",
            "bit_order": "COLUMN_MAJOR_UPPER_TRIANGLE",
        }
        if result != expected:
            return _reject("candidate does not match independent graph6 replay")
        return _accept("independent graph6 bitstream replay accepted candidate")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_graph6_decode"]
