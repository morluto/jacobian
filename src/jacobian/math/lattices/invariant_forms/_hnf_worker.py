"""Killable FLINT worker for invariant-form graph-lattice HNF."""

from __future__ import annotations

import hashlib
import sys

from flint import fmpz_mat

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)


def _decode_integer(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("worker integer must be a canonical string")
    parsed = parse_canonical_integer(value)
    if format_canonical_integer(parsed) != value:
        raise ValueError("worker integer is not canonical")
    return parsed


def main() -> None:
    input_bytes = sys.stdin.buffer.read()
    payload = loads_strict_json(
        input_bytes,
        limits=CanonicalLimits(max_input_bytes=len(input_bytes)),
    )
    coefficient_count = payload["coefficient_count"]
    constraints = payload["constraints"]
    constraint_count = len(constraints)
    graph = fmpz_mat(
        [
            [
                *(
                    _decode_integer(constraints[constraint][coordinate])
                    for constraint in range(constraint_count)
                ),
                *(int(coordinate == column) for column in range(coefficient_count)),
            ]
            for coordinate in range(coefficient_count)
        ]
    )
    graph_hnf = graph.hnf()
    primitive_kernel = [
        [
            format_canonical_integer(int(graph_hnf[row, constraint_count + column]))
            for column in range(coefficient_count)
        ]
        for row in range(coefficient_count)
        if all(
            graph_hnf[row, constraint] == 0 for constraint in range(constraint_count)
        )
    ]
    sys.stdout.buffer.write(
        encode_strict_json(
            {
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                "primitive_kernel": primitive_kernel,
                "constraint_rank": coefficient_count - len(primitive_kernel),
            }
        )
    )


if __name__ == "__main__":
    main()
