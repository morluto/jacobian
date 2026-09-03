"""Killable backend phase for finite-field fixed-subspace linear algebra."""

from __future__ import annotations

import json
import sys
from hashlib import sha256

from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)


def main() -> None:
    input_bytes = sys.stdin.buffer.read()
    source_digest = sha256(input_bytes).hexdigest()
    payload = json.loads(input_bytes)
    generator_ranks = []
    for entries in payload["matrices"]:
        generator_ranks.append(
            rank(
                PrimeFieldMatrix(
                    prime=payload["prime"],
                    entries=tuple(tuple(row) for row in entries),
                    columns=payload["generator_columns"],
                )
            )
        )
    if any(value != payload["generator_columns"] for value in generator_ranks):
        json.dump(
            {
                "source_digest": source_digest,
                "generators_invertible": False,
                "basis_rows": [],
            },
            sys.stdout,
            separators=(",", ":"),
        )
        return
    matrix = PrimeFieldMatrix(
        prime=payload["prime"],
        entries=tuple(tuple(row) for row in payload["equation_entries"]),
        columns=payload["equation_columns"],
    )
    nullspace_rows = nullspace(matrix)
    if nullspace_rows:
        reduced, pivots = rref(
            PrimeFieldMatrix(
                prime=matrix.prime,
                entries=nullspace_rows,
                columns=matrix.columns,
            )
        )
        basis_rows = reduced[: len(pivots)]
    else:
        basis_rows = ()
    json.dump(
        {
            "source_digest": source_digest,
            "generators_invertible": True,
            "basis_rows": basis_rows,
        },
        sys.stdout,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
