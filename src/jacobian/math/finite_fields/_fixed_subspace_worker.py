"""Killable backend phase for finite-field fixed-subspace linear algebra."""

from __future__ import annotations

import json
import sys

from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("operation") == "generator_ranks":
        generator_ranks = []
        for entries in payload["matrices"]:
            generator_ranks.append(
                rank(
                    PrimeFieldMatrix(
                        prime=payload["prime"],
                        entries=tuple(tuple(row) for row in entries),
                        columns=payload["columns"],
                    )
                )
            )
        json.dump({"ranks": generator_ranks}, sys.stdout, separators=(",", ":"))
        return
    matrix = PrimeFieldMatrix(
        prime=payload["prime"],
        entries=tuple(tuple(row) for row in payload["entries"]),
        columns=payload["columns"],
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
    json.dump({"basis_rows": basis_rows}, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
